//! Benchmarks to inform dispatch logic in process_batch_sorted
//!
//! This benchmark directly tests individual execution strategies to determine
//! optimal thresholds for dispatching between Tiled, DatasetMajor, and
//! ParameterMajor strategies.
//!
//! All benchmark metadata is encoded in the benchmark ID, allowing us to use
//! Criterion's native CSV export. The benchmark ID format is:
//!   `strategy/engine/cache_level/n<size>/d<datasets>/q<quantiles>/t<threads>`
//!
//! Example: `DatasetMajor/scalar_seq/L1_2/n3072/d5/q50/t1`
//!
//! To export results:
//!   cargo bench --bench dispatch_strategy_bench -- --save-baseline my_run
//!
//! The CSV data can be found in target/criterion/ and includes all the
//! encoded metadata in the benchmark names.

use criterion::{black_box, criterion_group, BenchmarkId, Criterion, Throughput};
#[cfg(feature = "parallel")]
use robust_core::execution::{scalar_parallel, simd_parallel};
use robust_core::{
    execution::{scalar_sequential, simd_sequential, ExecutionEngine},
    BatchProcessor, CachePolicy, ProcessingStrategy, UnifiedWeightCache,
};
use robust_quantile::{
    estimators::{harrell_davis, trimmed_harrell_davis},
    HDWeightComputer, SqrtWidth, TrimmedHDWeightComputer,
};
use std::time::Duration;

// No need for custom result storage - we'll use Criterion's native export

/// Cache topology information
#[derive(Debug, Clone)]
struct CacheTopology {
    l1_data_size: usize,
    l2_size: usize,
    l3_size: usize,
    cache_line_size: usize,
}

impl CacheTopology {
    /// Detect cache topology from system
    fn detect() -> Self {
        #[cfg(target_os = "linux")]
        {
            // Try to read from sysfs
            let l1_data = Self::read_cache_size("/sys/devices/system/cpu/cpu0/cache/index0/size")
                .or_else(|| Self::read_cache_size("/sys/devices/system/cpu/cpu0/cache/index1/size"))
                .unwrap_or(32 * 1024); // 32KB default

            let l2 = Self::read_cache_size("/sys/devices/system/cpu/cpu0/cache/index2/size")
                .unwrap_or(256 * 1024); // 256KB default

            let l3 = Self::read_cache_size("/sys/devices/system/cpu/cpu0/cache/index3/size")
                .unwrap_or(8 * 1024 * 1024); // 8MB default

            let cache_line = Self::read_cache_size(
                "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size",
            )
            .unwrap_or(64); // 64 bytes default

            Self {
                l1_data_size: l1_data,
                l2_size: l2,
                l3_size: l3,
                cache_line_size: cache_line,
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Conservative defaults for other platforms
            Self {
                l1_data_size: 32 * 1024,  // 32KB
                l2_size: 256 * 1024,      // 256KB
                l3_size: 8 * 1024 * 1024, // 8MB
                cache_line_size: 64,      // 64 bytes
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn read_cache_size(path: &str) -> Option<usize> {
        use std::fs;
        fs::read_to_string(path).ok().and_then(|s| {
            let s = s.trim();
            if s.ends_with('K') {
                s[..s.len() - 1].parse::<usize>().ok().map(|n| n * 1024)
            } else if s.ends_with('M') {
                s[..s.len() - 1]
                    .parse::<usize>()
                    .ok()
                    .map(|n| n * 1024 * 1024)
            } else {
                s.parse::<usize>().ok()
            }
        })
    }

    /// Get dataset sizes that test different cache levels
    fn get_test_sizes(&self) -> Vec<(usize, &'static str)> {
        let f64_size = std::mem::size_of::<f64>();

        vec![
            // L1 cache tests
            (self.l1_data_size / f64_size / 4, "L1/4"),
            (self.l1_data_size / f64_size / 2, "L1/2"),
            (self.l1_data_size * 3 / 4 / f64_size, "L1*3/4"),
            // L2 cache tests
            (self.l2_size / f64_size / 4, "L2/4"),
            (self.l2_size / f64_size / 2, "L2/2"),
            (self.l2_size * 3 / 4 / f64_size, "L2*3/4"),
            // L3 cache tests
            (self.l3_size / f64_size / 4, "L3/4"),
            (self.l3_size / f64_size / 2, "L3/2"),
            (self.l3_size * 3 / 4 / f64_size, "L3*3/4"),
            // Beyond L3 (RAM)
            (self.l3_size * 2 / f64_size, "L3*2"),
            (self.l3_size * 4 / f64_size, "L3*4"),
        ]
    }
}

/// Generate test data with specific characteristics
fn generate_test_data(n: usize, seed: u64) -> Vec<f64> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut data: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();

    // Pre-sort the data
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    data
}

/// Test configuration for benchmarking
struct BenchmarkConfig {
    name: &'static str,
    dataset_sizes: Vec<(usize, &'static str)>, // (size, cache_level)
    num_datasets: Vec<usize>,
    num_quantiles: Vec<usize>,
}

/// Run benchmarks with a specific engine type
fn run_with_engine<E: ExecutionEngine<f64>>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    config: &BenchmarkConfig,
    engine: E,
    engine_name: &str,
) {
    let hd = trimmed_harrell_davis(engine.clone(), SqrtWidth);

    for &(n, cache_level) in &config.dataset_sizes {
        for &n_datasets in &config.num_datasets {
            for &n_quantiles in &config.num_quantiles {
                // Create a fresh cache for each configuration to avoid cache pollution
                let cache = UnifiedWeightCache::new(
                    TrimmedHDWeightComputer::new(SqrtWidth),
                    CachePolicy::Lru {
                        max_entries: 10 * 1024 * 1024,
                    },
                );
                // Generate datasets
                let datasets: Vec<Vec<f64>> = (0..n_datasets)
                    .map(|i| generate_test_data(n, i as u64))
                    .collect();
                let dataset_refs: Vec<&[f64]> = datasets.iter().map(|d| d.as_slice()).collect();

                // Generate quantiles
                let quantiles: Vec<f64> = (1..=n_quantiles)
                    .map(|i| i as f64 / (n_quantiles + 1) as f64)
                    .collect();

                group.throughput(Throughput::Elements((n_datasets * n_quantiles) as u64));

                // Test different strategies
                let strategies = vec![
                    (ProcessingStrategy::DatasetMajor, "DatasetMajor"),
                    (ProcessingStrategy::ParameterMajor, "ParameterMajor"),
                    (ProcessingStrategy::TiledTileMajor, "TiledTileMajor"),
                    (ProcessingStrategy::TiledDatasetMajor, "TiledDatasetMajor"),
                ];

                for (strategy, strategy_name) in strategies {
                    // Encode all metadata in the benchmark ID for Criterion's CSV export
                    // Format that will be parseable from CSV
                    let bench_id = format!(
                        "{}/{}/{}/n{}/d{}/q{}/t{}",
                        strategy_name,
                        engine_name,
                        cache_level,
                        n,
                        n_datasets,
                        n_quantiles,
                        engine.num_threads()
                    );
                    group.bench_with_input(
                        BenchmarkId::from_parameter(bench_id),
                        &(&dataset_refs, &quantiles),
                        |b, (datasets, quantiles)| {
                            b.iter(|| {
                                black_box(
                                    hd.process_batch_sorted(datasets, quantiles, &cache, strategy),
                                )
                            });
                        },
                    );
                }
            }
        }
    }
}

/// Benchmark cache-aware configurations
fn bench_cache_aware(c: &mut Criterion) {
    let cache_topology = CacheTopology::detect();
    eprintln!("Detected cache topology: {:?}", cache_topology);

    let mut group = c.benchmark_group("cache_aware_dispatch");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(50); // Reduce sample size for more configurations

    // Get cache-aware test sizes
    let cache_test_sizes = cache_topology.get_test_sizes();

    // Test configurations with cache-aware sizes
    // Each configuration should test different combinations to avoid duplicates
    let configs = vec![
        BenchmarkConfig {
            name: "cache_fit_many_quantiles",
            dataset_sizes: cache_test_sizes.clone(),
            num_datasets: vec![1, 5, 20],
            num_quantiles: vec![10, 50, 100],
        },
        BenchmarkConfig {
            name: "cache_boundary_transitions",
            dataset_sizes: cache_test_sizes
                .iter()
                .filter(|(_, level)| level.starts_with("L2"))
                .cloned()
                .collect(),
            num_datasets: vec![10],
            num_quantiles: vec![25, 75],
        },
        BenchmarkConfig {
            name: "varying_parallelism",
            dataset_sizes: cache_test_sizes
                .iter()
                .filter(|(_, level)| level.starts_with("L3"))
                .cloned()
                .take(2) // Just test first 2 L3 sizes
                .collect(),
            num_datasets: vec![20, 50],
            num_quantiles: vec![200],
        },
    ];

    // Run benchmarks with different engine types
    // We'll only run the first config to avoid duplicate IDs
    let config = &configs[0];

    // Sequential engines
    run_with_engine(&mut group, config, scalar_sequential(), "scalar_seq");
    run_with_engine(&mut group, config, simd_sequential(), "simd_seq");

    // Parallel engines (if feature enabled)
    #[cfg(feature = "parallel")]
    {
        run_with_engine(&mut group, config, scalar_parallel(), "scalar_par");
        run_with_engine(&mut group, config, simd_parallel(), "simd_par");
    }

    group.finish();
}

/// Print cache topology information
fn print_cache_info() {
    let topology = CacheTopology::detect();
    println!("\nCache Topology:");
    println!("  L1 Data: {} KB", topology.l1_data_size / 1024);
    println!("  L2: {} KB", topology.l2_size / 1024);
    println!("  L3: {} MB", topology.l3_size / 1024 / 1024);
    println!("  Cache Line: {} bytes", topology.cache_line_size);
    println!("\nTest dataset sizes:");
    for (size, level) in topology.get_test_sizes() {
        println!("  {}: {} elements ({} KB)", level, size, size * 8 / 1024);
    }
    println!();
}

/// Benchmark varying-sized datasets (DatasetMajor should win)
fn bench_varying_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("varying_sizes");
    group.measurement_time(Duration::from_secs(5));

    let engine = simd_sequential();
    let hd = harrell_davis(engine);
    let cache = UnifiedWeightCache::new(
        HDWeightComputer::new(),
        CachePolicy::Lru {
            max_entries: 1024 * 1024,
        },
    );

    // Create datasets with different sizes
    let size_patterns = vec![
        ("uniform", vec![10_000; 10]),
        (
            "slight_variation",
            vec![
                9_000, 9_500, 10_000, 10_500, 11_000, 9_000, 9_500, 10_000, 10_500, 11_000,
            ],
        ),
        (
            "high_variation",
            vec![
                1_000, 5_000, 10_000, 20_000, 50_000, 2_000, 8_000, 15_000, 30_000, 40_000,
            ],
        ),
    ];

    let quantile_counts = vec![10, 25, 50];

    for (pattern_name, sizes) in size_patterns {
        for &n_quantiles in &quantile_counts {
            // Generate datasets with specified sizes
            let datasets: Vec<Vec<f64>> = sizes
                .iter()
                .enumerate()
                .map(|(i, &size)| generate_test_data(size, i as u64))
                .collect();
            let dataset_refs: Vec<&[f64]> = datasets.iter().map(|d| d.as_slice()).collect();

            // Generate quantiles
            let quantiles: Vec<f64> = (1..=n_quantiles)
                .map(|i| i as f64 / (n_quantiles + 1) as f64)
                .collect();

            let bench_name = format!("{}_q{}", pattern_name, n_quantiles);

            group.throughput(Throughput::Elements((datasets.len() * n_quantiles) as u64));

            // Compare strategies
            // For varying sizes, tiling shouldn't work, so we test DatasetMajor vs Auto
            for strategy in [ProcessingStrategy::DatasetMajor, ProcessingStrategy::Auto] {
                let strategy_name = match strategy {
                    ProcessingStrategy::DatasetMajor => "DatasetMajor",
                    ProcessingStrategy::Auto => "Auto",
                    _ => unreachable!(),
                };

                group.bench_with_input(
                    BenchmarkId::new(strategy_name, &bench_name),
                    &(&dataset_refs, &quantiles),
                    |b, (datasets, quantiles)| {
                        b.iter(|| {
                            let result =
                                hd.process_batch_sorted(datasets, quantiles, &cache, strategy);
                            black_box(result)
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark the threshold for when tiling becomes beneficial
fn bench_tiling_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiling_threshold");
    group.measurement_time(Duration::from_secs(5));

    let engine = simd_sequential();
    let hd = harrell_davis(engine);

    // Test around the current threshold: n >= 10000 && num_quantiles >= 50
    let test_points = vec![
        // (dataset_size, num_quantiles)
        (5_000, 50),
        (8_000, 50),
        (10_000, 30),
        (10_000, 40),
        (10_000, 50),
        (10_000, 60),
        (10_000, 100),
        (15_000, 50),
        (20_000, 50),
    ];

    let n_datasets = 5; // Fixed number of datasets

    for (n, n_quantiles) in test_points {
        // Create a fresh cache for each configuration
        let cache = UnifiedWeightCache::new(
            HDWeightComputer::new(),
            CachePolicy::Lru {
                max_entries: 1024 * 1024,
            },
        );

        // Generate datasets
        let datasets: Vec<Vec<f64>> = (0..n_datasets)
            .map(|i| generate_test_data(n, i as u64))
            .collect();
        let dataset_refs: Vec<&[f64]> = datasets.iter().map(|d| d.as_slice()).collect();

        // Generate quantiles
        let quantiles: Vec<f64> = (1..=n_quantiles)
            .map(|i| i as f64 / (n_quantiles + 1) as f64)
            .collect();

        let bench_name = format!("n{}_q{}", n, n_quantiles);

        group.throughput(Throughput::Elements((n_datasets * n_quantiles) as u64));

        // Compare tiling strategies vs DatasetMajor
        for strategy in [
            ProcessingStrategy::DatasetMajor,
            ProcessingStrategy::TiledTileMajor,
            ProcessingStrategy::TiledDatasetMajor,
            ProcessingStrategy::Auto,
        ] {
            let strategy_name = match strategy {
                ProcessingStrategy::DatasetMajor => "DatasetMajor",
                ProcessingStrategy::TiledTileMajor => "TiledTileMajor",
                ProcessingStrategy::TiledDatasetMajor => "TiledDatasetMajor",
                ProcessingStrategy::Auto => "Auto",
                _ => unreachable!(),
            };

            group.bench_with_input(
                BenchmarkId::new(strategy_name, &bench_name),
                &(&dataset_refs, &quantiles),
                |b, (datasets, quantiles)| {
                    b.iter(|| {
                        let result = hd.process_batch_sorted(datasets, quantiles, &cache, strategy);
                        black_box(result)
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark tile size impact
fn bench_tile_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("tile_sizes");
    group.measurement_time(Duration::from_secs(5));

    // This would require modifying the tile size in TiledSparseMatrix
    // For now, we'll skip this benchmark but note it as future work

    group.finish();
}

criterion_group!(
    benches,
    bench_cache_aware,
    bench_varying_sizes,
    bench_tiling_threshold,
);

fn main() {
    print_cache_info();

    println!("\nTo export benchmark results to CSV, run:");
    println!("  cargo bench --bench dispatch_strategy_bench -- --save-baseline my_run");
    println!("\nCriterion will save results to:");
    println!("  - HTML report: target/criterion/");
    println!("  - Raw data: target/criterion/<benchmark_name>/my_run/");
    println!("\nTo compare against a baseline:");
    println!("  cargo bench --bench dispatch_strategy_bench -- --baseline my_run");

    benches();
}
