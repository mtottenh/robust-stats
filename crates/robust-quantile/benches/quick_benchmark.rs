//! Quick benchmark to compare scalar vs SIMD performance
//!
//! Run with: cargo bench -p robust-quantile --bench quick_benchmark

#[cfg(feature = "parallel")]
use robust_core::execution::simd_parallel;
use robust_core::{
    execution::{auto_engine, scalar_sequential, simd_sequential},
    BatchProcessor, CachePolicy, ExecutionEngine, ProcessingStrategy, UnifiedWeightCache,
};
use robust_quantile::{
    estimators::{self, SqrtWidth},
    HDWeightComputer, QuantileEstimator, TrimmedHDWeightComputer,
};
use std::time::Instant;

fn benchmark_engine<E: ExecutionEngine<f64>>(
    name: &str,
    engine: E,
    data: &[f64],
    iterations: usize,
) {
    let hd = estimators::trimmed_harrell_davis(engine, SqrtWidth);
    let computer = TrimmedHDWeightComputer::new(robust_quantile::weights::SqrtWidthFn);
    let cache = UnifiedWeightCache::new(
        computer,
        CachePolicy::Lru {
            max_entries: 1024 * 1024,
        },
    );
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Warmup
    for _ in 0..100_000 {
        let _ = hd.quantile_sorted(&sorted, 0.5, &cache);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hd.quantile_sorted(&sorted, 0.5, &cache).unwrap();
    }
    let duration = start.elapsed();

    let ops_per_sec = iterations as f64 / duration.as_secs_f64();
    let nanos_per_op = duration.as_nanos() as f64 / iterations as f64;

    // Choose appropriate unit based on time scale
    let (time_per_op, unit) = if nanos_per_op < 1000.0 {
        (nanos_per_op, "ns")
    } else if nanos_per_op < 1_000_000.0 {
        (nanos_per_op / 1000.0, "μs")
    } else {
        (nanos_per_op / 1_000_000.0, "ms")
    };

    println!(
        "{:15} {:10.2} ops/sec  ({:7.2} {})",
        name, ops_per_sec, time_per_op, unit
    );
}

fn main() {
    println!("Quick Quantile Benchmark - Scalar vs SIMD");
    println!("==========================================\n");

    // Generate test data
    let sizes = vec![100, 1_000, 10_000, 100_000, 1_000_000];
    let iterations = 1_000_000;

    for size in sizes {
        println!("\nSample size: {}", size);
        println!("{:15} {:>15}  {:>15}", "Engine", "Throughput", "Latency");
        println!("{:-<50}", "");

        // Generate random data
        let data: Vec<f64> = (0..size).map(|i| (i as f64 * 3.14159).sin()).collect();

        // Benchmark different engines
        benchmark_engine("Scalar", scalar_sequential(), &data, iterations);
        benchmark_engine("SIMD Sequential", simd_sequential(), &data, iterations);
        benchmark_engine("SIMD Auto", auto_engine(), &data, iterations);
    }

    // Also test batch quantiles
    println!("\n\nBatch Quantile Performance (n=1000, 99 quantiles)");
    println!("{:-<50}", "");

    let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 3.14159).sin()).collect();
    let quantiles: Vec<f64> = (1..100).map(|i| i as f64 / 100.0).collect();
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Test scalar engine
    {
        let hd = estimators::trimmed_harrell_davis(scalar_sequential(), SqrtWidth);
        let computer = TrimmedHDWeightComputer::new(robust_quantile::weights::SqrtWidthFn);
        let cache = UnifiedWeightCache::new(
            computer,
            CachePolicy::Lru {
                max_entries: 10240 * 1024,
            },
        );

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = hd.quantiles_sorted(&sorted, &quantiles, &cache).unwrap();
        }
        let duration = start.elapsed();

        let ops_per_sec = iterations as f64 / duration.as_secs_f64();
        let micros_per_batch = duration.as_micros() as f64 / iterations as f64;
        println!(
            "{:15} {:10.2} batch ops/sec ({:7.2} μs per {} quantiles)",
            "Scalar",
            ops_per_sec,
            micros_per_batch,
            quantiles.len()
        );
    }

    // Test SIMD sequential engine
    {
        let hd = estimators::trimmed_harrell_davis(simd_sequential(), SqrtWidth);
        let computer = TrimmedHDWeightComputer::new(robust_quantile::weights::SqrtWidthFn);
        let cache = UnifiedWeightCache::new(
            computer,
            CachePolicy::Lru {
                max_entries: 1024 * 1024,
            },
        );

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = hd.quantiles_sorted(&sorted, &quantiles, &cache).unwrap();
        }
        let duration = start.elapsed();

        let ops_per_sec = iterations as f64 / duration.as_secs_f64();
        let micros_per_batch = duration.as_micros() as f64 / iterations as f64;
        println!(
            "{:15} {:10.2} batch ops/sec ({:7.2} μs per {} quantiles)",
            "SIMD Sequential",
            ops_per_sec,
            micros_per_batch,
            quantiles.len()
        );
    }

    // Test auto engine
    {
        let hd = estimators::trimmed_harrell_davis(auto_engine(), SqrtWidth);
        let computer = TrimmedHDWeightComputer::new(robust_quantile::weights::SqrtWidthFn);
        let cache = UnifiedWeightCache::new(
            computer,
            CachePolicy::Lru {
                max_entries: 1024 * 1024,
            },
        );

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = hd.quantiles_sorted(&sorted, &quantiles, &cache).unwrap();
        }
        let duration = start.elapsed();

        let ops_per_sec = iterations as f64 / duration.as_secs_f64();
        let micros_per_batch = duration.as_micros() as f64 / iterations as f64;
        println!(
            "{:15} {:10.2} batch ops/sec ({:7.2} μs per {} quantiles)",
            "SIMD Auto",
            ops_per_sec,
            micros_per_batch,
            quantiles.len()
        );
    }

    // Test cache effectiveness
    println!("\n\nCache Effectiveness Test (n=1000, repeated quantiles)");
    println!("{:-<60}", "");

    let mut data: Vec<f64> = (0..1000).map(|i| (i as f64 * 3.14159).sin()).collect();
    // Pre-sort the data since we're testing cache effectiveness, not sorting
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let engine = simd_sequential();
    let hd = estimators::trimmed_harrell_davis(engine, SqrtWidth);

    // Test with no cache
    {
        let computer = TrimmedHDWeightComputer::new(robust_quantile::weights::SqrtWidthFn);
        let cache = UnifiedWeightCache::new(computer, CachePolicy::NoCache);
        let start = Instant::now();
        for i in 0..1000 {
            let p = match i % 3 {
                0 => 0.25,
                1 => 0.5,
                _ => 0.75,
            };
            let _ = hd.quantile_sorted(&data, p, &cache).unwrap();
        }
        let duration = start.elapsed();
        let micros = duration.as_micros() as f64;
        println!("{:20} {:8.2} μs", "No cache:", micros);
    }

    // Test with cache
    {
        let computer = TrimmedHDWeightComputer::new(robust_quantile::weights::SqrtWidthFn);
        let cache = UnifiedWeightCache::new(
            computer,
            CachePolicy::Lru {
                max_entries: 10240 * 1024,
            },
        );
        let start = Instant::now();
        for i in 0..1000 {
            let p = match i % 3 {
                0 => 0.25,
                1 => 0.5,
                _ => 0.75,
            };
            let _ = hd.quantile_sorted(&data, p, &cache).unwrap();
        }
        let duration = start.elapsed();
        let micros = duration.as_micros() as f64;
        println!("{:20} {:8.2} μs", "With cache:", micros);

        // Print cache stats
        let stats = cache.stats();
        println!("\nCache statistics:");
        println!("  Hits:   {} ({:.1}%)", stats.hits, stats.hit_rate * 100.0);
        println!("  Misses: {}", stats.misses);
        println!("  Memory: {} bytes", stats.memory_bytes);
    }

    // Compare sorted vs unsorted performance
    println!("\n\nSorting Cost Analysis (n=10000, 99 quantiles)");
    println!("{:-<60}", "");

    {
        let test_data: Vec<f64> = (0..10000).map(|i| (i as f64 * 3.14159).sin()).collect();
        let quantiles: Vec<f64> = (1..100).map(|i| i as f64 / 100.0).collect();

        let engine = simd_sequential();
        let hd = estimators::trimmed_harrell_davis(engine, SqrtWidth);
        let computer = TrimmedHDWeightComputer::new(robust_quantile::weights::SqrtWidthFn);
        let cache = UnifiedWeightCache::new(
            computer,
            CachePolicy::Lru {
                max_entries: 1024 * 1024,
            },
        );

        // Test with unsorted data (includes sorting time)
        let mut unsorted_copy = test_data.clone();
        let start = Instant::now();
        let _ = hd
            .quantiles(&mut unsorted_copy, &quantiles, &cache)
            .unwrap();
        let unsorted_duration = start.elapsed();

        // Test with pre-sorted data
        let mut sorted_data = test_data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let start = Instant::now();
        let _ = hd
            .quantiles_sorted(&sorted_data, &quantiles, &cache)
            .unwrap();
        let sorted_duration = start.elapsed();

        // Just measure sorting time
        let mut sort_test = test_data.clone();
        let start = Instant::now();
        sort_test.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sort_only_duration = start.elapsed();

        println!(
            "Unsorted data (sort + compute): {:8.2} μs",
            unsorted_duration.as_micros() as f64
        );
        println!(
            "Pre-sorted data (compute only):  {:8.2} μs",
            sorted_duration.as_micros() as f64
        );
        println!(
            "Sorting only:                    {:8.2} μs",
            sort_only_duration.as_micros() as f64
        );
        println!(
            "Sorting overhead:                {:.1}%",
            (sort_only_duration.as_secs_f64() / unsorted_duration.as_secs_f64()) * 100.0
        );
    }

    // Multiple datasets, multiple quantiles test
    println!("\n\nMultiple Datasets, Multiple Quantiles Performance");
    println!("{:-<60}", "");

    // Generate multiple datasets of the same size
    let n_datasets = 100;
    let dataset_size = 1_000_000;
    let datasets: Vec<Vec<f64>> = (0..n_datasets)
        .map(|i| {
            (0..dataset_size)
                .map(|j| ((i * dataset_size + j) as f64 * 3.14159).sin())
                .collect()
        })
        .collect();

    // Quantiles to compute for each dataset
    let quantiles: Vec<f64> = (1..100).map(|i| i as f64 / 100.0).collect();

    println!(
        "Computing {} quantiles on {} datasets (size={} each)",
        quantiles.len(),
        n_datasets,
        dataset_size
    );
    println!("{:20} {:>15} {:>15}", "Engine", "Total Time", "Per Dataset");
    println!("{:-<60}", "");

    // Test scalar engine
    {
        let hd = estimators::trimmed_harrell_davis(scalar_sequential(), SqrtWidth);
        let computer = TrimmedHDWeightComputer::new(robust_quantile::weights::SqrtWidthFn);
        let cache = UnifiedWeightCache::new(
            computer,
            CachePolicy::Lru {
                max_entries: 1024 * 1024,
            },
        );

        // Clone all datasets BEFORE timing
        let mut dataset_copies: Vec<Vec<f64>> = datasets.clone();

        let start = Instant::now();
        let all_results: Vec<Vec<f64>> = dataset_copies
            .iter_mut()
            .map(|data| hd.quantiles(data, &quantiles, &cache).unwrap())
            .collect();
        let duration = start.elapsed();

        let total_micros = duration.as_micros() as f64;
        let micros_per_dataset = total_micros / n_datasets as f64;
        println!(
            "{:20} {:>12.2} μs {:>12.2} μs",
            "Scalar", total_micros, micros_per_dataset
        );

        // Verify we got results
        assert_eq!(all_results.len(), n_datasets);
        assert_eq!(all_results[0].len(), quantiles.len());
    }

    // Test SIMD sequential engine
    {
        let hd = estimators::trimmed_harrell_davis(simd_sequential(), SqrtWidth);
        let computer = TrimmedHDWeightComputer::new(robust_quantile::weights::SqrtWidthFn);
        let cache = UnifiedWeightCache::new(
            computer,
            CachePolicy::Lru {
                max_entries: 1024 * 1024,
            },
        );

        // Clone all datasets BEFORE timing
        let mut dataset_copies: Vec<Vec<f64>> = datasets.clone();

        let start = Instant::now();
        let _all_results: Vec<Vec<f64>> = dataset_copies
            .iter_mut()
            .map(|data| hd.quantiles(data, &quantiles, &cache).unwrap())
            .collect();
        let duration = start.elapsed();

        let total_micros = duration.as_micros() as f64;
        let micros_per_dataset = total_micros / n_datasets as f64;
        println!(
            "{:20} {:>12.2} μs {:>12.2} μs",
            "SIMD Sequential", total_micros, micros_per_dataset
        );
    }

    // Test with ProcessingStrategy::DatasetMajor
    {
        println!("\nUsing BatchProcessor with different strategies:");

        #[cfg(feature = "parallel")]
        let engine = simd_parallel();
        #[cfg(not(feature = "parallel"))]
        let engine = simd_sequential();

        let hd = estimators::trimmed_harrell_davis(engine, SqrtWidth);
        let computer = TrimmedHDWeightComputer::new(robust_quantile::weights::SqrtWidthFn);
        let cache = UnifiedWeightCache::new(
            computer,
            CachePolicy::Lru {
                max_entries: 1024 * 1024,
            },
        );

        // First, let's test processing multiple datasets for same quantile (median)

        let mut dataset_copies: Vec<Vec<f64>> = datasets.clone();
        let mut dataset_refs: Vec<&mut [f64]> = dataset_copies
            .iter_mut()
            .map(|d| d.as_mut_slice())
            .collect();
        let start = Instant::now();
        let _median_results = hd
            .process_batch(
                &mut dataset_refs,
                &quantiles,
                &cache,
                ProcessingStrategy::DatasetMajor,
            )
            .unwrap();
        let duration = start.elapsed();

        let total_micros = duration.as_micros() as f64;
        let micros_per_dataset = total_micros / n_datasets as f64;
        println!(
            "{:20} {:>12.2} μs {:>12.2} μs",
            "Dataset Major", total_micros, micros_per_dataset
        );
        println!(
            "  Cache hits: {} ({:.1}%)",
            cache.stats().hits,
            cache.stats().hit_rate * 100.0
        );
        println!("  Cache misses: {}", cache.stats().misses);
        // Test ParameterMajor for same thing

        let mut dataset_copies: Vec<Vec<f64>> = datasets.clone();
        let mut dataset_refs: Vec<&mut [f64]> = dataset_copies
            .iter_mut()
            .map(|d| d.as_mut_slice())
            .collect();
        let start = Instant::now();
        let _median_results2 = hd
            .process_batch(
                &mut dataset_refs,
                &quantiles,
                &cache,
                ProcessingStrategy::ParameterMajor,
            )
            .unwrap();
        let duration = start.elapsed();

        let total_micros = duration.as_micros() as f64;
        let micros_per_dataset = total_micros / n_datasets as f64;
        println!(
            "{:20} {:>12.2} μs {:>12.2} μs",
            "Parameter Major", total_micros, micros_per_dataset
        );
        println!(
            "  Cache hits: {} ({:.1}%)",
            cache.stats().hits,
            cache.stats().hit_rate * 100.0
        );
        println!("  Cache misses: {}", cache.stats().misses);
    }

    // Test parallel engine if available
    #[cfg(feature = "parallel")]
    {
        let hd = estimators::trimmed_harrell_davis(simd_parallel(), SqrtWidth);
        let computer = TrimmedHDWeightComputer::new(robust_quantile::weights::SqrtWidthFn);
        let cache = UnifiedWeightCache::new(
            computer,
            CachePolicy::Lru {
                max_entries: 1024 * 1024,
            },
        );

        let mut dataset_copies: Vec<Vec<f64>> = datasets.clone();
        let mut dataset_refs: Vec<&mut [f64]> = dataset_copies
            .iter_mut()
            .map(|d| d.as_mut_slice())
            .collect();
        let start = Instant::now();
        let all_results: Vec<Vec<f64>> = hd
            .process_batch(
                &mut dataset_refs,
                &quantiles,
                &cache,
                ProcessingStrategy::ParameterMajor,
            )
            .unwrap();
        let duration = start.elapsed();

        let total_micros = duration.as_micros() as f64;
        let micros_per_dataset = total_micros / n_datasets as f64;
        println!(
            "{:20} {:>12.2} μs {:>12.2} μs",
            "SIMD Parallel", total_micros, micros_per_dataset
        );
        println!(
            "  Cache hits: {} ({:.1}%)",
            cache.stats().hits,
            cache.stats().hit_rate * 100.0
        );
        println!("  Cache misses: {}", cache.stats().misses);
    }

    // Also test cache behavior across datasets
    {
        println!("\nCache behavior across multiple datasets:");

        let hd = estimators::harrell_davis(simd_sequential());
        let cache = UnifiedWeightCache::new(
            HDWeightComputer::new(),
            CachePolicy::Lru {
                max_entries: 10 * 1024 * 1024,
            },
        );

        // Clone the first 10 datasets upfront
        let mut first_10_copies: Vec<Vec<f64>> = datasets.iter().take(10).cloned().collect();

        // Process first 10 datasets (timing this)
        let start = Instant::now();
        for data in first_10_copies.iter_mut() {
            let _ = hd.quantiles(data, &quantiles, &cache).unwrap();
        }
        let first_pass_duration = start.elapsed();

        let stats = cache.stats();
        let micros = first_pass_duration.as_micros() as f64;
        println!(
            "After 10 datasets: {} hits, {} misses ({:.1}% hit rate) - {:.2} μs",
            stats.hits,
            stats.misses,
            stats.hit_rate * 100.0,
            micros
        );

        // Process same 10 datasets again (they're already sorted from first pass)
        let start = Instant::now();
        for data in first_10_copies.iter() {
            // Use quantiles_sorted since data is already sorted
            let _ = hd.quantiles_sorted(data, &quantiles, &cache).unwrap();
        }
        let second_pass_duration = start.elapsed();

        let stats = cache.stats();
        let micros = second_pass_duration.as_micros() as f64;
        println!(
            "After repeat:      {} hits, {} misses ({:.1}% hit rate) - {:.2} μs",
            stats.hits,
            stats.misses,
            stats.hit_rate * 100.0,
            micros
        );

        println!(
            "Speedup from cache and pre-sorted data: {:.2}x",
            first_pass_duration.as_secs_f64() / second_pass_duration.as_secs_f64()
        );
    }
}
