//! Statistical equivalence tests comparing optimized implementations
//! with naive reference implementations

use approx::assert_relative_eq;
use robust_core::{simd_sequential, CachePolicy, ProcessingStrategy, BatchProcessor, UnifiedWeightCache};
use robust_quantile::QuantileKernel;
use robust_quantile::{
    ConstantWidth, HarrellDavis, LinearWidth, QuantileEstimator, SqrtWidth, TrimmedHarrellDavis,
    HDWeightComputer, TrimmedHDWeightComputer,
};
// Import naive reference implementations
#[path = "../src/naive_reference.rs"]
mod naive_reference;

use naive_reference::{NaiveHarrellDavis, NaiveTrimmedHarrellDavis};

/// Test parameters for equivalence testing
struct TestParams {
    /// Relative tolerance for comparing results
    rel_tolerance: f64,
    /// Absolute tolerance for near-zero values
    abs_tolerance: f64,
    /// Random seed for reproducibility
    seed: u64,
}

impl Default for TestParams {
    fn default() -> Self {
        Self {
            rel_tolerance: 1e-10, // Very tight tolerance
            abs_tolerance: 1e-10,
            seed: 42,
        }
    }
}

/// Generate test data with various characteristics
fn generate_test_datasets() -> Vec<(&'static str, Vec<f64>)> {
    vec![
        // Small datasets
        ("tiny", vec![1.0, 2.0, 3.0]),
        ("small", vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        // Regular patterns
        ("uniform_10", (1..=10).map(|x| x as f64).collect()),
        ("uniform_100", (1..=100).map(|x| x as f64).collect()),
        // Random data (with fixed seed)
        ("random_normal", generate_normal(100, 0.0, 1.0, 42)),
        ("random_uniform", generate_uniform(100, 0.0, 10.0, 42)),
        // Edge cases
        ("single", vec![42.0]),
        ("duplicates", vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
        ("with_outliers", {
            let mut data = generate_normal(100, 0.0, 1.0, 42);
            data[0] = -100.0; // Extreme outlier
            data[99] = 100.0; // Extreme outlier
            data
        }),
        // Different scales
        ("large_values", (1000..1100).map(|x| x as f64).collect()),
        ("small_values", (0..100).map(|x| x as f64 * 0.001).collect()),
    ]
}

/// Generate normal distribution data
fn generate_normal(n: usize, mean: f64, std_dev: f64, seed: u64) -> Vec<f64> {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(mean, std_dev).unwrap();

    (0..n).map(|_| normal.sample(&mut rng)).collect()
}

/// Generate uniform distribution data
fn generate_uniform(n: usize, min: f64, max: f64, seed: u64) -> Vec<f64> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(min..max)).collect()
}

#[test]
fn test_harrell_davis_equivalence() {
    let params = TestParams::default();
    let datasets = generate_test_datasets();

    // Create optimized estimator
    let engine = simd_sequential();
    let hd = HarrellDavis::new(engine, HDWeightComputer::new());
    let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::Lru { max_entries: 10240 });

    // Test various quantiles
    let test_quantiles = vec![
        0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 0.001, 0.999, // Extreme quantiles
        0.123, 0.456, 0.789, // Arbitrary values
    ];

    for (name, data) in datasets {
        for &p in &test_quantiles {
            // Skip invalid cases
            if data.len() == 1 && (p == 0.0 || p == 1.0) {
                continue;
            }

            // Compute using naive implementation
            let naive_result = NaiveHarrellDavis::quantile(&data, p);

            // Compute using optimized implementation
            let mut data_copy = data.clone();
            let opt_result = hd.quantile(&mut data_copy, p, &cache).unwrap();

            // For debugging failures
            if (opt_result - naive_result).abs() > params.abs_tolerance {
                eprintln!(
                    "Dataset: {}, p: {}, naive: {}, optimized: {}, diff: {:e}",
                    name,
                    p,
                    naive_result,
                    opt_result,
                    (opt_result - naive_result).abs()
                );
            }
            // Check equivalence
            assert_relative_eq!(
                opt_result,
                naive_result,
                epsilon = params.rel_tolerance,
                max_relative = params.rel_tolerance,
            );
        }
    }
}

#[test]
fn test_trimmed_hd_sqrt_equivalence() {
    let params = TestParams::default();
    let datasets = generate_test_datasets();

    // Create optimized estimator
    let engine = simd_sequential();
    let thd = TrimmedHarrellDavis::new(engine, TrimmedHDWeightComputer::new(SqrtWidth));
    let cache = UnifiedWeightCache::new(TrimmedHDWeightComputer::new(SqrtWidth), CachePolicy::Lru { max_entries: 1024 });

    let test_quantiles = vec![0.1, 0.25, 0.5, 0.75, 0.9];

    for (name, data) in datasets {
        // Skip tiny datasets where sqrt width might not make sense
        if data.len() < 4 {
            continue;
        }

        for &p in &test_quantiles {
            // Compute using naive implementation
            let naive_result = NaiveTrimmedHarrellDavis::quantile_sqrt(&data, p);

            // Compute using optimized implementation
            let mut data_copy = data.clone();
            let opt_result = thd.quantile(&mut data_copy, p, &cache).unwrap();

            // Check equivalence (slightly relaxed tolerance for trimmed)
            assert_relative_eq!(
                opt_result,
                naive_result,
                epsilon = params.rel_tolerance * 10.0,
                max_relative = params.rel_tolerance * 10.0,
            );
        }
    }
}

#[test]
fn test_trimmed_hd_linear_equivalence() {
    let params = TestParams::default();
    let datasets = generate_test_datasets();

    // Create optimized estimator
    let engine = simd_sequential();
    let thd = TrimmedHarrellDavis::new(engine, TrimmedHDWeightComputer::new(LinearWidth));
    let cache = UnifiedWeightCache::new(TrimmedHDWeightComputer::new(LinearWidth), CachePolicy::Lru { max_entries: 1024 });

    let test_quantiles = vec![0.1, 0.25, 0.5, 0.75, 0.9];

    for (name, data) in datasets {
        // Skip tiny datasets
        if data.len() < 4 {
            continue;
        }

        for &p in &test_quantiles {
            // Compute using naive implementation
            let naive_result = NaiveTrimmedHarrellDavis::quantile_linear(&data, p);

            // Compute using optimized implementation
            let mut data_copy = data.clone();
            let opt_result = thd.quantile(&mut data_copy, p, &cache).unwrap();

            // Check equivalence
            assert_relative_eq!(
                opt_result,
                naive_result,
                epsilon = params.rel_tolerance * 10.0,
                max_relative = params.rel_tolerance * 10.0,
            );
        }
    }
}

#[test]
fn test_trimmed_hd_constant_width_equivalence() {
    let params = TestParams::default();
    let datasets = generate_test_datasets();

    // Test various constant widths
    let test_widths = vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0];

    for width in test_widths {
        // Create optimized estimator
        let engine = simd_sequential();
        let thd = TrimmedHarrellDavis::new(engine, TrimmedHDWeightComputer::new(ConstantWidth::new(width)));
        let cache = UnifiedWeightCache::new(TrimmedHDWeightComputer::new(ConstantWidth::new(width)), CachePolicy::Lru { max_entries: 1024 });

        let test_quantiles = vec![0.1, 0.25, 0.5, 0.75, 0.9];

        for (name, data) in &datasets {
            // Skip tiny datasets
            if data.len() < 4 {
                continue;
            }

            for &p in &test_quantiles {
                // Compute using naive implementation
                let naive_result = NaiveTrimmedHarrellDavis::quantile(&data, p, width);

                // Compute using optimized implementation
                let mut data_copy = data.clone();
                let opt_result = thd.quantile(&mut data_copy, p, &cache).unwrap();

                // Check equivalence
                assert_relative_eq!(
                    opt_result,
                    naive_result,
                    epsilon = params.rel_tolerance * 10.0,
                    max_relative = params.rel_tolerance * 10.0,
                );
            }
        }
    }
}

#[test]
fn test_moments_equivalence() {
    let params = TestParams::default();
    let datasets = generate_test_datasets();

    // We need to test that the moments computation matches
    let engine = simd_sequential();
    let kernel = robust_quantile::kernels::WeightedSumKernel::new(engine);

    for (name, data) in datasets {
        if data.len() < 2 {
            continue;
        }

        let mut sorted = data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for &p in &[0.25, 0.5, 0.75] {
            // Get naive moments
            let (naive_c1, naive_c2) = NaiveHarrellDavis::quantile_with_moments(&data, p);

            // Get optimized moments
            let weights = robust_quantile::weights::compute_hd_weights(data.len(), p);
            let (opt_c1, opt_c2) = kernel.apply_sparse_weights_with_moments(&sorted, &weights);

            // Check first moment (the quantile itself)
            assert_relative_eq!(
                opt_c1,
                naive_c1,
                epsilon = params.rel_tolerance,
                max_relative = params.rel_tolerance,
            );

            // Check second moment
            assert_relative_eq!(
                opt_c2,
                naive_c2,
                epsilon = params.rel_tolerance,
                max_relative = params.rel_tolerance,
            );
        }
    }
}

#[test]
fn test_edge_cases_equivalence() {
    let params = TestParams::default();

    // Test specific edge cases
    let edge_cases = vec![
        // Empty data is handled by error, not tested here

        // Single element
        (vec![42.0], vec![0.0, 0.5, 1.0]),
        // Two elements
        (vec![1.0, 2.0], vec![0.0, 0.5, 1.0]),
        // All same values
        (vec![5.0; 10], vec![0.0, 0.5, 1.0]),
        // Extreme values
        (
            vec![f64::MIN_POSITIVE, 0.0, f64::MAX / 2.0],
            vec![0.1, 0.5, 0.9],
        ),
    ];

    let engine = simd_sequential();
    let hd = HarrellDavis::new(engine, HDWeightComputer::new());
    let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);

    for (data, quantiles) in edge_cases {
        for &p in &quantiles {
            let naive_result = NaiveHarrellDavis::quantile(&data, p);
            let mut data_copy = data.clone();
            let opt_result = hd.quantile(&mut data_copy, p, &cache).unwrap();

            // For edge cases, we might need more tolerance
            if naive_result.is_finite() && opt_result.is_finite() {
                assert_relative_eq!(
                    opt_result,
                    naive_result,
                    epsilon = params.rel_tolerance * 100.0,
                    max_relative = params.rel_tolerance * 100.0,
                );
            }
        }
    }
}

#[cfg(feature = "simd")]
#[test]
fn test_simd_equivalence() {
    let params = TestParams::default();
    let datasets = generate_test_datasets();

    // Create SIMD estimator
    let engine = simd_sequential();
    let hd = HarrellDavis::new(engine, HDWeightComputer::new());
    let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::Lru { max_entries: 10240 });

    // Test various quantiles
    let test_quantiles = vec![0.1, 0.25, 0.5, 0.75, 0.9];

    for (name, data) in datasets {
        for &p in &test_quantiles {
            // Compute using naive implementation
            let naive_result = NaiveHarrellDavis::quantile(&data, p);

            // Compute using SIMD implementation
            let mut data_copy = data.clone();
            let simd_result = hd.quantile(&mut data_copy, p, &cache).unwrap();

            // Check equivalence
            if (simd_result - naive_result).abs() > params.rel_tolerance {
                panic!(
                    "SIMD mismatch for dataset '{}', p={}: simd={}, naive={}, diff={}",
                    name, p, simd_result, naive_result, (simd_result - naive_result).abs()
                );
            }
            assert_relative_eq!(
                simd_result,
                naive_result,
                epsilon = params.rel_tolerance,
                max_relative = params.rel_tolerance
            );
        }
    }
}

#[cfg(feature = "avx2")]
#[test]
fn test_avx2_specific_equivalence() {
    use robust_core::{
        execution::SequentialEngine,
        primitives::Avx2Backend,
    };
    
    // Only run if AVX2 is actually available
    if !is_x86_feature_detected!("avx2") {
        eprintln!("AVX2 not available on this CPU, skipping test");
        return;
    }
    
    let params = TestParams::default();
    let datasets = generate_test_datasets();

    // Create AVX2-specific estimator
    let avx2_backend = Avx2Backend::new();
    let engine = SequentialEngine::new(avx2_backend);
    let hd = HarrellDavis::new(engine, HDWeightComputer::new());
    let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::Lru { max_entries: 10240 });

    // Focus on larger datasets where AVX2 shines
    let large_datasets = vec![
        ("large_normal", generate_normal(10000, 0.0, 1.0, 123)),
        ("large_uniform", generate_uniform(10000, -10.0, 10.0, 456)),
    ];

    let test_quantiles = vec![0.05, 0.25, 0.5, 0.75, 0.95];

    for (name, data) in large_datasets {
        for &p in &test_quantiles {
            // Compute using naive implementation
            let naive_result = NaiveHarrellDavis::quantile(&data, p);

            // Compute using AVX2 implementation
            let mut data_copy = data.clone();
            let avx2_result = hd.quantile(&mut data_copy, p, &cache).unwrap();

            // Check equivalence
            if (avx2_result - naive_result).abs() > params.rel_tolerance * 10.0 {
                panic!(
                    "AVX2 mismatch for dataset '{}', p={}: avx2={}, naive={}, diff={}",
                    name, p, avx2_result, naive_result, (avx2_result - naive_result).abs()
                );
            }
            assert_relative_eq!(
                avx2_result,
                naive_result,
                epsilon = params.rel_tolerance * 10.0, // Slightly more tolerance for AVX2
                max_relative = params.rel_tolerance * 10.0
            );
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_equivalence() {
    use robust_core::execution::{scalar_parallel, simd_parallel};
    
    let params = TestParams::default();
    
    // Generate multiple datasets for batch processing
    let n_datasets = 20;
    let dataset_size = 1000;
    let datasets: Vec<Vec<f64>> = (0..n_datasets)
        .map(|i| generate_normal(dataset_size, i as f64, 1.0, i as u64))
        .collect();
    
    let quantiles = vec![0.1, 0.25, 0.5, 0.75, 0.9];
    
    // Test scalar parallel
    {
        let engine = scalar_parallel();
        let hd = HarrellDavis::new(engine, HDWeightComputer::new());
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::Lru { max_entries: 10240 });
        
        // Process each dataset individually (reference)
        let reference_results: Vec<Vec<f64>> = datasets
            .iter()
            .map(|data| {
                quantiles
                    .iter()
                    .map(|&p| NaiveHarrellDavis::quantile(data, p))
                    .collect()
            })
            .collect();
        
        // Process using parallel batch
        let mut dataset_copies: Vec<Vec<f64>> = datasets.clone();
        let mut dataset_refs: Vec<&mut [f64]> = dataset_copies.iter_mut().map(|d| d.as_mut_slice()).collect();
        let parallel_results = hd
            .process_batch(&mut dataset_refs, &quantiles, &cache, ProcessingStrategy::Auto)
            .unwrap();
        
        // Check equivalence
        for (i, (ref_res, par_res)) in reference_results.iter().zip(parallel_results.iter()).enumerate() {
            for (j, (&ref_val, &par_val)) in ref_res.iter().zip(par_res.iter()).enumerate() {
                assert_relative_eq!(
                    par_val,
                    ref_val,
                    epsilon = params.rel_tolerance,
                    max_relative = params.rel_tolerance
                );
            }
        }
    }
    
    // Test SIMD parallel if available
    #[cfg(feature = "simd")]
    {
        let engine = simd_parallel();
        let hd = HarrellDavis::new(engine, HDWeightComputer::new());
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::Lru { max_entries: 10240 });
        
        // Process using SIMD parallel batch
        let mut dataset_copies: Vec<Vec<f64>> = datasets.clone();
        let mut dataset_refs: Vec<&mut [f64]> = dataset_copies.iter_mut().map(|d| d.as_mut_slice()).collect();
        let simd_parallel_results = hd
            .process_batch(&mut dataset_refs, &quantiles, &cache, ProcessingStrategy::ParameterMajor)
            .unwrap();
        
        // Check against reference
        for (i, data) in datasets.iter().enumerate() {
            for (j, &p) in quantiles.iter().enumerate() {
                let ref_val = NaiveHarrellDavis::quantile(data, p);
                let simd_val = simd_parallel_results[i][j];
                
                assert_relative_eq!(
                    simd_val,
                    ref_val,
                    epsilon = params.rel_tolerance,
                    max_relative = params.rel_tolerance
                );
            }
        }
    }
}

#[cfg(all(feature = "simd", feature = "parallel"))]
#[test]
fn test_processing_strategies_equivalence() {
    use robust_core::execution::simd_parallel;
    
    let params = TestParams::default();
    
    // Create test datasets
    let datasets: Vec<Vec<f64>> = vec![
        generate_normal(500, 0.0, 1.0, 1),
        generate_normal(500, 5.0, 2.0, 2),
        generate_uniform(500, -10.0, 10.0, 3),
        generate_uniform(500, 0.0, 100.0, 4),
    ];
    
    let quantiles = vec![0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];
    
    let engine = simd_parallel();
    let hd = HarrellDavis::new(engine, HDWeightComputer::new());
    let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::Lru { max_entries: 10240 });
    
    // Prepare dataset copies for mutable processing
    let mut dataset_copies1: Vec<Vec<f64>> = datasets.clone();
    let mut dataset_refs1: Vec<&mut [f64]> = dataset_copies1.iter_mut().map(|d| d.as_mut_slice()).collect();
    
    // Test DatasetMajor strategy
    let dataset_major_results = hd
        .process_batch(&mut dataset_refs1, &quantiles, &cache, ProcessingStrategy::DatasetMajor)
        .unwrap();
    
    // Test ParameterMajor strategy
    let mut dataset_copies2: Vec<Vec<f64>> = datasets.clone();
    let mut dataset_refs2: Vec<&mut [f64]> = dataset_copies2.iter_mut().map(|d| d.as_mut_slice()).collect();
    let parameter_major_results = hd
        .process_batch(&mut dataset_refs2, &quantiles, &cache, ProcessingStrategy::ParameterMajor)
        .unwrap();
    
    // Test Tiled strategy
    let mut dataset_copies3: Vec<Vec<f64>> = datasets.clone();
    let mut dataset_refs3: Vec<&mut [f64]> = dataset_copies3.iter_mut().map(|d| d.as_mut_slice()).collect();
    let tiled_results = hd
        .process_batch(&mut dataset_refs3, &quantiles, &cache, ProcessingStrategy::TiledTileMajor)
        .unwrap();
    
    // Test Auto strategy
    let mut dataset_copies4: Vec<Vec<f64>> = datasets.clone();
    let mut dataset_refs4: Vec<&mut [f64]> = dataset_copies4.iter_mut().map(|d| d.as_mut_slice()).collect();
    let auto_results = hd
        .process_batch(&mut dataset_refs4, &quantiles, &cache, ProcessingStrategy::Auto)
        .unwrap();
    
    // All strategies should produce identical results
    for i in 0..datasets.len() {
        for j in 0..quantiles.len() {
            let dm_val = dataset_major_results[i][j];
            let pm_val = parameter_major_results[i][j];
            let tiled_val = tiled_results[i][j];
            let auto_val = auto_results[i][j];
            
            assert_relative_eq!(
                dm_val,
                pm_val,
                epsilon = params.rel_tolerance,
                max_relative = params.rel_tolerance
            );
            
            assert_relative_eq!(
                dm_val,
                tiled_val,
                epsilon = params.rel_tolerance,
                max_relative = params.rel_tolerance
            );
            
            assert_relative_eq!(
                dm_val,
                auto_val,
                epsilon = params.rel_tolerance,
                max_relative = params.rel_tolerance
            );
        }
    }
}

#[test]
fn test_harrell_davis_tiled_equivalence() {
    let params = TestParams::default();
    
    // Create test scenarios that benefit from tiling:
    // - Many quantiles (good for tiling)
    // - Large datasets (better cache utilization)
    let test_cases = vec![
        // (name, dataset, quantiles)
        ("small_many_quantiles", vec![1.0, 2.0, 3.0, 4.0, 5.0], (1..=99).map(|i| i as f64 / 100.0).collect::<Vec<_>>()),
        ("medium_percentiles", generate_normal(1000, 0.0, 1.0, 42), (1..=99).map(|i| i as f64 / 100.0).collect()),
        ("large_deciles", generate_uniform(10000, -100.0, 100.0, 123), (1..=9).map(|i| i as f64 / 10.0).collect()),
        ("large_fine_grained", generate_normal(5000, 50.0, 10.0, 456), (5..=95).step_by(5).map(|i| i as f64 / 100.0).collect()),
    ];
    
    let engine = simd_sequential();
    let hd = HarrellDavis::new(engine, HDWeightComputer::new());
    let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::Lru { max_entries: 10240 });
    
    for (name, data, quantiles) in test_cases {
        // Compute reference values using naive implementation
        let naive_results: Vec<f64> = quantiles
            .iter()
            .map(|&p| NaiveHarrellDavis::quantile(&data, p))
            .collect();
        
        // Test with Tiled strategy
        let mut data_copy = data.clone();
        data_copy.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let datasets = vec![data_copy.as_slice()];
        let tiled_results = hd
            .process_batch_sorted(&datasets, &quantiles, &cache, ProcessingStrategy::TiledTileMajor)
            .unwrap();
        
        // Compare results
        for (i, (&naive_val, &tiled_val)) in naive_results.iter().zip(tiled_results[0].iter()).enumerate() {
            if (tiled_val - naive_val).abs() > params.rel_tolerance * 10.0 {
                panic!(
                    "Mismatch in dataset '{}' at quantile index {}: tiled={}, naive={}, diff={}",
                    name, i, tiled_val, naive_val, (tiled_val - naive_val).abs()
                );
            }
            assert_relative_eq!(
                tiled_val,
                naive_val,
                epsilon = params.rel_tolerance * 10.0,
                max_relative = params.rel_tolerance * 10.0
            );
        }
    }
}

#[test]
fn test_trimmed_hd_tiled_equivalence() {
    let params = TestParams::default();
    
    // Test tiled processing for Trimmed HD with different width functions
    let test_cases = vec![
        ("many_quantiles_sqrt", generate_normal(1000, 0.0, 1.0, 42), (10..=90).step_by(5).map(|i| i as f64 / 100.0).collect::<Vec<_>>()),
        ("percentiles_linear", generate_uniform(2000, -50.0, 50.0, 123), (1..=99).map(|i| i as f64 / 100.0).collect()),
        ("large_quartiles", generate_normal(5000, 100.0, 20.0, 456), vec![0.25, 0.5, 0.75]),
    ];
    
    let engine = simd_sequential();
    
    // Test with sqrt width
    {
        let thd = TrimmedHarrellDavis::new(engine.clone(), TrimmedHDWeightComputer::new(SqrtWidth));
        let cache = UnifiedWeightCache::new(TrimmedHDWeightComputer::new(SqrtWidth), CachePolicy::Lru { max_entries: 10240 });
        
        for (name, data, quantiles) in &test_cases {
            if data.len() < 10 {
                continue; // Skip tiny datasets for trimmed estimators
            }
            
            // Compute reference values
            let naive_results: Vec<f64> = quantiles
                .iter()
                .map(|&p| NaiveTrimmedHarrellDavis::quantile_sqrt(&data, p))
                .collect();
            
            // Test with Tiled strategy
            let mut data_copy = data.clone();
            data_copy.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let datasets = vec![data_copy.as_slice()];
            let tiled_results = thd
                .process_batch_sorted(&datasets, &quantiles, &cache, ProcessingStrategy::TiledTileMajor)
                .unwrap();
            
            // Compare results
            for (i, (&naive_val, &tiled_val)) in naive_results.iter().zip(tiled_results[0].iter()).enumerate() {
                assert_relative_eq!(
                    tiled_val,
                    naive_val,
                    epsilon = params.rel_tolerance * 100.0,
                    max_relative = params.rel_tolerance * 100.0
                );
            }
        }
    }
    
    // Test with linear width
    {
        let thd = TrimmedHarrellDavis::new(engine.clone(), TrimmedHDWeightComputer::new(LinearWidth));
        let cache = UnifiedWeightCache::new(TrimmedHDWeightComputer::new(LinearWidth), CachePolicy::Lru { max_entries: 10240 });
        
        for (name, data, quantiles) in &test_cases {
            if data.len() < 10 {
                continue;
            }
            
            // Only test first few quantiles for linear width to save time
            let test_quantiles: Vec<f64> = quantiles.iter().take(10).copied().collect();
            
            // Compute reference values
            let naive_results: Vec<f64> = test_quantiles
                .iter()
                .map(|&p| NaiveTrimmedHarrellDavis::quantile_linear(&data, p))
                .collect();
            
            // Test with Tiled strategy
            let mut data_copy = data.clone();
            data_copy.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let datasets = vec![data_copy.as_slice()];
            let tiled_results = thd
                .process_batch_sorted(&datasets, &test_quantiles, &cache, ProcessingStrategy::TiledTileMajor)
                .unwrap();
            
            // Compare results
            for (i, (&naive_val, &tiled_val)) in naive_results.iter().zip(tiled_results[0].iter()).enumerate() {
                assert_relative_eq!(
                    tiled_val,
                    naive_val,
                    epsilon = params.rel_tolerance * 100.0,
                    max_relative = params.rel_tolerance * 100.0
                );
            }
        }
    }
}

#[test]
fn test_tiled_strategy_selection() {
    // Test that Auto strategy correctly selects Tiled when appropriate
    let engine = simd_sequential();
    let hd = HarrellDavis::new(engine, HDWeightComputer::new());
    let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::Lru { max_entries: 10240 });
    
    // Create a scenario where Tiled should be selected:
    // - Many quantiles (50+)
    // - Large datasets (10000+)
    // - All same size
    let large_datasets: Vec<Vec<f64>> = (0..5)
        .map(|i| generate_normal(10000, i as f64 * 10.0, 5.0, i as u64))
        .collect();
    
    let many_quantiles: Vec<f64> = (1..=99).map(|i| i as f64 / 100.0).collect();
    
    // Process with Auto strategy
    let mut auto_copies: Vec<Vec<f64>> = large_datasets.clone();
    let mut auto_refs: Vec<&mut [f64]> = auto_copies.iter_mut().map(|d| d.as_mut_slice()).collect();
    let auto_results = hd
        .process_batch(&mut auto_refs, &many_quantiles, &cache, ProcessingStrategy::Auto)
        .unwrap();
    
    // Process with explicit Tiled strategy
    let mut tiled_copies: Vec<Vec<f64>> = large_datasets.clone();
    let mut tiled_refs: Vec<&mut [f64]> = tiled_copies.iter_mut().map(|d| d.as_mut_slice()).collect();
    let tiled_results = hd
        .process_batch(&mut tiled_refs, &many_quantiles, &cache, ProcessingStrategy::TiledTileMajor)
        .unwrap();
    
    // Results should be identical if Auto correctly selected Tiled
    for i in 0..large_datasets.len() {
        for j in 0..many_quantiles.len() {
            assert_relative_eq!(
                auto_results[i][j],
                tiled_results[i][j],
                epsilon = 1e-15,
                max_relative = 1e-15
            );
        }
    }
}
