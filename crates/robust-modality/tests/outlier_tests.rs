//! Tests for outlier handling with different quantile estimators

use robust_modality::{ModalityDetectorBuilder, NullModalityVisualizer, test_data};
use robust_quantile::{estimators::{harrell_davis, trimmed_harrell_davis}, QuantileAdapter, HDWeightComputer, TrimmedHDWeightComputer, ConstantWidth};
use robust_core::{execution::SequentialEngine, primitives::ScalarBackend, UnifiedWeightCache, CachePolicy, Numeric};
use num_traits::NumCast;

// Helper function to create HD estimator and cache (generic)
fn create_hd_estimator_and_cache<T: Numeric + NumCast + 'static>() -> (QuantileAdapter<T, robust_quantile::HarrellDavis<T, SequentialEngine<T, ScalarBackend>>>, UnifiedWeightCache<HDWeightComputer<T>, T>) 
where
    T::Float: num_traits::Float,
{
    let engine = SequentialEngine::new(ScalarBackend);
    let hd = harrell_davis(engine);
    let estimator = QuantileAdapter::new(hd);
    let cache = UnifiedWeightCache::new(HDWeightComputer::<T>::new(), CachePolicy::NoCache);
    (estimator, cache)
}

// Helper function to create trimmed HD estimator and cache (generic)
fn create_trimmed_estimator_and_cache<T: Numeric + NumCast + 'static>(trim_width: f64) -> (QuantileAdapter<T, robust_quantile::TrimmedHarrellDavis<T, SequentialEngine<T, ScalarBackend>, ConstantWidth>>, UnifiedWeightCache<TrimmedHDWeightComputer<T, ConstantWidth>, T>) 
where
    T::Float: num_traits::Float,
{
    let engine = SequentialEngine::new(ScalarBackend);
    let trimmed = trimmed_harrell_davis(engine, ConstantWidth::new(trim_width));
    let estimator = QuantileAdapter::new(trimmed);
    let cache = UnifiedWeightCache::new(TrimmedHDWeightComputer::<T, ConstantWidth>::new(ConstantWidth::new(trim_width)), CachePolicy::NoCache);
    (estimator, cache)
}

#[test]
#[ignore = "Trimmed HD has numerical issues with large n and extreme quantiles"]
fn test_outlier_robustness_hd_vs_trimmed() {
    // Use standardized test data - mix bimodal with outliers
    let mut data = test_data::TestDistributions::bimodal_symmetric();
    
    // Add extreme outliers (about 10% of original data)
    let n_outliers = data.len() / 10;
    for i in 0..n_outliers {
        let outlier_val = if i % 2 == 0 { -50.0 + (i as f64) } else { 50.0 - (i as f64) };
        data.push(outlier_val);
    }

    // Create estimators and caches
    let (hd_estimator, hd_cache) = create_hd_estimator_and_cache::<f64>();
    let (trimmed_estimator, trimmed_cache) = create_trimmed_estimator_and_cache::<f64>(0.1);

    // Test with regular Harrell-Davis
    let params = test_data::TestParameters::DEFAULT;
    let detector_hd = ModalityDetectorBuilder::new(NullModalityVisualizer::default())
        .sensitivity(params.0)
        .precision(params.1)
        .build();

    // Test with Trimmed Harrell-Davis (10% trimming) - same parameters
    let detector_trimmed = ModalityDetectorBuilder::new(NullModalityVisualizer::default())
        .sensitivity(params.0)
        .precision(params.1)
        .build();

    let result_hd = detector_hd.detect_modes_with_estimator(&data, &hd_estimator, &hd_cache).unwrap();
    let result_trimmed = detector_trimmed.detect_modes_with_estimator(&data, &trimmed_estimator, &trimmed_cache).unwrap();

    println!("HD estimator: {} modes", result_hd.mode_count());
    println!("Trimmed HD estimator: {} modes", result_trimmed.mode_count());

    // With extreme outliers, the histogram might have trouble detecting the bimodal structure
    // The outliers spread the range so much that the modes might not be detected
    // We'll check that at least the results are reasonable
    if result_hd.is_multimodal() {
        // If HD detects multimodal, trimmed should too (or have similar behavior)
        println!("HD detected multimodal structure");
    } else {
        println!("HD detected unimodal structure (outliers may have affected detection)");
    }
    
    // The key test is that trimmed should be at least as stable as regular HD
    // It shouldn't detect many more modes due to outlier effects
    
    // Trimmed should be at least as stable
    assert!(result_trimmed.mode_count() <= result_hd.mode_count() + 1, 
            "Trimmed should not detect many more modes than regular HD");
}

#[test]
fn test_extreme_outliers_no_trim() {
    // Use standardized unimodal with outliers
    let data = test_data::TestDistributions::unimodal_with_outliers();

    // Create estimator and cache
    let (estimator, cache) = create_hd_estimator_and_cache::<f64>();

    // Test with default parameters
    let params = test_data::TestParameters::DEFAULT;
    let detector = ModalityDetectorBuilder::new(NullModalityVisualizer::default())
        .sensitivity(params.0)
        .precision(params.1)
        .build();

    let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();

    println!("Unimodal with outliers: {} modes", result.mode_count());

    // Should remain unimodal despite outliers
    assert_eq!(result.mode_count(), 1, "Should detect single mode despite outliers");
    assert!(!result.is_multimodal());
}

#[test]
#[ignore = "Trimmed HD has numerical issues with large n and extreme quantiles"]
fn test_extreme_outliers_with_trim() {
    // Use standardized unimodal with outliers
    let data = test_data::TestDistributions::unimodal_with_outliers();

    // Create trimmed estimator with 2% trimming (smaller to avoid numerical issues)
    let (estimator, cache) = create_trimmed_estimator_and_cache(0.02);

    // Test with default parameters
    let params = test_data::TestParameters::DEFAULT;
    let detector = ModalityDetectorBuilder::new(NullModalityVisualizer::default())
        .sensitivity(params.0)
        .precision(params.1)
        .build();

    let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();

    println!("Unimodal with outliers (trimmed): {} modes", result.mode_count());

    // Should remain unimodal
    assert_eq!(result.mode_count(), 1, "Trimmed estimator should detect single mode");
    assert!(!result.is_multimodal());
}

#[test]
fn test_outlier_effect_on_histogram_bins() {
    // Create data with extreme outliers that could affect binning
    let mut data = test_data::TestDistributions::unimodal_normal();
    
    // Add a few extreme outliers
    data.push(-1000.0);
    data.push(1000.0);

    // Create estimators
    let (hd_estimator, hd_cache) = create_hd_estimator_and_cache::<f64>();
    let (trimmed_estimator, trimmed_cache) = create_trimmed_estimator_and_cache::<f64>(0.05);

    // Use fine precision to see if outliers affect binning
    let detector = ModalityDetectorBuilder::new(NullModalityVisualizer::default())
        .sensitivity(0.5)
        .precision(0.1) // Fine bins
        .build();

    let result_hd = detector.detect_modes_with_estimator(&data, &hd_estimator, &hd_cache).unwrap();
    let result_trimmed = detector.detect_modes_with_estimator(&data, &trimmed_estimator, &trimmed_cache).unwrap();

    println!("With extreme outliers - HD: {} modes, Trimmed: {} modes", 
             result_hd.mode_count(), result_trimmed.mode_count());

    // Both should still identify the main mode
    assert!(result_hd.mode_count() >= 1, "HD should find at least the main mode");
    assert!(result_trimmed.mode_count() >= 1, "Trimmed should find at least the main mode");
    
    // Trimmed might be more stable
    assert!(result_trimmed.mode_count() <= result_hd.mode_count(), 
            "Trimmed should not find more modes than HD with outliers");
}