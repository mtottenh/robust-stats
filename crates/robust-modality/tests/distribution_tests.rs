//! Tests with various synthetic distributions

use robust_modality::{detector_with_params, sensible_defaults, test_data};
use robust_quantile::{estimators::harrell_davis, QuantileAdapter, HDWeightComputer};
use robust_core::{execution::SequentialEngine, primitives::ScalarBackend, UnifiedWeightCache, CachePolicy, Numeric};
use num_traits::NumCast;

// Helper function to create estimator and cache for tests (generic version)
fn create_estimator_and_cache<T: Numeric + NumCast + 'static>() -> (QuantileAdapter<T, robust_quantile::HarrellDavis<T, SequentialEngine<T, ScalarBackend>>>, UnifiedWeightCache<HDWeightComputer<T>, T>) 
where
    T::Float: num_traits::Float,
{
    let engine = SequentialEngine::new(ScalarBackend);
    let hd = harrell_davis(engine);
    let estimator = QuantileAdapter::new(hd);
    let cache = UnifiedWeightCache::new(HDWeightComputer::<T>::new(), CachePolicy::NoCache);
    (estimator, cache)
}

// Helper for f64 (most common case)
fn create_f64_estimator_and_cache() -> (QuantileAdapter<f64, robust_quantile::HarrellDavis<f64, SequentialEngine<f64, ScalarBackend>>>, UnifiedWeightCache<HDWeightComputer<f64>, f64>) {
    create_estimator_and_cache::<f64>()
}

#[test]
fn test_clear_bimodal() {
    // Use standardized bimodal distribution
    let data = test_data::TestDistributions::bimodal_symmetric();

    // Create estimator and cache
    let (estimator, cache) = create_f64_estimator_and_cache();

    // First test with sensible defaults
    let detector = sensible_defaults();
    let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();
    println!(
        "Clear bimodal with sensible defaults: {} modes",
        result.mode_count()
    );

    // Should detect exactly 2 modes for well-separated bimodal
    assert_eq!(result.mode_count(), 2, "Should detect exactly 2 modes");
    assert!(result.is_multimodal());

    // Check mode locations
    let modes = result.modes();
    let locations: Vec<f64> = modes.iter().map(|m| m.location).collect();

    assert!(
        locations.iter().any(|&x| (x + 3.0).abs() < 1.0),
        "Should find mode near -3"
    );
    assert!(
        locations.iter().any(|&x| (x - 3.0).abs() < 1.0),
        "Should find mode near 3"
    );
}

#[test]
fn test_overlapping_bimodal() {
    // Use standardized overlapping bimodal distribution
    let data = test_data::TestDistributions::bimodal_overlapping();

    // Create estimator and cache
    let (estimator, cache) = create_f64_estimator_and_cache();

    // Test with different sensitivity levels
    let conservative = test_data::TestParameters::CONSERVATIVE;
    let detector_conservative = detector_with_params(conservative.0, conservative.1);
    let result_conservative = detector_conservative.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();

    let sensitive = test_data::TestParameters::SENSITIVE;
    let detector_sensitive = detector_with_params(sensitive.0, sensitive.1);
    let result_sensitive = detector_sensitive.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();

    println!(
        "Overlapping bimodal - conservative: {} modes, sensitive: {} modes",
        result_conservative.mode_count(),
        result_sensitive.mode_count()
    );

    // Sensitive detector should find at least as many modes
    assert!(
        result_sensitive.mode_count() >= result_conservative.mode_count(),
        "Sensitive detector should find at least as many modes"
    );
}

#[test]
fn test_trimodal_equal_weights() {
    // Use standardized trimodal distribution
    let data = test_data::TestDistributions::trimodal_symmetric();

    // Create estimator and cache
    let (estimator, cache) = create_f64_estimator_and_cache();

    let detector = sensible_defaults();
    let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();

    println!("Trimodal equal weights: {} modes", result.mode_count());

    // Should detect at least 3 modes
    assert!(result.mode_count() >= 3, "Should detect at least 3 modes");
    assert!(result.is_multimodal());

    // Check that modes are roughly where expected
    let modes = result.modes();
    let locations: Vec<f64> = modes.iter().map(|m| m.location).collect();

    assert!(
        locations.iter().any(|&x| (x + 4.0).abs() < 1.0),
        "Should find mode near -4"
    );
    assert!(
        locations.iter().any(|&x| x.abs() < 1.0),
        "Should find mode near 0"
    );
    assert!(
        locations.iter().any(|&x| (x - 4.0).abs() < 1.0),
        "Should find mode near 4"
    );
}

#[test]
fn test_skewed_distribution() {
    // Use standardized skewed unimodal distribution
    let data = test_data::TestDistributions::unimodal_skewed();

    // Create estimator and cache
    let (estimator, cache) = create_f64_estimator_and_cache();

    let detector = sensible_defaults();
    let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();

    println!("Skewed unimodal: {} modes", result.mode_count());

    // Should still detect as unimodal despite skewness
    assert_eq!(result.mode_count(), 1, "Skewed distribution should still be unimodal");
    assert!(!result.is_multimodal());
}

#[test]
fn test_multimodal_with_different_spreads() {
    // Use standardized complex multimodal distribution
    let data = test_data::TestDistributions::multimodal_complex();

    // Create estimator and cache
    let (estimator, cache) = create_f64_estimator_and_cache();

    // Try with sensitive parameters for better detection
    let sensitive_params = test_data::TestParameters::SENSITIVE;
    let detector = detector_with_params(sensitive_params.0, sensitive_params.1);
    let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();

    println!("Complex multimodal: {} modes", result.mode_count());

    // Should detect multiple modes (at least 2, but ideally more)
    assert!(result.mode_count() >= 2, "Should detect at least 2 modes in complex distribution");
    assert!(result.is_multimodal());
    
    // The broad mode at -1.0 might overlap with adjacent modes, making exact detection difficult
    // So we just verify it's multimodal
}

#[test] 
fn test_mixture_of_different_distributions() {
    // Create a custom mixture with specific properties
    let data = test_data::mixture_normal(
        &[0.0, 5.0, 10.0], // means
        &[1.0, 0.5, 2.0],  // stds
        &[0.4, 0.4, 0.2],  // weights
        1000,
        Some(42)
    );

    // Create estimator and cache
    let (estimator, cache) = create_f64_estimator_and_cache();

    let detector = sensible_defaults();
    let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();

    println!("Custom mixture: {} modes", result.mode_count());

    // Should detect at least 2 modes (the third might be too weak)
    assert!(result.mode_count() >= 2, "Should detect at least 2 modes in mixture");
    assert!(result.is_multimodal());
}

#[test]
fn test_beta_distribution_modes() {
    // Beta distributions can be unimodal or bimodal depending on parameters
    // Use our standardized unimodal skewed (which is a Beta(2,5))
    let data = test_data::TestDistributions::unimodal_skewed();

    // Create estimator and cache
    let (estimator, cache) = create_f64_estimator_and_cache();

    let detector = sensible_defaults();
    let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();

    // Beta(2,5) should be unimodal
    assert_eq!(result.mode_count(), 1, "Beta(2,5) should be unimodal");
    assert!(!result.is_multimodal());
}

#[test]
fn test_generic_types() {
    // Test with f32 data
    // Note: Since harrell_davis creates f64 estimators, we'll test type conversion
    let data_f32: Vec<f32> = vec![1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 9.0, 9.1, 9.2];
    let data_f64: Vec<f64> = data_f32.iter().map(|&x| x as f64).collect();
    let (estimator, cache) = create_f64_estimator_and_cache();
    let detector = sensible_defaults();
    let result = detector.detect_modes_with_estimator(&data_f64, &estimator, &cache).unwrap();
    assert!(result.mode_count() >= 2, "Should detect multiple modes in f32 data (converted to f64)");
    
    // Test with integer data (converted to f64)
    let data_i32: Vec<i32> = vec![1, 1, 1, 5, 5, 5, 9, 9, 9];
    let data_f64: Vec<f64> = data_i32.iter().map(|&x| x as f64).collect();
    let (estimator, cache) = create_f64_estimator_and_cache();
    let detector = sensible_defaults();
    let result = detector.detect_modes_with_estimator(&data_f64, &estimator, &cache).unwrap();
    assert!(result.mode_count() >= 1, "Should detect modes in i32 data (converted to f64)");
}