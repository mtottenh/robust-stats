//! Tests for different sensitivity settings

use robust_modality::{sensible_defaults, ModalityDetectorBuilder, NullModalityVisualizer, test_data};
use robust_quantile::{estimators::harrell_davis, QuantileAdapter, HDWeightComputer};
use robust_core::{execution::SequentialEngine, primitives::ScalarBackend, UnifiedWeightCache, CachePolicy, Numeric};
use num_traits::NumCast;

// Helper function to create estimator and cache for tests (generic)
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

/// Helper to create a detector with specific sensitivity
fn create_detector_with_sensitivity(sensitivity: f64) -> robust_modality::LowlandModalityDetector<f64, robust_histogram::QRDEBuilderWithSteps, NullModalityVisualizer> {
    ModalityDetectorBuilder::new(NullModalityVisualizer::default())
        .sensitivity(sensitivity)
        .precision(1.0 / 101.0) // Use sensible default precision (101 bins)
        .build()
}

#[test]
fn test_sensitivity_on_weak_modes() {
    // Use asymmetric bimodal distribution (70% / 30% split)
    let data = test_data::TestDistributions::bimodal_asymmetric();
    

    // Create estimator and cache
    let (estimator, cache) = create_f64_estimator_and_cache();

    // First test with sensible defaults
    let default_detector = sensible_defaults();
    let result_default = default_detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();
    println!(
        "Weak modes with sensible defaults: {} modes",
        result_default.mode_count()
    );

    // Low sensitivity (high threshold) should detect fewer modes
    let low_sens = create_detector_with_sensitivity(0.1);
    let result_low = low_sens.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();
    println!(
        "Weak modes with low sensitivity (0.1): {} modes",
        result_low.mode_count()
    );

    // High sensitivity (low threshold) should detect both modes
    let high_sens = create_detector_with_sensitivity(0.7);
    let result_high = high_sens.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();
    println!(
        "Weak modes with high sensitivity (0.7): {} modes",
        result_high.mode_count()
    );
    assert!(
        result_high.mode_count() >= 2,
        "High sensitivity should detect weak mode, but found {} modes",
        result_high.mode_count()
    );

    assert!(
        result_low.mode_count() <= result_high.mode_count(),
        "Low sensitivity should detect same or fewer modes than high sensitivity"
    );
}

#[test]
fn test_sensitivity_on_noise() {
    // Use unimodal distribution with heavy tails (which adds noise)
    let data = test_data::TestDistributions::unimodal_heavy_tails();
    

    // Create estimator and cache
    let (estimator, cache) = create_f64_estimator_and_cache();

    // First test with sensible defaults
    let default_detector = sensible_defaults();
    let result_default = default_detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();
    println!(
        "Noisy unimodal with sensible defaults: {} modes",
        result_default.mode_count()
    );

    // Very high sensitivity (low threshold) might detect noise as modes
    let very_high_sens = create_detector_with_sensitivity(0.95);
    let result_very_high = very_high_sens.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();

    // Moderate sensitivity should detect single mode
    let moderate_sens = create_detector_with_sensitivity(0.3);
    let result_moderate = moderate_sens.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();

    // Moderate sensitivity should find fewer modes than very high
    assert!(
        result_moderate.mode_count() <= result_very_high.mode_count(),
        "Moderate sensitivity should find fewer modes than very high sensitivity"
    );
}

#[test]
fn test_sensitivity_gradient() {
    // Use complex multimodal distribution with different mode strengths
    let data = test_data::TestDistributions::multimodal_complex();
    

    // Create estimator and cache
    let (estimator, cache) = create_f64_estimator_and_cache();

    // First test with sensible defaults
    let default_detector = sensible_defaults();
    let result_default = default_detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();
    println!(
        "Gradient modes with sensible defaults: {} modes",
        result_default.mode_count()
    );

    // Test different sensitivities (from low to high)
    let sensitivities = vec![0.05, 0.1, 0.2, 0.3, 0.4];
    let mut mode_counts = Vec::new();

    for &sens in &sensitivities {
        let detector = create_detector_with_sensitivity(sens);
        let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();
        mode_counts.push(result.mode_count());
        println!("Sensitivity {}: {} modes", sens, result.mode_count());
    }

    // Mode counts should be non-decreasing with increasing sensitivity value
    // (because higher value = easier to find lowlands = more modes)
    for i in 1..mode_counts.len() {
        assert!(
            mode_counts[i] >= mode_counts[i - 1],
            "Mode count should not decrease with higher sensitivity value"
        );
    }

    // Highest sensitivity value should find multiple modes
    // The complex distribution has overlapping modes, so exact count varies
    assert!(
        mode_counts[mode_counts.len() - 1] >= 2,
        "Highest sensitivity should find at least 2 modes"
    );

    // Lowest sensitivity value should find fewer modes than highest
    assert!(
        mode_counts[0] < mode_counts[mode_counts.len() - 1],
        "Lowest sensitivity should find fewer modes than highest sensitivity"
    );
}
