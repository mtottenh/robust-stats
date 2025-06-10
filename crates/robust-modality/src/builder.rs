use crate::detector::LowlandModalityDetector;
use crate::visualization::{ModalityVisualizer, NullModalityVisualizer};
use robust_histogram::{QRDEBuilderWithSteps, QuantileStepFunction};

/// Builder for configuring and creating modality detectors.
/// 
/// The estimator is now provided at detection time to enable batch optimization.
pub struct ModalityDetectorBuilder<V> {
    visualizer: V,
    sensitivity: f64,
    precision: f64,
    step_function: Option<QuantileStepFunction>,
}

impl<V> ModalityDetectorBuilder<V>
where
    V: ModalityVisualizer,
{
    /// Creates a new modality detector builder.
    ///
    /// # Arguments  
    /// * `visualizer` - The visualizer for optional plotting
    pub fn new(visualizer: V) -> Self {
        Self {
            visualizer,
            sensitivity: 0.5,       // Default sensitivity
            precision: 1.0 / 101.0, // Default precision (101 bins)
            step_function: None,
        }
    }

    /// Sets the sensitivity parameter (0.0-1.0).
    ///
    /// Higher sensitivity detects more modes but may include noise.
    /// Lower sensitivity only detects very prominent modes.
    pub fn sensitivity(mut self, sensitivity: f64) -> Self {
        self.sensitivity = sensitivity.clamp(0.0, 1.0);
        self
    }

    /// Sets the precision parameter (0.001-1.0).
    ///
    /// Higher precision requires deeper valleys between modes.
    /// Lower precision allows modes separated by shallow valleys.
    ///
    /// The number of histogram bins is calculated as: round(1 / precision)
    pub fn precision(mut self, precision: f64) -> Self {
        self.precision = precision.clamp(0.001, 1.0);
        self
    }

    /// Sets a custom step function for the QRDE.
    ///
    /// This allows fine control over the quantile points used for binning.
    /// If set, this overrides the precision-based bin calculation.
    pub fn with_step_function<F>(mut self, step_fn: F) -> Self
    where
        F: Fn(usize) -> Vec<f64> + 'static,
    {
        self.step_function = Some(Box::new(step_fn));
        self
    }

    /// Builds the configured modality detector using QRDE.
    pub fn build(self) -> LowlandModalityDetector<f64, QRDEBuilderWithSteps, V> {
        let qrde_builder = if let Some(step_fn) = self.step_function {
            // Use custom step function  
            QRDEBuilderWithSteps::new(step_fn)
        } else {
            // Calculate number of bins from precision
            let num_bins = (1.0 / self.precision).round() as usize;
            // Use uniform step function based on precision
            QRDEBuilderWithSteps::uniform(num_bins)
        };

        LowlandModalityDetector::new(
            qrde_builder,
            self.visualizer,
            self.sensitivity,
            self.precision,
        )
    }
}

/// Convenience function for creating a default modality detector.
/// Uses default parameters with no estimator (estimator provided at detection time).
pub fn default_detector() -> LowlandModalityDetector<f64, QRDEBuilderWithSteps, NullModalityVisualizer> {
    ModalityDetectorBuilder::new(NullModalityVisualizer::default())
        .sensitivity(0.5)
        .precision(1.0 / 101.0)
        .build()
}

/// Convenience function for creating a modality detector with sensible defaults.
/// Uses:
/// - Sensitivity: 0.5
/// - Precision: 1/101 (based on num_bins)
/// - Number of bins: 101
/// - Estimator provided at detection time
pub fn sensible_defaults() -> LowlandModalityDetector<f64, QRDEBuilderWithSteps, NullModalityVisualizer> {
    ModalityDetectorBuilder::new(NullModalityVisualizer::default())
        .sensitivity(0.5)
        .precision(1.0 / 101.0) // This will result in 101 bins
        .build()
}

/// Convenience function for creating a modality detector with custom parameters.
pub fn detector_with_params(
    sensitivity: f64,
    precision: f64,
) -> LowlandModalityDetector<f64, QRDEBuilderWithSteps, NullModalityVisualizer> {
    ModalityDetectorBuilder::new(NullModalityVisualizer::default())
        .sensitivity(sensitivity)
        .precision(precision)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_data::{TestDistributions, TestParameters};
    use robust_quantile::{estimators::harrell_davis, QuantileAdapter, HDWeightComputer};
    use robust_core::{execution::scalar_sequential, UnifiedWeightCache, CachePolicy};

    #[test]
    fn test_builder_pattern() {
        let (sensitivity, precision) = TestParameters::DEFAULT;
        let detector = ModalityDetectorBuilder::new(NullModalityVisualizer::default())
            .sensitivity(sensitivity)
            .precision(precision)
            .build();

        // Test with standardized data
        let data = TestDistributions::unimodal_normal();
        let engine = scalar_sequential();
        let hd = harrell_davis(engine);
        let estimator = QuantileAdapter::new(hd);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        let result = detector.detect_modes_with_estimator(&data, &estimator, &cache);
        assert!(result.is_ok());
    }

    #[test]
    fn test_default_detector() {
        let detector = default_detector();
        // Test with standardized data
        let data = TestDistributions::unimodal_normal();
        let engine = scalar_sequential();
        let hd = harrell_davis(engine);
        let estimator = QuantileAdapter::new(hd);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        let result = detector.detect_modes_with_estimator(&data, &estimator, &cache);
        assert!(result.is_ok());
    }

    #[test]
    fn test_detector_with_params() {
        let (sensitivity, precision) = TestParameters::SENSITIVE;
        let detector = detector_with_params(sensitivity, precision);
        // Test with standardized data
        let data = TestDistributions::bimodal_symmetric();
        let engine = scalar_sequential();
        let hd = harrell_davis(engine);
        let estimator = QuantileAdapter::new(hd);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        let result = detector.detect_modes_with_estimator(&data, &estimator, &cache);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parameter_clamping() {
        let detector = ModalityDetectorBuilder::new(NullModalityVisualizer::default())
            .sensitivity(-0.5) // Should be clamped to 0.0
            .precision(2.0) // Should be clamped to 1.0 (resulting in 1 bin, clamped to min 5)
            .build();

        // Use small standardized test data
        let data = TestDistributions::unimodal_normal().into_iter().take(100).collect::<Vec<_>>();
        let engine = scalar_sequential();
        let hd = harrell_davis(engine);
        let estimator = QuantileAdapter::new(hd);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        let result = detector.detect_modes_with_estimator(&data, &estimator, &cache);
        assert!(result.is_ok());
    }
}
