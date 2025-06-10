//! Core traits for changepoint detection
//!
//! Following the three-layer architecture:
//! - Layer 3: High-level detection algorithms (this module)
//! - Layer 2: Domain-specific kernels (in respective detector modules)
//! - Layer 1: Computational primitives (from robust-core)
//!
//! All estimators and caches are passed as parameters to enable:
//! - Zero-cost rolling operations
//! - Perfect cache sharing across windows
//! - Composable algorithms
//! - Efficient batch processing

use crate::types::ChangePointResult;
use robust_core::{Result, Numeric};
use robust_spread::SpreadEstimator;
use robust_quantile::QuantileEstimator;

/// Properties of a changepoint detector that don't depend on estimators
pub trait ChangePointDetectorProperties {
    /// Get the name of the detection algorithm
    fn algorithm_name(&self) -> &'static str;
    
    /// Get the minimum sample size required for detection
    fn minimum_sample_size(&self) -> usize;
}

/// Core trait for changepoint detection with explicit dependencies
///
/// All detectors must specify their estimator requirements explicitly.
/// This enables cache sharing and efficient rolling operations.
pub trait ChangePointDetector<T, S, Q>: ChangePointDetectorProperties 
where
    T: Numeric,
    S: SpreadEstimator<T, Q>,
    Q: QuantileEstimator<T>,
{
    /// Detect changepoints in the given sample
    ///
    /// This method should:
    /// 1. Make a single copy of data if needed
    /// 2. Sort once if sorting is required
    /// 3. Use sorted APIs for all subsequent operations
    fn detect(
        &self, 
        sample: &[T], 
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<ChangePointResult>;
    
    /// Detect changepoints in pre-sorted data
    ///
    /// Implementations should use this to avoid redundant sorting
    fn detect_sorted(
        &self,
        sorted_sample: &[T],
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<ChangePointResult> {
        // Default: delegate to detect for algorithms that don't benefit from sorted data
        self.detect(sorted_sample, spread_est, quantile_est, cache)
    }
}

/// Trait for detectors that only need quantile estimation (no spread)
pub trait QuantileBasedDetector<T, Q>: ChangePointDetectorProperties 
where
    T: Numeric,
    Q: QuantileEstimator<T>,
{
    /// Detect using only quantile estimation
    fn detect_quantile(
        &self,
        sample: &[T],
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<ChangePointResult>;
    
    /// Detect in pre-sorted data
    fn detect_quantile_sorted(
        &self,
        sorted_sample: &[T],
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<ChangePointResult> {
        self.detect_quantile(sorted_sample, quantile_est, cache)
    }
}

/// Trait for simple detectors that don't need external estimators
pub trait SimpleDetector<T: Numeric>: ChangePointDetectorProperties {
    /// Detect without external dependencies
    fn detect_simple(&self, sample: &[T]) -> Result<ChangePointResult>;
    
    /// Detect in pre-sorted data
    fn detect_simple_sorted(&self, sorted_sample: &[T]) -> Result<ChangePointResult> {
        self.detect_simple(sorted_sample)
    }
}

/// Configuration trait remains unchanged (orthogonal to estimator parameterization)
pub trait ConfigurableDetector {
    type Parameters;
    
    fn with_parameters(params: Self::Parameters) -> Self;
    fn parameters(&self) -> &Self::Parameters;
    fn set_parameters(&mut self, params: Self::Parameters);
}

/// Confidence scoring trait remains unchanged
pub trait ConfidenceScoring {
    fn confidence_score(&self, changepoint_index: usize) -> f64;
    fn detection_threshold(&self) -> f64;
}

/// Online detection with explicit estimator dependencies
pub trait OnlineDetector<T, S, Q>
where
    T: Numeric,
    S: SpreadEstimator<T, Q>,
    Q: QuantileEstimator<T>,
{
    /// Process a single point with provided estimators
    fn process_point(
        &mut self,
        value: T,
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<Vec<usize>>;
    
    /// Reset internal state
    fn reset(&mut self);
    
    /// Get current position in stream
    fn current_index(&self) -> usize;
}

/// Batch processing for efficient multi-dataset analysis
pub trait BatchChangePointDetector<T, S, Q>: ChangePointDetector<T, S, Q>
where
    T: Numeric,
    S: SpreadEstimator<T, Q>,
    Q: QuantileEstimator<T>,
{
    /// Process multiple datasets with the same estimators for cache efficiency
    ///
    /// Default implementation processes sequentially
    fn detect_batch(
        &self,
        samples: &[&[T]],
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<Vec<ChangePointResult>> {
        samples.iter()
            .map(|sample| self.detect(sample, spread_est, quantile_est, cache))
            .collect()
    }
    
    /// Process pre-sorted datasets
    fn detect_batch_sorted(
        &self,
        sorted_samples: &[&[T]],
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<Vec<ChangePointResult>> {
        sorted_samples.iter()
            .map(|sample| self.detect_sorted(sample, spread_est, quantile_est, cache))
            .collect()
    }
}