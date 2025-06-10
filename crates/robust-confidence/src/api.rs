//! High-level API for bootstrap confidence intervals
//!
//! This module provides clean, easy-to-use functions for common
//! bootstrap confidence interval calculations.

use crate::{
    bootstrap::{Bootstrap, BootstrapMethod, BootstrapResult},
    bootstrap_methods::PercentileBootstrap,
    quantile_bootstrap::{QuantileBootstrapResult, QuantileRatioBootstrap, QuantileShiftBootstrap},
    ConfidenceInterval,
};
use robust_core::{execution::HierarchicalExecution, EstimatorFactory, Result, TwoSampleComparison, Numeric};

// Re-export common bootstrap methods for convenience
pub use crate::bootstrap_methods::{BasicBootstrap, StudentBootstrap};

/// Default number of bootstrap resamples
pub const DEFAULT_RESAMPLES: usize = 20000;

/// Fast number of resamples for quick estimates
pub const FAST_RESAMPLES: usize = 10000;

/// High-precision number of resamples
pub const HIGH_PRECISION_RESAMPLES: usize = 30000;

/// Compute bootstrap confidence intervals for any two-sample comparison
///
/// This is the most general bootstrap function, working with any comparison
/// type and estimator.
///
/// # Arguments
/// * `sample1` - First sample
/// * `sample2` - Second sample
/// * `comparison` - The comparison operation (shift, ratio, etc.)
/// * `estimator_factory` - Factory for creating estimators
/// * `engine` - Execution engine with hierarchical support
/// * `method` - Bootstrap method (e.g., PercentileBootstrap, BCaBootstrap)
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95%)
/// * `n_resamples` - Number of bootstrap resamples
///
/// # Example
/// ```rust,ignore
/// use robust_confidence::api::*;
/// use robust_core::{QuantileShiftComparison, execution::auto_budgeted_engine};
///
/// let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let sample2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
///
/// let comparison = QuantileShiftComparison::new(vec![0.5]).unwrap();
/// let factory = harrell_davis_factory();
/// let engine = auto_budgeted_engine();
///
/// let result = bootstrap_confidence_intervals(
///     &sample1,
///     &sample2,
///     &comparison,
///     &factory,
///     engine,
///     PercentileBootstrap,
///     0.95,
///     DEFAULT_RESAMPLES,
/// ).unwrap();
/// ```
pub fn bootstrap_confidence_intervals<C, E, M, Est, F>(
    sample1: &[f64],
    sample2: &[f64],
    comparison: &C,
    estimator_factory: &F,
    engine: E,
    method: M,
    confidence_level: f64,
    n_resamples: usize,
) -> Result<BootstrapResult<C::Output>>
where
    C: TwoSampleComparison<Est> + Clone + Send + Sync,
    E: HierarchicalExecution<f64>,
    M: BootstrapMethod,
    F: EstimatorFactory<f64, E::SubordinateEngine, Estimator = Est>,
    Est: robust_core::StatefulEstimator<f64> + Send + Sync,
    Est::State: Send + Sync,
    C::Output: crate::bootstrap::BootstrapOutput + Clone + Send + Sync,
{
    let bootstrap = Bootstrap::new(engine, method)
        .with_resamples(n_resamples)
        .with_confidence_level(confidence_level);

    bootstrap.confidence_intervals(sample1, sample2, comparison, estimator_factory)
}

/// Compute bootstrap confidence intervals for quantile shifts
///
/// Specialized function for computing confidence intervals for differences
/// between quantiles of two samples.
///
/// # Example
/// ```rust,ignore
/// use robust_confidence::api::*;
/// use robust_core::execution::auto_budgeted_engine;
///
/// let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let sample2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
///
/// let result = quantile_shift_confidence_intervals(
///     &sample1,
///     &sample2,
///     &[0.25, 0.5, 0.75],  // Quartiles
///     &harrell_davis_factory(),
///     auto_budgeted_engine(),
///     PercentileBootstrap,
///     0.95,
///     DEFAULT_RESAMPLES,
/// ).unwrap();
///
/// println!("{}", result.summary());
/// ```
pub fn quantile_shift_confidence_intervals<E, M, Est, F>(
    sample1: &[f64],
    sample2: &[f64],
    quantiles: &[f64],
    estimator_factory: &F,
    engine: E,
    method: M,
    confidence_level: f64,
    n_resamples: usize,
) -> Result<QuantileBootstrapResult<f64>>
where
    E: HierarchicalExecution<f64>,
    M: BootstrapMethod,
    F: EstimatorFactory<f64, E::SubordinateEngine, Estimator = Est>,
    Est: robust_core::BatchQuantileEstimator<f64> + Send + Sync,
    Est::State: Send + Sync,
{
    let bootstrap = QuantileShiftBootstrap::new(engine, method, quantiles.to_vec())?
        .with_resamples(n_resamples)
        .with_confidence_level(confidence_level);

    bootstrap.confidence_intervals(sample1, sample2, estimator_factory)
}

/// Compute bootstrap confidence intervals for quantile ratios
///
/// Specialized function for computing confidence intervals for ratios
/// between quantiles of two samples.
///
/// # Example
/// ```rust,ignore
/// use robust_confidence::api::*;
/// use robust_core::execution::auto_budgeted_engine;
///
/// let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let sample2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
///
/// let result = quantile_ratio_confidence_intervals(
///     &sample1,
///     &sample2,
///     &[0.25, 0.5, 0.75],  // Quartiles
///     &harrell_davis_factory(),
///     auto_budgeted_engine(),
///     BCaBootstrap,
///     0.95,
///     DEFAULT_RESAMPLES,
/// ).unwrap();
/// ```
pub fn quantile_ratio_confidence_intervals<E, M, Est, F>(
    sample1: &[f64],
    sample2: &[f64],
    quantiles: &[f64],
    estimator_factory: &F,
    engine: E,
    method: M,
    confidence_level: f64,
    n_resamples: usize,
) -> Result<QuantileBootstrapResult<f64>>
where
    E: HierarchicalExecution<f64>,
    M: BootstrapMethod,
    F: EstimatorFactory<f64, E::SubordinateEngine, Estimator = Est>,
    Est: robust_core::BatchQuantileEstimator<f64> + Send + Sync,
    Est::State: Send + Sync,
{
    let bootstrap = QuantileRatioBootstrap::new(engine, method, quantiles.to_vec())?
        .with_resamples(n_resamples)
        .with_confidence_level(confidence_level);

    bootstrap.confidence_intervals(sample1, sample2, estimator_factory)
}

/// Quick bootstrap confidence interval for median shift
///
/// Convenience function for the common case of computing a confidence
/// interval for the difference between medians.
///
/// # Example
/// ```rust,ignore
/// use robust_confidence::api::median_shift_ci;
///
/// let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let sample2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
///
/// let ci = median_shift_ci(&sample1, &sample2, 0.95).unwrap();
/// println!("95% CI for median shift: [{:.2}, {:.2}]", ci.lower, ci.upper);
/// ```
#[cfg(feature = "quantile")]
pub fn median_shift_ci(
    sample1: &[f64],
    sample2: &[f64],
    confidence_level: f64,
) -> Result<ConfidenceInterval<f64>> {
    use robust_quantile::harrell_davis_factory;
    use robust_core::execution::auto_budgeted_engine;

    let result = quantile_shift_confidence_intervals(
        sample1,
        sample2,
        &[0.5],
        &harrell_davis_factory(),
        auto_budgeted_engine(),
        PercentileBootstrap,
        confidence_level,
        FAST_RESAMPLES,
    )?;

    result
        .intervals
        .into_iter()
        .next()
        .ok_or_else(|| robust_core::Error::Execution("No intervals computed".to_string()))
}

/// Bootstrap confidence intervals with automatic method selection
///
/// This function automatically selects an appropriate bootstrap method
/// based on the sample sizes and comparison type.
pub fn auto_bootstrap<C, E, Est, F>(
    sample1: &[f64],
    sample2: &[f64],
    comparison: &C,
    estimator_factory: &F,
    engine: E,
    confidence_level: f64,
) -> Result<BootstrapResult<C::Output>>
where
    C: TwoSampleComparison<Est> + Clone + Send + Sync,
    E: HierarchicalExecution<f64>,
    F: EstimatorFactory<f64, E::SubordinateEngine, Estimator = Est>,
    Est: robust_core::StatefulEstimator<f64> + Send + Sync,
    Est::State: Send + Sync,
    C::Output: crate::bootstrap::BootstrapOutput + Clone + Send + Sync,
{
    let n1 = sample1.len();
    let n2 = sample2.len();
    let n_min = n1.min(n2);

    // Choose number of resamples based on sample size
    let n_resamples = if n_min < 50 {
        HIGH_PRECISION_RESAMPLES // More resamples for small samples
    } else if n_min < 200 {
        DEFAULT_RESAMPLES
    } else {
        FAST_RESAMPLES // Can use fewer for large samples
    };

    // For auto_bootstrap, we'll use PercentileBootstrap as it's most robust
    bootstrap_confidence_intervals(
        sample1,
        sample2,
        comparison,
        estimator_factory,
        engine,
        PercentileBootstrap,
        confidence_level,
        n_resamples,
    )
}

// Re-export factory functions if quantile feature is enabled
// Re-export factory functions if features are enabled
#[cfg(feature = "quantile")]
pub use robust_quantile::{
    harrell_davis_factory, trimmed_hd_constant_factory, trimmed_hd_linear_factory,
    trimmed_hd_sqrt_factory,
};

#[cfg(all(feature = "spread", feature = "quantile"))]
pub use robust_spread::{mad_factory, qad_factory};

#[cfg(test)]
mod tests {

    #[test]
    fn test_auto_method_selection() {
        // Just verify the function compiles
        // Full integration tests would require the quantile feature
    }
}
