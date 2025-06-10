//! Robust confidence interval estimation
//!
//! This crate provides various methods for constructing confidence intervals
//! for robust estimators, including:
//!
//! - **Maritz-Jarrett intervals**: For quantile estimators with moment calculation
//! - **Bootstrap intervals**: Percentile, BCa, and other bootstrap methods
//! - **Asymptotic intervals**: Based on theoretical distributions
//!
//! # Overview
//!
//! Confidence intervals provide a range of plausible values for a population
//! parameter. Unlike traditional methods that assume normality, the methods
//! in this crate are designed to work well with robust estimators and
//! non-normal distributions.
//!
//! # Examples
//!
//! ## Maritz-Jarrett Confidence Interval
//!
//! ```rust,ignore
//! # #[cfg(feature = "quantile")]
//! # {
//! use robust_quantile::{QuantileAdapter, HDWeightComputer};
//! use robust_confidence::{MaritzJarrettCI, ConfidenceInterval, ConfidenceIntervalEstimator};
//! use robust_core::{execution::scalar_sequential, UnifiedWeightCache, CachePolicy};
//!
//! let mut sample = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
//!
//! // Create CI estimator for median
//! let mj_ci = MaritzJarrettCI::new(
//!     0.5,  // median
//!     0.95  // 95% confidence level
//! );
//!
//! // Create the estimator and cache
//! let engine = scalar_sequential();
//! let hd = robust_quantile::estimators::harrell_davis(engine);
//! let estimator = QuantileAdapter::new(hd);
//! let cache = UnifiedWeightCache::new(HDWeightComputer, CachePolicy::NoCache);
//!
//! let ci = mj_ci.confidence_interval(&sample, &estimator, &cache).unwrap();
//! println!("95% CI for median: [{:.2}, {:.2}]", ci.lower, ci.upper);
//! # }
//! ```

pub mod api;
mod asymptotic;
mod bootstrap;
mod bootstrap_methods;
pub mod bootstrap_workspace;
mod estimator_factories;
mod maritz_jarrett;
mod maritz_jarrett_two_sample;
mod quantile_bootstrap;
mod traits;
mod types;


// Re-exports
pub use api::{
    bootstrap_confidence_intervals, quantile_shift_confidence_intervals,
    quantile_ratio_confidence_intervals, auto_bootstrap,
    DEFAULT_RESAMPLES, FAST_RESAMPLES, HIGH_PRECISION_RESAMPLES,
};
pub use asymptotic::{AsymptoticCI, MeanWithSE, MeanCache, StandardErrorEstimator};
pub use bootstrap::{Bootstrap, BootstrapMethod, BootstrapResult};
pub use robust_core::EstimatorFactory;
pub use bootstrap_methods::{
    BCaBootstrap, BasicBootstrap, PercentileBootstrap, StudentBootstrap,
};
pub use estimator_factories::{
    closure_factory, static_factory,
};
#[cfg(feature = "quantile")]
pub use robust_quantile::{
    harrell_davis_factory, HarrellDavisFactory,
    trimmed_hd_constant_factory, trimmed_hd_linear_factory, trimmed_hd_sqrt_factory,
    TrimmedHDConstantFactory, TrimmedHDLinearFactory, TrimmedHDSqrtFactory,
};
#[cfg(all(feature = "spread", feature = "quantile"))]
pub use robust_spread::{
    mad_factory, qad_factory, MADFactory, QADFactory,
};
pub use maritz_jarrett::MaritzJarrettCI;
pub use maritz_jarrett_two_sample::MaritzJarrettTwoSampleCI;
pub use quantile_bootstrap::{
    QuantileShiftBootstrap, QuantileRatioBootstrap, QuantileBootstrapResult,
    decile_bootstrap, quartile_bootstrap, percentile_bootstrap,
};
pub use traits::{
    ConfidenceIntervalEstimator, PairedConfidenceIntervalEstimator,
    TwoSampleConfidenceIntervalEstimator,
};
pub use types::{ConfidenceInterval, ConfidenceLevel};

// Convenience constructor
pub fn maritz_jarrett(probability: f64, confidence_level: f64) -> MaritzJarrettCI {
    MaritzJarrettCI::new(probability, confidence_level)
}
