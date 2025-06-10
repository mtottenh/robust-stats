//! Flexible histogram construction and analysis for robust statistics
//!
//! This crate provides generic histogram functionality that can be used across
//! the robust statistics ecosystem. It offers multiple strategies for histogram
//! construction, from simple fixed-width bins to sophisticated quantile-based
//! approaches that are robust to outliers.
//!
//! # Key Features
//!
//! - **Multiple binning strategies**: Fixed-width, quantile-based, Scott's rule, etc.
//! - **Robust to outliers**: Quantile-based binning handles extreme values gracefully
//! - **Generic design**: Works with any &[f64] slice
//! - **Histogram operations**: Merge, normalize, compare histograms
//! - **Integration ready**: Designed to work with other robust-stats crates
//!
//! # Examples
//!
//! ## Basic Usage with Fixed-Width Bins
//!
//! ```rust
//! use robust_histogram::{FixedWidthBuilder, HistogramBuilder};
//!
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
//! let builder = FixedWidthBuilder::new(5); // 5 bins
//! let histogram = builder.build(&data).unwrap();
//!
//! println!("Histogram with {} bins", histogram.len());
//! for bin in histogram.bins() {
//!     println!("  [{:.1}, {:.1}): count={}, density={:.3}",
//!              bin.left, bin.right, bin.count, bin.density);
//! }
//! ```
//!
//! ## Quantile-Based Histogram (Robust to Outliers)
//!
//! ```rust
//! # #[cfg(feature = "quantile")]
//! # {
//! use robust_histogram::QuantileBuilder;
//! use robust_quantile::{estimators::harrell_davis, QuantileAdapter, HDWeightComputer};
//! use robust_core::{execution::scalar_sequential, UnifiedWeightCache, CachePolicy};
//!
//! let data = vec![1.0, 2.0, 3.0, 100.0]; // Note the outlier
//! let builder = QuantileBuilder::new(4);
//! 
//! // Create estimator and cache
//! let engine = scalar_sequential();
//! let hd = harrell_davis(engine);
//! let estimator = QuantileAdapter::new(hd);
//! let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
//! 
//! let histogram = builder.build(&data, &estimator, &cache).unwrap();
//!
//! // The quantile-based approach handles the outlier gracefully
//! println!("Robust histogram: {} bins", histogram.len());
//! # }
//! ```
//!
//! ## Quantile-Respectful Density Estimator (QRDE)
//!
//! ```rust
//! # #[cfg(feature = "quantile")]
//! # {
//! use robust_histogram::{qrde, QRDEBuilder, HistogramBuilder};
//! use robust_quantile::{estimators::harrell_davis, QuantileAdapter, HDWeightComputer};
//! use robust_core::{execution::scalar_sequential, UnifiedWeightCache, CachePolicy};
//!
//! let data = vec![1.0, 2.0, 2.5, 3.0, 3.2, 3.5, 100.0]; // Skewed data
//! 
//! // Create estimator and cache
//! let engine = scalar_sequential();
//! let hd = harrell_davis(engine);
//! let estimator = QuantileAdapter::new(hd);
//! let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
//!
//! // Using the builder
//! let builder: QRDEBuilder = QRDEBuilder::new(10);
//! let density = builder.build(&data, &estimator, &cache).unwrap();
//!
//! // Or using the convenience function
//! let density = qrde(&data, &estimator, &cache, 10).unwrap();
//!
//! // QRDEs adapt bin boundaries to data distribution
//! println!("QRDE with {} bins captures density structure", density.len());
//! # }
//! ```
//!
//! ## QRDE with Custom Step Functions
//!
//! ```rust
//! # #[cfg(feature = "quantile")]
//! # {
//! use robust_histogram::{qrde_with_steps, adaptive_qrde, QRDEBuilderWithSteps};
//! use robust_quantile::{estimators::harrell_davis, QuantileAdapter, HDWeightComputer};
//! use robust_core::{execution::scalar_sequential, UnifiedWeightCache, CachePolicy};
//!
//! let data = vec![1.0, 2.0, 2.5, 3.0, 3.2, 3.5, 100.0];
//! 
//! // Create estimator and cache
//! let engine = scalar_sequential();
//! let hd = harrell_davis(engine);
//! let estimator = QuantileAdapter::new(hd);
//! let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
//! 
//! // Custom step function for specific quantiles
//! let density = qrde_with_steps(
//!     &data,
//!     &estimator,
//!     &cache,
//!     |_size| vec![0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
//! ).unwrap();
//!
//! // Adaptive QRDE that adjusts to sample size
//! let adaptive = adaptive_qrde(&data, &estimator, &cache).unwrap();
//!
//! // Exponential steps for better tail resolution
//! let builder = QRDEBuilderWithSteps::exponential(20);
//! let tail_focused = builder.build(&data, &estimator, &cache).unwrap();
//! # }
//! ```
//!
//! ## Histogram Operations
//!
//! ```rust
//! use robust_histogram::{FixedWidthBuilder, HistogramBuilder, HistogramOps};
//!
//! let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let data2 = vec![3.0, 4.0, 5.0, 6.0, 7.0];
//!
//! let builder = FixedWidthBuilder::new(5);
//! let hist1 = builder.build(&data1).unwrap();
//! let hist2 = builder.build(&data2).unwrap();
//!
//! // Normalize to compare shapes
//! let norm1 = hist1.normalize();
//! let norm2 = hist2.normalize();
//!
//! // Calculate histogram distance
//! let distance = norm1.wasserstein_distance(&norm2);
//! println!("Wasserstein distance: {:.3}", distance);
//! ```

pub mod builders;
pub mod ops;
pub mod traits;
pub mod types;

// Re-export main types and traits
pub use builders::{FixedWidthBuilder, ScottsRule, FreedmanDiaconisRule};
pub use traits::HistogramBuilder;
pub use types::{Histogram, HistogramBin};

#[cfg(feature = "quantile")]
pub use builders::{QuantileBuilder, QRDEBuilderWithSteps};

#[cfg(feature = "quantile")]
pub use builders::quantile::QuantileStepFunction;

#[cfg(feature = "quantile")]
/// Type alias for Quantile-Respectful Density Estimator (QRDE)
/// 
/// This is a more descriptive name for histogram builders that use quantile-based
/// binning strategies rather than fixed-width bins. QRDEs adapt bin boundaries to
/// the data distribution, providing better density estimates for skewed or
/// multi-modal distributions.
pub type QRDEBuilder = QuantileBuilder;

pub use ops::HistogramOps;

// Convenience functions
/// Create a histogram with a fixed number of equal-width bins
pub fn fixed_histogram(data: &[f64], num_bins: usize) -> crate::Result<Histogram> {
    FixedWidthBuilder::new(num_bins).build(data)
}

/// Create a histogram using Scott's rule for bin width
pub fn scott_histogram(data: &[f64]) -> crate::Result<Histogram> {
    ScottsRule.build(data)
}

#[cfg(feature = "quantile")]
/// Create a quantile-based histogram (robust to outliers)
pub fn quantile_histogram<Q>(data: &[f64], estimator: &Q, cache: &Q::State, num_bins: usize) -> crate::Result<Histogram>
where
    Q: robust_core::BatchQuantileEstimator,
{
    QuantileBuilder::new(num_bins).build(data, estimator, cache)
}

#[cfg(feature = "quantile")]
/// Create a Quantile-Respectful Density Estimate (QRDE)
/// 
/// This is an alias for quantile_histogram that better describes what the
/// resulting structure represents: a density estimate that respects the
/// quantile structure of the data.
pub fn qrde<Q>(data: &[f64], estimator: &Q, cache: &Q::State, num_bins: usize) -> crate::Result<Histogram>
where
    Q: robust_core::BatchQuantileEstimator,
{
    QuantileBuilder::new(num_bins).build(data, estimator, cache)
}

#[cfg(feature = "quantile")]
/// Create a QRDE with a custom step function
/// 
/// The step function takes the sample size and returns quantile points.
/// This allows flexible control over bin boundaries.
pub fn qrde_with_steps<Q, F>(data: &[f64], estimator: &Q, cache: &Q::State, step_fn: F) -> crate::Result<Histogram>
where
    Q: robust_core::BatchQuantileEstimator,
    F: Fn(usize) -> Vec<f64> + 'static,
{
    QRDEBuilderWithSteps::new(Box::new(step_fn)).build(data, estimator, cache)
}

#[cfg(feature = "quantile")]
/// Create an adaptive QRDE that adjusts bins based on sample size
pub fn adaptive_qrde<Q>(data: &[f64], estimator: &Q, cache: &Q::State) -> crate::Result<Histogram>
where
    Q: robust_core::BatchQuantileEstimator,
{
    QRDEBuilderWithSteps::adaptive().build(data, estimator, cache)
}

pub use robust_core::Result;
