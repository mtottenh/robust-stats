//! Confidence interval support for quantile estimation
//!
//! This module provides implementations of confidence intervals for
//! quantile estimators, particularly the Maritz-Jarrett method for
//! Harrell-Davis quantiles.

mod maritz_jarrett;

pub use maritz_jarrett::{ConfidenceInterval, MaritzJarrett};

/// Trait for quantile estimators that can compute moments
///
/// This is required for Maritz-Jarrett confidence intervals
pub trait QuantileWithMoments {
    /// Compute quantile and its first two moments
    ///
    /// Returns (quantile, second_moment)
    fn quantile_with_moments(&self, sorted_data: &[f64], p: f64) -> crate::Result<(f64, f64)>;
}
