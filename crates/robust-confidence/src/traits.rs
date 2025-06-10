//! Core traits for confidence interval estimation
//!
//! This module defines the parameterized traits for confidence interval estimation
//! following the three-layer architecture. Estimators, comparison operations, and
//! caches are passed as parameters rather than stored, enabling optimal reuse
//! and composition.

use crate::types::ConfidenceInterval;
use robust_core::{Result, TwoSampleComparison, StatefulEstimator};

/// Parameterized confidence interval estimator trait
///
/// This trait follows the three-layer architecture where estimators and
/// cache are passed as parameters. This enables:
/// - Cache reuse across multiple CI calculations
/// - Composition with different estimators
/// - Zero-cost abstractions through dependency injection
pub trait ConfidenceIntervalEstimator<E> 
where
    E: StatefulEstimator,
{
    /// Calculate confidence interval for the given sample
    ///
    /// # Arguments
    /// * `sample` - Data sample
    /// * `estimator` - The estimator to use for the underlying property
    /// * `cache` - Cached computational state for optimal performance
    fn confidence_interval(
        &self,
        sample: &[f64],
        estimator: &E,
        cache: &E::State,
    ) -> Result<ConfidenceInterval>;
    
    /// Calculate confidence interval from pre-sorted data (optimization)
    fn confidence_interval_sorted(
        &self,
        sorted_sample: &[f64],
        estimator: &E,
        cache: &E::State,
    ) -> Result<ConfidenceInterval> {
        // Default implementation uses sorted data directly
        self.confidence_interval(sorted_sample, estimator, cache)
    }

    /// Get the confidence level
    fn confidence_level(&self) -> f64;

    /// Check if the method supports weighted samples
    fn supports_weighted_samples(&self) -> bool {
        false
    }
}


/// Parameterized two-sample confidence interval estimator trait
///
/// This trait uses the comparison abstraction to enable confidence intervals
/// for any type of two-sample comparison (shift, ratio, etc.) with any
/// estimator (quantiles, means, spreads, etc.).
pub trait TwoSampleConfidenceIntervalEstimator<C, E> 
where
    C: TwoSampleComparison<E>,
    E: StatefulEstimator,
{
    /// Calculate confidence interval for the comparison between two samples
    ///
    /// # Arguments
    /// * `sample1` - First sample
    /// * `sample2` - Second sample  
    /// * `comparison` - The comparison operation (shift, ratio, etc.)
    /// * `estimator` - The estimator for the underlying property
    /// * `cache` - Cached computational state for optimal performance
    fn confidence_interval_two_sample(
        &self,
        sample1: &[f64],
        sample2: &[f64],
        comparison: &C,
        estimator: &E,
        cache: &E::State,
    ) -> Result<ConfidenceInterval>;

    /// Get the confidence level
    fn confidence_level(&self) -> f64;
}

/// Parameterized paired confidence interval estimator trait
///
/// This trait enables confidence intervals for paired sample comparisons
/// using the same parameterized design as the other traits.
pub trait PairedConfidenceIntervalEstimator<E> 
where
    E: StatefulEstimator,
{
    /// Calculate confidence interval for paired samples
    ///
    /// # Arguments
    /// * `paired_data` - Paired sample data as (x, y) tuples
    /// * `estimator` - The estimator for the underlying property of differences
    /// * `cache` - Cached computational state for optimal performance
    fn confidence_interval_paired(
        &self,
        paired_data: &[(f64, f64)],
        estimator: &E,
        cache: &E::State,
    ) -> Result<ConfidenceInterval>;

    /// Get the confidence level
    fn confidence_level(&self) -> f64;
}
