//! Core traits for robust statistical estimation
//!
//! This module provides the foundational traits that all statistical
//! estimators build upon. Algorithm-specific traits (like QuantileEstimator,
//! SpreadEstimator, etc.) are defined in their respective crates.

use crate::{Result, comparison::StatefulEstimator, Numeric};
use std::any::Any;
use bitflags::bitflags;

/// Core trait for robust estimators
///
/// This trait defines the interface for estimators that compute a single
/// statistic from a data sample. It's designed to work with the parameterized
/// architecture where cache and other dependencies are passed externally.
pub trait RobustEstimator<T: Numeric = f64> {
    /// Output type of the estimator
    type Output;
    
    /// Estimate the statistic from the given sample
    fn estimate(&self, sample: &[T]) -> Result<Self::Output>;
    
    /// Estimate from pre-sorted data (optimization opportunity)
    fn estimate_sorted(&self, sorted_sample: &[T]) -> Result<Self::Output> {
        self.estimate(sorted_sample)
    }
}

/// Trait for two-sample estimators
///
/// This trait enables estimators that compare properties between two samples.
/// The specific comparison operation is defined by the implementation.
pub trait TwoSampleEstimator<T: Numeric = f64> {
    /// Output type of the comparison
    type Output;
    
    /// Estimate the comparison between two samples
    fn estimate_two_sample(&self, sample1: &[T], sample2: &[T]) -> Result<Self::Output>;
}

/// Base trait for all central tendency estimators
///
/// This is the fundamental trait for estimators that measure the "center" 
/// or "typical value" of a distribution. Implementations include means, 
/// medians, trimmed means, and other robust measures.
///
/// # API Convention
///
/// - Methods without suffix sort data in place (take `&mut [T]`)
/// - Methods with `_sorted` suffix require pre-sorted data (take `&[T]`)
///
/// # Example
///
/// ```rust
/// use robust_core::{CentralTendencyEstimator, Result, Numeric};
/// use num_traits::{Zero, Float, NumCast};
///
/// #[derive(Clone)]
/// struct Mean<T: Numeric> {
///     _phantom: std::marker::PhantomData<T>,
/// }
///
/// impl<T: Numeric> CentralTendencyEstimator<T> for Mean<T> {
///     fn estimate(&self, data: &mut [T]) -> Result<T::Float> {
///         // No need to sort for mean, but API requires mutable reference
///         self.estimate_sorted(data)
///     }
///     
///     fn estimate_sorted(&self, sorted_data: &[T]) -> Result<T::Float> {
///         if sorted_data.is_empty() {
///             return Err(robust_core::Error::empty_input("mean"));
///         }
///         let sum: T::Float = sorted_data.iter()
///             .map(|x| x.to_float())
///             .fold(T::Float::zero(), |acc, x| acc + x);
///         let count = <T::Float as NumCast>::from(sorted_data.len()).unwrap();
///         Ok(sum / count)
///     }
///     
///     fn name(&self) -> &str { "Mean" }
///     fn is_robust(&self) -> bool { false }
///     fn breakdown_point(&self) -> f64 { 0.0 }
/// }
/// ```
pub trait CentralTendencyEstimator<T: Numeric = f64>: Send + Sync + Clone {
    /// Estimate the central tendency from a sample
    ///
    /// # Warning
    /// This method may sort the data in place! If you need to preserve the original
    /// order, use `estimate_sorted()` with pre-sorted data or make a copy first.
    fn estimate(&self, data: &mut [T]) -> Result<T::Float> {
        // Default implementation: sort and delegate to estimate_sorted
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.estimate_sorted(data)
    }
    
    /// Estimate the central tendency from pre-sorted data
    fn estimate_sorted(&self, sorted_data: &[T]) -> Result<T::Float>;
    
    /// Human-readable name of the estimator
    fn name(&self) -> &str;
    
    /// Whether this estimator is robust to outliers
    fn is_robust(&self) -> bool;
    
    /// The breakdown point (fraction of contamination the estimator can handle)
    /// - 0.0 for non-robust estimators (e.g., mean)
    /// - 0.5 for maximally robust estimators (e.g., median)
    fn breakdown_point(&self) -> f64;
}

/// Performance characteristics for batch processing
///
/// Communicates an estimator's optimal usage patterns to enable
/// automatic performance optimization.
#[derive(Debug, Clone)]
pub struct BatchCharacteristics {
    /// Optimal number of datasets to process together
    pub optimal_batch_size: Option<usize>,
    
    /// Whether this algorithm benefits from batching
    pub benefits_from_batching: bool,
    
    /// Preferred processing dimension
    pub preferred_dimension: Option<BatchDimension>,
    
    /// Whether parallel processing is beneficial
    pub supports_parallel: bool,
}

impl BatchCharacteristics {
    /// Create characteristics for sequential processing
    pub fn sequential() -> Self {
        Self {
            optimal_batch_size: None,
            benefits_from_batching: false,
            preferred_dimension: None,
            supports_parallel: false,
        }
    }
    
    /// Create characteristics for batch processing
    pub fn batch_optimized(optimal_size: usize) -> Self {
        Self {
            optimal_batch_size: Some(optimal_size),
            benefits_from_batching: true,
            preferred_dimension: Some(BatchDimension::Balanced),
            supports_parallel: true,
        }
    }
}

/// Dimension preference for batch processing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BatchDimension {
    /// Process many datasets with few parameters each
    Datasets,
    /// Process few datasets with many parameters each
    Parameters,
    /// No strong preference
    Balanced,
}

/// Parallelization hints for execution planning
#[derive(Debug, Clone, Copy)]
pub enum ParallelizationHint {
    /// Can be used from multiple threads without synchronization
    ThreadSafe,
    /// Should be cloned for each thread
    ThreadLocal,
    /// Must be accessed sequentially
    Sequential,
}

/// Trait for accessing internal kernels
///
/// Allows advanced users to access the underlying kernel for
/// custom operations or performance optimization.
pub trait HasKernel {
    /// The type of kernel used by this estimator
    type Kernel;
    
    /// Get a reference to the kernel
    fn kernel(&self) -> &Self::Kernel;
}

/// Trait for estimators that can compute their variance or second moment
///
/// This is required for methods like Maritz-Jarrett confidence intervals
/// that need variance estimates for asymptotic intervals.
pub trait EstimatorWithVariance<T: Numeric = f64>: StatefulEstimator {
    /// Estimate the value and its variance (second moment) in one pass
    ///
    /// Returns (estimate, variance)
    fn estimate_with_variance(
        &self,
        sample: &[T],
        cache: &<Self as StatefulEstimator>::State,
    ) -> Result<(T::Float, T::Float)>;
    
    /// Estimate from sorted data with variance
    fn estimate_sorted_with_variance(
        &self,
        sorted_sample: &[T],
        cache: &<Self as StatefulEstimator>::State,
    ) -> Result<(T::Float, T::Float)> {
        self.estimate_with_variance(sorted_sample, cache)
    }
}

/// Trait for runtime performance introspection
///
/// Enables estimators to report their active optimizations
/// and performance characteristics at runtime.
pub trait PerformanceIntrospection {
    /// Get the active optimization flags
    fn active_optimizations(&self) -> OptimizationFlags;
    
    /// Get cache statistics if caching is enabled
    fn cache_stats(&self) -> Option<crate::batch::CacheStats> {
        None
    }
    
    /// Check if SIMD is being used
    fn is_simd_enabled(&self) -> bool {
        self.active_optimizations().contains(OptimizationFlags::SIMD)
    }
    
    /// Check if parallel execution is enabled
    fn is_parallel_enabled(&self) -> bool {
        self.active_optimizations().contains(OptimizationFlags::PARALLEL)
    }
}

bitflags! {
    /// Flags indicating which optimizations are active
    pub struct OptimizationFlags: u32 {
        /// SIMD instructions are being used
        const SIMD = 0b00000001;
        /// Results are being cached
        const CACHE = 0b00000010;
        /// Parallel execution is enabled
        const PARALLEL = 0b00000100;
        /// Sparse representations are used
        const SPARSE = 0b00001000;
        /// Memory prefetching is enabled
        const PREFETCH = 0b00010000;
    }
}

/// Helper trait for types that can be downcast
///
/// Used internally for kernel access patterns.
pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
}

impl<T: 'static> AsAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}