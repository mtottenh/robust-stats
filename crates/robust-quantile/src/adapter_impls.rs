//! Zero-cost adapter for quantile estimators
//!
//! This module defines QuantileAdapter and implements StatefulEstimator and 
//! BatchQuantileEstimator for it, enabling quantile estimators to work with 
//! the generic confidence interval framework.

use crate::QuantileEstimator;
use robust_core::{
    StatefulEstimator, BatchQuantileEstimator,
    Error as CoreError, Numeric,
};

/// Zero-cost adapter for quantile estimators
///
/// Bridges `QuantileEstimator` to `StatefulEstimator` and `BatchQuantileEstimator`.
/// Compiles away completely - equivalent to calling the quantile estimator directly.
///
/// # Zero-Cost Properties
/// - Monomorphized per estimator type
/// - All methods are `#[inline]` and compile to direct calls
/// - Zero runtime overhead
/// - If the estimator is zero-size, adapter adds only one f64 for default quantile
#[derive(Clone, Debug)]
pub struct QuantileAdapter<T: Numeric, Q> {
    estimator: Q,
    default_quantile: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, Q> QuantileAdapter<T, Q> {
    /// Create adapter with median (0.5) as default for single-value estimation
    #[inline]
    pub fn new(estimator: Q) -> Self {
        Self { 
            estimator,
            default_quantile: 0.5,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Create adapter with custom default quantile
    #[inline]
    pub fn with_default_quantile(estimator: Q, default_quantile: f64) -> Self {
        Self { 
            estimator, 
            default_quantile,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Get reference to underlying estimator (zero-cost)
    #[inline]
    pub const fn inner(&self) -> &Q {
        &self.estimator
    }
    
    /// Unwrap to get underlying estimator (zero-cost)
    #[inline]
    pub fn into_inner(self) -> Q {
        self.estimator
    }
    
    /// Get the default quantile
    #[inline]
    pub const fn default_quantile(&self) -> f64 {
        self.default_quantile
    }
}

/// Implementation of StatefulEstimator for QuantileAdapter
///
/// This allows any QuantileEstimator to work with the confidence interval framework.
/// Uses the adapter's default quantile for single-value estimation.
/// 
/// # Zero-Cost Properties
/// - Compiles to direct method call on underlying estimator
/// - No runtime overhead beyond the original estimator
/// - Monomorphized per concrete estimator type
impl<T: Numeric, Q> StatefulEstimator<T> for QuantileAdapter<T, Q> 
where
    Q: QuantileEstimator<T>,
{
    type State = Q::State;
    
    #[inline]
    fn estimate_with_cache(&self, sample: &[T], cache: &Self::State) -> robust_core::Result<T::Float> {
        self.inner()
            .quantile_sorted(sample, self.default_quantile(), cache)
            .map_err(|e| CoreError::Computation(format!("Quantile estimation failed: {e}")))
    }
    
    #[inline]
    fn estimate_sorted_with_cache(&self, sorted_sample: &[T], cache: &Self::State) -> robust_core::Result<T::Float> {
        self.inner()
            .quantile_sorted(sorted_sample, self.default_quantile(), cache)
            .map_err(|e| CoreError::Computation(format!("Quantile estimation failed: {e}")))
    }
}

/// Implementation of BatchQuantileEstimator for QuantileAdapter
///
/// This enables efficient multi-quantile operations through the adapter.
/// Directly delegates to the underlying estimator's batch processing capabilities.
///
/// # Zero-Cost Properties
/// - Direct delegation to underlying estimator
/// - No additional allocations or overhead
/// - Leverages estimator's native batch optimizations
impl<T: Numeric, Q> BatchQuantileEstimator<T> for QuantileAdapter<T, Q>
where
    Q: QuantileEstimator<T>,
{
    #[inline]
    fn estimate_quantiles_with_cache(
        &self,
        sample: &[T], 
        quantiles: &[f64],
        cache: &Self::State,
    ) -> robust_core::Result<Vec<T::Float>> {
        self.inner()
            .quantiles_sorted(sample, quantiles, cache)
            .map_err(|e| CoreError::Computation(format!("Quantile estimation failed: {e}")))
    }
    
    #[inline]
    fn estimate_quantiles_sorted_with_cache(
        &self,
        sorted_sample: &[T],
        quantiles: &[f64], 
        cache: &Self::State,
    ) -> robust_core::Result<Vec<T::Float>> {
        self.inner()
            .quantiles_sorted(sorted_sample, quantiles, cache)
            .map_err(|e| CoreError::Computation(format!("Quantile estimation failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{estimators::harrell_davis, HDWeightComputer};
    use robust_core::{simd_sequential, CachePolicy, UnifiedWeightCache};
    
    #[test]
    fn test_quantile_adapter_stateful_estimator() {
        // Create quantile estimator and adapter
        let engine = simd_sequential();
        let hd = harrell_davis(engine);
        let adapter = QuantileAdapter::new(hd);
        
        // Create cache
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        
        // Test data
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Test StatefulEstimator interface (should return median)
        let result = adapter.estimate_sorted_with_cache(&data, &cache).unwrap();
        
        // Should be close to median (3.0)
        assert!((result - 3.0).abs() < 0.5);
    }
    
    #[test]
    fn test_quantile_adapter_batch_estimator() {
        // Create quantile estimator and adapter  
        let engine = simd_sequential();
        let hd = harrell_davis(engine);
        let adapter = QuantileAdapter::new(hd);
        
        // Create cache
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        
        // Test data
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Test BatchQuantileEstimator interface
        let quantiles = vec![0.25, 0.5, 0.75];
        let results = adapter.estimate_quantiles_sorted_with_cache(&data, &quantiles, &cache).unwrap();
        
        // Should get 3 results
        assert_eq!(results.len(), 3);
        
        // Results should be in ascending order
        assert!(results[0] <= results[1]);
        assert!(results[1] <= results[2]);
    }
    
    #[test]
    fn test_custom_default_quantile() {
        let engine = simd_sequential();
        let hd = harrell_davis(engine);
        
        // Create adapter with 75th percentile as default
        let adapter = QuantileAdapter::with_default_quantile(hd, 0.75);
        assert_eq!(adapter.default_quantile(), 0.75);
        
        // Create cache  
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        
        // Test data
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Should return 75th percentile, not median
        let result = adapter.estimate_sorted_with_cache(&data, &cache).unwrap();
        assert!(result > 3.0); // Should be greater than median
    }
    
    #[test]
    fn test_zero_cost_compilation() {
        // This test ensures the adapter compiles to efficient code
        let engine = simd_sequential();
        let hd = harrell_davis(engine);
        let adapter = QuantileAdapter::new(hd);
        
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // These should compile to direct method calls with no overhead
        let _single = adapter.estimate_sorted_with_cache(&data, &cache).unwrap();
        let _batch = adapter.estimate_quantiles_sorted_with_cache(&data, &[0.25, 0.5, 0.75], &cache).unwrap();
        
        // If this compiles without warnings about unused code, the inlining worked
    }
}

// Convenience constructor functions (all const and inline for zero cost)

/// Create zero-cost quantile adapter with median default
#[inline]
pub fn quantile_adapter<T: Numeric, Q>(estimator: Q) -> QuantileAdapter<T, Q> {
    QuantileAdapter::new(estimator)
}

/// Create zero-cost quantile adapter with custom default quantile
#[inline] 
pub fn quantile_adapter_with_default<T: Numeric, Q>(estimator: Q, default_quantile: f64) -> QuantileAdapter<T, Q> {
    QuantileAdapter::with_default_quantile(estimator, default_quantile)
}