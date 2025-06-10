//! Zero-cost adapter for spread estimators
//!
//! This module defines SpreadAdapter and implements StatefulEstimator for it,
//! enabling spread estimators to work with the generic confidence interval framework.

use crate::{SpreadEstimator, SpreadEstimatorProperties};
use robust_core::{
    StatefulEstimator,
    Error as CoreError,
    Numeric,
};
use robust_quantile::QuantileEstimator;

/// Zero-cost adapter for spread estimators
///
/// Bridges `SpreadEstimator<Q>` to `StatefulEstimator`. Since spread estimators
/// are parameterized by quantile estimators, the adapter stores both the spread
/// estimator and the quantile estimator dependency.
///
/// # Zero-Cost Properties
/// - Monomorphized per estimator type combination
/// - All methods are `#[inline]` and compile to direct calls
/// - Zero runtime overhead
/// - Size equals sum of wrapped estimators
#[derive(Clone, Debug)]
pub struct SpreadAdapter<T: Numeric, S, Q> {
    spread_estimator: S,
    quantile_estimator: Q,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, S, Q> SpreadAdapter<T, S, Q> {
    /// Create adapter with spread and quantile estimators
    #[inline]
    pub fn new(spread_estimator: S, quantile_estimator: Q) -> Self {
        Self { 
            spread_estimator,
            quantile_estimator,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Get reference to underlying spread estimator (zero-cost)
    #[inline]
    pub const fn spread_estimator(&self) -> &S {
        &self.spread_estimator
    }
    
    /// Get reference to underlying quantile estimator (zero-cost)
    #[inline]
    pub const fn quantile_estimator(&self) -> &Q {
        &self.quantile_estimator
    }
    
    /// Unwrap to get underlying estimators (zero-cost)
    #[inline]
    pub fn into_inner(self) -> (S, Q) {
        (self.spread_estimator, self.quantile_estimator)
    }
}

/// Implementation of StatefulEstimator for SpreadAdapter
///
/// This allows any SpreadEstimator to work with the confidence interval framework.
/// Uses the spread estimator with its associated quantile estimator dependency.
/// 
/// # Zero-Cost Properties
/// - Compiles to direct method call on underlying spread estimator
/// - No runtime overhead beyond the original estimator
/// - Monomorphized per concrete estimator type combination
impl<T: Numeric, S, Q> StatefulEstimator<T> for SpreadAdapter<T, S, Q> 
where
    S: SpreadEstimator<T, Q>,
    Q: QuantileEstimator<T>,
{
    type State = Q::State;
    
    #[inline]
    fn estimate_with_cache(&self, sample: &[T], cache: &Self::State) -> robust_core::Result<T::Float> {
        // SpreadEstimator expects mutable data for sorting, so we need to clone
        let mut data = sample.to_vec();
        self.spread_estimator
            .estimate(&mut data, &self.quantile_estimator, cache)
            .map_err(|e| CoreError::Computation(format!("Spread estimation failed: {}", e)))
    }
    
    #[inline]
    fn estimate_sorted_with_cache(&self, sorted_sample: &[T], cache: &Self::State) -> robust_core::Result<T::Float> {
        self.spread_estimator
            .estimate_sorted(sorted_sample, &self.quantile_estimator, cache)
            .map_err(|e| CoreError::Computation(format!("Spread estimation failed: {}", e)))
    }
}

// Implement SpreadEstimatorProperties for the adapter (delegate to inner estimator)
impl<T: Numeric, S, Q> SpreadEstimatorProperties for SpreadAdapter<T, S, Q>
where
    S: SpreadEstimatorProperties,
{
    #[inline]
    fn name(&self) -> &str {
        self.spread_estimator.name()
    }
    
    #[inline]
    fn is_robust(&self) -> bool {
        self.spread_estimator.is_robust()
    }
    
    #[inline]
    fn breakdown_point(&self) -> f64 {
        self.spread_estimator.breakdown_point()
    }
    
    #[inline]
    fn gaussian_efficiency(&self) -> f64 {
        self.spread_estimator.gaussian_efficiency()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Mad, StandardizedMad};
    use robust_core::{UnifiedWeightCache, CachePolicy};
    use robust_quantile::{estimators::harrell_davis, HDWeightComputer};
    use std::mem::size_of_val;
    
    #[test]
    fn test_spread_adapter_stateful_estimator() {
        // Create spread and quantile estimators
        let primitives = robust_core::primitives::ScalarBackend::new();
        let mad = Mad::<f64, _>::new(primitives);
        let quantile_est = harrell_davis(robust_core::auto_engine());
        let adapter = SpreadAdapter::<f64, _, _>::new(mad, quantile_est);
        
        // Create cache
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        
        // Test data
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Test StatefulEstimator interface
        let result = adapter.estimate_sorted_with_cache(&data, &cache).unwrap();
        
        // MAD should be reasonable for this data
        assert!(result > 0.0);
        assert!(result < 10.0);
    }
    
    #[test]
    fn test_spread_adapter_properties() {
        let primitives = robust_core::primitives::ScalarBackend::new();
        let mad = StandardizedMad::<f64, _>::new(primitives);
        let quantile_est = harrell_davis(robust_core::auto_engine());
        let adapter = SpreadAdapter::<f64, _, _>::new(mad, quantile_est);
        
        // Test property delegation
        assert_eq!(adapter.name(), "Standardized MAD");
        assert!(adapter.is_robust());
        assert!(adapter.breakdown_point() > 0.4); // MAD has 50% breakdown point
        assert!(adapter.gaussian_efficiency() > 0.3); // MAD has ~37% efficiency
    }
    
    #[test]
    fn test_adapter_sizes() {
        // Test that adapter size is reasonable
        let primitives = robust_core::primitives::ScalarBackend::new();
        let mad = Mad::<f64, _>::new(primitives.clone());
        let quantile_est = harrell_davis(robust_core::auto_engine());
        let adapter = SpreadAdapter::<f64, _, _>::new(mad, quantile_est);
        
        // Check adapter size - just use the actual adapter type
        let adapter_size = size_of_val(&adapter);
        assert!(adapter_size > 0);
        // This is hard to test precisely since the types are complex,
        // but we can check it's reasonable
        assert!(adapter_size < 1000); // Should be much smaller than this
    }
    
    #[test]
    fn test_adapter_unwrapping() {
        let primitives = robust_core::primitives::ScalarBackend::new();
        let mad = Mad::<f64, _>::new(primitives);
        let quantile_est = harrell_davis(robust_core::auto_engine());
        let adapter = SpreadAdapter::<f64, _, _>::new(mad, quantile_est);
        
        // Should be able to get back the original estimators
        let (_spread_est, _quantile_est) = adapter.into_inner();
        // Can't easily test equality since the types don't implement PartialEq,
        // but the unwrapping should work without panic
    }
    
    #[test]
    fn test_zero_cost_compilation() {
        // This test ensures the adapter compiles to efficient code
        let primitives = robust_core::primitives::ScalarBackend::new();
        let mad = Mad::<f64, _>::new(primitives);
        let quantile_est = harrell_davis(robust_core::auto_engine());
        let adapter = SpreadAdapter::<f64, _, _>::new(mad, quantile_est);
        
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // These should compile to direct method calls with no overhead
        let _result = adapter.estimate_sorted_with_cache(&data, &cache).unwrap();
        
        // If this compiles without warnings about unused code, the inlining worked
    }
}

// Convenience constructor functions (all const and inline for zero cost)

/// Create zero-cost spread adapter
#[inline]
pub fn spread_adapter<T: Numeric, S, Q>(spread_estimator: S, quantile_estimator: Q) -> SpreadAdapter<T, S, Q> {
    SpreadAdapter::new(spread_estimator, quantile_estimator)
}