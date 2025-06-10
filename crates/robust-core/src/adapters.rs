//! Zero-cost adapters for bridging estimator traits
//!
//! This module provides zero-overhead adapters that enable different estimator
//! types to work with the generic confidence interval framework. All adapters
//! are designed to compile away completely through monomorphization and inlining.
//!
//! # Architecture
//!
//! - Domain-specific estimators keep their natural interfaces (QuantileEstimator, SpreadEstimator, etc.)
//! - Zero-cost adapters bridge these to the generic StatefulEstimator interface
//! - Confidence interval framework only works with StatefulEstimator
//! - Adapters compile away completely at runtime - zero overhead!
//!
//! # Examples
//!
//! ```rust,ignore
//! use robust_core::adapters::*;
//! // These would come from other crates
//! // use robust_quantile::QuantileAdapter;
//! // use robust_spread::SpreadAdapter;
//! 
//! // Create adapters (zero runtime cost)
//! // let quantile_adapter = QuantileAdapter::new(harrell_davis);
//! // let spread_adapter = SpreadAdapter::new(mad_estimator, harrell_davis);
//! // let mean_adapter = CentralTendencyAdapter::new(mean_estimator);
//!
//! // All work uniformly with CI framework
//! // let ci1 = BootstrapCI::new(quantile_adapter, ...);
//! // let ci2 = BootstrapCI::new(spread_adapter, ...);
//! // let ci3 = BootstrapCI::new(mean_adapter, ...);
//! ```

use crate::{Result, CentralTendencyEstimator, Numeric};
use crate::comparison::StatefulEstimator;



/// Zero-cost adapter for central tendency estimators
///
/// Bridges `CentralTendencyEstimator` to `StatefulEstimator`. Since central
/// tendency estimators traditionally don't use caches, this provides a dummy
/// cache that compiles away.
///
/// # Zero-Cost Properties
/// - Monomorphized per estimator type
/// - All methods are `#[inline]` and compile away  
/// - Zero runtime overhead
/// - Zero-size if the estimator is zero-size
#[derive(Clone, Debug)]
pub struct CentralTendencyAdapter<C, T: Numeric = f64> {
    estimator: C,
    _phantom: std::marker::PhantomData<T>,
}

impl<C, T: Numeric> CentralTendencyAdapter<C, T> {
    /// Create adapter
    #[inline]
    pub fn new(estimator: C) -> Self {
        Self { 
            estimator,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Get reference to underlying estimator (zero-cost)
    #[inline]
    pub const fn inner(&self) -> &C {
        &self.estimator
    }
    
    /// Unwrap to get underlying estimator (zero-cost)
    #[inline]
    pub fn into_inner(self) -> C {
        self.estimator
    }
}

/// Zero-size dummy cache for estimators that don't need caching
///
/// This compiles away completely - zero memory footprint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoCache;

// Trait implementations for CentralTendencyAdapter (since CentralTendencyEstimator is local)

impl<C, T> StatefulEstimator<T> for CentralTendencyAdapter<C, T>
where
    C: CentralTendencyEstimator<T>,
    T: Numeric,
{
    type State = NoCache; // Central tendency estimators use no cache
    
    #[inline]
    fn estimate_with_cache(&self, sample: &[T], _cache: &Self::State) -> Result<T::Float> {
        // CentralTendencyEstimator expects sorted data, so we need to sort
        let mut sorted_data = sample.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.estimator.estimate_sorted(&sorted_data)
    }
    
    #[inline]  
    fn estimate_sorted_with_cache(&self, sorted_sample: &[T], _cache: &Self::State) -> Result<T::Float> {
        // Direct call since data is already sorted
        self.estimator.estimate_sorted(sorted_sample)
    }
}

// Convenience constructor functions (all const and inline for zero cost)


/// Create zero-cost central tendency adapter
#[inline]
pub fn central_tendency_adapter<C, T: Numeric>(estimator: C) -> CentralTendencyAdapter<C, T> {
    CentralTendencyAdapter::new(estimator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;
    
    // Mock zero-size estimators for testing
    #[derive(Clone, Debug, Copy)]
    struct ZeroSizeEstimator;
    
    #[derive(Clone, Debug)]
    struct SmallEstimator {
        _value: f64,
    }
    
    #[test]
    fn test_adapter_sizes() {
        // Zero-size estimator adapter should be minimal (plus PhantomData)
        assert_eq!(
            size_of::<CentralTendencyAdapter<ZeroSizeEstimator, f64>>(),
            0 // PhantomData is zero-size
        );
        
        // Small estimator adapter should be minimal overhead (plus PhantomData)
        assert_eq!(
            size_of::<CentralTendencyAdapter<SmallEstimator, f64>>(),
            size_of::<SmallEstimator>() // PhantomData adds no size
        );
    }
    
    #[test]
    fn test_construction() {
        // CentralTendency adapter should be constructible
        let _ct_adapter: CentralTendencyAdapter<ZeroSizeEstimator, f64> = CentralTendencyAdapter::new(ZeroSizeEstimator);
    }
    
    #[test]
    fn test_no_cache_size() {
        // NoCache should be zero-size
        assert_eq!(size_of::<NoCache>(), 0);
        
        // Default should work
        let _cache: NoCache = Default::default();
    }
    
    #[test]
    fn test_adapter_unwrapping() {
        let estimator = ZeroSizeEstimator;
        let adapter: CentralTendencyAdapter<ZeroSizeEstimator, f64> = CentralTendencyAdapter::new(estimator);
        
        // Should be able to get back the original
        let unwrapped = adapter.into_inner();
        // For ZST, we just check they're the same type
        let _: ZeroSizeEstimator = unwrapped;
    }
}