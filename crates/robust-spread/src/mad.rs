//! Median Absolute Deviation (MAD) estimators
//!
//! MAD is a special case of QAD with p=0.5

use crate::qad::Qad;
use crate::traits::{RobustScale, SpreadEstimator, SpreadEstimatorProperties};
use robust_core::{ComputePrimitives, Result, Numeric};
use robust_quantile::QuantileEstimator;

/// Median Absolute Deviation (MAD) estimator
/// 
/// MAD is QAD with p=0.5. This is a convenience wrapper.
#[derive(Debug, Clone)]
pub struct Mad<T: Numeric = f64, P: ComputePrimitives<T> = robust_core::primitives::ScalarBackend> {
    qad: Qad<T, P>,
}

impl<T: Numeric, P: ComputePrimitives<T>> Mad<T, P> {
    /// Create a new MAD estimator
    pub fn new(primitives: P) -> Self {
        Self {
            qad: Qad::mad(primitives),
        }
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> SpreadEstimatorProperties for Mad<T, P> {
    fn name(&self) -> &str {
        "MAD"
    }

    fn is_robust(&self) -> bool {
        true
    }

    fn breakdown_point(&self) -> f64 {
        0.5
    }
    
    fn gaussian_efficiency(&self) -> f64 {
        0.3675 // 36.75% efficiency for MAD
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> SpreadEstimator<T, Q> for Mad<T, P> {
    
    fn estimate(&self, data: &mut [T], quantile_est: &Q, cache: &Q::State) -> Result<T::Float> {
        self.qad.estimate(data, quantile_est, cache)
    }
    
    fn estimate_sorted(&self, sorted_data: &[T], quantile_est: &Q, cache: &Q::State) -> Result<T::Float> {
        self.qad.estimate_sorted(sorted_data, quantile_est, cache)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> RobustScale<T, Q> for Mad<T, P> {}

/// Standardized MAD estimator (scaled by consistency factor)
/// 
/// The consistency factor (1.4826) makes it comparable to standard deviation
/// for normally distributed data.
#[derive(Debug, Clone)]
pub struct StandardizedMad<T: Numeric = f64, P: ComputePrimitives<T> = robust_core::primitives::ScalarBackend> {
    qad: Qad<T, P>,
}

impl<T: Numeric, P: ComputePrimitives<T>> StandardizedMad<T, P> {
    /// Create with default consistency factor
    pub fn new(primitives: P) -> Self {
        Self {
            qad: Qad::standardized_mad(primitives),
        }
    }

    /// Create with custom consistency factor
    pub fn with_factor(primitives: P, consistency_factor: f64) -> Self {
        Self { 
            qad: Qad::with_constant(primitives, 0.5, consistency_factor).unwrap(),
        }
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> SpreadEstimatorProperties for StandardizedMad<T, P> {
    fn name(&self) -> &str {
        "Standardized MAD"
    }

    fn is_robust(&self) -> bool {
        true
    }

    fn breakdown_point(&self) -> f64 {
        0.5
    }
    
    fn gaussian_efficiency(&self) -> f64 {
        0.3675 // 36.75% efficiency for MAD
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> SpreadEstimator<T, Q> for StandardizedMad<T, P> {
    
    fn estimate(&self, data: &mut [T], quantile_est: &Q, cache: &Q::State) -> Result<T::Float> {
        self.qad.estimate(data, quantile_est, cache)
    }
    
    fn estimate_sorted(&self, sorted_data: &[T], quantile_est: &Q, cache: &Q::State) -> Result<T::Float> {
        self.qad.estimate_sorted(sorted_data, quantile_est, cache)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> RobustScale<T, Q> for StandardizedMad<T, P> {}

/// Compute MAD with cache support (for backward compatibility)
pub fn mad_with_cache<T: Numeric, Q: QuantileEstimator<T>>(
    sample: &mut [T], 
    estimator: &Q,
    cache: &Q::State
) -> Result<T::Float> {
    use robust_core::primitives::ScalarBackend;
    let mad = Mad::<T, ScalarBackend>::new(ScalarBackend::new());
    mad.estimate(sample, estimator, cache)
}

/// Compute MAD using sorted data with cache support
pub fn mad_sorted_with_cache<T: Numeric, Q: QuantileEstimator<T>>(
    sorted_sample: &[T], 
    estimator: &Q,
    cache: &Q::State
) -> Result<T::Float> {
    use robust_core::primitives::ScalarBackend;
    let mad = Mad::<T, ScalarBackend>::new(ScalarBackend::new());
    mad.estimate_sorted(sorted_sample, estimator, cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use robust_quantile::estimators::harrell_davis;

    #[test]
    fn test_mad_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let estimator = harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        
        let result = mad_with_cache(&mut data, &estimator, &cache).unwrap();
        // With Harrell-Davis estimator, the MAD is approximately 1.26
        // (smooth quantile estimation gives different results than simple quantiles)
        assert_relative_eq!(result, 1.26, epsilon = 0.1);
    }

    #[test]
    fn test_mad_struct() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let estimator = harrell_davis(robust_core::auto_engine());
        let mad_est = Mad::new(robust_core::primitives::ScalarBackend::new());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        
        let result = mad_est.estimate(&mut data, &estimator, &cache).unwrap();
        // With Harrell-Davis estimator, expecting ~1.26
        assert_relative_eq!(result, 1.26, epsilon = 0.1);
    }

    #[test]
    fn test_standardized_mad() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let estimator = harrell_davis(robust_core::auto_engine());
        let smad = StandardizedMad::new(robust_core::primitives::ScalarBackend::new());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        
        let result = smad.estimate(&mut data, &estimator, &cache).unwrap();
        // With Harrell-Davis: 1.26 * 1.4826 ≈ 1.868
        assert_relative_eq!(result, 1.868, epsilon = 0.1);
    }

    #[test]
    fn test_mad_with_outlier() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let estimator = harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        
        let result = mad_with_cache(&mut data, &estimator, &cache).unwrap();
        // MAD should be robust to the outlier
        // With Harrell-Davis estimator, the result is smoother and can be higher
        assert!(result < 6.0); // Much less than std dev would be
        assert!(result > 1.0); // But not too small
    }

    #[test]
    fn test_empty_sample() {
        let mut data: Vec<f64> = vec![];
        let estimator = harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        
        let result = mad_with_cache(&mut data, &estimator, &cache);
        assert!(result.is_err());
    }
}