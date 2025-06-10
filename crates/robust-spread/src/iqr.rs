//! Interquartile Range (IQR) estimator

use crate::kernels::IqrKernel;
use crate::traits::{RobustScale, SpreadEstimator, SpreadEstimatorProperties};
use robust_core::{ComputePrimitives, Result, Numeric};
use robust_quantile::QuantileEstimator;
use num_traits::NumCast;

/// Interquartile Range estimator
/// 
/// Requires a quantile estimator to be passed as a parameter for all operations.
#[derive(Debug, Clone)]
pub struct Iqr<T: Numeric, P: ComputePrimitives<T>> {
    kernel: IqrKernel<T, P>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> Iqr<T, P> {
    /// Create a new IQR estimator
    pub fn new(primitives: P) -> Self {
        Self {
            kernel: IqrKernel::new(primitives),
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Standardized IQR estimator (scaled by consistency factor)
/// 
/// The consistency factor (0.7413) makes it comparable to standard deviation
/// for normally distributed data.
#[derive(Debug, Clone)]
pub struct IqrEstimator<T: Numeric, P: ComputePrimitives<T>> {
    kernel: IqrKernel<T, P>,
    consistency_factor: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> IqrEstimator<T, P> {
    /// Create with default consistency factor
    pub fn new(primitives: P) -> Self {
        Self {
            kernel: IqrKernel::new(primitives),
            consistency_factor: 0.7413, // 1 / (2 * qnorm(0.75))
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create with custom consistency factor
    pub fn with_factor(primitives: P, consistency_factor: f64) -> Self {
        Self {
            kernel: IqrKernel::new(primitives),
            consistency_factor,
            _phantom: std::marker::PhantomData,
        }
    }
}

// Note: Default implementation removed - primitives must be explicitly provided

/// Generic IQR type for type-parameterized usage
/// 
/// This is provided for backward compatibility but doesn't store an estimator.
#[derive(Debug, Clone)]
pub struct GenericIqr<T: Numeric, Q> {
    _phantom: std::marker::PhantomData<(T, Q)>,
}

impl<T: Numeric, Q: QuantileEstimator<T>> GenericIqr<T, Q> {
    /// Create a new generic IQR estimator
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Numeric, Q: QuantileEstimator<T>> Default for GenericIqr<T, Q> {
    fn default() -> Self {
        Self::new()
    }
}

// Trait implementations for parameterized SpreadEstimator

impl<T: Numeric, P: ComputePrimitives<T>> SpreadEstimatorProperties for Iqr<T, P> {
    fn name(&self) -> &str {
        "IQR"
    }

    fn is_robust(&self) -> bool {
        true
    }

    fn breakdown_point(&self) -> f64 {
        0.5
    }
    
    fn gaussian_efficiency(&self) -> f64 {
        0.37 // Approximate efficiency for IQR
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> SpreadEstimator<T, Q> for Iqr<T, P> {
    fn estimate(&self, data: &mut [T], quantile_est: &Q, cache: &Q::State) -> Result<T::Float> {
        self.kernel.compute_iqr(data, quantile_est, cache)
    }
    
    fn estimate_sorted(&self, sorted_data: &[T], quantile_est: &Q, cache: &Q::State) -> Result<T::Float> {
        // For now, delegate to unsorted version since IQR doesn't really use the sorted property
        let mut data = sorted_data.to_vec();
        self.kernel.compute_iqr(&mut data, quantile_est, cache)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> RobustScale<T, Q> for Iqr<T, P> {}

impl<T: Numeric, P: ComputePrimitives<T>> SpreadEstimatorProperties for IqrEstimator<T, P> {
    fn name(&self) -> &str {
        "Standardized IQR"
    }

    fn is_robust(&self) -> bool {
        true
    }

    fn breakdown_point(&self) -> f64 {
        0.5
    }
    
    fn gaussian_efficiency(&self) -> f64 {
        0.37 // Approximate efficiency for IQR
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> SpreadEstimator<T, Q> for IqrEstimator<T, P> {
    fn estimate(&self, data: &mut [T], quantile_est: &Q, cache: &Q::State) -> Result<T::Float> {
        let iqr_value = self.kernel.compute_iqr(data, quantile_est, cache)?;
        let factor = <T::Float as NumCast>::from(self.consistency_factor).unwrap();
        Ok(iqr_value * factor)
    }
    
    fn estimate_sorted(&self, sorted_data: &[T], quantile_est: &Q, cache: &Q::State) -> Result<T::Float> {
        let mut data = sorted_data.to_vec();
        let iqr_value = self.kernel.compute_iqr(&mut data, quantile_est, cache)?;
        let factor = <T::Float as NumCast>::from(self.consistency_factor).unwrap();
        Ok(iqr_value * factor)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> RobustScale<T, Q> for IqrEstimator<T, P> {}

// Type aliases removed - use parameterized trait approach instead


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use robust_quantile::estimators::harrell_davis;
    use crate::traits::SpreadEstimator;

    #[test]
    fn test_iqr_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let estimator = harrell_davis(robust_core::auto_engine());
        let iqr_est = Iqr::<f64, _>::new(robust_core::primitives::ScalarBackend::new());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        
        let result = iqr_est.estimate(&mut data, &estimator, &cache).unwrap();
        // Approximate Q1 = 3.0, Q3 = 7.0, IQR = 4.0
        assert_relative_eq!(result, 4.0, epsilon = 1.0);
    }

    #[test]
    fn test_iqr_struct() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let estimator = harrell_davis(robust_core::auto_engine());
        let iqr_est = Iqr::<f64, _>::new(robust_core::primitives::ScalarBackend::new());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        
        let result = iqr_est.estimate(&mut data, &estimator, &cache).unwrap();
        assert_relative_eq!(result, 4.0, epsilon = 1.0);
    }

    #[test]
    fn test_standardized_iqr() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let estimator = harrell_davis(robust_core::auto_engine());
        let iqr_est = IqrEstimator::<f64, _>::new(robust_core::primitives::ScalarBackend::new());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        
        let result = iqr_est.estimate(&mut data, &estimator, &cache).unwrap();
        assert_relative_eq!(result, 4.0 * 0.7413, epsilon = 0.5);
    }

    #[test]
    fn test_insufficient_data() {
        let mut data = vec![1.0, 2.0];
        let estimator = harrell_davis(robust_core::auto_engine());
        let iqr_est = Iqr::<f64, _>::new(robust_core::primitives::ScalarBackend::new());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        
        let result = SpreadEstimator::estimate(&iqr_est, &mut data, &estimator, &cache);
        assert!(result.is_err());
    }

    #[test]
    fn test_sorted_iqr() {
        let mut data = vec![9.0, 2.0, 7.0, 4.0, 5.0, 6.0, 3.0, 8.0, 1.0];
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let estimator = harrell_davis(robust_core::auto_engine());
        let iqr_est = Iqr::<f64, _>::new(robust_core::primitives::ScalarBackend::new());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        
        let result = iqr_est.estimate_sorted(&data, &estimator, &cache).unwrap();
        assert_relative_eq!(result, 4.0, epsilon = 1.0);
    }
}