//! Trimmed and Winsorized standard deviation estimators

use crate::traits::{RobustScale, SpreadEstimator, SpreadEstimatorProperties};
use crate::kernels::{TrimmedKernel, WinsorizedKernel};
use robust_core::{ComputePrimitives, Result, Numeric};
use robust_quantile::QuantileEstimator;

/// Trimmed standard deviation estimator
#[derive(Debug, Clone)]
pub struct TrimmedStd<T: Numeric, P: ComputePrimitives<T>> {
    kernel: TrimmedKernel<T, P>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> TrimmedStd<T, P> {
    /// Create with specified trim proportion
    pub fn new(primitives: P, trim_proportion: f64) -> Result<Self> {
        if !(0.0..0.5).contains(&trim_proportion) {
            return Err(robust_core::Error::InvalidInput(
                "Trim proportion must be between 0 and 0.5".to_string(),
            ));
        }
        Ok(Self {
            kernel: TrimmedKernel::new(primitives, trim_proportion),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create with 20% trimming (10% from each tail)
    pub fn default_trim(primitives: P) -> Self {
        Self {
            kernel: TrimmedKernel::new(primitives, 0.1),
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Get the trim proportion
    pub fn trim_proportion(&self) -> f64 {
        self.kernel.trim_proportion
    }
}



impl<T: Numeric, P: ComputePrimitives<T>> SpreadEstimatorProperties for TrimmedStd<T, P> {
    fn name(&self) -> &str {
        "Trimmed Std Dev"
    }

    fn is_robust(&self) -> bool {
        self.kernel.trim_proportion > 0.0
    }

    fn breakdown_point(&self) -> f64 {
        self.kernel.trim_proportion
    }
    
    fn gaussian_efficiency(&self) -> f64 {
        // Efficiency decreases as trimming increases
        // Approximate formula for normal distributions
        1.0 - 2.0 * self.kernel.trim_proportion
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> SpreadEstimator<T, Q> for TrimmedStd<T, P> {
    fn estimate(&self, data: &mut [T], _quantile_est: &Q, _cache: &Q::State) -> Result<T::Float> {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.estimate_sorted(data, _quantile_est, _cache)
    }
    
    fn estimate_sorted(&self, sorted_sample: &[T], _quantile_est: &Q, _cache: &Q::State) -> Result<T::Float> {
        self.kernel.compute_trimmed_std(sorted_sample)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> RobustScale<T, Q> for TrimmedStd<T, P> {}

/// Winsorized standard deviation estimator
#[derive(Debug, Clone)]
pub struct WinsorizedStd<T: Numeric, P: ComputePrimitives<T>> {
    kernel: WinsorizedKernel<T, P>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> WinsorizedStd<T, P> {
    /// Create with specified winsorization proportion
    pub fn new(primitives: P, winsor_proportion: f64) -> Result<Self> {
        if !(0.0..0.5).contains(&winsor_proportion) {
            return Err(robust_core::Error::InvalidInput(
                "Winsorization proportion must be between 0 and 0.5".to_string(),
            ));
        }
        Ok(Self {
            kernel: WinsorizedKernel::new(primitives, winsor_proportion),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create with 20% winsorization (10% from each tail)
    pub fn default_winsor(primitives: P) -> Self {
        Self {
            kernel: WinsorizedKernel::new(primitives, 0.1),
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Get the winsor proportion
    pub fn winsor_proportion(&self) -> f64 {
        self.kernel.winsor_proportion
    }
}


impl<T: Numeric, P: ComputePrimitives<T>> SpreadEstimatorProperties for WinsorizedStd<T, P> {
    fn name(&self) -> &str {
        "Winsorized Std Dev"
    }

    fn is_robust(&self) -> bool {
        self.kernel.winsor_proportion > 0.0
    }

    fn breakdown_point(&self) -> f64 {
        self.kernel.winsor_proportion
    }
    
    fn gaussian_efficiency(&self) -> f64 {
        // Winsorization is more efficient than trimming
        // Approximate formula
        1.0 - self.kernel.winsor_proportion
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> SpreadEstimator<T, Q> for WinsorizedStd<T, P> {
    fn estimate(&self, data: &mut [T], _quantile_est: &Q, _cache: &Q::State) -> Result<T::Float> {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.estimate_sorted(data, _quantile_est, _cache)
    }
    
    fn estimate_sorted(&self, sorted_sample: &[T], _quantile_est: &Q, _cache: &Q::State) -> Result<T::Float> {
        self.kernel.compute_winsorized_std(sorted_sample)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> RobustScale<T, Q> for WinsorizedStd<T, P> {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use robust_quantile::estimators::harrell_davis;
    use crate::traits::{SpreadEstimator, SpreadEstimatorProperties};

    #[test]
    fn test_trimmed_std() {
        let mut sample = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let estimator = harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        let trimmed = TrimmedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.1).unwrap();
        let result = trimmed.estimate(&mut sample, &estimator, &cache).unwrap();

        // Should trim 1 from each end, giving std of [2,3,4,5,6,7,8,9]
        assert!(result > 2.0 && result < 3.0);
    }

    #[test]
    fn test_trimmed_with_outliers() {
        let mut sample = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let estimator = harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        let trimmed = TrimmedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.2).unwrap();
        let result = trimmed.estimate(&mut sample, &estimator, &cache).unwrap();

        // Should trim the outlier
        assert!(result < 10.0);
    }

    #[test]
    fn test_winsorized_std() {
        let mut sample = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let estimator = harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        let winsorized = WinsorizedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.1).unwrap();
        let result = SpreadEstimator::estimate(&winsorized, &mut sample, &estimator, &cache).unwrap();

        // Should winsorize 1 from each end
        assert!(result > 2.0 && result < 3.5);
    }

    #[test]
    fn test_winsorized_with_outliers() {
        let mut sample = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let estimator = harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        let winsorized = WinsorizedStd::new(robust_core::primitives::ScalarBackend, 0.2).unwrap();
        let result = SpreadEstimator::estimate(&winsorized, &mut sample, &estimator, &cache).unwrap();

        // Should reduce impact of outlier
        assert!(result < 40.0);
    }

    #[test]
    fn test_invalid_proportions() {
        assert!(TrimmedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), -0.1).is_err());
        assert!(TrimmedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.5).is_err());
        assert!(TrimmedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.6).is_err());

        assert!(WinsorizedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), -0.1).is_err());
        assert!(WinsorizedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.5).is_err());
        assert!(WinsorizedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.6).is_err());
    }

    #[test]
    fn test_breakdown_points() {
        use robust_quantile::estimators::HarrellDavis;
        use robust_core::execution::SequentialEngine;
        use robust_core::primitives::ScalarBackend;
        type HDEstimator = HarrellDavis<f64, SequentialEngine<f64, ScalarBackend>>;
        
        let trimmed = TrimmedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.2).unwrap();
        assert_eq!(SpreadEstimatorProperties::breakdown_point(&trimmed), 0.2);

        let winsorized = WinsorizedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.15).unwrap();
        assert_eq!(SpreadEstimatorProperties::breakdown_point(&winsorized), 0.15);
    }

    #[test]
    fn test_efficiency() {
        use robust_quantile::estimators::HarrellDavis;
        use robust_core::execution::SequentialEngine;
        use robust_core::primitives::ScalarBackend;
        type HDEstimator = HarrellDavis<f64, SequentialEngine<f64, ScalarBackend>>;
        
        let trimmed = TrimmedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.1).unwrap();
        assert_relative_eq!(
            SpreadEstimatorProperties::gaussian_efficiency(&trimmed),
            0.8,
            epsilon = 0.01
        );

        let winsorized = WinsorizedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.1).unwrap();
        assert_relative_eq!(
            SpreadEstimatorProperties::gaussian_efficiency(&winsorized),
            0.9,
            epsilon = 0.01
        );
    }

    #[test]
    fn test_small_samples() {
        let mut sample = vec![1.0, 2.0];
        let estimator = harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );

        let trimmed = TrimmedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.1).unwrap();
        assert!(trimmed.estimate(&mut sample, &estimator, &cache).is_err());

        let winsorized = WinsorizedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.1).unwrap();
        let mut sample2 = vec![1.0, 2.0];
        assert!(winsorized.estimate(&mut sample2, &estimator, &cache).is_err());
    }
    
    #[test]
    fn test_sorted_methods() {
        let mut unsorted = vec![10.0, 2.0, 8.0, 4.0, 6.0, 1.0, 9.0, 3.0, 7.0, 5.0];
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let estimator = harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        
        let trimmed = TrimmedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.1).unwrap();
        let result_unsorted = trimmed.estimate(&mut unsorted, &estimator, &cache).unwrap();
        let result_sorted = trimmed.estimate_sorted(&sorted, &estimator, &cache).unwrap();
        assert_relative_eq!(result_unsorted, result_sorted, epsilon = 1e-10);
        
        let winsorized = WinsorizedStd::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.1).unwrap();
        let mut unsorted2 = vec![10.0, 2.0, 8.0, 4.0, 6.0, 1.0, 9.0, 3.0, 7.0, 5.0];
        let result_unsorted = winsorized.estimate(&mut unsorted2, &estimator, &cache).unwrap();
        let result_sorted = winsorized.estimate_sorted(&sorted, &estimator, &cache).unwrap();
        assert_relative_eq!(result_unsorted, result_sorted, epsilon = 1e-10);
    }
}
