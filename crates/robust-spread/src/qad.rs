//! Quantile Absolute Deviation (QAD) estimators
//!
//! QAD is a robust scale estimator defined as:
//! QAD(p) = K_{p,n} * quantile_p(|X - median(X)|)
//!
//! This crate provides:
//! - Basic QAD with custom quantile
//! - Standard QAD (SQAD) with p = Φ(1) - Φ(-1) ≈ 0.6827
//! - Optimal QAD (OQAD) with p ≈ 0.8617
//!
//! Note: MAD is a special case of QAD with p = 0.5 and constant ≈ 1.4826
//! 
//! Trade-offs:
//! - MAD (p=0.5): 50% breakdown, 36.75% efficiency
//! - SQAD (p≈0.6827): 31.73% breakdown, 54.06% efficiency
//! - OQAD (p≈0.8617): 13.83% breakdown, 65.22% efficiency

use crate::kernels::QadKernel;
use crate::traits::{RobustScale, SpreadEstimator, SpreadEstimatorProperties};
use robust_core::{ComputePrimitives, Result, Numeric};
use robust_quantile::QuantileEstimator;

/// Quantile Absolute Deviation estimator
/// 
/// Computes the p-th quantile of absolute deviations from the median.
/// Uses kernel architecture for optimal performance.
#[derive(Debug, Clone)]
pub struct Qad<T: Numeric = f64, P: ComputePrimitives<T> = robust_core::primitives::ScalarBackend> {
    kernel: QadKernel<T, P>,
}

/// Standard Quantile Absolute Deviation (SQAD)
/// 
/// Uses p = Φ(1) - Φ(-1) ≈ 0.6827 with appropriate consistency constants.
#[derive(Debug, Clone)]
pub struct StandardQad<T: Numeric = f64, P: ComputePrimitives<T> = robust_core::primitives::ScalarBackend> {
    /// The base QAD estimator
    qad: Qad<T, P>,
}

/// Optimal Quantile Absolute Deviation (OQAD)  
/// 
/// Uses p ≈ 0.8617 with optimal consistency constants.
#[derive(Debug, Clone)]
pub struct OptimalQad<T: Numeric = f64, P: ComputePrimitives<T> = robust_core::primitives::ScalarBackend> {
    /// The base QAD estimator
    qad: Qad<T, P>,
}

impl<T: Numeric, P: ComputePrimitives<T>> Qad<T, P> {
    /// Create QAD with specified quantile and constant 1.0
    pub fn new(primitives: P, p: f64) -> Result<Self> {
        Self::with_constant(primitives, p, 1.0)
    }

    /// Create QAD with custom quantile and constant
    pub fn with_constant(primitives: P, p: f64, constant: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(robust_core::Error::InvalidInput(
                "Quantile must be between 0 and 1".to_string(),
            ));
        }
        if constant <= 0.0 {
            return Err(robust_core::Error::InvalidInput(
                "Constant must be positive".to_string(),
            ));
        }

        Ok(Self {
            kernel: QadKernel::new(primitives, p, constant),
        })
    }
    
    /// Create MAD as a special case of QAD (p=0.5, constant=1.0)
    /// 
    /// MAD has 50% breakdown point and 36.75% efficiency at Gaussian
    pub fn mad(primitives: P) -> Self {
        Self {
            kernel: QadKernel::new(primitives, 0.5, 1.0), // Raw MAD without standardization
        }
    }
    
    /// Create standardized MAD (p=0.5, constant=1.482602)
    /// 
    /// Standardized to be consistent with standard deviation for normal data
    pub fn standardized_mad(primitives: P) -> Self {
        Self {
            kernel: QadKernel::new(primitives, 0.5, 1.482602), // Exact constant from the paper
        }
    }

    /// Get the theoretical breakdown point (1 - p)
    pub fn theoretical_breakdown_point(&self) -> f64 {
        1.0 - self.kernel.p
    }
    
    /// Get the quantile parameter
    pub fn p(&self) -> f64 {
        self.kernel.p
    }
    
    /// Get the constant
    pub fn constant(&self) -> f64 {
        self.kernel.constant
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> StandardQad<T, P> {
    /// Create Standard QAD with appropriate constants for sample size
    pub fn new(primitives: P, n: usize) -> Self {
        // Constants from the R implementation
        const CONSTANTS: [f64; 100] = [
            0.0, 1.7724, 1.3506, 1.3762, 1.1881, 1.1773, 1.1289, 1.1248, 1.0920, 1.0943,
            1.0764, 1.0738, 1.0630, 1.0637, 1.0533, 1.0537, 1.0482, 1.0468, 1.0419, 1.0429,
            1.0377, 1.0376, 1.0351, 1.0343, 1.0314, 1.0320, 1.0292, 1.0290, 1.0272, 1.0271,
            1.0251, 1.0253, 1.0238, 1.0235, 1.0223, 1.0224, 1.0210, 1.0210, 1.0201, 1.0199,
            1.0189, 1.0192, 1.0180, 1.0180, 1.0174, 1.0172, 1.0165, 1.0166, 1.0158, 1.0158,
            1.0152, 1.0152, 1.0146, 1.0146, 1.0141, 1.0140, 1.0135, 1.0137, 1.0130, 1.0131,
            1.0127, 1.0126, 1.0123, 1.0124, 1.0118, 1.0119, 1.0115, 1.0115, 1.0111, 1.0112,
            1.0108, 1.0108, 1.0106, 1.0106, 1.0102, 1.0103, 1.0100, 1.0100, 1.0097, 1.0097,
            1.0095, 1.0095, 1.0093, 1.0092, 1.0090, 1.0091, 1.0089, 1.0088, 1.0086, 1.0086,
            1.0084, 1.0084, 1.0082, 1.0082, 1.0081, 1.0081, 1.0079, 1.0079, 1.0078, 1.0077
        ];
        
        let constant = if n < CONSTANTS.len() {
            CONSTANTS[n]
        } else {
            // Formula for n > 100
            let n_f = n as f64;
            1.0 + 0.762 / n_f + 0.967 / (n_f * n_f)
        };
        
        // p = Φ(1) - Φ(-1) ≈ 0.6827
        const P_STANDARD: f64 = 0.682689492137086;
        
        Self {
            qad: Qad {
                kernel: QadKernel::new(primitives, P_STANDARD, constant),
            },
        }
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> OptimalQad<T, P> {
    /// Create Optimal QAD with appropriate constants for sample size
    pub fn new(primitives: P, n: usize) -> Self {
        // Constants from the R implementation
        const CONSTANTS: [f64; 100] = [
            0.0, 1.7729, 0.9788, 0.9205, 0.8194, 0.8110, 0.7792, 0.7828, 0.7600, 0.7535,
            0.7388, 0.7365, 0.7282, 0.7284, 0.7241, 0.7234, 0.7170, 0.7155, 0.7113, 0.7110,
            0.7083, 0.7088, 0.7068, 0.7056, 0.7030, 0.7024, 0.7006, 0.7006, 0.6995, 0.6998,
            0.6979, 0.6974, 0.6960, 0.6958, 0.6949, 0.6949, 0.6944, 0.6940, 0.6929, 0.6927,
            0.6918, 0.6918, 0.6913, 0.6914, 0.6907, 0.6904, 0.6897, 0.6896, 0.6891, 0.6892,
            0.6888, 0.6887, 0.6882, 0.6880, 0.6875, 0.6875, 0.6871, 0.6872, 0.6870, 0.6868,
            0.6863, 0.6862, 0.6859, 0.6859, 0.6857, 0.6858, 0.6854, 0.6853, 0.6850, 0.6849,
            0.6847, 0.6847, 0.6846, 0.6845, 0.6842, 0.6841, 0.6839, 0.6839, 0.6837, 0.6838,
            0.6836, 0.6834, 0.6833, 0.6832, 0.6831, 0.6830, 0.6829, 0.6830, 0.6827, 0.6827,
            0.6825, 0.6825, 0.6823, 0.6823, 0.6823, 0.6822, 0.6820, 0.6820, 0.6819, 0.6819
        ];
        
        let constant = if n < CONSTANTS.len() {
            CONSTANTS[n]
        } else {
            // Formula for n > 100
            let n_f = n as f64;
            0.6747309 * (1.0 + 1.047 / n_f + 1.193 / (n_f * n_f))
        };
        
        // Optimal p value
        const P_OPTIMAL: f64 = 0.861678977787423;
        
        Self {
            qad: Qad {
                kernel: QadKernel::new(primitives, P_OPTIMAL, constant),
            },
        }
    }
}

// Trait implementations for SpreadEstimatorProperties

impl<T: Numeric, P: ComputePrimitives<T>> SpreadEstimatorProperties for Qad<T, P> {
    fn name(&self) -> &str {
        "QAD"
    }

    fn is_robust(&self) -> bool {
        true
    }

    fn breakdown_point(&self) -> f64 {
        1.0 - self.kernel.p // QAD has breakdown point of 1 - p
    }
    
    fn gaussian_efficiency(&self) -> f64 {
        // MAD has 36.75% efficiency, efficiency improves with higher p
        // This is an approximation - exact values depend on p
        if (self.kernel.p - 0.5).abs() < 0.001 {
            0.3675  // MAD
        } else if (self.kernel.p - 0.6827).abs() < 0.01 {
            0.5406  // Standard QAD
        } else if (self.kernel.p - 0.8617).abs() < 0.01 {
            0.6522  // Optimal QAD
        } else {
            // Linear interpolation as approximation
            0.3675 + (self.kernel.p - 0.5) * 0.5
        }
    }
}

// Trait implementations for parameterized SpreadEstimator

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> SpreadEstimator<T, Q> for Qad<T, P> {
    
    fn estimate(&self, data: &mut [T], quantile_est: &Q, cache: &Q::State) -> Result<T::Float> {
        self.kernel.compute_qad(data, quantile_est, cache)
    }
    
    fn estimate_sorted(&self, sorted_data: &[T], quantile_est: &Q, cache: &Q::State) -> Result<T::Float> {
        self.kernel.compute_qad_sorted(sorted_data, quantile_est, cache)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> SpreadEstimatorProperties for StandardQad<T, P> {
    fn name(&self) -> &str {
        "Standard QAD"
    }

    fn is_robust(&self) -> bool {
        true
    }

    fn breakdown_point(&self) -> f64 {
        1.0 - self.qad.p() // Standard QAD has breakdown point ≈ 31.73%
    }
    
    fn gaussian_efficiency(&self) -> f64 {
        0.5406 // Standard QAD has 54.06% efficiency at Gaussian
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> SpreadEstimator<T, Q> for StandardQad<T, P> {
    
    fn estimate(&self, data: &mut [T], quantile_est: &Q, cache: &Q::State) -> Result<T::Float> {
        self.qad.estimate(data, quantile_est, cache)
    }
    
    fn estimate_sorted(&self, sorted_data: &[T], quantile_est: &Q, cache: &Q::State) -> Result<T::Float> {
        self.qad.estimate_sorted(sorted_data, quantile_est, cache)
    }
}


impl<T: Numeric, P: ComputePrimitives<T>> SpreadEstimatorProperties for OptimalQad<T, P> {
    fn name(&self) -> &str {
        "Optimal QAD"
    }

    fn is_robust(&self) -> bool {
        true
    }

    fn breakdown_point(&self) -> f64 {
        1.0 - self.qad.p() // Optimal QAD has breakdown point ≈ 13.83%
    }
    
    fn gaussian_efficiency(&self) -> f64 {
        0.6522 // Optimal QAD has 65.22% efficiency at Gaussian
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> SpreadEstimator<T, Q> for OptimalQad<T, P> {
    
    fn estimate(&self, data: &mut [T], quantile_est: &Q, cache: &Q::State) -> Result<T::Float> {
        self.qad.estimate(data, quantile_est, cache)
    }
    
    fn estimate_sorted(&self, sorted_data: &[T], quantile_est: &Q, cache: &Q::State) -> Result<T::Float> {
        self.qad.estimate_sorted(sorted_data, quantile_est, cache)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> RobustScale<T, Q> for StandardQad<T, P> {}

impl<T: Numeric, P: ComputePrimitives<T>, Q: QuantileEstimator<T>> RobustScale<T, Q> for OptimalQad<T, P> {}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::traits::{SpreadEstimator, SpreadEstimatorProperties};

    #[test]
    fn test_basic_qad() {
        let mut sample = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let qad = Qad::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.5).unwrap(); // Median of absolute deviations
        let estimator = robust_quantile::estimators::harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        let result = qad.estimate(&mut sample, &estimator, &cache).unwrap();

        // For symmetric data centered at 5, deviations are [4,3,2,1,0,1,2,3,4]
        // Median of these is 2.0
        assert!(result > 1.5 && result < 2.5);
    }

    #[test]
    fn test_qad_custom_quantile() {
        let mut sample = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let qad = Qad::<f64, _>::with_constant(robust_core::primitives::ScalarBackend::new(), 0.75, 1.0).unwrap(); // 75th percentile of deviations
        let estimator = robust_quantile::estimators::harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        let result = qad.estimate(&mut sample, &estimator, &cache).unwrap();

        // Should be larger than median of deviations
        assert!(result > 1.0);
    }

    #[test]
    fn test_qad_breakdown() {
        // MAD (p=0.5) has 50% breakdown
        let mad = Qad::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.5).unwrap();
        assert_eq!(mad.breakdown_point(), 0.5);

        // Higher p means lower breakdown
        let qad_high_p = Qad::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.9).unwrap();
        assert_relative_eq!(qad_high_p.breakdown_point(), 0.1, epsilon = 0.0001);
        
        // Standard QAD should have ~31.73% breakdown
        let sqad = StandardQad::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 100);
        assert!((sqad.breakdown_point() - 0.3173).abs() < 0.001);
        
        // Optimal QAD should have ~13.83% breakdown
        let oqad = OptimalQad::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 100);
        assert!((oqad.breakdown_point() - 0.1383).abs() < 0.001);
    }

    #[test]
    fn test_standard_qad() {
        let mut sample = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let qad_est = StandardQad::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), sample.len());
        let estimator = robust_quantile::estimators::harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        let result = qad_est.estimate(&mut sample, &estimator, &cache).unwrap();

        // Should be standardized with appropriate constant
        assert!(result > 1.5 && result < 3.5);
    }
    
    #[test]
    fn test_optimal_qad() {
        let mut sample = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let qad_est = OptimalQad::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), sample.len());
        let estimator = robust_quantile::estimators::harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        let result = qad_est.estimate(&mut sample, &estimator, &cache).unwrap();

        // Optimal QAD should give different result due to different p and constant
        assert!(result > 1.0 && result < 3.0);
    }

    #[test]
    fn test_invalid_quantiles() {
        assert!(Qad::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), -0.1).is_err());
        assert!(Qad::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 1.1).is_err());
        assert!(Qad::<f64, _>::with_constant(robust_core::primitives::ScalarBackend::new(), 0.5, -1.0).is_err());
        assert!(Qad::<f64, _>::with_constant(robust_core::primitives::ScalarBackend::new(), 0.5, 0.0).is_err());
    }

    #[test]
    fn test_efficiency() {
        // Standard QAD should have 54.06% efficiency
        let sqad = StandardQad::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 100);
        assert_relative_eq!(
            sqad.gaussian_efficiency(),
            0.5406,
            epsilon = 0.001
        );

        // Optimal QAD should have 65.22% efficiency
        let oqad = OptimalQad::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 100);
        assert_relative_eq!(
            oqad.gaussian_efficiency(),
            0.6522,
            epsilon = 0.001
        );
        
        // MAD should have 36.75% efficiency
        let mad = Qad::<f64, _>::mad(robust_core::primitives::ScalarBackend::new());
        assert_relative_eq!(
            mad.gaussian_efficiency(),
            0.3675,
            epsilon = 0.001
        );
    }
    
    #[test]
    fn test_qad_sorted() {
        let mut unsorted = vec![9.0, 2.0, 5.0, 1.0, 7.0, 3.0, 8.0, 4.0, 6.0];
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        
        let qad = Qad::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.5).unwrap();
        let estimator = robust_quantile::estimators::harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        let result_unsorted = qad.estimate(&mut unsorted, &estimator, &cache).unwrap();
        let result_sorted = qad.estimate_sorted(&sorted, &estimator, &cache).unwrap();
        
        // Results should be identical
        assert!((result_unsorted - result_sorted).abs() < 0.1);
    }
    
    #[test]
    fn test_qad_with_outliers() {
        let mut sample = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // Outlier at the end
        let qad = Qad::<f64, _>::new(robust_core::primitives::ScalarBackend::new(), 0.5).unwrap();
        let estimator = robust_quantile::estimators::harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache
        );
        let result = qad.estimate(&mut sample, &estimator, &cache).unwrap();
        
        // QAD should be robust to outliers
        assert!(result < 10.0); // Much less than would be expected if outlier dominated
    }
}