//! Maritz-Jarrett confidence intervals for two-sample comparisons
//!
//! This module provides Maritz-Jarrett CI for two-sample comparisons
//! that support linear combinations and variance estimation.

use crate::{ConfidenceInterval, TwoSampleConfidenceIntervalEstimator};
use robust_core::{
    Error, Result, TwoSampleComparison, 
    LinearComparison, EstimatorWithVariance,
};
use statrs::distribution::{ContinuousCDF, Normal};

/// Maritz-Jarrett two-sample confidence interval estimator
///
/// This uses the asymptotic normality of the estimator and requires:
/// 1. The comparison to be linear (e.g., shift, not ratio)
/// 2. The estimator to provide variance estimates
#[derive(Debug, Clone)]
pub struct MaritzJarrettTwoSampleCI {
    confidence_level: f64,
}

impl MaritzJarrettTwoSampleCI {
    /// Create a new Maritz-Jarrett two-sample CI estimator
    pub fn new(confidence_level: f64) -> Self {
        assert!(
            confidence_level > 0.0 && confidence_level < 1.0,
            "Confidence level must be in (0, 1)"
        );
        
        Self { confidence_level }
    }
}

impl<C, E> TwoSampleConfidenceIntervalEstimator<C, E> for MaritzJarrettTwoSampleCI
where
    C: TwoSampleComparison<E> + LinearComparison,
    E: EstimatorWithVariance,
    C::Output: Into<f64>,
{
    fn confidence_interval_two_sample(
        &self,
        sample1: &[f64],
        sample2: &[f64],
        comparison: &C,
        estimator: &E,
        cache: &E::State,
    ) -> Result<ConfidenceInterval> {
        // Get estimates and variances for both samples
        let (_est1, var1) = estimator.estimate_with_variance(sample1, cache)?;
        let (_est2, var2) = estimator.estimate_with_variance(sample2, cache)?;
        
        // Get the comparison value (e.g., shift)
        let comparison_value = comparison.shift(estimator, sample1, sample2, cache)?;
        let estimate: f64 = comparison_value.into();
        
        // For independent samples and linear combinations:
        // Var(A - B) = Var(A) + Var(B)
        let combined_variance = var1 + var2;
        
        if combined_variance <= 0.0 {
            return Err(Error::Computation(
                "Combined variance is non-positive, cannot compute confidence interval".to_string(),
            ));
        }
        
        let std_error = combined_variance.sqrt();
        
        // Calculate critical value from normal distribution
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            Error::Computation(format!("Failed to create normal distribution: {}", e))
        })?;
        
        let alpha = 1.0 - self.confidence_level;
        let z_critical = normal.inverse_cdf(1.0 - alpha / 2.0);
        
        // Calculate confidence interval
        let margin = z_critical * std_error;
        let lower = estimate - margin;
        let upper = estimate + margin;
        
        Ok(ConfidenceInterval::new(
            lower,
            upper,
            estimate,
            self.confidence_level,
        ))
    }
    
    fn confidence_level(&self) -> f64 {
        self.confidence_level
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robust_core::{ShiftComparison, NoCache, StatefulEstimator};
    
    // Mock estimator with variance for testing
    #[derive(Clone)]
    struct MeanWithVariance;
    
    impl StatefulEstimator for MeanWithVariance {
        type State = NoCache;
        
        fn estimate_with_cache(&self, sample: &[f64], _cache: &Self::State) -> Result<f64> {
            if sample.is_empty() {
                return Err(Error::InvalidInput("Empty sample".to_string()));
            }
            Ok(sample.iter().sum::<f64>() / sample.len() as f64)
        }
    }
    
    impl EstimatorWithVariance for MeanWithVariance {
        fn estimate_with_variance(
            &self,
            sample: &[f64],
            cache: &Self::State,
        ) -> Result<(f64, f64)> {
            let mean = self.estimate_with_cache(sample, cache)?;
            let n = sample.len() as f64;
            
            // Sample variance of the mean: s²/n
            let sample_var = sample.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (n - 1.0);
            
            let variance_of_mean = sample_var / n;
            
            Ok((mean, variance_of_mean))
        }
    }
    
    #[test]
    fn test_maritz_jarrett_two_sample() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![4.0, 5.0, 6.0, 7.0, 8.0];
        
        let estimator = MeanWithVariance;
        let comparison = ShiftComparison;
        let cache = NoCache;
        
        let mj_ci = MaritzJarrettTwoSampleCI::new(0.95);
        
        let ci = mj_ci.confidence_interval_two_sample(
            &sample1,
            &sample2,
            &comparison,
            &estimator,
            &cache,
        ).unwrap();
        
        // Mean difference should be 3.0 - 6.0 = -3.0
        assert!((ci.estimate - (-3.0)).abs() < 1e-10);
        assert!(ci.contains(-3.0));
        assert_eq!(ci.confidence_level, 0.95);
    }
    
    #[test] 
    fn test_type_safety() {
        // This test verifies that non-linear comparisons cannot be used
        // with MaritzJarrettTwoSampleCI at compile time.
        
        // The following should NOT compile:
        // let ratio_comparison = RatioComparison;
        // let mj_ci = MaritzJarrettTwoSampleCI::new(0.95);
        // mj_ci.confidence_interval_two_sample(..., &ratio_comparison, ...);
        
        // This is enforced by the LinearComparison bound
    }
}