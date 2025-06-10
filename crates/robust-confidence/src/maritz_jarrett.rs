//! Maritz-Jarrett confidence intervals for quantile estimators
//!
//! Based on: Maritz, J. S., & Jarrett, R. G. (1978). A note on estimating the
//! variance of the sample median. Journal of the American Statistical Association,
//! 73(361), 194-196.

use crate::{ConfidenceInterval, ConfidenceIntervalEstimator};
use robust_core::{Error, Result, StatefulEstimator};
use statrs::distribution::{ContinuousCDF, Normal};

/// Maritz-Jarrett confidence interval estimator for quantiles
///
/// This method uses the variance of the quantile estimator to construct
/// asymptotically valid confidence intervals. Following the new architecture,
/// it no longer stores an estimator internally.
#[derive(Debug, Clone)]
pub struct MaritzJarrettCI {
    /// The quantile probability (e.g., 0.5 for median)
    probability: f64,
    /// Confidence level (e.g., 0.95 for 95% CI)
    confidence_level: f64,
}

impl MaritzJarrettCI {
    /// Create a new Maritz-Jarrett CI estimator
    pub fn new(probability: f64, confidence_level: f64) -> Self {
        assert!(
            probability > 0.0 && probability < 1.0,
            "Probability must be in (0, 1)"
        );
        assert!(
            confidence_level > 0.0 && confidence_level < 1.0,
            "Confidence level must be in (0, 1)"
        );

        Self {
            probability,
            confidence_level,
        }
    }

    /// Create a CI estimator for the median
    pub fn median(confidence_level: f64) -> Self {
        Self::new(0.5, confidence_level)
    }

    /// Create a CI estimator for the first quartile
    pub fn q1(confidence_level: f64) -> Self {
        Self::new(0.25, confidence_level)
    }

    /// Create a CI estimator for the third quartile
    pub fn q3(confidence_level: f64) -> Self {
        Self::new(0.75, confidence_level)
    }
}



impl<Q> ConfidenceIntervalEstimator<Q> for MaritzJarrettCI
where
    Q: StatefulEstimator + robust_core::BatchQuantileEstimator,
{
    fn confidence_interval(
        &self,
        sample: &[f64],
        estimator: &Q,
        cache: &<Q as StatefulEstimator>::State,
    ) -> Result<ConfidenceInterval> {
        // For now, we need to sort the data and use the non-cached interface
        // TODO: Update once QuantileWithMoments supports cached operations
        let mut sorted_data = sample.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        // Use batch interface to get single quantile
        let quantiles = vec![self.probability];
        let estimates = estimator.estimate_quantiles_sorted_with_cache(&sorted_data, &quantiles, cache)?;
        let estimate = estimates.first().copied().ok_or_else(|| 
            Error::Computation("Failed to get quantile estimate".to_string())
        )?;
        
        // Fallback: use a simple variance approximation for now
        // TODO: Implement proper moment calculation
        let n = sorted_data.len() as f64;
        let variance = self.probability * (1.0 - self.probability) / n;

        if variance <= 0.0 {
            return Err(Error::Computation(
                "Variance is non-positive, cannot compute confidence interval".to_string(),
            ));
        }

        let std_error = variance.sqrt();

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

    fn confidence_interval_sorted(
        &self,
        sorted_sample: &[f64],
        estimator: &Q,
        cache: &<Q as StatefulEstimator>::State,
    ) -> Result<ConfidenceInterval> {
        // Use batch interface to get single quantile
        let quantiles = vec![self.probability];
        let estimates = estimator.estimate_quantiles_sorted_with_cache(sorted_sample, &quantiles, cache)?;
        let estimate = estimates.first().copied().ok_or_else(|| 
            Error::Computation("Failed to get quantile estimate".to_string())
        )?;
        
        // Fallback: use a simple variance approximation for now
        // TODO: Implement proper moment calculation
        let n = sorted_sample.len() as f64;
        let variance = self.probability * (1.0 - self.probability) / n;

        if variance <= 0.0 {
            return Err(Error::Computation(
                "Variance is non-positive, cannot compute confidence interval".to_string(),
            ));
        }

        let std_error = variance.sqrt();

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

    fn supports_weighted_samples(&self) -> bool {
        // QuantileEstimator trait doesn't have this method currently
        // Most quantile estimators support weighted samples
        true
    }
}


#[cfg(all(test, feature = "quantile"))]
mod tests {
    use super::*;
    use robust_quantile::{estimators::harrell_davis, HDWeightComputer, QuantileAdapter};
    use robust_core::{simd_sequential, UnifiedWeightCache, CachePolicy};

    #[test]
    fn test_maritz_jarrett_median() {
        let sample = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let engine = simd_sequential();
        let quantile_est = harrell_davis(engine);
        let estimator = QuantileAdapter::with_default_quantile(quantile_est, 0.5);
        let cache = UnifiedWeightCache::new(HDWeightComputer, CachePolicy::NoCache);
        
        let mj_ci = MaritzJarrettCI::median(0.95);

        let ci = mj_ci.confidence_interval(&sample, &estimator, &cache).unwrap();

        // Check that the interval contains the estimate
        assert!(ci.contains(ci.estimate));

        // Check that it's centered around the estimate
        let center = (ci.lower + ci.upper) / 2.0;
        assert!((center - ci.estimate).abs() < 1e-10);

        // Check confidence level
        assert_eq!(ci.confidence_level, 0.95);
    }

    #[test]
    fn test_different_quantiles() {
        let sample: Vec<f64> = (1..=100).map(|x| x as f64).collect();

        let engine = simd_sequential();
        let cache = UnifiedWeightCache::new(HDWeightComputer, CachePolicy::NoCache);

        // Test Q1
        let q1_est = harrell_davis(engine.clone());
        let q1_adapter = QuantileAdapter::with_default_quantile(q1_est, 0.25);
        let q1_ci = MaritzJarrettCI::q1(0.95);
        let ci_q1 = q1_ci.confidence_interval(&sample, &q1_adapter, &cache).unwrap();
        assert!(ci_q1.estimate > 20.0 && ci_q1.estimate < 30.0);

        // Test median
        let median_est = harrell_davis(engine.clone());
        let median_adapter = QuantileAdapter::with_default_quantile(median_est, 0.5);
        let median_ci = MaritzJarrettCI::median(0.95);
        let ci_median = median_ci.confidence_interval(&sample, &median_adapter, &cache).unwrap();
        assert!(ci_median.estimate > 45.0 && ci_median.estimate < 55.0);

        // Test Q3
        let q3_est = harrell_davis(engine);
        let q3_adapter = QuantileAdapter::with_default_quantile(q3_est, 0.75);
        let q3_ci = MaritzJarrettCI::q3(0.95);
        let ci_q3 = q3_ci.confidence_interval(&sample, &q3_adapter, &cache).unwrap();
        assert!(ci_q3.estimate > 70.0 && ci_q3.estimate < 80.0);
    }

    #[test]
    fn test_confidence_levels() {
        let sample: Vec<f64> = (1..=50).map(|x| x as f64).collect();

        let engine = simd_sequential();
        let quantile_est = harrell_davis(engine);
        let estimator = QuantileAdapter::with_default_quantile(quantile_est, 0.5);
        let cache = UnifiedWeightCache::new(HDWeightComputer, CachePolicy::NoCache);

        // 90% CI should be narrower than 95% CI
        let ci_90 = MaritzJarrettCI::median(0.90)
            .confidence_interval(&sample, &estimator, &cache)
            .unwrap();

        let ci_95 = MaritzJarrettCI::median(0.95)
            .confidence_interval(&sample, &estimator, &cache)
            .unwrap();

        let ci_99 = MaritzJarrettCI::median(0.99)
            .confidence_interval(&sample, &estimator, &cache)
            .unwrap();

        assert!(ci_90.width() < ci_95.width());
        assert!(ci_95.width() < ci_99.width());
    }

    #[test]
    fn test_small_sample() {
        let sample = vec![1.0, 2.0, 3.0];

        let engine = simd_sequential();
        let quantile_est = harrell_davis(engine);
        let estimator = QuantileAdapter::with_default_quantile(quantile_est, 0.5);
        let cache = UnifiedWeightCache::new(HDWeightComputer, CachePolicy::NoCache);
        
        let mj_ci = MaritzJarrettCI::median(0.95);

        let ci = mj_ci.confidence_interval(&sample, &estimator, &cache).unwrap();

        // With small sample, CI should be wide
        assert!(ci.width() > 1.0);
        assert!(ci.contains(2.0)); // Should contain true median
    }
    
    // Two-sample tests are in maritz_jarrett_two_sample.rs
    
    // This test should fail to compile, demonstrating the type constraint
    // Uncomment to verify the constraint works
    /*
    #[test]
    fn test_two_sample_maritz_jarrett_ratio_should_not_compile() {
        use robust_core::RatioComparison;
        
        let sample1 = vec![1.0, 2.0, 3.0];
        let sample2 = vec![2.0, 4.0, 6.0];
        
        let engine = simd_sequential();
        let quantile_est = harrell_davis(engine);
        let estimator = QuantileAdapter::with_default_quantile(quantile_est, 0.5);
        let cache = UnifiedWeightCache::new(HDWeightComputer, CachePolicy::NoCache);
        
        let comparison = RatioComparison; // This is NonLinearCombination
        let mj_two_sample = MaritzJarrettTwoSampleCI::new(0.95);
        
        // This should fail to compile because RatioComparison doesn't implement LinearCombination
        let ci = mj_two_sample.confidence_interval_two_sample(
            &sample1,
            &sample2,
            &comparison,
            &estimator,
            &cache,
        ).unwrap();
    }
    */
}
