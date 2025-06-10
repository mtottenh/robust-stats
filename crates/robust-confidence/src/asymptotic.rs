//! Asymptotic confidence intervals based on theoretical distributions

use crate::{ConfidenceInterval, ConfidenceIntervalEstimator};
use robust_core::{Error, Result, StatefulEstimator};
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

/// Asymptotic confidence interval estimator
///
/// This estimator uses theoretical asymptotic distributions to construct
/// confidence intervals. It assumes the estimator is asymptotically normal
/// with a known or estimable standard error.
#[derive(Debug, Clone)]
pub struct AsymptoticCI {
    /// Confidence level
    confidence_level: f64,
    /// Whether to use t-distribution (true) or normal (false)
    use_t_distribution: bool,
}

impl AsymptoticCI {
    /// Create a new asymptotic CI estimator using normal distribution
    pub fn normal(confidence_level: f64) -> Self {
        assert!(
            confidence_level > 0.0 && confidence_level < 1.0,
            "Confidence level must be in (0, 1)"
        );

        Self {
            confidence_level,
            use_t_distribution: false,
        }
    }

    /// Create a new asymptotic CI estimator using t-distribution
    pub fn students_t(confidence_level: f64) -> Self {
        assert!(
            confidence_level > 0.0 && confidence_level < 1.0,
            "Confidence level must be in (0, 1)"
        );

        Self {
            confidence_level,
            use_t_distribution: true,
        }
    }
}

/// Trait for estimators that can provide their standard error
pub trait StandardErrorEstimator: StatefulEstimator {
    /// Calculate the standard error of the estimate
    fn standard_error(&self, sample: &[f64], cache: &Self::State) -> Result<f64>;
    
    /// Calculate the standard error from pre-sorted data
    fn standard_error_sorted(&self, sorted_sample: &[f64], cache: &Self::State) -> Result<f64> {
        // Default implementation uses unsorted method
        self.standard_error(sorted_sample, cache)
    }
}

impl<E> ConfidenceIntervalEstimator<E> for AsymptoticCI
where
    E: StandardErrorEstimator,
{
    fn confidence_interval(
        &self,
        sample: &[f64],
        estimator: &E,
        cache: &E::State,
    ) -> Result<ConfidenceInterval> {
        // Get point estimate
        let estimate = estimator.estimate_with_cache(sample, cache)?;

        // Get standard error
        let std_error = estimator.standard_error(sample, cache)?;

        if std_error <= 0.0 {
            return Err(Error::Computation(
                "Standard error is non-positive".to_string(),
            ));
        }

        // Calculate critical value
        let alpha = 1.0 - self.confidence_level;
        let critical_value = if self.use_t_distribution {
            // Use t-distribution with n-1 degrees of freedom
            let df = (sample.len() - 1) as f64;
            if df <= 0.0 {
                return Err(Error::InvalidInput(
                    "Not enough data for t-distribution".to_string(),
                ));
            }

            let t_dist = StudentsT::new(0.0, 1.0, df).map_err(|e| {
                Error::Computation(format!("Failed to create t-distribution: {}", e))
            })?;
            t_dist.inverse_cdf(1.0 - alpha / 2.0)
        } else {
            // Use normal distribution
            let normal = Normal::new(0.0, 1.0).map_err(|e| {
                Error::Computation(format!("Failed to create normal distribution: {}", e))
            })?;
            normal.inverse_cdf(1.0 - alpha / 2.0)
        };

        // Calculate confidence interval
        let margin = critical_value * std_error;
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
        estimator: &E,
        cache: &E::State,
    ) -> Result<ConfidenceInterval> {
        // Get point estimate
        let estimate = estimator.estimate_sorted_with_cache(sorted_sample, cache)?;

        // Get standard error
        let std_error = estimator.standard_error_sorted(sorted_sample, cache)?;

        if std_error <= 0.0 {
            return Err(Error::Computation(
                "Standard error is non-positive".to_string(),
            ));
        }

        // Calculate critical value
        let alpha = 1.0 - self.confidence_level;
        let critical_value = if self.use_t_distribution {
            // Use t-distribution with n-1 degrees of freedom
            let df = (sorted_sample.len() - 1) as f64;
            if df <= 0.0 {
                return Err(Error::InvalidInput(
                    "Not enough data for t-distribution".to_string(),
                ));
            }

            let t_dist = StudentsT::new(0.0, 1.0, df).map_err(|e| {
                Error::Computation(format!("Failed to create t-distribution: {}", e))
            })?;
            t_dist.inverse_cdf(1.0 - alpha / 2.0)
        } else {
            // Use normal distribution
            let normal = Normal::new(0.0, 1.0).map_err(|e| {
                Error::Computation(format!("Failed to create normal distribution: {}", e))
            })?;
            normal.inverse_cdf(1.0 - alpha / 2.0)
        };

        // Calculate confidence interval
        let margin = critical_value * std_error;
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

/// Example: Mean estimator with standard error
#[derive(Debug, Clone, Copy)]
pub struct MeanWithSE;

/// No-op cache for mean estimator
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanCache;

impl StatefulEstimator for MeanWithSE {
    type State = MeanCache;

    fn estimate_with_cache(&self, sample: &[f64], _cache: &Self::State) -> Result<f64> {
        if sample.is_empty() {
            return Err(Error::InvalidInput("Empty sample".to_string()));
        }
        Ok(sample.iter().sum::<f64>() / sample.len() as f64)
    }
}

impl StandardErrorEstimator for MeanWithSE {
    fn standard_error(&self, sample: &[f64], _cache: &Self::State) -> Result<f64> {
        if sample.len() < 2 {
            return Err(Error::InvalidInput(
                "Need at least 2 observations for standard error".to_string(),
            ));
        }

        let mean = self.estimate_with_cache(sample, &MeanCache)?;
        let variance = sample
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (sample.len() - 1) as f64;

        Ok((variance / sample.len() as f64).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asymptotic_normal() {
        let sample: Vec<f64> = (1..=30).map(|x| x as f64).collect();

        let ci_estimator = AsymptoticCI::normal(0.95);
        let estimator = MeanWithSE;
        let cache = MeanCache;
        let ci = ci_estimator.confidence_interval(&sample, &estimator, &cache).unwrap();

        // Check that CI contains the true mean (15.5)
        assert!(ci.contains(15.5));

        // Check confidence level
        assert_eq!(ci.confidence_level, 0.95);
    }

    #[test]
    fn test_asymptotic_t() {
        let sample = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let estimator = MeanWithSE;
        let cache = MeanCache;

        let ci_normal = AsymptoticCI::normal(0.95)
            .confidence_interval(&sample, &estimator, &cache)
            .unwrap();

        let ci_t = AsymptoticCI::students_t(0.95)
            .confidence_interval(&sample, &estimator, &cache)
            .unwrap();

        // t-distribution CI should be wider with small sample
        assert!(ci_t.width() > ci_normal.width());

        // Both should contain the true mean (3.0)
        assert!(ci_normal.contains(3.0));
        assert!(ci_t.contains(3.0));
    }

    #[test]
    fn test_confidence_levels() {
        let sample: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let estimator = MeanWithSE;
        let cache = MeanCache;

        let ci_90 = AsymptoticCI::normal(0.90)
            .confidence_interval(&sample, &estimator, &cache)
            .unwrap();

        let ci_95 = AsymptoticCI::normal(0.95)
            .confidence_interval(&sample, &estimator, &cache)
            .unwrap();

        let ci_99 = AsymptoticCI::normal(0.99)
            .confidence_interval(&sample, &estimator, &cache)
            .unwrap();

        // Higher confidence level should give wider interval
        assert!(ci_90.width() < ci_95.width());
        assert!(ci_95.width() < ci_99.width());
    }
}
