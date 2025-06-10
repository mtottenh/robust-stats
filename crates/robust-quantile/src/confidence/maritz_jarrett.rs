//! Maritz-Jarrett confidence intervals for Harrell-Davis quantiles
//!
//! Based on the method described in:
//! Maritz, J.S. and Jarrett, R.G. (1978). "A note on estimating the variance
//! of the sample median." Journal of the American Statistical Association.

use crate::{Error, QuantileKernel, Result};
use robust_core::SparseWeights;
// use robust_core::{ExecutionEngine, StatisticalKernel};
use statrs::distribution::{ContinuousCDF, Normal};

/// Confidence interval representation
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceInterval {
    /// Point estimate (quantile value)
    pub point: f64,
    /// Lower bound of the interval
    pub lower: f64,
    /// Upper bound of the interval
    pub upper: f64,
    /// Confidence level (e.g., 0.95 for 95%)
    pub level: f64,
}

impl ConfidenceInterval {
    /// Width of the confidence interval
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Check if a value is within the interval
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Margin of error (half-width)
    pub fn margin_of_error(&self) -> f64 {
        self.width() / 2.0
    }
}

/// Maritz-Jarrett confidence interval calculator
///
/// This method estimates the variance of Harrell-Davis quantiles
/// using the first two moments of the weighted order statistics.
#[derive(Clone)]
pub struct MaritzJarrett<K: QuantileKernel> {
    kernel: K,
}

impl<K: QuantileKernel> MaritzJarrett<K> {
    /// Create new Maritz-Jarrett calculator
    pub fn new(kernel: K) -> Self {
        Self { kernel }
    }

    /// Compute confidence interval for a quantile
    ///
    /// # Arguments
    /// * `sorted_data` - Pre-sorted data
    /// * `weights` - Sparse weights for the quantile
    /// * `confidence_level` - Desired confidence level (e.g., 0.95)
    pub fn confidence_interval(
        &self,
        sorted_data: &[f64],
        weights: &SparseWeights,
        confidence_level: f64,
    ) -> Result<ConfidenceInterval> {
        // Validate confidence level
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(Error::InvalidConfidenceLevel {
                level: confidence_level,
            });
        }

        let n = sorted_data.len();
        if n < 2 {
            return Err(Error::Core(robust_core::Error::InsufficientData {
                expected: 2,
                actual: n,
            }));
        }

        // Compute first and second moments
        let (c1, c2) = self
            .kernel
            .apply_sparse_weights_with_moments(sorted_data, weights);

        // Estimate variance using Maritz-Jarrett formula
        // Var(Q_p) ≈ (c2 - c1²) / n
        let variance = (c2 - c1 * c1) / (n as f64);

        if variance <= 0.0 {
            return Err(Error::Numerical(
                "Negative or zero variance in Maritz-Jarrett calculation".to_string(),
            ));
        }

        let std_error = variance.sqrt();

        // Get critical value from normal distribution
        let alpha = 1.0 - confidence_level;
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z = normal.inverse_cdf(1.0 - alpha / 2.0);

        // Construct confidence interval
        let margin = z * std_error;

        Ok(ConfidenceInterval {
            point: c1,
            lower: c1 - margin,
            upper: c1 + margin,
            level: confidence_level,
        })
    }

    /// Compute confidence intervals for multiple quantiles
    ///
    /// This is more efficient than computing them separately as it can
    /// reuse computations across quantiles.
    pub fn confidence_intervals_batch(
        &self,
        sorted_data: &[f64],
        weights_batch: &[&SparseWeights],
        confidence_level: f64,
    ) -> Result<Vec<ConfidenceInterval>> {
        // Validate confidence level
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(Error::InvalidConfidenceLevel {
                level: confidence_level,
            });
        }

        let n = sorted_data.len();
        if n < 2 {
            return Err(Error::Core(robust_core::Error::InsufficientData {
                expected: 2,
                actual: n,
            }));
        }

        // Get critical value once
        let alpha = 1.0 - confidence_level;
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z = normal.inverse_cdf(1.0 - alpha / 2.0);

        // Process each set of weights
        weights_batch
            .iter()
            .map(|weights| {
                let (c1, c2) = self
                    .kernel
                    .apply_sparse_weights_with_moments(sorted_data, weights);
                let variance = (c2 - c1 * c1) / (n as f64);

                if variance <= 0.0 {
                    return Err(Error::Numerical(
                        "Negative or zero variance in Maritz-Jarrett calculation".to_string(),
                    ));
                }

                let std_error = variance.sqrt();
                let margin = z * std_error;

                Ok(ConfidenceInterval {
                    point: c1,
                    lower: c1 - margin,
                    upper: c1 + margin,
                    level: confidence_level,
                })
            })
            .collect()
    }
}

/// Extension trait for computing confidence intervals
pub trait ConfidenceIntervalExt {
    /// Compute quantile with confidence interval
    fn quantile_with_ci(
        &self,
        data: &[f64],
        p: f64,
        confidence_level: f64,
    ) -> Result<ConfidenceInterval>;

    /// Compute multiple quantiles with confidence intervals
    fn quantiles_with_ci(
        &self,
        data: &[f64],
        ps: &[f64],
        confidence_level: f64,
    ) -> Result<Vec<ConfidenceInterval>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{kernels::WeightedSumKernel, weights::compute_hd_weights};
    use robust_core::simd_sequential;

    #[test]
    fn test_confidence_interval_basic() {
        let engine = simd_sequential();
        let kernel = WeightedSumKernel::new(engine);
        let mj = MaritzJarrett::new(kernel);

        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let weights = compute_hd_weights(data.len(), 0.5);

        let ci = mj.confidence_interval(&data, &weights, 0.95).unwrap();

        // Check basic properties
        assert!(ci.lower < ci.point);
        assert!(ci.point < ci.upper);
        assert_eq!(ci.level, 0.95);
        assert!(ci.contains(ci.point));

        // For uniform data, median should be close to 50.5
        assert!((ci.point - 50.5).abs() < 1.0);
    }

    #[test]
    fn test_confidence_interval_width() {
        let engine = simd_sequential();
        let kernel = WeightedSumKernel::new(engine);
        let mj = MaritzJarrett::new(kernel);

        // Use standard normal distribution samples for both
        // This ensures same underlying distribution, different sample sizes
        use rand::{SeedableRng, distributions::Distribution};
        use rand_chacha::ChaCha8Rng;
        use rand_distr::Normal;
        
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        // Generate samples from same distribution
        let mut small_data: Vec<f64> = (0..20)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let mut large_data: Vec<f64> = (0..2000)
            .map(|_| normal.sample(&mut rng))
            .collect();
            
        // Sort the data
        small_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        large_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let weights_small = compute_hd_weights(small_data.len(), 0.5);
        let weights_large = compute_hd_weights(large_data.len(), 0.5);

        let ci_small = mj
            .confidence_interval(&small_data, &weights_small, 0.95)
            .unwrap();
        let ci_large = mj
            .confidence_interval(&large_data, &weights_large, 0.95)
            .unwrap();
        
        println!(
            "Small sample (n={}): point={:.4}, width={:.4}",
            small_data.len(), ci_small.point, ci_small.width()
        );
        println!(
            "Large sample (n={}): point={:.4}, width={:.4}",
            large_data.len(), ci_large.point, ci_large.width()
        );
        
        // Larger sample should have narrower CI
        assert!(ci_small.width() > ci_large.width(), 
            "Expected small sample CI width ({}) > large sample CI width ({})",
            ci_small.width(), ci_large.width());
    }

    #[test]
    fn test_confidence_level_effect() {
        let engine = simd_sequential();
        let kernel = WeightedSumKernel::new(engine);
        let mj = MaritzJarrett::new(kernel);

        let data: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let weights = compute_hd_weights(data.len(), 0.5);

        let ci_90 = mj.confidence_interval(&data, &weights, 0.90).unwrap();
        let ci_95 = mj.confidence_interval(&data, &weights, 0.95).unwrap();
        let ci_99 = mj.confidence_interval(&data, &weights, 0.99).unwrap();

        // Higher confidence level should give wider intervals
        assert!(ci_90.width() < ci_95.width());
        assert!(ci_95.width() < ci_99.width());

        // All should have the same point estimate
        assert_eq!(ci_90.point, ci_95.point);
        assert_eq!(ci_95.point, ci_99.point);
    }

    #[test]
    fn test_batch_confidence_intervals() {
        let engine = simd_sequential();
        let kernel = WeightedSumKernel::new(engine);
        let mj = MaritzJarrett::new(kernel);

        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();

        // Multiple quantiles
        let ps = vec![0.25, 0.5, 0.75];
        let weights: Vec<_> = ps
            .iter()
            .map(|&p| compute_hd_weights(data.len(), p))
            .collect();
        let weight_refs: Vec<_> = weights.iter().collect();

        let cis = mj
            .confidence_intervals_batch(&data, &weight_refs, 0.95)
            .unwrap();

        assert_eq!(cis.len(), 3);

        // Check ordering: Q1 < median < Q3
        assert!(cis[0].point < cis[1].point);
        assert!(cis[1].point < cis[2].point);
    }
}
