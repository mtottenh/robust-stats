//! Bootstrap method implementations
//!
//! This module provides various bootstrap methods for constructing
//! confidence intervals from bootstrap distributions.

use crate::{bootstrap::BootstrapMethod, ConfidenceInterval};
use robust_core::{math::distributions::normal, Error, Result};
use tracing::{debug, instrument};

/// Percentile bootstrap method
///
/// The simplest bootstrap method. Uses the empirical percentiles of the
/// bootstrap distribution to construct the confidence interval.
#[derive(Debug, Clone, Copy, Default)]
pub struct PercentileBootstrap;

impl BootstrapMethod for PercentileBootstrap {
    fn calculate_interval(
        &self,
        bootstrap_estimates: &[f64],
        original_estimate: f64,
        confidence_level: f64,
    ) -> Result<ConfidenceInterval> {
        if bootstrap_estimates.is_empty() {
            return Err(Error::InvalidInput("No bootstrap estimates".to_string()));
        }

        let mut sorted = bootstrap_estimates.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = 1.0 - confidence_level;
        let lower_idx = ((alpha / 2.0) * sorted.len() as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * sorted.len() as f64) as usize;

        let lower_idx = lower_idx.min(sorted.len() - 1);
        let upper_idx = upper_idx.min(sorted.len() - 1);

        Ok(ConfidenceInterval::new(
            sorted[lower_idx],
            sorted[upper_idx],
            original_estimate,
            confidence_level,
        ))
    }

    fn name(&self) -> &'static str {
        "Percentile Bootstrap"
    }
}

/// BCa (bias-corrected and accelerated) bootstrap method
///
/// An improved bootstrap method that corrects for bias and skewness
/// in the bootstrap distribution.
#[derive(Debug, Clone, Copy, Default)]
pub struct BCaBootstrap;

impl BootstrapMethod for BCaBootstrap {
    #[instrument(skip(bootstrap_estimates), fields(n_estimates = bootstrap_estimates.len(), confidence_level))]
    fn calculate_interval(
        &self,
        bootstrap_estimates: &[f64],
        original_estimate: f64,
        confidence_level: f64,
    ) -> Result<ConfidenceInterval> {
        if bootstrap_estimates.is_empty() {
            return Err(Error::InvalidInput("No bootstrap estimates".to_string()));
        }

        debug!("BCa bootstrap: calculating interval from {} estimates", bootstrap_estimates.len());

        // Calculate bias correction factor
        let z0 = {
            let count_less = bootstrap_estimates
                .iter()
                .filter(|&&x| x < original_estimate)
                .count() as f64;
            let proportion = count_less / bootstrap_estimates.len() as f64;
            
            debug!("Bias correction: {} estimates < original (proportion: {:.4})", 
                   count_less, proportion);

            // Use normal quantile function
            if proportion <= 0.0 || proportion >= 1.0 {
                0.0
            } else {
                normal::quantile(proportion)
            }
        };

        // For now, use simplified BCa without acceleration
        // Full BCa would require jackknife estimates
        let a = 0.0; // Acceleration factor (simplified)
        
        debug!("BCa parameters: z0={:.4}, a={:.4}", z0, a);

        let alpha = 1.0 - confidence_level;
        let z_alpha_2 = normal::quantile(alpha / 2.0);
        let z_1_alpha_2 = normal::quantile(1.0 - alpha / 2.0);

        // Adjusted percentiles
        let alpha1 = normal::cdf(z0 + (z0 + z_alpha_2) / (1.0 - a * (z0 + z_alpha_2)));
        let alpha2 = normal::cdf(z0 + (z0 + z_1_alpha_2) / (1.0 - a * (z0 + z_1_alpha_2)));
        
        debug!("Adjusted percentiles: α1={:.4}, α2={:.4}", alpha1, alpha2);

        let mut sorted = bootstrap_estimates.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let lower_idx = (alpha1 * sorted.len() as f64) as usize;
        let upper_idx = (alpha2 * sorted.len() as f64) as usize;

        let lower_idx = lower_idx.min(sorted.len() - 1);
        let upper_idx = upper_idx.min(sorted.len() - 1);
        
        debug!("BCa indices: lower={}, upper={}", lower_idx, upper_idx);

        Ok(ConfidenceInterval::new(
            sorted[lower_idx],
            sorted[upper_idx],
            original_estimate,
            confidence_level,
        ))
    }

    fn name(&self) -> &'static str {
        "BCa Bootstrap"
    }
}

/// Basic bootstrap method
///
/// Uses the reflection principle: if θ̂* is the bootstrap estimate and θ̂ is the
/// original estimate, then the interval is [2θ̂ - q_{1-α/2}, 2θ̂ - q_{α/2}]
#[derive(Debug, Clone, Copy, Default)]
pub struct BasicBootstrap;

impl BootstrapMethod for BasicBootstrap {
    fn calculate_interval(
        &self,
        bootstrap_estimates: &[f64],
        original_estimate: f64,
        confidence_level: f64,
    ) -> Result<ConfidenceInterval> {
        if bootstrap_estimates.is_empty() {
            return Err(Error::InvalidInput("No bootstrap estimates".to_string()));
        }

        let mut sorted = bootstrap_estimates.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = 1.0 - confidence_level;
        let lower_idx = ((alpha / 2.0) * sorted.len() as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * sorted.len() as f64) as usize;

        let lower_idx = lower_idx.min(sorted.len() - 1);
        let upper_idx = upper_idx.min(sorted.len() - 1);

        // Basic method: 2*theta - quantile
        let lower = 2.0 * original_estimate - sorted[upper_idx];
        let upper = 2.0 * original_estimate - sorted[lower_idx];

        Ok(ConfidenceInterval::new(
            lower,
            upper,
            original_estimate,
            confidence_level,
        ))
    }

    fn name(&self) -> &'static str {
        "Basic Bootstrap"
    }
}

/// Student-t bootstrap method
///
/// Uses the t-distribution to account for variance estimation uncertainty.
/// This method is particularly useful for small samples.
#[derive(Debug, Clone, Copy, Default)]
pub struct StudentBootstrap;

impl BootstrapMethod for StudentBootstrap {
    fn calculate_interval(
        &self,
        bootstrap_estimates: &[f64],
        original_estimate: f64,
        confidence_level: f64,
    ) -> Result<ConfidenceInterval> {
        if bootstrap_estimates.is_empty() {
            return Err(Error::InvalidInput("No bootstrap estimates".to_string()));
        }

        // Calculate standard error from bootstrap distribution
        let mean = bootstrap_estimates.iter().sum::<f64>() / bootstrap_estimates.len() as f64;
        let variance = bootstrap_estimates
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (bootstrap_estimates.len() - 1) as f64;
        let se = variance.sqrt();

        // Calculate t-statistics
        let mut t_stats: Vec<f64> = bootstrap_estimates
            .iter()
            .map(|&x| (x - original_estimate) / se)
            .collect();
        
        t_stats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = 1.0 - confidence_level;
        let lower_idx = ((alpha / 2.0) * t_stats.len() as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * t_stats.len() as f64) as usize;

        let lower_idx = lower_idx.min(t_stats.len() - 1);
        let upper_idx = upper_idx.min(t_stats.len() - 1);

        // Convert back from t-statistics
        let lower = original_estimate - t_stats[upper_idx] * se;
        let upper = original_estimate - t_stats[lower_idx] * se;

        Ok(ConfidenceInterval::new(
            lower,
            upper,
            original_estimate,
            confidence_level,
        ))
    }

    fn name(&self) -> &'static str {
        "Student-t Bootstrap"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_percentile_bootstrap() {
        let bootstrap_estimates = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let original_estimate = 5.5;
        
        let method = PercentileBootstrap;
        let ci = method.calculate_interval(&bootstrap_estimates, original_estimate, 0.90).unwrap();
        
        // For 90% CI with 10 samples, should use indices 0 and 9
        assert_eq!(ci.lower, 1.0);
        assert_eq!(ci.upper, 10.0);
        assert_eq!(ci.estimate, 5.5);
    }

    #[test]
    fn test_basic_bootstrap() {
        let bootstrap_estimates = vec![4.0, 5.0, 6.0];
        let original_estimate = 5.0;
        
        let method = BasicBootstrap;
        let ci = method.calculate_interval(&bootstrap_estimates, original_estimate, 0.95).unwrap();
        
        // Basic method: 2*5.0 - 6.0 = 4.0 (lower), 2*5.0 - 4.0 = 6.0 (upper)
        assert_eq!(ci.lower, 4.0);
        assert_eq!(ci.upper, 6.0);
    }

    #[test]
    fn test_bca_bootstrap_no_bias() {
        // When bootstrap distribution is centered on original estimate, z0 should be ~0
        let bootstrap_estimates: Vec<f64> = (-50..=50).map(|i| 5.0 + i as f64 * 0.1).collect();
        let original_estimate = 5.0;
        
        let method = BCaBootstrap;
        let ci = method.calculate_interval(&bootstrap_estimates, original_estimate, 0.95).unwrap();
        
        // Should be roughly symmetric
        assert_relative_eq!(original_estimate - ci.lower, ci.upper - original_estimate, epsilon = 0.5);
    }

    #[test]
    fn test_student_bootstrap() {
        let bootstrap_estimates = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let original_estimate = 5.5;
        
        let method = StudentBootstrap;
        let ci = method.calculate_interval(&bootstrap_estimates, original_estimate, 0.95).unwrap();
        
        // Should produce a valid interval
        assert!(ci.lower < original_estimate);
        assert!(ci.upper > original_estimate);
        assert_eq!(ci.estimate, original_estimate);
    }
}