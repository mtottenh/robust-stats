//! Variance explained effect size measures (eta-squared, omega-squared)

use crate::{EffectSize, EffectSizeEstimator, EffectSizeType, NonParametricEffectSize};
use robust_core::{Result, Numeric};
use num_traits::{Zero, One};

/// Eta-squared (η²) effect size estimator
///
/// Eta-squared represents the proportion of total variance that is explained
/// by group membership. It ranges from 0 to 1, where 0 indicates no effect
/// and 1 indicates that group membership explains all variance.
///
/// η² = SS_between / SS_total
#[derive(Debug, Clone, Copy)]
pub struct EtaSquared;

impl EtaSquared {
    /// Create a new eta-squared estimator
    pub fn new() -> Self {
        Self
    }
}

impl Default for EtaSquared {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Numeric> NonParametricEffectSize<T> for EtaSquared {
    fn compute(&self, group1: &[T], group2: &[T]) -> Result<EffectSize> {
        if group1.is_empty() || group2.is_empty() {
            return Err(robust_core::Error::InvalidInput(
                "Both groups must be non-empty".to_string(),
            ));
        }

        let values1 = group1;
        let values2 = group2;

        let n1 = values1.len();
        let n2 = values2.len();
        let n_total = n1 + n2;

        use num_traits::{NumCast, Float};
        
        // Calculate group means
        let sum1: T::Float = values1.iter().fold(T::Float::zero(), |acc, &v| acc + v.to_float());
        let sum2: T::Float = values2.iter().fold(T::Float::zero(), |acc, &v| acc + v.to_float());
        let n1_f: T::Float = NumCast::from(n1).unwrap();
        let n2_f: T::Float = NumCast::from(n2).unwrap();
        let n_total_f: T::Float = NumCast::from(n_total).unwrap();
        
        let mean1 = sum1 / n1_f;
        let mean2 = sum2 / n2_f;

        // Calculate grand mean
        let grand_mean = (sum1 + sum2) / n_total_f;

        // Calculate sum of squares between groups
        let ss_between = n1_f * (mean1 - grand_mean).powi(2) + n2_f * (mean2 - grand_mean).powi(2);

        // Calculate sum of squares within groups
        let ss_within1 = values1.iter().fold(T::Float::zero(), |acc, &x| {
            let x_f: T::Float = x.to_float();
            acc + (x_f - mean1).powi(2)
        });
        let ss_within2 = values2.iter().fold(T::Float::zero(), |acc, &x| {
            let x_f: T::Float = x.to_float();
            acc + (x_f - mean2).powi(2)
        });
        let ss_within = ss_within1 + ss_within2;

        // Calculate total sum of squares
        let ss_total = ss_between + ss_within;

        if ss_total <= T::Float::zero() {
            return Err(robust_core::Error::Computation(
                "Total sum of squares is non-positive".to_string(),
            ));
        }

        // Calculate eta-squared
        let eta_squared = ss_between / ss_total;
        let eta_squared_f64: f64 = NumCast::from(eta_squared).unwrap();

        Ok(EffectSize::new(
            eta_squared_f64,
            EffectSizeType::VarianceExplained,
            Some((n1, n2)),
        ))
    }

    fn compute_sorted(&self, sorted_group1: &[T], sorted_group2: &[T]) -> Result<EffectSize> {
        // Variance measures don't benefit from sorted data
        self.compute(sorted_group1, sorted_group2)
    }
}

impl<T: Numeric> EffectSizeEstimator<T> for EtaSquared {
    fn effect_size_type(&self) -> EffectSizeType {
        EffectSizeType::VarianceExplained
    }

    fn supports_weighted_samples(&self) -> bool {
        false
    }
}

/// Omega-squared (ω²) effect size estimator
///
/// Omega-squared is a less biased estimator of effect size compared to eta-squared.
/// It provides a better estimate of the population effect size by accounting for
/// the bias in eta-squared, especially with small samples.
///
/// ω² = (SS_between - (k-1) * MS_within) / (SS_total + MS_within)
/// where k is the number of groups (2 in our case)
#[derive(Debug, Clone, Copy)]
pub struct OmegaSquared;

impl OmegaSquared {
    /// Create a new omega-squared estimator
    pub fn new() -> Self {
        Self
    }
}

impl Default for OmegaSquared {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Numeric> NonParametricEffectSize<T> for OmegaSquared {
    fn compute(&self, group1: &[T], group2: &[T]) -> Result<EffectSize> {
        if group1.is_empty() || group2.is_empty() {
            return Err(robust_core::Error::InvalidInput(
                "Both groups must be non-empty".to_string(),
            ));
        }

        let values1 = group1;
        let values2 = group2;

        let n1 = values1.len();
        let n2 = values2.len();
        let n_total = n1 + n2;

        if n_total < 4 {
            return Err(robust_core::Error::InvalidInput(
                "Need at least 4 total observations for omega-squared".to_string(),
            ));
        }

        use num_traits::{NumCast, Float};
        
        // Calculate group means
        let sum1: T::Float = values1.iter().fold(T::Float::zero(), |acc, &v| acc + v.to_float());
        let sum2: T::Float = values2.iter().fold(T::Float::zero(), |acc, &v| acc + v.to_float());
        let n1_f: T::Float = NumCast::from(n1).unwrap();
        let n2_f: T::Float = NumCast::from(n2).unwrap();
        let n_total_f: T::Float = NumCast::from(n_total).unwrap();
        
        let mean1 = sum1 / n1_f;
        let mean2 = sum2 / n2_f;

        // Calculate grand mean
        let grand_mean = (sum1 + sum2) / n_total_f;

        // Calculate sum of squares between groups
        let ss_between = n1_f * (mean1 - grand_mean).powi(2) + n2_f * (mean2 - grand_mean).powi(2);

        // Calculate sum of squares within groups
        let ss_within1 = values1.iter().fold(T::Float::zero(), |acc, &x| {
            let x_f: T::Float = x.to_float();
            acc + (x_f - mean1).powi(2)
        });
        let ss_within2 = values2.iter().fold(T::Float::zero(), |acc, &x| {
            let x_f: T::Float = x.to_float();
            acc + (x_f - mean2).powi(2)
        });
        let ss_within = ss_within1 + ss_within2;

        // Calculate mean square within
        let df_within = n_total - 2; // degrees of freedom within groups
        let df_within_f: T::Float = NumCast::from(df_within).unwrap();
        let ms_within = ss_within / df_within_f;

        // Calculate total sum of squares
        let ss_total = ss_between + ss_within;

        // Calculate omega-squared with bias correction
        let k: T::Float = NumCast::from(2.0).unwrap(); // number of groups
        let one: T::Float = T::Float::one();
        let numerator = ss_between - (k - one) * ms_within;
        let denominator = ss_total + ms_within;

        if denominator <= T::Float::zero() {
            return Err(robust_core::Error::Computation(
                "Denominator is non-positive in omega-squared calculation".to_string(),
            ));
        }

        // Omega-squared can be negative (indicating no effect), but we bound it at 0
        let omega_squared = (numerator / denominator).max(T::Float::zero());
        let omega_squared_f64: f64 = NumCast::from(omega_squared).unwrap();

        Ok(EffectSize::new(
            omega_squared_f64,
            EffectSizeType::VarianceExplained,
            Some((n1, n2)),
        ))
    }

    fn compute_sorted(&self, sorted_group1: &[T], sorted_group2: &[T]) -> Result<EffectSize> {
        // Variance measures don't benefit from sorted data
        self.compute(sorted_group1, sorted_group2)
    }
}

impl<T: Numeric> EffectSizeEstimator<T> for OmegaSquared {
    fn effect_size_type(&self) -> EffectSizeType {
        EffectSizeType::VarianceExplained
    }

    fn supports_weighted_samples(&self) -> bool {
        false
    }
}

/// Convert between eta-squared and Cohen's d (approximate relationship)
pub fn eta_squared_to_cohen_d(eta_squared: f64) -> f64 {
    if eta_squared >= 1.0 {
        return f64::INFINITY;
    }
    if eta_squared <= 0.0 {
        return 0.0;
    }

    // Approximate relationship: d ≈ 2 * sqrt(η² / (1 - η²))
    2.0 * (eta_squared / (1.0 - eta_squared)).sqrt()
}

/// Convert between Cohen's d and eta-squared (approximate relationship)
pub fn cohen_d_to_eta_squared(cohen_d: f64) -> f64 {
    let d_squared = cohen_d * cohen_d;
    // Approximate relationship: η² ≈ d² / (d² + 4)
    d_squared / (d_squared + 4.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_eta_squared_no_difference() {
        // Identical groups
        let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let eta_sq = EtaSquared::new();
        let effect_size = eta_sq.compute(&group1, &group2).unwrap();

        // Should be zero (no variance explained by groups)
        assert_abs_diff_eq!(effect_size.magnitude, 0.0, epsilon = 1e-10);
        assert_eq!(effect_size.effect_type, EffectSizeType::VarianceExplained);
    }

    #[test]
    fn test_eta_squared_clear_difference() {
        // Groups with clear separation
        let group1 = vec![1.0, 1.0, 1.0]; // no within-group variance
        let group2 = vec![5.0, 5.0, 5.0]; // no within-group variance

        let eta_sq = EtaSquared::new();
        let effect_size = eta_sq.compute(&group1, &group2).unwrap();

        // Should be 1.0 (all variance explained by groups)
        assert_abs_diff_eq!(effect_size.magnitude, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_eta_squared_partial_variance() {
        let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // mean = 3
        let group2 = vec![6.0, 7.0, 8.0, 9.0, 10.0]; // mean = 8

        let eta_sq = EtaSquared::new();
        let effect_size = eta_sq.compute(&group1, &group2).unwrap();

        // Should be substantial (groups are well separated)
        assert!(effect_size.magnitude > 0.5);
        assert!(effect_size.magnitude < 1.0);
        assert_eq!(effect_size.sample_sizes, Some((5, 5)));
    }

    #[test]
    fn test_omega_squared_vs_eta_squared() {
        let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];

        let eta_sq = EtaSquared::new();
        let omega_sq = OmegaSquared::new();

        let eta_effect = eta_sq.compute(&group1, &group2).unwrap();
        let omega_effect = omega_sq.compute(&group1, &group2).unwrap();

        // Omega-squared should be smaller (less biased)
        assert!(omega_effect.magnitude <= eta_effect.magnitude);

        // Both should be positive
        assert!(omega_effect.magnitude > 0.0);
        assert!(eta_effect.magnitude > 0.0);
    }

    #[test]
    fn test_omega_squared_small_sample() {
        // Very small sample
        let group1 = vec![1.0, 2.0];
        let group2 = vec![3.0, 4.0];

        let omega_sq = OmegaSquared::new();
        let effect_size = omega_sq.compute(&group1, &group2).unwrap();

        // Should work with minimum sample size
        assert!(effect_size.magnitude >= 0.0);
        assert!(effect_size.magnitude <= 1.0);
    }

    #[test]
    fn test_omega_squared_too_small_sample() {
        // Too small sample
        let group1 = vec![1.0];
        let group2 = vec![2.0];

        let omega_sq = OmegaSquared::new();

        // Should error with insufficient data
        assert!(omega_sq.compute(&group1, &group2).is_err());
    }

    #[test]
    fn test_cohen_d_eta_squared_conversion() {
        // Test conversion functions
        let cohen_d_values = vec![0.0, 0.2, 0.5, 0.8, 1.0, 2.0];

        for &d in &cohen_d_values {
            let eta_sq = cohen_d_to_eta_squared(d);
            let d_back = eta_squared_to_cohen_d(eta_sq);

            // Should be roughly invertible
            assert_abs_diff_eq!(d, d_back, epsilon = 0.01);

            // Eta-squared should be in valid range
            assert!((0.0..=1.0).contains(&eta_sq));
        }
    }

    #[test]
    fn test_eta_squared_bounds() {
        let group1 = vec![1.0, 2.0, 3.0];
        let group2 = vec![4.0, 5.0, 6.0];

        let eta_sq = EtaSquared::new();
        let omega_sq = OmegaSquared::new();

        let eta_effect = eta_sq.compute(&group1, &group2).unwrap();
        let omega_effect = omega_sq.compute(&group1, &group2).unwrap();

        // Both should be between 0 and 1
        assert!(eta_effect.magnitude >= 0.0 && eta_effect.magnitude <= 1.0);
        assert!(omega_effect.magnitude >= 0.0 && omega_effect.magnitude <= 1.0);
    }

    #[test]
    fn test_empty_groups() {
        let empty_group = vec![];
        let group = vec![1.0, 2.0, 3.0];

        let eta_sq = EtaSquared::new();
        let omega_sq = OmegaSquared::new();

        assert!(eta_sq.compute(&empty_group, &group).is_err());
        assert!(omega_sq.compute(&empty_group, &group).is_err());
    }

    #[test]
    fn test_constant_values() {
        // All values are the same (no variance)
        let group1 = vec![5.0, 5.0, 5.0];
        let group2 = vec![5.0, 5.0, 5.0];

        let eta_sq = EtaSquared::new();

        // Should error due to zero total variance
        assert!(eta_sq.compute(&group1, &group2).is_err());
    }
}
