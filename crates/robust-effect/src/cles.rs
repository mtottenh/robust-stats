//! Common Language Effect Size (CLES)
//!
//! CLES represents the probability that a randomly selected score from one group
//! will be greater than a randomly selected score from another group. It provides
//! an intuitive interpretation of effect size in terms of probability.

use crate::{EffectSize, EffectSizeEstimator, EffectSizeType, NonParametricEffectSize};
use robust_core::{Result, Numeric};

/// Common Language Effect Size (CLES) estimator
///
/// CLES is calculated as the probability that a randomly selected observation
/// from group 2 will be greater than a randomly selected observation from group 1.
///
/// CLES = P(X₂ > X₁)
///
/// The result ranges from 0 to 1:
/// - CLES = 0.5: no difference between groups (50% chance)
/// - CLES > 0.5: group 2 tends to have higher values
/// - CLES < 0.5: group 1 tends to have higher values
/// - CLES = 1.0: all values in group 2 are greater than all values in group 1
/// - CLES = 0.0: all values in group 1 are greater than all values in group 2
#[derive(Debug, Clone, Copy)]
pub struct CommonLanguageEffectSize {
    /// Whether to use continuity correction for ties
    use_continuity_correction: bool,
}

impl CommonLanguageEffectSize {
    /// Create a new CLES estimator
    pub fn new() -> Self {
        Self {
            use_continuity_correction: true, // Default to using correction
        }
    }

    /// Disable continuity correction for ties
    pub fn without_continuity_correction(mut self) -> Self {
        self.use_continuity_correction = false;
        self
    }
}

impl Default for CommonLanguageEffectSize {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Numeric> NonParametricEffectSize<T> for CommonLanguageEffectSize {
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
        let total_pairs = n1 * n2;

        let mut wins = 0; // group2 > group1
        let mut ties = 0; // group1 == group2

        // Compare all pairs
        for &x1 in values1.iter() {
            for &x2 in values2.iter() {
                match x2.partial_cmp(&x1) {
                    Some(std::cmp::Ordering::Greater) => wins += 1,
                    Some(std::cmp::Ordering::Equal) => ties += 1,
                    Some(std::cmp::Ordering::Less) => {} // losses, don't count
                    None => {
                        return Err(robust_core::Error::Computation(
                            "Cannot compare values (NaN encountered)".to_string(),
                        ));
                    }
                }
            }
        }

        // Calculate CLES
        let cles = if self.use_continuity_correction {
            // Continuity correction: treat ties as 0.5 probability
            (wins as f64 + (ties as f64 * 0.5)) / total_pairs as f64
        } else {
            // Standard calculation: ignore ties
            wins as f64 / total_pairs as f64
        };

        Ok(EffectSize::new(
            cles,
            EffectSizeType::Probability,
            Some((n1, n2)),
        ))
    }

    fn compute_sorted(&self, sorted_group1: &[T], sorted_group2: &[T]) -> Result<EffectSize> {
        // CLES doesn't benefit from sorted data since we need all pairwise comparisons
        self.compute(sorted_group1, sorted_group2)
    }
}

impl<T: Numeric> EffectSizeEstimator<T> for CommonLanguageEffectSize {
    fn effect_size_type(&self) -> EffectSizeType {
        EffectSizeType::Probability
    }

    fn supports_weighted_samples(&self) -> bool {
        false // Not supported for pairwise comparisons
    }

    fn is_symmetric(&self) -> bool {
        false // CLES(A,B) = 1 - CLES(B,A), not negative
    }
}

impl CommonLanguageEffectSize {
    /// Convert CLES to Cliff's delta
    ///
    /// Relationship: δ = 2 × CLES - 1
    /// where δ is Cliff's delta and CLES is Common Language Effect Size
    pub fn to_cliff_delta(cles: f64) -> f64 {
        2.0 * cles - 1.0
    }

    /// Convert Cliff's delta to CLES
    ///
    /// Relationship: CLES = (δ + 1) / 2
    /// where δ is Cliff's delta and CLES is Common Language Effect Size
    pub fn from_cliff_delta(delta: f64) -> f64 {
        (delta + 1.0) / 2.0
    }

    /// Calculate the confidence interval for CLES using normal approximation
    pub fn confidence_interval(
        cles: f64,
        n1: usize,
        n2: usize,
        confidence_level: f64,
    ) -> Result<(f64, f64)> {
        use statrs::distribution::{ContinuousCDF, Normal};

        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(robust_core::Error::InvalidInput(
                "Confidence level must be in (0, 1)".to_string(),
            ));
        }

        let n1_f = n1 as f64;
        let n2_f = n2 as f64;

        // Variance of CLES using the relationship with Cliff's delta
        // Var(CLES) = Var(δ)/4 where Var(δ) = (n1+n2+1)/(3*n1*n2)
        let variance = (n1_f + n2_f + 1.0) / (12.0 * n1_f * n2_f);
        let standard_error = variance.sqrt();

        // Normal approximation (valid for large samples)
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            robust_core::Error::Computation(format!("Failed to create normal distribution: {}", e))
        })?;

        let alpha = 1.0 - confidence_level;
        let z_critical = normal.inverse_cdf(1.0 - alpha / 2.0);

        let margin = z_critical * standard_error;
        let lower = (cles - margin).max(0.0); // CLES is bounded by [0, 1]
        let upper = (cles + margin).min(1.0);

        Ok((lower, upper))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cles_no_overlap() {
        // Group 1: all smaller than group 2
        let group1 = vec![1.0, 2.0, 3.0];
        let group2 = vec![4.0, 5.0, 6.0];

        let cles = CommonLanguageEffectSize::new();
        let effect_size = cles.compute(&group1, &group2).unwrap();

        // All pairs have group2 > group1, so CLES = 1.0
        assert_abs_diff_eq!(effect_size.magnitude, 1.0, epsilon = 1e-10);
        assert_eq!(effect_size.effect_type, EffectSizeType::Probability);
    }

    #[test]
    fn test_cles_complete_overlap() {
        // Identical groups
        let group1 = vec![1.0, 2.0, 3.0];
        let group2 = vec![1.0, 2.0, 3.0];

        let cles = CommonLanguageEffectSize::new();
        let effect_size = cles.compute(&group1, &group2).unwrap();

        // All pairs are ties, with continuity correction CLES = 0.5
        assert_abs_diff_eq!(effect_size.magnitude, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_cles_partial_overlap() {
        let group1 = vec![1.0, 2.0, 3.0, 4.0];
        let group2 = vec![2.5, 3.5, 4.5, 5.5];

        let cles = CommonLanguageEffectSize::new();
        let effect_size = cles.compute(&group1, &group2).unwrap();

        // Count manually: how many times group2 > group1
        // 1.0 vs [2.5, 3.5, 4.5, 5.5]: 4 wins
        // 2.0 vs [2.5, 3.5, 4.5, 5.5]: 4 wins
        // 3.0 vs [2.5, 3.5, 4.5, 5.5]: 3 wins
        // 4.0 vs [2.5, 3.5, 4.5, 5.5]: 2 wins
        // Total: 13 wins out of 16 pairs = 0.8125
        assert_abs_diff_eq!(effect_size.magnitude, 0.8125, epsilon = 1e-10);
    }

    #[test]
    fn test_cles_with_ties() {
        let group1 = vec![1.0, 2.0, 3.0];
        let group2 = vec![2.0, 3.0, 4.0];

        let cles_corrected = CommonLanguageEffectSize::new();
        let cles_standard = CommonLanguageEffectSize::new().without_continuity_correction();

        let effect_corrected = cles_corrected.compute(&group1, &group2).unwrap();
        let effect_standard = cles_standard.compute(&group1, &group2).unwrap();

        // With ties, the corrected version should be different
        assert_ne!(effect_corrected.magnitude, effect_standard.magnitude);

        // Both should be > 0.5 (group2 tends to be larger)
        assert!(effect_corrected.magnitude > 0.5);
        assert!(effect_standard.magnitude > 0.5);

        // Corrected version should be higher (ties count as 0.5)
        assert!(effect_corrected.magnitude > effect_standard.magnitude);
    }

    #[test]
    fn test_cles_cliff_delta_relationship() {
        let group1 = vec![1.0, 2.0, 3.0];
        let group2 = vec![4.0, 5.0, 6.0];

        let cles_estimator = CommonLanguageEffectSize::new();
        let effect_size = cles_estimator.compute(&group1, &group2).unwrap();

        // Convert to Cliff's delta and back
        let cliff_delta = CommonLanguageEffectSize::to_cliff_delta(effect_size.magnitude);
        let cles_back = CommonLanguageEffectSize::from_cliff_delta(cliff_delta);

        assert_abs_diff_eq!(effect_size.magnitude, cles_back, epsilon = 1e-10);

        // For complete separation, CLES = 1.0 and Cliff's delta = 1.0
        assert_abs_diff_eq!(effect_size.magnitude, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cliff_delta, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cles_asymmetry() {
        let group1 = vec![1.0, 2.0, 3.0];
        let group2 = vec![4.0, 5.0, 6.0];

        let cles = CommonLanguageEffectSize::new();
        let cles_12 = cles.compute(&group1, &group2).unwrap();
        let cles_21 = cles.compute(&group2, &group1).unwrap();

        // CLES(A,B) + CLES(B,A) should equal 1.0 (assuming no ties)
        assert_abs_diff_eq!(cles_12.magnitude + cles_21.magnitude, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cles_confidence_interval() {
        let cles_value = 0.75;
        let n1 = 20;
        let n2 = 25;
        let confidence_level = 0.95;

        let (lower, upper) =
            CommonLanguageEffectSize::confidence_interval(cles_value, n1, n2, confidence_level)
                .unwrap();

        // CI should contain the estimate
        assert!(lower <= cles_value && cles_value <= upper);

        // CI should be within valid range
        assert!(lower >= 0.0 && upper <= 1.0);

        // CI should be reasonably wide for this sample size
        assert!(upper - lower > 0.05);
        assert!(upper - lower < 0.5);
    }

    #[test]
    fn test_empty_groups() {
        let empty_group = vec![];
        let group = vec![1.0, 2.0, 3.0];

        let cles = CommonLanguageEffectSize::new();

        assert!(cles.compute(&empty_group, &group).is_err());
        assert!(cles.compute(&group, &empty_group).is_err());
    }

    #[test]
    fn test_interpretation() {
        // Test different CLES values and their interpretations
        // Distance from 0.5: <0.06=negligible, <0.14=small, <0.21=medium, >=0.21=large
        let effect_small = EffectSize::new(0.6, EffectSizeType::Probability, None); // dist=0.1
        let effect_medium = EffectSize::new(0.65, EffectSizeType::Probability, None); // dist=0.15
        let effect_large = EffectSize::new(0.75, EffectSizeType::Probability, None); // dist=0.25

        // Test that interpretations follow expected pattern
        assert!(effect_small.interpretation.to_string().contains("small"));
        assert!(effect_medium.interpretation.to_string().contains("medium"));
        assert!(effect_large.interpretation.to_string().contains("large"));
    }
}
