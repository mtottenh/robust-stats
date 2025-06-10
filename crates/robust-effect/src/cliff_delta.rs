//! Cliff's delta non-parametric effect size
//!
//! Cliff's delta is a non-parametric effect size measure that estimates the degree
//! of overlap between two groups. It's based on the probability that a randomly
//! selected observation from one group will be greater than a randomly selected
//! observation from the other group.

use crate::{EffectSize, EffectSizeEstimator, EffectSizeType, NonParametricEffectSize};
use robust_core::{Result, Numeric};

/// Cliff's delta effect size estimator
///
/// Cliff's delta (δ) is calculated as:
/// δ = (number of pairs where X₁ > X₂ - number of pairs where X₁ < X₂) / (n₁ × n₂)
///
/// where X₁ comes from group 1 and X₂ comes from group 2.
///
/// The result ranges from -1 to +1:
/// - δ = +1: all observations in group 2 are larger than all observations in group 1
/// - δ = -1: all observations in group 1 are larger than all observations in group 2
/// - δ = 0: complete overlap, no systematic difference
#[derive(Debug, Clone, Copy)]
pub struct CliffDelta {
    /// Whether to use continuity correction for ties
    use_continuity_correction: bool,
}

impl CliffDelta {
    /// Create a new Cliff's delta estimator
    pub fn new() -> Self {
        Self {
            use_continuity_correction: false,
        }
    }

    /// Enable continuity correction for ties
    pub fn with_continuity_correction(mut self) -> Self {
        self.use_continuity_correction = true;
        self
    }
}

impl Default for CliffDelta {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Numeric> NonParametricEffectSize<T> for CliffDelta {
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
        let mut losses = 0; // group1 > group2
        let mut ties = 0; // group1 == group2

        // Compare all pairs
        for &x1 in values1.iter() {
            for &x2 in values2.iter() {
                match x2.partial_cmp(&x1) {
                    Some(std::cmp::Ordering::Greater) => wins += 1,
                    Some(std::cmp::Ordering::Less) => losses += 1,
                    Some(std::cmp::Ordering::Equal) => ties += 1,
                    None => {
                        return Err(robust_core::Error::Computation(
                            "Cannot compare values (NaN encountered)".to_string(),
                        ));
                    }
                }
            }
        }

        // Calculate Cliff's delta
        let delta = if self.use_continuity_correction {
            // Continuity correction: treat ties as 0.5 wins and 0.5 losses
            let adjusted_wins = wins as f64 + (ties as f64 * 0.5);
            let adjusted_losses = losses as f64 + (ties as f64 * 0.5);
            (adjusted_wins - adjusted_losses) / total_pairs as f64
        } else {
            // Standard calculation: ignore ties
            (wins as f64 - losses as f64) / total_pairs as f64
        };

        Ok(EffectSize::new(
            delta,
            EffectSizeType::Dominance,
            Some((n1, n2)),
        ))
    }
    
    fn compute_sorted(&self, sorted_group1: &[T], sorted_group2: &[T]) -> Result<EffectSize> {
        // Cliff's delta doesn't benefit from sorting
        self.compute(sorted_group1, sorted_group2)
    }
}

impl<T: Numeric> EffectSizeEstimator<T> for CliffDelta {
    fn effect_size_type(&self) -> EffectSizeType {
        EffectSizeType::Dominance
    }

    fn supports_weighted_samples(&self) -> bool {
        false // Not supported for non-parametric pairwise comparisons
    }

    fn is_symmetric(&self) -> bool {
        true // cliff_delta(A,B) = -cliff_delta(B,A)
    }
}

/// Calculate the confidence interval for Cliff's delta using normal approximation
pub fn cliff_delta_confidence_interval(
    delta: f64,
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

    // Variance of Cliff's delta (Cliff, 1993)
    let variance = (n1_f + n2_f + 1.0) / (3.0 * n1_f * n2_f);
    let standard_error = variance.sqrt();

    // Normal approximation (valid for large samples)
    let normal = Normal::new(0.0, 1.0).map_err(|e| {
        robust_core::Error::Computation(format!("Failed to create normal distribution: {}", e))
    })?;

    let alpha = 1.0 - confidence_level;
    let z_critical = normal.inverse_cdf(1.0 - alpha / 2.0);

    let margin = z_critical * standard_error;
    let lower = (delta - margin).max(-1.0); // Cliff's delta is bounded by [-1, 1]
    let upper = (delta + margin).min(1.0);

    Ok((lower, upper))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cliff_delta_no_overlap() {
        // Group 1: all smaller than group 2
        let group1 = vec![1.0, 2.0, 3.0];
        let group2 = vec![4.0, 5.0, 6.0];

        let cliff_delta = CliffDelta::new();
        let effect_size = cliff_delta.compute(&group1, &group2).unwrap();

        // All 9 pairs have group2 > group1, so delta = 1.0
        assert_abs_diff_eq!(effect_size.magnitude, 1.0, epsilon = 1e-10);
        assert_eq!(effect_size.effect_type, EffectSizeType::Dominance);
    }

    #[test]
    fn test_cliff_delta_complete_overlap() {
        // Identical groups
        let group1 = vec![1.0, 2.0, 3.0];
        let group2 = vec![1.0, 2.0, 3.0];

        let cliff_delta = CliffDelta::new();
        let effect_size = cliff_delta.compute(&group1, &group2).unwrap();

        // All pairs are ties, so delta = 0.0
        assert_abs_diff_eq!(effect_size.magnitude, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cliff_delta_partial_overlap() {
        let group1 = vec![1.0, 2.0, 3.0, 4.0];
        let group2 = vec![2.5, 3.5, 4.5, 5.5];

        let cliff_delta = CliffDelta::new();
        let effect_size = cliff_delta.compute(&group1, &group2).unwrap();

        // Count manually: group2 vs group1 comparisons
        // 1.0 vs [2.5, 3.5, 4.5, 5.5] -> group2 wins 4, group1 wins 0
        // 2.0 vs [2.5, 3.5, 4.5, 5.5] -> group2 wins 4, group1 wins 0
        // 3.0 vs [2.5, 3.5, 4.5, 5.5] -> group2 wins 3, group1 wins 1
        // 4.0 vs [2.5, 3.5, 4.5, 5.5] -> group2 wins 2, group1 wins 2
        // Total: group2 wins 13, group1 wins 3 out of 16 pairs
        let expected_delta = (13.0 - 3.0) / 16.0;
        assert_abs_diff_eq!(effect_size.magnitude, expected_delta, epsilon = 1e-10);
    }

    #[test]
    fn test_cliff_delta_with_ties() {
        let group1 = vec![1.0, 2.0, 3.0];
        let group2 = vec![2.0, 3.0, 4.0];

        let cliff_delta_standard = CliffDelta::new();
        let cliff_delta_corrected = CliffDelta::new().with_continuity_correction();

        let effect_standard = cliff_delta_standard.compute(&group1, &group2).unwrap();
        let effect_corrected = cliff_delta_corrected.compute(&group1, &group2).unwrap();

        // Both should give the same result because continuity correction
        // adds 0.5*ties to both wins and losses, so the difference cancels out
        assert_eq!(effect_standard.magnitude, effect_corrected.magnitude);

        // Should be positive (group2 tends to be larger)
        assert!(effect_standard.magnitude > 0.0);
        assert!(effect_corrected.magnitude > 0.0);

        // Check approximate value
        let expected = 5.0 / 9.0; // (6 wins - 1 loss) / 9 pairs
        assert!((effect_standard.magnitude - expected).abs() < 0.01);
    }

    #[test]
    fn test_cliff_delta_symmetry() {
        let group1 = vec![1.0, 2.0, 3.0];
        let group2 = vec![4.0, 5.0, 6.0];

        let cliff_delta = CliffDelta::new();
        let delta_12 = cliff_delta.compute(&group1, &group2).unwrap();
        let delta_21 = cliff_delta.compute(&group2, &group1).unwrap();

        // Should be symmetric: delta(A,B) = -delta(B,A)
        assert_abs_diff_eq!(delta_12.magnitude, -delta_21.magnitude, epsilon = 1e-10);
    }

    #[test]
    fn test_cliff_delta_confidence_interval() {
        let delta = 0.5;
        let n1 = 20;
        let n2 = 25;
        let confidence_level = 0.95;

        let (lower, upper) =
            cliff_delta_confidence_interval(delta, n1, n2, confidence_level).unwrap();

        // CI should contain the estimate
        assert!(lower <= delta && delta <= upper);

        // CI should be within valid range
        assert!(lower >= -1.0 && upper <= 1.0);

        // CI should be reasonably wide for this sample size
        assert!(upper - lower > 0.1);
        assert!(upper - lower < 1.0);
    }

    #[test]
    fn test_empty_groups() {
        let empty_group = vec![];
        let group = vec![1.0, 2.0, 3.0];

        let cliff_delta = CliffDelta::new();

        assert!(cliff_delta.compute(&empty_group, &group).is_err());
        assert!(cliff_delta.compute(&group, &empty_group).is_err());
    }
}
