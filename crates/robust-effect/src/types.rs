//! Types for effect size representation

use std::fmt;

/// Types of effect sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectSizeType {
    /// Standardized mean difference (Cohen's d family)
    StandardizedMeanDifference,
    /// Non-parametric dominance measure (Cliff's delta)
    Dominance,
    /// Probability-based measure (CLES)
    Probability,
    /// Correlation-based measure
    Correlation,
    /// Proportion of variance explained
    VarianceExplained,
    /// Custom effect size type
    Custom(&'static str),
}

impl EffectSizeType {
    /// Get the name of the effect size type
    pub fn name(&self) -> &'static str {
        match self {
            Self::StandardizedMeanDifference => "Standardized Mean Difference",
            Self::Dominance => "Dominance",
            Self::Probability => "Probability",
            Self::Correlation => "Correlation",
            Self::VarianceExplained => "Variance Explained",
            Self::Custom(name) => name,
        }
    }

    /// Get the typical range for this effect size type
    pub fn typical_range(&self) -> (f64, f64) {
        match self {
            Self::StandardizedMeanDifference => (f64::NEG_INFINITY, f64::INFINITY),
            Self::Dominance => (-1.0, 1.0),
            Self::Probability => (0.0, 1.0),
            Self::Correlation => (-1.0, 1.0),
            Self::VarianceExplained => (0.0, 1.0),
            Self::Custom(_) => (f64::NEG_INFINITY, f64::INFINITY),
        }
    }
}

/// An effect size measurement with magnitude and interpretation
#[derive(Debug, Clone, PartialEq)]
pub struct EffectSize {
    /// The effect size magnitude
    pub magnitude: f64,
    /// The type of effect size
    pub effect_type: EffectSizeType,
    /// Interpretation of the magnitude
    pub interpretation: EffectSizeInterpretation,
    /// Sample sizes (group1, group2)
    pub sample_sizes: Option<(usize, usize)>,
}

impl EffectSize {
    /// Create a new effect size
    pub fn new(
        magnitude: f64,
        effect_type: EffectSizeType,
        sample_sizes: Option<(usize, usize)>,
    ) -> Self {
        let interpretation = EffectSizeInterpretation::from_magnitude(magnitude, effect_type);

        Self {
            magnitude,
            effect_type,
            interpretation,
            sample_sizes,
        }
    }

    /// Get the absolute magnitude
    pub fn abs_magnitude(&self) -> f64 {
        self.magnitude.abs()
    }

    /// Check if the effect size is practically significant
    pub fn is_practically_significant(&self) -> bool {
        matches!(
            self.interpretation,
            EffectSizeInterpretation::Medium | EffectSizeInterpretation::Large
        )
    }

    /// Check if the effect is in favor of group 2 (positive) or group 1 (negative)
    pub fn favors_group2(&self) -> bool {
        self.magnitude > 0.0
    }
}

impl fmt::Display for EffectSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.3} ({})",
            self.effect_type.name(),
            self.magnitude,
            self.interpretation
        )
    }
}

/// Interpretation of effect size magnitude following Cohen's conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectSizeInterpretation {
    /// Negligible effect (very small)
    Negligible,
    /// Small effect
    Small,
    /// Medium effect
    Medium,
    /// Large effect
    Large,
}

impl EffectSizeInterpretation {
    /// Get interpretation from magnitude based on effect size type
    pub fn from_magnitude(magnitude: f64, effect_type: EffectSizeType) -> Self {
        let abs_magnitude = magnitude.abs();

        match effect_type {
            EffectSizeType::StandardizedMeanDifference => {
                // Cohen's conventions for d
                if abs_magnitude < 0.2 {
                    Self::Negligible
                } else if abs_magnitude < 0.5 {
                    Self::Small
                } else if abs_magnitude < 0.8 {
                    Self::Medium
                } else {
                    Self::Large
                }
            }
            EffectSizeType::Correlation => {
                // Cohen's conventions for correlations
                if abs_magnitude < 0.1 {
                    Self::Negligible
                } else if abs_magnitude < 0.3 {
                    Self::Small
                } else if abs_magnitude < 0.5 {
                    Self::Medium
                } else {
                    Self::Large
                }
            }
            EffectSizeType::Dominance => {
                // Cliff's delta interpretations
                if abs_magnitude < 0.147 {
                    Self::Negligible
                } else if abs_magnitude < 0.33 {
                    Self::Small
                } else if abs_magnitude < 0.474 {
                    Self::Medium
                } else {
                    Self::Large
                }
            }
            EffectSizeType::VarianceExplained => {
                // Eta-squared, omega-squared interpretations
                if abs_magnitude < 0.01 {
                    Self::Negligible
                } else if abs_magnitude < 0.06 {
                    Self::Small
                } else if abs_magnitude < 0.14 {
                    Self::Medium
                } else {
                    Self::Large
                }
            }
            EffectSizeType::Probability => {
                // CLES interpretations (distance from 0.5)
                let distance_from_chance = (abs_magnitude - 0.5).abs();
                if distance_from_chance < 0.06 {
                    Self::Negligible
                } else if distance_from_chance < 0.14 {
                    Self::Small
                } else if distance_from_chance < 0.21 {
                    Self::Medium
                } else {
                    Self::Large
                }
            }
            EffectSizeType::Custom(_) => {
                // Default to Cohen's d conventions
                if abs_magnitude < 0.2 {
                    Self::Negligible
                } else if abs_magnitude < 0.5 {
                    Self::Small
                } else if abs_magnitude < 0.8 {
                    Self::Medium
                } else {
                    Self::Large
                }
            }
        }
    }
}

impl fmt::Display for EffectSizeInterpretation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Negligible => "negligible",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
        };
        write!(f, "{}", s)
    }
}

/// Categories of effect size magnitude for quick reference
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EffectSizeMagnitude {
    /// Effect size is negligible or very small
    Negligible,
    /// Effect size indicates a small practical difference
    Small,
    /// Effect size indicates a medium practical difference
    Medium,
    /// Effect size indicates a large practical difference
    Large,
}

impl From<EffectSizeInterpretation> for EffectSizeMagnitude {
    fn from(interpretation: EffectSizeInterpretation) -> Self {
        match interpretation {
            EffectSizeInterpretation::Negligible => Self::Negligible,
            EffectSizeInterpretation::Small => Self::Small,
            EffectSizeInterpretation::Medium => Self::Medium,
            EffectSizeInterpretation::Large => Self::Large,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_size_interpretation() {
        // Test Cohen's d interpretations
        assert_eq!(
            EffectSizeInterpretation::from_magnitude(
                0.1,
                EffectSizeType::StandardizedMeanDifference
            ),
            EffectSizeInterpretation::Negligible
        );
        assert_eq!(
            EffectSizeInterpretation::from_magnitude(
                0.3,
                EffectSizeType::StandardizedMeanDifference
            ),
            EffectSizeInterpretation::Small
        );
        assert_eq!(
            EffectSizeInterpretation::from_magnitude(
                0.6,
                EffectSizeType::StandardizedMeanDifference
            ),
            EffectSizeInterpretation::Medium
        );
        assert_eq!(
            EffectSizeInterpretation::from_magnitude(
                1.0,
                EffectSizeType::StandardizedMeanDifference
            ),
            EffectSizeInterpretation::Large
        );
    }

    #[test]
    fn test_cliff_delta_interpretation() {
        assert_eq!(
            EffectSizeInterpretation::from_magnitude(0.1, EffectSizeType::Dominance),
            EffectSizeInterpretation::Negligible
        );
        assert_eq!(
            EffectSizeInterpretation::from_magnitude(0.2, EffectSizeType::Dominance),
            EffectSizeInterpretation::Small
        );
        assert_eq!(
            EffectSizeInterpretation::from_magnitude(0.4, EffectSizeType::Dominance),
            EffectSizeInterpretation::Medium
        );
        assert_eq!(
            EffectSizeInterpretation::from_magnitude(0.6, EffectSizeType::Dominance),
            EffectSizeInterpretation::Large
        );
    }

    #[test]
    fn test_effect_size_display() {
        let effect_size = EffectSize::new(
            0.6,
            EffectSizeType::StandardizedMeanDifference,
            Some((10, 10)),
        );

        let display = format!("{}", effect_size);
        assert!(display.contains("Standardized Mean Difference"));
        assert!(display.contains("0.600"));
        assert!(display.contains("medium"));
    }
}
