//! Common types for confidence intervals

use robust_core::Numeric;
use std::fmt;

/// A confidence interval with lower and upper bounds
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceInterval<T: Numeric = f64> {
    /// Lower bound of the interval
    pub lower: T,
    /// Upper bound of the interval
    pub upper: T,
    /// The point estimate (center of interval)
    pub estimate: T,
    /// Confidence level (e.g., 0.95 for 95% CI)
    pub confidence_level: f64,
}

impl<T: Numeric> ConfidenceInterval<T> {
    /// Create a new confidence interval
    pub fn new(lower: T, upper: T, estimate: T, confidence_level: f64) -> Self {
        Self {
            lower,
            upper,
            estimate,
            confidence_level,
        }
    }

    /// Width of the confidence interval
    pub fn width(&self) -> T {
        self.upper - self.lower
    }

    /// Margin of error (half-width)
    pub fn margin_of_error(&self) -> T::Float 
    where 
        T::Float: num_traits::cast::FromPrimitive,
    {
        let width = (self.upper - self.lower).to_float();
        use num_traits::cast::FromPrimitive;
        let two = <T::Float as FromPrimitive>::from_f64(2.0).unwrap();
        width / two
    }

    /// Check if a value is contained in the interval
    pub fn contains(&self, value: T) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Check if intervals overlap
    pub fn overlaps(&self, other: &ConfidenceInterval<T>) -> bool {
        self.lower <= other.upper && other.lower <= self.upper
    }
}

impl<T: Numeric + fmt::Display> fmt::Display for ConfidenceInterval<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.1}% CI: [{}, {}], estimate: {}",
            self.confidence_level * 100.0,
            self.lower,
            self.upper,
            self.estimate
        )
    }
}

/// Confidence level type with validation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceLevel(f64);

impl ConfidenceLevel {
    /// Create a new confidence level
    ///
    /// # Panics
    /// Panics if level is not in (0, 1)
    pub fn new(level: f64) -> Self {
        assert!(
            level > 0.0 && level < 1.0,
            "Confidence level must be in (0, 1)"
        );
        Self(level)
    }

    /// Get the confidence level value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Get the alpha level (1 - confidence level)
    pub fn alpha(&self) -> f64 {
        1.0 - self.0
    }

    /// Get the tail probability (alpha/2 for two-tailed)
    pub fn tail_probability(&self) -> f64 {
        self.alpha() / 2.0
    }

    /// Common confidence levels
    pub const NINETY: Self = Self(0.90);
    pub const NINETY_FIVE: Self = Self(0.95);
    pub const NINETY_NINE: Self = Self(0.99);
}

impl From<f64> for ConfidenceLevel {
    fn from(level: f64) -> Self {
        Self::new(level)
    }
}

impl fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}%", self.0 * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_interval() {
        let ci = ConfidenceInterval::new(2.0, 8.0, 5.0, 0.95);

        assert_eq!(ci.width(), 6.0);
        assert_eq!(ci.margin_of_error(), 3.0);
        assert!(ci.contains(5.0));
        assert!(!ci.contains(1.0));
        assert!(!ci.contains(9.0));
    }

    #[test]
    fn test_overlap() {
        let ci1 = ConfidenceInterval::new(2.0, 6.0, 4.0, 0.95);
        let ci2 = ConfidenceInterval::new(4.0, 8.0, 6.0, 0.95);
        let ci3 = ConfidenceInterval::new(7.0, 9.0, 8.0, 0.95);

        assert!(ci1.overlaps(&ci2));
        assert!(ci2.overlaps(&ci1));
        assert!(!ci1.overlaps(&ci3));
    }
    
    #[test]
    fn test_confidence_interval_generic() {
        // Test with i32
        let ci_i32 = ConfidenceInterval::new(2i32, 8i32, 5i32, 0.95);
        assert_eq!(ci_i32.width(), 6);
        assert!(ci_i32.contains(5));
        assert!(!ci_i32.contains(1));
        
        // Test with f32
        let ci_f32 = ConfidenceInterval::new(2.0f32, 8.0f32, 5.0f32, 0.95);
        assert_eq!(ci_f32.width(), 6.0f32);
        assert_eq!(ci_f32.margin_of_error(), 3.0f32);
    }

    #[test]
    fn test_confidence_level() {
        let level = ConfidenceLevel::new(0.95);
        assert_eq!(level.value(), 0.95);
        assert!((level.alpha() - 0.05).abs() < 1e-10);
        assert!((level.tail_probability() - 0.025).abs() < 1e-10);
    }

    #[test]
    #[should_panic]
    fn test_invalid_confidence_level() {
        ConfidenceLevel::new(1.5);
    }
    
    #[test]
    fn test_confidence_interval_display() {
        let ci = ConfidenceInterval::new(2.5, 7.5, 5.0, 0.95);
        let display = format!("{}", ci);
        assert!(display.contains("95.0%"));
        assert!(display.contains("2.5000"));
        assert!(display.contains("7.5000"));
        assert!(display.contains("5.0000"));
    }
    
    #[test]
    fn test_confidence_level_display() {
        let level = ConfidenceLevel::new(0.95);
        assert_eq!(format!("{}", level), "95.0%");
        
        let level = ConfidenceLevel::new(0.99);
        assert_eq!(format!("{}", level), "99.0%");
    }
}
