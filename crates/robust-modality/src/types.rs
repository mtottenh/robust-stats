//! Core types for modality detection

use robust_histogram::Histogram;
use robust_core::Numeric;
use std::fmt;

/// A detected mode in the data distribution
#[derive(Debug, Clone, PartialEq)]
pub struct Mode<T: Numeric = f64> 
where
    T::Float: fmt::Debug + PartialEq + num_traits::ToPrimitive,
{
    /// The location (center) of the mode
    pub location: T::Float,
    /// The left boundary of the mode
    pub left_bound: T::Float,
    /// The right boundary of the mode
    pub right_bound: T::Float,
    /// The height (density) of the mode
    pub height: T::Float,
}

impl<T: Numeric> Mode<T> 
where
    T::Float: fmt::Debug + PartialEq + num_traits::ToPrimitive,
{
    /// Create a new mode
    pub fn new(location: T::Float, left_bound: T::Float, right_bound: T::Float, height: T::Float) -> Self {
        Self {
            location,
            left_bound,
            right_bound,
            height,
        }
    }

    /// Get the width of the mode
    pub fn width(&self) -> T::Float {
        self.right_bound - self.left_bound
    }

    /// Check if a value falls within this mode's bounds
    pub fn contains(&self, value: T::Float) -> bool {
        value >= self.left_bound && value <= self.right_bound
    }

    /// Get the relative position of a value within the mode (0.0 to 1.0)
    pub fn relative_position(&self, value: T::Float) -> T::Float {
        use num_traits::{Zero, One};
        if self.width() == T::Float::zero() {
            let half: T::Float = num_traits::NumCast::from(0.5).unwrap();
            return half;
        }
        let zero = T::Float::zero();
        let one = T::Float::one();
        let pos = (value - self.left_bound) / self.width();
        if pos <= zero { zero } else if pos >= one { one } else { pos }
    }
}

impl<T: Numeric> fmt::Display for Mode<T> 
where
    T::Float: fmt::Display + fmt::Debug + PartialEq + num_traits::ToPrimitive,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mode(location={:.3}, bounds=[{:.3}, {:.3}], height={:.3})",
            self.location, self.left_bound, self.right_bound, self.height
        )
    }
}

/// Result of modality detection containing detected modes and metadata
#[derive(Debug, Clone, PartialEq)]
pub struct ModalityResult<T: Numeric = f64> 
where
    T::Float: fmt::Debug + PartialEq + num_traits::ToPrimitive,
{
    /// The detected modes
    modes: Vec<Mode<T>>,
    /// The histogram used for detection
    histogram: Histogram<T>,
    /// Indices of lowland bins (areas of low density between modes)
    lowland_indices: Vec<usize>,
}

impl<T: Numeric> ModalityResult<T> 
where
    T::Float: fmt::Debug + PartialEq + num_traits::ToPrimitive,
{
    /// Create a new modality result
    pub fn new(modes: Vec<Mode<T>>, histogram: Histogram<T>, lowland_indices: Vec<usize>) -> Self {
        Self {
            modes,
            histogram,
            lowland_indices,
        }
    }

    /// Get the detected modes
    pub fn modes(&self) -> &[Mode<T>] {
        &self.modes
    }

    /// Get the number of detected modes
    pub fn mode_count(&self) -> usize {
        self.modes.len()
    }

    /// Check if the distribution is unimodal
    pub fn is_unimodal(&self) -> bool {
        self.modes.len() == 1
    }

    /// Check if the distribution is multimodal
    pub fn is_multimodal(&self) -> bool {
        self.modes.len() > 1
    }

    /// Get the histogram used for detection
    pub fn histogram(&self) -> &Histogram<T> {
        &self.histogram
    }

    /// Get the lowland indices
    pub fn lowland_indices(&self) -> &[usize] {
        &self.lowland_indices
    }

    /// Find which mode (if any) contains a given value
    pub fn find_mode(&self, value: T::Float) -> Option<usize> {
        self.modes.iter().position(|mode| mode.contains(value))
    }

    /// Get mode assignments for a set of values
    pub fn assign_modes(&self, values: &[T]) -> Vec<Option<usize>> {
        values.iter().map(|&value| self.find_mode(value.to_float())).collect()
    }

    /// Get summary statistics
    pub fn summary(&self) -> ModalitySummary<T> {
        use num_traits::{Zero, Float};
        
        let zero = T::Float::zero();
        let inf = T::Float::infinity();
        
        let max_height = self.modes.iter()
            .map(|m| m.height)
            .fold(zero, |a, b| if a > b { a } else { b });
        
        let min_height = self.modes.iter()
            .map(|m| m.height)
            .fold(inf, |a, b| if a < b { a } else { b });
            
        ModalitySummary {
            mode_count: self.mode_count(),
            total_histogram_bins: self.histogram.len(),
            lowland_count: self.lowland_indices.len(),
            data_range: self.histogram.range(),
            max_mode_height: max_height,
            min_mode_height: min_height,
        }
    }
}

impl<T: Numeric> fmt::Display for ModalityResult<T> 
where
    T::Float: fmt::Debug + PartialEq + num_traits::ToPrimitive,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModalityResult({} modes detected)", self.mode_count())
    }
}

/// Summary statistics for a modality detection result
#[derive(Debug, Clone, PartialEq)]
pub struct ModalitySummary<T: Numeric = f64> 
where
    T::Float: fmt::Debug + PartialEq + num_traits::ToPrimitive,
{
    /// Number of detected modes
    pub mode_count: usize,
    /// Total number of histogram bins
    pub total_histogram_bins: usize,
    /// Number of lowland regions
    pub lowland_count: usize,
    /// Range of the data
    pub data_range: T::Float,
    /// Maximum mode height
    pub max_mode_height: T::Float,
    /// Minimum mode height
    pub min_mode_height: T::Float,
}

impl<T: Numeric> fmt::Display for ModalitySummary<T> 
where
    T::Float: fmt::Display + fmt::Debug + PartialEq + num_traits::ToPrimitive,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Modes: {}, Histogram bins: {}, Lowlands: {}, Range: {:.3}",
            self.mode_count, self.total_histogram_bins, self.lowland_count, self.data_range
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robust_histogram::Histogram;

    #[test]
    fn test_mode_creation() {
        let mode = Mode::<f64>::new(5.0, 3.0, 7.0, 2.5);
        assert_eq!(mode.location, 5.0);
        assert_eq!(mode.width(), 4.0);
        assert!(mode.contains(5.0));
        assert!(!mode.contains(2.0));
        assert_eq!(mode.relative_position(5.0), 0.5);
    }

    #[test]
    fn test_modality_result() {
        let modes = vec![Mode::<f64>::new(1.0, 0.5, 1.5, 0.8), Mode::<f64>::new(3.0, 2.5, 3.5, 0.6)];
        let hist = Histogram::<f64>::new(vec![], 0, 0.0, 4.0);
        let result = ModalityResult::<f64>::new(modes, hist, vec![2]);

        assert_eq!(result.mode_count(), 2);
        assert!(result.is_multimodal());
        assert!(!result.is_unimodal());
        assert_eq!(result.find_mode(1.0), Some(0));
        assert_eq!(result.find_mode(3.0), Some(1));
        assert_eq!(result.find_mode(2.0), None);
    }
}
