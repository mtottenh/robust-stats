//! Core types for histogram representation

use std::fmt;
use robust_core::Numeric;
use num_traits::{Zero, Float, NumCast};

/// A single bin in a histogram
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBin<T: Numeric = f64> {
    /// Left edge of the bin (inclusive)
    pub left: T::Float,
    /// Right edge of the bin (exclusive, except for the last bin)
    pub right: T::Float,
    /// Number of values in this bin
    pub count: usize,
    /// Density (count / (total_count * bin_width))
    pub density: T::Float,
}

impl<T: Numeric> HistogramBin<T> 
where
    T::Float: Float,
{
    /// Create a new histogram bin
    pub fn new(left: T::Float, right: T::Float, count: usize, total_count: usize) -> Self {
        let width = right - left;
        let density = if width > T::Float::zero() && total_count > 0 {
            let count_f: T::Float = NumCast::from(count).unwrap();
            let total_f: T::Float = NumCast::from(total_count).unwrap();
            count_f / (total_f * width)
        } else {
            T::Float::zero()
        };

        Self {
            left,
            right,
            count,
            density,
        }
    }

    /// Get the center point of the bin
    pub fn center(&self) -> T::Float {
        let two: T::Float = NumCast::from(2.0).unwrap();
        (self.left + self.right) / two
    }

    /// Get the width of the bin
    pub fn width(&self) -> T::Float {
        self.right - self.left
    }

    /// Check if a value falls within this bin
    pub fn contains(&self, value: T::Float) -> bool {
        value >= self.left && value < self.right
    }

    /// Get the relative frequency (count / total_count)
    pub fn frequency(&self, total_count: usize) -> T::Float {
        if total_count > 0 {
            let count_f: T::Float = NumCast::from(self.count).unwrap();
            let total_f: T::Float = NumCast::from(total_count).unwrap();
            count_f / total_f
        } else {
            T::Float::zero()
        }
    }
}

impl<T: Numeric> fmt::Display for HistogramBin<T> 
where
    T::Float: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:.3}, {:.3}): count={}, density={:.3}",
            self.left, self.right, self.count, self.density
        )
    }
}

/// A histogram representation of data
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram<T: Numeric = f64> {
    /// The bins that make up the histogram
    bins: Vec<HistogramBin<T>>,
    /// Total number of data points
    total_count: usize,
    /// Minimum value in the data
    min: T::Float,
    /// Maximum value in the data  
    max: T::Float,
}

impl<T: Numeric> Histogram<T> {
    /// Create a new histogram
    pub fn new(bins: Vec<HistogramBin<T>>, total_count: usize, min: T::Float, max: T::Float) -> Self {
        Self {
            bins,
            total_count,
            min,
            max,
        }
    }

    /// Get the bins
    pub fn bins(&self) -> &[HistogramBin<T>] {
        &self.bins
    }

    /// Get mutable access to bins (for internal use)
    pub(crate) fn bins_mut(&mut self) -> &mut Vec<HistogramBin<T>> {
        &mut self.bins
    }

    /// Get the number of bins
    pub fn len(&self) -> usize {
        self.bins.len()
    }

    /// Check if the histogram is empty
    pub fn is_empty(&self) -> bool {
        self.bins.is_empty()
    }

    /// Get the total count of data points
    pub fn total_count(&self) -> usize {
        self.total_count
    }

    /// Get the minimum value
    pub fn min(&self) -> T::Float {
        self.min
    }

    /// Get the maximum value
    pub fn max(&self) -> T::Float {
        self.max
    }

    /// Get the range of the histogram
    pub fn range(&self) -> T::Float {
        self.max - self.min
    }

    /// Get the maximum density in the histogram
    pub fn max_density(&self) -> T::Float {
        self.bins.iter().map(|bin| bin.density).fold(T::Float::zero(), |acc, d| if d > acc { d } else { acc })
    }

    /// Get the minimum density in the histogram
    pub fn min_density(&self) -> T::Float {
        self.bins
            .iter()
            .map(|bin| bin.density)
            .fold(T::Float::infinity(), |acc, d| if d < acc { d } else { acc })
    }

    /// Get the maximum count in any bin
    pub fn max_count(&self) -> usize {
        self.bins.iter().map(|bin| bin.count).max().unwrap_or(0)
    }

    /// Find which bin contains a given value
    pub fn find_bin(&self, value: T::Float) -> Option<usize> {
        // Handle last bin specially (includes right boundary)
        if !self.bins.is_empty() {
            let last_idx = self.bins.len() - 1;
            if value == self.bins[last_idx].right {
                return Some(last_idx);
            }
        }

        self.bins.iter().position(|bin| bin.contains(value))
    }

    /// Get counts as a vector
    pub fn counts(&self) -> Vec<usize> {
        self.bins.iter().map(|bin| bin.count).collect()
    }

    /// Get densities as a vector
    pub fn densities(&self) -> Vec<T::Float> {
        self.bins.iter().map(|bin| bin.density).collect()
    }

    /// Get frequencies as a vector
    pub fn frequencies(&self) -> Vec<T::Float> {
        self.bins
            .iter()
            .map(|bin| bin.frequency(self.total_count))
            .collect()
    }

    /// Get bin centers as a vector
    pub fn centers(&self) -> Vec<T::Float> {
        self.bins.iter().map(|bin| bin.center()).collect()
    }

    /// Get bin edges (including rightmost edge)
    pub fn edges(&self) -> Vec<T::Float> {
        if self.bins.is_empty() {
            return vec![];
        }

        let mut edges = Vec::with_capacity(self.bins.len() + 1);
        for bin in &self.bins {
            edges.push(bin.left);
        }
        edges.push(self.bins.last().unwrap().right);
        edges
    }

    /// Create a normalized copy (total density = 1)
    pub fn normalize(&self) -> Self {
        let mut normalized = self.clone();
        let total_density: T::Float = self.bins.iter().map(|bin| bin.density * bin.width()).fold(T::Float::zero(), |acc, x| acc + x);

        if total_density > T::Float::zero() {
            for bin in &mut normalized.bins {
                bin.density = bin.density / total_density;
            }
        }

        normalized
    }
}

impl<T: Numeric> fmt::Display for Histogram<T> 
where
    T::Float: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Histogram({} bins, n={}, range=[{:.3}, {:.3}])",
            self.len(),
            self.total_count,
            self.min,
            self.max
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_bin() {
        let bin = HistogramBin::<f64>::new(0.0, 1.0, 5, 10);
        assert_eq!(bin.center(), 0.5);
        assert_eq!(bin.width(), 1.0);
        assert!(bin.contains(0.5));
        assert!(!bin.contains(1.0)); // Right edge is exclusive
        assert_eq!(bin.frequency(10), 0.5);
        assert_eq!(bin.density, 0.5); // 5 / (10 * 1.0)
    }

    #[test]
    fn test_histogram() {
        let bins = vec![
            HistogramBin::<f64>::new(0.0, 1.0, 2, 10),
            HistogramBin::<f64>::new(1.0, 2.0, 5, 10),
            HistogramBin::<f64>::new(2.0, 3.0, 3, 10),
        ];
        let hist = Histogram::<f64>::new(bins, 10, 0.0, 3.0);

        assert_eq!(hist.len(), 3);
        assert_eq!(hist.total_count(), 10);
        assert_eq!(hist.range(), 3.0);
        assert_eq!(hist.max_count(), 5);
        assert_eq!(hist.find_bin(1.5), Some(1));
        assert_eq!(hist.find_bin(3.0), Some(2)); // Last bin includes right edge
        assert_eq!(hist.counts(), vec![2, 5, 3]);

        let edges = hist.edges();
        assert_eq!(edges, vec![0.0, 1.0, 2.0, 3.0]);
    }
}
