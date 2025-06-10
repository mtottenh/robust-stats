//! Core traits for histogram building

use crate::types::Histogram;
use robust_core::{Result, Numeric};

/// Trait for building histograms from sample data
pub trait HistogramBuilder<T: Numeric = f64> {
    /// Build a histogram from the given sample
    fn build(&self, sample: &[T]) -> Result<Histogram<T>>;
    
    /// Build a histogram from pre-sorted data
    /// 
    /// This can be more efficient for some builders that need sorted data.
    /// Default implementation just calls build().
    fn build_sorted(&self, sorted_sample: &[T]) -> Result<Histogram<T>> {
        self.build(sorted_sample)
    }

    /// Get the target number of bins (if known)
    fn target_bins(&self) -> Option<usize> {
        None
    }

    /// Check if this builder supports weighted samples
    fn supports_weighted_samples(&self) -> bool {
        false
    }
}
