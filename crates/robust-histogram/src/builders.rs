//! Various histogram building strategies

use crate::traits::HistogramBuilder;
use crate::types::{Histogram, HistogramBin};
use robust_core::{Result, Numeric};
use num_traits::{Zero, Float, NumCast};

/// Fixed-width histogram builder
///
/// Creates a histogram with a specified number of equal-width bins.
pub struct FixedWidthBuilder {
    num_bins: usize,
}

impl FixedWidthBuilder {
    /// Create a new fixed-width histogram builder
    pub fn new(num_bins: usize) -> Self {
        Self {
            num_bins: num_bins.max(1),
        }
    }
}

impl<T: Numeric> HistogramBuilder<T> for FixedWidthBuilder {
    fn build(&self, sample: &[T]) -> Result<Histogram<T>> {
        let mut sorted = sample.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.build_sorted(&sorted)
    }
    
    fn build_sorted(&self, sorted_sample: &[T]) -> Result<Histogram<T>> {
        if sorted_sample.is_empty() {
            return Ok(Histogram::new(vec![], 0, T::Float::zero(), T::Float::zero()));
        }

        // O(1) min/max from sorted data
        let min = sorted_sample[0].to_float();
        let max = sorted_sample[sorted_sample.len() - 1].to_float();

        let epsilon = <T::Float as NumCast>::from(1e-10).unwrap();
        if T::Float::abs(max - min) < epsilon {
            // All values are the same
            let bin = HistogramBin::new(min, max, sorted_sample.len(), sorted_sample.len());
            return Ok(Histogram::new(vec![bin], sorted_sample.len(), min, max));
        }

        let num_bins_f = <T::Float as NumCast>::from(self.num_bins).unwrap();
        let width = (max - min) / num_bins_f;
        let mut bins = Vec::with_capacity(self.num_bins);

        // Create bins
        for i in 0..self.num_bins {
            let i_f = <T::Float as NumCast>::from(i).unwrap();
            let i_plus_1_f = <T::Float as NumCast>::from(i + 1).unwrap();
            let left = min + i_f * width;
            let right = if i == self.num_bins - 1 {
                max // Ensure last bin includes max
            } else {
                min + i_plus_1_f * width
            };
            bins.push(HistogramBin::new(left, right, 0, sorted_sample.len()));
        }

        // Count values in each bin - efficient single pass through sorted data
        let mut current_bin = 0;
        
        for &value in sorted_sample {
            let value_f = value.to_float();
            // Move to the correct bin
            while current_bin < self.num_bins - 1 && value_f >= bins[current_bin].right {
                current_bin += 1;
            }
            
            if current_bin < self.num_bins {
                bins[current_bin].count += 1;
            }
        }

        // Update densities
        let total = sorted_sample.len();
        for bin in &mut bins {
            let count_f = <T::Float as NumCast>::from(bin.count).unwrap();
            let total_f = <T::Float as NumCast>::from(total).unwrap();
            bin.density = count_f / (total_f * bin.width());
        }

        Ok(Histogram::new(bins, sorted_sample.len(), min, max))
    }

    fn target_bins(&self) -> Option<usize> {
        Some(self.num_bins)
    }
}

/// Scott's rule for optimal bin width
/// 
/// Uses the formula: h = 3.5 * σ * n^(-1/3)
/// where σ is the standard deviation and n is the sample size.
pub struct ScottsRule;

impl<T: Numeric> HistogramBuilder<T> for ScottsRule {
    fn build(&self, sample: &[T]) -> Result<Histogram<T>> {
        if sample.is_empty() {
            return Ok(Histogram::new(vec![], 0, T::Float::zero(), T::Float::zero()));
        }

        // Calculate mean and standard deviation
        let n = sample.len();
        let n_f = <T::Float as NumCast>::from(n).unwrap();
        let sum = sample.iter().fold(T::Float::zero(), |acc, &x| acc + x.to_float());
        let mean = sum / n_f;

        let variance = sample
            .iter()
            .fold(T::Float::zero(), |acc, &x| {
                let diff = x.to_float() - mean;
                acc + diff * diff
            })
            / n_f;
        let std_dev = variance.sqrt();

        // Scott's rule
        let three_point_five = <T::Float as NumCast>::from(3.5).unwrap();
        let one_third = <T::Float as NumCast>::from(1.0 / 3.0).unwrap();
        let bin_width = three_point_five * std_dev * n_f.powf(-one_third);

        let min = sample.iter().map(|v| v.to_float()).fold(T::Float::infinity(), |a, b| if b < a { b } else { a });
        let max = sample.iter().map(|v| v.to_float()).fold(T::Float::neg_infinity(), |a, b| if b > a { b } else { a });

        let range = max - min;
        let epsilon = <T::Float as NumCast>::from(1e-10).unwrap();
        let num_bins = if bin_width > epsilon {
            use num_traits::cast::ToPrimitive;
            ((range / bin_width).ceil().to_usize().unwrap_or(1)).max(1)
        } else {
            1
        };

        FixedWidthBuilder::new(num_bins).build(sample)
    }
}

/// Freedman-Diaconis rule for optimal bin width
/// 
/// Uses the formula: h = 2 * IQR * n^(-1/3)
/// where IQR is the interquartile range.
pub struct FreedmanDiaconisRule;

impl<T: Numeric> HistogramBuilder<T> for FreedmanDiaconisRule {
    fn build(&self, sample: &[T]) -> Result<Histogram<T>> {
        if sample.is_empty() {
            return Ok(Histogram::new(vec![], 0, T::Float::zero(), T::Float::zero()));
        }

        let mut sorted = sample.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let q1_idx = n / 4;
        let q3_idx = (3 * n) / 4;

        let q1 = sorted[q1_idx].to_float();
        let q3 = sorted[q3_idx].to_float();
        let iqr = q3 - q1;

        // Freedman-Diaconis rule
        let two = <T::Float as NumCast>::from(2.0).unwrap();
        let one_third = <T::Float as NumCast>::from(1.0 / 3.0).unwrap();
        let n_f = <T::Float as NumCast>::from(n).unwrap();
        let bin_width = two * iqr * n_f.powf(-one_third);

        let min = sorted[0].to_float();
        let max = sorted[n - 1].to_float();
        let range = max - min;

        let epsilon = <T::Float as NumCast>::from(1e-10).unwrap();
        let num_bins = if bin_width > epsilon {
            use num_traits::cast::ToPrimitive;
            ((range / bin_width).ceil().to_usize().unwrap_or(1)).max(1)
        } else {
            1
        };

        FixedWidthBuilder::new(num_bins).build_sorted(&sorted)
    }
}

/// Quantile-based histogram builders that require a quantile estimator
pub mod quantile {
    use super::*;

    /// Quantile-based histogram builder
    ///
    /// Creates bins with boundaries at quantiles, making the histogram
    /// robust to outliers. Each bin (except possibly the last) contains
    /// approximately the same number of points.
    /// 
    /// The estimator is now provided at build time to enable batch optimization.
    pub struct QuantileBuilder {
        num_bins: usize,
    }

    impl QuantileBuilder {
        /// Create a new quantile-based histogram builder
        pub fn new(num_bins: usize) -> Self {
            Self {
                num_bins: num_bins.max(1),
            }
        }

        /// Build a histogram using the provided quantile estimator
        /// 
        /// This enables batch optimization when a batch-capable estimator is provided.
        pub fn build<T, Q>(&self, sample: &[T], estimator: &Q, cache: &Q::State) -> Result<Histogram<T>>
        where
            T: Numeric,
            Q: robust_core::BatchQuantileEstimator<T>,
        {
            if sample.is_empty() {
                use num_traits::Zero;
                return Ok(Histogram::new(vec![], 0, T::Float::zero(), T::Float::zero()));
            }

            if sample.len() < self.num_bins {
                // Not enough data for requested bins
                return FixedWidthBuilder::new(sample.len()).build(sample);
            }

            // Generate quantile points for uniform bins
            let quantile_points: Vec<f64> = (0..=self.num_bins)
                .map(|i| i as f64 / self.num_bins as f64)
                .collect();

            // Sort the data once
            let mut sorted = sample.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Use batch quantiles for efficiency
            let edges = if quantile_points.len() > 4 {
                // For many quantiles, use batch operation
                match estimator.estimate_quantiles_sorted_with_cache(&sorted, &quantile_points, cache) {
                    Ok(batch_result) => batch_result,
                    Err(e) => return Err(e),
                }
            } else {
                // For few quantiles, still use batch for consistency
                estimator.estimate_quantiles_sorted_with_cache(&sorted, &quantile_points, cache)?
            };

            // Remove duplicate edges (can happen with discrete data)
            let mut unique_edges = edges;
            let epsilon: T::Float = NumCast::from(1e-10).unwrap();
            unique_edges.dedup_by(|a, b| (*a - *b).abs() < epsilon);

            if unique_edges.len() < 2 {
                // All values are the same
                let value = unique_edges[0];
                let bin = HistogramBin::new(value, value, sample.len(), sample.len());
                return Ok(Histogram::new(vec![bin], sample.len(), value, value));
            }

            // Create bins from edges
            let mut bins = Vec::new();
            for i in 0..unique_edges.len() - 1 {
                bins.push(HistogramBin::new(unique_edges[i], unique_edges[i + 1], 0, sample.len()));
            }

            // Count values in each bin
            for &value in sample {
                let val_f = value.to_float();
                if let Some(bin_idx) = bins.iter().position(|b| {
                    val_f >= b.left
                        && (val_f < b.right
                            || (val_f == b.right && b.right == unique_edges[unique_edges.len() - 1]))
                }) {
                    bins[bin_idx].count += 1;
                }
            }

            // Update densities
            let total = sample.len();
            for bin in &mut bins {
                let count_f: T::Float = NumCast::from(bin.count).unwrap();
                let total_f: T::Float = NumCast::from(total).unwrap();
                bin.density = count_f / (total_f * bin.width());
            }

            let min = sample.iter().map(|v| v.to_float()).fold(T::Float::infinity(), |a, b| if b < a { b } else { a });
            let max = sample.iter().map(|v| v.to_float()).fold(T::Float::neg_infinity(), |a, b| if b > a { b } else { a });

            Ok(Histogram::new(bins, sample.len(), min, max))
        }
    }

    // Note: QuantileBuilder no longer implements HistogramBuilder trait directly.
    // Users must explicitly provide an estimator via the build method.

    /// Function type for generating quantile steps
    pub type QuantileStepFunction = Box<dyn Fn(usize) -> Vec<f64>>; // TODO: Make this generic when needed

    /// QRDE (Quantile-Respectful Density Estimator) builder with custom step functions
    /// 
    /// This builder allows custom quantile step functions for more
    /// flexible density estimation than fixed bin counts.
    /// 
    /// The estimator is now provided at build time to enable batch optimization.
    pub struct QRDEBuilderWithSteps {
        step_function: QuantileStepFunction,
    }

    impl QRDEBuilderWithSteps {
        /// Create a new QRDE builder with a custom step function
        /// 
        /// The step function takes a sample size and returns the quantile points
        /// to use for binning (e.g., [0.0, 0.1, 0.2, ..., 1.0])
        pub fn new(step_function: QuantileStepFunction) -> Self {
            Self {
                step_function,
            }
        }

        /// Create a QRDE with uniform steps
        pub fn uniform(num_bins: usize) -> Self {
            let step_fn = Box::new(move |_sample_size: usize| {
                (0..=num_bins)
                    .map(|i| i as f64 / num_bins as f64)
                    .collect()
            });
            Self::new(step_fn)
        }

        /// Create a QRDE with adaptive steps based on sample size
        /// 
        /// Uses more bins for larger samples: sqrt(n) bins
        pub fn adaptive() -> Self {
            let step_fn = Box::new(|sample_size: usize| {
                let num_bins = (sample_size as f64).sqrt().ceil() as usize;
                (0..=num_bins)
                    .map(|i| i as f64 / num_bins as f64)
                    .collect()
            });
            Self::new(step_fn)
        }

        /// Create a QRDE with exponential steps for better tail resolution
        /// 
        /// Concentrates more bins in the tails of the distribution
        pub fn exponential(num_bins: usize) -> Self {
            let step_fn = Box::new(move |_sample_size: usize| {
                let mut steps = Vec::with_capacity(num_bins + 1);
                steps.push(0.0);
                
                // First half: exponentially spaced from 0 to 0.5
                let half = num_bins / 2;
                for i in 1..=half {
                    let t = i as f64 / half as f64;
                    steps.push(0.5 * (1.0 - (-5.0 * t).exp()));
                }
                
                // Second half: mirror of first half
                for i in 1..half {
                    steps.push(1.0 - steps[half - i]);
                }
                steps.push(1.0);
                
                steps
            });
            Self::new(step_fn)
        }

        /// Build a histogram using the provided quantile estimator
        /// 
        /// This is the new primary method that enables batch optimization
        /// when a batch-capable estimator is provided.
        pub fn build<T, Q>(&self, sample: &[T], estimator: &Q, cache: &Q::State) -> Result<Histogram<T>>
        where
            T: Numeric,
            Q: robust_core::BatchQuantileEstimator<T>,
        {
            if sample.is_empty() {
                use num_traits::Zero;
                return Ok(Histogram::new(vec![], 0, T::Float::zero(), T::Float::zero()));
            }

            // Get quantile points from step function
            let quantile_points_f64 = (self.step_function)(sample.len());
            
            if quantile_points_f64.len() < 2 {
                return Err(robust_core::Error::InvalidInput(
                    "Step function must return at least 2 quantile points".to_string()
                ));
            }

            // Use f64 quantile points directly (BatchQuantileEstimator expects &[f64])
            let quantile_points = quantile_points_f64;

            // Sort the data once
            let mut sorted = sample.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Use batch quantiles for efficiency
            let edges = estimator.estimate_quantiles_sorted_with_cache(&sorted, &quantile_points, cache)?;

            // Remove duplicate edges (can happen with discrete data)
            let mut unique_edges = edges;
            let epsilon: T::Float = NumCast::from(1e-10).unwrap();
            unique_edges.dedup_by(|a, b| (*a - *b).abs() < epsilon);

            if unique_edges.len() < 2 {
                // All values are the same
                let value = unique_edges[0];
                let bin = HistogramBin::new(value, value, sample.len(), sample.len());
                return Ok(Histogram::new(vec![bin], sample.len(), value, value));
            }

            // Create bins from edges
            let mut bins = Vec::new();
            for i in 0..unique_edges.len() - 1 {
                bins.push(HistogramBin::new(unique_edges[i], unique_edges[i + 1], 0, sample.len()));
            }

            // Count values in each bin
            for &value in sample {
                let val_f = value.to_float();
                if let Some(bin_idx) = bins.iter().position(|b| {
                    val_f >= b.left
                        && (val_f < b.right
                            || (val_f == b.right && b.right == unique_edges[unique_edges.len() - 1]))
                }) {
                    bins[bin_idx].count += 1;
                }
            }

            // Update densities
            let total = sample.len();
            for bin in &mut bins {
                let count_f: T::Float = NumCast::from(bin.count).unwrap();
                let total_f: T::Float = NumCast::from(total).unwrap();
                bin.density = count_f / (total_f * bin.width());
            }

            let min = sample.iter().map(|v| v.to_float()).fold(T::Float::infinity(), |a, b| if b < a { b } else { a });
            let max = sample.iter().map(|v| v.to_float()).fold(T::Float::neg_infinity(), |a, b| if b > a { b } else { a });

            Ok(Histogram::new(bins, sample.len(), min, max))
        }
    }

    // Note: QRDEBuilderWithSteps no longer implements HistogramBuilder trait directly.
    // Users must explicitly provide an estimator via the build method.
}

// Re-export for convenience
pub use quantile::{QRDEBuilderWithSteps, QuantileBuilder};