//! Weight computation for quantile estimators

mod harrell_davis;
mod hdi;
#[cfg(feature = "parallel")]
pub mod parallel;

pub use harrell_davis::{compute_hd_weights, compute_hd_weights_range};
pub use hdi::{compute_beta_hdi, compute_trimmed_weights};
#[cfg(feature = "parallel")]
pub use parallel::{ParallelBatchProcessor};
use robust_core::{SparseWeights, WeightComputer as RCWeightComputer, Numeric};
use std::marker::PhantomData;




// Trait for width computation
pub trait WidthComputation: Clone + Send + Sync {
    fn compute(&self, n: usize) -> f64;
}

// Zero-cost width function types
#[derive(Clone, Copy, Debug)]
pub struct SqrtWidthFn;

impl WidthComputation for SqrtWidthFn {
    fn compute(&self, n: usize) -> f64 {
        (1.0 / (n as f64).sqrt()).min(1.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LinearWidthFn;

impl WidthComputation for LinearWidthFn {
    fn compute(&self, n: usize) -> f64 {
        (1.0 / n as f64).min(1.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConstantWidthFn(f64);

impl ConstantWidthFn {
    /// Create a new constant width function
    pub fn new(width: f64) -> Self {
        Self(width)
    }
}

impl WidthComputation for ConstantWidthFn {
    fn compute(&self, _n: usize) -> f64 {
        self.0
    }
}

// Convenience constructors
impl<T: Numeric> TrimmedHDWeightComputer<T, SqrtWidthFn> {
    /// Create with sqrt(n) width
    pub fn sqrt() -> Self {
        Self::new(SqrtWidthFn)
    }
}

impl<T: Numeric> TrimmedHDWeightComputer<T, LinearWidthFn> {
    /// Create with linear width
    pub fn linear() -> Self {
        Self::new(LinearWidthFn)
    }
}

impl<T: Numeric> TrimmedHDWeightComputer<T, ConstantWidthFn> {
    /// Create with constant width
    pub fn constant(width: f64) -> Self {
        Self::new(ConstantWidthFn(width))
    }
}


// ===== New unified weight computers =====

/// Harrell-Davis weight computer that can produce both sparse and tiled representations
#[derive(Clone, Debug)]
pub struct HDWeightComputer<T: Numeric = f64> {
    _phantom: PhantomData<T>,
}

impl<T: Numeric> HDWeightComputer<T> {
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

impl<T: Numeric> Default for HDWeightComputer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Numeric> RCWeightComputer<T> for HDWeightComputer<T> {
    fn compute_sparse(&self, n: usize, p: f64) -> SparseWeights<T> {
        compute_hd_weights(n, p)
    }
    
    fn compute_sparse_range(&self, n: usize, p: f64, start: usize, end: usize) -> SparseWeights<T> {
        compute_hd_weights_range(n, p, start, end)
    }
    
    // Remove custom compute_tiled implementation to inherit the optimized default
    // which uses compute_tiled_with_engine and compute_sparse_range
}

/// Trimmed Harrell-Davis weight computer
#[derive(Clone, Debug)]
pub struct TrimmedHDWeightComputer<T: Numeric = f64, F: WidthComputation = SqrtWidthFn> {
    width_fn: F,
    _phantom: PhantomData<T>,
}

impl<T: Numeric, F: WidthComputation> TrimmedHDWeightComputer<T, F> {
    pub fn new(width_fn: F) -> Self {
        Self { width_fn, _phantom: PhantomData }
    }
}

impl<T: Numeric, F: WidthComputation> RCWeightComputer<T> for TrimmedHDWeightComputer<T, F> {
    fn compute_sparse(&self, n: usize, p: f64) -> SparseWeights<T> {
        let width = self.width_fn.compute(n);
        compute_trimmed_weights(n, p, width)
    }
    
    fn compute_sparse_range(&self, n: usize, p: f64, start: usize, end: usize) -> SparseWeights<T> {
        let width = self.width_fn.compute(n);
        let trim_start = ((n as f64 * (1.0 - width) / 2.0).floor() as usize).max(0);
        let trim_end = ((n as f64 * (1.0 + width) / 2.0).ceil() as usize).min(n);

        // Adjust range to account for trimming
        let effective_start = start.max(trim_start);
        let effective_end = end.min(trim_end);

        if effective_start >= effective_end {
            return SparseWeights {
                indices: vec![],
                weights: vec![],
                n,
            };
        }

        // Compute weights for the trimmed range
        let mut weights = compute_hd_weights_range::<T>(
            trim_end - trim_start,
            p,
            effective_start - trim_start,
            effective_end - trim_start,
        );

        // Adjust indices to account for trimming
        for idx in &mut weights.indices {
            *idx += trim_start;
        }

        // Update n to reflect the original size
        weights.n = n;

        weights
    }
    
    // Remove custom compute_tiled implementation to inherit the optimized default
    // which uses compute_tiled_with_engine and compute_sparse_range
}
