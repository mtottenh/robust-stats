//! Trimmed Harrell-Davis quantile estimator using generic implementation

use crate::weights::{TrimmedHDWeightComputer, WidthComputation};
use robust_core::{ExecutionEngine, Numeric};

use super::generic::GenericQuantileEstimator;

// Re-export the width function types with more user-friendly names
pub use crate::weights::{
    ConstantWidthFn as ConstantWidth, LinearWidthFn as LinearWidth, SqrtWidthFn as SqrtWidth,
};

/// Trait alias for width functions (for documentation)
pub trait WidthFunction: WidthComputation {}

/// Trimmed Harrell-Davis quantile estimator
///
/// This estimator reduces the influence of extreme order statistics
/// by trimming the weights at the tails.
pub type TrimmedHarrellDavis<T, E, W> = GenericQuantileEstimator<T, E, TrimmedHDWeightComputer<T, W>>;

/// State for Trimmed Harrell-Davis estimator
pub type TrimmedHDState<T, W> = super::generic::GenericState<T, TrimmedHDWeightComputer<T, W>>;

/// Convenience constructor for Trimmed Harrell-Davis estimator
pub fn trimmed_harrell_davis<T: Numeric, E: ExecutionEngine<T>, W: WidthComputation>(
    engine: E,
    width_fn: W,
) -> TrimmedHarrellDavis<T, E, W> {
    TrimmedHarrellDavis::new(engine, TrimmedHDWeightComputer::new(width_fn))
}

/// Convenience type aliases
pub type TrimmedHDSqrt<T, E> = TrimmedHarrellDavis<T, E, SqrtWidth>;
pub type TrimmedHDLinear<T, E> = TrimmedHarrellDavis<T, E, LinearWidth>;
pub type TrimmedHDConstant<T, E> = TrimmedHarrellDavis<T, E, ConstantWidth>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QuantileEstimator;
    use robust_core::{execution::scalar_sequential, CachePolicy, UnifiedWeightCache};

    #[test]
    fn test_trimmed_hd_median() {
        let engine = scalar_sequential();
        let thd = trimmed_harrell_davis(engine, SqrtWidth);
        let cache = UnifiedWeightCache::new(
            TrimmedHDWeightComputer::new(SqrtWidth),
            CachePolicy::NoCache,
        );

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let median = thd.quantile_sorted(&data, 0.5, &cache).unwrap();
        assert!((median - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_different_width_functions() {
        let engine = scalar_sequential();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Test with different width functions
        let sqrt_thd = trimmed_harrell_davis(engine.clone(), SqrtWidth);
        let linear_thd = trimmed_harrell_davis(engine.clone(), LinearWidth);
        let const_thd = trimmed_harrell_davis(engine, ConstantWidth::new(0.2));

        let sqrt_cache = UnifiedWeightCache::new(
            TrimmedHDWeightComputer::new(SqrtWidth),
            CachePolicy::NoCache,
        );
        let linear_cache = UnifiedWeightCache::new(
            TrimmedHDWeightComputer::new(LinearWidth),
            CachePolicy::NoCache,
        );
        let const_cache = UnifiedWeightCache::new(
            TrimmedHDWeightComputer::new(ConstantWidth::new(0.2)),
            CachePolicy::NoCache,
        );

        let sqrt_median = sqrt_thd.quantile_sorted(&data, 0.5, &sqrt_cache).unwrap();
        let linear_median = linear_thd
            .quantile_sorted(&data, 0.5, &linear_cache)
            .unwrap();
        let const_median = const_thd.quantile_sorted(&data, 0.5, &const_cache).unwrap();

        // All should be close to the true median of 5.5
        assert!((sqrt_median - 5.5).abs() < 1.0);
        assert!((linear_median - 5.5).abs() < 1.0);
        assert!((const_median - 5.5).abs() < 1.0);
    }
}
