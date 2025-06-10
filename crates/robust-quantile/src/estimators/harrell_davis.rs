//! Harrell-Davis quantile estimator using generic implementation

use crate::weights::HDWeightComputer;
use robust_core::{ExecutionEngine, Numeric};

use super::generic::GenericQuantileEstimator;

/// Harrell-Davis quantile estimator
///
/// This estimator provides smooth, differentiable quantile estimates
/// using a weighted average of order statistics.
pub type HarrellDavis<T, E> = GenericQuantileEstimator<T, E, HDWeightComputer<T>>;

/// State for Harrell-Davis estimator
pub type HDState<T> = super::generic::GenericState<T, HDWeightComputer<T>>;

/// Convenience constructor for Harrell-Davis estimator
pub fn harrell_davis<T: Numeric, E: ExecutionEngine<T>>(engine: E) -> HarrellDavis<T, E> {
    HarrellDavis::new(engine, HDWeightComputer::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QuantileEstimator;
    use robust_core::{execution::scalar_sequential, BatchProcessor, CachePolicy, ProcessingStrategy, UnifiedWeightCache};

    #[test]
    fn test_harrell_davis_median() {
        let engine = scalar_sequential();
        let hd = harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let median = hd.quantile_sorted(&data, 0.5, &cache).unwrap();
        assert!((median - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_harrell_davis_quartiles() {
        let engine = scalar_sequential();
        let hd = harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let quartiles = hd
            .quantiles_sorted(&data, &[0.25, 0.5, 0.75], &cache)
            .unwrap();

        assert_eq!(quartiles.len(), 3);
        assert!(quartiles[0] < quartiles[1]);
        assert!(quartiles[1] < quartiles[2]);
    }

    #[test]
    fn test_batch_processing() {
        let engine = scalar_sequential();
        let hd = harrell_davis(engine);
        let cache =
            UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::Lru { max_entries: 1024 });

        let datasets = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3.0, 4.0, 5.0, 6.0, 7.0],
        ];

        let sorted_refs: Vec<&[f64]> = datasets.iter().map(|d| d.as_slice()).collect();
        let quantiles = vec![0.25, 0.5, 0.75];

        let results = hd
            .process_batch_sorted(&sorted_refs, &quantiles, &cache, ProcessingStrategy::Auto)
            .unwrap();

        assert_eq!(results.len(), 3); // 3 datasets
        assert_eq!(results[0].len(), 3); // 3 quantiles each
    }
}