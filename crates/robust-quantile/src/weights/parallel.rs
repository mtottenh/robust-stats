//! Parallel weight computation implementations for quantile estimators

use robust_core::{
    execution::ExecutionEngine, BatchProcessor, Numeric,
    Result,
};

// Re-export the function for backward compatibility if needed
pub use crate::weights::compute_hd_weights_range as compute_hd_weights_range_direct;

/// Integration with batch processing for parallel weight computation
pub trait ParallelBatchProcessor<T: Numeric>: BatchProcessor<T> {
    /// Process batch with parallel weight computation
    fn process_batch_parallel<E: ExecutionEngine<T>>(
        &self,
        inputs: &mut [&mut [T]],
        params: &[f64],
        engine: &E,
    ) -> Result<Vec<T::Float>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::compute_hd_weights;
    use approx::assert_relative_eq;
    use robust_core::WeightComputer;

    #[test]
    fn test_hd_weights_range() {
        let n = 100;
        let p = 0.5;

        // Compute full weights
        let full_weights = compute_hd_weights::<f64>(n, p);

        // Compute range weights
        let start = 40;
        let end = 60;
        let range_weights = compute_hd_weights_range_direct::<f64>(n, p, start, end);

        // Verify that range weights match the corresponding subset of full weights
        for (idx, weight) in range_weights.indices.iter().zip(&range_weights.weights) {
            if let Some(pos) = full_weights.indices.iter().position(|&i| i == *idx) {
                assert_relative_eq!(*weight, full_weights.weights[pos], epsilon = 1e-10);
            }
        }

        // Verify all indices are in range
        for &idx in &range_weights.indices {
            assert!(idx >= start && idx < end);
        }
    }

    #[test]
    fn test_weight_computer_with_engine() {
        use crate::weights::HDWeightComputer;
        use robust_core::execution::auto_engine;
        
        let computer = HDWeightComputer::<f64>::new();
        let n = 50;
        let quantiles = vec![0.25, 0.5, 0.75];
        let engine = auto_engine();

        // Test compute_tiled_with_engine
        let tiled = computer.compute_tiled_with_engine(n, &quantiles, &engine, 32, 256);
        
        // Verify dimensions
        assert_eq!(tiled.n_rows, quantiles.len());
        assert_eq!(tiled.n_cols, n);
        
        // Test that weights are non-empty
        assert!(tiled.tiles.len() > 0);
    }
}
