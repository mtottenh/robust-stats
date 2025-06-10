//! Weighted sum kernel for quantile operations

use crate::QuantileKernel;
use robust_core::{ComputePrimitives, ExecutionEngine, SparseWeights, StatisticalKernel, Numeric};
use std::marker::PhantomData;

/// Kernel for weighted sum operations in quantile estimation
/// 
/// This kernel handles single weighted sum operations only.
/// For batch or tiled operations, use specialized kernels.
#[derive(Clone)]
pub struct WeightedSumKernel<T: Numeric, E: ExecutionEngine<T>> {
    engine: E,
    _phantom: PhantomData<T>,
}

impl<T: Numeric, E: ExecutionEngine<T>> WeightedSumKernel<T, E> {
    /// Create new kernel
    pub fn new(engine: E) -> Self {
        Self { engine, _phantom: PhantomData }
    }
}

impl<T: Numeric, E: ExecutionEngine<T>> StatisticalKernel<T> for WeightedSumKernel<T, E> {
    type Primitives = E::Primitives;

    fn primitives(&self) -> &Self::Primitives {
        self.engine.primitives()
    }

    fn name(&self) -> &'static str {
        "Weighted Sum Kernel"
    }
}

impl<T: Numeric, E: ExecutionEngine<T>> QuantileKernel<T> for WeightedSumKernel<T, E> {
    fn apply_sparse_weights(&self, sorted_data: &[T], weights: &SparseWeights<T>) -> T::Float {
        // Always validate that weights match data size (not just in debug mode)
        // This is critical for correctness
        if sorted_data.len() != weights.n {
            panic!(
                "CRITICAL ERROR: Data length {} does not match weights dimension {}. \
                This indicates a cache corruption or incorrect weight reuse.",
                sorted_data.len(),
                weights.n
            );
        }
        
        // Direct to primitives - pure computation
        let aggregate = self.primitives()
            .sparse_weighted_sum(sorted_data, &weights.indices, &weights.weights);
        // Convert Aggregate to Float
        <T::Float as num_traits::NumCast>::from(aggregate.into()).unwrap()
    }

    fn apply_sparse_weights_with_moments(
        &self,
        sorted_data: &[T],
        weights: &SparseWeights<T>,
    ) -> (T::Float, T::Float) {
        let primitives = self.primitives();

        // First moment
        let c1_aggregate = primitives.sparse_weighted_sum(sorted_data, &weights.indices, &weights.weights);
        let c1 = <T::Float as num_traits::NumCast>::from(c1_aggregate.into()).unwrap();

        // Second moment - need squared values
        // We'll compute in the original type space then convert
        let squared_values: Vec<T> = weights
            .indices
            .iter()
            .map(|&i| {
                let val = sorted_data[i];
                // This won't work for all types...
                // For now, convert to float, square, convert back
                let float_val = val.to_float();
                let squared = float_val * float_val;
                T::from_f64(squared.into())
            })
            .collect();

        let c2_aggregate = primitives.dot_product(&squared_values, &weights.weights);
        let c2 = <T::Float as num_traits::NumCast>::from(c2_aggregate.into()).unwrap();

        (c1, c2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::compute_hd_weights;

    #[test]
    fn test_weighted_sum_kernel_basic() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();

        // Generate weight for median
        let weights = compute_hd_weights(100, 0.5);

        // Test with scalar engine
        let engine = robust_core::execution::scalar_sequential();
        let kernel = WeightedSumKernel::new(engine);

        let result = kernel.apply_sparse_weights(&data, &weights);
        
        // Median of 0..99 should be around 49.5
        assert!((result - 49.5).abs() < 1.0);
    }

    #[test]
    fn test_moments_computation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = compute_hd_weights(5, 0.5);

        let engine = robust_core::execution::scalar_sequential();
        let kernel = WeightedSumKernel::new(engine);

        let (c1, c2) = kernel.apply_sparse_weights_with_moments(&data, &weights);
        
        // First moment should be around 3 (median)
        assert!((c1 - 3.0).abs() < 0.5);
        
        // Second moment should be positive
        assert!(c2 > 0.0);
    }
}