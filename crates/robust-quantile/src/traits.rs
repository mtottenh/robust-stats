//! Core traits for quantile estimation

use crate::Result;
use robust_core::{
    CentralTendencyEstimator, ComputePrimitives, StatisticalKernel,
    BatchProcessor, BatchCharacteristics, SparseWeights, Numeric,
};
use num_traits::NumCast;

/// Main trait for quantile estimation
///
/// This extends `CentralTendencyEstimator` since median (p=0.5) is a
/// measure of central tendency.
pub trait QuantileEstimator<T: Numeric = f64>: CentralTendencyEstimator<T> + BatchProcessor<T> {
    /// Estimate a single quantile
    ///
    /// # Warning
    /// This method will sort the data in place! If you need to preserve the original
    /// order, use `quantile_sorted()` with pre-sorted data or make a copy first.
    ///
    /// # Arguments
    /// * `data` - The data sample (will be sorted in place)
    /// * `p` - The probability (0.0 to 1.0)
    /// * `cache` - Cache for expensive computations
    fn quantile(&self, data: &mut [T], p: f64, cache: &Self::State) -> Result<T::Float> {
        // Sort in place
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.quantile_sorted(data, p, cache)
    }
    
    /// Estimate a single quantile from pre-sorted data
    ///
    /// # Arguments
    /// * `sorted_data` - The data sample, already sorted
    /// * `p` - The probability (0.0 to 1.0)
    /// * `cache` - Cache for expensive computations
    fn quantile_sorted(&self, sorted_data: &[T], p: f64, cache: &Self::State) -> Result<T::Float>;
    
    /// Estimate multiple quantiles efficiently
    ///
    /// # Warning
    /// This method will sort the data in place! If you need to preserve the original
    /// order, use `quantiles_sorted()` with pre-sorted data or make a copy first.
    ///
    /// # Arguments
    /// * `data` - The data sample (will be sorted in place)
    /// * `ps` - The probabilities to estimate
    /// * `cache` - Cache for expensive computations
    fn quantiles(&self, data: &mut [T], ps: &[f64], cache: &Self::State) -> Result<Vec<T::Float>> {
        // Sort in place
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.quantiles_sorted(data, ps, cache)
    }
    
    /// Estimate multiple quantiles from pre-sorted data
    ///
    /// Implementations should provide efficient batch processing by leveraging
    /// the BatchProcessor infrastructure.
    ///
    /// # Arguments
    /// * `sorted_data` - The data sample, already sorted
    /// * `ps` - The probabilities to estimate
    /// * `cache` - Cache for expensive computations
    fn quantiles_sorted(&self, sorted_data: &[T], ps: &[f64], cache: &Self::State) -> Result<Vec<T::Float>>; 

    /// Get batch processing characteristics
    fn batch_characteristics(&self, _num_quantiles: usize) -> BatchCharacteristics {
        BatchCharacteristics {
            optimal_batch_size: Some(100),
            benefits_from_batching: true,
            preferred_dimension: Some(robust_core::BatchDimension::Parameters),
            supports_parallel: true,
        }
    }
}

/// Kernel trait for quantile-specific operations
pub trait QuantileKernel<T: Numeric = f64>: StatisticalKernel<T> {
    /// Apply sparse weights to sorted data
    fn apply_sparse_weights(&self, sorted_data: &[T], weights: &SparseWeights<T>) -> T::Float {
        let aggregate = self.primitives().sparse_weighted_sum(
            sorted_data,
            &weights.indices,
            &weights.weights,
        );
        // Convert Aggregate to Float
        <T::Float as NumCast>::from(aggregate.into()).unwrap()
    }
    
    /// Apply weights and compute both first and second moments
    fn apply_sparse_weights_with_moments(
        &self,
        sorted_data: &[T],
        weights: &SparseWeights<T>,
    ) -> (T::Float, T::Float) {
        let primitives = self.primitives();
        
        // First moment
        let c1_aggregate = primitives.sparse_weighted_sum(
            sorted_data,
            &weights.indices,
            &weights.weights,
        );
        let c1 = <T::Float as NumCast>::from(c1_aggregate.into()).unwrap();
        
        // Second moment - need squared values
        // We'll compute in the original type space then convert
        let squared_values: Vec<T> = weights.indices.iter()
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
        let c2 = <T::Float as NumCast>::from(c2_aggregate.into()).unwrap();
        
        (c1, c2)
    }
}
