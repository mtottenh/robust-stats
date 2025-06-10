//! Unified generic compute primitives trait
//!
//! This is the new consolidated trait that will replace all existing primitive traits.

use crate::numeric::Numeric;
use num_traits::Zero;

/// Unified trait for computational primitives with generic numeric support
///
/// This trait provides low-level operations optimized for different backends
/// (scalar, AVX2, AVX512, etc.) with compile-time dispatch.
pub trait ComputePrimitives<T: Numeric = f64>: Clone + Send + Sync {
    /// Get the name of this backend
    fn backend_name(&self) -> &'static str;
    
    /// Get the SIMD width (number of elements processed in parallel)
    fn simd_width(&self) -> usize {
        1
    }
    
    /// Compute sparse weighted sum: Σ(data[indices[i]] * weights[i])
    fn sparse_weighted_sum(&self, data: &[T], indices: &[usize], weights: &[T]) -> T::Aggregate {
        debug_assert_eq!(
            indices.len(),
            weights.len(),
            "Indices and weights must have same length"
        );
        
        indices
            .iter()
            .zip(weights.iter())
            .map(|(&idx, &weight)| {
                debug_assert!(
                    idx < data.len(),
                    "Index {} out of bounds for data of length {}",
                    idx,
                    data.len()
                );
                <T::Aggregate as From<T>>::from(data[idx]) * <T::Aggregate as From<T>>::from(weight)
            })
            .fold(<T::Aggregate as Zero>::zero(), |acc, x| acc + x)
    }
    
    /// Compute dot product of two vectors
    fn dot_product(&self, a: &[T], b: &[T]) -> T::Aggregate {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| <T::Aggregate as From<T>>::from(x) * <T::Aggregate as From<T>>::from(y))
            .fold(<T::Aggregate as Zero>::zero(), |acc, x| acc + x)
    }
    
    /// Sum all elements in a slice
    fn sum(&self, data: &[T]) -> T::Aggregate {
        data.iter()
            .map(|&x| <T::Aggregate as From<T>>::from(x))
            .fold(<T::Aggregate as Zero>::zero(), |acc, x| acc + x)
    }
    
    /// Compute sum of squares
    fn sum_of_squares(&self, data: &[T]) -> T::Float {
        data.iter()
            .map(|&x| {
                let f = <T::Float as From<T>>::from(x);
                f * f
            })
            .fold(<T::Float as Zero>::zero(), |acc, x| acc + x)
    }
    
    /// Compute mean of a slice
    fn mean(&self, data: &[T]) -> T::Float {
        if data.is_empty() {
            return <T::Float as Zero>::zero();
        }
        let sum = self.sum(data);
        // Convert through f64 as intermediate step
        let sum_f64: f64 = sum.into();
        let len_f64 = data.len() as f64;
        let mean_f64 = sum_f64 / len_f64;
        // Now convert f64 to T::Float
        num_traits::NumCast::from(mean_f64).unwrap_or_else(<T::Float as Zero>::zero)
    }
    
    /// Compute variance of a slice
    fn variance(&self, data: &[T]) -> T::Float {
        if data.len() <= 1 {
            return <T::Float as Zero>::zero();
        }
        
        let mean = self.mean(data);
        let sum_sq_diff = data.iter()
            .map(|&x| {
                let diff = <T::Float as From<T>>::from(x) - mean;
                diff * diff
            })
            .fold(<T::Float as Zero>::zero(), |acc, x| acc + x);
            
        // Convert through f64 as intermediate step
        let sum_sq_diff_f64: f64 = sum_sq_diff.into();
        let divisor = (data.len() - 1) as f64;
        let variance_f64 = sum_sq_diff_f64 / divisor;
        // Now convert f64 to T::Float
        num_traits::NumCast::from(variance_f64).unwrap_or_else(<T::Float as Zero>::zero)
    }
    
    /// Find minimum value (requires T: Ord)
    fn min(&self, data: &[T]) -> Option<T> 
    where T: Ord
    {
        data.iter().copied().min()
    }
    
    /// Find maximum value (requires T: Ord)
    fn max(&self, data: &[T]) -> Option<T>
    where T: Ord
    {
        data.iter().copied().max()
    }
    
    /// Find index of minimum value  
    fn argmin(&self, data: &[T]) -> Option<usize> {
        data.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }
    
    /// Find index of maximum value
    fn argmax(&self, data: &[T]) -> Option<usize> {
        data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }
    
    /// Apply sparse tile to result vector
    /// This is specialized for sparse matrix operations
    fn apply_sparse_tile<S: crate::SparseTileData<T>>(
        &self,
        tile_data: &[T],
        tile: &S,
        result: &mut [T::Aggregate],
    ) {
        let local_rows = tile.local_rows();
        let local_cols = tile.local_cols();
        let weights = tile.weights();
        
        for i in 0..weights.len() {
            let row_offset = local_rows[i] as usize;
            let col_offset = local_cols[i] as usize;
            let weight = weights[i];
            let data_val = <T::Aggregate as From<T>>::from(tile_data[col_offset]);
            let weight_val = <T::Aggregate as From<T>>::from(weight);
            result[row_offset] += data_val * weight_val;
        }
    }
    
    /// Apply sparse tile without bounds checks
    /// 
    /// # Safety
    /// The caller must ensure:
    /// - All indices in tile are within bounds of tile_data and result
    unsafe fn apply_sparse_tile_unchecked<S: crate::SparseTileData<T>>(
        &self,
        tile_data: &[T],
        tile: &S,
        result: &mut [T::Aggregate],
    ) {
        let local_rows = tile.local_rows();
        let local_cols = tile.local_cols();
        let weights = tile.weights();
        
        for i in 0..weights.len() {
            let row_offset = *local_rows.get_unchecked(i) as usize;
            let col_offset = *local_cols.get_unchecked(i) as usize;
            let weight = *weights.get_unchecked(i);
            let data_val = <T::Aggregate as From<T>>::from(*tile_data.get_unchecked(col_offset));
            let weight_val = <T::Aggregate as From<T>>::from(weight);
            let result_ref = result.get_unchecked_mut(row_offset);
            *result_ref += data_val * weight_val;
        }
    }
}


