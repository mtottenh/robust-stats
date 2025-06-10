//! Scalar fallback implementation for sparse weighted sum

use crate::Numeric;

/// Scalar implementation of sparse weighted sum
pub fn sparse_weighted_sum_scalar<T: Numeric>(
    data: &[T],
    indices: &[usize],
    weights: &[T],
) -> T {
    let n = indices.len().min(weights.len());
    let mut sum = <T as Numeric>::zero();
    
    for i in 0..n {
        let idx = indices[i];
        if idx < data.len() {
            sum = sum + data[idx] * weights[i];
        }
    }
    
    sum
}