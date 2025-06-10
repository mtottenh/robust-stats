//! Sparse weighted sum operation for AVX2 backend
//!
//! Computes the weighted sum of sparse data: sum(data[indices[i]] * weights[i])

mod f32;
mod f64;
mod scalar;

use crate::{Numeric, primitives::backends::avx2::Avx2Backend};

/// Trait for sparse weighted sum operation
pub trait SparseWeightedSum<T: Numeric> {
    /// Compute sparse weighted sum with AVX2 optimizations
    unsafe fn compute(
        backend: &Avx2Backend,
        data: &[T],
        indices: &[usize],
        weights: &[T],
    ) -> T::Aggregate;
}

// Type-specific implementations
impl SparseWeightedSum<f32> for f32 {
    unsafe fn compute(
        backend: &Avx2Backend,
        data: &[f32],
        indices: &[usize],
        weights: &[f32],
    ) -> f64 {
        f32::sparse_weighted_sum_f32(backend, data, indices, weights)
    }
}

impl SparseWeightedSum<f64> for f64 {
    unsafe fn compute(
        backend: &Avx2Backend,
        data: &[f64],
        indices: &[usize],
        weights: &[f64],
    ) -> f64 {
        f64::sparse_weighted_sum_f64(backend, data, indices, weights)
    }
}

// Generic fallback for other types
macro_rules! impl_sparse_weighted_sum_fallback {
    ($type:ty) => {
        impl SparseWeightedSum<$type> for $type {
            unsafe fn compute(
                _backend: &Avx2Backend,
                data: &[$type],
                indices: &[usize],
                weights: &[$type],
            ) -> <$type as Numeric>::Aggregate {
                let result = scalar::sparse_weighted_sum_scalar(data, indices, weights);
                <$type as Numeric>::Aggregate::from(result)
            }
        }
    };
}

impl_sparse_weighted_sum_fallback!(i32);
impl_sparse_weighted_sum_fallback!(i64);
impl_sparse_weighted_sum_fallback!(u32);
impl_sparse_weighted_sum_fallback!(u64);