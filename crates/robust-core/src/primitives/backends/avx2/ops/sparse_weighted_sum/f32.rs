//! AVX2 sparse weighted sum implementation for f32

use crate::primitives::backends::avx2::Avx2Backend;

/// AVX2 implementation of sparse weighted sum for f32
#[target_feature(enable = "avx2")]
pub unsafe fn sparse_weighted_sum_f32(
    _backend: &Avx2Backend,
    data: &[f32],
    indices: &[usize],
    weights: &[f32],
) -> f64 {
    // TODO: Implement AVX2 version for f32
    // For now, use scalar fallback
    super::scalar::sparse_weighted_sum_scalar(data, indices, weights) as f64
}