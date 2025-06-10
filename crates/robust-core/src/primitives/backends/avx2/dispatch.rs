//! Compile-time dispatch system for AVX2 type-specific implementations
//!
//! This module provides the trait for zero-overhead type dispatch
//! using operation-centric modules and stable Rust features.

use crate::{Numeric, SparseTileData};
use super::Avx2Backend;
use super::ops::{SparseWeightedSum, DotProduct, Sum, SparseTile};

/// Trait for type-specific AVX2 dispatch
/// 
/// This trait is implemented for each supported type, allowing compile-time
/// dispatch to type-specific implementations without runtime overhead.
pub trait Avx2TypeDispatch: Numeric + SparseWeightedSum<Self> + DotProduct<Self> + Sum<Self> + SparseTile<Self> {
    fn backend_name() -> &'static str;
    fn simd_width() -> usize;
    
    unsafe fn sparse_weighted_sum_impl(
        backend: &Avx2Backend,
        data: &[Self],
        indices: &[usize],
        weights: &[Self],
    ) -> Self::Aggregate {
        <Self as SparseWeightedSum<Self>>::compute(backend, data, indices, weights)
    }
    
    unsafe fn dot_product_impl(
        backend: &Avx2Backend,
        a: &[Self],
        b: &[Self],
    ) -> Self::Aggregate {
        <Self as DotProduct<Self>>::compute(backend, a, b)
    }
    
    unsafe fn sum_impl(
        backend: &Avx2Backend,
        data: &[Self],
    ) -> Self::Aggregate {
        <Self as Sum<Self>>::compute(backend, data)
    }
    
    unsafe fn apply_sparse_tile_impl<S: SparseTileData<Self>>(
        backend: &Avx2Backend,
        tile_data: &[Self],
        tile: &S,
        result: &mut [Self::Aggregate],
    ) {
        <Self as SparseTile<Self>>::apply(backend, tile_data, tile, result)
    }
    
    unsafe fn apply_sparse_tile_unchecked_impl<S: SparseTileData<Self>>(
        backend: &Avx2Backend,
        tile_data: &[Self],
        tile: &S,
        result: &mut [Self::Aggregate],
    ) {
        <Self as SparseTile<Self>>::apply_unchecked(backend, tile_data, tile, result)
    }
}

// Implement for all numeric types
impl Avx2TypeDispatch for f32 {
    fn backend_name() -> &'static str {
        "avx2"
    }
    
    fn simd_width() -> usize {
        8 // AVX2 processes 8 f32s at once
    }
}

impl Avx2TypeDispatch for f64 {
    fn backend_name() -> &'static str {
        "avx2"
    }
    
    fn simd_width() -> usize {
        4 // AVX2 processes 4 f64s at once
    }
}

// Integer types use scalar fallback
impl Avx2TypeDispatch for i32 {
    fn backend_name() -> &'static str {
        "avx2 (scalar fallback)"
    }
    
    fn simd_width() -> usize {
        1 // Scalar fallback
    }
}

impl Avx2TypeDispatch for i64 {
    fn backend_name() -> &'static str {
        "avx2 (scalar fallback)"
    }
    
    fn simd_width() -> usize {
        1 // Scalar fallback
    }
}

impl Avx2TypeDispatch for u32 {
    fn backend_name() -> &'static str {
        "avx2 (scalar fallback)"
    }
    
    fn simd_width() -> usize {
        1 // Scalar fallback
    }
}

impl Avx2TypeDispatch for u64 {
    fn backend_name() -> &'static str {
        "avx2 (scalar fallback)"
    }
    
    fn simd_width() -> usize {
        1 // Scalar fallback
    }
}