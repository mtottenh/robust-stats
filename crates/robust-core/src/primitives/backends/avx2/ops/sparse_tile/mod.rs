//! Sparse tile operations for AVX2 backend
//!
//! Provides optimized sparse matrix-vector multiplication operations
//! with support for different memory access patterns

mod f32;
mod f64;
mod scalar;

use crate::{Numeric, SparseTileData, primitives::backends::avx2::Avx2Backend};

/// Trait for sparse tile operations
pub trait SparseTile<T: Numeric> {
    /// Apply sparse tile with AVX2 optimizations
    unsafe fn apply<S: SparseTileData<T>>(
        backend: &Avx2Backend,
        tile_data: &[T],
        tile: &S,
        result: &mut [T::Aggregate],
    );
    
    /// Apply sparse tile unchecked (no bounds checking)
    unsafe fn apply_unchecked<S: SparseTileData<T>>(
        backend: &Avx2Backend,
        tile_data: &[T],
        tile: &S,
        result: &mut [T::Aggregate],
    );
}

// Type-specific implementations
impl SparseTile<f32> for f32 {
    unsafe fn apply<S: SparseTileData<f32>>(
        backend: &Avx2Backend,
        tile_data: &[f32],
        tile: &S,
        result: &mut [f64],
    ) {
        f32::apply_sparse_tile_f32(backend, tile_data, tile, result)
    }
    
    unsafe fn apply_unchecked<S: SparseTileData<f32>>(
        backend: &Avx2Backend,
        tile_data: &[f32],
        tile: &S,
        result: &mut [f64],
    ) {
        f32::apply_sparse_tile_unchecked_f32(backend, tile_data, tile, result)
    }
}

impl SparseTile<f64> for f64 {
    unsafe fn apply<S: SparseTileData<f64>>(
        backend: &Avx2Backend,
        tile_data: &[f64],
        tile: &S,
        result: &mut [f64],
    ) {
        f64::apply_sparse_tile_f64(backend, tile_data, tile, result)
    }
    
    unsafe fn apply_unchecked<S: SparseTileData<f64>>(
        backend: &Avx2Backend,
        tile_data: &[f64],
        tile: &S,
        result: &mut [f64],
    ) {
        f64::apply_sparse_tile_unchecked_f64(backend, tile_data, tile, result)
    }
}

// Generic fallback for other types
macro_rules! impl_sparse_tile_fallback {
    ($type:ty) => {
        impl SparseTile<$type> for $type {
            unsafe fn apply<S: SparseTileData<$type>>(
                _backend: &Avx2Backend,
                tile_data: &[$type],
                tile: &S,
                result: &mut [<$type as Numeric>::Aggregate],
            ) {
                scalar::apply_sparse_tile_scalar(tile_data, tile, result)
            }
            
            unsafe fn apply_unchecked<S: SparseTileData<$type>>(
                _backend: &Avx2Backend,
                tile_data: &[$type],
                tile: &S,
                result: &mut [<$type as Numeric>::Aggregate],
            ) {
                scalar::apply_sparse_tile_scalar(tile_data, tile, result)
            }
        }
    };
}

impl_sparse_tile_fallback!(i32);
impl_sparse_tile_fallback!(i64);
impl_sparse_tile_fallback!(u32);
impl_sparse_tile_fallback!(u64);