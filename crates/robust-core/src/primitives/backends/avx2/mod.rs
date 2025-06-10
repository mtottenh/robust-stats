//! AVX2 backend implementation with modular operation-centric organization
//!
//! This module provides AVX2-optimized implementations with compile-time
//! type dispatch to eliminate runtime overhead. Operations are organized
//! by primitive type for better modularity and testing.

mod dispatch;
mod ops;
mod utils;

use crate::primitives::ComputePrimitives;
use crate::Numeric;

/// AVX2 backend for x86_64 processors
#[derive(Clone, Copy, Debug)]
pub struct Avx2Backend;

impl Avx2Backend {
    /// Create a new AVX2 backend
    ///
    /// # Panics
    /// Panics if the CPU doesn't support AVX2 instructions
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        {
            if !is_x86_feature_detected!("avx2") {
                panic!("AVX2 backend requested but CPU doesn't support AVX2 instructions");
            }
            Self
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
        {
            panic!("AVX2 backend not available: not compiled with AVX2 support");
        }
    }

    /// Check if AVX2 is available on this CPU
    pub fn is_available() -> bool {
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
        {
            false
        }
    }
}

// AVX2 implementations delegate to type-specific modules via compile-time dispatch
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
impl<T> ComputePrimitives<T> for Avx2Backend 
where 
    T: Numeric + self::dispatch::Avx2TypeDispatch
{
    fn backend_name(&self) -> &'static str {
        T::backend_name()
    }

    fn simd_width(&self) -> usize {
        T::simd_width()
    }

    fn sparse_weighted_sum(&self, data: &[T], indices: &[usize], weights: &[T]) -> T::Aggregate {
        // Safety: We checked CPU support in new()
        unsafe { T::sparse_weighted_sum_impl(self, data, indices, weights) }
    }

    fn dot_product(&self, a: &[T], b: &[T]) -> T::Aggregate {
        // Safety: We checked CPU support in new()
        unsafe { T::dot_product_impl(self, a, b) }
    }

    fn sum(&self, data: &[T]) -> T::Aggregate {
        // Safety: We checked CPU support in new()
        unsafe { T::sum_impl(self, data) }
    }

    fn apply_sparse_tile<S: crate::SparseTileData<T>>(
        &self,
        tile_data: &[T],
        tile: &S,
        result: &mut [T::Aggregate],
    ) {
        // Safety: We checked CPU support in new()
        unsafe { T::apply_sparse_tile_impl(self, tile_data, tile, result) }
    }

    unsafe fn apply_sparse_tile_unchecked<S: crate::SparseTileData<T>>(
        &self,
        tile_data: &[T],
        tile: &S,
        result: &mut [T::Aggregate],
    ) {
        T::apply_sparse_tile_unchecked_impl(self, tile_data, tile, result)
    }
}

// Fallback for non-AVX2 builds
#[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
impl<T: Numeric> ComputePrimitives<T> for Avx2Backend {
    fn backend_name(&self) -> &'static str {
        "avx2 (unavailable)"
    }
}