//! AVX2 sparse tile implementation for f32
//!
//! Since SoATileBuffer already provides optimal memory layout with:
//! - 32-byte alignment for AVX2
//! - Contiguous memory for all arrays
//! - Cache-friendly access patterns
//!
//! We only need a single, straightforward AVX2 implementation.

use crate::{SparseTileData, primitives::backends::avx2::Avx2Backend};
use std::arch::x86_64::*;

/// AVX2 implementation of sparse tile application for f32
#[target_feature(enable = "avx2")]
pub unsafe fn apply_sparse_tile_f32<S: SparseTileData<f32>>(
    _backend: &Avx2Backend,
    tile_data: &[f32],
    tile: &S,
    result: &mut [f64],  // f32::Aggregate = f64
) {
    apply_sparse_tile_avx2(tile_data, tile, result)
}

/// Unchecked version (same as checked for AVX2)
#[target_feature(enable = "avx2")]
pub unsafe fn apply_sparse_tile_unchecked_f32<S: SparseTileData<f32>>(
    backend: &Avx2Backend,
    tile_data: &[f32],
    tile: &S,
    result: &mut [f64],  // f32::Aggregate = f64
) {
    apply_sparse_tile_f32(backend, tile_data, tile, result)
}

/// AVX2 sparse tile implementation for f32
/// Processes entries in chunks of 8, leveraging SoATileBuffer's aligned layout
#[target_feature(enable = "avx2")]
unsafe fn apply_sparse_tile_avx2<S: SparseTileData<f32>>(
    tile_data: &[f32],
    tile: &S,
    result: &mut [f64],  // f32::Aggregate = f64
) {
    let local_rows = tile.local_rows();
    let local_cols = tile.local_cols();
    let weights = tile.weights();
    let n = weights.len();

    if n == 0 {
        return;
    }

    let chunks = n / 8;
    let remainder = n % 8;

    // Process 8 elements at a time
    for chunk in 0..chunks {
        let base = chunk * 8;

        // Load 8 weights
        let weights_vec = _mm256_loadu_ps(weights.get_unchecked(base));

        // Load column indices and check if consecutive
        let col0 = *local_cols.get_unchecked(base) as usize;
        let mut consecutive_cols = true;
        for i in 1..8 {
            if *local_cols.get_unchecked(base + i) as usize != col0 + i {
                consecutive_cols = false;
                break;
            }
        }

        let data_vec = if consecutive_cols {
            // Load consecutive data values
            _mm256_loadu_ps(tile_data.get_unchecked(col0))
        } else {
            // Gather scattered data values
            let data = [
                *tile_data.get_unchecked(*local_cols.get_unchecked(base) as usize),
                *tile_data.get_unchecked(*local_cols.get_unchecked(base + 1) as usize),
                *tile_data.get_unchecked(*local_cols.get_unchecked(base + 2) as usize),
                *tile_data.get_unchecked(*local_cols.get_unchecked(base + 3) as usize),
                *tile_data.get_unchecked(*local_cols.get_unchecked(base + 4) as usize),
                *tile_data.get_unchecked(*local_cols.get_unchecked(base + 5) as usize),
                *tile_data.get_unchecked(*local_cols.get_unchecked(base + 6) as usize),
                *tile_data.get_unchecked(*local_cols.get_unchecked(base + 7) as usize),
            ];
            _mm256_loadu_ps(data.as_ptr())
        };

        // Multiply data by weights
        let products = _mm256_mul_ps(data_vec, weights_vec);

        // Extract products and update result
        #[repr(align(32))]
        struct Aligned([f32; 8]);
        let mut products_arr = Aligned([0.0; 8]);
        _mm256_store_ps(products_arr.0.as_mut_ptr(), products);

        // Convert f32 products to f64 and accumulate
        for i in 0..8 {
            let row = *local_rows.get_unchecked(base + i) as usize;
            *result.get_unchecked_mut(row) += products_arr.0[i] as f64;
        }
    }

    // Process remainder
    for i in 0..remainder {
        let idx = chunks * 8 + i;
        let row = *local_rows.get_unchecked(idx) as usize;
        let col = *local_cols.get_unchecked(idx) as usize;
        let weight = *weights.get_unchecked(idx);
        let data = *tile_data.get_unchecked(col);
        *result.get_unchecked_mut(row) += (data * weight) as f64;
    }
}