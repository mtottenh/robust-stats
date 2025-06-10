//! Scalar fallback implementation for sparse tile operations

use crate::{Numeric, SparseTileData};

/// Scalar implementation of sparse tile application
pub fn apply_sparse_tile_scalar<T: Numeric, S: SparseTileData<T>>(
    tile_data: &[T],
    tile: &S,
    result: &mut [T::Aggregate],
) {
    let local_rows = tile.local_rows();
    let local_cols = tile.local_cols();
    let weights = tile.weights();
    let n = weights.len();
    
    for i in 0..n {
        let row = local_rows[i] as usize;
        let col = local_cols[i] as usize;
        
        if row < result.len() && col < tile_data.len() {
            // Convert T to T::Aggregate using From trait
            let data_val = T::Aggregate::from(tile_data[col]);
            let weight_val = T::Aggregate::from(weights[i]);
            let product = data_val * weight_val;
            result[row] += product;
        }
    }
}