//! AVX2 sparse tile implementation for f64
//!
//! Since SoATileBuffer already provides optimal memory layout with:
//! - 32-byte alignment for AVX2
//! - Contiguous memory for all arrays
//! - Cache-friendly access patterns
//!
//! We only need a single, straightforward AVX2 implementation.

use crate::{primitives::backends::avx2::Avx2Backend, SparseTileData};
use std::arch::x86_64::*;

/// AVX2 implementation of sparse tile application for f64
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn apply_sparse_tile_f64<S: SparseTileData<f64>>(
    _backend: &Avx2Backend,
    tile_data: &[f64],
    tile: &S,
    result: &mut [f64],
) {
    apply_sparse_tile_avx2(tile_data, tile, result)
}

/// Unchecked version (same as checked for AVX2)
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn apply_sparse_tile_unchecked_f64<S: SparseTileData<f64>>(
    backend: &Avx2Backend,
    tile_data: &[f64],
    tile: &S,
    result: &mut [f64],
) {
    apply_sparse_tile_f64(backend, tile_data, tile, result)
}

/// AVX2 sparse tile implementation for f64
/// Uses row-grouped processing if row_starts metadata is available,
/// otherwise falls back to the standard approach
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_sparse_tile_avx2<S: SparseTileData<f64>>(
    tile_data: &[f64],
    tile: &S,
    result: &mut [f64],
) {
    // Check if row_starts metadata is available for row-grouped processing
    if let Some(row_starts) = tile.row_starts() {
        apply_sparse_tile_row_grouped_avx2(tile_data, tile, result, row_starts);
    } else {
        apply_sparse_tile_standard_avx2(tile_data, tile, result);
    }
}

/// Row-grouped AVX2 implementation - processes all entries for each row together
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_sparse_tile_row_grouped_avx2<S: SparseTileData<f64>>(
    tile_data: &[f64],
    tile: &S,
    result: &mut [f64],
    row_starts: &[u16],
) {
    let local_cols = tile.local_cols();
    let weights = tile.weights();
    
    if weights.is_empty() {
        return;
    }
    
    // Debug: Check if arrays are consistent
    if local_cols.len() != weights.len() {
        panic!("Array length mismatch: local_cols.len()={}, weights.len()={}", 
               local_cols.len(), weights.len());
    }
    
    let n_rows = row_starts.len() - 1; // Last element is sentinel
    
    
    // Process each row
    for row in 0..n_rows {
        let start = row_starts[row] as usize;
        let end = row_starts[row + 1] as usize;
        
        // Debug check
        if start > weights.len() || end > weights.len() {
            eprintln!("ERROR: Invalid row_starts values for row {}: start={}, end={}, weights.len()={}", 
                     row, start, end, weights.len());
            continue;
        }
        
        if start == end {
            continue; // Empty row
        }
        
        let n_entries = end - start;
        let chunks = n_entries / 4;
        let remainder = n_entries % 4;
        
        // Accumulate in SIMD register
        let mut sum = _mm256_setzero_pd();
        
        // Process 4 entries at a time
        for chunk in 0..chunks {
            let base = start + chunk * 4;
            
            
            // Load weights
            let weights_vec = _mm256_loadu_pd(weights.get_unchecked(base));
            
            // Gather data values
            let col0 = *local_cols.get_unchecked(base) as usize;
            let col1 = *local_cols.get_unchecked(base + 1) as usize;
            let col2 = *local_cols.get_unchecked(base + 2) as usize;
            let col3 = *local_cols.get_unchecked(base + 3) as usize;
            
            let data_vec = _mm256_set_pd(
                *tile_data.get_unchecked(col3),
                *tile_data.get_unchecked(col2),
                *tile_data.get_unchecked(col1),
                *tile_data.get_unchecked(col0),
            );
            
            // FMA: sum += data * weights
            sum = _mm256_fmadd_pd(data_vec, weights_vec, sum);
        }
        
        // Horizontal sum of the SIMD register
        let sum_scalar = horizontal_sum_pd(sum);
        
        // Handle remainder entries
        let mut remainder_sum = 0.0;
        for i in 0..remainder {
            let idx = start + chunks * 4 + i;
            let col = *local_cols.get_unchecked(idx) as usize;
            remainder_sum += *weights.get_unchecked(idx) * *tile_data.get_unchecked(col);
        }
        
        // Single write to result
        *result.get_unchecked_mut(row) += sum_scalar + remainder_sum;
    }
}

/// Horizontal sum for __m256d
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_pd(v: __m256d) -> f64 {
    // v = [a, b, c, d]
    let high = _mm256_extractf128_pd(v, 1); // [c, d]
    let low = _mm256_castpd256_pd128(v);    // [a, b]
    let sum128 = _mm_add_pd(low, high);     // [a+c, b+d]
    let high64 = _mm_unpackhi_pd(sum128, sum128); // [b+d, b+d]
    let sum = _mm_add_sd(sum128, high64);   // [a+c+b+d]
    _mm_cvtsd_f64(sum)
}

/// Standard AVX2 implementation - the original approach
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_sparse_tile_standard_avx2<S: SparseTileData<f64>>(
    tile_data: &[f64],
    tile: &S,
    result: &mut [f64],
) {
    let local_rows = tile.local_rows();
    let local_cols = tile.local_cols();
    let weights = tile.weights();
    let n = weights.len();

    if n == 0 {
        return;
    }

    let chunks = n / 4;
    let remainder = n % 4;

    // Check for special case: consecutive rows
    // This allows us to use vectorized FMA on the result
    for chunk in 0..chunks {
        let base = chunk * 4;

        let weights_vec = _mm256_loadu_pd(weights.get_unchecked(base));

        // Load row indices first to check pattern
        let row0 = *local_rows.get_unchecked(base) as usize;
        let row1 = *local_rows.get_unchecked(base + 1) as usize;
        let row2 = *local_rows.get_unchecked(base + 2) as usize;
        let row3 = *local_rows.get_unchecked(base + 3) as usize;

        // Load column indices
        let col0 = *local_cols.get_unchecked(base) as usize;
        let col1 = *local_cols.get_unchecked(base + 1) as usize;
        let col2 = *local_cols.get_unchecked(base + 2) as usize;
        let col3 = *local_cols.get_unchecked(base + 3) as usize;

        // Load data values
        let data0 = *tile_data.get_unchecked(col0);
        let data1 = *tile_data.get_unchecked(col1);
        let data2 = *tile_data.get_unchecked(col2);
        let data3 = *tile_data.get_unchecked(col3);

        let data_vec = _mm256_set_pd(data3, data2, data1, data0);

        // Check if rows are consecutive - if so, use vectorized FMA
        if row1 == row0 + 1 && row2 == row0 + 2 && row3 == row0 + 3 {
            // Load current result values
            let result_vec = _mm256_loadu_pd(result.get_unchecked(row0));
            // FMA: result = result + data * weights
            let updated = _mm256_fmadd_pd(data_vec, weights_vec, result_vec);
            _mm256_storeu_pd(result.get_unchecked_mut(row0), updated);
        } else {
            // Fall back to scalar accumulation
            let products = _mm256_mul_pd(data_vec, weights_vec);

            #[repr(align(32))]
            struct Aligned([f64; 4]);
            let mut products_arr = Aligned([0.0; 4]);
            _mm256_store_pd(products_arr.0.as_mut_ptr(), products);

            *result.get_unchecked_mut(row0) += products_arr.0[0];
            *result.get_unchecked_mut(row1) += products_arr.0[1];
            *result.get_unchecked_mut(row2) += products_arr.0[2];
            *result.get_unchecked_mut(row3) += products_arr.0[3];
        }
    }

    // Process remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let idx = base + i;
        let row_offset = *local_rows.get_unchecked(idx) as usize;
        let col_offset = *local_cols.get_unchecked(idx) as usize;
        let weight = *weights.get_unchecked(idx);
        let data_val = *tile_data.get_unchecked(col_offset);
        *result.get_unchecked_mut(row_offset) += data_val * weight;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SparseTile, TileEntry};

    // Helper to check if AVX2 is available
    fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    // Helper to create test data that's aligned properly
    fn create_aligned_vec(size: usize, value: f64) -> Vec<f64> {
        let mut v = vec![value; size];
        // Ensure alignment for AVX2
        assert_eq!(v.as_ptr() as usize % 8, 0); // f64 alignment
        v
    }

    #[test]
    fn test_horizontal_sum_pd() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        unsafe {
            // Test case 1: All same values
            let v = _mm256_set_pd(3.0, 3.0, 3.0, 3.0);
            let sum = horizontal_sum_pd(v);
            assert!((sum - 12.0).abs() < 1e-10);

            // Test case 2: Different values
            let v = _mm256_set_pd(4.0, 3.0, 2.0, 1.0);
            let sum = horizontal_sum_pd(v);
            assert!((sum - 10.0).abs() < 1e-10);

            // Test case 3: With negatives
            let v = _mm256_set_pd(5.0, -3.0, 2.0, -1.0);
            let sum = horizontal_sum_pd(v);
            assert!((sum - 3.0).abs() < 1e-10);

            // Test case 4: Zeros
            let v = _mm256_setzero_pd();
            let sum = horizontal_sum_pd(v);
            assert_eq!(sum, 0.0);
        }
    }

    #[test]
    fn test_sparse_tile_empty() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let tile_data = create_aligned_vec(16, 1.0);
        let mut result = create_aligned_vec(16, 0.0);
        
        // Empty tile
        let tile = SparseTile::<f64>::new(0, 0, 0, 4, 0, 4, vec![]);
        
        unsafe {
            apply_sparse_tile_f64(&backend, &tile_data, &tile, &mut result);
        }
        
        // Result should remain unchanged
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_sparse_tile_single_entry() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let tile_data = create_aligned_vec(4, 2.0);
        let mut result = create_aligned_vec(4, 0.0);
        
        let entries = vec![
            TileEntry { local_row: 1, local_col: 2, weight: 0.5 }
        ];
        let tile = SparseTile::<f64>::new(0, 0, 0, 4, 0, 4, entries);
        
        unsafe {
            apply_sparse_tile_f64(&backend, &tile_data, &tile, &mut result);
        }
        
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 1.0); // 2.0 * 0.5
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 0.0);
    }

    #[test]
    fn test_sparse_tile_multiple_entries_same_row() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let tile_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut result = create_aligned_vec(4, 0.0);
        
        // Multiple entries in the same row should accumulate
        let entries = vec![
            TileEntry { local_row: 1, local_col: 0, weight: 0.5 },
            TileEntry { local_row: 1, local_col: 2, weight: 0.3 },
            TileEntry { local_row: 1, local_col: 4, weight: 0.2 },
        ];
        let tile = SparseTile::<f64>::new(0, 0, 0, 4, 0, 5, entries);
        
        unsafe {
            apply_sparse_tile_f64(&backend, &tile_data, &tile, &mut result);
        }
        
        assert_eq!(result[0], 0.0);
        assert!((result[1] - 2.4).abs() < 1e-10); // 1.0*0.5 + 3.0*0.3 + 5.0*0.2
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 0.0);
    }

    #[test]
    fn test_sparse_tile_consecutive_rows() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let tile_data = vec![1.0, 2.0, 3.0, 4.0];
        let mut result = create_aligned_vec(4, 0.0);
        
        // Four consecutive entries to trigger vectorized path
        let entries = vec![
            TileEntry { local_row: 0, local_col: 0, weight: 0.1 },
            TileEntry { local_row: 1, local_col: 1, weight: 0.2 },
            TileEntry { local_row: 2, local_col: 2, weight: 0.3 },
            TileEntry { local_row: 3, local_col: 3, weight: 0.4 },
        ];
        let tile = SparseTile::<f64>::new(0, 0, 0, 4, 0, 4, entries);
        
        unsafe {
            apply_sparse_tile_f64(&backend, &tile_data, &tile, &mut result);
        }
        
        assert!((result[0] - 0.1).abs() < 1e-10); // 1.0 * 0.1
        assert!((result[1] - 0.4).abs() < 1e-10); // 2.0 * 0.2
        assert!((result[2] - 0.9).abs() < 1e-10); // 3.0 * 0.3
        assert!((result[3] - 1.6).abs() < 1e-10); // 4.0 * 0.4
    }

    #[test]
    fn test_sparse_tile_with_remainder() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let tile_data = vec![1.0; 8];
        let mut result = create_aligned_vec(8, 0.0);
        
        // 6 entries = 1 chunk of 4 + 2 remainder
        let entries = vec![
            TileEntry { local_row: 0, local_col: 0, weight: 0.1 },
            TileEntry { local_row: 1, local_col: 1, weight: 0.2 },
            TileEntry { local_row: 2, local_col: 2, weight: 0.3 },
            TileEntry { local_row: 3, local_col: 3, weight: 0.4 },
            TileEntry { local_row: 4, local_col: 4, weight: 0.5 },
            TileEntry { local_row: 5, local_col: 5, weight: 0.6 },
        ];
        let tile = SparseTile::<f64>::new(0, 0, 0, 8, 0, 8, entries);
        
        unsafe {
            apply_sparse_tile_f64(&backend, &tile_data, &tile, &mut result);
        }
        
        // Check all results
        for (i, &val) in result.iter().enumerate().take(6) {
            assert!((val - (i as f64 + 1.0) * 0.1).abs() < 1e-10);
        }
        assert_eq!(result[6], 0.0);
        assert_eq!(result[7], 0.0);
    }

    #[test]
    fn test_sparse_tile_row_grouped() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let tile_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut result = create_aligned_vec(4, 0.0);
        
        // Entries grouped by row - this should trigger row-grouped processing
        let entries = vec![
            // Row 0: 3 entries
            TileEntry { local_row: 0, local_col: 0, weight: 0.1 },
            TileEntry { local_row: 0, local_col: 2, weight: 0.2 },
            TileEntry { local_row: 0, local_col: 4, weight: 0.3 },
            // Row 1: 2 entries  
            TileEntry { local_row: 1, local_col: 1, weight: 0.4 },
            TileEntry { local_row: 1, local_col: 3, weight: 0.5 },
            // Row 2: 1 entry
            TileEntry { local_row: 2, local_col: 5, weight: 0.6 },
            // Row 3: empty
        ];
        let tile = SparseTile::<f64>::new(0, 0, 0, 4, 0, 6, entries);
        
        unsafe {
            apply_sparse_tile_f64(&backend, &tile_data, &tile, &mut result);
        }
        
        // Row 0: 1*0.1 + 3*0.2 + 5*0.3 = 0.1 + 0.6 + 1.5 = 2.2
        assert!((result[0] - 2.2).abs() < 1e-10);
        // Row 1: 2*0.4 + 4*0.5 = 0.8 + 2.0 = 2.8
        assert!((result[1] - 2.8).abs() < 1e-10);
        // Row 2: 6*0.6 = 3.6
        assert!((result[2] - 3.6).abs() < 1e-10);
        // Row 3: empty
        assert_eq!(result[3], 0.0);
    }

    #[test]
    fn test_sparse_tile_edge_cases() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        
        // Test 1: Very large weights
        let tile_data = vec![1e100; 4];
        let mut result = vec![0.0; 4];
        let entries = vec![
            TileEntry { local_row: 0, local_col: 0, weight: 1e-100 },
        ];
        let tile = SparseTile::<f64>::new(0, 0, 0, 4, 0, 4, entries);
        
        unsafe {
            apply_sparse_tile_f64(&backend, &tile_data, &tile, &mut result);
        }
        assert!((result[0] - 1.0).abs() < 1e-10); // 1e100 * 1e-100 = 1
        
        // Test 2: Negative weights
        let tile_data = vec![2.0; 4];
        let mut result = vec![1.0; 4];
        let entries = vec![
            TileEntry { local_row: 1, local_col: 1, weight: -0.5 },
        ];
        let tile = SparseTile::<f64>::new(0, 0, 0, 4, 0, 4, entries);
        
        unsafe {
            apply_sparse_tile_f64(&backend, &tile_data, &tile, &mut result);
        }
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 0.0); // 1.0 + (2.0 * -0.5) = 0
        
        // Test 3: Zero weights
        let mut result = vec![5.0; 4];
        let entries = vec![
            TileEntry { local_row: 2, local_col: 2, weight: 0.0 },
        ];
        let tile = SparseTile::<f64>::new(0, 0, 0, 4, 0, 4, entries);
        
        unsafe {
            apply_sparse_tile_f64(&backend, &tile_data, &tile, &mut result);
        }
        assert!(result.iter().all(|&x| x == 5.0)); // No change
    }

    #[test]
    fn test_sparse_tile_accumulation() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let tile_data = vec![1.0; 8];
        let mut result = vec![10.0; 4]; // Start with non-zero values
        
        let entries = vec![
            TileEntry { local_row: 0, local_col: 0, weight: 1.0 },
            TileEntry { local_row: 1, local_col: 1, weight: 2.0 },
            TileEntry { local_row: 2, local_col: 2, weight: 3.0 },
            TileEntry { local_row: 3, local_col: 3, weight: 4.0 },
        ];
        let tile = SparseTile::<f64>::new(0, 0, 0, 4, 0, 8, entries);
        
        unsafe {
            apply_sparse_tile_f64(&backend, &tile_data, &tile, &mut result);
        }
        
        // Results should accumulate on top of existing values
        assert_eq!(result[0], 11.0); // 10 + 1*1
        assert_eq!(result[1], 12.0); // 10 + 1*2
        assert_eq!(result[2], 13.0); // 10 + 1*3
        assert_eq!(result[3], 14.0); // 10 + 1*4
    }

    #[test]
    fn test_unchecked_version() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let tile_data = vec![2.0, 3.0, 4.0, 5.0];
        let mut result1 = vec![0.0; 4];
        let mut result2 = vec![0.0; 4];
        
        let entries = vec![
            TileEntry { local_row: 0, local_col: 1, weight: 0.5 },
            TileEntry { local_row: 2, local_col: 3, weight: 0.7 },
        ];
        let tile = SparseTile::<f64>::new(0, 0, 0, 4, 0, 4, entries);
        
        unsafe {
            // Both versions should produce the same result
            apply_sparse_tile_f64(&backend, &tile_data, &tile, &mut result1);
            apply_sparse_tile_unchecked_f64(&backend, &tile_data, &tile, &mut result2);
        }
        
        assert_eq!(result1, result2);
        assert_eq!(result1[0], 1.5); // 3.0 * 0.5
        assert_eq!(result1[2], 3.5); // 5.0 * 0.7
    }

    #[test]
    fn test_sparse_tile_scalar_equivalence() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        // Test that AVX2 implementation matches scalar computation
        let backend = Avx2Backend;
        let tile_data: Vec<f64> = (0..16).map(|i| i as f64 * 0.5).collect();
        let mut avx2_result = vec![0.0; 8];
        let mut scalar_result = [0.0; 8];
        
        // Create a complex pattern with various entry counts per row
        let entries = vec![
            // Row 0: 5 entries (1 chunk + 1 remainder)
            TileEntry { local_row: 0, local_col: 0, weight: 0.1 },
            TileEntry { local_row: 0, local_col: 3, weight: 0.2 },
            TileEntry { local_row: 0, local_col: 7, weight: 0.3 },
            TileEntry { local_row: 0, local_col: 11, weight: 0.4 },
            TileEntry { local_row: 0, local_col: 15, weight: 0.5 },
            // Row 1: empty
            // Row 2: 3 entries
            TileEntry { local_row: 2, local_col: 2, weight: 0.6 },
            TileEntry { local_row: 2, local_col: 5, weight: 0.7 },
            TileEntry { local_row: 2, local_col: 9, weight: 0.8 },
            // Row 3: 8 entries (2 chunks)
            TileEntry { local_row: 3, local_col: 1, weight: 0.1 },
            TileEntry { local_row: 3, local_col: 2, weight: 0.2 },
            TileEntry { local_row: 3, local_col: 4, weight: 0.3 },
            TileEntry { local_row: 3, local_col: 6, weight: 0.4 },
            TileEntry { local_row: 3, local_col: 8, weight: 0.5 },
            TileEntry { local_row: 3, local_col: 10, weight: 0.6 },
            TileEntry { local_row: 3, local_col: 12, weight: 0.7 },
            TileEntry { local_row: 3, local_col: 14, weight: 0.8 },
        ];
        
        let tile = SparseTile::<f64>::new(0, 0, 0, 8, 0, 16, entries.clone());
        
        // Compute with AVX2
        unsafe {
            apply_sparse_tile_f64(&backend, &tile_data, &tile, &mut avx2_result);
        }
        
        // Compute with scalar
        for entry in &entries {
            scalar_result[entry.local_row as usize] += 
                tile_data[entry.local_col as usize] * entry.weight;
        }
        
        // Compare results
        for i in 0..8 {
            assert!(
                (avx2_result[i] - scalar_result[i]).abs() < 1e-10,
                "Mismatch at row {}: AVX2={}, scalar={}",
                i, avx2_result[i], scalar_result[i]
            );
        }
    }
}
