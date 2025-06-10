//! Unified weight caching system for quantile estimators
//!
//! This module provides a unified cache that can handle both sparse and tiled
//! weight representations, with automatic eviction based on usage patterns.

use crate::{
    batch::{CachePolicy, CacheableComputation, ComputationCache},
    sparse::SparseWeights,
    tiled::TiledSparseMatrix,
    Numeric,
};
use ordered_float::OrderedFloat;
use std::sync::Arc;

/// Efficient representation of quantile sets using bitmaps and common patterns
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuantileSet {
    /// All percentiles [0.01, 0.02, ..., 0.99]
    AllPercentiles,
    /// All permilles [0.001, 0.002, ..., 0.999]
    AllPermilles,
    /// Quartiles [0.25, 0.50, 0.75]
    Quartiles,
    /// Deciles [0.10, 0.20, ..., 0.90]
    Deciles,

    /// Custom selection of percentiles (bitmap where bit i represents (i+1)/100)
    Percentiles(u128),

    /// High precision in a specific range
    /// E.g., range_start=900, precision=1 means [0.900, 0.901, ..., 0.999]
    HighPrecisionRange {
        /// Starting value in permilles (900 = 0.900)
        range_start: u16,
        /// Precision: 1=0.001, 10=0.01, 100=0.1
        precision: u8,
        /// Bitmap of which values in range are included
        bitmap: u128,
    },
}

impl QuantileSet {
    /// Create a QuantileSet from a slice of quantiles
    pub fn from_quantiles(quantiles: &[f64]) -> Self {
        // Check common patterns first
        if Self::is_all_percentiles(quantiles) {
            return Self::AllPercentiles;
        }
        if Self::is_all_permilles(quantiles) {
            return Self::AllPermilles;
        }
        if Self::is_quartiles(quantiles) {
            return Self::Quartiles;
        }
        if Self::is_deciles(quantiles) {
            return Self::Deciles;
        }

        // Check if all values are at 0.01 resolution
        let all_percentile_resolution = quantiles.iter().all(|&q| {
            let rounded = (q * 100.0).round() / 100.0;
            (rounded - q).abs() < 1e-9 && (0.01..=0.99).contains(&q)
        });

        if all_percentile_resolution {
            // Create bitmap for custom percentiles
            let mut bitmap = 0u128;
            for &q in quantiles {
                let idx = ((q * 100.0).round() as u32) - 1;
                if idx < 99 {
                    bitmap |= 1u128 << idx;
                }
            }
            return Self::Percentiles(bitmap);
        }

        // Try to fit in a high precision range
        if let Some(range) = Self::try_high_precision_range(quantiles) {
            return range;
        }

        // Fall back to custom percentiles, losing precision
        // In practice, this should rarely happen
        let mut bitmap = 0u128;
        for &q in quantiles {
            let idx = ((q * 100.0).round() as u32).saturating_sub(1);
            if idx < 99 {
                bitmap |= 1u128 << idx;
            }
        }
        Self::Percentiles(bitmap)
    }

    /// Convert back to a vector of quantiles
    pub fn to_quantiles(&self) -> Vec<f64> {
        match self {
            Self::AllPercentiles => (1..=99).map(|i| i as f64 / 100.0).collect(),
            Self::AllPermilles => (1..=999).map(|i| i as f64 / 1000.0).collect(),
            Self::Quartiles => vec![0.25, 0.50, 0.75],
            Self::Deciles => (1..=9).map(|i| i as f64 / 10.0).collect(),

            Self::Percentiles(bitmap) => {
                let mut quantiles = Vec::with_capacity(99);
                for i in 0..99 {
                    if (bitmap >> i) & 1 == 1 {
                        quantiles.push((i + 1) as f64 / 100.0);
                    }
                }
                quantiles
            }

            Self::HighPrecisionRange {
                range_start,
                precision,
                bitmap,
            } => {
                let mut quantiles = Vec::with_capacity(128);

                for i in 0..128 {
                    if (bitmap >> i) & 1 == 1 {
                        let value = (*range_start as f64 + i as f64 * *precision as f64) / 1000.0;
                        if value <= 1.0 {
                            quantiles.push(value);
                        }
                    }
                }
                quantiles
            }
        }
    }

    fn is_all_percentiles(quantiles: &[f64]) -> bool {
        quantiles.len() == 99
            && quantiles
                .windows(2)
                .all(|w| (w[1] - w[0] - 0.01).abs() < 1e-9)
            && (quantiles[0] - 0.01).abs() < 1e-9
    }

    fn is_all_permilles(quantiles: &[f64]) -> bool {
        quantiles.len() == 999
            && quantiles
                .windows(2)
                .all(|w| (w[1] - w[0] - 0.001).abs() < 1e-9)
            && (quantiles[0] - 0.001).abs() < 1e-9
    }

    fn is_quartiles(quantiles: &[f64]) -> bool {
        quantiles.len() == 3
            && (quantiles[0] - 0.25).abs() < 1e-9
            && (quantiles[1] - 0.50).abs() < 1e-9
            && (quantiles[2] - 0.75).abs() < 1e-9
    }

    fn is_deciles(quantiles: &[f64]) -> bool {
        quantiles.len() == 9
            && quantiles
                .iter()
                .enumerate()
                .all(|(i, &q)| ((i + 1) as f64 / 10.0 - q).abs() < 1e-9)
    }

    fn try_high_precision_range(quantiles: &[f64]) -> Option<Self> {
        if quantiles.is_empty() || quantiles.len() > 128 {
            return None;
        }

        // Find range and precision
        let min_q = quantiles[0];
        let max_q = quantiles[quantiles.len() - 1];

        // Try different precisions
        for &precision in &[1, 10] {
            let range_start = (min_q * 1000.0).floor() as u16;
            let range_start_aligned = (range_start / precision) * precision;

            // Check if all quantiles fit in 128 positions at this precision
            let max_positions =
                ((max_q * 1000.0 - range_start_aligned as f64) / precision as f64).ceil() as u32;
            if max_positions <= 128 {
                // Build bitmap
                let mut bitmap = 0u128;
                let mut all_fit = true;

                for &q in quantiles {
                    let pos = ((q * 1000.0 - range_start_aligned as f64) / precision as f64).round()
                        as i32;
                    if !(0..128).contains(&pos) {
                        all_fit = false;
                        break;
                    }
                    bitmap |= 1u128 << pos;
                }

                if all_fit {
                    return Some(Self::HighPrecisionRange {
                        range_start: range_start_aligned,
                        precision: precision as u8,
                        bitmap,
                    });
                }
            }
        }

        None
    }
}

/// Unified key for both sparse and tiled weights
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightKey {
    /// Key for sparse weights: (n, p)
    Sparse { n: usize, p: OrderedFloat<f64> },
    /// Key for tiled weights using efficient quantile representation
    Tiled { n: usize, quantiles: QuantileSet },
    /// Key for tiled weights with custom tile configuration
    TiledWithConfig {
        n: usize,
        quantiles: QuantileSet,
        tile_row_size: usize,
        tile_col_size: usize,
    },
}

/// Unified value for both sparse and tiled weights
#[derive(Clone)]
pub enum WeightValue<T: Numeric = f64> {
    Sparse(Arc<SparseWeights<T>>),
    Tiled(Arc<TiledSparseMatrix<T>>),
}

/// Trait for weight computers that can generate both representations
/// 
/// This trait is now parameterized by ExecutionEngine to enable compile-time
/// dispatch between sequential and parallel implementations.
pub trait WeightComputer<T: Numeric = f64>: Send + Sync {
    /// Compute sparse weights for a single quantile
    fn compute_sparse(&self, n: usize, p: f64) -> SparseWeights<T>;

    /// Compute sparse weights for a specific column range
    /// 
    /// This method MUST compute weights directly for the range [start, end),
    /// not compute all weights and filter. This is critical for performance
    /// when building tiled matrices.
    /// 
    /// Implementations should:
    /// - Start computation at column `start` (not 0)
    /// - Stop computation at column `end` (not n)
    /// - Only include indices in the range [start, end)
    fn compute_sparse_range(&self, n: usize, p: f64, start: usize, end: usize) -> SparseWeights<T>;

    /// Compute tiled weights for multiple quantiles
    /// 
    /// This method now accepts an ExecutionEngine parameter for compile-time dispatch.
    /// The default implementation provides backward compatibility.
    fn compute_tiled(&self, n: usize, quantiles: &[f64]) -> Arc<TiledSparseMatrix<T>> {
        // Default implementation for backward compatibility
        self.compute_tiled_with_config(n, quantiles, 32, 256)
    }

    /// Compute tiled weights with custom tile configuration
    /// Default implementation uses standard tile sizes
    fn compute_tiled_with_config(
        &self,
        n: usize,
        quantiles: &[f64],
        tile_row_size: usize,
        tile_col_size: usize,
    ) -> Arc<TiledSparseMatrix<T>> {
        // Default implementation: compute sparse weights and convert to tiled
        let sparse_rows: Vec<SparseWeights<T>> = quantiles
            .iter()
            .map(|&p| self.compute_sparse(n, p))
            .collect();

        Arc::new(TiledSparseMatrix::from_sparse_rows(
            sparse_rows,
            tile_row_size,
            tile_col_size,
        ))
    }

    /// Compute tiled weights with execution engine for optimal performance
    /// 
    /// This method provides compile-time dispatch between sequential and parallel
    /// implementations based on the engine's capabilities.
    fn compute_tiled_with_engine<E: crate::execution::ExecutionEngine<T>>(
        &self,
        n: usize,
        quantiles: &[f64],
        engine: &E,
        tile_row_size: usize,
        tile_col_size: usize,
    ) -> Arc<TiledSparseMatrix<T>>
    where
        T: num_traits::NumCast,
    {
        
        // Use compile-time dispatch based on engine capabilities
        if E::IS_SEQUENTIAL || quantiles.len() < 4 || n < 100 {
            // Sequential implementation
            self.compute_tiled_sequential_impl(n, quantiles, tile_row_size, tile_col_size)
        } else {
            // Parallel implementation
            self.compute_tiled_parallel_impl(n, quantiles, engine, tile_row_size, tile_col_size)
        }
    }

    /// Internal sequential implementation
    fn compute_tiled_sequential_impl(
        &self,
        n: usize,
        quantiles: &[f64],
        tile_row_size: usize,
        tile_col_size: usize,
    ) -> Arc<TiledSparseMatrix<T>>
    where
        T: num_traits::NumCast,
    {
        let n_rows = quantiles.len();
        let n_cols = n;
        
        let n_tile_rows = n_rows.div_ceil(tile_row_size);
        let n_tile_cols = n_cols.div_ceil(tile_col_size);
        
        let mut tiles = Vec::with_capacity(n_tile_rows * n_tile_cols);
        
        // Process each tile sequentially
        for tile_row in 0..n_tile_rows {
            for tile_col in 0..n_tile_cols {
                let row_start = tile_row * tile_row_size;
                let row_end = ((tile_row + 1) * tile_row_size).min(n_rows);
                let col_start = tile_col * tile_col_size;
                let col_end = ((tile_col + 1) * tile_col_size).min(n_cols);
                
                let mut temp_entries = Vec::new();
                
                for (local_row, &quantile) in quantiles[row_start..row_end].iter().enumerate() {
                    let weights = self.compute_sparse_range(n, quantile, col_start, col_end);
                    
                    for (&col, &weight) in weights.indices.iter().zip(&weights.weights) {
                        temp_entries.push((local_row as u16, (col - col_start) as u16, weight));
                    }
                }
                
                if !temp_entries.is_empty() {
                    let tile_n_rows = row_end - row_start;
                    let mut buffer = crate::tiled::SoaTileBuffer::new(temp_entries.len(), tile_n_rows);
                    
                    // Fill buffer directly
                    unsafe {
                        let rows_ptr = buffer.local_rows_mut().as_mut_ptr();
                        let cols_ptr = buffer.local_cols_mut().as_mut_ptr();
                        let weights_ptr: *mut T = buffer.weights_mut().as_mut_ptr();
                        
                        for (i, &(row, col, weight)) in temp_entries.iter().enumerate() {
                            *rows_ptr.add(i) = row;
                            *cols_ptr.add(i) = col;
                            *weights_ptr.add(i) = weight;
                        }
                    }
                    
                    buffer.build_row_starts();
                    
                    tiles.push(crate::tiled::SparseTile::from_buffer(
                        tile_row,
                        tile_col,
                        row_start,
                        row_end,
                        col_start,
                        col_end,
                        buffer,
                    ));
                }
            }
        }
        
        let mut matrix = TiledSparseMatrix::new(
            n_rows,
            n_cols,
            tile_row_size,
            tile_col_size,
            tiles,
        );
        
        // Normalize rows to sum to 1.0
        matrix.normalize_rows();
        Arc::new(matrix)
    }

    /// Internal parallel implementation
    fn compute_tiled_parallel_impl<E: crate::execution::ExecutionEngine<T>>(
        &self,
        n: usize,
        quantiles: &[f64],
        engine: &E,
        tile_row_size: usize,
        tile_col_size: usize,
    ) -> Arc<TiledSparseMatrix<T>>
    where
        T: num_traits::NumCast,
    {
        let n_rows = quantiles.len();
        let n_cols = n;
        
        let n_tile_rows = n_rows.div_ceil(tile_row_size);
        let n_tile_cols = n_cols.div_ceil(tile_col_size);
        let total_tiles = n_tile_rows * n_tile_cols;
        
        // Use the execution engine to parallelize tile computation
        let tiles: Vec<Option<crate::tiled::SparseTile<T>>> = engine.execute_batch(total_tiles, |tile_idx| {
            let tile_row = tile_idx / n_tile_cols;
            let tile_col = tile_idx % n_tile_cols;
            
            let row_start = tile_row * tile_row_size;
            let row_end = ((tile_row + 1) * tile_row_size).min(n_rows);
            let col_start = tile_col * tile_col_size;
            let col_end = ((tile_col + 1) * tile_col_size).min(n_cols);
            
            let mut temp_entries = Vec::new();
            
            for (local_row, &quantile) in quantiles[row_start..row_end].iter().enumerate() {
                let weights = self.compute_sparse_range(n, quantile, col_start, col_end);
                
                for (&col, &weight) in weights.indices.iter().zip(&weights.weights) {
                    // The indices from compute_sparse_range should already be absolute indices
                    // We need to make them tile-local
                    debug_assert!(
                        col >= col_start && col < col_end,
                        "Index {} is outside tile column range [{}, {})",
                        col, col_start, col_end
                    );
                    temp_entries.push((local_row as u16, (col - col_start) as u16, weight));
                }
            }
            
            if temp_entries.is_empty() {
                None
            } else {
                let tile_n_rows = row_end - row_start;
                let mut buffer = crate::tiled::SoaTileBuffer::new(temp_entries.len(), tile_n_rows);
                
                // Fill buffer directly
                unsafe {
                    let rows_ptr = buffer.local_rows_mut().as_mut_ptr();
                    let cols_ptr = buffer.local_cols_mut().as_mut_ptr();
                    let weights_ptr: *mut T = buffer.weights_mut().as_mut_ptr();
                    
                    for (i, &(row, col, weight)) in temp_entries.iter().enumerate() {
                        *rows_ptr.add(i) = row;
                        *cols_ptr.add(i) = col;
                        *weights_ptr.add(i) = weight;
                    }
                }
                
                buffer.build_row_starts();
                
                Some(crate::tiled::SparseTile::from_buffer(
                    tile_row,
                    tile_col,
                    row_start,
                    row_end,
                    col_start,
                    col_end,
                    buffer,
                ))
            }
        });
        
        let tiles: Vec<crate::tiled::SparseTile<T>> = tiles.into_iter().flatten().collect();
        
        let mut matrix = TiledSparseMatrix::new(
            n_rows,
            n_cols,
            tile_row_size,
            tile_col_size,
            tiles,
        );
        
        // Normalize rows to sum to 1.0
        matrix.normalize_rows();
        Arc::new(matrix)
    }
}

/// Wrapper to adapt WeightComputer to CacheableComputation
#[derive(Clone)]
pub struct WeightComputerAdapter<C, T>
where
    C: WeightComputer<T>,
    T: Numeric,
{
    computer: C,
    _phantom: std::marker::PhantomData<T>,
}

impl<C, T> WeightComputerAdapter<C, T>
where
    C: WeightComputer<T>,
    T: Numeric,
{
    pub fn new(computer: C) -> Self {
        Self { 
            computer,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<C, T> CacheableComputation<T> for WeightComputerAdapter<C, T>
where
    C: WeightComputer<T>,
    T: Numeric + num_traits::NumCast,
{
    type CacheKey = WeightKey;
    type CacheValue = WeightValue<T>;

    fn compute_cached_value<E: crate::execution::ExecutionEngine<T>>(
        &self, 
        key: &Self::CacheKey,
        engine: &E,
    ) -> Self::CacheValue {
        match key {
            WeightKey::Sparse { n, p } => {
                WeightValue::Sparse(Arc::new(self.computer.compute_sparse(*n, p.0)))
            }
            WeightKey::Tiled { n, quantiles } => {
                let q_vec = quantiles.to_quantiles();
                WeightValue::Tiled(self.computer.compute_tiled_with_engine(*n, &q_vec, engine, 32, 256))
            }
            WeightKey::TiledWithConfig {
                n,
                quantiles,
                tile_row_size,
                tile_col_size,
            } => {
                let q_vec = quantiles.to_quantiles();
                WeightValue::Tiled(self.computer.compute_tiled_with_engine(
                    *n,
                    &q_vec,
                    engine,
                    *tile_row_size,
                    *tile_col_size,
                ))
            }
        }
    }
}

/// Unified weight cache that handles both sparse and tiled representations
pub struct UnifiedWeightCache<C, T = f64>
where
    C: WeightComputer<T>,
    T: Numeric + num_traits::NumCast,
{
    cache: ComputationCache<WeightComputerAdapter<C, T>, T>,
    adapter: WeightComputerAdapter<C, T>,
    _phantom: std::marker::PhantomData<T>,
}

impl<C, T> UnifiedWeightCache<C, T>
where
    C: WeightComputer<T> + Clone + 'static,
    T: Numeric + num_traits::NumCast + 'static,
{
    /// Create a new unified cache with the given computer and cache policy
    pub fn new(computer: C, policy: CachePolicy) -> Self {
        let adapter = WeightComputerAdapter::new(computer);
        let cache = ComputationCache::new(policy);

        Self { 
            cache, 
            adapter,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get or compute sparse weights with execution engine
    pub fn get_sparse_with_engine<E: crate::execution::ExecutionEngine<T>>(
        &self, 
        n: usize, 
        p: f64,
        engine: &E,
    ) -> Arc<SparseWeights<T>> {
        let key = WeightKey::Sparse {
            n,
            p: OrderedFloat(p),
        };
        
        let value = self.cache.get_or_compute_with_engine(key, engine, |k, e| {
            self.adapter.compute_cached_value(k, e)
        });

        match value.as_ref() {
            WeightValue::Sparse(weights) => weights.clone(),
            _ => unreachable!("Cache returned wrong type"),
        }
    }

    

    /// Get or compute tiled weights with execution engine for optimal performance
    /// 
    /// This method allows using different execution engines with the same cache,
    /// providing compile-time optimization for sequential vs parallel execution.
    pub fn get_tiled_with_engine<E: crate::execution::ExecutionEngine<T>>(
        &self,
        n: usize,
        quantiles: &[f64],
        engine: &E,
        tile_row_size: usize,
        tile_col_size: usize,
    ) -> Arc<TiledSparseMatrix<T>>
    where
        C: WeightComputer<T>,
        T: num_traits::NumCast,
    {
        let quantile_set = QuantileSet::from_quantiles(quantiles);
        let key = WeightKey::TiledWithConfig {
            n,
            quantiles: quantile_set,
            tile_row_size,
            tile_col_size,
        };
        
        let value = self.cache.get_or_compute_with_engine(key, engine, |k, e| {
            self.adapter.compute_cached_value(k, e)
        });
        
        match value.as_ref() {
            WeightValue::Tiled(matrix) => matrix.clone(),
            _ => unreachable!("Type mismatch in cache"),
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> crate::batch::CacheStats {
        self.cache.stats()
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.clear()
    }
}

impl<C, T> Clone for UnifiedWeightCache<C, T>
where
    C: WeightComputer<T> + Clone,
    T: Numeric + num_traits::NumCast,
{
    fn clone(&self) -> Self {
        Self {
            cache: self.cache.clone(),
            adapter: self.adapter.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantile_set_common_patterns() {
        // Test all percentiles
        let quantiles: Vec<f64> = (1..=99).map(|i| i as f64 / 100.0).collect();
        assert_eq!(
            QuantileSet::from_quantiles(&quantiles),
            QuantileSet::AllPercentiles
        );

        // Test quartiles
        let quantiles = vec![0.25, 0.50, 0.75];
        assert_eq!(
            QuantileSet::from_quantiles(&quantiles),
            QuantileSet::Quartiles
        );

        // Test deciles
        let quantiles: Vec<f64> = (1..=9).map(|i| i as f64 / 10.0).collect();
        assert_eq!(
            QuantileSet::from_quantiles(&quantiles),
            QuantileSet::Deciles
        );
    }

    #[test]
    fn test_quantile_set_custom_percentiles() {
        let quantiles = vec![0.05, 0.25, 0.50, 0.75, 0.95];
        let set = QuantileSet::from_quantiles(&quantiles);

        match set {
            QuantileSet::Percentiles(bitmap) => {
                // Check that bits 4, 24, 49, 74, 94 are set
                assert!((bitmap >> 4) & 1 == 1);
                assert!((bitmap >> 24) & 1 == 1);
                assert!((bitmap >> 49) & 1 == 1);
                assert!((bitmap >> 74) & 1 == 1);
                assert!((bitmap >> 94) & 1 == 1);
            }
            _ => panic!("Expected Percentiles variant"),
        }

        // Round trip
        let recovered = set.to_quantiles();
        assert_eq!(recovered, quantiles);
    }

    #[test]
    fn test_quantile_set_high_precision() {
        let quantiles = vec![0.900, 0.950, 0.990, 0.995, 0.999];
        let set = QuantileSet::from_quantiles(&quantiles);

        match set {
            QuantileSet::HighPrecisionRange {
                range_start,
                precision,
                bitmap,
            } => {
                assert_eq!(range_start, 900);
                assert_eq!(precision, 1);
                // Check specific bits
                assert!((bitmap & 1) == 1); // 0.900
                assert!(((bitmap >> 50) & 1) == 1); // 0.950
                assert!(((bitmap >> 90) & 1) == 1); // 0.990
                assert!(((bitmap >> 95) & 1) == 1); // 0.995
                assert!(((bitmap >> 99) & 1) == 1); // 0.999
            }
            _ => panic!("Expected HighPrecisionRange variant"),
        }
    }
}
