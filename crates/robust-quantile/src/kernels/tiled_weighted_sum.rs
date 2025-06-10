//! Tiled weighted sum kernel for efficient multi-quantile computation

use robust_core::{
    ComputePrimitives, ExecutionEngine, SparseTile, StatisticalKernel, TiledSparseMatrix, Numeric,
};
use std::marker::PhantomData;
use num_traits::{Zero, NumCast};

/// Strategy for parallel tile processing
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TileProcessingStrategy {
    /// Parallelize over tiles (good when many tiles, fewer datasets)
    TileMajor,
    /// Parallelize over dataset chunks (good when many datasets, fewer tiles)
    DatasetMajor,
    /// Automatically choose based on workload characteristics
    Auto,
}

/// Kernel for tiled weighted sum operations
///
/// This kernel is optimized for computing multiple quantiles simultaneously
/// using tiled sparse matrix representations. It processes data in cache-friendly
/// tiles and can leverage SIMD operations for processing multiple datasets.
#[derive(Clone)]
pub struct TiledWeightedSumKernel<T: Numeric, E: ExecutionEngine<T>> {
    engine: E,
    _phantom: PhantomData<T>,
}

impl<T: Numeric, E: ExecutionEngine<T>> TiledWeightedSumKernel<T, E> {
    /// Create new tiled kernel
    pub fn new(engine: E) -> Self {
        Self { engine, _phantom: PhantomData }
    }

    /// Process a single tile for multiple datasets
    ///
    /// This is the unit of work that can be parallelized.
    /// The kernel processes one tile at a time, updating results in place.
    pub fn apply_single_tile(
        &self,
        datasets: &[&[T]],
        tile: &SparseTile<T>,
        results: &mut [Vec<T::Float>],
    ) {
        // Use the primitives' tiled operations for better performance
        let primitives = self.primitives();
        
        // For each dataset
        for (dataset_idx, &dataset) in datasets.iter().enumerate() {
            // Extract the relevant portion of the dataset for this tile
            let tile_data = &dataset[tile.col_start..tile.col_end];
            
            // Create temporary buffer for aggregate results
            let tile_rows = tile.row_end - tile.row_start;
            let mut aggregate_buffer = vec![<T::Aggregate as Zero>::zero(); tile_rows];
            
            // Use optimized tile processing from primitives
            primitives.apply_sparse_tile(tile_data, tile, &mut aggregate_buffer);
            
            // Convert aggregate results to float and add to existing results
            let result_slice = &mut results[dataset_idx][tile.row_start..tile.row_end];
            for (i, &agg_val) in aggregate_buffer.iter().enumerate() {
                // Convert aggregate to float and add to result
                let float_val = <T::Float as NumCast>::from(agg_val.into()).unwrap();
                result_slice[i] += float_val;
            }
        }
    }

    /// Process a range of tiles
    ///
    /// This method is called by the execution engine to process
    /// a chunk of tiles assigned to a specific thread.
    pub fn apply_tile_range(
        &self,
        datasets: &[&[T]],
        tiles: &[SparseTile<T>],
        results: &mut [Vec<T::Float>],
    ) {
        for tile in tiles {
            self.apply_single_tile(datasets, tile, results);
        }
    }

    /// Process all tiles for a subset of datasets
    ///
    /// This is used when parallelizing over datasets rather than tiles.
    pub fn apply_all_tiles_to_datasets(
        &self,
        datasets: &[&[T]],
        tiled_weights: &TiledSparseMatrix<T>,
    ) -> Vec<Vec<T::Float>> {
        let mut results = vec![vec![<T::Float as Zero>::zero(); tiled_weights.n_rows]; datasets.len()];

        // Process each tile
        for tile in &tiled_weights.tiles {
            self.apply_single_tile(datasets, tile, &mut results);
        }

        results
    }
}

impl<T: Numeric, E: ExecutionEngine<T>> StatisticalKernel<T> for TiledWeightedSumKernel<T, E> {
    type Primitives = E::Primitives;

    fn primitives(&self) -> &Self::Primitives {
        self.engine.primitives()
    }

    fn name(&self) -> &'static str {
        "Tiled Weighted Sum Kernel"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::compute_hd_weights;
    use robust_core::SparseWeights;

    #[test]
    fn test_tiled_kernel_basic() {
        // Create test data
        let datasets = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3.0, 4.0, 5.0, 6.0, 7.0],
        ];
        let dataset_refs: Vec<&[f64]> = datasets.iter().map(|d| d.as_slice()).collect();

        // Create sparse weights for multiple quantiles
        let quantiles = vec![0.25, 0.5, 0.75];
        let sparse_rows: Vec<SparseWeights> = quantiles
            .iter()
            .map(|&p| compute_hd_weights(5, p))
            .collect();

        // Convert to tiled format (small tiles for testing)
        let tiled = TiledSparseMatrix::from_sparse_rows(sparse_rows, 2, 3);

        // Create kernel
        let engine = robust_core::execution::scalar_sequential();
        let kernel = TiledWeightedSumKernel::new(engine);

        // Apply tiled weights
        let results = kernel.apply_all_tiles_to_datasets(&dataset_refs, &tiled);

        // Verify we got results for each dataset and quantile
        assert_eq!(results.len(), 3); // 3 datasets
        assert_eq!(results[0].len(), 3); // 3 quantiles

        // Results should be increasing for each dataset (since data is sorted)
        for dataset_results in &results {
            assert!(dataset_results[0] < dataset_results[1]); // Q1 < median
            assert!(dataset_results[1] < dataset_results[2]); // median < Q3
        }
    }

    #[test]
    fn test_single_tile_processing() {
        // Create a simple tile
        let tile = SparseTile::new(
            0,
            0,
            0,
            2,
            0,
            3,
            vec![
                robust_core::TileEntry {
                    local_row: 0,
                    local_col: 1,
                    weight: 0.5,
                },
                robust_core::TileEntry {
                    local_row: 1,
                    local_col: 2,
                    weight: 0.5,
                },
            ],
        );

        let datasets = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let dataset_refs: Vec<&[f64]> = datasets.iter().map(|d| d.as_slice()).collect();

        let mut results = vec![vec![0.0; 2]];

        let engine = robust_core::execution::scalar_sequential();
        let kernel = TiledWeightedSumKernel::new(engine);

        kernel.apply_single_tile(&dataset_refs, &tile, &mut results);

        // First row: 0.5 * data[1] = 0.5 * 2.0 = 1.0
        assert_eq!(results[0][0], 1.0);
        // Second row: 0.5 * data[2] = 0.5 * 3.0 = 1.5
        assert_eq!(results[0][1], 1.5);
    }
}