//! Compile-time specialization for processing strategies
//!
//! This module provides function-level specialization that enables
//! the compiler to generate optimal code paths for sequential vs parallel
//! execution without runtime overhead.

use crate::kernels::TiledWeightedSumKernel;
use crate::QuantileKernel;
use num_traits::Zero;
use robust_core::{
    execution::{ExecutionEngine, ExecutionMode},
    ComputePrimitives, Numeric, SparseWeights, TiledSparseMatrix,
};
use std::sync::Arc;

/// Apply sparse weights to a single dataset - sequential version
///
/// This function avoids closure overhead by directly calling the kernel.
#[inline]
pub fn apply_sparse_weights_sequential<T: Numeric, K: QuantileKernel<T>>(
    kernel: &K,
    sorted_data: &[T],
    weights: &Arc<SparseWeights<T>>,
) -> T::Float {
    kernel.apply_sparse_weights(sorted_data, weights)
}

/// Apply sparse weights across multiple datasets - sequential version
///
/// This processes multiple datasets without using closures.
pub fn apply_sparse_weights_batch_sequential<T: Numeric, K: QuantileKernel<T>>(
    kernel: &K,
    sorted_datasets: &[&[T]],
    weights: &Arc<SparseWeights<T>>,
) -> Vec<T::Float> {
    sorted_datasets
        .iter()
        .map(|data| kernel.apply_sparse_weights(data, weights))
        .collect()
}

/// Apply sparse weights across multiple datasets - parallel version
///
/// This uses the engine for parallel execution.
pub fn apply_sparse_weights_batch_parallel<
    T: Numeric,
    E: ExecutionEngine<T>,
    K: QuantileKernel<T>,
>(
    engine: &E,
    kernel: &K,
    sorted_datasets: &[&[T]],
    weights: &Arc<SparseWeights<T>>,
) -> Vec<T::Float> {
    engine.execute_batch(sorted_datasets.len(), |i| {
        kernel.apply_sparse_weights(sorted_datasets[i], weights)
    })
}

/// Apply sparse weights with compile-time dispatch
#[inline]
pub fn apply_sparse_weights_batch<
    T: Numeric,
    E: ExecutionEngine<T> + ExecutionMode,
    K: QuantileKernel<T>,
>(
    engine: &E,
    kernel: &K,
    sorted_datasets: &[&[T]],
    weights: &Arc<SparseWeights<T>>,
) -> Vec<T::Float> {
    if E::IS_SEQUENTIAL {
        apply_sparse_weights_batch_sequential(kernel, sorted_datasets, weights)
    } else {
        apply_sparse_weights_batch_parallel(engine, kernel, sorted_datasets, weights)
    }
}

/// Process tiles sequentially with direct function calls
///
/// This function is optimized for sequential execution and avoids
/// all closure overhead by using direct iteration.
pub fn process_tiles_sequential<T: Numeric, P: ComputePrimitives<T>>(
    kernel: &TiledWeightedSumKernel<T, robust_core::execution::SequentialEngine<T, P>>,
    tiled_weights: &TiledSparseMatrix<T>,
    sorted_inputs: &[&[T]],
    n_quantiles: usize,
) -> Vec<Vec<T::Float>> {
    let n_datasets = sorted_inputs.len();
    let mut final_results = vec![vec![<T::Float as Zero>::zero(); n_quantiles]; n_datasets];

    // Direct iteration - no closures, perfect for compiler optimization
    for tile in &tiled_weights.tiles {
        kernel.apply_single_tile(sorted_inputs, tile, &mut final_results);
    }

    final_results
}

/// Process tiles sequentially for any sequential engine type
///
/// This avoids unsafe code by using a generic function that works
/// with any sequential engine type.
pub fn process_tiles_sequential_generic<T: Numeric, E: ExecutionEngine<T>>(
    kernel: &TiledWeightedSumKernel<T, E>,
    tiled_weights: &TiledSparseMatrix<T>,
    sorted_inputs: &[&[T]],
    n_quantiles: usize,
) -> Vec<Vec<T::Float>> {
    let n_datasets = sorted_inputs.len();
    let mut final_results = vec![vec![<T::Float as Zero>::zero(); n_quantiles]; n_datasets];

    // Direct iteration - no closures, perfect for compiler optimization
    for tile in &tiled_weights.tiles {
        kernel.apply_single_tile(sorted_inputs, tile, &mut final_results);
    }

    final_results
}

/// Process tiles in parallel using the execution engine
///
/// This function is optimized for parallel execution and uses
/// the engine's thread pool for efficient work distribution.
pub fn process_tiles_parallel<T: Numeric, E: ExecutionEngine<T>>(
    engine: &E,
    kernel: &TiledWeightedSumKernel<T, E>,
    tiled_weights: &TiledSparseMatrix<T>,
    sorted_inputs: &[&[T]],
    n_quantiles: usize,
) -> Vec<Vec<T::Float>> {
    let n_datasets = sorted_inputs.len();
    let n_tiles = tiled_weights.tiles.len();

    // Compute optimal chunk size for parallel processing
    let chunk_size = compute_optimal_chunk_size(n_tiles, engine.num_threads());
    let n_chunks = n_tiles.div_ceil(chunk_size);

    // Process tiles in chunks using the engine's thread pool
    let chunk_results = engine.execute_batch(n_chunks, |chunk_idx| {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(n_tiles);

        // Process multiple tiles in one task to amortize overhead
        let mut chunk_results = vec![vec![<T::Float as Zero>::zero(); n_quantiles]; n_datasets];
        for tile_idx in start..end {
            let tile = &tiled_weights.tiles[tile_idx];
            kernel.apply_single_tile(sorted_inputs, tile, &mut chunk_results);
        }
        chunk_results
    });

    // Aggregate results from all chunks
    let mut final_results = vec![vec![<T::Float as Zero>::zero(); n_quantiles]; n_datasets];
    for chunk_result in chunk_results {
        for (dataset_idx, dataset_results) in chunk_result.into_iter().enumerate() {
            for (quantile_idx, value) in dataset_results.into_iter().enumerate() {
                final_results[dataset_idx][quantile_idx] += value;
            }
        }
    }

    final_results
}

/// Compile-time dispatcher for tile processing
///
/// This function uses the engine's compile-time constants to choose
/// between sequential and parallel processing without runtime overhead.
#[inline]
pub fn process_tiles<T: Numeric, E: ExecutionEngine<T> + ExecutionMode>(
    engine: &E,
    kernel: &TiledWeightedSumKernel<T, E>,
    tiled_weights: &TiledSparseMatrix<T>,
    sorted_inputs: &[&[T]],
    n_quantiles: usize,
) -> Vec<Vec<T::Float>> {
    // This constant is known at compile time!
    if E::IS_SEQUENTIAL {
        // For sequential engines, use direct processing
        // The compiler will optimize away the unused branch
        process_tiles_sequential_generic(kernel, tiled_weights, sorted_inputs, n_quantiles)
    } else {
        // For parallel engines, use the closure-based approach
        process_tiles_parallel(engine, kernel, tiled_weights, sorted_inputs, n_quantiles)
    }
}

/// Process datasets in tile-major order, parallelizing by dataset chunks
///
/// This is optimized for when there are many datasets relative to tiles.
pub fn process_tiles_dataset_major<T: Numeric, E: ExecutionEngine<T> + ExecutionMode>(
    engine: &E,
    kernel: &TiledWeightedSumKernel<T, E>,
    tiled_weights: &TiledSparseMatrix<T>,
    sorted_inputs: &[&[T]],
    n_quantiles: usize,
) -> Vec<Vec<T::Float>> {
    let n_datasets = sorted_inputs.len();

    if E::IS_SEQUENTIAL {
        // Sequential version - process all datasets for each tile
        let mut results = vec![vec![<T::Float as Zero>::zero(); n_quantiles]; n_datasets];

        // Process all tiles for all datasets
        for tile in &tiled_weights.tiles {
            kernel.apply_single_tile(sorted_inputs, tile, &mut results);
        }

        results
    } else {
        // Parallel version - chunk by datasets
        let chunk_size = compute_optimal_chunk_size(n_datasets, engine.num_threads())
            .max(engine.primitives().simd_width());
        let n_chunks = n_datasets.div_ceil(chunk_size);

        let chunk_results = engine.execute_batch(n_chunks, |chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(n_datasets);
            let chunk_datasets = &sorted_inputs[start..end];

            let mut chunk_results =
                vec![vec![<T::Float as Zero>::zero(); n_quantiles]; chunk_datasets.len()];

            // Process all tiles for this dataset chunk
            for tile in &tiled_weights.tiles {
                kernel.apply_single_tile(chunk_datasets, tile, &mut chunk_results);
            }

            chunk_results
        });

        // Flatten chunk results into final results
        chunk_results.into_iter().flatten().collect()
    }
}

/// Process each quantile across all datasets
///
/// This strategy is optimal when there are many datasets with the same size
/// and relatively few quantiles.
pub fn process_parameter_major<
    T: Numeric,
    E: ExecutionEngine<T> + ExecutionMode,
    K: QuantileKernel<T>,
>(
    engine: &E,
    kernel: &K,
    sorted_inputs: &[&[T]],
    params: &[f64],
    weights_for_quantiles: &[Option<Arc<SparseWeights<T>>>],
) -> Vec<Vec<T::Float>> {
    let n_datasets = sorted_inputs.len();
    let n_params = params.len();
    let mut all_results = vec![vec![<T::Float as Zero>::zero(); n_params]; n_datasets];

    // Process each quantile
    for (q_idx, (weight_opt, &p)) in weights_for_quantiles.iter().zip(params).enumerate() {
        if p == 0.0 {
            // First element of each dataset
            for (dataset_idx, data) in sorted_inputs.iter().enumerate() {
                all_results[dataset_idx][q_idx] = data[0].to_float();
            }
        } else if p == 1.0 {
            // Last element of each dataset
            for (dataset_idx, data) in sorted_inputs.iter().enumerate() {
                all_results[dataset_idx][q_idx] = data[data.len() - 1].to_float();
            }
        } else if let Some(weights) = weight_opt {
            // Apply weights to all datasets
            let results = if E::IS_SEQUENTIAL || sorted_inputs.len() <= 10 {
                // Sequential or small batch - direct iteration
                sorted_inputs
                    .iter()
                    .map(|data| kernel.apply_sparse_weights(data, weights))
                    .collect::<Vec<_>>()
            } else {
                // Parallel over datasets for large batches
                apply_sparse_weights_batch(engine, kernel, sorted_inputs, weights)
            };

            // Store results
            for (dataset_idx, &result) in results.iter().enumerate() {
                all_results[dataset_idx][q_idx] = result;
            }
        }
    }

    all_results
}

/// Process datasets with pre-computed weights (all same length)
///
/// This strategy is optimal when all datasets have the same size and
/// weights can be reused across datasets.
pub fn process_dataset_major_same_length<
    T: Numeric,
    E: ExecutionEngine<T> + ExecutionMode,
    K: QuantileKernel<T>,
>(
    engine: &E,
    kernel: &K,
    sorted_inputs: &[&[T]],
    params: &[f64],
    weights_for_quantiles: &[Option<Arc<SparseWeights<T>>>],
) -> Vec<Vec<T::Float>> {
    // Validate that all datasets have the same length
    #[cfg(debug_assertions)]
    if !sorted_inputs.is_empty() {
        let expected_len = sorted_inputs[0].len();
        for (i, data) in sorted_inputs.iter().enumerate() {
            debug_assert_eq!(
                data.len(), 
                expected_len,
                "Dataset {} has length {} but expected {}",
                i, data.len(), expected_len
            );
        }
    }
    if E::IS_SEQUENTIAL {
        // Sequential version - direct iteration
        sorted_inputs
            .iter()
            .map(|data| {
                let mut results = Vec::with_capacity(params.len());

                for (weight_opt, &p) in weights_for_quantiles.iter().zip(params) {
                    if p == 0.0 {
                        results.push(data[0].to_float());
                    } else if p == 1.0 {
                        results.push(data[data.len() - 1].to_float());
                    } else if let Some(weights) = weight_opt {
                        results.push(kernel.apply_sparse_weights(data, weights));
                    } else {
                        // This should not happen - weights should be Some for p in (0,1)
                        panic!("Missing weights for quantile p={}", p);
                    }
                }
                results
            })
            .collect()
    } else {
        // Parallel version
        engine.execute_batch(sorted_inputs.len(), |i| {
            let data = sorted_inputs[i];
            let mut results = Vec::with_capacity(params.len());

            for (weight_opt, &p) in weights_for_quantiles.iter().zip(params) {
                if p == 0.0 {
                    results.push(data[0].to_float());
                } else if p == 1.0 {
                    results.push(data[data.len() - 1].to_float());
                } else if let Some(weights) = weight_opt {
                    results.push(kernel.apply_sparse_weights(data, weights));
                } else {
                    // This should not happen - weights should be Some for p in (0,1)
                    panic!("Missing weights for quantile p={}", p);
                }
            }
            results
        })
    }
}

/// Process each dataset independently
///
/// This strategy is optimal when datasets have different sizes or there are
/// relatively few datasets.
pub fn process_dataset_major<
    T: Numeric + num_traits::NumCast,
    E: ExecutionEngine<T> + ExecutionMode,
    K: QuantileKernel<T>,
    W: robust_core::WeightComputer<T> + Clone + 'static,
>(
    engine: &E,
    kernel: &K,
    sorted_inputs: &[&[T]],
    params: &[f64],
    cache: &robust_core::UnifiedWeightCache<W, T>,
) -> Vec<Vec<T::Float>> {
    if E::IS_SEQUENTIAL {
        // Sequential version - direct iteration
        sorted_inputs
            .iter()
            .map(|data| {
                let mut results = Vec::with_capacity(params.len());

                // Handle special cases
                if data.len() == 1 {
                    results.resize(params.len(), data[0].to_float());
                    return results;
                }

                // Process each quantile for this dataset
                for &p in params {
                    if p == 0.0 {
                        results.push(data[0].to_float());
                    } else if p == 1.0 {
                        results.push(data[data.len() - 1].to_float());
                    } else {
                        let weights = cache.get_sparse_with_engine(data.len(), p, engine);
                        results.push(kernel.apply_sparse_weights(data, &weights));
                    }
                }
                results
            })
            .collect()
    } else {
        // Parallel version
        engine.execute_batch(sorted_inputs.len(), |i| {
            let data = sorted_inputs[i];
            let mut results = Vec::with_capacity(params.len());

            // Handle special cases
            if data.len() == 1 {
                results.resize(params.len(), data[0].to_float());
                return results;
            }

            // Process each quantile for this dataset
            for &p in params {
                if p == 0.0 {
                    results.push(data[0].to_float());
                } else if p == 1.0 {
                    results.push(data[data.len() - 1].to_float());
                } else {
                    let weights = cache.get_sparse_with_engine(data.len(), p, engine);
                    results.push(kernel.apply_sparse_weights(data, &weights));
                }
            }
            results
        })
    }
}

/// Compute optimal chunk size for parallel processing
pub(crate) fn compute_optimal_chunk_size(n_items: usize, n_threads: usize) -> usize {
    // Heuristic: aim for 4-8 chunks per thread for good load balancing
    // but ensure chunks are large enough to amortize overhead
    let target_chunks = n_threads * 6;
    let chunk_size = n_items.div_ceil(target_chunks);

    // Minimum chunk size to amortize task creation overhead
    const MIN_CHUNK_SIZE: usize = 4;
    chunk_size.max(MIN_CHUNK_SIZE).min(n_items)
}

/// Tile size configuration based on workload characteristics
pub struct TileConfig {
    pub row_size: usize,
    pub col_size: usize,
}

impl TileConfig {
    /// Compute optimal tile size based on problem characteristics
    pub fn optimal(n_rows: usize, n_cols: usize, n_datasets: usize) -> Self {
        // Base tile sizes for good cache locality
        let base_row_size = match n_rows {
            ..=1000 => 64,
            1001..=10000 => 128,
            _ => 255,
        };

        let base_col_size = match n_cols {
            ..=1000 => 64,
            1001..=10000 => 128,
            _ => 255,
        };

        // Adjust column size for SIMD processing of multiple datasets
        // We want to process multiple datasets together for better SIMD utilization
        let simd_width = 4; // Typical SIMD width for f64
        let datasets_per_simd = n_datasets.div_ceil(simd_width);
        let adjusted_col_size = base_col_size.max(datasets_per_simd * 16);

        // But don't make tiles too large (diminishing returns and memory pressure)
        let max_tile_size = 255;

        Self {
            row_size: base_row_size.min(max_tile_size).min(n_rows),
            col_size: adjusted_col_size.min(max_tile_size).min(n_cols),
        }
    }
}
