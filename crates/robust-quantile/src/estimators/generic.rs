//! Generic quantile estimator implementation
//!
//! This provides a unified implementation for all quantile estimators
//! that use weighted sums of order statistics.

use crate::{
    processing_traits::{process_dataset_major, process_dataset_major_same_length, process_parameter_major, process_tiles, process_tiles_dataset_major, TileConfig},
    kernels::{TiledWeightedSumKernel, WeightedSumKernel},
    Error, QuantileEstimator, Result,
};
use robust_core::{
    execution::{ExecutionEngine, ExecutionMode},
    BatchProcessor, CachePolicy, CentralTendencyEstimator, ProcessingStrategy,
    UnifiedWeightCache, WeightComputer, Numeric, SparseWeights,
};
use num_traits::Zero;

/// Generic quantile estimator that works with any weight computer
///
/// This provides the core implementation for both Harrell-Davis and
/// Trimmed Harrell-Davis estimators, eliminating code duplication.
#[derive(Clone)]
pub struct GenericQuantileEstimator<T: Numeric, E: ExecutionEngine<T>, C: WeightComputer<T> + Clone> {
    engine: E,
    kernel: WeightedSumKernel<T, E>,
    tiled_kernel: TiledWeightedSumKernel<T, E>,
    computer: C,
}

impl<T: Numeric, E: ExecutionEngine<T>, C: WeightComputer<T> + Clone> GenericQuantileEstimator<T, E, C> {
    /// Create new generic quantile estimator
    pub fn new(engine: E, computer: C) -> Self {
        let kernel = WeightedSumKernel::new(engine.clone());
        let tiled_kernel = TiledWeightedSumKernel::new(engine.clone());
        Self {
            engine,
            kernel,
            tiled_kernel,
            computer,
        }
    }

    /// Get the execution engine
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Get the weight computer
    pub fn computer(&self) -> &C {
        &self.computer
    }
}

/// State for generic estimator using unified cache
pub type GenericState<T, C> = UnifiedWeightCache<C, T>;

// Implement CentralTendencyEstimator
impl<T: Numeric + num_traits::NumCast, E: ExecutionEngine<T> + ExecutionMode, C: WeightComputer<T> + Clone + 'static> CentralTendencyEstimator<T>
    for GenericQuantileEstimator<T, E, C>
{
    fn estimate_sorted(&self, sorted_data: &[T]) -> robust_core::Result<T::Float> {
        // Create temporary cache for single estimation
        let cache = UnifiedWeightCache::new(self.computer.clone(), CachePolicy::NoCache);

        self.quantile_sorted(sorted_data, 0.5, &cache)
            .map_err(|e| robust_core::Error::Other(e.into()))
    }

    fn name(&self) -> &str {
        // This could be made configurable if needed
        "Generic Quantile Estimator"
    }

    fn is_robust(&self) -> bool {
        true
    }

    fn breakdown_point(&self) -> f64 {
        0.5 // Median has 50% breakdown point
    }
}

// Implement BatchProcessor
impl<T: Numeric + num_traits::NumCast, E: ExecutionEngine<T> + ExecutionMode, C: WeightComputer<T> + Clone + 'static> BatchProcessor<T>
    for GenericQuantileEstimator<T, E, C>
{
    type Input = [T];
    type Output = Vec<T::Float>;
    type State = GenericState<T, C>;
    type Params = [f64];

    fn process_batch(
        &self,
        inputs: &mut [&mut Self::Input],
        params: &Self::Params,
        state: &Self::State,
        strategy: ProcessingStrategy,
    ) -> robust_core::Result<Vec<Self::Output>> {
        // Sort each dataset in place
        for data in inputs.iter_mut() {
            data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }

        // Now delegate to process_batch_sorted
        // Optimize for common case of single dataset
        if inputs.len() == 1 {
            let sorted_ref = &*inputs[0] as &[T];
            self.process_batch_sorted(&[sorted_ref], params, state, strategy)
        } else {
            // For multiple datasets, we need to collect immutable references
            let sorted_refs: Vec<&[T]> = inputs.iter().map(|data| &**data).collect();
            self.process_batch_sorted(&sorted_refs, params, state, strategy)
        }
    }

    fn process_batch_sorted(
        &self,
        sorted_inputs: &[&Self::Input],
        params: &Self::Params,
        state: &Self::State,
        strategy: ProcessingStrategy,
    ) -> robust_core::Result<Vec<Self::Output>> {
        use crate::Error as QError;
        use std::sync::Arc;
        
        // Validate inputs
        if sorted_inputs.is_empty() {
            return Ok(vec![]);
        }
        
        // Check all probabilities upfront
        for &p in params {
            QError::check_probability(p)
                .map_err(|e| robust_core::Error::Other(e.into()))?;
        }
        
        // Check that all datasets have the same length (required for some strategies)
        let n = sorted_inputs[0].len();
        for data in sorted_inputs {
            if data.is_empty() {
                return Err(robust_core::Error::Other(QError::EmptyData.into()));
            }
        }
        
        // Check if all datasets have the same length
        let all_same_length = sorted_inputs.iter().all(|data| data.len() == n);
        
        match strategy {
            ProcessingStrategy::TiledTileMajor => {
                if !all_same_length {
                    return Err(robust_core::Error::Other(
                        QError::Numerical("Tiled strategies require all datasets to have the same length".to_string()).into()
                    ));
                }
                
                // Compute optimal tile size based on workload
                let tile_config = TileConfig::optimal(n, n, sorted_inputs.len());
                
                // Get tiled weights with optimized tile size
                let tiled_weights = state.get_tiled_with_engine(n, params, &self.engine, tile_config.row_size, tile_config.col_size);
                let n_quantiles = params.len();
                
                // Use specialized processing based on engine type
                Ok(process_tiles(
                    &self.engine,
                    &self.tiled_kernel,
                    &tiled_weights,
                    sorted_inputs,
                    n_quantiles,
                ))
            }
            
            ProcessingStrategy::TiledDatasetMajor => {
                if !all_same_length {
                    return Err(robust_core::Error::Other(
                        QError::Numerical("Tiled strategies require all datasets to have the same length".to_string()).into()
                    ));
                }
                
                // Compute optimal tile size
                let tile_config = TileConfig::optimal(n, n, sorted_inputs.len());
                let tiled_weights = state.get_tiled_with_engine(n, params, &self.engine, tile_config.row_size, tile_config.col_size);
                let n_quantiles = params.len();
                
                // Use compile-time dispatched processing
                Ok(process_tiles_dataset_major(
                    &self.engine,
                    &self.tiled_kernel,
                    &tiled_weights,
                    sorted_inputs,
                    n_quantiles,
                ))
            }
            
            ProcessingStrategy::DatasetMajor => {
                if all_same_length && n > 1 {
                    // Pre-compute weights for all quantiles (cache-friendly)
                    let weights_for_quantiles: Vec<Option<Arc<robust_core::SparseWeights<T>>>> = params
                        .iter()
                        .map(|&p| {
                            if p == 0.0 || p == 1.0 {
                                None
                            } else {
                                let weights = state.get_sparse_with_engine(n, p, &self.engine);
                                
                                // Validate weights
                                #[cfg(debug_assertions)]
                                {
                                    for &idx in &weights.indices {
                                        debug_assert!(
                                            idx < n,
                                            "Weight index {} out of bounds for n={}, p={}",
                                            idx, n, p
                                        );
                                    }
                                    debug_assert_eq!(
                                        weights.n, n,
                                        "Weight dimension {} doesn't match data size {}, p={}",
                                        weights.n, n, p
                                    );
                                }
                                
                                Some(weights)
                            }
                        })
                        .collect();
                    
                    // Use compile-time dispatched processing with pre-computed weights
                    Ok(process_dataset_major_same_length(
                        &self.engine,
                        &self.kernel,
                        sorted_inputs,
                        params,
                        &weights_for_quantiles,
                    ))
                } else {
                    // Datasets have different lengths - use general dataset major processing
                    Ok(process_dataset_major(
                        &self.engine,
                        &self.kernel,
                        sorted_inputs,
                        params,
                        state,
                    ))
                }
            }
            
            ProcessingStrategy::ParameterMajor => {
                if !all_same_length {
                    return Err(robust_core::Error::Other(
                        QError::Numerical("ParameterMajor strategy requires all datasets to have the same length".to_string()).into()
                    ));
                }
                
                // Handle special case: single data point in all datasets
                if n == 1 {
                    let mut all_results: Vec<Vec<T::Float>> = vec![vec![<T::Float as Zero>::zero(); params.len()]; sorted_inputs.len()];
                    for (i, results) in all_results.iter_mut().enumerate() {
                        let value = sorted_inputs[i][0].to_float();
                        for r in results.iter_mut() {
                            *r = value;
                        }
                    }
                    return Ok(all_results);
                }
                
                // Pre-compute weights for all quantiles
                let weights_for_quantiles: Vec<Option<Arc<robust_core::SparseWeights<T>>>> = params
                    .iter()
                    .map(|&p| {
                        if p == 0.0 || p == 1.0 {
                            None
                        } else {
                            let weights = state.get_sparse_with_engine(n, p, &self.engine);
                            
                            // Validate weights
                            #[cfg(debug_assertions)]
                            {
                                for &idx in &weights.indices {
                                    debug_assert!(
                                        idx < n,
                                        "Weight index {} out of bounds for n={}, p={}",
                                        idx, n, p
                                    );
                                }
                                debug_assert_eq!(
                                    weights.n, n,
                                    "Weight dimension {} doesn't match data size {}, p={}",
                                    weights.n, n, p
                                );
                            }
                            
                            Some(weights)
                        }
                    })
                    .collect();
                
                // Use compile-time dispatched processing
                Ok(process_parameter_major(
                    &self.engine,
                    &self.kernel,
                    sorted_inputs,
                    params,
                    &weights_for_quantiles,
                ))
            }
            
            ProcessingStrategy::Auto => {
                // Choose strategy based on workload characteristics
                let num_datasets = sorted_inputs.len();
                let num_quantiles = params.len();
                
                // Updated heuristics based on clean strategy separation
                if all_same_length && num_quantiles >= 50 && n >= 10000 {
                    // Many quantiles and large datasets -> Tiled
                    // Now we also choose between tile-major and dataset-major
                    let tiled_weights = state.get_tiled_with_engine(n, params, &self.engine, 32, 256);
                    let n_tiles = tiled_weights.tiles.len();
                    
                    if n_tiles > num_datasets * 4 {
                        // Many tiles, fewer datasets -> TileMajor
                        self.process_batch_sorted(
                            sorted_inputs, params, state, 
                            ProcessingStrategy::TiledTileMajor
                        )
                    } else {
                        // Many datasets, fewer tiles -> TiledDatasetMajor
                        self.process_batch_sorted(
                            sorted_inputs, params, state,
                            ProcessingStrategy::TiledDatasetMajor
                        )
                    }
                } else if all_same_length && num_datasets > num_quantiles * 2 {
                    // Many datasets with same size, few quantiles -> ParameterMajor
                    self.process_batch_sorted(
                        sorted_inputs, params, state,
                        ProcessingStrategy::ParameterMajor
                    )
                } else {
                    // Few datasets, varying sizes, or few quantiles -> DatasetMajor
                    self.process_batch_sorted(
                        sorted_inputs, params, state,
                        ProcessingStrategy::DatasetMajor
                    )
                }
            }
        }
    }

    fn optimal_batch_size(&self) -> usize {
        100
    }

    fn benefits_from_batching(&self) -> bool {
        true
    }
}

/// Simple weight computer for generic quantile estimation (linear interpolation)
#[derive(Clone, Debug)]
pub struct SimpleWeightComputer;

impl<T: Numeric> WeightComputer<T> for SimpleWeightComputer {
    fn compute_sparse(&self, n: usize, p: f64) -> SparseWeights<T> {
        if n < 2 {
            // For single element, just return full weight on that element
            return SparseWeights::new(vec![0], vec![T::from_f64(1.0)], n);
        }
        
        // Simple linear interpolation
        let pos = p * (n - 1) as f64;
        let lower = pos.floor() as usize;
        let upper = (lower + 1).min(n - 1);
        let weight = T::from_f64(pos - lower as f64);
        
        let indices = vec![lower, upper];
        let weights = vec![
            T::from_f64(1.0) - weight,
            weight,
        ];
        
        SparseWeights::new(indices, weights, n)
    }
    
    fn compute_sparse_range(&self, n: usize, p: f64, start: usize, end: usize) -> SparseWeights<T> {
        if n < 2 || start >= end {
            return SparseWeights::new(vec![], vec![], n);
        }
        
        // Simple linear interpolation
        let pos = p * (n - 1) as f64;
        let lower = pos.floor() as usize;
        let upper = (lower + 1).min(n - 1);
        
        // Check if the interpolation points are within the requested range
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        
        let weight = T::from_f64(pos - lower as f64);
        
        if lower >= start && lower < end {
            indices.push(lower);
            weights.push(T::from_f64(1.0) - weight);
        }
        
        if upper >= start && upper < end && upper != lower {
            indices.push(upper);
            weights.push(weight);
        }
        
        SparseWeights::new(indices, weights, n)
    }
    
    fn compute_tiled(&self, n: usize, quantiles: &[f64]) -> std::sync::Arc<robust_core::TiledSparseMatrix<T>> {
        let sparse_rows: Vec<SparseWeights<T>> = quantiles
            .iter()
            .map(|&p| self.compute_sparse(n, p))
            .collect();
            
        std::sync::Arc::new(robust_core::TiledSparseMatrix::from_sparse_rows(
            sparse_rows,
            100, // default tile row size
            n,   // default tile col size (full width)
        ))
    }
}

/// Type alias for convenience - a generic quantile estimator with simple interpolation
pub type Generic<T, E> = GenericQuantileEstimator<T, E, SimpleWeightComputer>;

// Implement QuantileEstimator
impl<T: Numeric + num_traits::NumCast, E: ExecutionEngine<T> + ExecutionMode, C: WeightComputer<T> + Clone + 'static> QuantileEstimator<T>
    for GenericQuantileEstimator<T, E, C>
{
    fn quantile_sorted(&self, sorted_data: &[T], p: f64, cache: &Self::State) -> Result<T::Float> {
        // Delegate to quantiles_sorted for a single quantile
        // This ensures we use the same optimized code path
        let quantiles = self.quantiles_sorted(sorted_data, &[p], cache)?;
        Ok(quantiles.into_iter().next().unwrap())
    }

    fn quantiles_sorted(
        &self,
        sorted_data: &[T],
        ps: &[f64],
        cache: &Self::State,
    ) -> Result<Vec<T::Float>> {
        // Delegate to BatchProcessor for a single dataset
        // This gives us all the optimizations (tiling, SIMD, etc.)
        // Using array syntax to avoid Vec allocation
        let batch_results = self
            .process_batch_sorted(&[sorted_data], ps, cache, ProcessingStrategy::Auto)
            .map_err(Error::Core)?;

        // Extract the single result vector
        Ok(batch_results.into_iter().next().unwrap())
    }
}