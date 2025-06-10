//! Quantile estimation implementations

use polars::prelude::*;
use robust_core::{
    execution::{SequentialEngine, auto_engine}, 
    EstimatorFactory,
    UnifiedWeightCache,
    CachePolicy,
    best_available_backend,
};
use robust_quantile::{
    factories::{HarrellDavisFactory, trimmed_hd_constant_factory, trimmed_hd_sqrt_factory, trimmed_hd_linear_factory},
    HDWeightComputer,
    TrimmedHDWeightComputer,
    weights::{ConstantWidthFn, SqrtWidthFn, LinearWidthFn},
};
use robust_core::BatchQuantileEstimator;
use crate::{Result, Error, QuantileMethod, RobustStatsExt, TrimWidth, QuantileCache};
use lazy_static::lazy_static;
use std::sync::Arc;

// Global thread-safe caches for common operations
lazy_static! {
    // Harrell-Davis caches
    pub(super) static ref HD_CACHE_F64: Arc<UnifiedWeightCache<HDWeightComputer<f64>, f64>> = 
        Arc::new(UnifiedWeightCache::new(
            HDWeightComputer::new(), 
            CachePolicy::Lru { max_entries: 1024 * 1024 }
        ));
    
    pub(super) static ref HD_CACHE_F32: Arc<UnifiedWeightCache<HDWeightComputer<f32>, f32>> = 
        Arc::new(UnifiedWeightCache::new(
            HDWeightComputer::new(), 
            CachePolicy::Lru { max_entries: 1024 * 1024 }
        ));
    
    // Trimmed HD with sqrt width caches
    static ref THD_SQRT_CACHE_F64: Arc<UnifiedWeightCache<TrimmedHDWeightComputer<f64, SqrtWidthFn>, f64>> = 
        Arc::new(UnifiedWeightCache::new(
            TrimmedHDWeightComputer::new(SqrtWidthFn), 
            CachePolicy::Lru { max_entries: 1024 * 1024 }
        ));
    
    static ref THD_SQRT_CACHE_F32: Arc<UnifiedWeightCache<TrimmedHDWeightComputer<f32, SqrtWidthFn>, f32>> = 
        Arc::new(UnifiedWeightCache::new(
            TrimmedHDWeightComputer::new(SqrtWidthFn), 
            CachePolicy::Lru { max_entries: 1024 * 1024 }
        ));
    
    // Trimmed HD with linear width caches
    static ref THD_LINEAR_CACHE_F64: Arc<UnifiedWeightCache<TrimmedHDWeightComputer<f64, LinearWidthFn>, f64>> = 
        Arc::new(UnifiedWeightCache::new(
            TrimmedHDWeightComputer::new(LinearWidthFn), 
            CachePolicy::Lru { max_entries: 1024 * 1024 }
        ));
    
    static ref THD_LINEAR_CACHE_F32: Arc<UnifiedWeightCache<TrimmedHDWeightComputer<f32, LinearWidthFn>, f32>> = 
        Arc::new(UnifiedWeightCache::new(
            TrimmedHDWeightComputer::new(LinearWidthFn), 
            CachePolicy::Lru { max_entries: 1024 * 1024 }
        ));
}

impl RobustStatsExt for DataFrame {
    fn robust_quantiles(
        &self,
        columns: &[&str],
        quantiles: &[f64],
        method: QuantileMethod,
    ) -> Result<DataFrame> {
        let mut result_series = Vec::new();
        
        for col_name in columns {
            let column = self.column(col_name)
                .map_err(|_| Error::InvalidColumn(col_name.to_string()))?;
            
            // Process based on data type
            let values = match column.dtype() {
                DataType::Float64 => {
                    compute_quantiles_f64(column, quantiles, method)?
                }
                DataType::Float32 => {
                    compute_quantiles_f32(column, quantiles, method)?
                }
                DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                    // Convert to f64 for computation
                    let float_column = column.cast(&DataType::Float64)?;
                    compute_quantiles_f64(&float_column, quantiles, method)?
                }
                dt => {
                    return Err(Error::TypeMismatch {
                        expected: "numeric".to_string(),
                        got: format!("{:?}", dt),
                    });
                }
            };
            
            // Create result columns
            for (i, &q) in quantiles.iter().enumerate() {
                let col_name = format!("{}_q{:.2}", col_name, q);
                let series = Series::new(col_name.as_str().into(), vec![values[i]]);
                result_series.push(series.into());
            }
        }
        
        Ok(DataFrame::new(result_series)?)
    }
    
    fn robust_quantiles_cached<T>(
        &self,
        columns: &[&str],
        quantiles: &[f64],
        method: QuantileMethod,
        cache: &QuantileCache<T>,
    ) -> Result<DataFrame>
    where
        T: robust_core::Numeric + num_traits::NumCast + 'static,
    {
        let mut result_series = Vec::new();
        
        for col_name in columns {
            let column = self.column(col_name)
                .map_err(|_| Error::InvalidColumn(col_name.to_string()))?;
            
            // Process based on data type and cache type
            let values = match (column.dtype(), std::any::TypeId::of::<T>()) {
                (DataType::Float64, id) if id == std::any::TypeId::of::<f64>() => {
                    // Safe to transmute cache reference since we checked the type
                    let cache_f64 = unsafe { std::mem::transmute::<&QuantileCache<T>, &QuantileCache<f64>>(cache) };
                    compute_quantiles_cached_f64(column, quantiles, method, cache_f64)?
                }
                (DataType::Float32, id) if id == std::any::TypeId::of::<f32>() => {
                    // Safe to transmute cache reference since we checked the type
                    let cache_f32 = unsafe { std::mem::transmute::<&QuantileCache<T>, &QuantileCache<f32>>(cache) };
                    compute_quantiles_cached_f32(column, quantiles, method, cache_f32)?
                }
                (DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8, id) if id == std::any::TypeId::of::<f64>() => {
                    // Convert to f64 for computation
                    let float_column = column.cast(&DataType::Float64)?;
                    let cache_f64 = unsafe { std::mem::transmute::<&QuantileCache<T>, &QuantileCache<f64>>(cache) };
                    compute_quantiles_cached_f64(&float_column, quantiles, method, cache_f64)?
                }
                (dt, _) => {
                    return Err(Error::TypeMismatch {
                        expected: "numeric type matching cache type".to_string(),
                        got: format!("{:?}", dt),
                    });
                }
            };
            
            // Create result columns
            for (i, &q) in quantiles.iter().enumerate() {
                let col_name = format!("{}_q{:.2}", col_name, q);
                let series = Series::new(col_name.as_str().into(), vec![values[i]]);
                result_series.push(series.into());
            }
        }
        
        Ok(DataFrame::new(result_series)?)
    }

    fn robust_spread(
        &self,
        columns: &[&str],
        method: crate::SpreadMethod,
    ) -> Result<DataFrame> {
        // Delegate to spread module
        crate::methods::spread::robust_spread_impl(self, columns, method)
    }
    
    fn robust_spread_cached<T>(
        &self,
        columns: &[&str],
        method: crate::SpreadMethod,
        cache: &crate::QuantileCache<T>,
    ) -> Result<DataFrame>
    where
        T: robust_core::Numeric + num_traits::NumCast + 'static,
    {
        // Delegate to spread module
        crate::methods::spread::robust_spread_cached_impl(self, columns, method, cache)
    }

    fn robust_confidence(
        &self,
        _columns: &[&str],
        _level: f64,
        _method: crate::ConfidenceMethod,
    ) -> Result<DataFrame> {
        // Implemented in confidence.rs
        todo!()
    }

    fn robust_changepoints(
        &self,
        _column: &str,
        _method: crate::ChangePointMethod,
    ) -> Result<DataFrame> {
        // Implemented in changepoint.rs
        todo!()
    }

    fn robust_stability(
        &self,
        _columns: &[&str],
        _window_size: usize,
    ) -> Result<DataFrame> {
        // Implemented in stability.rs
        todo!()
    }

    fn robust_modality(
        &self,
        _columns: &[&str],
        _sensitivity: f64,
    ) -> Result<DataFrame> {
        // Implemented in modality.rs
        todo!()
    }

    fn robust_effect_size(
        &self,
        _group_col: &str,
        _value_col: &str,
        _method: crate::EffectSizeMethod,
    ) -> Result<DataFrame> {
        // Implemented in effect.rs
        todo!()
    }
}

fn compute_quantiles_f64(
    column: &Column,
    quantiles: &[f64],
    method: QuantileMethod,
) -> Result<Vec<f64>> {
    let ca = column.f64()?;
    
    if ca.is_empty() {
        return Ok(vec![f64::NAN; quantiles.len()]);
    }
    
    // Sort using Polars' native sorting (ascending order, nulls last)
    let sorted_ca = ca.sort(false);
    
    // Rechunk to ensure contiguous memory if possible
    let rechunked = sorted_ca.rechunk();
    
    // Create the best available engine for f64
    let engine = auto_engine();
    
    // Try to get zero-copy access to the data
    let result = if let Ok(slice) = rechunked.cont_slice() {
        // Zero-copy path: data is contiguous
        match method {
            QuantileMethod::HarrellDavis => {
                // Use factory to create estimator
                let factory = HarrellDavisFactory::<f64>::default();
                let estimator = factory.create(engine);
                
                // Use global cache
                estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, &**HD_CACHE_F64)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))
            }
            QuantileMethod::TrimmedHarrellDavis { width } => {
                match width {
                    TrimWidth::Constant(proportion) => {
                        let factory = trimmed_hd_constant_factory::<f64>(proportion);
                        let estimator = factory.create(engine);
                        
                        // TODO: Create a global cache per constant width value
                        // For now, create a new cache
                        let cache = UnifiedWeightCache::<TrimmedHDWeightComputer<f64, ConstantWidthFn>, f64>::new(
                            TrimmedHDWeightComputer::new(ConstantWidthFn::new(proportion)), 
                            CachePolicy::Lru { max_entries: 1024 * 1024 }
                        );
                        
                        estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, &cache)
                            .map_err(|e| Error::ComputationFailed(e.to_string()))
                    }
                    TrimWidth::Sqrt => {
                        let factory = trimmed_hd_sqrt_factory::<f64>();
                        let estimator = factory.create(engine);
                        
                        // Use global sqrt cache
                        estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, &**THD_SQRT_CACHE_F64)
                            .map_err(|e| Error::ComputationFailed(e.to_string()))
                    }
                    TrimWidth::Linear => {
                        let factory = trimmed_hd_linear_factory::<f64>();
                        let estimator = factory.create(engine);
                        
                        // Use global linear cache
                        estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, &**THD_LINEAR_CACHE_F64)
                            .map_err(|e| Error::ComputationFailed(e.to_string()))
                    }
                }
            }
        }
    } else {
        // Fallback path: collect into Vec if data is not contiguous
        let data: Vec<f64> = rechunked.into_no_null_iter().collect();
        
        match method {
            QuantileMethod::HarrellDavis => {
                let factory = HarrellDavisFactory::<f64>::default();
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, &**HD_CACHE_F64)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))
            }
            QuantileMethod::TrimmedHarrellDavis { width } => {
                match width {
                    TrimWidth::Constant(proportion) => {
                        let factory = trimmed_hd_constant_factory::<f64>(proportion);
                        let estimator = factory.create(engine);
                        let cache = UnifiedWeightCache::<TrimmedHDWeightComputer<f64, ConstantWidthFn>, f64>::new(
                            TrimmedHDWeightComputer::new(ConstantWidthFn::new(proportion)), 
                            CachePolicy::Lru { max_entries: 1024 * 1024 }
                        );
                        estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, &cache)
                            .map_err(|e| Error::ComputationFailed(e.to_string()))
                    }
                    TrimWidth::Sqrt => {
                        let factory = trimmed_hd_sqrt_factory::<f64>();
                        let estimator = factory.create(engine);
                        estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, &**THD_SQRT_CACHE_F64)
                            .map_err(|e| Error::ComputationFailed(e.to_string()))
                    }
                    TrimWidth::Linear => {
                        let factory = trimmed_hd_linear_factory::<f64>();
                        let estimator = factory.create(engine);
                        estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, &**THD_LINEAR_CACHE_F64)
                            .map_err(|e| Error::ComputationFailed(e.to_string()))
                    }
                }
            }
        }
    };
    
    result
}

fn compute_quantiles_f32(
    column: &Column,
    quantiles: &[f64],
    method: QuantileMethod,
) -> Result<Vec<f64>> {
    let ca = column.f32()?;
    
    if ca.is_empty() {
        return Ok(vec![f64::NAN; quantiles.len()]);
    }
    
    // Sort using Polars' native sorting (ascending order, nulls last)
    let sorted_ca = ca.sort(false);
    
    // Rechunk to ensure contiguous memory if possible
    let rechunked = sorted_ca.rechunk();
    
    // Create the best available engine for f32
    let backend = best_available_backend::<f32>();
    let engine = SequentialEngine::new(backend);
    
    // Try to get zero-copy access to the data
    let results = if let Ok(slice) = rechunked.cont_slice() {
        // Zero-copy path: data is contiguous
        match method {
            QuantileMethod::HarrellDavis => {
                let factory = HarrellDavisFactory::<f32>::default();
                let estimator = factory.create(engine);
                
                // Use global cache
                estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, &**HD_CACHE_F32)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))?
            }
            QuantileMethod::TrimmedHarrellDavis { width } => {
                match width {
                    TrimWidth::Constant(proportion) => {
                        let factory = trimmed_hd_constant_factory::<f32>(proportion);
                        let estimator = factory.create(engine);
                        
                        // TODO: Create a global cache per constant width value
                        // For now, create a new cache
                        let cache = UnifiedWeightCache::<TrimmedHDWeightComputer<f32, ConstantWidthFn>, f32>::new(
                            TrimmedHDWeightComputer::new(ConstantWidthFn::new(proportion)), 
                            CachePolicy::Lru { max_entries: 1024 * 1024 }
                        );
                        
                        estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, &cache)
                            .map_err(|e| Error::ComputationFailed(e.to_string()))?
                    }
                    TrimWidth::Sqrt => {
                        let factory = trimmed_hd_sqrt_factory::<f32>();
                        let estimator = factory.create(engine);
                        
                        // Use global sqrt cache
                        estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, &**THD_SQRT_CACHE_F32)
                            .map_err(|e| Error::ComputationFailed(e.to_string()))?
                    }
                    TrimWidth::Linear => {
                        let factory = trimmed_hd_linear_factory::<f32>();
                        let estimator = factory.create(engine);
                        
                        // Use global linear cache
                        estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, &**THD_LINEAR_CACHE_F32)
                            .map_err(|e| Error::ComputationFailed(e.to_string()))?
                    }
                }
            }
        }
    } else {
        // Fallback path: collect into Vec if data is not contiguous
        let data: Vec<f32> = rechunked.into_no_null_iter().collect();
        
        match method {
            QuantileMethod::HarrellDavis => {
                let factory = HarrellDavisFactory::<f32>::default();
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, &**HD_CACHE_F32)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))?
            }
            QuantileMethod::TrimmedHarrellDavis { width } => {
                match width {
                    TrimWidth::Constant(proportion) => {
                        let factory = trimmed_hd_constant_factory::<f32>(proportion);
                        let estimator = factory.create(engine);
                        let cache = UnifiedWeightCache::<TrimmedHDWeightComputer<f32, ConstantWidthFn>, f32>::new(
                            TrimmedHDWeightComputer::new(ConstantWidthFn::new(proportion)), 
                            CachePolicy::Lru { max_entries: 1024 * 1024 }
                        );
                        estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, &cache)
                            .map_err(|e| Error::ComputationFailed(e.to_string()))?
                    }
                    TrimWidth::Sqrt => {
                        let factory = trimmed_hd_sqrt_factory::<f32>();
                        let estimator = factory.create(engine);
                        estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, &**THD_SQRT_CACHE_F32)
                            .map_err(|e| Error::ComputationFailed(e.to_string()))?
                    }
                    TrimWidth::Linear => {
                        let factory = trimmed_hd_linear_factory::<f32>();
                        let estimator = factory.create(engine);
                        estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, &**THD_LINEAR_CACHE_F32)
                            .map_err(|e| Error::ComputationFailed(e.to_string()))?
                    }
                }
            }
        }
    };
    
    // Convert f32 results to f64
    Ok(results.into_iter().map(|x| x as f64).collect())
}

// Helper function to compute quantiles with custom cache for f64
fn compute_quantiles_cached_f64(
    column: &Column,
    quantiles: &[f64],
    method: QuantileMethod,
    cache: &QuantileCache<f64>,
) -> Result<Vec<f64>> {
    // Validate cache matches method
    if !cache.matches_method(&method) {
        return Err(Error::InvalidInput(
            "Provided cache does not match the quantile method".to_string()
        ));
    }
    
    let ca = column.f64()?;
    
    if ca.is_empty() {
        return Ok(vec![f64::NAN; quantiles.len()]);
    }
    
    // Sort using Polars' native sorting (ascending order, nulls last)
    let sorted_ca = ca.sort(false);
    
    // Rechunk to ensure contiguous memory if possible
    let rechunked = sorted_ca.rechunk();
    
    // Create the best available engine for f64
    let engine = auto_engine();
    
    // Try to get zero-copy access to the data
    if let Ok(slice) = rechunked.cont_slice() {
        // Zero-copy path: data is contiguous
        match (method, cache) {
            (QuantileMethod::HarrellDavis, QuantileCache::HarrellDavis(cache)) => {
                let factory = HarrellDavisFactory::<f64>::default();
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))
            }
            (
                QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Constant(_) }, 
                QuantileCache::TrimmedConstant { cache, .. }
            ) => {
                // Extract proportion from method since cache validation already checked it matches
                let proportion = match method {
                    QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Constant(p) } => p,
                    _ => unreachable!(),
                };
                let factory = trimmed_hd_constant_factory::<f64>(proportion);
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))
            }
            (
                QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Sqrt }, 
                QuantileCache::TrimmedSqrt(cache)
            ) => {
                let factory = trimmed_hd_sqrt_factory::<f64>();
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))
            }
            (
                QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Linear }, 
                QuantileCache::TrimmedLinear(cache)
            ) => {
                let factory = trimmed_hd_linear_factory::<f64>();
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))
            }
            _ => {
                // This shouldn't happen due to earlier validation
                Err(Error::InvalidInput("Cache type mismatch".to_string()))
            }
        }
    } else {
        // Fallback path: collect into Vec if data is not contiguous
        let data: Vec<f64> = rechunked.into_no_null_iter().collect();
        
        match (method, cache) {
            (QuantileMethod::HarrellDavis, QuantileCache::HarrellDavis(cache)) => {
                let factory = HarrellDavisFactory::<f64>::default();
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))
            }
            (
                QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Constant(_) }, 
                QuantileCache::TrimmedConstant { cache, .. }
            ) => {
                let proportion = match method {
                    QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Constant(p) } => p,
                    _ => unreachable!(),
                };
                let factory = trimmed_hd_constant_factory::<f64>(proportion);
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))
            }
            (
                QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Sqrt }, 
                QuantileCache::TrimmedSqrt(cache)
            ) => {
                let factory = trimmed_hd_sqrt_factory::<f64>();
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))
            }
            (
                QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Linear }, 
                QuantileCache::TrimmedLinear(cache)
            ) => {
                let factory = trimmed_hd_linear_factory::<f64>();
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))
            }
            _ => {
                // This shouldn't happen due to earlier validation
                Err(Error::InvalidInput("Cache type mismatch".to_string()))
            }
        }
    }
}

// Helper function to compute quantiles with custom cache for f32
fn compute_quantiles_cached_f32(
    column: &Column,
    quantiles: &[f64],
    method: QuantileMethod,
    cache: &QuantileCache<f32>,
) -> Result<Vec<f64>> {
    // Validate cache matches method
    if !cache.matches_method(&method) {
        return Err(Error::InvalidInput(
            "Provided cache does not match the quantile method".to_string()
        ));
    }
    
    let ca = column.f32()?;
    
    if ca.is_empty() {
        return Ok(vec![f64::NAN; quantiles.len()]);
    }
    
    // Sort using Polars' native sorting (ascending order, nulls last)
    let sorted_ca = ca.sort(false);
    
    // Rechunk to ensure contiguous memory if possible
    let rechunked = sorted_ca.rechunk();
    
    // Create the best available engine for f32
    let backend = best_available_backend::<f32>();
    let engine = SequentialEngine::new(backend);
    
    // Try to get zero-copy access to the data
    let results = if let Ok(slice) = rechunked.cont_slice() {
        // Zero-copy path: data is contiguous
        match (method, cache) {
            (QuantileMethod::HarrellDavis, QuantileCache::HarrellDavis(cache)) => {
                let factory = HarrellDavisFactory::<f32>::default();
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))?
            }
            (
                QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Constant(_) }, 
                QuantileCache::TrimmedConstant { cache, .. }
            ) => {
                let proportion = match method {
                    QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Constant(p) } => p,
                    _ => unreachable!(),
                };
                let factory = trimmed_hd_constant_factory::<f32>(proportion);
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))?
            }
            (
                QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Sqrt }, 
                QuantileCache::TrimmedSqrt(cache)
            ) => {
                let factory = trimmed_hd_sqrt_factory::<f32>();
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))?
            }
            (
                QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Linear }, 
                QuantileCache::TrimmedLinear(cache)
            ) => {
                let factory = trimmed_hd_linear_factory::<f32>();
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(slice, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))?
            }
            _ => {
                // This shouldn't happen due to earlier validation
                return Err(Error::InvalidInput("Cache type mismatch".to_string()));
            }
        }
    } else {
        // Fallback path: collect into Vec if data is not contiguous
        let data: Vec<f32> = rechunked.into_no_null_iter().collect();
        
        match (method, cache) {
            (QuantileMethod::HarrellDavis, QuantileCache::HarrellDavis(cache)) => {
                let factory = HarrellDavisFactory::<f32>::default();
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))?
            }
            (
                QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Constant(_) }, 
                QuantileCache::TrimmedConstant { cache, .. }
            ) => {
                let proportion = match method {
                    QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Constant(p) } => p,
                    _ => unreachable!(),
                };
                let factory = trimmed_hd_constant_factory::<f32>(proportion);
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))?
            }
            (
                QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Sqrt }, 
                QuantileCache::TrimmedSqrt(cache)
            ) => {
                let factory = trimmed_hd_sqrt_factory::<f32>();
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))?
            }
            (
                QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Linear }, 
                QuantileCache::TrimmedLinear(cache)
            ) => {
                let factory = trimmed_hd_linear_factory::<f32>();
                let estimator = factory.create(engine);
                estimator.estimate_quantiles_sorted_with_cache(&data, quantiles, cache)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))?
            }
            _ => {
                // This shouldn't happen due to earlier validation
                return Err(Error::InvalidInput("Cache type mismatch".to_string()));
            }
        }
    };
    
    // Convert f32 results to f64
    Ok(results.into_iter().map(|x| x as f64).collect())
}
