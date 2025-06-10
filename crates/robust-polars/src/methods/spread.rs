//! Spread estimation implementations

use polars::prelude::*;
use robust_core::{
    execution::{SequentialEngine, auto_engine}, 
    best_available_backend,
    StatefulEstimator,
};
use robust_quantile::{
    estimators::{harrell_davis, trimmed_harrell_davis},
    weights::{ConstantWidthFn, SqrtWidthFn, LinearWidthFn},
};
use crate::{Result, Error, SpreadMethod, QuantileCache};

// Import the global caches from the quantile module
use super::quantile::{HD_CACHE_F64, HD_CACHE_F32};

// Helper function to create appropriate quantile estimator and estimate spread
fn create_and_estimate_with_cache<T, S, E>(
    spread_est: S,
    data: &[T],
    engine: E,
    cache: &QuantileCache<T>,
) -> Result<T::Float>
where
    T: robust_core::Numeric + num_traits::NumCast,
    S: robust_spread::SpreadEstimator<T, robust_quantile::HarrellDavis<T, E>> 
        + robust_spread::SpreadEstimator<T, robust_quantile::TrimmedHarrellDavis<T, E, ConstantWidthFn>>
        + robust_spread::SpreadEstimator<T, robust_quantile::TrimmedHarrellDavis<T, E, SqrtWidthFn>>
        + robust_spread::SpreadEstimator<T, robust_quantile::TrimmedHarrellDavis<T, E, LinearWidthFn>>,
    E: robust_core::ExecutionEngine<T> + Clone,
{
    use robust_spread::SpreadAdapter;
    
    match cache {
        QuantileCache::HarrellDavis(hd_cache) => {
            let quantile_est = harrell_davis(engine);
            let adapter = SpreadAdapter::new(spread_est, quantile_est);
            adapter.estimate_with_cache(data, hd_cache)
                .map_err(|e| Error::ComputationFailed(e.to_string()))
        }
        QuantileCache::TrimmedConstant { cache: thd_cache, width } => {
            let quantile_est = trimmed_harrell_davis(engine, ConstantWidthFn::new(*width));
            let adapter = SpreadAdapter::new(spread_est, quantile_est);
            adapter.estimate_with_cache(data, thd_cache)
                .map_err(|e| Error::ComputationFailed(e.to_string()))
        }
        QuantileCache::TrimmedSqrt(thd_cache) => {
            let quantile_est = trimmed_harrell_davis(engine, SqrtWidthFn);
            let adapter = SpreadAdapter::new(spread_est, quantile_est);
            adapter.estimate_with_cache(data, thd_cache)
                .map_err(|e| Error::ComputationFailed(e.to_string()))
        }
        QuantileCache::TrimmedLinear(thd_cache) => {
            let quantile_est = trimmed_harrell_davis(engine, LinearWidthFn);
            let adapter = SpreadAdapter::new(spread_est, quantile_est);
            adapter.estimate_with_cache(data, thd_cache)
                .map_err(|e| Error::ComputationFailed(e.to_string()))
        }
    }
}

fn compute_spread_f64(
    column: &Column,
    method: &SpreadMethod,
) -> Result<f64> {
    let hd_cache = QuantileCache::HarrellDavis(HD_CACHE_F64.clone());
    compute_spread_cached_f64(column, method, &hd_cache)
}

fn compute_spread_cached_f64(
    column: &Column,
    method: &SpreadMethod,
    cache: &QuantileCache<f64>,
) -> Result<f64> {
    let ca = column.f64()?;
    
    if ca.is_empty() {
        return Ok(f64::NAN);
    }
    
    // Sort using Polars' native sorting (ascending order, nulls last)
    let sorted_ca = ca.sort(false);
    
    // Rechunk to ensure contiguous memory if possible
    let rechunked = sorted_ca.rechunk();
    
    // Create the best available engine
    let engine = auto_engine();
    let backend = best_available_backend::<f64>();
    
    // Create the appropriate spread estimator
    use robust_spread::{Mad, Iqr, Qad};
    
    // Try to get zero-copy access to the data
    if let Ok(slice) = rechunked.cont_slice() {
        // Zero-copy path: data is contiguous
        match method {
            SpreadMethod::Mad => {
                let mad = Mad::new(backend.clone());
                create_and_estimate_with_cache(mad, slice, engine, cache)
            }
            SpreadMethod::Iqr => {
                let iqr = Iqr::new(backend.clone());
                create_and_estimate_with_cache(iqr, slice, engine, cache)
            }
            SpreadMethod::Qad { probability } => {
                let qad = Qad::new(backend.clone(), *probability)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))?;
                create_and_estimate_with_cache(qad, slice, engine, cache)
            }
            SpreadMethod::TrimmedStd { trim_proportion: _ } => {
                // TODO: Implement trimmed standard deviation
                Err(Error::ComputationFailed("TrimmedStd not yet implemented".to_string()))
            }
            SpreadMethod::WinsorizedStd { winsor_proportion: _ } => {
                // TODO: Implement winsorized standard deviation
                Err(Error::ComputationFailed("WinsorizedStd not yet implemented".to_string()))
            }
        }
    } else {
        // Fallback path: collect into Vec if data is not contiguous
        let data: Vec<f64> = rechunked.into_no_null_iter().collect();
        
        match method {
            SpreadMethod::Mad => {
                let mad = Mad::new(backend.clone());
                create_and_estimate_with_cache(mad, &data, engine, cache)
            }
            SpreadMethod::Iqr => {
                let iqr = Iqr::new(backend.clone());
                create_and_estimate_with_cache(iqr, &data, engine, cache)
            }
            SpreadMethod::Qad { probability } => {
                let qad = Qad::new(backend.clone(), *probability)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))?;
                create_and_estimate_with_cache(qad, &data, engine, cache)
            }
            SpreadMethod::TrimmedStd { trim_proportion: _ } => {
                // TODO: Implement trimmed standard deviation
                Err(Error::ComputationFailed("TrimmedStd not yet implemented".to_string()))
            }
            SpreadMethod::WinsorizedStd { winsor_proportion: _ } => {
                // TODO: Implement winsorized standard deviation
                Err(Error::ComputationFailed("WinsorizedStd not yet implemented".to_string()))
            }
        }
    }
}

fn compute_spread_f32(
    column: &Column,
    method: &SpreadMethod,
) -> Result<f64> {
    let hd_cache = QuantileCache::HarrellDavis(HD_CACHE_F32.clone());
    compute_spread_cached_f32(column, method, &hd_cache)
}

fn compute_spread_cached_f32(
    column: &Column,
    method: &SpreadMethod,
    cache: &QuantileCache<f32>,
) -> Result<f64> {
    let ca = column.f32()?;
    
    if ca.is_empty() {
        return Ok(f64::NAN);
    }
    
    // Sort using Polars' native sorting (ascending order, nulls last)
    let sorted_ca = ca.sort(false);
    
    // Rechunk to ensure contiguous memory if possible
    let rechunked = sorted_ca.rechunk();
    
    // Create the best available engine
    let backend = best_available_backend::<f32>();
    let engine = SequentialEngine::new(backend.clone());
    
    // Create the appropriate spread estimator
    use robust_spread::{Mad, Iqr, Qad};
    
    // Try to get zero-copy access to the data
    let result = if let Ok(slice) = rechunked.cont_slice() {
        // Zero-copy path: data is contiguous
        match method {
            SpreadMethod::Mad => {
                let mad = Mad::new(backend.clone());
                create_and_estimate_with_cache(mad, slice, engine, cache)?
            }
            SpreadMethod::Iqr => {
                let iqr = Iqr::new(backend.clone());
                create_and_estimate_with_cache(iqr, slice, engine, cache)?
            }
            SpreadMethod::Qad { probability } => {
                let qad = Qad::new(backend.clone(), *probability)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))?;
                create_and_estimate_with_cache(qad, slice, engine, cache)?
            }
            SpreadMethod::TrimmedStd { trim_proportion: _ } => {
                // TODO: Implement trimmed standard deviation
                return Err(Error::ComputationFailed("TrimmedStd not yet implemented".to_string()));
            }
            SpreadMethod::WinsorizedStd { winsor_proportion: _ } => {
                // TODO: Implement winsorized standard deviation
                return Err(Error::ComputationFailed("WinsorizedStd not yet implemented".to_string()));
            }
        }
    } else {
        // Fallback path: collect into Vec if data is not contiguous
        let data: Vec<f32> = rechunked.into_no_null_iter().collect();
        
        match method {
            SpreadMethod::Mad => {
                let mad = Mad::new(backend.clone());
                create_and_estimate_with_cache(mad, &data, engine, cache)?
            }
            SpreadMethod::Iqr => {
                let iqr = Iqr::new(backend.clone());
                create_and_estimate_with_cache(iqr, &data, engine, cache)?
            }
            SpreadMethod::Qad { probability } => {
                let qad = Qad::new(backend.clone(), *probability)
                    .map_err(|e| Error::ComputationFailed(e.to_string()))?;
                create_and_estimate_with_cache(qad, &data, engine, cache)?
            }
            SpreadMethod::TrimmedStd { trim_proportion: _ } => {
                // TODO: Implement trimmed standard deviation
                return Err(Error::ComputationFailed("TrimmedStd not yet implemented".to_string()));
            }
            SpreadMethod::WinsorizedStd { winsor_proportion: _ } => {
                // TODO: Implement winsorized standard deviation
                return Err(Error::ComputationFailed("WinsorizedStd not yet implemented".to_string()));
            }
        }
    };
    
    // Convert f32 result to f64
    Ok(result as f64)
}

/// Helper function to process spread with a QuantileCache
fn compute_spread_with_cache<T>(
    column: &Column,
    method: &SpreadMethod,
    cache: &QuantileCache<T>,
) -> Result<f64>
where
    T: robust_core::Numeric + num_traits::NumCast + 'static,
{
    // Match on the column type and verify T matches at runtime
    match (column.dtype(), std::any::TypeId::of::<T>()) {
        (DataType::Float64, id) if id == std::any::TypeId::of::<f64>() => {
            // Safe to transmute cache reference since we checked the type
            let cache_f64 = unsafe { 
                std::mem::transmute::<&QuantileCache<T>, &QuantileCache<f64>>(cache) 
            };
            compute_spread_cached_f64(column, method, cache_f64)
        }
        (DataType::Float32, id) if id == std::any::TypeId::of::<f32>() => {
            // Safe to transmute cache reference since we checked the type
            let cache_f32 = unsafe { 
                std::mem::transmute::<&QuantileCache<T>, &QuantileCache<f32>>(cache) 
            };
            compute_spread_cached_f32(column, method, cache_f32)
        }
        (DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8, id) if id == std::any::TypeId::of::<f64>() => {
            // Convert to f64 for computation
            let float_column = column.cast(&DataType::Float64)?;
            let cache_f64 = unsafe { 
                std::mem::transmute::<&QuantileCache<T>, &QuantileCache<f64>>(cache) 
            };
            compute_spread_cached_f64(&float_column, method, cache_f64)
        }
        (dt, _) => {
            Err(Error::TypeMismatch {
                expected: "numeric type matching cache type".to_string(),
                got: format!("{:?}", dt),
            })
        }
    }
}

/// Public implementation function called from the trait implementation
pub(crate) fn robust_spread_impl(
    df: &DataFrame,
    columns: &[&str],
    method: SpreadMethod,
) -> Result<DataFrame> {
    let mut result_series = Vec::new();
    
    for col_name in columns {
        let column = df.column(col_name)
            .map_err(|_| Error::InvalidColumn(col_name.to_string()))?;
        
        // Process based on data type
        let value = match column.dtype() {
            DataType::Float64 => {
                compute_spread_f64(column, &method)?
            }
            DataType::Float32 => {
                compute_spread_f32(column, &method)?
            }
            DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                // Convert to f64 for computation
                let float_column = column.cast(&DataType::Float64)?;
                compute_spread_f64(&float_column, &method)?
            }
            dt => {
                return Err(Error::TypeMismatch {
                    expected: "numeric".to_string(),
                    got: format!("{:?}", dt),
                });
            }
        };
        
        // Create result column
        let col_name = format!("{}_{}", col_name, method.name());
        let series = Series::new(col_name.as_str().into(), vec![value]);
        result_series.push(series.into());
    }
    
    Ok(DataFrame::new(result_series)?)
}

/// Public implementation function for cached spread computation
pub(crate) fn robust_spread_cached_impl<T>(
    df: &DataFrame,
    columns: &[&str],
    method: SpreadMethod,
    cache: &QuantileCache<T>,
) -> Result<DataFrame>
where
    T: robust_core::Numeric + num_traits::NumCast + 'static,
{
    let mut result_series = Vec::new();
    
    for col_name in columns {
        let column = df.column(col_name)
            .map_err(|_| Error::InvalidColumn(col_name.to_string()))?;
        
        // Process with the provided cache
        let value = compute_spread_with_cache(column, &method, cache)?;
        
        // Create result column
        let col_name = format!("{}_{}", col_name, method.name());
        let series = Series::new(col_name.as_str().into(), vec![value]);
        result_series.push(series.into());
    }
    
    Ok(DataFrame::new(result_series)?)
}