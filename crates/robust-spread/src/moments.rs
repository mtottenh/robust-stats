//! Higher-order moments: skewness and kurtosis
//!
//! This module provides robust estimators for skewness and kurtosis.

use crate::kernels::{SkewnessKernel, KurtosisKernel};
use robust_core::{ComputePrimitives, Error, Result, Numeric};
use robust_quantile::QuantileEstimator;
use num_traits::{NumCast, Float};
use std::marker::PhantomData;
use std::iter::Sum;

/// Robust skewness estimator using quantiles
/// 
/// Requires a quantile estimator to be passed as a parameter for all operations.
#[derive(Debug, Clone)]
pub struct RobustSkewness<T: Numeric, P: ComputePrimitives<T>> {
    kernel: SkewnessKernel<T, P>,
    _phantom: PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> RobustSkewness<T, P> {
    pub fn new(primitives: P) -> Self {
        Self {
            kernel: SkewnessKernel::new(primitives),
            _phantom: PhantomData,
        }
    }
    
    /// Calculate robust skewness using quantile-based method
    /// Skewness = (Q3 + Q1 - 2*Q2) / (Q3 - Q1)
    pub fn estimate<Q: QuantileEstimator<T>>(
        &self,
        data: &mut [T],
        estimator: &Q,
        cache: &Q::State,
    ) -> Result<T::Float> {
        self.kernel.compute_quantile_skewness(data, estimator, cache)
    }
    
    /// Alternative: Medcouple skewness (more robust to outliers)
    pub fn medcouple<Q: QuantileEstimator<T>>(
        &self,
        data: &mut [T],
        estimator: &Q,
        cache: &Q::State,
    ) -> Result<T::Float> {
        medcouple_skewness(data, estimator, cache)
    }
}

/// Robust kurtosis estimator using quantiles
/// 
/// Requires a quantile estimator to be passed as a parameter for all operations.
#[derive(Debug, Clone)]
pub struct RobustKurtosis<T: Numeric, P: ComputePrimitives<T>> {
    kernel: KurtosisKernel<T, P>,
    _phantom: PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> RobustKurtosis<T, P> {
    pub fn new(primitives: P) -> Self {
        Self {
            kernel: KurtosisKernel::new(primitives),
            _phantom: PhantomData,
        }
    }
    
    /// Calculate robust kurtosis using quantile-based method
    /// Based on Moors' kurtosis: ((Q7 - Q5) + (Q3 - Q1)) / (Q6 - Q2)
    pub fn estimate<Q: QuantileEstimator<T>>(
        &self,
        data: &mut [T],
        estimator: &Q,
        cache: &Q::State,
    ) -> Result<T::Float> {
        self.kernel.compute_moors_kurtosis(data, estimator, cache)
    }
    
    /// Alternative: Crow-Siddiqui kurtosis
    pub fn crow_siddiqui<Q: QuantileEstimator<T>>(
        &self,
        data: &mut [T],
        estimator: &Q,
        cache: &Q::State,
    ) -> Result<T::Float> {
        crow_siddiqui_kurtosis(data, estimator, cache)
    }
}

/// Calculate robust skewness using quantile estimator
/// Skewness = (Q3 + Q1 - 2*Q2) / (Q3 - Q1)
pub fn robust_skewness<T: Numeric, Q: QuantileEstimator<T>>(
    data: &mut [T],
    estimator: &Q,
    cache: &Q::State,
) -> Result<T::Float> {
    let q1 = estimator
        .quantile(data, 0.25, cache)
        .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
    let q2 = estimator
        .quantile(data, 0.5, cache)
        .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
    let q3 = estimator
        .quantile(data, 0.75, cache)
        .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
    
    let iqr = q3 - q1;
    let epsilon = <T::Float as NumCast>::from(f64::EPSILON).unwrap();
    if iqr.abs() < epsilon {
        return Ok(<T::Float as NumCast>::from(0.0).unwrap());
    }
    
    let two = <T::Float as NumCast>::from(2.0).unwrap();
    Ok((q3 + q1 - two * q2) / iqr)
}

/// Medcouple skewness (more robust to outliers)
pub fn medcouple_skewness<T: Numeric, Q: QuantileEstimator<T>>(
    data: &mut [T],
    estimator: &Q,
    cache: &Q::State,
) -> Result<T::Float> {
    if data.len() < 3 {
        return Err(robust_core::Error::InsufficientData { expected: 3, actual: data.len() });
    }
    
    let median = estimator
        .quantile(data, 0.5, cache)
        .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
    
    let mut left_values = Vec::new();
    let mut right_values = Vec::new();
    
    for &x in data.iter() {
        let x_float = x.to_float();
        if x_float < median {
            left_values.push(x_float);
        } else if x_float > median {
            right_values.push(x_float);
        }
    }
    
    if left_values.is_empty() || right_values.is_empty() {
        return Ok(<T::Float as NumCast>::from(0.0).unwrap());
    }
    
    // Compute medcouple
    let mut h_values = Vec::new();
    for &xi in &left_values {
        for &xj in &right_values {
            let h = (xj - median - (median - xi)) / (xj - xi);
            h_values.push(T::from_f64(h.into()));
        }
    }
    
    // Use quantile estimator to find median of h_values
    estimator
        .quantile(&mut h_values, 0.5, cache)
        .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))
}

/// Calculate robust kurtosis using quantile-based method
/// Based on Moors' kurtosis: ((Q7 - Q5) + (Q3 - Q1)) / (Q6 - Q2)
pub fn robust_kurtosis<T: Numeric, Q: QuantileEstimator<T>>(
    data: &mut [T],
    estimator: &Q,
    cache: &Q::State,
) -> Result<T::Float> {
    // Compute octiles using the quantile estimator
    let q1 = estimator.quantile(data, 1.0 / 8.0, cache)
        .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
    let q2 = estimator.quantile(data, 2.0 / 8.0, cache)
        .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
    let q3 = estimator.quantile(data, 3.0 / 8.0, cache)
        .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
    let q5 = estimator.quantile(data, 5.0 / 8.0, cache)
        .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
    let q6 = estimator.quantile(data, 6.0 / 8.0, cache)
        .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
    let q7 = estimator.quantile(data, 7.0 / 8.0, cache)
        .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
    
    let numerator = (q7 - q5) + (q3 - q1);
    let denominator = q6 - q2;
    
    let epsilon = <T::Float as NumCast>::from(f64::EPSILON).unwrap();
    if denominator.abs() < epsilon {
        return Ok(<T::Float as NumCast>::from(3.0).unwrap()); // Return normal kurtosis if no spread
    }
    
    Ok(numerator / denominator)
}

/// Crow-Siddiqui kurtosis using quantile estimator
pub fn crow_siddiqui_kurtosis<T: Numeric, Q: QuantileEstimator<T>>(
    data: &mut [T],
    estimator: &Q,
    cache: &Q::State,
) -> Result<T::Float> {
    if data.len() < 4 {
        return Err(robust_core::Error::InvalidInput(
            "Need at least 4 observations for kurtosis".to_string(),
        ));
    }
    
    let q025 = estimator.quantile(data, 0.025, cache)
        .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
    let q25 = estimator.quantile(data, 0.25, cache)
        .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
    let q75 = estimator.quantile(data, 0.75, cache)
        .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
    let q975 = estimator.quantile(data, 0.975, cache)
        .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
    
    let iqr = q75 - q25;
    let epsilon = <T::Float as NumCast>::from(f64::EPSILON).unwrap();
    if iqr.abs() < epsilon {
        return Ok(<T::Float as NumCast>::from(0.0).unwrap());
    }
    
    let cs_const = <T::Float as NumCast>::from(2.91).unwrap();
    Ok((q975 - q025) / iqr - cs_const)
}

/// Calculate traditional (non-robust) skewness
pub fn classical_skewness<T: Numeric>(sample: &[T]) -> Result<T::Float> 
where
    T::Float: Sum,
{
    if sample.len() < 3 {
        return Err(Error::InsufficientData { expected: 3, actual: sample.len() });
    }
    
    let n = <T::Float as NumCast>::from(sample.len()).unwrap();
    let sum: T::Float = sample.iter().map(|&x| x.to_float()).sum();
    let mean = sum / n;
    
    let variance: T::Float = sample.iter()
        .map(|&x| {
            let x_float = x.to_float();
            let diff = x_float - mean;
            diff * diff
        })
        .sum::<T::Float>() / n;
    
    let epsilon = <T::Float as NumCast>::from(f64::EPSILON).unwrap();
    if variance.abs() < epsilon {
        return Ok(<T::Float as NumCast>::from(0.0).unwrap());
    }
    
    let std_dev = <T::Float as Float>::sqrt(variance);
    let skewness: T::Float = sample.iter()
        .map(|&x| {
            let x_float = x.to_float();
            let normalized = (x_float - mean) / std_dev;
            normalized * normalized * normalized
        })
        .sum::<T::Float>() / n;
    
    Ok(skewness)
}

/// Calculate traditional (non-robust) kurtosis
pub fn classical_kurtosis<T: Numeric>(sample: &[T]) -> Result<T::Float> 
where
    T::Float: Sum,
{
    if sample.len() < 4 {
        return Err(Error::InsufficientData { expected: 4, actual: sample.len() });
    }
    
    let n = <T::Float as NumCast>::from(sample.len()).unwrap();
    let sum: T::Float = sample.iter().map(|&x| x.to_float()).sum();
    let mean = sum / n;
    
    let variance: T::Float = sample.iter()
        .map(|&x| {
            let x_float = x.to_float();
            let diff = x_float - mean;
            diff * diff
        })
        .sum::<T::Float>() / n;
    
    let epsilon = <T::Float as NumCast>::from(f64::EPSILON).unwrap();
    if variance.abs() < epsilon {
        return Ok(<T::Float as NumCast>::from(0.0).unwrap());
    }
    
    let std_dev = <T::Float as Float>::sqrt(variance);
    let kurtosis: T::Float = sample.iter()
        .map(|&x| {
            let x_float = x.to_float();
            let normalized = (x_float - mean) / std_dev;
            let squared = normalized * normalized;
            squared * squared
        })
        .sum::<T::Float>() / n - <T::Float as NumCast>::from(3.0).unwrap(); // Excess kurtosis
    
    Ok(kurtosis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use robust_quantile::estimators::harrell_davis;
    use robust_core::primitives::ScalarBackend;
    
    #[test]
    fn test_skewness_symmetric() {
        let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let skewness = RobustSkewness::<f64, _>::new(ScalarBackend::new());
        let estimator = harrell_davis::<f64, _>(robust_core::execution::scalar_sequential());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::default(),
            robust_core::CachePolicy::NoCache
        );
        let result = skewness.estimate(&mut data, &estimator, &cache).unwrap();
        assert_relative_eq!(result, 0.0, epsilon = 0.1);
    }
    
    #[test]
    fn test_skewness_right_skewed() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0];
        let skewness = RobustSkewness::<f64, _>::new(ScalarBackend::new());
        let estimator = harrell_davis::<f64, _>(robust_core::execution::scalar_sequential());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::default(),
            robust_core::CachePolicy::NoCache
        );
        let result = skewness.estimate(&mut data, &estimator, &cache).unwrap();
        assert!(result > 0.0);
    }
    
    #[test]
    fn test_kurtosis_normal_like() {
        // Approximately normal data
        let mut data = vec![-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
        let kurtosis = RobustKurtosis::<f64, _>::new(ScalarBackend::new());
        let estimator = harrell_davis::<f64, _>(robust_core::execution::scalar_sequential());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::default(),
            robust_core::CachePolicy::NoCache
        );
        let result = kurtosis.estimate(&mut data, &estimator, &cache).unwrap();
        // Should be close to normal kurtosis value
        assert!(result.is_finite());
    }
    
    #[test]
    fn test_robust_skewness_function() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let estimator = harrell_davis::<f64, _>(robust_core::execution::scalar_sequential());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::default(),
            robust_core::CachePolicy::NoCache
        );
        let result = robust_skewness(&mut data, &estimator, &cache).unwrap();
        assert_relative_eq!(result, 0.0, epsilon = 0.1);
    }
    
    #[test]
    fn test_medcouple_skewness_function() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0];
        let estimator = harrell_davis::<f64, _>(robust_core::execution::scalar_sequential());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::default(),
            robust_core::CachePolicy::NoCache
        );
        let result = medcouple_skewness(&mut data, &estimator, &cache).unwrap();
        assert!(result > 0.0);
    }
    
    #[test]
    fn test_robust_kurtosis_function() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let estimator = harrell_davis::<f64, _>(robust_core::execution::scalar_sequential());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::default(),
            robust_core::CachePolicy::NoCache
        );
        let result = robust_kurtosis(&mut data, &estimator, &cache).unwrap();
        assert!(result.is_finite());
    }
    
    #[test]
    fn test_crow_siddiqui_function() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let estimator = harrell_davis::<f64, _>(robust_core::execution::scalar_sequential());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::default(),
            robust_core::CachePolicy::NoCache
        );
        let result = crow_siddiqui_kurtosis(&mut data, &estimator, &cache).unwrap();
        assert!(result.is_finite());
    }
}