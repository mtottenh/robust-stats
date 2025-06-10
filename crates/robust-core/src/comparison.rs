//! Two-sample comparison operations for robust statistical analysis
//!
//! This module provides traits and implementations for comparing distributional
//! properties between two samples, supporting both shift (A - B) and ratio (A / B)
//! operations. These are fundamental for confidence interval estimation and
//! effect size calculations.

use crate::{Result, Error, Numeric};
use num_traits::{Zero, NumCast, Float};

/// Marker trait for linear two-sample comparisons
///
/// Linear comparisons preserve the mathematical properties needed for
/// certain confidence interval methods (like Maritz-Jarrett). These are
/// operations of the form: a*X + b*Y + c, where a, b, c are constants.
///
/// Examples of linear comparisons:
/// - Shift: X - Y (a=1, b=-1, c=0)  
/// - Weighted average: 0.3*X + 0.7*Y
///
/// Linear comparisons preserve asymptotic normality and allow variance 
/// to be computed as Var(A - B) = Var(A) + Var(B) for independent samples.
/// This is required for methods like Maritz-Jarrett that rely on asymptotic theory.
///
/// This trait is a marker trait with no methods - it's used purely for
/// type-level guarantees.
pub trait LinearComparison {}

/// Core trait for two-sample comparison operations
///
/// This trait defines how to compare a specific distributional property
/// between two samples. It's parameterized by an estimator type E to enable
/// dependency injection and reuse of computational components.
pub trait TwoSampleComparison<E, T: Numeric = f64> {
    /// Output type of the comparison operation
    type Output;
    
    /// Calculate the shift (difference) between two samples: property(A) - property(B)
    ///
    /// # Arguments
    /// * `estimator` - The estimator to use for calculating the property
    /// * `sample_a` - First sample
    /// * `sample_b` - Second sample  
    /// * `cache` - Cached computational state for the estimator
    fn shift(
        &self,
        estimator: &E,
        sample_a: &[T],
        sample_b: &[T],
        cache: &E::State,
    ) -> Result<Self::Output>
    where
        E: StatefulEstimator<T>;
    
    /// Calculate the ratio between two samples: property(A) / property(B)
    ///
    /// # Arguments
    /// * `estimator` - The estimator to use for calculating the property
    /// * `sample_a` - First sample (numerator)
    /// * `sample_b` - Second sample (denominator)
    /// * `cache` - Cached computational state for the estimator
    fn ratio(
        &self,
        estimator: &E,
        sample_a: &[T],
        sample_b: &[T], 
        cache: &E::State,
    ) -> Result<Self::Output>
    where
        E: StatefulEstimator<T>;
}

/// Trait for estimators that have reusable computational state
///
/// This enables cache sharing across multiple comparison operations
/// for optimal performance.
pub trait StatefulEstimator<T: Numeric = f64> {
    /// Type representing cached computational state
    type State;
    
    /// Estimate the property using cached state
    fn estimate_with_cache(&self, sample: &[T], cache: &Self::State) -> Result<T::Float>;
    
    /// Estimate the property using sorted data with cached state  
    fn estimate_sorted_with_cache(&self, sorted_sample: &[T], cache: &Self::State) -> Result<T::Float> {
        self.estimate_with_cache(sorted_sample, cache)
    }
}

/// Simple shift comparison for single-valued properties
///
/// Computes property(A) - property(B) for any estimator that returns a single value.
/// This is a linear combination, suitable for Maritz-Jarrett and other asymptotic methods.
#[derive(Debug, Clone)]
pub struct ShiftComparison;

impl LinearComparison for ShiftComparison {}

impl<E, T> TwoSampleComparison<E, T> for ShiftComparison 
where
    E: StatefulEstimator<T>,
    T: Numeric,
{
    type Output = T::Float;
    
    fn shift(
        &self,
        estimator: &E,
        sample_a: &[T],
        sample_b: &[T],
        cache: &E::State,
    ) -> Result<Self::Output> {
        let estimate_a = estimator.estimate_with_cache(sample_a, cache)?;
        let estimate_b = estimator.estimate_with_cache(sample_b, cache)?;
        Ok(estimate_a - estimate_b)
    }
    
    fn ratio(
        &self,
        estimator: &E,
        sample_a: &[T],
        sample_b: &[T],
        cache: &E::State,
    ) -> Result<Self::Output> {
        let estimate_a = estimator.estimate_with_cache(sample_a, cache)?;
        let estimate_b = estimator.estimate_with_cache(sample_b, cache)?;
        
        if estimate_b.abs() < <T::Float as num_traits::NumCast>::from(f64::EPSILON).unwrap() {
            return Err(Error::Computation("Division by zero in ratio comparison".to_string()));
        }
        
        Ok(estimate_a / estimate_b)
    }
}

/// Ratio comparison for single-valued properties  
///
/// Computes property(A) / property(B) for any estimator that returns a single value.
/// This is a non-linear combination, NOT suitable for Maritz-Jarrett.
#[derive(Debug, Clone)]
pub struct RatioComparison;

// Note: RatioComparison does NOT implement LinearComparison

impl<E, T> TwoSampleComparison<E, T> for RatioComparison
where
    E: StatefulEstimator<T>,
    T: Numeric,
{
    type Output = T::Float;
    
    fn shift(
        &self,
        estimator: &E,
        sample_a: &[T],
        sample_b: &[T],
        cache: &E::State,
    ) -> Result<Self::Output> {
        // For ratio comparison, "shift" doesn't make sense, but we provide it for API completeness
        // Could be interpreted as log(A/B) = log(A) - log(B)
        let estimate_a = estimator.estimate_with_cache(sample_a, cache)?;
        let estimate_b = estimator.estimate_with_cache(sample_b, cache)?;
        
        if estimate_a <= <T::Float as num_traits::Zero>::zero() || estimate_b <= <T::Float as num_traits::Zero>::zero() {
            return Err(Error::Computation("Logarithm of non-positive value in ratio shift".to_string()));
        }
        
        let a_f64: f64 = estimate_a.into();
        let b_f64: f64 = estimate_b.into();
        Ok(<T::Float as num_traits::NumCast>::from(a_f64.ln() - b_f64.ln()).unwrap())
    }
    
    fn ratio(
        &self,
        estimator: &E,
        sample_a: &[T],
        sample_b: &[T],
        cache: &E::State,
    ) -> Result<Self::Output> {
        let estimate_a = estimator.estimate_with_cache(sample_a, cache)?;
        let estimate_b = estimator.estimate_with_cache(sample_b, cache)?;
        
        if estimate_b.abs() < <T::Float as num_traits::NumCast>::from(f64::EPSILON).unwrap() {
            return Err(Error::Computation("Division by zero in ratio comparison".to_string()));
        }
        
        Ok(estimate_a / estimate_b)
    }
}

/// Batch comparison for multiple quantiles
///
/// Enables efficient computation of shifts/ratios across many quantiles
/// simultaneously, such as "shift over quantiles 0.01->0.99".
/// This is a linear combination at each quantile level.
#[derive(Debug, Clone)]
pub struct QuantileShiftComparison {
    /// List of quantiles to compute (e.g., [0.01, 0.02, ..., 0.99])
    pub quantiles: Vec<f64>,
}

impl LinearComparison for QuantileShiftComparison {}

impl QuantileShiftComparison {
    /// Create a new quantile shift comparison
    pub fn new(quantiles: Vec<f64>) -> Result<Self> {
        if quantiles.is_empty() {
            return Err(Error::InvalidInput("Quantiles list cannot be empty".to_string()));
        }
        
        for &q in &quantiles {
            if !(0.0..=1.0).contains(&q) {
                return Err(Error::InvalidInput(format!("Invalid quantile: {q}")));
            }
        }
        
        Ok(Self { quantiles })
    }
    
    /// Create a comparison for quantiles from 0.01 to 0.99 in steps of 0.01
    pub fn range_01_to_99() -> Self {
        let quantiles = (1..100).map(|i| i as f64 / 100.0).collect();
        Self { quantiles }
    }
    
    /// Create a comparison for specific quantile ranges
    pub fn range(start: f64, end: f64, step: f64) -> Result<Self> {
        if start >= end || step <= 0.0 {
            return Err(Error::InvalidInput("Invalid range parameters".to_string()));
        }
        
        let mut quantiles = Vec::new();
        let mut current = start;
        while current <= end {
            quantiles.push(current);
            current += step;
        }
        
        Self::new(quantiles)
    }
}

/// Trait for quantile estimators that support batch operations
pub trait BatchQuantileEstimator<T: Numeric = f64>: StatefulEstimator<T> {
    /// Estimate multiple quantiles efficiently using cached state
    fn estimate_quantiles_with_cache(
        &self,
        sample: &[T],
        quantiles: &[f64],
        cache: &Self::State,
    ) -> Result<Vec<T::Float>>;
    
    /// Estimate multiple quantiles from sorted data with cached state
    fn estimate_quantiles_sorted_with_cache(
        &self,
        sorted_sample: &[T],
        quantiles: &[f64],
        cache: &Self::State,
    ) -> Result<Vec<T::Float>> {
        self.estimate_quantiles_with_cache(sorted_sample, quantiles, cache)
    }
}

impl<E, T> TwoSampleComparison<E, T> for QuantileShiftComparison
where
    E: BatchQuantileEstimator<T>,
    T: Numeric,
{
    type Output = Vec<T::Float>;
    
    fn shift(
        &self,
        estimator: &E,
        sample_a: &[T],
        sample_b: &[T],
        cache: &E::State,
    ) -> Result<Self::Output> {
        let estimates_a = estimator.estimate_quantiles_with_cache(sample_a, &self.quantiles, cache)?;
        let estimates_b = estimator.estimate_quantiles_with_cache(sample_b, &self.quantiles, cache)?;
        
        if estimates_a.len() != estimates_b.len() {
            return Err(Error::Computation("Quantile estimate vectors have different lengths".to_string()));
        }
        
        Ok(estimates_a.into_iter()
            .zip(estimates_b)
            .map(|(a, b)| a - b)
            .collect())
    }
    
    fn ratio(
        &self,
        estimator: &E,
        sample_a: &[T],
        sample_b: &[T],
        cache: &E::State,
    ) -> Result<Self::Output> {
        let estimates_a = estimator.estimate_quantiles_with_cache(sample_a, &self.quantiles, cache)?;
        let estimates_b = estimator.estimate_quantiles_with_cache(sample_b, &self.quantiles, cache)?;
        
        if estimates_a.len() != estimates_b.len() {
            return Err(Error::Computation("Quantile estimate vectors have different lengths".to_string()));
        }
        
        let mut ratios = Vec::with_capacity(estimates_a.len());
        for (a, b) in estimates_a.into_iter().zip(estimates_b.into_iter()) {
            if b.abs() < <T::Float as NumCast>::from(f64::EPSILON).unwrap() {
                return Err(Error::Computation(
                    format!("Division by zero in quantile ratio comparison at quantile with value {}", Into::<f64>::into(b))
                ));
            }
            ratios.push(a / b);
        }
        
        Ok(ratios)
    }
}

/// Batch comparison for multiple quantiles using ratio operations
/// This is a non-linear combination, NOT suitable for Maritz-Jarrett.
#[derive(Debug, Clone)]
pub struct QuantileRatioComparison {
    /// List of quantiles to compute
    pub quantiles: Vec<f64>,
}

// Note: QuantileRatioComparison does NOT implement LinearComparison

impl QuantileRatioComparison {
    /// Create a new quantile ratio comparison
    pub fn new(quantiles: Vec<f64>) -> Result<Self> {
        QuantileShiftComparison::new(quantiles.clone())?; // Validate using existing validation
        Ok(Self { quantiles })
    }
    
    /// Create a comparison for quantiles from 0.01 to 0.99 in steps of 0.01
    pub fn range_01_to_99() -> Self {
        let quantiles = (1..100).map(|i| i as f64 / 100.0).collect();
        Self { quantiles }
    }
}

impl<E, T> TwoSampleComparison<E, T> for QuantileRatioComparison
where
    E: BatchQuantileEstimator<T>,
    T: Numeric,
{
    type Output = Vec<T::Float>;
    
    fn shift(
        &self,
        estimator: &E,
        sample_a: &[T],
        sample_b: &[T],
        cache: &E::State,
    ) -> Result<Self::Output> {
        // For ratio comparison, shift = log(A) - log(B) = log(A/B)
        let estimates_a = estimator.estimate_quantiles_with_cache(sample_a, &self.quantiles, cache)?;
        let estimates_b = estimator.estimate_quantiles_with_cache(sample_b, &self.quantiles, cache)?;
        
        if estimates_a.len() != estimates_b.len() {
            return Err(Error::Computation("Quantile estimate vectors have different lengths".to_string()));
        }
        
        let mut log_ratios = Vec::with_capacity(estimates_a.len());
        for (a, b) in estimates_a.into_iter().zip(estimates_b.into_iter()) {
            if a <= <T::Float as Zero>::zero() || b <= <T::Float as Zero>::zero() {
                return Err(Error::Computation(
                    "Logarithm of non-positive quantile value in ratio shift".to_string()
                ));
            }
            let a_f64: f64 = a.into();
            let b_f64: f64 = b.into();
            log_ratios.push(<T::Float as NumCast>::from(a_f64.ln() - b_f64.ln()).unwrap());
        }
        
        Ok(log_ratios)
    }
    
    fn ratio(
        &self,
        estimator: &E,
        sample_a: &[T],
        sample_b: &[T],
        cache: &E::State,
    ) -> Result<Self::Output> {
        let estimates_a = estimator.estimate_quantiles_with_cache(sample_a, &self.quantiles, cache)?;
        let estimates_b = estimator.estimate_quantiles_with_cache(sample_b, &self.quantiles, cache)?;
        
        if estimates_a.len() != estimates_b.len() {
            return Err(Error::Computation("Quantile estimate vectors have different lengths".to_string()));
        }
        
        let mut ratios = Vec::with_capacity(estimates_a.len());
        for (a, b) in estimates_a.into_iter().zip(estimates_b.into_iter()) {
            if b.abs() < <T::Float as NumCast>::from(f64::EPSILON).unwrap() {
                return Err(Error::Computation(
                    format!("Division by zero in quantile ratio comparison at quantile with value {}", Into::<f64>::into(b))
                ));
            }
            ratios.push(a / b);
        }
        
        Ok(ratios)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock estimator for testing
    #[derive(Clone)]
    struct MockEstimator;
    
    struct MockCache;
    
    impl StatefulEstimator for MockEstimator {
        type State = MockCache;
        
        fn estimate_with_cache(&self, sample: &[f64], _cache: &Self::State) -> Result<f64> {
            Ok(sample.iter().sum::<f64>() / sample.len() as f64)
        }
    }
    
    impl BatchQuantileEstimator for MockEstimator {
        fn estimate_quantiles_with_cache(
            &self,
            sample: &[f64],
            quantiles: &[f64],
            _cache: &Self::State,
        ) -> Result<Vec<f64>> {
            let mut sorted = sample.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            Ok(quantiles.iter()
                .map(|&q| {
                    let index = (q * (sorted.len() - 1) as f64) as usize;
                    sorted[index]
                })
                .collect())
        }
    }
    
    #[test]
    fn test_shift_comparison() {
        let estimator = MockEstimator;
        let cache = MockCache;
        let comparison = ShiftComparison;
        
        let sample_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample_b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        
        let result = comparison.shift(&estimator, &sample_a, &sample_b, &cache).unwrap();
        assert!((result - (-1.0)).abs() < 1e-10); // mean(a) - mean(b) = 3 - 4 = -1
    }
    
    #[test]
    fn test_ratio_comparison() {
        let estimator = MockEstimator;
        let cache = MockCache;
        let comparison = ShiftComparison;
        
        let sample_a = vec![2.0, 4.0, 6.0];
        let sample_b = vec![1.0, 2.0, 3.0];
        
        let result = comparison.ratio(&estimator, &sample_a, &sample_b, &cache).unwrap();
        assert!((result - 2.0).abs() < 1e-10); // mean(a) / mean(b) = 4 / 2 = 2
    }
    
    #[test]
    fn test_quantile_shift_comparison() {
        let estimator = MockEstimator;
        let cache = MockCache;
        let comparison = QuantileShiftComparison::new(vec![0.0, 0.5, 1.0]).unwrap();
        
        let sample_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample_b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        
        let result = comparison.shift(&estimator, &sample_a, &sample_b, &cache).unwrap();
        assert_eq!(result.len(), 3);
        // Each quantile of a should be 1 less than corresponding quantile of b
        for diff in result {
            assert!((diff - (-1.0)).abs() < 1e-10);
        }
    }
}