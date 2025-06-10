//! Slopes-based changepoint detection
//!
//! This method detects changes in linear trends by analyzing the slopes
//! of consecutive segments of the time series.

use crate::kernel::{PolynomialKernel, WindowKernel};
use crate::traits::{
    ChangePointDetector, ChangePointDetectorProperties,
    ConfigurableDetector,
};
use crate::types::{ChangePoint, ChangePointResult, ChangeType};
use robust_core::{ComputePrimitives, Error, Result, StatisticalKernel, Numeric};
use robust_spread::SpreadEstimator;
use robust_quantile::QuantileEstimator;
use num_traits::{FromPrimitive, NumCast, Zero, One};

/// Parameters for slopes-based detection
#[derive(Debug, Clone, PartialEq)]
pub struct SlopesParameters<T: Numeric> {
    /// Window size for slope calculation
    pub window_size: usize,
    /// Threshold multiplier for slope difference detection
    pub threshold_multiplier: T::Float,
    /// Minimum confidence for a detection
    pub min_confidence: f64,
}

impl<T: Numeric> Default for SlopesParameters<T>
where
    T::Float: FromPrimitive,
{
    fn default() -> Self {
        Self {
            window_size: 10,
            threshold_multiplier: T::Float::from_f64(3.0).unwrap(),
            min_confidence: 0.5,
        }
    }
}

/// Slopes-based changepoint detector using kernels
#[derive(Clone)]
pub struct SlopesDetector<T: Numeric, P: ComputePrimitives<T>> {
    polynomial_kernel: PolynomialKernel<T, P>,
    window_kernel: WindowKernel<T, P>,
    params: SlopesParameters<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> std::fmt::Debug for SlopesDetector<T, P>
where
    T::Float: std::fmt::Debug,
    P: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlopesDetector")
            .field("polynomial_kernel", &self.polynomial_kernel)
            .field("window_kernel", &self.window_kernel)
            .field("params", &self.params)
            .finish()
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> SlopesDetector<T, P>
where
    T::Float: FromPrimitive,
{
    /// Create a new slopes detector
    pub fn new(primitives: P, window_size: usize, threshold_multiplier: T::Float) -> Self {
        Self {
            polynomial_kernel: PolynomialKernel::new(primitives.clone(), 1), // Linear fit
            window_kernel: WindowKernel::new(primitives, window_size),
            params: SlopesParameters {
                window_size,
                threshold_multiplier,
                min_confidence: 0.5,
            },
        }
    }

    /// Create with all parameters
    pub fn with_params(
        primitives: P,
        window_size: usize,
        threshold_multiplier: T::Float,
        min_confidence: f64,
    ) -> Self {
        Self {
            polynomial_kernel: PolynomialKernel::new(primitives.clone(), 1),
            window_kernel: WindowKernel::new(primitives, window_size),
            params: SlopesParameters {
                window_size,
                threshold_multiplier,
                min_confidence,
            },
        }
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> ChangePointDetectorProperties for SlopesDetector<T, P> {
    fn algorithm_name(&self) -> &'static str {
        "Slopes"
    }
    
    fn minimum_sample_size(&self) -> usize {
        self.params.window_size * 2
    }
}

impl<T, P, S, Q> ChangePointDetector<T, S, Q> for SlopesDetector<T, P>
where
    T: Numeric,
    P: ComputePrimitives<T>,
    S: SpreadEstimator<T, Q>,
    Q: QuantileEstimator<T>,
    T::Float: FromPrimitive + PartialOrd + nalgebra::Scalar + nalgebra::RealField,
{
    fn detect(
        &self,
        sample: &[T],
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<ChangePointResult> {
        if sample.len() < self.minimum_sample_size() {
            return Err(Error::InsufficientData {
                expected: self.minimum_sample_size(),
                actual: sample.len(),
            });
        }
        
        // Step 1: Calculate slopes for all windows using kernel
        let times: Vec<T> = (0..self.params.window_size).map(|i| T::from_f64(i as f64)).collect();
        
        let slopes = self.window_kernel.compute_window_stats(sample, |window, _prims| {
            match self.polynomial_kernel.fit_polynomial(&times, window) {
                Ok(coeffs) => {
                    // For linear fit, slope is the second coefficient
                    if coeffs.len() > 1 {
                        coeffs[1]
                    } else {
                        T::Float::zero()
                    }
                }
                Err(_) => T::Float::zero(),
            }
        });
        
        if slopes.is_empty() {
            return Ok(ChangePointResult::new(
                vec![],
                self.algorithm_name().to_string(),
                sample.len(),
                vec![],
            ));
        }
        
        // Step 2: Calculate slope differences
        let mut slope_diffs = Vec::with_capacity(slopes.len().saturating_sub(1));
        for i in 1..slopes.len() {
            let diff = slopes[i] - slopes[i - 1];
            slope_diffs.push(if diff < T::Float::zero() { -diff } else { diff });
        }
        
        // Step 3: Determine threshold using spread estimator
        // Convert to T for spread estimation
        let diffs_t: Vec<T> = slope_diffs.iter()
            .map(|&d| T::from_f64(NumCast::from(d).unwrap()))
            .collect();
        let mut diffs_sorted = diffs_t.clone();
        diffs_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let threshold = spread_est.estimate_sorted(&diffs_sorted, quantile_est, cache)
            .map_err(|e| Error::Computation(format!("Spread estimation failed: {}", e)))? 
                       * self.params.threshold_multiplier;
        
        // Step 4: Detect changepoints
        let mut changepoints = Vec::new();
        
        for (i, &diff) in slope_diffs.iter().enumerate() {
            if diff > threshold {
                let two = T::Float::from_f64(2.0).unwrap();
                let one = T::Float::one();
                let confidence_float = if (diff / (two * threshold)) < one {
                    diff / (two * threshold)
                } else {
                    one
                };
                let confidence = NumCast::from(confidence_float).unwrap();
                
                if confidence >= self.params.min_confidence {
                    changepoints.push(ChangePoint::with_type(
                        i + self.params.window_size,
                        confidence,
                        ChangeType::TrendChange,
                    ));
                }
            }
        }
        
        // Remove consecutive detections
        changepoints.dedup_by(|a, b| a.index.abs_diff(b.index) <= 2);
        
        // Convert slopes to f64 for statistics
        let statistics = slopes.iter().map(|&s| NumCast::from(s).unwrap()).collect();
        
        Ok(ChangePointResult::new(
            changepoints,
            self.algorithm_name().to_string(),
            sample.len(),
            statistics,
        ))
    }
    
    fn detect_sorted(
        &self,
        sorted_sample: &[T],
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<ChangePointResult> {
        // Slopes detection requires temporal order
        self.detect(sorted_sample, spread_est, quantile_est, cache)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> ConfigurableDetector for SlopesDetector<T, P>
where
    T::Float: FromPrimitive,
{
    type Parameters = SlopesParameters<T>;
    
    fn with_parameters(_params: Self::Parameters) -> Self {
        unreachable!("Use new() or with_params() to create SlopesDetector")
    }
    
    fn parameters(&self) -> &Self::Parameters {
        &self.params
    }
    
    fn set_parameters(&mut self, params: Self::Parameters) {
        let window_size = params.window_size;
        self.params = params;
        // Update window kernel if window size changed
        self.window_kernel = WindowKernel::new(
            self.window_kernel.primitives().clone(),
            window_size,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robust_core::{simd_sequential, ExecutionEngine};
    use robust_spread::StandardizedMad;
    use robust_quantile::estimators::harrell_davis;
    
    #[test]
    fn test_slopes_detection() {
        let primitives = simd_sequential().primitives().clone();
        let engine = robust_core::simd_sequential();
        let quantile_est = harrell_davis(engine);
        let spread_est = StandardizedMad::new(primitives.clone());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::default(),
            robust_core::CachePolicy::NoCache,
        );
        
        let detector = SlopesDetector::new(primitives, 10, 3.0);
        
        // Data with trend change (up then down)
        let mut data = Vec::with_capacity(100);
        for i in 0..50 {
            data.push(i as f64);
        }
        for i in 50..100 {
            data.push(50.0 - (i - 50) as f64);
        }
        
        let result = detector.detect(&data, &spread_est, &quantile_est, &cache).unwrap();
        
        // Should detect trend change around index 50
        assert!(!result.changepoints().is_empty(), "No changepoints detected!");
        
        // Look for any detection near the expected change point (index 50)
        let expected_change = 50;
        let tolerance = 15;
        let near_expected = result.changepoints().iter()
            .any(|cp| (cp.index as i32 - expected_change).abs() < tolerance);
        
        assert!(near_expected, 
            "No changepoint detected near expected position {}. Detected: {:?}", 
            expected_change, result.changepoints());
    }
    
    #[test]
    fn test_constant_data() {
        let primitives = simd_sequential().primitives().clone();
        let engine = robust_core::simd_sequential();
        let quantile_est = harrell_davis(engine);
        let spread_est = StandardizedMad::new(primitives.clone());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::default(),
            robust_core::CachePolicy::NoCache,
        );
        
        let detector = SlopesDetector::new(primitives, 5, 3.0);
        
        // Constant data should have no changepoints
        let data = vec![5.0; 50];
        let result = detector.detect(&data, &spread_est, &quantile_est, &cache).unwrap();
        
        assert!(result.changepoints().is_empty());
    }
}