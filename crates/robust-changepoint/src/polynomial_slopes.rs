//! Polynomial-based slopes changepoint detection - redesigned
//!
//! This module provides advanced changepoint detection by fitting polynomials
//! to rolling windows of data and analyzing the slopes. This is particularly
//! useful for detecting steady-state transitions and more complex trend changes.

use crate::kernel::{PolynomialKernel, WindowKernel};
use crate::traits::{
    ChangePointDetector, ChangePointDetectorProperties,
    ConfigurableDetector,
};
use crate::types::{ChangePoint, ChangePointResult, ChangeType};
use crate::visualization::{ChangePointVisualizer, NullChangePointVisualizer};
use robust_core::{ComputePrimitives, Error, Result, StatisticalKernel, Numeric};
use robust_spread::SpreadEstimator;
use robust_quantile::QuantileEstimator;
use num_traits::{FromPrimitive, NumCast, One, Zero};

/// Parameters for polynomial slopes detection
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialSlopesParameters<T: Numeric> {
    /// Window size for polynomial fitting
    pub window_size: usize,
    /// Degree of polynomial to fit (1 = linear, 2 = quadratic, etc.)
    pub polynomial_degree: usize,
    /// Threshold multiplier for slope change detection
    pub slope_threshold_multiplier: T::Float,
    /// Threshold multiplier for steady-state detection
    pub steady_state_threshold_multiplier: T::Float,
    /// Number of consecutive windows needed for steady-state confirmation
    pub steady_state_windows: usize,
    /// Minimum confidence for a detection
    pub min_confidence: f64,
    /// Whether to normalize time values for numerical stability
    pub normalize_time: bool,
}

impl<T: Numeric> Default for PolynomialSlopesParameters<T> 
where
    T::Float: FromPrimitive,
{
    fn default() -> Self {
        Self {
            window_size: 25,
            polynomial_degree: 1,
            slope_threshold_multiplier: T::Float::from_f64(3.0).unwrap(),
            steady_state_threshold_multiplier: T::Float::from_f64(2.0).unwrap(),
            steady_state_windows: 3,
            min_confidence: 0.5,
            normalize_time: true,
        }
    }
}

/// Parameters for adaptive polynomial slopes detection
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptivePolynomialSlopesParameters<T: Numeric> {
    /// Base parameters
    pub base: PolynomialSlopesParameters<T>,
    /// Whether to use acceleration (second derivative) analysis
    pub use_acceleration: bool,
    /// Acceleration threshold multiplier
    pub acceleration_threshold_multiplier: T::Float,
    /// Weight for acceleration signal when combined with slope
    pub acceleration_weight: f64,
}

impl<T: Numeric> Default for AdaptivePolynomialSlopesParameters<T>
where
    T::Float: FromPrimitive,
{
    fn default() -> Self {
        Self {
            base: PolynomialSlopesParameters::default(),
            use_acceleration: false,
            acceleration_threshold_multiplier: T::Float::from_f64(2.5).unwrap(),
            acceleration_weight: 0.3,
        }
    }
}

/// Polynomial slopes changepoint detector using kernels
#[derive(Clone)]
pub struct PolynomialSlopesDetector<T: Numeric, P: ComputePrimitives<T>, V = NullChangePointVisualizer> {
    polynomial_kernel: PolynomialKernel<T, P>,
    window_kernel: WindowKernel<T, P>,
    params: PolynomialSlopesParameters<T>,
    visualizer: V,
}

impl<T: Numeric, P: ComputePrimitives<T>, V: ChangePointVisualizer> std::fmt::Debug for PolynomialSlopesDetector<T, P, V>
where
    T::Float: std::fmt::Debug,
    P: std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolynomialSlopesDetector")
            .field("polynomial_kernel", &self.polynomial_kernel)
            .field("window_kernel", &self.window_kernel)
            .field("params", &self.params)
            .field("visualizer", &self.visualizer)
            .finish()
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> PolynomialSlopesDetector<T, P, NullChangePointVisualizer> 
where
    T::Float: FromPrimitive,
{
    /// Create a new polynomial slopes detector
    pub fn new(
        primitives: P,
        window_size: usize,
        polynomial_degree: usize,
        slope_threshold_multiplier: T::Float,
    ) -> Self {
        Self {
            polynomial_kernel: PolynomialKernel::new(primitives.clone(), polynomial_degree),
            window_kernel: WindowKernel::new(primitives, window_size),
            params: PolynomialSlopesParameters {
                window_size,
                polynomial_degree,
                slope_threshold_multiplier,
                steady_state_threshold_multiplier: T::Float::from_f64(2.0).unwrap(),
                steady_state_windows: 3,
                min_confidence: 0.5,
                normalize_time: true,
            },
            visualizer: NullChangePointVisualizer,
        }
    }

    /// Create with full parameters
    pub fn with_params(primitives: P, params: PolynomialSlopesParameters<T>) -> Self {
        Self {
            polynomial_kernel: PolynomialKernel::new(primitives.clone(), params.polynomial_degree),
            window_kernel: WindowKernel::new(primitives, params.window_size),
            params,
            visualizer: NullChangePointVisualizer,
        }
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, V: ChangePointVisualizer> PolynomialSlopesDetector<T, P, V> 
where
    T::Float: FromPrimitive + PartialOrd,
{
    /// Create with visualizer
    pub fn with_visualizer(
        primitives: P,
        params: PolynomialSlopesParameters<T>,
        visualizer: V,
    ) -> Self {
        Self {
            polynomial_kernel: PolynomialKernel::new(primitives.clone(), params.polynomial_degree),
            window_kernel: WindowKernel::new(primitives, params.window_size),
            params,
            visualizer,
        }
    }

    /// Generate time values for polynomial fitting
    fn generate_time_values(&self) -> Vec<T> {
        let times: Vec<T> = (0..self.params.window_size).map(|i| T::from_f64(i as f64)).collect();
        
        if self.params.normalize_time {
            // Normalize to [-1, 1] for numerical stability
            let max_val = T::from_f64((self.params.window_size - 1) as f64);
            let two = T::from_f64(2.0);
            let one = <T as Numeric>::one();
            times.into_iter().map(|t| {
                let normalized = (t.to_float() * two.to_float()) / max_val.to_float() - one.to_float();
                T::from_f64(NumCast::from(normalized).unwrap())
            }).collect()
        } else {
            times
        }
    }

    /// Compute slopes for all windows
    fn compute_window_slopes(&self, sample: &[T]) -> Result<Vec<T::Float>> 
    where
        T::Float: nalgebra::Scalar + nalgebra::RealField,
    {
        let times = self.generate_time_values();
        
        let slopes = self.window_kernel.compute_window_stats(sample, |window, _prims| {
            match self.polynomial_kernel.fit_polynomial(&times, window) {
                Ok(coeffs) => {
                    // For normalized time, the slope coefficient needs adjustment
                    if coeffs.len() > 1 {
                        if self.params.normalize_time {
                            // Adjust slope for normalized time scale
                            let time_scale = T::Float::from_f64((self.params.window_size - 1) as f64 / 2.0).unwrap();
                            coeffs[1] * time_scale
                        } else {
                            coeffs[1]
                        }
                    } else {
                        T::Float::zero()
                    }
                }
                Err(_) => T::Float::zero(),
            }
        });
        
        Ok(slopes)
    }

    /// Compute slope differences
    fn compute_slope_differences(&self, slopes: &[T::Float]) -> Vec<T::Float> {
        let mut slope_diffs = Vec::with_capacity(slopes.len().saturating_sub(1));
        for i in 1..slopes.len() {
            let diff = slopes[i] - slopes[i - 1];
            slope_diffs.push(if diff < T::Float::zero() { -diff } else { diff });
        }
        slope_diffs
    }

    /// Detect steady state periods
    fn detect_steady_states(&self, slopes: &[T::Float], steady_threshold: T::Float) -> Vec<ChangePoint> {
        let mut steady_states = Vec::new();
        let mut consecutive_count = 0;
        
        for (i, &slope) in slopes.iter().enumerate() {
            let abs_slope = if slope < T::Float::zero() { -slope } else { slope };
            if abs_slope < steady_threshold {
                consecutive_count += 1;
                
                if consecutive_count >= self.params.steady_state_windows {
                    // Found a steady state
                    let epsilon = T::Float::from_f64(1e-10).unwrap();
                    let one = T::Float::one();
                    let confidence_float = if (steady_threshold / (abs_slope + epsilon)) < one {
                        steady_threshold / (abs_slope + epsilon)
                    } else {
                        one
                    };
                    let confidence = NumCast::from(confidence_float).unwrap();
                    steady_states.push(ChangePoint::with_type(
                        i + self.params.window_size,
                        confidence,
                        ChangeType::SteadyState,
                    ));
                }
            } else {
                consecutive_count = 0;
            }
        }
        
        steady_states
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, V: ChangePointVisualizer> ChangePointDetectorProperties 
    for PolynomialSlopesDetector<T, P, V> 
{
    fn algorithm_name(&self) -> &'static str {
        "PolynomialSlopes"
    }
    
    fn minimum_sample_size(&self) -> usize {
        self.params.window_size * 2
    }
}

impl<T, P, S, Q, V> ChangePointDetector<T, S, Q> for PolynomialSlopesDetector<T, P, V>
where
    T: Numeric,
    P: ComputePrimitives<T>,
    S: SpreadEstimator<T, Q>,
    Q: QuantileEstimator<T>,
    V: ChangePointVisualizer,
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
        
        // Step 1: Compute slopes for all windows using kernel
        let slopes = self.compute_window_slopes(sample)?;
        
        if slopes.is_empty() {
            return Ok(ChangePointResult::new(
                vec![],
                self.algorithm_name().to_string(),
                sample.len(),
                vec![],
            ));
        }
        
        // Step 2: Calculate slope differences
        let slope_diffs = self.compute_slope_differences(&slopes);
        
        // Step 3: Determine thresholds using spread estimator (single copy/sort)
        // Convert to T for spread estimation
        let diffs_t: Vec<T> = slope_diffs.iter()
            .map(|&d| T::from_f64(NumCast::from(d).unwrap()))
            .collect();
        let mut diffs_sorted = diffs_t.clone();
        diffs_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let slope_threshold = spread_est.estimate_sorted(&diffs_sorted, quantile_est, cache)
            .map_err(|e| Error::Computation(format!("Spread estimation failed: {}", e)))? 
                             * self.params.slope_threshold_multiplier;
        
        // For steady state, use slopes directly (convert abs slopes)
        let slopes_abs_t: Vec<T> = slopes.iter()
            .map(|&s| {
                let abs_s = if s < T::Float::zero() { -s } else { s };
                T::from_f64(NumCast::from(abs_s).unwrap())
            })
            .collect();
        let mut slopes_sorted = slopes_abs_t.clone();
        slopes_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let steady_threshold = spread_est.estimate_sorted(&slopes_sorted, quantile_est, cache)
            .map_err(|e| Error::Computation(format!("Spread estimation failed: {}", e)))? 
                              * self.params.steady_state_threshold_multiplier;
        
        // Step 4: Detect trend changepoints
        let mut changepoints = Vec::new();
        
        for (i, &diff) in slope_diffs.iter().enumerate() {
            if diff > slope_threshold {
                let two = T::Float::from_f64(2.0).unwrap();
                let one = T::Float::one();
                let confidence_float = if (diff / (two * slope_threshold)) < one {
                    diff / (two * slope_threshold)
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
        
        // Step 5: Detect steady states
        let steady_states = self.detect_steady_states(&slopes, steady_threshold);
        changepoints.extend(steady_states);
        
        // Remove consecutive detections
        changepoints.dedup_by(|a, b| a.index.abs_diff(b.index) <= 2);
        
        // Sort by index
        changepoints.sort_by_key(|cp| cp.index);
        
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
        // Polynomial fitting requires temporal order
        self.detect(sorted_sample, spread_est, quantile_est, cache)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, V: ChangePointVisualizer> ConfigurableDetector 
    for PolynomialSlopesDetector<T, P, V>
where
    T::Float: FromPrimitive,
{
    type Parameters = PolynomialSlopesParameters<T>;
    
    fn with_parameters(_params: Self::Parameters) -> Self {
        unreachable!("Use new() or with_params() to create PolynomialSlopesDetector")
    }
    
    fn parameters(&self) -> &Self::Parameters {
        &self.params
    }
    
    fn set_parameters(&mut self, params: Self::Parameters) {
        let polynomial_degree = params.polynomial_degree;
        let window_size = params.window_size;
        self.params = params;
        // Update kernels if needed
        self.polynomial_kernel = PolynomialKernel::new(
            self.polynomial_kernel.primitives().clone(),
            polynomial_degree,
        );
        self.window_kernel = WindowKernel::new(
            self.window_kernel.primitives().clone(),
            window_size,
        );
    }
}

/// Adaptive polynomial slopes detector with acceleration analysis
#[derive(Clone)]
pub struct AdaptivePolynomialSlopesDetector<T: Numeric, P: ComputePrimitives<T>, V = NullChangePointVisualizer> {
    base_detector: PolynomialSlopesDetector<T, P, V>,
    params: AdaptivePolynomialSlopesParameters<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>, V: ChangePointVisualizer> std::fmt::Debug for AdaptivePolynomialSlopesDetector<T, P, V>
where
    T::Float: std::fmt::Debug,
    P: std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptivePolynomialSlopesDetector")
            .field("base_detector", &self.base_detector)
            .field("params", &self.params)
            .finish()
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> AdaptivePolynomialSlopesDetector<T, P, NullChangePointVisualizer>
where
    T::Float: FromPrimitive,
{
    /// Create a new adaptive polynomial slopes detector
    pub fn new(
        primitives: P,
        window_size: usize,
        polynomial_degree: usize,
    ) -> Self {
        let params = AdaptivePolynomialSlopesParameters {
            base: PolynomialSlopesParameters {
                window_size,
                polynomial_degree,
                ..Default::default()
            },
            ..Default::default()
        };
        
        Self {
            base_detector: PolynomialSlopesDetector::with_params(primitives, params.base.clone()),
            params,
        }
    }

    /// Create with full parameters
    pub fn with_params(primitives: P, params: AdaptivePolynomialSlopesParameters<T>) -> Self {
        Self {
            base_detector: PolynomialSlopesDetector::with_params(primitives, params.base.clone()),
            params,
        }
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, V: ChangePointVisualizer> AdaptivePolynomialSlopesDetector<T, P, V>
where
    T::Float: FromPrimitive + PartialOrd,
{
    /// Create with visualizer
    pub fn with_visualizer(
        primitives: P,
        params: AdaptivePolynomialSlopesParameters<T>,
        visualizer: V,
    ) -> Self {
        Self {
            base_detector: PolynomialSlopesDetector::with_visualizer(primitives, params.base.clone(), visualizer),
            params,
        }
    }

    /// Compute acceleration (second derivative) of slopes
    fn compute_acceleration(&self, slopes: &[T::Float]) -> Vec<T::Float> {
        let mut acceleration = Vec::with_capacity(slopes.len().saturating_sub(2));
        let two = T::Float::from_f64(2.0).unwrap();
        for i in 2..slopes.len() {
            // Second difference approximation
            let accel = slopes[i] - two * slopes[i - 1] + slopes[i - 2];
            acceleration.push(if accel < T::Float::zero() { -accel } else { accel });
        }
        acceleration
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, V: ChangePointVisualizer> ChangePointDetectorProperties 
    for AdaptivePolynomialSlopesDetector<T, P, V> 
{
    fn algorithm_name(&self) -> &'static str {
        "AdaptivePolynomialSlopes"
    }
    
    fn minimum_sample_size(&self) -> usize {
        self.base_detector.minimum_sample_size()
    }
}

impl<T, P, S, Q, V> ChangePointDetector<T, S, Q> for AdaptivePolynomialSlopesDetector<T, P, V>
where
    T: Numeric,
    P: ComputePrimitives<T>,
    S: SpreadEstimator<T, Q>,
    Q: QuantileEstimator<T>,
    V: ChangePointVisualizer,
    T::Float: FromPrimitive + PartialOrd + nalgebra::Scalar + nalgebra::RealField,
{
    fn detect(
        &self,
        sample: &[T],
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<ChangePointResult> {
        // Start with base detection
        let result = self.base_detector.detect(sample, spread_est, quantile_est, cache)?;
        
        if !self.params.use_acceleration {
            return Ok(result);
        }
        
        // Add acceleration-based detection
        let slopes = self.base_detector.compute_window_slopes(sample)?;
        
        if slopes.len() < 3 {
            return Ok(result);
        }
        
        let acceleration = self.compute_acceleration(&slopes);
        
        // Convert to T for spread estimation
        let accel_t: Vec<T> = acceleration.iter()
            .map(|&a| T::from_f64(NumCast::from(a).unwrap()))
            .collect();
        let mut accel_sorted = accel_t.clone();
        accel_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let accel_threshold = spread_est.estimate_sorted(&accel_sorted, quantile_est, cache)
            .map_err(|e| Error::Computation(format!("Spread estimation failed: {}", e)))? 
                             * self.params.acceleration_threshold_multiplier;
        
        // Detect acceleration changes
        let mut accel_changes = Vec::new();
        let two = T::Float::from_f64(2.0).unwrap();
        for (i, &accel) in acceleration.iter().enumerate() {
            if accel > accel_threshold {
                let one = T::Float::one();
                let confidence_float = if (accel / (two * accel_threshold)) < one {
                    accel / (two * accel_threshold)
                } else {
                    one
                };
                let confidence = NumCast::from(confidence_float).unwrap();
                if confidence >= self.params.base.min_confidence {
                    accel_changes.push(ChangePoint::with_type(
                        i + self.params.base.window_size + 2, // Account for second difference
                        confidence,
                        ChangeType::AccelerationChange,
                    ));
                }
            }
        }
        
        // Combine with base detections
        let mut all_changepoints = result.changepoints().to_vec();
        all_changepoints.extend(accel_changes);
        
        // Remove duplicates and sort
        all_changepoints.dedup_by(|a, b| a.index.abs_diff(b.index) <= 3);
        all_changepoints.sort_by_key(|cp| cp.index);
        
        Ok(ChangePointResult::new(
            all_changepoints,
            self.algorithm_name().to_string(),
            sample.len(),
            result.statistics().to_vec(),
        ))
    }
    
    fn detect_sorted(
        &self,
        sorted_sample: &[T],
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<ChangePointResult> {
        // Polynomial fitting requires temporal order
        self.detect(sorted_sample, spread_est, quantile_est, cache)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>, V: ChangePointVisualizer> ConfigurableDetector 
    for AdaptivePolynomialSlopesDetector<T, P, V>
where
    T::Float: FromPrimitive,
{
    type Parameters = AdaptivePolynomialSlopesParameters<T>;
    
    fn with_parameters(_params: Self::Parameters) -> Self {
        unreachable!("Use new() or with_params() to create AdaptivePolynomialSlopesDetector")
    }
    
    fn parameters(&self) -> &Self::Parameters {
        &self.params
    }
    
    fn set_parameters(&mut self, params: Self::Parameters) {
        let base_params = params.base.clone();
        self.params = params;
        self.base_detector.set_parameters(base_params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robust_core::{simd_sequential, ExecutionEngine};
    use robust_spread::StandardizedMad;
    use robust_quantile::estimators::harrell_davis;
    
    #[test]
    fn test_polynomial_slopes_detection() {
        let primitives = simd_sequential().primitives().clone();
        let engine = robust_core::simd_sequential();
        let quantile_est = harrell_davis(engine);
        let spread_est = StandardizedMad::new(primitives.clone());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::default(),
            robust_core::CachePolicy::NoCache,
        );
        
        let detector = PolynomialSlopesDetector::new(primitives, 10, 1, 3.0);
        
        // Data with changing trend (up then flat then down)
        let mut data = Vec::with_capacity(100);
        for i in 0..30 {
            data.push(i as f64);
        }
        for _i in 30..70 {
            data.push(30.0);
        }
        for i in 70..100 {
            data.push(30.0 - (i - 70) as f64);
        }
        
        let result = detector.detect(&data, &spread_est, &quantile_est, &cache).unwrap();
        
        // Should detect trend changes
        assert!(!result.changepoints().is_empty());
        
        // Should have some steady state detections around the flat section
        let steady_states: Vec<_> = result.changepoints().iter()
            .filter(|cp| matches!(cp.change_type, Some(ChangeType::SteadyState)))
            .collect();
        assert!(!steady_states.is_empty());
    }
    
    #[test]
    fn test_adaptive_with_acceleration() {
        let primitives = simd_sequential().primitives().clone();
        let engine = robust_core::simd_sequential();
        let quantile_est = harrell_davis(engine);
        let spread_est = StandardizedMad::new(primitives.clone());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::default(),
            robust_core::CachePolicy::NoCache,
        );
        
        let mut params = AdaptivePolynomialSlopesParameters::default();
        params.use_acceleration = true;
        params.base.window_size = 10;
        
        let detector = AdaptivePolynomialSlopesDetector::with_params(primitives, params);
        
        // Data with quadratic trend change
        let data: Vec<f64> = (0..100).map(|i| {
            let x = i as f64;
            if x < 50.0 {
                x
            } else {
                x + 0.1 * (x - 50.0).powi(2)
            }
        }).collect();
        
        let result = detector.detect(&data, &spread_est, &quantile_est, &cache).unwrap();
        
        // Should detect changes
        assert!(!result.changepoints().is_empty());
    }
}