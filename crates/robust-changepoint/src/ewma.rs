//! EWMA (Exponentially Weighted Moving Average) changepoint detection
//!
//! EWMA is particularly effective for detecting gradual changes and small shifts
//! in time series data by giving more weight to recent observations.

use crate::kernel::EwmaKernel;
use crate::traits::{
    ChangePointDetector, ChangePointDetectorProperties,
    ConfigurableDetector, OnlineDetector,
};
use crate::types::{ChangePoint, ChangePointResult, ChangeType};
use robust_core::{ComputePrimitives, Error, Result, StatisticalKernel, Numeric};
use robust_spread::SpreadEstimator;
use robust_quantile::QuantileEstimator;
use num_traits::{FromPrimitive, NumCast, Zero};

/// Parameters for EWMA detection
#[derive(Clone, PartialEq)]
pub struct EwmaParameters<T: Numeric> {
    /// Smoothing parameter (0 < lambda <= 1)
    pub lambda: T::Float,
    /// Detection threshold multiplier (typically 2-3)
    pub threshold_multiplier: T::Float,
    /// Target mean (if known, otherwise estimated from initial data)
    pub target_mean: Option<T::Float>,
    /// Number of initial observations to use for parameter estimation
    pub warmup_period: usize,
}

impl<T: Numeric> std::fmt::Debug for EwmaParameters<T>
where
    T::Float: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EwmaParameters")
            .field("lambda", &self.lambda)
            .field("threshold_multiplier", &self.threshold_multiplier)
            .field("target_mean", &self.target_mean)
            .field("warmup_period", &self.warmup_period)
            .finish()
    }
}

impl<T: Numeric> Default for EwmaParameters<T>
where
    T::Float: FromPrimitive,
{
    fn default() -> Self {
        Self {
            lambda: T::Float::from_f64(0.2).unwrap(),
            threshold_multiplier: T::Float::from_f64(3.0).unwrap(),
            target_mean: None,
            warmup_period: 20,
        }
    }
}

/// EWMA changepoint detector using parameterized design
#[derive(Clone)]
pub struct EwmaDetector<T: Numeric, P: ComputePrimitives<T>> {
    kernel: EwmaKernel<T, P>,
    params: EwmaParameters<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> std::fmt::Debug for EwmaDetector<T, P>
where
    T::Float: std::fmt::Debug,
    P: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EwmaDetector")
            .field("kernel", &self.kernel)
            .field("params", &self.params)
            .finish()
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> EwmaDetector<T, P> 
where
    T::Float: FromPrimitive + PartialOrd,
{
    /// Create a new EWMA detector
    pub fn new(primitives: P, lambda: T::Float, threshold_multiplier: T::Float) -> Self {
        let min_lambda = T::Float::from_f64(0.001).unwrap();
        let max_lambda = T::Float::from_f64(1.0).unwrap();
        let lambda = if lambda < min_lambda { min_lambda } else if lambda > max_lambda { max_lambda } else { lambda };
        
        Self {
            kernel: EwmaKernel::new(primitives, lambda),
            params: EwmaParameters {
                lambda,
                threshold_multiplier,
                target_mean: None,
                warmup_period: 20,
            },
        }
    }
    
    /// Create with a known target mean
    pub fn with_target_mean(
        primitives: P, 
        lambda: T::Float, 
        threshold_multiplier: T::Float, 
        target_mean: T::Float
    ) -> Self {
        let min_lambda = T::Float::from_f64(0.001).unwrap();
        let max_lambda = T::Float::from_f64(1.0).unwrap();
        let lambda = if lambda < min_lambda { min_lambda } else if lambda > max_lambda { max_lambda } else { lambda };
        
        Self {
            kernel: EwmaKernel::new(primitives, lambda),
            params: EwmaParameters {
                lambda,
                threshold_multiplier,
                target_mean: Some(target_mean),
                warmup_period: 20,
            },
        }
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> ChangePointDetectorProperties for EwmaDetector<T, P> {
    fn algorithm_name(&self) -> &'static str {
        "EWMA"
    }
    
    fn minimum_sample_size(&self) -> usize {
        self.params.warmup_period + 10
    }
}

impl<T, P, S, Q> ChangePointDetector<T, S, Q> for EwmaDetector<T, P>
where
    T: Numeric,
    P: ComputePrimitives<T>,
    S: SpreadEstimator<T, Q>,
    Q: QuantileEstimator<T>,
    T::Float: FromPrimitive + PartialOrd,
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
        
        // Step 1: Determine reference mean and spread
        let warmup_data = &sample[..self.params.warmup_period];
        
        // Make a single copy and sort for warmup statistics
        let mut warmup_sorted = warmup_data.to_vec();
        warmup_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let reference_mean = if let Some(target) = self.params.target_mean {
            target
        } else {
            quantile_est.quantile_sorted(&warmup_sorted, 0.5, cache)
                .map_err(|e| Error::Computation(format!("Quantile estimation failed: {}", e)))?
        };
        
        // Use spread estimator for robust variance estimation
        let reference_spread = spread_est.estimate_sorted(&warmup_sorted, quantile_est, cache)
            .map_err(|e| Error::Computation(format!("Spread estimation failed: {}", e)))?;
        
        // Step 2: Compute EWMA using kernel
        let ewma_values = self.kernel.compute_ewma(sample, Some(reference_mean));
        
        // Step 3: Compute deviations and detect changes
        let threshold = reference_spread * self.params.threshold_multiplier;
        let mut changepoints = Vec::new();
        
        // Skip warmup period
        for i in self.params.warmup_period..sample.len() {
            let deviation = if ewma_values[i] > reference_mean {
                ewma_values[i] - reference_mean
            } else {
                reference_mean - ewma_values[i]
            };
            
            if deviation > threshold {
                let two = T::Float::from_f64(2.0).unwrap();
                let one = T::Float::from_f64(1.0).unwrap();
                let confidence_float = if deviation / (two * threshold) < one {
                    deviation / (two * threshold)
                } else {
                    one
                };
                let confidence = NumCast::from(confidence_float).unwrap();
                changepoints.push(ChangePoint::with_type(
                    i,
                    confidence,
                    ChangeType::Drift,
                ));
            }
        }
        
        // Remove consecutive detections
        changepoints.dedup_by(|a, b| a.index.abs_diff(b.index) <= 3);
        
        // Convert ewma_values to f64 for statistics
        let statistics = ewma_values.iter().map(|&x| NumCast::from(x).unwrap()).collect();
        
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
        // EWMA requires temporal order, so sorted data doesn't help
        self.detect(sorted_sample, spread_est, quantile_est, cache)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> ConfigurableDetector for EwmaDetector<T, P> 
where
    T::Float: FromPrimitive,
{
    type Parameters = EwmaParameters<T>;
    
    fn with_parameters(_params: Self::Parameters) -> Self {
        unreachable!("Use new() or with_target_mean() to create EwmaDetector")
    }
    
    fn parameters(&self) -> &Self::Parameters {
        &self.params
    }
    
    fn set_parameters(&mut self, params: Self::Parameters) {
        let lambda = params.lambda;
        self.params = params;
        // Update kernel with new lambda
        self.kernel = EwmaKernel::new(self.kernel.primitives().clone(), lambda);
    }
}

/// Online EWMA detector with state
pub struct OnlineEwmaDetector<T, P, S, Q> 
where
    T: Numeric,
    P: ComputePrimitives<T>, 
    S: SpreadEstimator<T, Q>, 
    Q: QuantileEstimator<T>,
{
    detector: EwmaDetector<T, P>,
    ewma: T::Float,
    reference_mean: T::Float,
    threshold: T::Float,
    current_index: usize,
    warmup_buffer: Vec<T>,
    initialized: bool,
    _phantom: std::marker::PhantomData<(S, Q)>,
}

impl<T, P, S, Q> OnlineEwmaDetector<T, P, S, Q>
where
    T: Numeric,
    P: ComputePrimitives<T>,
    S: SpreadEstimator<T, Q>,
    Q: QuantileEstimator<T>,
    T::Float: FromPrimitive,
{
    pub fn new(detector: EwmaDetector<T, P>) -> Self {
        Self {
            detector,
            ewma: T::Float::zero(),
            reference_mean: T::Float::zero(),
            threshold: T::Float::zero(),
            current_index: 0,
            warmup_buffer: Vec::new(),
            initialized: false,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, P, S, Q> OnlineDetector<T, S, Q> for OnlineEwmaDetector<T, P, S, Q>
where
    T: Numeric,
    P: ComputePrimitives<T>,
    S: SpreadEstimator<T, Q>,
    Q: QuantileEstimator<T>,
    T::Float: FromPrimitive + PartialOrd,
{
    fn process_point(
        &mut self,
        value: T,
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<Vec<usize>> {
        if !self.initialized {
            self.warmup_buffer.push(value);
            
            if self.warmup_buffer.len() >= self.detector.params.warmup_period {
                // Initialize with warmup data
                let mut warmup_sorted = self.warmup_buffer.clone();
                warmup_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                
                self.reference_mean = if let Some(target) = self.detector.params.target_mean {
                    target
                } else {
                    quantile_est.quantile_sorted(&warmup_sorted, 0.5, cache)
                        .map_err(|e| Error::Computation(format!("Quantile estimation failed: {}", e)))?
                };
                
                let reference_spread = spread_est.estimate_sorted(&warmup_sorted, quantile_est, cache)
                    .map_err(|e| Error::Computation(format!("Spread estimation failed: {}", e)))?;
                self.threshold = reference_spread * self.detector.params.threshold_multiplier;
                
                self.ewma = self.reference_mean;
                self.initialized = true;
            }
            
            self.current_index += 1;
            return Ok(vec![]);
        }
        
        // Update EWMA
        let lambda = self.detector.params.lambda;
        let one_minus_lambda = T::Float::from_f64(1.0).unwrap() - lambda;
        self.ewma = lambda * value.to_float() + one_minus_lambda * self.ewma;
        
        // Check for changepoint
        let mut detected = Vec::new();
        let deviation = if self.ewma > self.reference_mean {
            self.ewma - self.reference_mean
        } else {
            self.reference_mean - self.ewma
        };
        
        if deviation > self.threshold {
            detected.push(self.current_index);
        }
        
        self.current_index += 1;
        Ok(detected)
    }
    
    fn reset(&mut self) {
        self.ewma = T::Float::zero();
        self.reference_mean = T::Float::zero();
        self.threshold = T::Float::zero();
        self.current_index = 0;
        self.warmup_buffer.clear();
        self.initialized = false;
    }
    
    fn current_index(&self) -> usize {
        self.current_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robust_core::{simd_sequential, ExecutionEngine};
    use robust_spread::StandardizedMad;
    use robust_quantile::estimators::harrell_davis;
    
    #[test]
    fn test_ewma_detection() {
        let primitives = simd_sequential().primitives().clone();
        let engine = robust_core::simd_sequential();
        let quantile_est = harrell_davis(engine);
        let spread_est = StandardizedMad::new(primitives.clone());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::default(),
            robust_core::CachePolicy::NoCache,
        );
        
        let detector = EwmaDetector::new(primitives, 0.2, 3.0);
        
        // Data with gradual shift
        let mut data = vec![0.0; 100];
        for i in 50..100 {
            data[i] = (i - 50) as f64 * 0.1;
        }
        
        let result = detector.detect(&data, &spread_est, &quantile_est, &cache).unwrap();
        
        // Should detect changes during the gradual shift
        assert!(!result.changepoints().is_empty());
    }
    
    #[test]
    fn test_online_ewma() {
        let primitives = simd_sequential().primitives().clone();
        let engine = robust_core::simd_sequential();
        let quantile_est = harrell_davis(engine);
        let spread_est = StandardizedMad::new(primitives.clone());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::default(),
            robust_core::CachePolicy::NoCache,
        );
        
        let detector = EwmaDetector::new(primitives, 0.2, 2.5);
        let mut online = OnlineEwmaDetector::new(detector);
        
        // Process normal data
        for _ in 0..30 {
            let detected = online.process_point(0.0, &spread_est, &quantile_est, &cache).unwrap();
            assert!(detected.is_empty());
        }
        
        // Process shifted data
        let mut any_detected = false;
        for _ in 0..20 {
            let detected = online.process_point(5.0, &spread_est, &quantile_est, &cache).unwrap();
            if !detected.is_empty() {
                any_detected = true;
            }
        }
        
        assert!(any_detected);
    }
}
