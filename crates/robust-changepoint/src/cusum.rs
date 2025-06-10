//! CUSUM (Cumulative Sum) changepoint detection - redesigned
//!
//! This implementation follows the three-layer architecture properly

use crate::kernel::CusumKernel;
use crate::traits::{
    ChangePointDetectorProperties, SimpleDetector, 
    ConfigurableDetector, ConfidenceScoring,
};
use crate::types::{ChangePoint, ChangePointResult, ChangeType};
use robust_core::{ComputePrimitives, Error, Result, StatisticalKernel, Numeric};
use num_traits::{FromPrimitive, NumCast};

/// CUSUM parameters
#[derive(Clone, PartialEq)]
pub struct CusumParameters<T: Numeric> {
    /// Detection threshold (typically 3-5)
    pub threshold: T::Float,
    /// Reference value for change detection (drift parameter)
    pub drift: T::Float,
    /// Target mean (if known, otherwise estimated from data)
    pub target_mean: Option<T::Float>,
}

impl<T: Numeric> std::fmt::Debug for CusumParameters<T>
where
    T::Float: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CusumParameters")
            .field("threshold", &self.threshold)
            .field("drift", &self.drift)
            .field("target_mean", &self.target_mean)
            .finish()
    }
}

impl<T: Numeric> Default for CusumParameters<T> 
where
    T::Float: num_traits::FromPrimitive,
{
    fn default() -> Self {
        use num_traits::FromPrimitive;
        Self {
            threshold: T::Float::from_f64(4.0).unwrap(),
            drift: T::Float::from_f64(0.5).unwrap(),
            target_mean: None,
        }
    }
}

/// CUSUM changepoint detector using kernels
#[derive(Clone)]
pub struct CusumDetector<T: Numeric, P: ComputePrimitives<T>> {
    kernel: CusumKernel<T, P>,
    params: CusumParameters<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> std::fmt::Debug for CusumDetector<T, P>
where
    T::Float: std::fmt::Debug,
    P: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CusumDetector")
            .field("kernel", &self.kernel)
            .field("params", &self.params)
            .finish()
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> CusumDetector<T, P> {
    /// Create a new CUSUM detector
    pub fn new(primitives: P, threshold: T::Float, drift: T::Float) -> Self {
        Self {
            kernel: CusumKernel::new(primitives, drift),
            params: CusumParameters {
                threshold,
                drift,
                target_mean: None,
            },
        }
    }
    
    /// Create with a known target mean
    pub fn with_target_mean(primitives: P, threshold: T::Float, drift: T::Float, target_mean: T::Float) -> Self {
        Self {
            kernel: CusumKernel::new(primitives, drift),
            params: CusumParameters {
                threshold,
                drift,
                target_mean: Some(target_mean),
            },
        }
    }
    
    /// Estimate reference mean robustly
    fn estimate_reference_mean(&self, sample: &[T]) -> T::Float {
        if let Some(target) = self.params.target_mean {
            return target;
        }
        
        // Use median for robustness
        let mut sorted = sample.to_vec();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let n = sorted.len();
        if n % 2 == 0 {
            let a = sorted[n / 2 - 1].to_float();
            let b = sorted[n / 2].to_float();
            (a + b) / num_traits::NumCast::from(2.0).unwrap()
        } else {
            sorted[n / 2].to_float()
        }
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> ChangePointDetectorProperties for CusumDetector<T, P> {
    fn algorithm_name(&self) -> &'static str {
        "CUSUM"
    }
    
    fn minimum_sample_size(&self) -> usize {
        5 // Need enough data to estimate mean
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> SimpleDetector<T> for CusumDetector<T, P> 
where
    T::Float: num_traits::FromPrimitive + PartialOrd,
{
    fn detect_simple(&self, sample: &[T]) -> Result<ChangePointResult> {
        if sample.len() < self.minimum_sample_size() {
            return Err(Error::InsufficientData {
                expected: self.minimum_sample_size(),
                actual: sample.len(),
            });
        }
        
        // Estimate reference mean
        let reference = self.estimate_reference_mean(sample);
        
        // Use kernel to compute CUSUM statistics
        let (positive_cusum, negative_cusum) = self.kernel.compute_cusum(sample, reference);
        
        // Detect changepoints
        let mut changepoints = Vec::new();
        
        for (i, (&pos, &neg)) in positive_cusum.iter().zip(&negative_cusum).enumerate() {
            if pos > self.params.threshold {
                let two = T::Float::from_f64(2.0).unwrap();
                let one = T::Float::from_f64(1.0).unwrap();
                let confidence_float = if pos / (self.params.threshold * two) < one {
                    pos / (self.params.threshold * two)
                } else {
                    one
                };
                let confidence = num_traits::NumCast::from(confidence_float).unwrap();
                changepoints.push(ChangePoint::with_type(i, confidence, ChangeType::MeanShift));
            }
            
            if neg > self.params.threshold {
                let two = T::Float::from_f64(2.0).unwrap();
                let one = T::Float::from_f64(1.0).unwrap();
                let confidence_float = if neg / (self.params.threshold * two) < one {
                    neg / (self.params.threshold * two)
                } else {
                    one
                };
                let confidence = num_traits::NumCast::from(confidence_float).unwrap();
                changepoints.push(ChangePoint::with_type(i, confidence, ChangeType::MeanShift));
            }
        }
        
        // Remove consecutive detections
        changepoints.dedup_by(|a, b| a.index.abs_diff(b.index) <= 1);
        
        // Combine statistics for output - convert to f64
        let mut all_statistics: Vec<f64> = positive_cusum.iter().map(|&x| NumCast::from(x).unwrap()).collect();
        all_statistics.extend(negative_cusum.iter().map(|&x| -> f64 { NumCast::from(x).unwrap() }));
        
        Ok(ChangePointResult::new(
            changepoints,
            self.algorithm_name().to_string(),
            sample.len(),
            all_statistics,
        ))
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> ConfigurableDetector for CusumDetector<T, P> {
    type Parameters = CusumParameters<T>;
    
    fn with_parameters(_params: Self::Parameters) -> Self {
        unreachable!("Use new() or with_target_mean() to create CusumDetector")
    }
    
    fn parameters(&self) -> &Self::Parameters {
        &self.params
    }
    
    fn set_parameters(&mut self, params: Self::Parameters) {
        let drift = params.drift;
        self.params = params;
        // Update kernel drift if needed
        self.kernel = CusumKernel::new(self.kernel.primitives().clone(), drift);
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> ConfidenceScoring for CusumDetector<T, P> {
    fn confidence_score(&self, _changepoint_index: usize) -> f64 {
        // Would need to store state to implement this properly
        0.5
    }
    
    fn detection_threshold(&self) -> f64 {
        num_traits::NumCast::from(self.params.threshold).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robust_core::{simd_sequential, ExecutionEngine};
    
    #[test]
    fn test_cusum_detection() {
        let primitives = simd_sequential().primitives().clone();
        // Use a detector with known target mean of 0.0 to test shift to 3.0
        let detector = CusumDetector::with_target_mean(primitives, 4.0, 0.5, 0.0);
        
        // Data with mean shift at index 25
        let mut data = vec![0.0; 50];
        for i in 25..50 {
            data[i] = 3.0;
        }
        
        let result = detector.detect_simple(&data).unwrap();
        
        assert!(!result.changepoints().is_empty(), "No changepoints detected!");
        
        // Should detect around index 25
        let detected = result.changepoints()[0].index;
        assert!((detected as i32 - 25).abs() < 10, "Detection too far from expected position: {} vs 25", detected);
    }
    
    #[test]
    fn test_with_target_mean() {
        let primitives = simd_sequential().primitives().clone();
        let detector = CusumDetector::with_target_mean(primitives, 3.0, 0.5, 0.0);
        
        let data = vec![0.0, 0.1, -0.1, 5.0, 5.1, 4.9, 5.0];
        let result = detector.detect_simple(&data).unwrap();
        
        // Should detect the jump to 5.0
        assert!(!result.changepoints().is_empty());
    }
}