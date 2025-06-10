//! Adapter to provide EstimatorWithVariance implementation for quantile estimators
//!
//! This bridges the gap between quantile estimators that can compute moments
//! and the generic EstimatorWithVariance interface.

use crate::{QuantileEstimator, confidence::QuantileWithMoments};
use robust_core::{EstimatorWithVariance, StatefulEstimator, Result as CoreResult};

/// Adapter that provides EstimatorWithVariance for quantile estimators
///
/// This wraps a quantile estimator that implements QuantileWithMoments
/// and provides the EstimatorWithVariance interface needed for Maritz-Jarrett CI.
#[derive(Clone, Debug)]
pub struct QuantileVarianceAdapter<Q> {
    estimator: Q,
    quantile: f64,
}

impl<Q> QuantileVarianceAdapter<Q> {
    /// Create a new adapter for a specific quantile
    pub fn new(estimator: Q, quantile: f64) -> Self {
        assert!(
            quantile > 0.0 && quantile < 1.0,
            "Quantile must be in (0, 1)"
        );
        Self { estimator, quantile }
    }
    
    /// Create adapter for median
    pub fn median(estimator: Q) -> Self {
        Self::new(estimator, 0.5)
    }
}

impl<Q> StatefulEstimator for QuantileVarianceAdapter<Q>
where
    Q: QuantileEstimator + QuantileWithMoments,
{
    type State = Q::State;
    
    fn estimate_with_cache(&self, sample: &[f64], cache: &Self::State) -> CoreResult<f64> {
        let mut sorted = sample.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        self.estimator
            .quantile_sorted(&sorted, self.quantile, cache)
            .map_err(|e| robust_core::Error::Computation(format!("Quantile estimation failed: {e}")))
    }
    
    fn estimate_sorted_with_cache(&self, sorted_sample: &[f64], cache: &Self::State) -> CoreResult<f64> {
        self.estimator
            .quantile_sorted(sorted_sample, self.quantile, cache)
            .map_err(|e| robust_core::Error::Computation(format!("Quantile estimation failed: {e}")))
    }
}

impl<Q> EstimatorWithVariance for QuantileVarianceAdapter<Q>
where
    Q: QuantileEstimator + QuantileWithMoments,
{
    fn estimate_with_variance(
        &self,
        sample: &[f64],
        _cache: &Self::State,
    ) -> CoreResult<(f64, f64)> {
        let mut sorted = sample.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        self.estimator
            .quantile_with_moments(&sorted, self.quantile)
            .map_err(|e| robust_core::Error::Computation(format!("Quantile with moments failed: {e}")))
    }
    
    fn estimate_sorted_with_variance(
        &self,
        sorted_sample: &[f64],
        _cache: &Self::State,
    ) -> CoreResult<(f64, f64)> {
        self.estimator
            .quantile_with_moments(sorted_sample, self.quantile)
            .map_err(|e| robust_core::Error::Computation(format!("Quantile with moments failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robust_core::BatchProcessor;
    
    // Mock quantile estimator for testing
    #[derive(Clone)]
    struct MockQuantileEstimator;
    
    impl BatchProcessor<f64> for MockQuantileEstimator {
        type Input = [f64];
        type Output = f64;
        type State = ();
        type Params = f64;
        
        fn process_batch(
            &self,
            inputs: &mut [&mut Self::Input],
            params: &Self::Params,
            state: &Self::State,
            _strategy: robust_core::ProcessingStrategy,
        ) -> robust_core::Result<Vec<Self::Output>> {
            inputs.iter_mut().map(|data| {
                self.quantile(data, *params, state)
                    .map_err(|e| robust_core::Error::Computation(format!("Quantile estimation failed: {}", e)))
            }).collect()
        }
        
        fn process_batch_sorted(
            &self,
            sorted_inputs: &[&Self::Input],
            params: &Self::Params,
            state: &Self::State,
            _strategy: robust_core::ProcessingStrategy,
        ) -> robust_core::Result<Vec<Self::Output>> {
            sorted_inputs.iter().map(|data| {
                self.quantile_sorted(data, *params, state)
                    .map_err(|e| robust_core::Error::Computation(format!("Quantile estimation failed: {}", e)))
            }).collect()
        }
    }
    
    impl robust_core::CentralTendencyEstimator for MockQuantileEstimator {
        fn estimate_sorted(&self, sorted_data: &[f64]) -> robust_core::Result<f64> {
            self.quantile_sorted(sorted_data, 0.5, &())
                .map_err(|e| robust_core::Error::Computation(format!("Quantile estimation failed: {}", e)))
        }
        
        fn name(&self) -> &str {
            "MockQuantileEstimator"
        }
        
        fn is_robust(&self) -> bool {
            true
        }
        
        fn breakdown_point(&self) -> f64 {
            0.5
        }
    }
    
    impl QuantileEstimator for MockQuantileEstimator {
        fn quantile(&self, data: &mut [f64], p: f64, _cache: &Self::State) -> crate::Result<f64> {
            data.sort_by(|a, b| a.partial_cmp(b).unwrap());
            self.quantile_sorted(data, p, _cache)
        }
        
        fn quantile_sorted(&self, sorted_data: &[f64], p: f64, _cache: &Self::State) -> crate::Result<f64> {
            if sorted_data.is_empty() {
                return Err(crate::Error::EmptyData);
            }
            let index = (p * (sorted_data.len() - 1) as f64) as usize;
            Ok(sorted_data[index])
        }
        
        fn quantiles(&self, data: &mut [f64], ps: &[f64], cache: &Self::State) -> crate::Result<Vec<f64>> {
            data.sort_by(|a, b| a.partial_cmp(b).unwrap());
            self.quantiles_sorted(data, ps, cache)
        }
        
        fn quantiles_sorted(&self, sorted_data: &[f64], ps: &[f64], cache: &Self::State) -> crate::Result<Vec<f64>> {
            ps.iter()
                .map(|&p| self.quantile_sorted(sorted_data, p, cache))
                .collect()
        }
    }
    
    impl QuantileWithMoments for MockQuantileEstimator {
        fn quantile_with_moments(&self, sorted_data: &[f64], p: f64) -> crate::Result<(f64, f64)> {
            let q = self.quantile_sorted(sorted_data, p, &())?;
            // Mock variance calculation
            let variance = 0.1; // Simplified for testing
            Ok((q, variance))
        }
    }
    
    #[test]
    fn test_quantile_variance_adapter() {
        let estimator = MockQuantileEstimator;
        let adapter = QuantileVarianceAdapter::median(estimator);
        
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cache = ();
        
        // Test basic estimation
        let estimate = adapter.estimate_with_cache(&data, &cache).unwrap();
        assert_eq!(estimate, 3.0);
        
        // Test with variance
        let (est, var) = adapter.estimate_with_variance(&data, &cache).unwrap();
        assert_eq!(est, 3.0);
        assert_eq!(var, 0.1);
    }
}