//! Specialized bootstrap implementation for quantile estimators
//!
//! This module provides optimized bootstrap confidence intervals for
//! quantile-based comparisons, leveraging batch APIs for efficiency.

use crate::{
    bootstrap::{Bootstrap, BootstrapMethod},
    ConfidenceInterval,
};
use robust_core::{
    execution::HierarchicalExecution,
    BatchQuantileEstimator, EstimatorFactory, QuantileShiftComparison, QuantileRatioComparison,
    Result, Numeric,
};

/// Specialized bootstrap for quantile shift comparisons
///
/// This implementation is optimized for computing confidence intervals
/// for differences between quantiles of two samples.
pub struct QuantileShiftBootstrap<E, M> {
    bootstrap: Bootstrap<E, M>,
    quantiles: Vec<f64>,
}

impl<E, M> QuantileShiftBootstrap<E, M>
where
    E: HierarchicalExecution<f64>,
    M: BootstrapMethod,
{
    /// Create a new quantile shift bootstrap
    pub fn new(engine: E, method: M, quantiles: Vec<f64>) -> Result<Self> {
        // Validate quantiles
        for &q in &quantiles {
            if q <= 0.0 || q >= 1.0 {
                return Err(robust_core::Error::InvalidInput(
                    format!("Quantile {} must be in (0, 1)", q)
                ));
            }
        }
        
        Ok(Self {
            bootstrap: Bootstrap::new(engine, method),
            quantiles,
        })
    }
    
    /// Set the number of bootstrap resamples
    pub fn with_resamples(mut self, n_resamples: usize) -> Self {
        self.bootstrap = self.bootstrap.with_resamples(n_resamples);
        self
    }
    
    /// Set the confidence level
    pub fn with_confidence_level(mut self, confidence_level: f64) -> Self {
        self.bootstrap = self.bootstrap.with_confidence_level(confidence_level);
        self
    }
    
    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.bootstrap = self.bootstrap.with_seed(seed);
        self
    }
    
    /// Compute confidence intervals for quantile shifts
    pub fn confidence_intervals<Est, F>(
        &self,
        sample1: &[f64],
        sample2: &[f64],
        estimator_factory: &F,
    ) -> Result<QuantileBootstrapResult<f64>>
    where
        F: EstimatorFactory<f64, E::SubordinateEngine, Estimator = Est>,
        Est: BatchQuantileEstimator<f64> + Send + Sync,
        Est::State: Send + Sync,
    {
        // Create comparison for quantile shifts
        let comparison = QuantileShiftComparison::new(self.quantiles.clone())?;
        
        // Run bootstrap with vector output
        let result = self.bootstrap.vector_confidence_intervals(
            sample1,
            sample2,
            &comparison,
            estimator_factory,
        )?;
        
        Ok(QuantileBootstrapResult {
            intervals: result.intervals,
            quantiles: self.quantiles.clone(),
            n_resamples: result.n_resamples,
            estimates: result.estimates,
            bootstrap_time_ms: result.bootstrap_time_ms,
        })
    }
}

/// Specialized bootstrap for quantile ratio comparisons
///
/// This implementation is optimized for computing confidence intervals
/// for ratios between quantiles of two samples.
pub struct QuantileRatioBootstrap<E, M> {
    bootstrap: Bootstrap<E, M>,
    quantiles: Vec<f64>,
}

impl<E, M> QuantileRatioBootstrap<E, M>
where
    E: HierarchicalExecution<f64>,
    M: BootstrapMethod,
{
    /// Create a new quantile ratio bootstrap
    pub fn new(engine: E, method: M, quantiles: Vec<f64>) -> Result<Self> {
        // Validate quantiles
        for &q in &quantiles {
            if q <= 0.0 || q >= 1.0 {
                return Err(robust_core::Error::InvalidInput(
                    format!("Quantile {} must be in (0, 1)", q)
                ));
            }
        }
        
        Ok(Self {
            bootstrap: Bootstrap::new(engine, method),
            quantiles,
        })
    }
    
    /// Set the number of bootstrap resamples
    pub fn with_resamples(mut self, n_resamples: usize) -> Self {
        self.bootstrap = self.bootstrap.with_resamples(n_resamples);
        self
    }
    
    /// Set the confidence level
    pub fn with_confidence_level(mut self, confidence_level: f64) -> Self {
        self.bootstrap = self.bootstrap.with_confidence_level(confidence_level);
        self
    }
    
    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.bootstrap = self.bootstrap.with_seed(seed);
        self
    }
    
    /// Compute confidence intervals for quantile ratios
    pub fn confidence_intervals<Est, F>(
        &self,
        sample1: &[f64],
        sample2: &[f64],
        estimator_factory: &F,
    ) -> Result<QuantileBootstrapResult<f64>>
    where
        F: EstimatorFactory<f64, E::SubordinateEngine, Estimator = Est>,
        Est: BatchQuantileEstimator<f64> + Send + Sync,
        Est::State: Send + Sync,
    {
        // Create comparison for quantile ratios
        let comparison = QuantileRatioComparison::new(self.quantiles.clone())?;
        
        // Run bootstrap with vector output
        let result = self.bootstrap.vector_confidence_intervals(
            sample1,
            sample2,
            &comparison,
            estimator_factory,
        )?;
        
        Ok(QuantileBootstrapResult {
            intervals: result.intervals,
            quantiles: self.quantiles.clone(),
            n_resamples: result.n_resamples,
            estimates: result.estimates,
            bootstrap_time_ms: result.bootstrap_time_ms,
        })
    }
}

/// Result of quantile bootstrap with metadata
#[derive(Debug, Clone)]
pub struct QuantileBootstrapResult<T: Numeric = f64> {
    /// Confidence intervals for each quantile
    pub intervals: Vec<ConfidenceInterval<T>>,
    /// The quantile probabilities
    pub quantiles: Vec<f64>,
    /// Number of bootstrap resamples
    pub n_resamples: usize,
    /// Original estimates for each quantile
    pub estimates: Vec<T>,
    /// Time taken for bootstrap (if measured)
    pub bootstrap_time_ms: Option<u64>,
}

impl<T: Numeric> QuantileBootstrapResult<T> {
    /// Get confidence interval for a specific quantile
    pub fn interval_for_quantile(&self, quantile: f64) -> Option<&ConfidenceInterval<T>> {
        self.quantiles.iter()
            .position(|&q| (q - quantile).abs() < 1e-10)
            .and_then(|idx| self.intervals.get(idx))
    }
    
    /// Get the median confidence interval (if 0.5 quantile was included)
    pub fn median_interval(&self) -> Option<&ConfidenceInterval<T>> {
        self.interval_for_quantile(0.5)
    }
    
    /// Get quartile confidence intervals (if included)
    pub fn quartile_intervals(&self) -> (Option<&ConfidenceInterval<T>>, Option<&ConfidenceInterval<T>>, Option<&ConfidenceInterval<T>>) {
        (
            self.interval_for_quantile(0.25),
            self.interval_for_quantile(0.50),
            self.interval_for_quantile(0.75),
        )
    }
    
    /// Create a summary string of the results
    pub fn summary(&self) -> String 
    where 
        T: std::fmt::Display,
    {
        let mut summary = format!("Quantile Bootstrap Results (n={})\n", self.n_resamples);
        summary.push_str("─────────────────────────────────────────────────\n");
        summary.push_str("Quantile │ Estimate │ Lower    │ Upper    │ Width\n");
        summary.push_str("─────────────────────────────────────────────────\n");
        
        for (i, (&q, ci)) in self.quantiles.iter().zip(&self.intervals).enumerate() {
            let width = ci.upper - ci.lower;
            summary.push_str(&format!(
                "{:8.2} │ {:8} │ {:8} │ {:8} │ {:6}\n",
                q, self.estimates[i], ci.lower, ci.upper, width
            ));
        }
        
        if let Some(ms) = self.bootstrap_time_ms {
            summary.push_str(&format!("\nComputation time: {}ms\n", ms));
        }
        
        summary
    }
}

/// Convenience functions for common quantile sets

/// Create bootstrap for standard quantiles (deciles)
pub fn decile_bootstrap<E, M>(engine: E, method: M) -> Result<QuantileShiftBootstrap<E, M>>
where
    E: HierarchicalExecution<f64>,
    M: BootstrapMethod,
{
    let quantiles = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    QuantileShiftBootstrap::new(engine, method, quantiles)
}

/// Create bootstrap for quartiles
pub fn quartile_bootstrap<E, M>(engine: E, method: M) -> Result<QuantileShiftBootstrap<E, M>>
where
    E: HierarchicalExecution<f64>,
    M: BootstrapMethod,
{
    let quantiles = vec![0.25, 0.5, 0.75];
    QuantileShiftBootstrap::new(engine, method, quantiles)
}

/// Create bootstrap for standard percentiles
pub fn percentile_bootstrap<E, M>(engine: E, method: M) -> Result<QuantileShiftBootstrap<E, M>>
where
    E: HierarchicalExecution<f64>,
    M: BootstrapMethod,
{
    let quantiles = vec![0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];
    QuantileShiftBootstrap::new(engine, method, quantiles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap_methods::PercentileBootstrap;
    use robust_core::execution::scalar_sequential;
    
    #[test]
    fn test_quantile_shift_bootstrap() {
        let engine = scalar_sequential();
        let method = PercentileBootstrap;
        
        let bootstrap = QuantileShiftBootstrap::new(engine, method, vec![0.25, 0.5, 0.75])
            .unwrap()
            .with_resamples(100)
            .with_seed(42);
        
        assert_eq!(bootstrap.quantiles.len(), 3);
    }
    
    #[test]
    fn test_invalid_quantiles() {
        let engine = scalar_sequential();
        let method = PercentileBootstrap;
        
        // Test invalid quantiles
        assert!(QuantileShiftBootstrap::new(engine.clone(), method, vec![0.0]).is_err());
        assert!(QuantileShiftBootstrap::new(engine.clone(), method, vec![1.0]).is_err());
        assert!(QuantileShiftBootstrap::new(engine, method, vec![-0.1]).is_err());
    }
    
    #[test]
    fn test_quantile_result_helpers() {
        let result = QuantileBootstrapResult::<f64> {
            intervals: vec![
                ConfidenceInterval::new(1.0, 2.0, 1.5, 0.95),
                ConfidenceInterval::new(2.0, 3.0, 2.5, 0.95),
                ConfidenceInterval::new(3.0, 4.0, 3.5, 0.95),
            ],
            quantiles: vec![0.25, 0.50, 0.75],
            n_resamples: 1000,
            estimates: vec![1.5, 2.5, 3.5],
            bootstrap_time_ms: Some(123),
        };
        
        // Test median lookup
        let median = result.median_interval().unwrap();
        assert_eq!(median.estimate, 2.5);
        
        // Test quartile lookup
        let (q1, q2, q3) = result.quartile_intervals();
        assert!(q1.is_some());
        assert!(q2.is_some());
        assert!(q3.is_some());
        assert_eq!(q1.unwrap().estimate, 1.5);
        assert_eq!(q3.unwrap().estimate, 3.5);
    }
}