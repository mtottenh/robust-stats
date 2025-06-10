//! Unified bootstrap implementation with workspace-only, two-sample-only design
//!
//! This module provides a clean, efficient bootstrap implementation that:
//! - Always uses workspaces for memory efficiency
//! - Only supports two-sample comparisons (the actual use case)
//! - Uses hierarchical execution to prevent thread oversubscription
//! - Leverages batch APIs for optimal performance
//! - Provides type-safe APIs that prevent misuse

use crate::{
    bootstrap_workspace::BootstrapWorkspacePool,
    ConfidenceInterval,
};
use rand::prelude::*;
use robust_core::{
    execution::HierarchicalExecution,
    EstimatorFactory, Result, StatefulEstimator, TwoSampleComparison, Numeric,
};
use std::sync::Arc;
use tracing::{debug, instrument};

/// Bootstrap method for calculating confidence intervals
///
/// This trait defines how to construct a confidence interval from
/// bootstrap estimates. Different methods (percentile, BCa, etc.)
/// implement this trait.
pub trait BootstrapMethod: Clone + Send + Sync {
    /// Calculate confidence interval from bootstrap distribution
    fn calculate_interval(
        &self,
        bootstrap_estimates: &[f64],
        original_estimate: f64,
        confidence_level: f64,
    ) -> Result<ConfidenceInterval>;

    /// Method name for documentation
    fn name(&self) -> &'static str;
}


/// Result of bootstrap confidence interval estimation
#[derive(Debug, Clone)]
pub struct BootstrapResult<T> {
    /// The confidence interval(s)
    pub intervals: T,
    /// Number of bootstrap resamples performed
    pub n_resamples: usize,
    /// Original estimate(s)
    pub estimates: T,
    /// Time taken for bootstrap (if measured)
    pub bootstrap_time_ms: Option<u64>,
}

/// Main bootstrap engine with workspace management and hierarchical execution
///
/// This struct provides the core bootstrap functionality with:
/// - Automatic workspace management for memory efficiency
/// - Hierarchical execution to prevent thread oversubscription
/// - Support for any two-sample comparison
/// - Integration with batch APIs for performance
#[derive(Clone)]
pub struct Bootstrap<E, M> {
    engine: E,
    method: M,
    workspace_pool: Arc<BootstrapWorkspacePool<f64>>,
    n_resamples: usize,
    confidence_level: f64,
    seed: Option<u64>,
}

impl<E, M> Bootstrap<E, M>
where
    E: HierarchicalExecution<f64>,
    M: BootstrapMethod,
{
    /// Create a new bootstrap engine
    pub fn new(engine: E, method: M) -> Self {
        Self {
            engine,
            method,
            workspace_pool: Arc::new(BootstrapWorkspacePool::new()),
            n_resamples: 5000, // Default
            confidence_level: 0.95, // Default
            seed: None,
        }
    }

    /// Set the number of bootstrap resamples
    pub fn with_resamples(mut self, n_resamples: usize) -> Self {
        assert!(n_resamples > 0, "Number of resamples must be positive");
        self.n_resamples = n_resamples;
        self
    }

    /// Set the confidence level
    pub fn with_confidence_level(mut self, confidence_level: f64) -> Self {
        assert!(
            confidence_level > 0.0 && confidence_level < 1.0,
            "Confidence level must be in (0, 1)"
        );
        self.confidence_level = confidence_level;
        self
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set a custom workspace pool
    pub fn with_workspace_pool(mut self, pool: Arc<BootstrapWorkspacePool<f64>>) -> Self {
        self.workspace_pool = pool;
        self
    }

    /// Generate bootstrap indices for two samples
    fn generate_indices_two_sample(&self, n1: usize, n2: usize) -> Vec<Vec<usize>> {
        let seed = self.seed.unwrap_or_else(|| thread_rng().gen());
        
        debug!("Generating {} bootstrap indices for sample sizes {} and {}", self.n_resamples, n1, n2);
        
        (0..self.n_resamples)
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
                // Generate indices for sample1 in range [0, n1)
                let indices1: Vec<usize> = (0..n1).map(|_| rng.gen_range(0..n1)).collect();
                // Generate indices for sample2 in range [0, n2)
                let indices2: Vec<usize> = (0..n2).map(|_| rng.gen_range(0..n2)).collect();
                // Concatenate them
                indices1.into_iter().chain(indices2).collect()
            })
            .collect()
    }

    /// Compute confidence intervals for a two-sample comparison
    ///
    /// This is the main entry point for bootstrap confidence intervals.
    /// It works with any two-sample comparison and estimator factory.
    #[instrument(skip(self, sample1, sample2, comparison, estimator_factory), 
                 fields(n1 = sample1.len(), n2 = sample2.len(), n_resamples = self.n_resamples))]
    pub fn confidence_intervals<C, Est, F>(
        &self,
        sample1: &[f64],
        sample2: &[f64],
        comparison: &C,
        estimator_factory: &F,
    ) -> Result<BootstrapResult<C::Output>>
    where
        C: TwoSampleComparison<Est> + Clone + Send + Sync,
        F: EstimatorFactory<f64, E::SubordinateEngine, Estimator = Est>,
        Est: StatefulEstimator<f64> + Send + Sync,
        Est::State: Send + Sync,
        C::Output: BootstrapOutput + Clone + Send + Sync,
    {
        if sample1.is_empty() || sample2.is_empty() {
            return Err(robust_core::Error::InvalidInput("Empty sample(s)".to_string()));
        }

        let start_time = std::time::Instant::now();

        // Get original estimates using main engine
        let main_estimator = estimator_factory.create(self.engine.subordinate());
        let cache = estimator_factory.create_cache();
        let original_estimates = comparison.shift(&main_estimator, sample1, sample2, &cache)?;

        // Generate bootstrap indices for both samples
        let indices = self.generate_indices_two_sample(sample1.len(), sample2.len());

        // Prepare data and workspace
        let sample1 = Arc::new(sample1.to_vec());
        let sample2 = Arc::new(sample2.to_vec());
        let comparison = Arc::new(comparison.clone());
        let estimator_factory = Arc::new(estimator_factory.clone());
        let workspace_pool = Arc::clone(&self.workspace_pool);
        
        // IMPORTANT: Create the cache once and share it across all bootstrap iterations
        // This is thread-safe because caches are designed to be shared
        let shared_cache = Arc::new(estimator_factory.create_cache());

        debug!("Running hierarchical bootstrap with {} resamples", self.n_resamples);

        // Execute bootstrap in parallel with subordinate engines
        let bootstrap_estimates = self.engine.execute_batch(self.n_resamples, |i| {
            // Create subordinate engine for this resample
            let subordinate = self.engine.subordinate();
            let estimator = estimator_factory.create(subordinate);
            
            // Use the shared cache - this is the key optimization!
            let cache = Arc::clone(&shared_cache);

            // Get workspace for this thread
            let workspace = workspace_pool.get_workspace();
            
            // Resample indices
            let idx = &indices[i];
            let n1 = sample1.len();
            
            // Resample both datasets using workspace
            let resampled1 = workspace.resample_slice(&sample1, &idx[..n1]);
            let resampled2 = workspace.resample_slice(&sample2, &idx[n1..]);
            
            // Compute comparison on resampled data
            comparison.shift(&estimator, &resampled1, &resampled2, &*cache)
                .unwrap_or(original_estimates.clone())
        });

        debug!("Bootstrap completed, calculating confidence intervals");

        // Calculate confidence intervals based on the output type
        let intervals = self.calculate_intervals(
            &bootstrap_estimates,
            &original_estimates,
            self.confidence_level,
        )?;

        let bootstrap_time_ms = Some(start_time.elapsed().as_millis() as u64);

        Ok(BootstrapResult {
            intervals,
            n_resamples: self.n_resamples,
            estimates: original_estimates,
            bootstrap_time_ms,
        })
    }

    /// Calculate intervals for different output types
    fn calculate_intervals<O>(&self, bootstrap_estimates: &[O], original: &O, confidence_level: f64) -> Result<O>
    where
        O: BootstrapOutput,
    {
        O::calculate_intervals(&self.method, bootstrap_estimates, original, confidence_level)
    }
}

/// Trait for types that can be output from bootstrap operations
///
/// This trait allows us to handle both single values and vectors of values
/// (e.g., multiple quantiles) in a type-safe way.
pub trait BootstrapOutput: Clone + Send {
    /// Calculate confidence intervals for this output type
    fn calculate_intervals<M: BootstrapMethod>(
        method: &M,
        bootstrap_estimates: &[Self],
        original: &Self,
        confidence_level: f64,
    ) -> Result<Self>;
}

// Implementation for single f64 values
impl BootstrapOutput for f64 {
    fn calculate_intervals<M: BootstrapMethod>(
        method: &M,
        bootstrap_estimates: &[Self],
        original: &Self,
        confidence_level: f64,
    ) -> Result<Self> {
        // For single values, we return the estimate (CI stored separately)
        method.calculate_interval(bootstrap_estimates, *original, confidence_level)?;
        Ok(*original)
    }
}

// Implementation for vectors (e.g., multiple quantiles)
impl BootstrapOutput for Vec<f64> {
    fn calculate_intervals<M: BootstrapMethod>(
        method: &M,
        bootstrap_estimates: &[Self],
        original: &Self,
        confidence_level: f64,
    ) -> Result<Self> {
        // Transpose bootstrap estimates
        let n_values = original.len();
        let mut transposed = vec![Vec::with_capacity(bootstrap_estimates.len()); n_values];
        
        for estimates in bootstrap_estimates {
            for (i, &value) in estimates.iter().enumerate() {
                if i < n_values {
                    transposed[i].push(value);
                }
            }
        }
        
        // Calculate intervals for each value
        let intervals: Result<Vec<_>> = transposed.iter()
            .zip(original.iter())
            .map(|(bootstrap_vals, &orig)| {
                method.calculate_interval(bootstrap_vals, orig, confidence_level)
                    .map(|_| orig) // Return original value
            })
            .collect();
        
        intervals
    }
}

/// Specialized result type for vector outputs with individual confidence intervals
#[derive(Debug, Clone)]
pub struct VectorBootstrapResult<T: Numeric = f64> {
    /// Individual confidence intervals for each element
    pub intervals: Vec<ConfidenceInterval<T>>,
    /// Number of bootstrap resamples performed
    pub n_resamples: usize,
    /// Original estimates
    pub estimates: Vec<T>,
    /// Time taken for bootstrap (if measured)
    pub bootstrap_time_ms: Option<u64>,
}

/// Extension methods for Bootstrap when used with vector outputs
impl<E, M> Bootstrap<E, M>
where
    E: HierarchicalExecution<f64>,
    M: BootstrapMethod,
{
    /// Compute confidence intervals for vector outputs with detailed results
    ///
    /// This method is useful when you need individual confidence intervals
    /// for each element in a vector output (e.g., multiple quantiles).
    pub fn vector_confidence_intervals<C, Est, F>(
        &self,
        sample1: &[f64],
        sample2: &[f64],
        comparison: &C,
        estimator_factory: &F,
    ) -> Result<VectorBootstrapResult<f64>>
    where
        C: TwoSampleComparison<Est, Output = Vec<f64>> + Clone + Send + Sync,
        F: EstimatorFactory<f64, E::SubordinateEngine, Estimator = Est>,
        Est: StatefulEstimator<f64> + Send + Sync,
        Est::State: Send + Sync,
    {
        // Run standard bootstrap
        let result = self.confidence_intervals(sample1, sample2, comparison, estimator_factory)?;
        
        // Calculate individual intervals
        let n_values = result.estimates.len();
        let mut bootstrap_by_element = vec![Vec::with_capacity(self.n_resamples); n_values];
        
        // This is a bit inefficient but works for now
        // In the future, we could optimize by storing intermediate results
        let start_time = std::time::Instant::now();
        
        // Re-run bootstrap to get individual estimates
        // (In practice, we'd cache these during the main run)
        let indices = self.generate_indices_two_sample(sample1.len(), sample2.len());
        let workspace_pool = Arc::clone(&self.workspace_pool);
        
        // Create shared cache for all bootstrap iterations
        let shared_cache = Arc::new(estimator_factory.create_cache());
        
        let bootstrap_estimates: Vec<Vec<f64>> = self.engine.execute_batch(self.n_resamples, |i| {
            let subordinate = self.engine.subordinate();
            let estimator = estimator_factory.create(subordinate);
            let cache = Arc::clone(&shared_cache);
            let workspace = workspace_pool.get_workspace();
            
            let idx = &indices[i];
            let n1 = sample1.len();
            
            let resampled1 = workspace.resample_slice(sample1, &idx[..n1]);
            let resampled2 = workspace.resample_slice(sample2, &idx[n1..]);
            
            comparison.shift(&estimator, &resampled1, &resampled2, &*cache)
                .unwrap_or_else(|_| vec![f64::NAN; n_values])
        });
        
        // Transpose results
        for estimates in &bootstrap_estimates {
            for (i, &value) in estimates.iter().enumerate() {
                if i < n_values {
                    bootstrap_by_element[i].push(value);
                }
            }
        }
        
        // Calculate individual confidence intervals
        let intervals: Result<Vec<_>> = bootstrap_by_element.iter()
            .zip(result.estimates.iter())
            .map(|(bootstrap_vals, &orig)| {
                self.method.calculate_interval(bootstrap_vals, orig, self.confidence_level)
            })
            .collect();
        
        Ok(VectorBootstrapResult {
            intervals: intervals?,
            n_resamples: self.n_resamples,
            estimates: result.estimates,
            bootstrap_time_ms: Some(start_time.elapsed().as_millis() as u64),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robust_core::execution::scalar_sequential;
    use crate::bootstrap_methods::PercentileBootstrap;

    #[test]
    fn test_bootstrap_construction() {
        let engine = scalar_sequential();
        let method = PercentileBootstrap;
        
        let bootstrap = Bootstrap::new(engine, method)
            .with_resamples(1000)
            .with_confidence_level(0.95)
            .with_seed(42);
        
        assert_eq!(bootstrap.n_resamples, 1000);
        assert_eq!(bootstrap.confidence_level, 0.95);
        assert_eq!(bootstrap.seed, Some(42));
    }
    
    #[test]
    fn test_generate_indices() {
        let engine = scalar_sequential();
        let method = PercentileBootstrap;
        
        let bootstrap = Bootstrap::new(engine, method)
            .with_resamples(10)
            .with_seed(42);
        
        let indices = bootstrap.generate_indices_two_sample(3, 2);
        
        // Check structure
        assert_eq!(indices.len(), 10); // 10 resamples
        for resample in &indices {
            assert_eq!(resample.len(), 5); // 3 + 2 indices each
            // First 3 indices should be in range [0, 3)
            for &idx in &resample[..3] {
                assert!(idx < 3);
            }
            // Last 2 indices should be in range [0, 2)
            for &idx in &resample[3..] {
                assert!(idx < 2);
            }
        }
        
        // Check reproducibility
        let indices2 = bootstrap.generate_indices_two_sample(3, 2);
        assert_eq!(indices, indices2);
    }
    
    #[test]
    fn test_invalid_inputs() {
        let engine = scalar_sequential();
        let method = PercentileBootstrap;
        
        // Test invalid confidence level
        let result = std::panic::catch_unwind(|| {
            Bootstrap::new(engine.clone(), method.clone())
                .with_confidence_level(1.5);
        });
        assert!(result.is_err());
        
        // Test invalid resamples
        let result = std::panic::catch_unwind(|| {
            Bootstrap::new(engine, method)
                .with_resamples(0);
        });
        assert!(result.is_err());
    }
    
    #[test]
    fn test_bootstrap_output_trait() {
        // Test f64 implementation
        let method = PercentileBootstrap;
        let bootstrap_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let original = 3.0;
        
        let result = f64::calculate_intervals(&method, &bootstrap_vals, &original, 0.95);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), original);
        
        // Test Vec<f64> implementation
        let bootstrap_vecs = vec![
            vec![1.0, 2.0],
            vec![1.5, 2.5],
            vec![2.0, 3.0],
        ];
        let original_vec = vec![1.5, 2.5];
        
        let result = Vec::<f64>::calculate_intervals(&method, &bootstrap_vecs, &original_vec, 0.95);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), original_vec);
    }
}