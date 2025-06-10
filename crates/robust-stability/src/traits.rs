//! Core traits for stability analysis
//!
//! Following the parameterized trait design pattern to enable:
//! - Zero-cost rolling operations
//! - Perfect cache sharing
//! - Composable algorithms
//! - Static dispatch throughout
//!
//! # Migration Guide
//!
//! The traits have been redesigned to support parameterization while maintaining
//! backward compatibility. Here's how to migrate:
//!
//! ## For Users
//! 
//! Existing code using `StabilityAnalyzer` will continue to work unchanged:
//! ```ignore
//! let analyzer = StatisticalStabilityAnalyzer::default();
//! let result = analyzer.analyze(&data)?;
//! ```
//!
//! ## For Advanced Users
//!
//! New code can use parameterized traits for better performance and flexibility:
//! ```ignore
//! let primitives = PrimitivesCPU::new();
//! let spread_est = Mad::new(primitives.clone());
//! let quantile_est = HDQuantile::new(CachingHDWeight::new(primitives));
//! let cache = quantile_est.create_state();
//! 
//! let analyzer = StatisticalStabilityAnalyzer::default();
//! let result = analyzer.analyze_with_estimators(&data, &spread_est, &quantile_est, &cache)?;
//! ```

use std::fmt;
use crate::types::{StabilityMetrics, StabilityStatus};
use robust_core::{Result, Numeric};
use robust_spread::SpreadEstimator;
use robust_quantile::QuantileEstimator;

/// Intrinsic properties of a stability analyzer that don't depend on dependencies
pub trait StabilityAnalyzerProperties {
    /// Get the minimum number of samples required for analysis
    fn minimum_samples(&self) -> usize;
    
    /// Get the name of the analysis method
    fn method_name(&self) -> &str;
    
    /// Check if the analyzer can handle the given sample size
    fn can_handle_size(&self, size: usize) -> bool {
        size >= self.minimum_samples()
    }
}

/// The main trait for stability analysis algorithms
/// 
/// This is the original trait maintained for backward compatibility.
/// New code should use the parameterized versions below.
pub trait StabilityAnalyzer<T: Numeric>: StabilityAnalyzerProperties {
    /// Analyze a time series for stability
    fn analyze(&self, data: &[T]) -> Result<StabilityResult<T>>;
}

/// Parameterized stability analyzer that uses spread estimation
/// 
/// This trait is for analyzers that need spread/scale estimates (like MAD or IQR)
/// for variability analysis or robust statistics.
pub trait StabilityAnalyzerWithSpread<T, S, Q>: StabilityAnalyzerProperties 
where
    T: Numeric,
    S: SpreadEstimator<T, Q>,
    Q: QuantileEstimator<T>,
{
    /// Analyze stability using the provided spread estimator
    fn analyze_with_spread(
        &self, 
        data: &[T], 
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<StabilityResult<T>>;
}

/// Parameterized stability analyzer that uses quantile estimation
/// 
/// This trait is for analyzers that need quantile estimates directly
/// (e.g., for robust location estimation or outlier detection).
pub trait StabilityAnalyzerWithQuantile<T, Q>: StabilityAnalyzerProperties 
where
    T: Numeric,
    Q: QuantileEstimator<T>,
{
    /// Analyze stability using the provided quantile estimator
    fn analyze_with_quantile(
        &self,
        data: &[T],
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<StabilityResult<T>>;
}

/// Combined parameterized trait for analyzers that need both spread and quantile estimators
/// 
/// Many stability algorithms need both location (median) and scale (MAD) estimates,
/// so this combined trait reduces boilerplate.
pub trait StabilityAnalyzerWithEstimators<T, S, Q>: StabilityAnalyzerProperties 
where
    T: Numeric,
    S: SpreadEstimator<T, Q>,
    Q: QuantileEstimator<T>,
{
    /// Analyze stability using the provided estimators
    fn analyze_with_estimators(
        &self,
        data: &[T],
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<StabilityResult<T>>;
}

/// Result of stability analysis
#[derive(Clone)]
pub struct StabilityResult<T: Numeric> {
    /// Overall stability status
    pub status: StabilityStatus<T>,
    
    /// Detailed metrics from the analysis
    pub metrics: StabilityMetrics<T>,
    
    /// Confidence level (0.0 to 1.0)
    pub confidence: T::Float,
    
    /// Time index when stability was achieved (if applicable)
    pub stability_index: Option<usize>,
    
    /// Detailed explanation of the result
    pub explanation: String,
    
    /// Method used for analysis
    pub method: String,
}

impl<T: Numeric> std::fmt::Debug for StabilityResult<T>
where
    T::Float: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StabilityResult")
            .field("status", &self.status)
            .field("metrics", &self.metrics)
            .field("confidence", &self.confidence)
            .field("stability_index", &self.stability_index)
            .field("explanation", &self.explanation)
            .field("method", &self.method)
            .finish()
    }
}

impl<T: Numeric> StabilityResult<T> {
    /// Create a new stability result
    pub fn new(
        status: StabilityStatus<T>,
        metrics: StabilityMetrics<T>,
        confidence: T::Float,
        method: String,
    ) -> Self {
        Self {
            status,
            metrics,
            confidence,
            stability_index: None,
            explanation: String::new(),
            method,
        }
    }
    
    /// Set the stability index
    pub fn with_stability_index(mut self, index: usize) -> Self {
        self.stability_index = Some(index);
        self
    }
    
    /// Set the explanation
    pub fn with_explanation(mut self, explanation: String) -> Self {
        self.explanation = explanation;
        self
    }
    
    /// Check if the system is stable
    pub fn is_stable(&self) -> bool {
        matches!(self.status, StabilityStatus::Stable)
    }
    
    /// Check if the system is unstable
    pub fn is_unstable(&self) -> bool {
        matches!(self.status, StabilityStatus::Unstable { .. })
    }
    
    /// Check if stability is unknown
    pub fn is_unknown(&self) -> bool {
        matches!(self.status, StabilityStatus::Unknown)
    }
}

/// Trait for online (real-time) stability detection
/// 
/// Online analyzers maintain state and don't fit the parameterized pattern.
/// They may internally use spread/quantile estimators but manage them as state.
pub trait OnlineStabilityAnalyzer<T: Numeric> {
    /// Add a new observation and update stability assessment
    fn add_observation(&mut self, value: T) -> StabilityStatus<T>;
    
    /// Get the current stability status
    fn current_status(&self) -> StabilityStatus<T>;
    
    /// Get the current metrics
    fn current_metrics(&self) -> StabilityMetrics<T>;
    
    /// Reset the analyzer
    fn reset(&mut self);
}

/// Parameterized online stability analyzer that uses external estimators
/// 
/// For online analyzers that benefit from shared estimator state across
/// multiple analyzer instances (e.g., in ensemble methods).
pub trait OnlineStabilityAnalyzerWithEstimators<T, S, Q>
where
    T: Numeric,
    S: SpreadEstimator<T, Q>,
    Q: QuantileEstimator<T>,
{
    /// Add observation with external estimators
    fn add_observation_with_estimators(
        &mut self,
        value: T,
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> StabilityStatus<T>;
    
    /// Get the current stability status
    fn current_status(&self) -> StabilityStatus<T>;
    
    /// Get the current metrics
    fn current_metrics(&self) -> StabilityMetrics<T>;
    
    /// Reset the analyzer
    fn reset(&mut self);
}

/// Trait for multi-method stability analysis
/// 
/// This is the original trait maintained for backward compatibility.
/// New code should use the parameterized version below.
pub trait CompositeStabilityAnalyzer<T: Numeric> {
    /// Analyze using multiple methods and combine results
    fn analyze_composite(&self, data: &[T]) -> Result<CompositeStabilityResult<T>>;
}

/// Parameterized composite stability analyzer
/// 
/// For ensemble methods that combine multiple analyzers with shared estimators
pub trait CompositeStabilityAnalyzerWithEstimators<T, S, Q>
where
    T: Numeric,
    S: SpreadEstimator<T, Q>,
    Q: QuantileEstimator<T>,
{
    /// Analyze using multiple methods with shared estimators
    fn analyze_composite_with_estimators(
        &self,
        data: &[T],
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<CompositeStabilityResult<T>>;
}

/// Result from composite stability analysis
#[derive(Clone)]
pub struct CompositeStabilityResult<T: Numeric> {
    /// Individual results from each method
    pub individual_results: Vec<StabilityResult<T>>,
    
    /// Combined stability status
    pub combined_status: StabilityStatus<T>,
    
    /// Overall confidence based on method agreement
    pub overall_confidence: T::Float,
    
    /// Agreement between methods
    pub method_agreement: T::Float,
}

impl<T: Numeric> fmt::Debug for CompositeStabilityResult<T>
where
    T::Float: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompositeStabilityResult")
            .field("individual_results", &self.individual_results)
            .field("combined_status", &self.combined_status)
            .field("overall_confidence", &self.overall_confidence)
            .field("method_agreement", &self.method_agreement)
            .finish()
    }
}

impl<T: Numeric> CompositeStabilityResult<T> {
    /// Check if all methods agree on stability
    pub fn unanimous_stable(&self) -> bool {
        self.individual_results.iter().all(|r| r.is_stable())
    }
    
    /// Check if any method detects instability
    pub fn any_unstable(&self) -> bool {
        self.individual_results.iter().any(|r| r.is_unstable())
    }
    
    /// Get the most confident result
    pub fn most_confident(&self) -> Option<&StabilityResult<T>> {
        self.individual_results
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
    }
}