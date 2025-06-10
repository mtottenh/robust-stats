//! Core traits for robust statistics on Polars DataFrames

use polars::prelude::*;
use crate::{Result, QuantileMethod, SpreadMethod, ConfidenceMethod, ChangePointMethod, EffectSizeMethod, QuantileCache};

/// Extension trait for robust statistics operations on Polars DataFrames
pub trait RobustStatsExt {
    /// Compute quantiles for specified columns using global caches
    ///
    /// # Arguments
    /// * `columns` - Column names to compute quantiles for
    /// * `quantiles` - Quantile probabilities (0.0 to 1.0)
    /// * `method` - Quantile estimation method
    ///
    /// # Returns
    /// DataFrame with columns named `{column}_q{quantile}`
    fn robust_quantiles(
        &self,
        columns: &[&str],
        quantiles: &[f64],
        method: QuantileMethod,
    ) -> Result<DataFrame>;
    
    /// Compute quantiles with a custom cache
    ///
    /// # Arguments
    /// * `columns` - Column names to compute quantiles for
    /// * `quantiles` - Quantile probabilities (0.0 to 1.0)
    /// * `method` - Quantile estimation method
    /// * `cache` - Custom cache instance (must match method type)
    ///
    /// # Returns
    /// DataFrame with columns named `{column}_q{quantile}`
    fn robust_quantiles_cached<T>(
        &self,
        columns: &[&str],
        quantiles: &[f64],
        method: QuantileMethod,
        cache: &QuantileCache<T>,
    ) -> Result<DataFrame>
    where
        T: robust_core::Numeric + num_traits::NumCast + 'static;

    /// Compute spread measures (MAD, IQR, etc.)
    ///
    /// # Arguments
    /// * `columns` - Column names to compute spread for
    /// * `method` - Spread estimation method
    ///
    /// # Returns
    /// DataFrame with columns named `{column}_{method}`
    fn robust_spread(
        &self,
        columns: &[&str],
        method: SpreadMethod,
    ) -> Result<DataFrame>;
    
    /// Compute spread measures with a custom cache
    ///
    /// # Arguments
    /// * `columns` - Column names to compute spread for
    /// * `method` - Spread estimation method
    /// * `cache` - Custom cache instance (must be compatible with spread method)
    ///
    /// # Returns
    /// DataFrame with columns named `{column}_{method}`
    fn robust_spread_cached<T>(
        &self,
        columns: &[&str],
        method: SpreadMethod,
        cache: &QuantileCache<T>,
    ) -> Result<DataFrame>
    where
        T: robust_core::Numeric + num_traits::NumCast + 'static;

    /// Compute confidence intervals
    ///
    /// # Arguments
    /// * `columns` - Column names to compute CIs for
    /// * `level` - Confidence level (e.g., 0.95 for 95% CI)
    /// * `method` - CI computation method
    ///
    /// # Returns
    /// DataFrame with columns `{column}_lower` and `{column}_upper`
    fn robust_confidence(
        &self,
        columns: &[&str],
        level: f64,
        method: ConfidenceMethod,
    ) -> Result<DataFrame>;

    /// Detect changepoints in time series
    ///
    /// # Arguments
    /// * `column` - Column name containing time series data
    /// * `method` - Changepoint detection method
    ///
    /// # Returns
    /// DataFrame with detected changepoint indices and metadata
    fn robust_changepoints(
        &self,
        column: &str,
        method: ChangePointMethod,
    ) -> Result<DataFrame>;

    /// Analyze stability of time series
    ///
    /// # Arguments
    /// * `columns` - Column names to analyze
    /// * `window_size` - Size of rolling window
    ///
    /// # Returns
    /// DataFrame with stability metrics
    fn robust_stability(
        &self,
        columns: &[&str],
        window_size: usize,
    ) -> Result<DataFrame>;

    /// Detect modality (number of modes)
    ///
    /// # Arguments
    /// * `columns` - Column names to analyze
    /// * `sensitivity` - Detection sensitivity (0.0 to 1.0)
    ///
    /// # Returns
    /// DataFrame with modality information
    fn robust_modality(
        &self,
        columns: &[&str],
        sensitivity: f64,
    ) -> Result<DataFrame>;

    /// Compute effect sizes between groups
    ///
    /// # Arguments
    /// * `group_col` - Column containing group labels
    /// * `value_col` - Column containing values to compare
    /// * `method` - Effect size method
    ///
    /// # Returns
    /// DataFrame with effect sizes between all group pairs
    fn robust_effect_size(
        &self,
        group_col: &str,
        value_col: &str,
        method: EffectSizeMethod,
    ) -> Result<DataFrame>;
}

/// Extension trait for lazy evaluation support
pub trait RobustStatsLazyExt {
    /// Create a quantile expression
    fn robust_quantile(self, p: f64, method: QuantileMethod) -> Self;
    
    /// Create a MAD expression
    fn robust_mad(self) -> Self;
    
    /// Create an IQR expression
    fn robust_iqr(self) -> Self;
}

/// Trait for creating expressions
pub trait RobustExpr {
    /// Convert to a Polars expression
    fn to_expr(self) -> Expr;
}