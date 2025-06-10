//! Configuration types for statistical methods

use std::sync::Arc;
use robust_core::UnifiedWeightCache;
use robust_quantile::{
    HDWeightComputer,
    TrimmedHDWeightComputer,
    weights::{ConstantWidthFn, SqrtWidthFn, LinearWidthFn},
};

/// Width function for trimmed estimators
#[derive(Debug, Clone, Copy)]
pub enum TrimWidth {
    /// Constant width (fixed proportion)
    Constant(f64),
    /// Square root of n width
    Sqrt,
    /// Linear (1/n) width
    Linear,
}

/// Quantile estimation methods
#[derive(Debug, Clone, Copy)]
pub enum QuantileMethod {
    /// Harrell-Davis estimator
    HarrellDavis,
    
    /// Trimmed Harrell-Davis estimator
    TrimmedHarrellDavis { width: TrimWidth },
}

impl Default for QuantileMethod {
    fn default() -> Self {
        Self::HarrellDavis
    }
}

/// Type-safe cache wrapper for different quantile methods
pub enum QuantileCache<T> 
where
    T: robust_core::Numeric + num_traits::NumCast + 'static,
{
    /// Cache for Harrell-Davis estimator
    HarrellDavis(Arc<UnifiedWeightCache<HDWeightComputer<T>, T>>),
    
    /// Cache for Trimmed HD with constant width
    TrimmedConstant {
        cache: Arc<UnifiedWeightCache<TrimmedHDWeightComputer<T, ConstantWidthFn>, T>>,
        width: f64,
    },
    
    /// Cache for Trimmed HD with sqrt width
    TrimmedSqrt(Arc<UnifiedWeightCache<TrimmedHDWeightComputer<T, SqrtWidthFn>, T>>),
    
    /// Cache for Trimmed HD with linear width
    TrimmedLinear(Arc<UnifiedWeightCache<TrimmedHDWeightComputer<T, LinearWidthFn>, T>>),
}

impl<T> QuantileCache<T> 
where
    T: robust_core::Numeric + num_traits::NumCast + 'static,
{
    /// Check if this cache matches the given method
    pub fn matches_method(&self, method: &QuantileMethod) -> bool {
        match (self, method) {
            (QuantileCache::HarrellDavis(_), QuantileMethod::HarrellDavis) => true,
            (
                QuantileCache::TrimmedConstant { width: cache_width, .. }, 
                QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Constant(method_width) }
            ) => (cache_width - method_width).abs() < 1e-10,
            (
                QuantileCache::TrimmedSqrt(_), 
                QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Sqrt }
            ) => true,
            (
                QuantileCache::TrimmedLinear(_), 
                QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Linear }
            ) => true,
            _ => false,
        }
    }
}

/// Spread estimation methods
#[derive(Debug, Clone)]
pub enum SpreadMethod {
    /// Median Absolute Deviation
    Mad,
    
    /// Interquartile Range
    Iqr,
    
    /// Quantile Absolute Deviation
    Qad { probability: f64 },
    
    /// Trimmed Standard Deviation
    TrimmedStd { trim_proportion: f64 },
    
    /// Winsorized Standard Deviation
    WinsorizedStd { winsor_proportion: f64 },
}

impl SpreadMethod {
    /// Get the name of this method
    pub fn name(&self) -> &'static str {
        match self {
            Self::Mad => "mad",
            Self::Iqr => "iqr",
            Self::Qad { .. } => "qad",
            Self::TrimmedStd { .. } => "trimmed_std",
            Self::WinsorizedStd { .. } => "winsorized_std",
        }
    }
}

impl Default for SpreadMethod {
    fn default() -> Self {
        Self::Mad
    }
}

/// Confidence interval methods
#[derive(Debug, Clone)]
pub enum ConfidenceMethod {
    /// Maritz-Jarrett method for quantiles
    MaritzJarrett,
    
    /// Bootstrap methods
    Bootstrap { 
        n_resamples: usize, 
        method: BootstrapType 
    },
    
    /// Asymptotic normal approximation
    Asymptotic,
}

/// Bootstrap CI types
#[derive(Debug, Clone)]
pub enum BootstrapType {
    /// Percentile method
    Percentile,
    
    /// Basic bootstrap
    Basic,
    
    /// Bias-corrected and accelerated
    BCa,
}

/// Change point detection methods
#[derive(Debug, Clone)]
pub enum ChangePointMethod {
    /// CUSUM algorithm
    Cusum { threshold: f64 },
    
    /// EWMA algorithm
    Ewma { lambda: f64 },
    
    /// Polynomial slopes
    PolynomialSlopes { 
        window_size: usize,
        degree: usize,
    },
}

/// Effect size methods
#[derive(Debug, Clone)]
pub enum EffectSizeMethod {
    /// Cohen's d
    CohenD,
    
    /// Hedges' g
    HedgesG,
    
    /// Cliff's Delta
    CliffDelta,
    
    /// Common Language Effect Size
    Cles,
    
    /// Glass's Delta
    GlassDelta { control_group: String },
}