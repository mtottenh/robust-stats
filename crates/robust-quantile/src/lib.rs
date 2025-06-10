//! High-performance quantile estimation
//!
//! This crate provides implementations of the Harrell-Davis quantile estimator
//! and its trimmed variant, leveraging SIMD operations and intelligent caching
//! for maximum performance.
//!
//! # Features
//!
//! - **Harrell-Davis Quantile Estimator**: Smooth, differentiable quantile estimation
//! - **Trimmed Harrell-Davis**: Reduces influence of extreme order statistics
//! - **SIMD Optimization**: Leverages robust-core's three-layer architecture
//! - **Intelligent Caching**: Reuses expensive weight computations
//! - **Batch Processing**: Efficient processing of multiple datasets
//! - **Confidence Intervals**: Maritz-Jarrett confidence intervals
//!
//! # Example
//!
//! ```rust
//! use robust_quantile::{estimators::harrell_davis, QuantileEstimator, HDWeightComputer};
//! use robust_core::{simd_sequential, CachePolicy, UnifiedWeightCache};
//!
//! // Create estimator with SIMD engine
//! let engine = simd_sequential();
//! let hd = harrell_davis(engine);
//!
//! // Create cache for weight reuse
//! let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::Lru { max_entries: 1024 });
//!
//! // Estimate median
//! let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let median = hd.quantile(&mut data, 0.5, &cache).unwrap();
//! ```

pub mod adapter_impls;
pub mod confidence;
pub mod error;
pub mod estimators;
pub mod factories;
pub mod kernels;
pub mod processing_traits;
pub mod quantile_variance_adapter;
pub mod traits;
pub mod weights;

// Re-export main types
pub use error::{Error, Result};
pub use estimators::{harrell_davis, HarrellDavis};
pub use traits::{QuantileEstimator, QuantileKernel};
pub use weights::{HDWeightComputer, TrimmedHDWeightComputer};
// Re-export from robust-core
pub use robust_core::SparseWeights;

// Re-export concrete estimators
pub use estimators::{ConstantWidth, LinearWidth, SqrtWidth, TrimmedHarrellDavis, WidthFunction};

// Re-export confidence interval support
pub use confidence::{MaritzJarrett, QuantileWithMoments};

// Re-export adapters
pub use adapter_impls::{quantile_adapter, quantile_adapter_with_default, QuantileAdapter};
pub use quantile_variance_adapter::QuantileVarianceAdapter;

// Re-export factories
pub use factories::{
    harrell_davis_factory, trimmed_hd_constant_factory, trimmed_hd_linear_factory,
    trimmed_hd_sqrt_factory, HarrellDavisFactory, TrimmedHDConstantFactory, TrimmedHDLinearFactory,
    TrimmedHDSqrtFactory,
};

// Type aliases for common use cases
pub type DefaultQuantileEstimator<T = f64> =
    HarrellDavis<T, robust_core::SequentialEngine<T, robust_core::ScalarBackend>>;

/// Function type for quantile ratio calculations
pub type QuantileRatioFunction = fn(&[f64], &[f64]) -> f64;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{
        Error, HarrellDavis, LinearWidth, QuantileEstimator, Result, SqrtWidth, TrimmedHarrellDavis,
    };
}
