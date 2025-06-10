//! Robust spread measurements for statistical analysis
//!
//! This crate provides various robust estimators of scale/spread including:
//! - MAD (Median Absolute Deviation)
//! - QAD (Quantile Absolute Deviation)
//! - IQR (Interquartile Range)
//! - Trimmed standard deviation
//! - Winsorized standard deviation
//!
//! # Overview
//!
//! Spread (or scale) estimators measure the variability in data. Traditional
//! measures like standard deviation are sensitive to outliers. This crate
//! provides robust alternatives with high breakdown points.
//!
//! # Estimator Properties
//!
//! | Estimator | Breakdown Point | Efficiency | Use When |
//! |-----------|----------------|------------|----------|
//! | Std Dev | 0% | 100% | Data is clean, normal |
//! | MAD | 50% | 37% | Heavy contamination possible |
//! | IQR | 25% | 65% | Moderate robustness needed |
//! | Trimmed Std | Variable | Variable | Known contamination level |
//! | QAD | Variable | Variable | Custom robustness needed |
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```rust,ignore
//! # #[cfg(feature = "quantile")]
//! # {
//! use robust_core::{ComputePrimitives, simd_primitives};
//! use robust_spread::{Mad, StandardizedMad, SpreadEstimator, SpreadEstimatorProperties};
//! use robust_quantile::estimators::harrell_davis;
//!
//! let mut sample = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // Outlier!
//!
//! // Create primitives and estimator
//! let primitives = simd_primitives();
//! let estimator = harrell_davis(robust_core::simd_sequential());
//! let cache = robust_core::UnifiedWeightCache::new(
//!     robust_quantile::HDWeightComputer,
//!     robust_core::CachePolicy::NoCache
//! );
//!
//! // MAD is robust
//! let mad = Mad::new(primitives.clone());
//! let mad_value = mad.estimate(&mut sample.clone(), &estimator, &cache).unwrap();
//! println!("MAD: {:.2}", mad_value); // ~1.5, ignores outlier
//!
//! // Standardized MAD (comparable to std dev for normal data)
//! let smad = StandardizedMad::new(primitives);
//! let smad_value = smad.estimate(&mut sample, &estimator, &cache).unwrap();
//! println!("Standardized MAD: {:.2}", smad_value);
//! # }
//! ```
//!
//! ## Choosing an Estimator
//!
//! ```rust,ignore
//! use robust_spread::{SpreadEstimator, SpreadEstimatorProperties};
//! use robust_core::ComputePrimitives;
//! 
//! // Example of selecting estimators based on contamination level
//! // (actual implementation would need to specify quantile estimator parameter)
//! ```
//!
//! ## Scale Ratio Analysis
//!
//! ```rust,ignore
//! # #[cfg(feature = "quantile")]
//! # {
//! use robust_spread::{StandardizedMad, SpreadEstimator};
//! use robust_core::primitives::ScalarBackend;
//! use robust_quantile::estimators::harrell_davis;
//!
//! let mut sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let mut sample2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
//!
//! let primitives = ScalarBackend::new();
//! let estimator = harrell_davis(robust_core::auto_engine());
//! let cache = robust_core::UnifiedWeightCache::new(
//!     robust_quantile::HDWeightComputer::new(),
//!     robust_core::CachePolicy::NoCache
//! );
//! 
//! let smad = StandardizedMad::new(primitives);
//! let scale1 = smad.estimate(&mut sample1, &estimator, &cache).unwrap();
//! let scale2 = smad.estimate(&mut sample2, &estimator, &cache).unwrap();
//! let scale_change = scale2 / scale1;
//!
//! if scale_change > 1.5 {
//!     println!("Significant increase in variability");
//! }
//! # }
//! ```

#[cfg(feature = "quantile")]
pub mod adapters;
#[cfg(feature = "quantile")]
pub mod factories;
#[cfg(feature = "quantile")]
pub mod iqr;
#[cfg(feature = "quantile")]
pub mod kernels;
#[cfg(feature = "quantile")]
pub mod mad;
#[cfg(feature = "quantile")]
pub mod moments;
#[cfg(feature = "quantile")]
pub mod qad;
#[cfg(feature = "quantile")]
pub mod traits;
#[cfg(feature = "quantile")]
pub mod trimmed;


// Re-exports
#[cfg(feature = "quantile")]
pub use adapters::{SpreadAdapter, spread_adapter};
#[cfg(feature = "quantile")]
pub use iqr::{Iqr, IqrEstimator};
#[cfg(feature = "quantile")]
pub use kernels::{IqrKernel, KurtosisKernel, MadKernel, QadKernel, SkewnessKernel, TrimmedKernel, WinsorizedKernel};
#[cfg(feature = "quantile")]
pub use mad::{mad_sorted_with_cache, mad_with_cache, Mad, StandardizedMad};
#[cfg(feature = "quantile")]
pub use qad::{OptimalQad, Qad, StandardQad};
#[cfg(feature = "quantile")]
pub use traits::{RobustScale, SpreadEstimator, SpreadEstimatorProperties, SpreadKernel};
#[cfg(feature = "quantile")]
pub use trimmed::{TrimmedStd, WinsorizedStd};

// Moments re-exports
#[cfg(feature = "quantile")]
pub use moments::{
    classical_kurtosis, classical_skewness, crow_siddiqui_kurtosis, medcouple_skewness,
    robust_kurtosis, robust_skewness, RobustKurtosis, RobustSkewness,
};

// Factory re-exports
#[cfg(feature = "quantile")]
pub use factories::{mad_factory, qad_factory, iqr_factory, QADFactory, MADFactory, IqrFactory};

#[cfg(all(test, feature = "quantile"))]
mod tests {
    use super::*;
    use robust_quantile::estimators::harrell_davis;
    #[test]
    fn test_mad_basic() {
        let mut sample = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let estimator = harrell_davis(robust_core::auto_engine());
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            robust_core::CachePolicy::NoCache,
        );
        let mad_est = Mad::<f64, _>::new(robust_core::primitives::ScalarBackend::new());
        let mad = mad_est.estimate(&mut sample, &estimator, &cache).unwrap();
        // With Harrell-Davis estimator, expecting ~1.26
        assert!((mad - 1.26).abs() < 0.1);
    }
}
