//! Robust effect size measurement
//!
//! This crate provides various effect size measures designed to work with
//! robust estimators and be resistant to outliers. Effect sizes quantify
//! the magnitude of differences between groups or relationships between variables.
//!
//! # Overview
//!
//! Effect sizes are crucial in statistics for understanding practical significance
//! beyond statistical significance. Traditional effect size measures like Cohen's d
//! can be heavily influenced by outliers. This crate provides robust alternatives
//! and allows using any robust location and scale estimators.
//!
//! # Supported Effect Sizes
//!
//! ## Location-based measures:
//! - **Cohen's d**: Standardized mean difference (with robust variants)
//! - **Glass's delta**: Uses control group standard deviation only
//! - **Hedges' g**: Bias-corrected Cohen's d
//!
//! ## Non-parametric measures:
//! - **Cliff's delta**: Non-parametric effect size based on dominance
//! - **Common Language Effect Size (CLES)**: Probability of superiority
//!
//! ## Correlation-based:
//! - **Point-biserial correlation**: For binary/continuous relationships
//! - **Eta-squared (η²)**: Proportion of variance explained
//! - **Omega-squared (ω²)**: Less biased than eta-squared
//!
//! # Examples
//!
//! ## Basic Cohen's d with robust estimators
//!
//! ```rust,ignore
//! use robust_effect::{CohenD, StandardizedEffectSize};
//! use robust_spread::Mad;
//! use robust_quantile::estimators::harrell_davis;
//! use robust_core::{execution::scalar_sequential, UnifiedWeightCache, CachePolicy};
//!
//! let mut group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let mut group2 = vec![3.0, 4.0, 5.0, 6.0, 7.0];
//!
//! // Set up estimators
//! let engine = scalar_sequential();
//! let quantile_est = harrell_davis(engine.clone());
//! let mad = Mad::new(engine.primitives().clone());
//! let cache = UnifiedWeightCache::new(
//!     robust_quantile::HDWeightComputer::new(),
//!     CachePolicy::NoCache,
//! );
//!
//! // Compute Cohen's d using median and MAD  
//! let cohen_d = CohenD::new();
//! let effect_size = cohen_d.compute_with_estimators(
//!     &mut group1,
//!     &mut group2,
//!     &quantile_est,  // Using quantile estimator as location
//!     &mad,
//!     &quantile_est,
//!     &cache
//! ).unwrap();
//! println!("Robust Cohen's d: {:.3}", effect_size.magnitude);
//! ```
//!
//! ## Cliff's Delta (non-parametric)
//!
//! ```rust
//! use robust_effect::{CliffDelta, NonParametricEffectSize};
//!
//! let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let group2 = vec![3.0, 4.0, 5.0, 6.0, 7.0];
//!
//! let cliff_delta = CliffDelta::new();
//! let effect_size = cliff_delta.compute(&group1, &group2).unwrap();
//! println!("Cliff's delta: {:.3}", effect_size.magnitude);
//! ```

mod cles;
mod cliff_delta;
mod cohen_d;
mod correlation;
mod glass_delta;
mod hedges_g;
mod traits;
mod types;
mod variance_explained;


// Re-exports
pub use cles::CommonLanguageEffectSize;
pub use cliff_delta::CliffDelta;
pub use cohen_d::{CohenD, MeanEstimator, StdEstimator};
pub use correlation::PointBiserialCorrelation;
pub use glass_delta::GlassDelta;
pub use hedges_g::HedgesG;
pub use traits::{EffectSizeEstimator, StandardizedEffectSize, NonParametricEffectSize};
pub use types::{EffectSize, EffectSizeInterpretation, EffectSizeMagnitude, EffectSizeType};
pub use variance_explained::{EtaSquared, OmegaSquared};

// Convenience constructors
pub fn cohen_d() -> CohenD {
    CohenD::new()
}

pub fn cohen_d_welch() -> CohenD {
    CohenD::new().with_welch_correction()
}

pub fn cliff_delta() -> CliffDelta {
    CliffDelta::new()
}

pub fn cles() -> CommonLanguageEffectSize {
    CommonLanguageEffectSize::new()
}
