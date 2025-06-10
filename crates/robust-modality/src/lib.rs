//! Robust multimodality detection using the Lowlands algorithm
//!
//! This crate provides algorithms for detecting multiple modes in data distributions
//! using the "lowland" algorithm. The approach identifies modes by finding areas of
//! low density (lowlands) between peaks in the data distribution.
//!
//! # Algorithm Overview
//!
//! The Lowlands algorithm works by:
//! 1. Creating a density histogram of the data
//! 2. Identifying peaks (local maxima) in the histogram
//! 3. Finding "lowlands" - areas of low density between peaks
//! 4. Determining mode boundaries based on lowland positions
//!
//! # Key Features
//!
//! - **Robust**: Works well with outliers and non-normal distributions
//! - **Configurable**: Adjustable sensitivity and precision parameters
//! - **Flexible**: Generic over quantile estimators and histogram builders
//! - **Visualizable**: Optional plotting support for analysis
//! - **Integration**: Works with the robust statistics framework
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```rust
//! use robust_modality::{default_detector, ModalityDetector};
//! use robust_quantile::{estimators::harrell_davis, QuantileAdapter, HDWeightComputer};
//! use robust_core::{execution::scalar_sequential, UnifiedWeightCache, CachePolicy};
//!
//! // Create bimodal data
//! let mut data = Vec::new();
//! data.extend((0..50).map(|x| x as f64 / 10.0));  // First mode: 0-5
//! data.extend((80..130).map(|x| x as f64 / 10.0)); // Second mode: 8-13
//!
//! // Create detector with default settings
//! let detector = default_detector();
//! 
//! // Create estimator and cache
//! let engine = scalar_sequential();
//! let hd = harrell_davis(engine);
//! let estimator = QuantileAdapter::new(hd);
//! let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
//!
//! // Detect modes
//! let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();
//!
//! println!("Detected {} modes", result.mode_count());
//! for (i, mode) in result.modes().iter().enumerate() {
//!     println!("Mode {}: center={:.2}, range=[{:.2}, {:.2}]",
//!         i + 1, mode.location, mode.left_bound, mode.right_bound);
//! }
//! ```
//!
//! ## Custom Configuration
//!
//! ```rust
//! use robust_modality::{detector_with_params, ModalityDetector};
//! use robust_quantile::{estimators::harrell_davis, QuantileAdapter, HDWeightComputer};
//! use robust_core::{execution::scalar_sequential, UnifiedWeightCache, CachePolicy};
//!
//! let data = vec![1.0, 2.0, 3.0, 8.0, 9.0, 10.0];
//!
//! // Create detector with custom parameters
//! let detector = detector_with_params(0.3, 0.02);
//! 
//! // Create estimator and cache
//! let engine = scalar_sequential();
//! let hd = harrell_davis(engine);
//! let estimator = QuantileAdapter::new(hd);
//! let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
//!
//! let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();
//! ```

pub mod builder;
pub mod detector;
pub mod traits;
pub mod types;
pub mod visualization;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_data;

// Re-exports
pub use builder::{
    default_detector, detector_with_params, sensible_defaults, ModalityDetectorBuilder,
};
pub use detector::LowlandModalityDetector;
pub use traits::ModalityDetector;
pub use types::{ModalityResult, Mode};
pub use visualization::{ModalityVisualizer, NullModalityVisualizer};

// Type aliases for convenience
pub type LowlandDetector = LowlandModalityDetector<
    f64,
    robust_histogram::QRDEBuilderWithSteps,
    NullModalityVisualizer,
>;

// Convenience functions
/// Create a detector with default settings
pub fn create_default_detector() -> LowlandDetector {
    self::builder::default_detector()
}

/// Create a detector with custom sensitivity and precision
pub fn detector(sensitivity: f64, precision: f64) -> LowlandDetector {
    self::builder::detector_with_params(sensitivity, precision)
}
