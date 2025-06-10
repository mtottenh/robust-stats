//! Robust changepoint detection algorithms
//!
//! This crate provides algorithms for detecting changepoints in time series data,
//! focusing on robust methods that work well in the presence of noise and outliers.
//!
//! # Algorithms
//!
//! ## Simple Methods
//! - **CUSUM** (Cumulative Sum): Detects changes in mean
//! - **Slopes**: Detects changes in linear trends
//! - **EWMA** (Exponentially Weighted Moving Average): Detects gradual changes
//!
//! ## Advanced Methods
//! - **PolynomialSlopes**: Fits polynomials to rolling windows for complex trend detection
//!   - Supports steady-state detection for system performance analysis
//!   - Can fit polynomials of any degree (linear, quadratic, etc.)
//!   - Available with Polars integration for efficient large-scale analysis
//!
//! ## Usage
//!
//! ```rust
//! use robust_changepoint::{CusumDetector, SimpleDetector};
//! use robust_core::{simd_sequential, ExecutionEngine};
//!
//! // Create sample data with a changepoint
//! let data: Vec<f64> = (0..50).map(|i| if i < 25 { 0.0 } else { 5.0 }).collect();
//!
//! // Detect changepoints using CUSUM
//! let primitives = simd_sequential().primitives().clone();
//! let detector = CusumDetector::new(primitives, 4.0, 0.5);
//! let result = detector.detect_simple(&data).unwrap();
//!
//! println!("Detected changepoints: {:?}", result.changepoints());
//! ```

pub mod cusum;
pub mod ewma;
pub mod kernel;
pub mod polynomial_slopes;
pub mod slopes;
pub mod traits;
pub mod types;
pub mod visualization;


// Examples are now in the examples/ directory

// Re-exports - New parameterized detectors
pub use cusum::CusumDetector;
pub use ewma::EwmaDetector;
pub use polynomial_slopes::{PolynomialSlopesDetector, AdaptivePolynomialSlopesDetector, PolynomialSlopesParameters, AdaptivePolynomialSlopesParameters};
pub use slopes::SlopesDetector;

// Core traits - new parameterized design
pub use traits::{
    ChangePointDetector, ChangePointDetectorProperties,
    QuantileBasedDetector, SimpleDetector,
    ConfigurableDetector, ConfidenceScoring,
    OnlineDetector, BatchChangePointDetector,
};

// Kernels for domain-specific operations
pub use kernel::{
    CusumKernel, PolynomialKernel, EwmaKernel, WindowKernel,
};

pub use types::{ChangePoint, ChangePointResult};
pub use visualization::{ChangePointVisualizer, NullChangePointVisualizer, PolynomialVisualizer};

// Note: PolynomialSlopesDetector is now parameterized by ComputePrimitives
// Example usage:
// let primitives = simd_primitives();
// let detector = PolynomialSlopesDetector::new(primitives, window_size, degree, threshold);

// Note: To use visualization, import robust-viz and create a type alias like:
// pub type VisualizingPolynomialSlopesDetector =
//     robust_changepoint::polynomial_slopes::PolynomialSlopesDetector<robust_viz::CharmingChangePointVisualizer>;
