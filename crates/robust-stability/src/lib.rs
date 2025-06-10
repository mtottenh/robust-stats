//! # Robust Stability Analysis
//!
//! This crate provides tools for determining when performance systems reach 
//! reliable, measurable steady states. It implements various stability detection
//! methods including statistical tests, Hilbert transform-based oscillation
//! detection, and multi-scale wavelet analysis.
//!
//! ## Key Features
//!
//! - **Multiple Detection Methods**: Statistical, oscillatory, and trend-based analysis
//! - **Online and Offline**: Real-time detection or retrospective analysis
//! - **Hilbert Transform Integration**: Detect hidden oscillations and instabilities
//! - **Flexible Framework**: Trait-based design for extensibility
//! - **Performance Focused**: Designed for CI/CD and performance testing pipelines
//! - **Zero-Copy Polars Integration**: Native DataFrame support with no allocations

pub mod traits;
pub mod statistical;
pub mod hilbert_analysis;
pub mod online;
pub mod offline;
pub mod types;
pub mod visualization;
pub mod window_traits;
pub mod algorithms;


// Re-exports
pub use traits::{
    StabilityAnalyzer, StabilityResult, OnlineStabilityAnalyzer, 
    CompositeStabilityAnalyzer, CompositeStabilityResult,
    StabilityAnalyzerProperties, StabilityAnalyzerWithSpread,
    StabilityAnalyzerWithQuantile, StabilityAnalyzerWithEstimators
};
pub use types::{StabilityStatus, StabilityMetrics, StabilityParameters};
pub use hilbert_analysis::HilbertStabilityAnalyzer;
pub use statistical::StatisticalStabilityAnalyzer;
pub use online::OnlineStabilityDetector;
pub use offline::OfflineStabilityAnalyzer;