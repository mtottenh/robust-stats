//! Polars integration for robust statistical analysis
//!
//! This crate provides a simple, efficient interface for using robust statistics
//! with Polars DataFrames through a single generic extension trait.
//!
//! # Example
//!
//! ```rust,ignore
//! use polars::prelude::*;
//! use robust_polars::{RobustStatsExt, QuantileMethod, SpreadMethod};
//! use robust_core::ExecutionEngine;
//!
//! let df = df!["values" => [1.0, 2.0, 3.0, 4.0, 5.0]]?;
//!
//! // Compute robust statistics
//! let stats = df
//!     .robust_quantiles(&["values"], &[0.25, 0.5, 0.75], QuantileMethod::HarrellDavis)
//!     .unwrap();
//! ```

mod config;
mod error;
mod methods;
mod traits;

#[cfg(test)]
mod tests;

pub use config::*;
pub use error::{Error, Result};
pub use traits::*;

// Re-export commonly used types from dependencies
pub use robust_core::execution::{ExecutionEngine, HierarchicalExecution, SequentialEngine};
#[cfg(feature = "parallel")]
pub use robust_core::execution::ParallelEngine;