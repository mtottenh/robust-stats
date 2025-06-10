//! Pipeline infrastructure for statistical analysis
//!
//! This module provides the core types and infrastructure for building
//! composable statistical analysis pipelines. The actual pipeline
//! implementation lives in the `statistical-analysis-pipeline` crate.

pub mod context;
pub mod events;
pub mod types;

// Re-export commonly used types
pub use context::{PipelineContext, PipelineContextBuilder, Value};
pub use events::{EventBus, EventHandler, PipelineEvent};
pub use types::*;