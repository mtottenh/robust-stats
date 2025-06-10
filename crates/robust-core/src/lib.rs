//! Core traits and types for robust statistical analysis
//!
//! This crate provides the foundational three-layer architecture that enables
//! high-performance statistical computing with compile-time optimization.

#![cfg_attr(feature = "nightly", feature(stdarch_x86_avx512))]
//!
//! # Architecture Overview
//!
//! The library is organized into three layers:
//!
//! 1. **Layer 1: Primitives** - Type-safe SIMD operations with zero-cost dispatch
//! 2. **Layer 2: Execution Engines** - Unified parallelism and SIMD control
//! 3. **Layer 3: Batch Processing** - Orchestration and caching
//!
//! # Design Philosophy
//!
//! - **Zero-Cost Abstractions**: All optimizations resolved at compile time
//! - **Type-Safe Operations**: SIMD dispatch through type-level operations
//! - **Composable**: Small, focused components that combine efficiently
//! - **No Hidden Allocations**: Explicit memory management throughout
//!
//! # Example
//!
//! ```rust
//! use robust_core::{
//!     execution::{scalar_sequential, ExecutionEngine},
//!     ComputePrimitives, ScalarBackend,
//! };
//!
//! // Create an execution engine
//! let engine = scalar_sequential();  // Uses ScalarBackend
//!
//! // Basic operations with the engine's primitives
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let sum = engine.primitives().sum(&data);
//! let mean = engine.primitives().mean(&data);
//!
//! println!("Sum: {}, Mean: {}", sum, mean);
//! ```
#![feature(portable_simd)]
// Re-export submodules
pub mod adapters;
pub mod batch;
pub mod builder;
pub mod comparison;
pub mod error;
pub mod execution;
pub mod factory;
pub mod kernels;
pub mod math;
pub mod numeric;
pub mod pipeline;
pub mod primitives;
pub mod sparse;
pub mod tiled;
pub mod traits;
pub mod utils;
pub mod weights;
pub mod workspace;

// Re-export core types
pub use error::{Error, Result};

pub use execution::{
    auto_engine, scalar_sequential, simd_sequential, ExecutionEngine, ExecutionStrategy,
    SequentialEngine,
};
#[cfg(feature = "parallel")]
pub use execution::{scalar_parallel, simd_parallel, ParallelEngine};

pub use primitives::{
    best_available_backend, best_backend_name, scalar_backend, Avx2Backend, ComputePrimitives,
    ScalarBackend, SelectBackend,
};

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
pub use primitives::avx2_backend;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use primitives::{avx512_backend, Avx512Backend};

#[cfg(all(target_arch = "x86_64", feature = "sse"))]
pub use primitives::{sse_backend, SseBackend};

pub use kernels::StatisticalKernel;
pub use sparse::{DenseWeights, SparseWeights, Weights};
pub use tiled::{
    SoaTileBuffer, SparseTile, SparseTileData, TileEntry, TileStats, TiledSparseMatrix,
};
pub use weights::{QuantileSet, UnifiedWeightCache, WeightComputer, WeightKey, WeightValue};

#[cfg(feature = "benchmark-variants")]
pub use tiled::benchmark_variants;

pub use batch::{
    // Workspace types from new module
    BatchProcessor,
    CachePolicy,
    CacheStats,
    CacheableComputation,
    ComputationCache,
    MemoryLayoutAware,
    ProcessingStrategy,
};
pub use workspace::{
    with_bootstrap_workspace, with_f64_bootstrap_workspace, AlignedBuffer, BootstrapWorkspace,
    BufferPool, CheckedOutBuffer, IndexWorkspace, ResampleWorkspace, SortWorkspace,
    ArcBufferPool, ArcBuffer,
};

pub use traits::{
    BatchCharacteristics, BatchDimension, CentralTendencyEstimator, EstimatorWithVariance,
    HasKernel, OptimizationFlags, ParallelizationHint, PerformanceIntrospection, RobustEstimator,
    TwoSampleEstimator,
};

pub use builder::{BuilderState, EstimatorBuilder, NeedsCache, NeedsEngine, Ready};

pub use comparison::{
    BatchQuantileEstimator, LinearComparison, QuantileRatioComparison, QuantileShiftComparison,
    RatioComparison, ShiftComparison, StatefulEstimator, TwoSampleComparison,
};

pub use factory::EstimatorFactory;

pub use adapters::{central_tendency_adapter, CentralTendencyAdapter, NoCache};

// Numeric traits
pub use numeric::{F64Aggregate, Numeric};

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports
pub mod prelude {

    pub use crate::{
        Avx2Backend,
        BatchProcessor,
        CacheableComputation,
        CentralTendencyEstimator,
        ComputePrimitives,
        ExecutionEngine,
        // Generic numeric traits
        Numeric,
        ProcessingStrategy,
        Result,
        // Backend types
        ScalarBackend,
        StatefulEstimator,
        TwoSampleComparison,
    };

    pub use crate::error::Error;

    // Common engine configurations
    #[cfg(feature = "parallel")]
    pub use crate::execution::simd_parallel;
    pub use crate::execution::{
        scalar_sequential, simd_sequential, ExecutionStrategy, SequentialEngine,
    };
    // Builder
    pub use crate::builder::EstimatorBuilder;
}
