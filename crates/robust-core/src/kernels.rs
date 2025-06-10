//! Layer 2: Algorithm-Specific Execution Kernel Traits
//!
//! This module provides the kernel trait definitions that algorithm-specific
//! crates use to implement their computational patterns. Kernels use the 
//! primitive operations from Layer 1 to enable efficient statistical algorithms.
//!
//! # Design Philosophy
//!
//! - **Interface Only**: This module only defines traits, not implementations
//! - **Algorithm-Specific**: Each statistical domain defines its own kernel trait
//! - **Composable**: Kernels can depend on other kernels
//! - **Zero-Copy**: Designed to work directly on slices without allocations

use crate::{primitives::ComputePrimitives, Numeric};

/// Base trait for all statistical kernels
///
/// This trait provides common functionality that all kernels share.
pub trait StatisticalKernel<T: Numeric = f64>: Clone + Send + Sync {
    /// The type of primitives used by this kernel
    type Primitives: ComputePrimitives<T>;
    
    /// Get the primitives used by this kernel
    fn primitives(&self) -> &Self::Primitives;
    
    /// Name of this kernel for debugging/logging
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}