//! Scalar backend implementation
//!
//! This backend provides generic implementations that work for all numeric types
//! without using any SIMD instructions.

use crate::primitives::ComputePrimitives;
use crate::Numeric;

/// Scalar backend - works for all numeric types
#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarBackend;

impl ScalarBackend {
    pub fn new() -> Self {
        Self
    }
}

// Generic implementation for all types
impl<T: Numeric> ComputePrimitives<T> for ScalarBackend {
    fn backend_name(&self) -> &'static str {
        "scalar"
    }

    // All operations use the default implementations from the trait
}