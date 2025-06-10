//! Clean backend implementations without delegation
//!
//! This module provides concrete backend types with direct implementations.
//! No Box, no dyn, just simple types with compile-time dispatch.

pub mod scalar;
pub mod avx2;
#[cfg(feature = "avx512")]
pub mod avx512;

// Re-export the main backend types
pub use scalar::ScalarBackend;
pub use avx2::Avx2Backend;
#[cfg(feature = "avx512")]
pub use avx512::Avx512Backend;

use crate::Numeric;
use super::ComputePrimitives;

/// SSE backend for older x86_64 processors
#[derive(Clone, Copy, Debug)]
pub struct SseBackend;

impl SseBackend {
    /// Create a new SSE backend
    ///
    /// # Panics
    /// Panics if the CPU doesn't support SSE2 instructions
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        #[cfg(all(target_arch = "x86_64", feature = "sse"))]
        {
            if !is_x86_feature_detected!("sse2") {
                panic!("SSE backend requested but CPU doesn't support SSE2 instructions");
            }
            Self
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "sse")))]
        {
            panic!("SSE backend not available: not compiled with SSE support");
        }
    }

    /// Check if SSE is available on this CPU
    pub fn is_available() -> bool {
        #[cfg(all(target_arch = "x86_64", feature = "sse"))]
        {
            is_x86_feature_detected!("sse2")
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "sse")))]
        {
            false
        }
    }
}

// SSE implementations would go here
#[cfg(all(target_arch = "x86_64", feature = "sse"))]
impl ComputePrimitives<f64> for SseBackend {
    fn backend_name(&self) -> &'static str {
        "sse"
    }

    fn simd_width(&self) -> usize {
        2 // SSE processes 2 f64s at once
    }

    // TODO: Add SSE implementations
}

// Fallback for non-SSE builds
#[cfg(not(all(target_arch = "x86_64", feature = "sse")))]
impl<T: Numeric> ComputePrimitives<T> for SseBackend {
    fn backend_name(&self) -> &'static str {
        "sse (unavailable)"
    }
}

/// Backend selection trait for automatic backend choice
pub trait SelectBackend: Numeric {
    /// The backend type to use for this numeric type
    type Backend: ComputePrimitives<Self>;

    /// Get an instance of the backend
    fn backend() -> Self::Backend;
}

// Default implementations - most types use scalar
impl SelectBackend for i32 {
    type Backend = ScalarBackend;
    fn backend() -> Self::Backend {
        ScalarBackend
    }
}

impl SelectBackend for u32 {
    type Backend = ScalarBackend;
    fn backend() -> Self::Backend {
        ScalarBackend
    }
}

impl SelectBackend for i64 {
    type Backend = ScalarBackend;
    fn backend() -> Self::Backend {
        ScalarBackend
    }
}

impl SelectBackend for u64 {
    type Backend = ScalarBackend;
    fn backend() -> Self::Backend {
        ScalarBackend
    }
}

// f64 uses AVX2 when available
impl SelectBackend for f64 {
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    type Backend = Avx2Backend;

    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    type Backend = ScalarBackend;

    fn backend() -> Self::Backend {
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        {
            if Avx2Backend::is_available() {
                return Avx2Backend::new();
            }
            // This line should never be reached if the cfg matches our type
            panic!("AVX2 backend unavailable");
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
        {
            ScalarBackend
        }
    }
}

// f32 uses AVX2 when available
impl SelectBackend for f32 {
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    type Backend = Avx2Backend;

    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    type Backend = ScalarBackend;

    fn backend() -> Self::Backend {
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        {
            if Avx2Backend::is_available() {
                return Avx2Backend::new();
            }
            // This line should never be reached if the cfg matches our type
            panic!("AVX2 backend unavailable");
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
        {
            ScalarBackend
        }
    }
}

/// Get the best available backend for the current platform
pub fn best_available_backend<T: SelectBackend>() -> T::Backend {
    T::backend()
}