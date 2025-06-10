//! High-performance computational primitives with compile-time dispatch
//!
//! This module provides optimized primitive operations for statistical algorithms
//! with support for multiple numeric types and SIMD backends.
//!
//! # Architecture
//!
//! - Single unified `ComputePrimitives<T>` trait for all operations
//! - Concrete backend types: `ScalarBackend`, `Avx2Backend`, `Avx512Backend`, `SseBackend`
//! - Compile-time backend selection with runtime validation
//! - Zero-cost abstractions - no heap allocation or dynamic dispatch
//!
//! # Usage
//!
//! ```rust,ignore
//! // Explicit backend selection - panics if not supported
//! let backend = Avx2Backend::new();
//! let sum = backend.sum(&data);
//!
//! // Automatic backend selection based on type
//! let backend = f64::backend(); // Uses AVX2 if available
//! let sum = backend.sum(&data);
//!
//! // Use with execution engines
//! let engine = SequentialEngine::new(Avx2Backend::new());
//! ```

pub mod backends;
pub mod traits;

// Re-export the new unified trait and backends
pub use backends::{
    Avx2Backend, ScalarBackend, SelectBackend, SseBackend,
    best_available_backend,
};
#[cfg(feature = "avx512")]
pub use backends::Avx512Backend;
pub use traits::ComputePrimitives;

// Migration complete - old aliases removed

// Convenience functions for backend creation
/// Create a scalar backend (always available)
pub fn scalar_backend() -> ScalarBackend {
    ScalarBackend::new()
}

/// Create an AVX2 backend (panics if not supported)
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
pub fn avx2_backend() -> Avx2Backend {
    Avx2Backend::new()
}

/// Create an AVX512 backend (panics if not supported)
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub fn avx512_backend() -> Avx512Backend {
    Avx512Backend::new()
}

/// Create an SSE backend (panics if not supported)
#[cfg(all(target_arch = "x86_64", feature = "sse"))]
pub fn sse_backend() -> SseBackend {
    SseBackend::new()
}

// Migration complete - old functions removed

/// Get the best available backend name
pub fn best_backend_name() -> &'static str {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if Avx512Backend::is_available() {
            return "avx512";
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        if Avx2Backend::is_available() {
            return "avx2";
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "sse"))]
    {
        if SseBackend::is_available() {
            return "sse";
        }
    }
    "scalar"
}

