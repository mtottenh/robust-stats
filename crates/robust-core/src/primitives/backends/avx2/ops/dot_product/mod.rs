//! Dot product operation for AVX2 backend
//!
//! Computes the dot product of two vectors: sum(a[i] * b[i])

mod f32;
mod f64;
mod scalar;

use crate::{Numeric, primitives::backends::avx2::Avx2Backend};

/// Trait for dot product operation
pub trait DotProduct<T: Numeric> {
    /// Compute dot product with AVX2 optimizations
    unsafe fn compute(
        backend: &Avx2Backend,
        a: &[T],
        b: &[T],
    ) -> T::Aggregate;
}

// Type-specific implementations
impl DotProduct<f32> for f32 {
    unsafe fn compute(
        backend: &Avx2Backend,
        a: &[f32],
        b: &[f32],
    ) -> f64 {
        f32::dot_product_f32(backend, a, b)
    }
}

impl DotProduct<f64> for f64 {
    unsafe fn compute(
        backend: &Avx2Backend,
        a: &[f64],
        b: &[f64],
    ) -> f64 {
        f64::dot_product_f64(backend, a, b)
    }
}

// Generic fallback for other types
macro_rules! impl_dot_product_fallback {
    ($type:ty) => {
        impl DotProduct<$type> for $type {
            unsafe fn compute(
                _backend: &Avx2Backend,
                a: &[$type],
                b: &[$type],
            ) -> <$type as Numeric>::Aggregate {
                let result = scalar::dot_product_scalar(a, b);
                <$type as Numeric>::Aggregate::from(result)
            }
        }
    };
}

impl_dot_product_fallback!(i32);
impl_dot_product_fallback!(i64);
impl_dot_product_fallback!(u32);
impl_dot_product_fallback!(u64);