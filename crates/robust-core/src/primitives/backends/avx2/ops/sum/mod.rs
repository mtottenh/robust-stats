//! Sum operation for AVX2 backend
//!
//! Computes the sum of all elements in a vector

mod f32;
mod f64;
mod scalar;

use crate::{Numeric, primitives::backends::avx2::Avx2Backend};

/// Trait for sum operation
pub trait Sum<T: Numeric> {
    /// Compute sum with AVX2 optimizations
    unsafe fn compute(
        backend: &Avx2Backend,
        data: &[T],
    ) -> T::Aggregate;
}

// Type-specific implementations
impl Sum<f32> for f32 {
    unsafe fn compute(
        backend: &Avx2Backend,
        data: &[f32],
    ) -> f64 {
        f32::sum_f32(backend, data)
    }
}

impl Sum<f64> for f64 {
    unsafe fn compute(
        backend: &Avx2Backend,
        data: &[f64],
    ) -> f64 {
        f64::sum_f64(backend, data)
    }
}

// Generic fallback for other types
macro_rules! impl_sum_fallback {
    ($type:ty) => {
        impl Sum<$type> for $type {
            unsafe fn compute(
                _backend: &Avx2Backend,
                data: &[$type],
            ) -> <$type as Numeric>::Aggregate {
                let result = scalar::sum_scalar(data);
                <$type as Numeric>::Aggregate::from(result)
            }
        }
    };
}

impl_sum_fallback!(i32);
impl_sum_fallback!(i64);
impl_sum_fallback!(u32);
impl_sum_fallback!(u64);