//! AVX2 dot product implementation for f64

use crate::primitives::backends::avx2::Avx2Backend;
use std::arch::x86_64::*;

/// AVX2 implementation of dot product for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_product_f64(
    _backend: &Avx2Backend,
    a: &[f64],
    b: &[f64],
) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum_vec = _mm256_setzero_pd();

    // Main loop - process 4 elements at a time
    for i in 0..chunks {
        let offset = i * 4;
        let a_ptr = a.as_ptr().add(offset);
        let b_ptr = b.as_ptr().add(offset);

        let a_vec = _mm256_loadu_pd(a_ptr);
        let b_vec = _mm256_loadu_pd(b_ptr);

        sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
    }

    // Sum the vector elements
    let sum_array = std::mem::transmute::<__m256d, [f64; 4]>(sum_vec);
    let mut sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

    // Handle remainder
    let remainder_start = chunks * 4;
    for i in 0..remainder {
        sum += a[remainder_start + i] * b[remainder_start + i];
    }

    sum
}