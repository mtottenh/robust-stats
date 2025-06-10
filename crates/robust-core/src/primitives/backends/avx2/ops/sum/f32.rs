//! AVX2 sum implementation for f32

use crate::primitives::backends::avx2::Avx2Backend;
use std::arch::x86_64::*;

/// AVX2 implementation of sum for f32
#[target_feature(enable = "avx2")]
pub unsafe fn sum_f32(
    _backend: &Avx2Backend,
    data: &[f32],
) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }

    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum_vec = _mm256_setzero_ps();

    // Main loop - process 8 elements at a time
    for i in 0..chunks {
        let offset = i * 8;
        let data_ptr = data.as_ptr().add(offset);
        let data_vec = _mm256_loadu_ps(data_ptr);
        sum_vec = _mm256_add_ps(sum_vec, data_vec);
    }

    // Sum the vector elements
    let sum_array = std::mem::transmute::<__m256, [f32; 8]>(sum_vec);
    let mut sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3]
                + sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in 0..remainder {
        sum += data[remainder_start + i];
    }

    sum as f64
}