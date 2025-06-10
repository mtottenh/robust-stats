//! Shared utilities for AVX2 implementations
//!
//! This module contains common operations used across different AVX2 implementations.

#![allow(dead_code)]

use std::arch::x86_64::*;

/// Horizontal sum of a __m256d (4 f64s)
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn horizontal_sum_pd(v: __m256d) -> f64 {
    // Extract high and low 128-bit lanes
    let high = _mm256_extractf128_pd(v, 1);
    let low = _mm256_castpd256_pd128(v);
    
    // Add high and low
    let sum128 = _mm_add_pd(high, low);
    
    // Horizontal add within 128-bit lane
    let sum = _mm_hadd_pd(sum128, sum128);
    
    // Extract result
    _mm_cvtsd_f64(sum)
}

/// Horizontal sum of a __m256 (8 f32s)
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn horizontal_sum_ps(v: __m256) -> f32 {
    // Extract high and low 128-bit lanes
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    
    // Add high and low
    let sum128 = _mm_add_ps(high, low);
    
    // Horizontal add within 128-bit lane (two steps)
    let shuf = _mm_shuffle_ps(sum128, sum128, 0b00_11_00_01);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf = _mm_shuffle_ps(sums, sums, 0b00_00_00_10);
    let result = _mm_add_ps(sums, shuf);
    
    // Extract result
    _mm_cvtss_f32(result)
}

/// Load 4 values from scattered indices
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn gather_4_f64(data: &[f64], indices: &[usize], offset: usize) -> __m256d {
    let idx0 = *indices.get_unchecked(offset);
    let idx1 = *indices.get_unchecked(offset + 1);
    let idx2 = *indices.get_unchecked(offset + 2);
    let idx3 = *indices.get_unchecked(offset + 3);
    
    let val0 = *data.get_unchecked(idx0);
    let val1 = *data.get_unchecked(idx1);
    let val2 = *data.get_unchecked(idx2);
    let val3 = *data.get_unchecked(idx3);
    
    _mm256_set_pd(val3, val2, val1, val0)
}

/// Load 8 values from scattered indices
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn gather_8_f32(data: &[f32], indices: &[usize], offset: usize) -> __m256 {
    let mut values = [0.0f32; 8];
    for (i, value) in values.iter_mut().enumerate() {
        let idx = *indices.get_unchecked(offset + i);
        *value = *data.get_unchecked(idx);
    }
    _mm256_loadu_ps(values.as_ptr())
}

/// Check if 4 indices are consecutive
#[inline]
pub fn are_consecutive_4(indices: &[u16], offset: usize) -> bool {
    if offset + 3 >= indices.len() {
        return false;
    }
    
    let first = indices[offset] as usize;
    indices[offset + 1] as usize == first + 1 &&
    indices[offset + 2] as usize == first + 2 &&
    indices[offset + 3] as usize == first + 3
}

/// Check if 8 indices are consecutive
#[inline]
pub fn are_consecutive_8(indices: &[u16], offset: usize) -> bool {
    if offset + 7 >= indices.len() {
        return false;
    }
    
    let first = indices[offset] as usize;
    for i in 1..8 {
        if indices[offset + i] as usize != first + i {
            return false;
        }
    }
    true
}

/// Aligned allocation helper
#[repr(align(32))]
pub struct Aligned32<T> {
    pub data: T,
}

/// Create aligned storage for AVX2 operations
#[macro_export]
macro_rules! aligned_array {
    ($type:ty, $size:expr) => {
        $crate::primitives::backends::avx2::utils::Aligned32 {
            data: [0 as $type; $size]
        }
    };
}