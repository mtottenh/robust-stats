//! Shared utilities for integration tests

pub use approx::assert_relative_eq;

pub const EPSILON: f64 = 1e-10;

/// Generate array lengths that test edge cases for SIMD operations
pub fn edge_case_lengths() -> Vec<usize> {
    vec![
        0,   // Empty
        1,   // Single element
        2,   // SSE width
        3,   // SSE width + 1
        4,   // AVX2 width
        5,   // AVX2 width + 1
        7,   // AVX2 width - 1 + remainder
        8,   // AVX512 width
        9,   // AVX512 width + 1
        15,  // Multiple of no SIMD width
        16,  // Power of 2
        17,  // Power of 2 + 1
        31,  // Prime
        32,  // Common cache line size / 8
        63,  // Almost cache line
        64,  // Cache line
        100, // Round number
        127, // Mersenne prime
        128, // Power of 2
    ]
}

/// Special floating-point values for edge case testing
pub fn special_values() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::MIN,
        f64::MAX,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::EPSILON,
        -f64::EPSILON,
        std::f64::consts::PI,
        std::f64::consts::E,
        1e-308, // Near underflow
        1e308,  // Near overflow
    ]
}

/// Assert two vectors are equal within tolerance
pub fn assert_vectors_equal(actual: &[f64], expected: &[f64], op: &str, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{} length mismatch for {}",
        op,
        context
    );

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_relative_eq!(
            a,
            e,
            epsilon = EPSILON
        );
    }
}

/// Generate test data with specific patterns
pub fn generate_test_data(len: usize) -> Vec<f64> {
    (0..len).map(|i| i as f64 + 0.1).collect()
}

/// Generate complementary test data for binary operations
pub fn generate_test_data_complement(len: usize) -> Vec<f64> {
    (0..len).map(|i| (len - i) as f64 + 0.2).collect()
}
