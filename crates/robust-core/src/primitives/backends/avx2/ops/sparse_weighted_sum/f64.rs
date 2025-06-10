//! AVX2 sparse weighted sum implementation for f64

use crate::primitives::backends::avx2::Avx2Backend;

/// AVX2 implementation of sparse weighted sum for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn sparse_weighted_sum_f64(
    _backend: &Avx2Backend,
    data: &[f64],
    indices: &[usize],
    weights: &[f64],
) -> f64 {
    sparse_weighted_sum_avx2(data, indices, weights)
}

/// Highly optimized AVX2 implementation with hand-rolled assembly
/// This implementation uses inline assembly for maximum performance
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sparse_weighted_sum_avx2(
    data: &[f64],
    indices: &[usize],
    weights: &[f64],
) -> f64 {
    use std::arch::asm;
    let n = indices.len();

    // Safety check
    if n == 0 {
        return 0.0;
    }

    let data_ptr = data.as_ptr();
    let indices_ptr = indices.as_ptr();
    let weights_ptr = weights.as_ptr();

    let mut result: f64;
    asm!(
        // Initialize sum accumulator
        "vxorpd ymm0, ymm0, ymm0",

        // Calculate how many groups of 4 we can process
        "mov {n_groups}, {n}",
        "shr {n_groups}, 2",  // n_groups = n / 4

        // Skip SIMD if we have less than 1 full group
        "test {n_groups}, {n_groups}",
        "jz 15f",  // Jump to scalar_only

        // Main loop counter
        "xor {i}, {i}",

        // SIMD loop - process 4 elements at a time
        "3:",
            // Calculate actual array offset
            "mov {offset}, {i}",
            "shl {offset}, 2",  // offset = i * 4

            // Load 4 indices
            "lea {temp}, [{indices} + {offset}*8]",
            "mov {idx0}, [{temp}]",
            "mov {idx1}, [{temp} + 8]",
            "mov {idx2}, [{temp} + 16]",
            "mov {idx3}, [{temp} + 24]",

            // Gather data values
            "vmovsd xmm1, [{data} + {idx0}*8]",
            "vmovsd xmm2, [{data} + {idx1}*8]",
            "vmovsd xmm3, [{data} + {idx2}*8]",
            "vmovsd xmm4, [{data} + {idx3}*8]",

            "vunpcklpd xmm1, xmm1, xmm2",
            "vunpcklpd xmm3, xmm3, xmm4",
            "vinsertf128 ymm1, ymm1, xmm3, 1",

            // Load weights
            "lea {temp}, [{weights} + {offset}*8]",
            "vmovupd ymm2, [{temp}]",

            // Multiply and accumulate
            "vfmadd231pd ymm0, ymm1, ymm2",

            "inc {i}",
            "cmp {i}, {n_groups}",
            "jb 3b",

        // Reduce ymm0 to scalar
        "vextractf128 xmm1, ymm0, 1",
        "vaddpd xmm0, xmm0, xmm1",
        "vhaddpd xmm0, xmm0, xmm0",

        // Calculate remaining elements
        "mov {remainder}, {n}",
        "and {remainder}, 3",  // remainder = n % 4
        "test {remainder}, {remainder}",
        "jz 50f",  // No remainder, we're done

        // Process remainder
        "mov {offset}, {n_groups}",
        "shl {offset}, 2",  // offset = n_groups * 4

        "5:",
            "lea {temp}, [{indices} + {offset}*8]",
            "mov {idx0}, [{temp}]",
            "vmovsd xmm1, [{data} + {idx0}*8]",
            "lea {temp}, [{weights} + {offset}*8]",
            "vmulsd xmm1, xmm1, [{temp}]",
            "vaddsd xmm0, xmm0, xmm1",

            "inc {offset}",
            "dec {remainder}",
            "jnz 5b",
            "jmp 50f",

        // Scalar only path (n < 4)
        "15:",
            "vxorpd xmm0, xmm0, xmm0",
            "xor {offset}, {offset}",
            "mov {remainder}, {n}",

        "16:",
            "mov {idx0}, [{indices} + {offset}*8]",
            "vmovsd xmm1, [{data} + {idx0}*8]",
            "vmulsd xmm1, xmm1, [{weights} + {offset}*8]",
            "vaddsd xmm0, xmm0, xmm1",

            "inc {offset}",
            "dec {remainder}",
            "jnz 16b",

        // Done
        "50:",
            "vzeroupper",

        // Input/output constraints
        n = in(reg) n,
        n_groups = out(reg) _,
        i = out(reg) _,
        offset = out(reg) _,
        remainder = out(reg) _,
        temp = out(reg) _,
        idx0 = out(reg) _,
        idx1 = out(reg) _,
        idx2 = out(reg) _,
        idx3 = out(reg) _,
        data = in(reg) data_ptr,
        indices = in(reg) indices_ptr,
        weights = in(reg) weights_ptr,
        out("xmm0") result,
        out("xmm1") _, out("xmm2") _, out("xmm3") _, out("xmm4") _,

        options(nostack)
    );

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Helper to check if AVX2 is available
    fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    // Helper for computing reference result
    fn sparse_weighted_sum_scalar(data: &[f64], indices: &[usize], weights: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..indices.len() {
            sum += data[indices[i]] * weights[i];
        }
        sum
    }

    #[test]
    fn test_sparse_weighted_sum_empty() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let indices: Vec<usize> = vec![];
        let weights: Vec<f64> = vec![];
        
        unsafe {
            let result = sparse_weighted_sum_f64(&backend, &data, &indices, &weights);
            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn test_sparse_weighted_sum_single() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let indices = vec![2];
        let weights = vec![0.5];
        
        unsafe {
            let result = sparse_weighted_sum_f64(&backend, &data, &indices, &weights);
            assert_eq!(result, 1.5); // 3.0 * 0.5
        }
    }

    #[test]
    fn test_sparse_weighted_sum_less_than_4() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Test with 2 elements (scalar path)
        let indices = vec![1, 3];
        let weights = vec![0.5, 0.3];
        
        unsafe {
            let result = sparse_weighted_sum_f64(&backend, &data, &indices, &weights);
            let expected = 2.0 * 0.5 + 4.0 * 0.3;
            assert!((result - expected).abs() < 1e-10);
        }
        
        // Test with 3 elements (scalar path)
        let indices = vec![0, 2, 4];
        let weights = vec![0.2, 0.3, 0.5];
        
        unsafe {
            let result = sparse_weighted_sum_f64(&backend, &data, &indices, &weights);
            let expected = 1.0 * 0.2 + 3.0 * 0.3 + 5.0 * 0.5;
            assert!((result - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparse_weighted_sum_exact_4() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = vec![0, 2, 4, 5];
        let weights = vec![0.1, 0.2, 0.3, 0.4];
        
        unsafe {
            let result = sparse_weighted_sum_f64(&backend, &data, &indices, &weights);
            let expected = 1.0 * 0.1 + 3.0 * 0.2 + 5.0 * 0.3 + 6.0 * 0.4;
            assert!((result - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparse_weighted_sum_with_remainder() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        
        // Test 5 elements (1 SIMD group + 1 remainder)
        let indices = vec![1, 3, 5, 7, 9];
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        
        unsafe {
            let result = sparse_weighted_sum_f64(&backend, &data, &indices, &weights);
            let expected = sparse_weighted_sum_scalar(&data, &indices, &weights);
            assert!((result - expected).abs() < 1e-10);
        }
        
        // Test 7 elements (1 SIMD group + 3 remainder)
        let indices = vec![0, 2, 3, 4, 6, 8, 9];
        let weights = vec![0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4];
        
        unsafe {
            let result = sparse_weighted_sum_f64(&backend, &data, &indices, &weights);
            let expected = sparse_weighted_sum_scalar(&data, &indices, &weights);
            assert!((result - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparse_weighted_sum_multiple_groups() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let data: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        
        // Test 8 elements (2 SIMD groups)
        let indices = vec![1, 3, 5, 7, 11, 13, 15, 17];
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        
        unsafe {
            let result = sparse_weighted_sum_f64(&backend, &data, &indices, &weights);
            let expected = sparse_weighted_sum_scalar(&data, &indices, &weights);
            assert!((result - expected).abs() < 1e-10);
        }
        
        // Test 10 elements (2 SIMD groups + 2 remainder)
        let indices = vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18];
        let weights = vec![0.05; 10];
        
        unsafe {
            let result = sparse_weighted_sum_f64(&backend, &data, &indices, &weights);
            let expected = sparse_weighted_sum_scalar(&data, &indices, &weights);
            assert!((result - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparse_weighted_sum_edge_cases() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        
        // Test with zeros
        let data = vec![0.0; 10];
        let indices = vec![1, 3, 5, 7];
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        
        unsafe {
            let result = sparse_weighted_sum_f64(&backend, &data, &indices, &weights);
            assert_eq!(result, 0.0);
        }
        
        // Test with negative values
        let data = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
        let indices = vec![0, 2, 4];
        let weights = vec![1.0, -1.0, 2.0];
        
        unsafe {
            let result = sparse_weighted_sum_f64(&backend, &data, &indices, &weights);
            let expected = -1.0 * 1.0 + -3.0 * -1.0 + -5.0 * 2.0;
            assert!((result - expected).abs() < 1e-10);
        }
        
        // Test with very large/small values
        let data = vec![1e100, 1e-100, 1e50, 1e-50];
        let indices = vec![0, 1, 2, 3];
        let weights = vec![1e-100, 1e100, 1e-50, 1e50];
        
        unsafe {
            let result = sparse_weighted_sum_f64(&backend, &data, &indices, &weights);
            let expected = 1.0 + 1.0 + 1.0 + 1.0; // All products equal 1
            assert!((result - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparse_weighted_sum_indices_out_of_order() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        
        // Indices not in ascending order
        let indices = vec![7, 2, 9, 0, 5, 3];
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        
        unsafe {
            let result = sparse_weighted_sum_f64(&backend, &data, &indices, &weights);
            let expected = sparse_weighted_sum_scalar(&data, &indices, &weights);
            assert!((result - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparse_weighted_sum_repeated_indices() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        
        // Same index accessed multiple times
        let indices = vec![1, 1, 2, 2];
        let weights = vec![0.5, 0.5, 0.3, 0.7];
        
        unsafe {
            let result = sparse_weighted_sum_f64(&backend, &data, &indices, &weights);
            // 2.0 * 0.5 + 2.0 * 0.5 + 3.0 * 0.3 + 3.0 * 0.7
            let expected = 1.0 + 1.0 + 0.9 + 2.1;
            assert!((result - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparse_weighted_sum_comprehensive() {
        if !has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2");
            return;
        }

        let backend = Avx2Backend;
        
        // Test various sizes to ensure all code paths work correctly
        for n in 0..20 {
            let data: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
            let indices: Vec<usize> = (0..n).map(|i| i % data.len()).collect();
            let weights: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.1).collect();
            
            unsafe {
                let avx2_result = sparse_weighted_sum_f64(&backend, &data, &indices, &weights);
                let scalar_result = sparse_weighted_sum_scalar(&data, &indices, &weights);
                
                assert!(
                    (avx2_result - scalar_result).abs() < 1e-10,
                    "Mismatch for n={}: AVX2={}, scalar={}",
                    n, avx2_result, scalar_result
                );
            }
        }
    }
}