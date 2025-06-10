//! Harrell-Davis weight computation

use robust_core::{SparseWeights, Numeric};
use statrs::distribution::{Beta, ContinuousCDF};
use num_traits::One;

/// Threshold for considering a weight as zero
const WEIGHT_THRESHOLD: f64 = 1e-12;

/// Compute Harrell-Davis weights
///
/// Returns sparse representation of weights for the quantile estimator.
pub fn compute_hd_weights<T: Numeric>(n: usize, p: f64) -> SparseWeights<T> 
where
    T::Float: num_traits::Float,
{
    assert!(n > 0, "n must be positive");
    assert!((0.0..=1.0).contains(&p), "p must be in [0, 1]");

    // Special cases
    if n == 1 {
        return SparseWeights::new(vec![0], vec![<T as One>::one()], 1);
    }

    if p == 0.0 {
        return SparseWeights::new(vec![0], vec![<T as One>::one()], n);
    }

    if p == 1.0 {
        return SparseWeights::new(vec![n - 1], vec![<T as One>::one()], n);
    }

    // Compute beta distribution parameters
    let n_f = n as f64;
    let alpha = (n_f + 1.0) * p;
    let beta_param = (n_f + 1.0) * (1.0 - p);

    let beta_dist = Beta::new(alpha, beta_param).unwrap();

    // Compute weights
    let mut indices = Vec::new();
    let mut weights = Vec::new();

    let mut beta_cdf_right = 0.0;
    let mut cumulative_prob = 0.0;

    for j in 0..n {
        let beta_cdf_left = beta_cdf_right;

        // For unweighted sample, each element contributes 1/n
        cumulative_prob += 1.0 / n_f;
        beta_cdf_right = beta_dist.cdf(cumulative_prob);

        let weight = beta_cdf_right - beta_cdf_left;

        // Only store weights above threshold
        if weight > WEIGHT_THRESHOLD {
            indices.push(j);
            weights.push(T::from_f64(weight));
        }
    }

    // Normalize weights to ensure they sum to exactly 1.0
    let sum: f64 = weights.iter().map(|w| w.to_f64()).sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w = T::from_f64(w.to_f64() / sum);
        }
    }

    SparseWeights::new(indices, weights, n)
}

/// Compute Harrell-Davis weights directly for a column range
/// This avoids computing all weights and then filtering
pub fn compute_hd_weights_range<T: Numeric>(
    n: usize,
    p: f64,
    col_start: usize,
    col_end: usize,
) -> SparseWeights<T>
where
    T::Float: num_traits::Float,
{
    use statrs::distribution::{Beta, ContinuousCDF};
    
    if n == 0 || col_start >= n || col_start >= col_end {
        return SparseWeights {
            indices: vec![],
            weights: vec![],
            n,
        };
    }
    
    // Handle edge cases
    if p == 0.0 {
        if col_start == 0 && col_end > 0 {
            return SparseWeights {
                indices: vec![0],
                weights: vec![<T as One>::one()],
                n,
            };
        } else {
            return SparseWeights {
                indices: vec![],
                weights: vec![],
                n,
            };
        }
    }
    
    if p == 1.0 {
        if col_start < n && col_end > n - 1 {
            return SparseWeights {
                indices: vec![n - 1],
                weights: vec![<T as One>::one()],
                n,
            };
        } else {
            return SparseWeights {
                indices: vec![],
                weights: vec![],
                n,
            };
        }
    }

    // Beta distribution parameters
    let n_f = n as f64;
    let alpha = (n_f + 1.0) * p;
    let beta_param = (n_f + 1.0) * (1.0 - p);
    let beta_dist = Beta::new(alpha, beta_param).unwrap();
    
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    
    // Start from the CDF at col_start instead of computing from 0
    let start_cumulative = if col_start == 0 { 
        0.0 
    } else { 
        (col_start as f64) / n_f 
    };
    
    let mut beta_cdf_left = beta_dist.cdf(start_cumulative);
    
    // Only compute weights for the range we need
    for j in col_start..col_end.min(n) {
        let j_plus_1_cumulative = (j + 1) as f64 / n_f;
        let beta_cdf_right = beta_dist.cdf(j_plus_1_cumulative);
        
        let weight = beta_cdf_right - beta_cdf_left;
        
        if weight > WEIGHT_THRESHOLD {
            indices.push(j);
            weights.push(T::from_f64(weight));
        }
        
        beta_cdf_left = beta_cdf_right;
    }

    SparseWeights {
        indices,
        weights,
        n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hd_weights_basic() {
        let weights: SparseWeights<f64> = compute_hd_weights(5, 0.5);
        use robust_core::primitives::ScalarBackend;
        let primitives = ScalarBackend::new();
        assert!(weights.validate_sum(&primitives, 1.0, 1e-10));
        assert_eq!(weights.n, 5);

        // Median should have symmetric weights
        let sum: f64 = weights.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hd_weights_extremes() {
        // p = 0 should put all weight on first element
        let weights: SparseWeights<f64> = compute_hd_weights(10, 0.0);
        assert_eq!(weights.nnz(), 1);
        assert_eq!(weights.indices[0], 0);
        assert!((weights.weights[0] - 1.0).abs() < 1e-10);

        // p = 1 should put all weight on last element
        let weights: SparseWeights<f64> = compute_hd_weights(10, 1.0);
        assert_eq!(weights.nnz(), 1);
        assert_eq!(weights.indices[0], 9);
        assert!((weights.weights[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weight_sparsity() {
        // For large n, weights should be sparse
        let weights: SparseWeights<f64> = compute_hd_weights(10000, 0.5);
        assert!(weights.sparsity() > 0.5);
    }
}
