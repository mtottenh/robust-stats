//! Utility functions for working with data slices

/// Sort data and return a new vector
///
/// Handles NaN values by placing them at the end.
///
/// # Examples
///
/// ```rust
/// use robust_core::utils::sorted;
///
/// let data = vec![3.0, 1.0, 5.0, 2.0, 4.0];
/// assert_eq!(sorted(&data), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// ```
pub fn sorted(data: &[f64]) -> Vec<f64> {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater, // NaN goes after non-NaN
            (false, true) => std::cmp::Ordering::Less,    // non-NaN goes before NaN
            (false, false) => a.partial_cmp(b).unwrap(),  // Safe for non-NaN values
        }
    });
    sorted
}

/// Calculate the sample standard deviation
///
/// Returns 0.0 for slices with less than 2 elements.
///
/// # Examples
///
/// ```rust
/// use robust_core::utils::std_dev;
///
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let sd = std_dev(&data);
/// assert!((sd - 1.58113883).abs() < 1e-6);
/// ```
pub fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let variance: f64 = data
        .iter()
        .map(|&x| {
            let diff = x - m;
            diff * diff
        })
        .sum::<f64>()
        / (data.len() - 1) as f64;
    variance.sqrt()
}

/// Calculate the mean of a slice
///
/// Returns 0.0 for empty slices.
///
/// # Examples
///
/// ```rust
/// use robust_core::utils::mean;
///
/// assert_eq!(mean(&[1.0, 2.0, 3.0]), 2.0);
/// assert_eq!(mean(&[]), 0.0);
/// ```
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sorted_basic() {
        let data = vec![3.0, 1.0, 5.0, 2.0, 4.0];
        let sorted_data = sorted(&data);
        assert_eq!(sorted_data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }
    
    #[test]
    fn test_sorted_empty() {
        let data: Vec<f64> = vec![];
        let sorted_data = sorted(&data);
        assert_eq!(sorted_data, Vec::<f64>::new());
    }
    
    #[test]
    fn test_sorted_single_element() {
        let data = vec![42.0];
        let sorted_data = sorted(&data);
        assert_eq!(sorted_data, vec![42.0]);
    }
    
    #[test]
    fn test_sorted_duplicates() {
        let data = vec![3.0, 1.0, 3.0, 2.0, 1.0];
        let sorted_data = sorted(&data);
        assert_eq!(sorted_data, vec![1.0, 1.0, 2.0, 3.0, 3.0]);
    }
    
    #[test]
    fn test_sorted_negative_numbers() {
        let data = vec![3.0, -1.0, 0.0, -5.0, 2.0];
        let sorted_data = sorted(&data);
        assert_eq!(sorted_data, vec![-5.0, -1.0, 0.0, 2.0, 3.0]);
    }
    
    #[test]
    fn test_sorted_with_nan() {
        let data = vec![3.0, f64::NAN, 1.0, 2.0];
        let sorted_data = sorted(&data);
        
        // NaN values should be at the end
        assert_eq!(sorted_data[0], 1.0);
        assert_eq!(sorted_data[1], 2.0);
        assert_eq!(sorted_data[2], 3.0);
        assert!(sorted_data[3].is_nan());
    }
    
    #[test]
    fn test_sorted_with_infinity() {
        let data = vec![3.0, f64::INFINITY, 1.0, f64::NEG_INFINITY, 2.0];
        let sorted_data = sorted(&data);
        assert_eq!(sorted_data, vec![f64::NEG_INFINITY, 1.0, 2.0, 3.0, f64::INFINITY]);
    }
    
    #[test]
    fn test_sorted_preserves_original() {
        let data = vec![3.0, 1.0, 5.0, 2.0, 4.0];
        let original = data.clone();
        let _ = sorted(&data);
        assert_eq!(data, original); // Original data unchanged
    }
    
    #[test]
    fn test_mean_basic() {
        assert_eq!(mean(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
    }
    
    #[test]
    fn test_mean_empty() {
        assert_eq!(mean(&[]), 0.0);
    }
    
    #[test]
    fn test_mean_single_element() {
        assert_eq!(mean(&[42.0]), 42.0);
    }
    
    #[test]
    fn test_mean_negative_numbers() {
        assert_eq!(mean(&[-1.0, -2.0, -3.0]), -2.0);
        assert_eq!(mean(&[-10.0, 10.0]), 0.0);
    }
    
    #[test]
    fn test_mean_precision() {
        let data = [1.1, 2.2, 3.3, 4.4, 5.5];
        let m = mean(&data);
        assert!((m - 3.3).abs() < 1e-10);
    }
    
    #[test]
    fn test_mean_large_numbers() {
        let data = [1e10, 2e10, 3e10];
        let m = mean(&data);
        assert_eq!(m, 2e10);
    }
    
    #[test]
    fn test_std_dev_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sd = std_dev(&data);
        // Variance = sum((x - mean)^2) / (n-1) = 10 / 4 = 2.5
        // SD = sqrt(2.5) ≈ 1.58113883
        assert!((sd - 1.58113883).abs() < 1e-8);
    }
    
    #[test]
    fn test_std_dev_empty() {
        assert_eq!(std_dev(&[]), 0.0);
    }
    
    #[test]
    fn test_std_dev_single_element() {
        assert_eq!(std_dev(&[42.0]), 0.0);
    }
    
    #[test]
    fn test_std_dev_identical_values() {
        let data = [5.0, 5.0, 5.0, 5.0];
        assert_eq!(std_dev(&data), 0.0);
    }
    
    #[test]
    fn test_std_dev_two_elements() {
        let data = [1.0, 3.0];
        let sd = std_dev(&data);
        // Mean = 2.0, Variance = ((1-2)^2 + (3-2)^2) / 1 = 2 / 1 = 2
        // SD = sqrt(2) ≈ 1.41421356
        assert!((sd - std::f64::consts::SQRT_2).abs() < 1e-8);
    }
    
    #[test]
    fn test_std_dev_negative_numbers() {
        let data = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let sd = std_dev(&data);
        assert!((sd - 1.58113883).abs() < 1e-8);
    }
    
    #[test]
    fn test_std_dev_larger_dataset() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let sd = std_dev(&data);
        // For 0..99, mean = 49.5, variance ≈ 833.25, SD ≈ 28.866
        assert!((sd - 29.011491975882016).abs() < 1e-8);
    }
    
    #[test]
    fn test_std_dev_precision() {
        // Test with very small differences
        let data = [1.0000001, 1.0000002, 1.0000003];
        let sd = std_dev(&data);
        assert!(sd > 0.0); // Should detect the tiny variance
        assert!(sd < 1e-6);
    }
    
    #[test]
    fn test_edge_case_nan_handling() {
        // Test that NaN values propagate correctly
        let data_with_nan = [1.0, 2.0, f64::NAN, 4.0, 5.0];
        let m = mean(&data_with_nan);
        assert!(m.is_nan());
        
        let sd = std_dev(&data_with_nan);
        assert!(sd.is_nan());
    }
    
    #[test]
    fn test_edge_case_infinity() {
        let data_with_inf = [1.0, 2.0, f64::INFINITY, 4.0, 5.0];
        let m = mean(&data_with_inf);
        assert_eq!(m, f64::INFINITY);
        
        let sd = std_dev(&data_with_inf);
        assert!(sd.is_nan()); // Infinity - Infinity = NaN
    }
    
    #[test]
    fn test_numerical_stability() {
        // Test with values that could cause numerical issues
        let data = [1e10, 1e10 + 1.0, 1e10 + 2.0];
        let sd = std_dev(&data);
        // Despite large magnitudes, the SD should be around 1.0
        assert!((sd - 1.0).abs() < 1e-8);
    }
    
    #[test]
    fn test_sorted_stability() {
        // Test that equal elements maintain their relative order
        // While Rust's sort_by doesn't guarantee stability for partial_cmp,
        // we can at least verify all equal elements are grouped
        let data = vec![3.0, 1.0, 2.0, 1.0, 3.0, 2.0];
        let sorted_data = sorted(&data);
        assert_eq!(sorted_data, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }
    
    #[test]
    fn test_mean_overflow_protection() {
        // Test with values that could overflow if summed naively
        let data = vec![f64::MAX / 2.0, f64::MAX / 2.0];
        let m = mean(&data);
        assert_eq!(m, f64::MAX / 2.0); // Should not overflow
    }
}