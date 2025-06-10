//! Naive reference implementations of Harrell-Davis estimators
//! 
//! These implementations are intentionally simple and unoptimized.
//! They serve as a reference for verifying the correctness of optimized implementations.
//! 
//! DO NOT USE IN PRODUCTION - these are for testing and debugging only!

use statrs::distribution::{Beta, ContinuousCDF};

/// Naive Harrell-Davis quantile estimator
/// 
/// This implementation:
/// - Computes all weights from scratch every time
/// - Does not cache anything
/// - Does not use sparse representations
/// - Is intentionally simple for verification purposes
/// 
/// Based on the reference C# implementation from Perfolizer
pub struct NaiveHarrellDavis;

impl NaiveHarrellDavis {
    /// Compute a single quantile using the naive approach
    /// 
    /// # Arguments
    /// * `data` - The data sample (will be sorted internally)
    /// * `p` - The probability (0.0 to 1.0)
    pub fn quantile(data: &[f64], p: f64) -> f64 {
        assert!(!data.is_empty(), "Cannot compute quantile of empty data");
        assert!((0.0..=1.0).contains(&p), "Probability must be in [0, 1]");
        
        // Sort the data
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted.len();
        
        // Special cases
        if n == 1 {
            return sorted[0];
        }
        
        if p == 0.0 {
            return sorted[0];
        }
        
        if p == 1.0 {
            return sorted[n - 1];
        }
        
        // Compute beta distribution parameters
        // Following the reference implementation: a = (n + 1) * p
        let n_f = n as f64;
        let alpha = (n_f + 1.0) * p;
        let beta_param = (n_f + 1.0) * (1.0 - p);
        
        let beta_dist = Beta::new(alpha, beta_param).unwrap();
        
        // Compute weights following the reference implementation
        let mut c1 = 0.0;
        let mut beta_cdf_right = 0.0;
        let mut current_probability = 0.0;
        
        for j in 0..n {
            let beta_cdf_left = beta_cdf_right;
            
            // For unweighted sample, each element contributes 1/n to cumulative probability
            current_probability += 1.0 / n_f;
            
            // Get CDF value at this cumulative probability
            beta_cdf_right = beta_dist.cdf(current_probability);
            
            // Weight is the difference in CDF values
            let w = beta_cdf_right - beta_cdf_left;
            
            // Accumulate weighted value
            c1 += w * sorted[j];
        }
        
        c1
    }
    
    /// Compute quantile with moments (for Maritz-Jarrett CI)
    pub fn quantile_with_moments(data: &[f64], p: f64) -> (f64, f64) {
        assert!(!data.is_empty(), "Cannot compute quantile of empty data");
        assert!((0.0..=1.0).contains(&p), "Probability must be in [0, 1]");
        
        // Sort the data
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted.len();
        
        // Special cases
        if n == 1 {
            let val = sorted[0];
            return (val, val * val);
        }
        
        // Compute beta distribution parameters
        let n_f = n as f64;
        let alpha = (n_f + 1.0) * p;
        let beta_param = (n_f + 1.0) * (1.0 - p);
        
        let beta_dist = Beta::new(alpha, beta_param).unwrap();
        
        // Compute both moments
        let mut c1 = 0.0;
        let mut c2 = 0.0;
        let mut beta_cdf_right = 0.0;
        let mut current_probability = 0.0;
        
        for j in 0..n {
            let beta_cdf_left = beta_cdf_right;
            current_probability += 1.0 / n_f;
            beta_cdf_right = beta_dist.cdf(current_probability);
            let w = beta_cdf_right - beta_cdf_left;
            
            c1 += w * sorted[j];
            c2 += w * sorted[j] * sorted[j];
        }
        
        (c1, c2)
    }
}

/// Naive Trimmed Harrell-Davis quantile estimator
/// 
/// This implementation:
/// - Computes HDI from scratch every time
/// - Computes all weights from scratch
/// - Does not cache anything
/// - Is intentionally simple for verification purposes
pub struct NaiveTrimmedHarrellDavis;

impl NaiveTrimmedHarrellDavis {
    /// Compute a single quantile using trimmed HD with specified width
    /// 
    /// Following the C# reference implementation from Perfolizer
    pub fn quantile(data: &[f64], p: f64, width: f64) -> f64 {
        assert!(!data.is_empty(), "Cannot compute quantile of empty data");
        assert!((0.0..=1.0).contains(&p), "Probability must be in [0, 1]");
        assert!((0.0..=1.0).contains(&width), "Width must be in [0, 1]");
        
        // If width is essentially 1, use regular HD
        if width >= 1.0 - 1e-9 {
            return NaiveHarrellDavis::quantile(data, p);
        }
        
        // Sort the data
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted.len();
        
        // Special cases
        if n == 1 {
            return sorted[0];
        }
        
        // Compute beta distribution parameters
        let n_f = n as f64;
        let alpha = (n_f + 1.0) * p;
        let beta_param = (n_f + 1.0) * (1.0 - p);
        
        // Find HDI bounds
        let (hdi_lower, hdi_upper) = compute_beta_hdi(alpha, beta_param, width);
        
        let beta_dist = Beta::new(alpha, beta_param).unwrap();
        let hdi_cdf_l = beta_dist.cdf(hdi_lower);
        let hdi_cdf_r = beta_dist.cdf(hdi_upper);
        
        // Define the normalized CDF function within HDI (matching C#)
        let cdf = |x: f64| -> f64 {
            if x <= hdi_lower {
                0.0
            } else if x > hdi_upper {
                1.0
            } else {
                (beta_dist.cdf(x) - hdi_cdf_l) / (hdi_cdf_r - hdi_cdf_l)
            }
        };
        
        // Compute weights - optimized for unweighted samples
        let mut c1 = 0.0;
        let mut beta_cdf_right = 0.0;
        
        // Following C# optimization: only process indices within HDI range
        let j_l = (hdi_lower * n_f).floor() as usize;
        let j_r = ((hdi_upper * n_f).ceil() as usize).saturating_sub(1).min(n - 1);
        
        for j in j_l..=j_r {
            let beta_cdf_left = beta_cdf_right;
            let current_probability = (j + 1) as f64 / n_f;
            
            let cdf_value = cdf(current_probability);
            beta_cdf_right = cdf_value;
            let w = beta_cdf_right - beta_cdf_left;
            c1 += w * sorted[j];
        }
        
        c1
    }
    
    /// Compute with sqrt(n) width
    pub fn quantile_sqrt(data: &[f64], p: f64) -> f64 {
        let width = (1.0 / (data.len() as f64).sqrt()).min(1.0);
        Self::quantile(data, p, width)
    }
    
    /// Compute with linear width
    pub fn quantile_linear(data: &[f64], p: f64) -> f64 {
        let width = (1.0 / data.len() as f64).min(1.0);
        Self::quantile(data, p, width)
    }
}

/// Compute Highest Density Interval (HDI) for a beta distribution
/// 
/// Based on the C# reference implementation from Perfolizer
fn compute_beta_hdi(alpha: f64, beta_param: f64, width: f64) -> (f64, f64) {
    const EPS: f64 = 1e-9;
    
    if width >= 1.0 - EPS {
        return (0.0, 1.0);
    }
    
    // Special cases matching C# implementation
    if alpha < 1.0 + EPS && beta_param < 1.0 + EPS {
        return (0.5 - width / 2.0, 0.5 + width / 2.0);
    }
    
    if alpha < 1.0 + EPS && beta_param > 1.0 {
        return (0.0, width);
    }
    
    if alpha > 1.0 && beta_param < 1.0 + EPS {
        return (1.0 - width, 1.0);
    }
    
    if (alpha - beta_param).abs() < EPS {
        return (0.5 - width / 2.0, 0.5 + width / 2.0);
    }
    
    // For other cases, use binary search
    let mode = (alpha - 1.0) / (alpha + beta_param - 2.0);
    let left_bound = f64::max(0.0, mode - width);
    let right_bound = f64::min(mode, 1.0 - width);
    
    let l = binary_search(
        |x| beta_denormalized_log_pdf(alpha, beta_param, x) - 
            beta_denormalized_log_pdf(alpha, beta_param, x + width),
        left_bound,
        right_bound
    );
    
    (l, l + width)
}

/// Binary search helper function matching C# implementation
fn binary_search<F>(f: F, mut left: f64, mut right: f64) -> f64 
where
    F: Fn(f64) -> f64
{
    let mut fl = f(left);
    let fr = f(right);
    
    // Check if function has same sign at both ends
    if (fl < 0.0 && fr < 0.0) || (fl > 0.0 && fr > 0.0) {
        // Return midpoint as fallback
        return (left + right) / 2.0;
    }
    
    while right - left > 1e-9 {
        let m = (left + right) / 2.0;
        let fm = f(m);
        
        if (fl < 0.0 && fm < 0.0) || (fl > 0.0 && fm > 0.0) {
            fl = fm;
            left = m;
        } else {
            right = m;
        }
    }
    
    (left + right) / 2.0
}

/// Denormalized log Beta PDF matching C# implementation
fn beta_denormalized_log_pdf(alpha: f64, beta_param: f64, x: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return f64::NEG_INFINITY;
    }
    
    (alpha - 1.0) * x.ln() + (beta_param - 1.0) * (1.0 - x).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_naive_hd_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let q1 = NaiveHarrellDavis::quantile(&data, 0.25);
        let median = NaiveHarrellDavis::quantile(&data, 0.5);
        let q3 = NaiveHarrellDavis::quantile(&data, 0.75);
        
        assert!(q1 > 1.0 && q1 < 3.0);
        assert_relative_eq!(median, 3.0, epsilon = 0.1);
        assert!(q3 > 3.0 && q3 < 5.0);
    }
    
    #[test]
    fn test_naive_hd_edge_cases() {
        let single = vec![42.0];
        assert_eq!(NaiveHarrellDavis::quantile(&single, 0.5), 42.0);
        
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(NaiveHarrellDavis::quantile(&data, 0.0), 1.0);
        assert_eq!(NaiveHarrellDavis::quantile(&data, 1.0), 3.0);
    }
    
    #[test]
    fn test_naive_trimmed_hd_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        
        let regular = NaiveTrimmedHarrellDavis::quantile(&data, 0.5, 1.0);
        let trimmed = NaiveTrimmedHarrellDavis::quantile(&data, 0.5, 0.5);
        
        // Regular should be close to 5.5
        assert_relative_eq!(regular, 5.5, epsilon = 0.1);
        
        // Trimmed should also be close but potentially different
        assert!(trimmed > 4.0 && trimmed < 7.0);
    }
    
    #[test]
    fn test_moments() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (c1, c2) = NaiveHarrellDavis::quantile_with_moments(&data, 0.5);
        
        // First moment should be the quantile
        assert_relative_eq!(c1, 3.0, epsilon = 0.1);
        
        // Second moment should be positive
        assert!(c2 > 0.0);
        assert!(c2 >= c1 * c1); // E[X²] ≥ E[X]²
    }
}