//! Highest Density Interval (HDI) computation for trimmed Harrell-Davis

use super::compute_hd_weights;
use num_traits::One;
use robust_core::{Numeric, SparseWeights};
use statrs::distribution::{Beta, ContinuousCDF};
use statrs::function::beta::ln_beta;
/// Compute Highest Density Interval for a Beta distribution
///
/// Returns (lower, upper) bounds of the HDI with the specified width.
pub fn compute_beta_hdi(alpha: f64, beta_param: f64, width: f64) -> (f64, f64) {
    const EPS: f64 = 1e-9;

    if width >= 1.0 - EPS {
        return (0.0, 1.0);
    }

    // Special cases
    if alpha < 1.0 + EPS && beta_param < 1.0 + EPS {
        // U-shaped distribution - HDI at extremes
        return (0.5 - width / 2.0, 0.5 + width / 2.0);
    }

    if alpha < 1.0 + EPS && beta_param > 1.0 {
        // J-shaped (decreasing) - HDI at left
        return (0.0, width);
    }

    if alpha > 1.0 && beta_param < 1.0 + EPS {
        // J-shaped (increasing) - HDI at right
        return (1.0 - width, 1.0);
    }

    if (alpha - beta_param).abs() < EPS {
        // Symmetric - HDI centered at 0.5
        return (0.5 - width / 2.0, 0.5 + width / 2.0);
    }

    // General case: use optimization
    let mode = if alpha > 1.0 && beta_param > 1.0 {
        (alpha - 1.0) / (alpha + beta_param - 2.0)
    } else {
        0.5
    };

    let left_bound = (mode - width).max(0.0);
    let right_bound = mode.min(1.0 - width);

    let l = binary_search(
        |x| beta_log_pdf(x, alpha, beta_param) - beta_log_pdf(x + width, alpha, beta_param),
        left_bound,
        right_bound,
    );

    (l, l + width)
}

/// Compute trimmed Harrell-Davis weights
pub fn compute_trimmed_weights<T: Numeric>(n: usize, p: f64, width: f64) -> SparseWeights<T> {
    if width >= 1.0 - 1e-9 {
        return compute_hd_weights(n, p);
    }

    let n_f = n as f64;
    let alpha = (n_f + 1.0) * p;
    let beta_param = (n_f + 1.0) * (1.0 - p);
    if n == 1 {
        return SparseWeights::new(vec![0], vec![<T as One>::one()], 1);
    }

    if p == 0.0 {
        return SparseWeights::new(vec![0], vec![<T as One>::one()], n);
    }

    if p == 1.0 {
        return SparseWeights::new(vec![n - 1], vec![<T as One>::one()], n);
    }

    // Find HDI bounds
    let (hdi_lower, hdi_upper) = compute_beta_hdi(alpha, beta_param, width);

    let beta_dist = Beta::new(alpha, beta_param).unwrap();
    let hdi_cdf_l = beta_dist.cdf(hdi_lower);
    let hdi_cdf_r = beta_dist.cdf(hdi_upper);

    // Normalized CDF within HDI
    let cdf = |x: f64| -> f64 {
        if x <= hdi_lower {
            0.0
        } else if x >= hdi_upper {
            1.0
        } else {
            (beta_dist.cdf(x) - hdi_cdf_l) / (hdi_cdf_r - hdi_cdf_l)
        }
    };

    // Compute weights only for indices within HDI range
    let j_l = (hdi_lower * n_f).floor() as usize;
    let j_r = ((hdi_upper * n_f).ceil() as usize).min(n);

    let mut indices = Vec::new();
    let mut weights = Vec::new();

    let mut cdf_right = if j_l > 0 { cdf(j_l as f64 / n_f) } else { 0.0 };

    for j in j_l..j_r {
        let cdf_left = cdf_right;
        let cumulative_prob = (j + 1) as f64 / n_f;
        cdf_right = cdf(cumulative_prob);

        let weight = cdf_right - cdf_left;

        if weight > 1e-12 {
            indices.push(j);
            weights.push(T::from_f64(weight));
        }
    }

    // Normalize
    let sum: f64 = weights.iter().map(|w| w.to_f64()).sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w = T::from_f64(w.to_f64() / sum);
        }
    }

    SparseWeights::new(indices, weights, n)
}

/// Binary search helper
fn binary_search<F>(f: F, mut left: f64, mut right: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    const TOL: f64 = 1e-9;

    let fl = f(left);
    let fr = f(right);

    // Check if function has same sign at both ends
    if fl * fr > 0.0 {
        // Return midpoint as fallback
        return (left + right) / 2.0;
    }

    while right - left > TOL {
        let m = (left + right) / 2.0;
        let fm = f(m);

        if fl * fm < 0.0 {
            right = m;
        } else {
            left = m;
        }
    }

    (left + right) / 2.0
}

/// Log PDF of Beta distribution (unnormalized)
fn beta_log_pdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return f64::NEG_INFINITY;
    }

    (alpha - 1.0) * x.ln() + (beta - 1.0) * (1.0 - x).ln() - ln_beta(alpha, beta)
}
