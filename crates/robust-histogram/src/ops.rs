//! Operations on histograms

use crate::types::Histogram;
use robust_core::Numeric;

/// Operations that can be performed on histograms
pub trait HistogramOps<T: Numeric = f64> {
    /// Calculate the Wasserstein distance between two histograms
    fn wasserstein_distance(&self, other: &Self) -> T::Float;

    /// Calculate the Kullback-Leibler divergence from this histogram to another
    fn kl_divergence(&self, other: &Self) -> T::Float;

    /// Calculate the chi-squared distance between two histograms
    fn chi_squared_distance(&self, other: &Self) -> T::Float;

    /// Calculate the intersection (overlap) between two histograms
    fn intersection(&self, other: &Self) -> T::Float;

    /// Calculate the Bhattacharyya distance between two histograms
    fn bhattacharyya_distance(&self, other: &Self) -> T::Float;
}

impl HistogramOps for Histogram {
    fn wasserstein_distance(&self, other: &Self) -> f64 {
        // Simple implementation for histograms with same support
        if self.is_empty() || other.is_empty() {
            return 0.0;
        }

        // Convert to CDFs
        let cdf1 = cumulative_distribution(self);
        let cdf2 = cumulative_distribution(other);

        // Calculate area between CDFs
        let mut distance = 0.0;
        let all_edges = merge_edges(self, other);

        for i in 0..all_edges.len() - 1 {
            let x1 = all_edges[i];
            let x2 = all_edges[i + 1];
            let width = x2 - x1;

            let y1 = interpolate_cdf(&cdf1, x1);
            let y2 = interpolate_cdf(&cdf2, x1);

            distance += (y1 - y2).abs() * width;
        }

        distance
    }

    fn kl_divergence(&self, other: &Self) -> f64 {
        if self.is_empty() || other.is_empty() {
            return 0.0;
        }

        let mut kl = 0.0;
        let epsilon = 1e-10; // Avoid log(0)

        // Align histograms to same bins
        let (aligned_self, aligned_other) = align_histograms(self, other);

        for (p, q) in aligned_self.iter().zip(aligned_other.iter()) {
            if *p > epsilon {
                let q_safe = q.max(epsilon);
                kl += p * (p / q_safe).ln();
            }
        }

        kl
    }

    fn chi_squared_distance(&self, other: &Self) -> f64 {
        if self.is_empty() || other.is_empty() {
            return 0.0;
        }

        let (aligned_self, aligned_other) = align_histograms(self, other);

        aligned_self
            .iter()
            .zip(aligned_other.iter())
            .map(|(a, b)| {
                let sum = a + b;
                if sum > 0.0 {
                    (a - b).powi(2) / sum
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            * 0.5
    }

    fn intersection(&self, other: &Self) -> f64 {
        if self.is_empty() || other.is_empty() {
            return 0.0;
        }

        let (aligned_self, aligned_other) = align_histograms(self, other);

        aligned_self
            .iter()
            .zip(aligned_other.iter())
            .map(|(a, b)| a.min(*b))
            .sum()
    }

    fn bhattacharyya_distance(&self, other: &Self) -> f64 {
        if self.is_empty() || other.is_empty() {
            return 0.0;
        }

        let (aligned_self, aligned_other) = align_histograms(self, other);

        let bc = aligned_self
            .iter()
            .zip(aligned_other.iter())
            .map(|(a, b)| (a * b).sqrt())
            .sum::<f64>();

        if bc >= 1.0 {
            0.0
        } else {
            (-bc.ln()).max(0.0)
        }
    }
}

// Helper functions

fn cumulative_distribution(hist: &Histogram) -> Vec<(f64, f64)> {
    let mut cdf = Vec::new();
    let mut cumsum = 0.0;

    for bin in hist.bins() {
        cdf.push((bin.left, cumsum));
        cumsum += bin.frequency(hist.total_count());
        cdf.push((bin.right, cumsum));
    }

    cdf
}

fn interpolate_cdf(cdf: &[(f64, f64)], x: f64) -> f64 {
    if cdf.is_empty() {
        return 0.0;
    }

    if x <= cdf[0].0 {
        return cdf[0].1;
    }

    if x >= cdf[cdf.len() - 1].0 {
        return cdf[cdf.len() - 1].1;
    }

    // Binary search for the right interval
    let idx = cdf.partition_point(|(xi, _)| *xi <= x);
    if idx == 0 || idx >= cdf.len() {
        return cdf[cdf.len() - 1].1;
    }

    let (x0, y0) = cdf[idx - 1];
    let (x1, y1) = cdf[idx];

    // Linear interpolation
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

fn merge_edges(hist1: &Histogram, hist2: &Histogram) -> Vec<f64> {
    let mut edges = hist1.edges();
    edges.extend(hist2.edges());
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
    edges.dedup();
    edges
}

fn align_histograms(hist1: &Histogram, hist2: &Histogram) -> (Vec<f64>, Vec<f64>) {
    // Create common bin edges
    let edges = merge_edges(hist1, hist2);

    if edges.len() < 2 {
        return (vec![], vec![]);
    }

    let mut aligned1 = Vec::new();
    let mut aligned2 = Vec::new();

    // Rebin both histograms to common edges
    for i in 0..edges.len() - 1 {
        let left = edges[i];
        let right = edges[i + 1];
        let center = (left + right) / 2.0;

        // Find density at this bin center for both histograms
        let density1 = hist1
            .bins()
            .iter()
            .find(|b| b.contains(center) || (center == b.right && i == edges.len() - 2))
            .map(|b| b.density * b.width())
            .unwrap_or(0.0);

        let density2 = hist2
            .bins()
            .iter()
            .find(|b| b.contains(center) || (center == b.right && i == edges.len() - 2))
            .map(|b| b.density * b.width())
            .unwrap_or(0.0);

        aligned1.push(density1);
        aligned2.push(density2);
    }

    // Normalize to ensure they sum to 1
    let sum1: f64 = aligned1.iter().sum();
    let sum2: f64 = aligned2.iter().sum();

    if sum1 > 0.0 {
        for v in &mut aligned1 {
            *v /= sum1;
        }
    }

    if sum2 > 0.0 {
        for v in &mut aligned2 {
            *v /= sum2;
        }
    }

    (aligned1, aligned2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::FixedWidthBuilder;
    use crate::traits::HistogramBuilder;
    #[test]
    fn test_identical_histograms() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let builder = FixedWidthBuilder::new(5);
        let hist = builder.build(&data).unwrap();

        assert_eq!(hist.wasserstein_distance(&hist), 0.0);
        assert_eq!(hist.kl_divergence(&hist), 0.0);
        assert_eq!(hist.chi_squared_distance(&hist), 0.0);
        assert_eq!(hist.bhattacharyya_distance(&hist), 0.0);
        assert_eq!(hist.intersection(&hist), 1.0);
    }

    #[test]
    fn test_different_histograms() {
        let data1 = vec![1.0, 2.0, 3.0];
        let data2 = vec![4.0, 5.0, 6.0];
        let builder = FixedWidthBuilder::new(3);

        let hist1 = builder.build(&data1).unwrap();
        let hist2 = builder.build(&data2).unwrap();

        assert!(hist1.wasserstein_distance(&hist2) > 0.0);
        assert!(hist1.chi_squared_distance(&hist2) > 0.0);
        assert!(hist1.bhattacharyya_distance(&hist2) > 0.0);
        assert!(hist1.intersection(&hist2) < 1.0);
    }
}
