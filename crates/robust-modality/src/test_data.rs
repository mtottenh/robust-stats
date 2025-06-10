//! Test data generators for modality testing
//!
//! This module provides standardized test distributions that can be used
//! across all tests to ensure consistency and reproducibility.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Beta, Distribution, Normal, Uniform};

/// Standard test distributions for modality detection
pub struct TestDistributions;

impl TestDistributions {
    /// Create a clearly unimodal normal distribution
    ///
    /// Returns 10000 samples from N(0, 1)
    pub fn unimodal_normal() -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();
        (0..10000).map(|_| normal.sample(&mut rng)).collect()
    }

    /// Create a unimodal distribution with heavy tails
    ///
    /// Returns 10000 samples from a t-distribution approximation
    pub fn unimodal_heavy_tails() -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let uniform = Uniform::new(0.0, 1.0);

        (0..10000)
            .map(|_| {
                let u = uniform.sample(&mut rng);
                if u < 0.9 {
                    // 90% from normal
                    normal.sample(&mut rng)
                } else {
                    // 10% from wider distribution
                    normal.sample(&mut rng) * 3.0
                }
            })
            .collect()
    }

    /// Create a clearly bimodal distribution
    ///
    /// Returns 10000 samples: 5000 from N(-3, 0.8) and 5000 from N(3, 0.8)
    pub fn bimodal_symmetric() -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(42);
        let normal1 = Normal::new(-3.0, 0.8).unwrap();
        let normal2 = Normal::new(3.0, 0.8).unwrap();

        let mut data = Vec::with_capacity(10000);
        for _ in 0..5000 {
            data.push(normal1.sample(&mut rng));
        }
        for _ in 0..5000 {
            data.push(normal2.sample(&mut rng));
        }
        data
    }

    /// Create an asymmetric bimodal distribution
    ///
    /// Returns 10000 samples: 7000 from N(-2, 1.0) and 3000 from N(3, 0.5)
    pub fn bimodal_asymmetric() -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(42);
        let normal1 = Normal::new(-2.0, 1.0).unwrap();
        let normal2 = Normal::new(3.0, 0.5).unwrap();

        let mut data = Vec::with_capacity(10000);
        for _ in 0..7000 {
            data.push(normal1.sample(&mut rng));
        }
        for _ in 0..3000 {
            data.push(normal2.sample(&mut rng));
        }
        data
    }

    /// Create a trimodal distribution
    ///
    /// Returns 12000 samples: 4000 each from N(-4, 0.6), N(0, 0.6), N(4, 0.6)
    pub fn trimodal_symmetric() -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(42);
        let normal1 = Normal::new(-4.0, 0.6).unwrap();
        let normal2 = Normal::new(0.0, 0.6).unwrap();
        let normal3 = Normal::new(4.0, 0.6).unwrap();

        let mut data = Vec::with_capacity(12000);
        for _ in 0..4000 {
            data.push(normal1.sample(&mut rng));
        }
        for _ in 0..4000 {
            data.push(normal2.sample(&mut rng));
        }
        for _ in 0..4000 {
            data.push(normal3.sample(&mut rng));
        }
        data
    }

    /// Create a complex bimodal distribution with overlapping modes
    ///
    /// Returns 10000 samples from two modes that are close together
    pub fn bimodal_overlapping() -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(42);
        let normal1 = Normal::new(-1.0, 1.0).unwrap();
        let normal2 = Normal::new(1.0, 1.0).unwrap();

        let mut data = Vec::with_capacity(10000);
        for _ in 0..5000 {
            data.push(normal1.sample(&mut rng));
        }
        for _ in 0..5000 {
            data.push(normal2.sample(&mut rng));
        }
        data
    }

    /// Create a complex multimodal distribution
    ///
    /// Returns 16000 samples from 4 modes with different weights and spreads
    pub fn multimodal_complex() -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(42);
        let normal1 = Normal::new(-5.0, 0.5).unwrap(); // Sharp mode
        let normal2 = Normal::new(-1.0, 1.5).unwrap(); // Broad mode
        let normal3 = Normal::new(2.0, 0.7).unwrap(); // Medium mode
        let normal4 = Normal::new(5.0, 0.3).unwrap(); // Very sharp mode

        let mut data = Vec::with_capacity(16000);
        // Different weights for each mode
        for _ in 0..2000 {
            data.push(normal1.sample(&mut rng));
        }
        for _ in 0..6000 {
            data.push(normal2.sample(&mut rng));
        }
        for _ in 0..5000 {
            data.push(normal3.sample(&mut rng));
        }
        for _ in 0..3000 {
            data.push(normal4.sample(&mut rng));
        }
        data
    }

    /// Create a uniform distribution (no clear modes)
    ///
    /// Returns 10000 samples from U(-5, 5)
    pub fn uniform_flat() -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(42);
        let uniform = Uniform::new(-5.0, 5.0);
        (0..10000).map(|_| uniform.sample(&mut rng)).collect()
    }

    /// Create a beta distribution (unimodal but skewed)
    ///
    /// Returns 10000 samples from Beta(2, 5) scaled to [-2, 2]
    pub fn unimodal_skewed() -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(42);
        let beta = Beta::new(2.0, 5.0).unwrap();
        (0..10000)
            .map(|_| {
                let x = beta.sample(&mut rng);
                // Scale from [0, 1] to [-2, 2]
                x * 4.0 - 2.0
            })
            .collect()
    }

    /// Create a distribution with outliers
    ///
    /// Returns 10000 samples: 9600 from N(0, 1) and 400 outliers
    pub fn unimodal_with_outliers() -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let uniform = Uniform::new(-10.0, 10.0);

        let mut data = Vec::with_capacity(10000);
        // Main distribution
        for _ in 0..9600 {
            data.push(normal.sample(&mut rng));
        }
        // Outliers
        for _ in 0..400 {
            data.push(uniform.sample(&mut rng));
        }
        data
    }
}

/// Creates a sample from a mixture of normal distributions
///
/// This is useful for creating custom multimodal distributions with precise control
/// over the location, spread, and weight of each mode.
pub fn mixture_normal(
    means: &[f64],
    stds: &[f64],
    weights: &[f64],
    n: usize,
    seed: Option<u64>,
) -> Vec<f64> {
    assert_eq!(means.len(), stds.len());
    assert_eq!(means.len(), weights.len());
    assert!(!means.is_empty());

    // Normalize weights
    let total_weight: f64 = weights.iter().sum();
    let normalized_weights: Vec<f64> = weights.iter().map(|&w| w / total_weight).collect();

    // Calculate cumulative weights for sampling
    let mut cumulative = vec![0.0];
    for &w in &normalized_weights {
        cumulative.push(cumulative.last().unwrap() + w);
    }

    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };

    let uniform = Uniform::new(0.0, 1.0);
    let mut data = Vec::with_capacity(n);

    for _ in 0..n {
        // Select component
        let u = uniform.sample(&mut rng);
        let component = cumulative.iter().position(|&c| c > u).unwrap() - 1;

        // Sample from selected component
        let normal = Normal::new(means[component], stds[component]).unwrap();
        data.push(normal.sample(&mut rng));
    }

    data
}

/// Test parameters for different sensitivity levels
pub struct TestParameters;

impl TestParameters {
    /// Conservative parameters - less likely to detect weak modes
    pub const CONSERVATIVE: (f64, f64) = (0.3, 0.01);

    /// Default parameters - balanced detection
    pub const DEFAULT: (f64, f64) = (0.5, 0.01);

    /// Sensitive parameters - more likely to detect weak modes
    pub const SENSITIVE: (f64, f64) = (0.7, 0.01);

    /// Very sensitive parameters - may over-detect modes
    pub const VERY_SENSITIVE: (f64, f64) = (0.9, 0.01);
}
