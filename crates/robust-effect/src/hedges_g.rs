//! Hedges' g effect size
//!
//! Hedges' g is a bias-corrected version of Cohen's d that provides a less biased
//! estimate of effect size, especially for small samples. It applies a correction
//! factor to account for the upward bias in Cohen's d.

use crate::{EffectSize, EffectSizeEstimator, EffectSizeType, StandardizedEffectSize};
use robust_core::{Result, CentralTendencyEstimator, Numeric};
use robust_spread::SpreadEstimator;
use robust_quantile::QuantileEstimator;
use crate::cohen_d::CohenD;

/// Hedges' g effect size estimator
///
/// Hedges' g is calculated as:
/// g = d × J
///
/// where d is Cohen's d and J is the bias correction factor:
/// J ≈ 1 - 3/(4(n₁ + n₂) - 9)
///
/// For large samples, g ≈ d, but for small samples g < d.
#[derive(Debug, Clone, Copy)]
pub struct HedgesG {
    /// Underlying Cohen's d estimator
    cohen_d: CohenD,
}

impl HedgesG {
    /// Create a new Hedges' g estimator
    pub fn new() -> Self {
        Self {
            cohen_d: CohenD::new(),
        }
    }

    /// Create Hedges' g with Welch correction
    pub fn with_welch_correction() -> Self {
        Self {
            cohen_d: CohenD::new().with_welch_correction(),
        }
    }

    /// Calculate the bias correction factor J
    fn bias_correction_factor(n1: usize, n2: usize) -> f64 {
        let total_n = n1 + n2;
        if total_n <= 9 {
            // For very small samples, use the exact correction
            let df = total_n - 2;
            if df <= 0 {
                return 1.0; // fallback
            }
            // Exact correction using gamma functions (approximated)
            1.0 - 3.0 / (4.0 * df as f64 - 1.0)
        } else {
            // Hedges' approximation for larger samples
            1.0 - 3.0 / (4.0 * (total_n as f64) - 9.0)
        }
    }
}

impl Default for HedgesG {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Numeric> StandardizedEffectSize<T> for HedgesG {
    fn compute_with_estimators<L, S, Q>(
        &self,
        group1: &mut [T],
        group2: &mut [T],
        location_estimator: &L,
        spread_estimator: &S,
        quantile_estimator: &Q,
        cache: &Q::State,
    ) -> Result<EffectSize>
    where
        L: CentralTendencyEstimator<T>,
        S: SpreadEstimator<T, Q>,
        Q: QuantileEstimator<T>,
    {
        // First compute Cohen's d
        let cohens_d = self.cohen_d.compute_with_estimators(
            group1,
            group2,
            location_estimator,
            spread_estimator,
            quantile_estimator,
            cache,
        )?;

        // Apply bias correction
        let j = Self::bias_correction_factor(group1.len(), group2.len());
        let g = cohens_d.magnitude * j;

        Ok(EffectSize::new(
            g,
            EffectSizeType::StandardizedMeanDifference,
            Some((group1.len(), group2.len())),
        ))
    }

    fn compute_sorted_with_estimators<L, S, Q>(
        &self,
        sorted_group1: &[T],
        sorted_group2: &[T],
        location_estimator: &L,
        spread_estimator: &S,
        quantile_estimator: &Q,
        cache: &Q::State,
    ) -> Result<EffectSize>
    where
        L: CentralTendencyEstimator<T>,
        S: SpreadEstimator<T, Q>,
        Q: QuantileEstimator<T>,
    {
        // First compute Cohen's d from sorted data
        let cohens_d = self.cohen_d.compute_sorted_with_estimators(
            sorted_group1,
            sorted_group2,
            location_estimator,
            spread_estimator,
            quantile_estimator,
            cache,
        )?;

        // Apply bias correction
        let j = Self::bias_correction_factor(sorted_group1.len(), sorted_group2.len());
        let g = cohens_d.magnitude * j;

        Ok(EffectSize::new(
            g,
            EffectSizeType::StandardizedMeanDifference,
            Some((sorted_group1.len(), sorted_group2.len())),
        ))
    }
}

impl<T: Numeric> EffectSizeEstimator<T> for HedgesG {
    fn effect_size_type(&self) -> EffectSizeType {
        EffectSizeType::StandardizedMeanDifference
    }

    fn supports_weighted_samples(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use robust_core::execution::scalar_sequential;
    use robust_quantile::HDWeightComputer;
    use robust_core::UnifiedWeightCache;
    use robust_core::CachePolicy;
    use crate::cohen_d::{MeanEstimator, StdEstimator};

    #[test]
    fn test_bias_correction_factor() {
        // Test small sample correction
        let j_small = HedgesG::bias_correction_factor(5, 5);
        assert!(j_small < 1.0); // Should be less than 1
        assert!(j_small > 0.9); // But not too small

        // Test large sample correction
        let j_large = HedgesG::bias_correction_factor(50, 50);
        assert!(j_large > 0.99); // Should be very close to 1
        assert!(j_large < 1.0); // But still less than 1
    }

    #[test]
    fn test_hedges_g_vs_cohen_d() {
        let mut group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut group2 = vec![3.0, 4.0, 5.0, 6.0, 7.0];

        let hedges_g = HedgesG::new();
        let cohen_d = CohenD::new();
        
        let engine = scalar_sequential();
        let mean_est = MeanEstimator;
        let std_est = StdEstimator::<f64>::new();
        let quantile_est = robust_quantile::estimators::harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);

        let g_effect = hedges_g.compute_with_estimators(
            &mut group1.clone(),
            &mut group2.clone(),
            &mean_est,
            &std_est,
            &quantile_est,
            &cache
        ).unwrap();
        
        let d_effect = cohen_d.compute_with_estimators(
            &mut group1,
            &mut group2,
            &mean_est,
            &std_est,
            &quantile_est,
            &cache
        ).unwrap();

        // Hedges' g should be smaller than Cohen's d for small samples
        assert!(g_effect.magnitude.abs() < d_effect.magnitude.abs());
        
        // But they should have the same sign
        assert_eq!(g_effect.magnitude.signum(), d_effect.magnitude.signum());
    }

    #[test]
    fn test_small_sample_correction() {
        let group1 = vec![1.0, 2.0, 3.0]; // n=3
        let group2 = vec![4.0, 5.0, 6.0]; // n=3

        let hedges_g = HedgesG::new();
        let engine = scalar_sequential();
        let mean_est = MeanEstimator;
        let std_est = StdEstimator::<f64>::new();
        let quantile_est = robust_quantile::estimators::harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        
        let mut g1 = group1.clone();
        let mut g2 = group2.clone();
        let effect_size = hedges_g.compute_with_estimators(&mut g1, &mut g2, &mean_est, &std_est, &quantile_est, &cache).unwrap();

        // With such small samples, the correction should be substantial
        let j = HedgesG::bias_correction_factor(3, 3);
        assert!(j < 0.9); // Correction factor should be noticeably less than 1
        
        // Effect size should be finite and reasonable
        assert!(effect_size.magnitude.is_finite());
    }

    #[test]
    fn test_large_sample_convergence() {
        let mut group1: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut group2: Vec<f64> = (0..100).map(|i| (i + 5) as f64).collect();

        let hedges_g = HedgesG::new();
        let cohen_d = CohenD::new();
        
        let engine = scalar_sequential();
        let mean_est = MeanEstimator;
        let std_est = StdEstimator::<f64>::new();
        let quantile_est = robust_quantile::estimators::harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);

        let g_effect = hedges_g.compute_with_estimators(
            &mut group1.clone(),
            &mut group2.clone(),
            &mean_est,
            &std_est,
            &quantile_est,
            &cache
        ).unwrap();
        
        let d_effect = cohen_d.compute_with_estimators(
            &mut group1,
            &mut group2,
            &mean_est,
            &std_est,
            &quantile_est,
            &cache
        ).unwrap();

        // For large samples, Hedges' g should be very close to Cohen's d
        assert_abs_diff_eq!(g_effect.magnitude, d_effect.magnitude, epsilon = 0.01);
    }

    #[test]
    fn test_empty_groups() {
        let empty_group = vec![];
        let group = vec![1.0, 2.0, 3.0];

        let hedges_g = HedgesG::new();

        let engine = scalar_sequential();
        let mean_est = MeanEstimator;
        let std_est = StdEstimator::<f64>::new();
        let quantile_est = robust_quantile::estimators::harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        
        assert!(hedges_g.compute_with_estimators(&mut empty_group.clone(), &mut group.clone(), &mean_est, &std_est, &quantile_est, &cache).is_err());
        assert!(hedges_g.compute_with_estimators(&mut group.clone(), &mut empty_group.clone(), &mean_est, &std_est, &quantile_est, &cache).is_err());
    }

    #[test]
    fn test_welch_correction() {
        let group1 = vec![1.0, 2.0, 3.0]; // small group
        let group2 = vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]; // larger group

        let hedges_g_pooled = HedgesG::new();
        let hedges_g_welch = HedgesG::with_welch_correction();

        let engine = scalar_sequential();
        let mean_est = MeanEstimator;
        let std_est = StdEstimator::<f64>::new();
        let quantile_est = robust_quantile::estimators::harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        
        let mut g1_pooled = group1.clone();
        let mut g2_pooled = group2.clone();
        let effect_pooled = hedges_g_pooled.compute_with_estimators(&mut g1_pooled, &mut g2_pooled, &mean_est, &std_est, &quantile_est, &cache).unwrap();
        
        let mut g1_welch = group1.clone();
        let mut g2_welch = group2.clone();
        let effect_welch = hedges_g_welch.compute_with_estimators(&mut g1_welch, &mut g2_welch, &mean_est, &std_est, &quantile_est, &cache).unwrap();

        // Welch correction should give different result
        assert_ne!(effect_pooled.magnitude, effect_welch.magnitude);
    }
}