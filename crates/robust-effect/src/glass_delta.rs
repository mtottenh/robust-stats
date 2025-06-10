//! Glass's delta effect size
//!
//! Glass's delta is similar to Cohen's d but uses only the control group's
//! standard deviation as the denominator, making it useful when one group
//! can be considered a control or baseline.

use crate::{EffectSize, EffectSizeEstimator, EffectSizeType, StandardizedEffectSize};
use robust_core::{Result, CentralTendencyEstimator, Numeric};
use robust_spread::SpreadEstimator;
use robust_quantile::QuantileEstimator;

/// Glass's delta effect size estimator
///
/// Glass's delta (Δ) is calculated as:
/// Δ = (μ₂ - μ₁) / σ₁
///
/// where μ₁, μ₂ are the group means and σ₁ is the control group (group 1) standard deviation.
/// This implementation follows the parameterized design where estimators are passed as parameters.
#[derive(Debug, Clone, Copy)]
pub struct GlassDelta;

impl GlassDelta {
    /// Create a new Glass's delta estimator
    pub fn new() -> Self {
        Self
    }

    /// Compute effect size using provided estimators
    fn compute<T, L, S, Q>(
        &self,
        control_group: &mut [T],
        treatment_group: &mut [T],
        location_estimator: &L,
        scale_estimator: &S,
        quantile_estimator: &Q,
        cache: &Q::State,
    ) -> Result<EffectSize>
    where
        T: Numeric,
        L: CentralTendencyEstimator<T>,
        S: SpreadEstimator<T, Q>,
        Q: QuantileEstimator<T>,
    {
        if control_group.is_empty() || treatment_group.is_empty() {
            return Err(robust_core::Error::InvalidInput(
                "Both groups must be non-empty".to_string(),
            ));
        }

        // Calculate locations
        let location1 = location_estimator.estimate(control_group)?;
        let location2 = location_estimator.estimate(treatment_group)?;

        // Calculate scale for control group (group1) only
        let scale1 = scale_estimator.estimate(control_group, quantile_estimator, cache)?;

        use num_traits::{Zero, NumCast};
        
        if scale1 <= T::Float::zero() {
            return Err(robust_core::Error::Computation(
                "Control group scale estimate must be positive".to_string(),
            ));
        }

        // Calculate Glass's delta
        let delta = (location2 - location1) / scale1;
        let delta_f64: f64 = NumCast::from(delta).unwrap();

        Ok(EffectSize::new(
            delta_f64,
            EffectSizeType::StandardizedMeanDifference,
            Some((control_group.len(), treatment_group.len())),
        ))
    }

    /// Compute effect size from sorted data
    fn compute_sorted<T, L, S, Q>(
        &self,
        sorted_control_group: &[T],
        sorted_treatment_group: &[T],
        location_estimator: &L,
        scale_estimator: &S,
        quantile_estimator: &Q,
        cache: &Q::State,
    ) -> Result<EffectSize>
    where
        T: Numeric,
        L: CentralTendencyEstimator<T>,
        S: SpreadEstimator<T, Q>,
        Q: QuantileEstimator<T>,
    {
        if sorted_control_group.is_empty() || sorted_treatment_group.is_empty() {
            return Err(robust_core::Error::InvalidInput(
                "Both groups must be non-empty".to_string(),
            ));
        }

        // Calculate locations from sorted data
        let location1 = location_estimator.estimate_sorted(sorted_control_group)?;
        let location2 = location_estimator.estimate_sorted(sorted_treatment_group)?;

        // Calculate scale for control group only
        let scale1 = scale_estimator.estimate_sorted(sorted_control_group, quantile_estimator, cache)?;

        use num_traits::{Zero, NumCast};
        
        if scale1 <= T::Float::zero() {
            return Err(robust_core::Error::Computation(
                "Control group scale estimate must be positive".to_string(),
            ));
        }

        // Calculate Glass's delta
        let delta = (location2 - location1) / scale1;
        let delta_f64: f64 = NumCast::from(delta).unwrap();

        Ok(EffectSize::new(
            delta_f64,
            EffectSizeType::StandardizedMeanDifference,
            Some((sorted_control_group.len(), sorted_treatment_group.len())),
        ))
    }
}

impl Default for GlassDelta {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Numeric> StandardizedEffectSize<T> for GlassDelta {
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
        self.compute(group1, group2, location_estimator, spread_estimator, quantile_estimator, cache)
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
        self.compute_sorted(sorted_group1, sorted_group2, location_estimator, spread_estimator, quantile_estimator, cache)
    }
}

impl<T: Numeric> EffectSizeEstimator<T> for GlassDelta {
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
    use robust_core::ExecutionEngine;
    use robust_quantile::HDWeightComputer;
    use robust_core::UnifiedWeightCache;
    use robust_core::CachePolicy;
    use crate::cohen_d::{MeanEstimator, StdEstimator};

    #[test]
    fn test_traditional_glass_delta() {
        let mut control = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // mean=3, sd≈1.58
        let mut treatment = vec![3.0, 4.0, 5.0, 6.0, 7.0]; // mean=5

        let glass_delta = GlassDelta::new();
        let mean_est = MeanEstimator;
        let engine = scalar_sequential();
        let std_est = StdEstimator::<f64>::new();
        
        // Create quantile estimator and cache
        let quantile_est = robust_quantile::estimators::harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        
        let effect_size = glass_delta.compute(
            &mut control,
            &mut treatment,
            &mean_est,
            &std_est,
            &quantile_est,
            &cache
        ).unwrap();

        // Expected: (5-3) / 1.58 ≈ 1.26
        assert_abs_diff_eq!(effect_size.magnitude, 1.26, epsilon = 0.1);
        assert_eq!(
            effect_size.effect_type,
            EffectSizeType::StandardizedMeanDifference
        );
    }

    #[test]
    fn test_robust_glass_delta() {
        let mut control = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // outlier in control
        let mut treatment = vec![3.0, 4.0, 5.0, 6.0, 7.0];

        let glass_delta = GlassDelta::new();
        let engine = scalar_sequential();
        
        // Robust version using median and MAD
        let quantile_est = robust_quantile::estimators::harrell_davis(engine.clone());
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        let mad = robust_spread::Mad::new(engine.primitives().clone());
        
        // Use median as location estimator
        let median_est = MedianEstimator;
        
        let effect_size = glass_delta.compute(
            &mut control,
            &mut treatment,
            &median_est,
            &mad,
            &quantile_est,
            &cache
        ).unwrap();

        // Effect size should be computed successfully
        assert!(effect_size.magnitude.is_finite());
    }

    #[test]
    fn test_zero_difference() {
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // identical

        let glass_delta = GlassDelta::new();
        let engine = scalar_sequential();
        let mean_est = MeanEstimator;
        let std_est = StdEstimator::<f64>::new();
        let quantile_est = robust_quantile::estimators::harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        
        let mut control_mut = control.clone();
        let mut treatment_mut = treatment.clone();
        let effect_size = glass_delta.compute(&mut control_mut, &mut treatment_mut, &mean_est, &std_est, &quantile_est, &cache).unwrap();

        assert_abs_diff_eq!(effect_size.magnitude, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_glass_vs_cohen() {
        use crate::cohen_d::CohenD;

        // Groups with different variances
        let control = vec![1.0, 2.0, 3.0]; // smaller variance
        let treatment = vec![10.0, 20.0, 30.0, 40.0, 50.0]; // larger variance

        let glass_delta = GlassDelta::new();
        let cohen_d = CohenD::new();

        let engine = scalar_sequential();
        let mean_est = MeanEstimator;
        let std_est = StdEstimator::<f64>::new();
        let quantile_est = robust_quantile::estimators::harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        
        let mut control_mut1 = control.clone();
        let mut treatment_mut1 = treatment.clone();
        let glass_effect = glass_delta.compute_with_estimators(&mut control_mut1, &mut treatment_mut1, &mean_est, &std_est, &quantile_est, &cache).unwrap();
        
        let mut control_mut2 = control.clone();
        let mut treatment_mut2 = treatment.clone();
        let cohen_effect = cohen_d.compute_with_estimators(&mut control_mut2, &mut treatment_mut2, &mean_est, &std_est, &quantile_est, &cache).unwrap();

        // Glass's delta uses only control SD, Cohen's d uses pooled SD
        // With unequal variances, they should give different results
        assert_ne!(glass_effect.magnitude, cohen_effect.magnitude);
    }

    #[test]
    fn test_empty_groups() {
        let empty_group = vec![];
        let group = vec![1.0, 2.0, 3.0];

        let glass_delta = GlassDelta::new();

        let engine = scalar_sequential();
        let mean_est = MeanEstimator;
        let std_est = StdEstimator::<f64>::new();
        let quantile_est = robust_quantile::estimators::harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        
        assert!(glass_delta.compute(&mut empty_group.clone(), &mut group.clone(), &mean_est, &std_est, &quantile_est, &cache).is_err());
        assert!(glass_delta.compute(&mut group.clone(), &mut empty_group.clone(), &mean_est, &std_est, &quantile_est, &cache).is_err());
    }

    #[test]
    fn test_zero_control_variance() {
        let control = vec![5.0, 5.0, 5.0]; // constant values
        let treatment = vec![1.0, 2.0, 3.0];

        let glass_delta = GlassDelta::new();
        let engine = scalar_sequential();
        let mean_est = MeanEstimator;
        let std_est = StdEstimator::<f64>::new();
        let quantile_est = robust_quantile::estimators::harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        
        let mut control_mut = control.clone();
        let mut treatment_mut = treatment.clone();
        // Should error because control group has zero variance
        assert!(glass_delta.compute(&mut control_mut, &mut treatment_mut, &mean_est, &std_est, &quantile_est, &cache).is_err());
    }
    
    // Helper median estimator for tests
    #[derive(Clone)]
    struct MedianEstimator;
    
    impl<T: Numeric> CentralTendencyEstimator<T> for MedianEstimator {
        fn estimate(&self, sample: &mut [T]) -> Result<T::Float> {
            if sample.is_empty() {
                return Err(robust_core::Error::InvalidInput("Empty sample".to_string()));
            }
            sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = sample.len();
            use num_traits::{Float, NumCast};
            if n % 2 == 0 {
                let v1: T::Float = sample[n/2 - 1].to_float();
                let v2: T::Float = sample[n/2].to_float();
                let two: T::Float = NumCast::from(2.0).unwrap();
                Ok((v1 + v2) / two)
            } else {
                Ok(sample[n/2].to_float())
            }
        }

        fn estimate_sorted(&self, sorted_sample: &[T]) -> Result<T::Float> {
            if sorted_sample.is_empty() {
                return Err(robust_core::Error::InvalidInput("Empty sample".to_string()));
            }
            let n = sorted_sample.len();
            use num_traits::{Float, NumCast};
            if n % 2 == 0 {
                let v1: T::Float = sorted_sample[n/2 - 1].to_float();
                let v2: T::Float = sorted_sample[n/2].to_float();
                let two: T::Float = NumCast::from(2.0).unwrap();
                Ok((v1 + v2) / two)
            } else {
                Ok(sorted_sample[n/2].to_float())
            }
        }
        
        fn name(&self) -> &str {
            "median"
        }
        
        fn is_robust(&self) -> bool {
            true
        }
        
        fn breakdown_point(&self) -> f64 {
            0.5
        }
    }
}