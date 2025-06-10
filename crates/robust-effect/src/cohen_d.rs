//! Cohen's d effect size with robust estimators
//!
//! Cohen's d is a standardized effect size measure that expresses the difference
//! between two group means in terms of the pooled standard deviation. This implementation
//! follows the parameterized design where estimators are passed as parameters rather
//! than stored internally.

use crate::{EffectSize, EffectSizeEstimator, EffectSizeType, StandardizedEffectSize};
use robust_core::{Result, CentralTendencyEstimator, Numeric};
use robust_spread::{SpreadEstimator, SpreadEstimatorProperties};
use robust_quantile::QuantileEstimator;

/// Cohen's d effect size estimator
///
/// The traditional Cohen's d is calculated as:
/// d = (μ₁ - μ₂) / σ_pooled
///
/// This implementation generalizes it to:
/// d = (location₁ - location₂) / scale_pooled
///
/// where location and scale can be any robust estimators passed as parameters.
#[derive(Debug, Clone, Copy)]
pub struct CohenD {
    /// Whether to use Welch's correction for unequal variances
    use_welch_correction: bool,
}

impl CohenD {
    /// Create a new Cohen's d estimator
    pub fn new() -> Self {
        Self {
            use_welch_correction: false,
        }
    }

    /// Enable Welch's correction for unequal variances
    pub fn with_welch_correction(mut self) -> Self {
        self.use_welch_correction = true;
        self
    }

    fn compute<T, L, S, Q>(
        &self,
        group1: &mut [T],
        group2: &mut [T],
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
        if group1.is_empty() || group2.is_empty() {
            return Err(robust_core::Error::InvalidInput(
                "Both groups must be non-empty".to_string(),
            ));
        }

        // Calculate locations
        let location1 = location_estimator.estimate(group1)?;
        let location2 = location_estimator.estimate(group2)?;

        // Calculate scales
        let scale1 = scale_estimator.estimate(group1, quantile_estimator, cache)?;
        let scale2 = scale_estimator.estimate(group2, quantile_estimator, cache)?;

        use num_traits::{Zero, NumCast, ToPrimitive};
        
        let zero: T::Float = NumCast::from(0.0).unwrap();
        if scale1 <= zero || scale2 <= zero {
            return Err(robust_core::Error::Computation(
                "Scale estimates must be positive".to_string(),
            ));
        }

        // Calculate pooled scale
        let pooled_scale = if self.use_welch_correction {
            // Welch's approach: don't pool, use separate variances
            let n1: T::Float = NumCast::from(group1.len()).unwrap();
            let n2: T::Float = NumCast::from(group2.len()).unwrap();
            let scale1_sq = scale1 * scale1;
            let scale2_sq = scale2 * scale2;
            let var_sum = scale1_sq / n1 + scale2_sq / n2;
            NumCast::from(var_sum.to_f64().unwrap().sqrt()).unwrap()
        } else {
            // Traditional pooled approach
            let n1: T::Float = NumCast::from(group1.len()).unwrap();
            let n2: T::Float = NumCast::from(group2.len()).unwrap();
            let one: T::Float = NumCast::from(1.0).unwrap();
            let two: T::Float = NumCast::from(2.0).unwrap();
            let n1_minus_one: T::Float = NumCast::from(group1.len() - 1).unwrap();
            let n2_minus_one: T::Float = NumCast::from(group2.len() - 1).unwrap();
            let n_total_minus_two: T::Float = NumCast::from(group1.len() + group2.len() - 2).unwrap();
            let scale1_sq = scale1 * scale1;
            let scale2_sq = scale2 * scale2;
            let pooled_variance: T::Float =
                (n1_minus_one * scale1_sq + n2_minus_one * scale2_sq) / n_total_minus_two;
            NumCast::from(pooled_variance.to_f64().unwrap().sqrt()).unwrap()
        };

        let zero: T::Float = NumCast::from(0.0).unwrap();
        if pooled_scale <= zero {
            return Err(robust_core::Error::Computation(
                "Pooled scale is non-positive".to_string(),
            ));
        }

        // Calculate Cohen's d
        let d = (location1 - location2) / pooled_scale;
        let d_f64: f64 = NumCast::from(d).unwrap();

        Ok(EffectSize::new(
            d_f64,
            EffectSizeType::StandardizedMeanDifference,
            Some((group1.len(), group2.len())),
        ))
    }

    /// Compute effect size from sorted data
    pub fn compute_sorted<T, L, S, Q>(
        &self,
        sorted_group1: &[T],
        sorted_group2: &[T],
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
        if sorted_group1.is_empty() || sorted_group2.is_empty() {
            return Err(robust_core::Error::InvalidInput(
                "Both groups must be non-empty".to_string(),
            ));
        }

        // Calculate locations from sorted data
        let location1 = location_estimator.estimate_sorted(sorted_group1)?;
        let location2 = location_estimator.estimate_sorted(sorted_group2)?;

        // Calculate scales from sorted data
        let scale1 = scale_estimator.estimate_sorted(sorted_group1, quantile_estimator, cache)?;
        let scale2 = scale_estimator.estimate_sorted(sorted_group2, quantile_estimator, cache)?;

        use num_traits::{Zero, NumCast, ToPrimitive};

        let zero: T::Float = NumCast::from(0.0).unwrap();
        if scale1 <= zero || scale2 <= zero {
            return Err(robust_core::Error::Computation(
                "Scale estimates must be positive".to_string(),
            ));
        }

        // Calculate pooled scale
        let pooled_scale = if self.use_welch_correction {
            let n1: T::Float = NumCast::from(sorted_group1.len()).unwrap();
            let n2: T::Float = NumCast::from(sorted_group2.len()).unwrap();
            let scale1_sq = scale1 * scale1;
            let scale2_sq = scale2 * scale2;
            let var_sum = scale1_sq / n1 + scale2_sq / n2;
            NumCast::from(var_sum.to_f64().unwrap().sqrt()).unwrap()
        } else {
            let n1: T::Float = NumCast::from(sorted_group1.len()).unwrap();
            let n2: T::Float = NumCast::from(sorted_group2.len()).unwrap();
            let one: T::Float = NumCast::from(1.0).unwrap();
            let two: T::Float = NumCast::from(2.0).unwrap();
            let n1_minus_one: T::Float = NumCast::from(sorted_group1.len() - 1).unwrap();
            let n2_minus_one: T::Float = NumCast::from(sorted_group2.len() - 1).unwrap();
            let n_total_minus_two: T::Float = NumCast::from(sorted_group1.len() + sorted_group2.len() - 2).unwrap();
            let scale1_sq = scale1 * scale1;
            let scale2_sq = scale2 * scale2;
            let pooled_variance: T::Float =
                (n1_minus_one * scale1_sq + n2_minus_one * scale2_sq) / n_total_minus_two;
            NumCast::from(pooled_variance.to_f64().unwrap().sqrt()).unwrap()
        };

        let zero: T::Float = NumCast::from(0.0).unwrap();
        if pooled_scale <= zero {
            return Err(robust_core::Error::Computation(
                "Pooled scale is non-positive".to_string(),
            ));
        }

        // Calculate Cohen's d
        let d = (location1 - location2) / pooled_scale;
        let d_f64: f64 = NumCast::from(d).unwrap();

        Ok(EffectSize::new(
            d_f64,
            EffectSizeType::StandardizedMeanDifference,
            Some((sorted_group1.len(), sorted_group2.len())),
        ))
    }
}

impl Default for CohenD {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Numeric> StandardizedEffectSize<T> for CohenD {
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

impl<T: Numeric> EffectSizeEstimator<T> for CohenD {
    fn effect_size_type(&self) -> EffectSizeType {
        EffectSizeType::StandardizedMeanDifference
    }

    fn supports_weighted_samples(&self) -> bool {
        false
    }
}

// Simple mean estimator for traditional Cohen's d
#[derive(Debug, Clone, Copy)]
pub struct MeanEstimator;

impl<T: Numeric> CentralTendencyEstimator<T> for MeanEstimator {
    fn estimate(&self, sample: &mut [T]) -> Result<T::Float> {
        if sample.is_empty() {
            return Err(robust_core::Error::InvalidInput("Empty sample".to_string()));
        }
        use num_traits::{Zero, NumCast, ToPrimitive};
        let sum = sample.iter().fold(T::Float::zero(), |acc, &x| {
            acc + x.to_float()
        });
        let n: T::Float = NumCast::from(sample.len()).unwrap();
        Ok(sum / n)
    }

    fn estimate_sorted(&self, sorted_sample: &[T]) -> Result<T::Float> {
        if sorted_sample.is_empty() {
            return Err(robust_core::Error::InvalidInput("Empty sample".to_string()));
        }
        use num_traits::{Zero, NumCast, ToPrimitive};
        let sum = sorted_sample.iter().fold(T::Float::zero(), |acc, &x| {
            acc + x.to_float()
        });
        let n: T::Float = NumCast::from(sorted_sample.len()).unwrap();
        Ok(sum / n)
    }
    
    fn name(&self) -> &str {
        "mean"
    }
    
    fn is_robust(&self) -> bool {
        false
    }
    
    fn breakdown_point(&self) -> f64 {
        0.0
    }
}

// Simple standard deviation estimator
#[derive(Debug, Clone, Copy)]
pub struct StdEstimator<T: Numeric> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric> StdEstimator<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

impl<T: Numeric, Q: QuantileEstimator<T>> SpreadEstimator<T, Q> for StdEstimator<T> {
    fn estimate(&self, data: &mut [T], _quantile_est: &Q, _cache: &Q::State) -> Result<T::Float> {
        if data.len() < 2 {
            return Err(robust_core::Error::InvalidInput(
                "Need at least 2 observations for standard deviation".to_string(),
            ));
        }

        use num_traits::{Float, Zero, One, NumCast, ToPrimitive};
        let sum = data.iter().fold(T::Float::zero(), |acc, &x| {
            acc + x.to_float()
        });
        let n: T::Float = NumCast::from(data.len()).unwrap();
        let mean = sum / n;
        
        let variance = data
            .iter()
            .fold(T::Float::zero(), |acc, &x| {
                let x_f: T::Float = x.to_float();
                acc + (x_f - mean).powi(2)
            })
            / (n - T::Float::one());

        Ok(NumCast::from(variance.to_f64().unwrap().sqrt()).unwrap())
    }

    fn estimate_sorted(&self, sorted_data: &[T], _quantile_est: &Q, _cache: &Q::State) -> Result<T::Float> {
        if sorted_data.len() < 2 {
            return Err(robust_core::Error::InvalidInput(
                "Need at least 2 observations for standard deviation".to_string(),
            ));
        }

        use num_traits::{Float, Zero, One, NumCast, ToPrimitive};
        let sum = sorted_data.iter().fold(T::Float::zero(), |acc, &x| {
            acc + x.to_float()
        });
        let n: T::Float = NumCast::from(sorted_data.len()).unwrap();
        let mean = sum / n;
        
        let variance = sorted_data
            .iter()
            .fold(T::Float::zero(), |acc, &x| {
                let x_f: T::Float = x.to_float();
                acc + (x_f - mean).powi(2)
            })
            / (n - T::Float::one());

        Ok(NumCast::from(variance.to_f64().unwrap().sqrt()).unwrap())
    }
}

impl<T: Numeric> SpreadEstimatorProperties for StdEstimator<T> {
    fn name(&self) -> &str {
        "Standard Deviation"
    }

    fn is_robust(&self) -> bool {
        false
    }

    fn breakdown_point(&self) -> f64 {
        0.0
    }

    fn gaussian_efficiency(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use robust_core::execution::scalar_sequential;
    use robust_core::{ExecutionEngine, UnifiedWeightCache, CachePolicy};
    use robust_quantile::HDWeightComputer;

    #[test]
    fn test_traditional_cohens_d() {
        let mut group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut group2 = vec![3.0, 4.0, 5.0, 6.0, 7.0];

        let cohen_d = CohenD::new();
        let mean_est = MeanEstimator;
        let engine = scalar_sequential();
        let std_est = StdEstimator::<f64>::new();
        
        // Create quantile estimator and cache (needed for SpreadEstimator interface)
        let quantile_est = robust_quantile::estimators::harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        
        let effect_size = cohen_d.compute(
            &mut group1,
            &mut group2,
            &mean_est,
            &std_est,
            &quantile_est,
            &cache
        ).unwrap();

        // Expected: (5-3) / pooled_sd ≈ -2.0 / 1.58 ≈ -1.26
        assert_abs_diff_eq!(effect_size.magnitude, -1.26, epsilon = 0.1);
        assert_eq!(
            effect_size.effect_type,
            EffectSizeType::StandardizedMeanDifference
        );
        assert_eq!(effect_size.sample_sizes, Some((5, 5)));
    }

    #[test]
    fn test_robust_cohens_d() {
        let mut group1 = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // outlier!
        let mut group2 = vec![3.0, 4.0, 5.0, 6.0, 7.0];

        let cohen_d = CohenD::new();
        let engine = scalar_sequential();
        
        // Robust version using median and MAD
        let quantile_est = robust_quantile::estimators::harrell_davis(engine.clone());
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        let mad = robust_spread::Mad::new(engine.primitives().clone());
        
        // Use median as location estimator (via quantile at 0.5)
        let median_est = MedianEstimator;
        
        let effect_size = cohen_d.compute(
            &mut group1,
            &mut group2,
            &median_est,
            &mad,
            &quantile_est,
            &cache
        ).unwrap();

        // Effect size should be computed successfully and be finite
        assert!(effect_size.magnitude.is_finite());
    }

    #[test]
    fn test_welch_correction() {
        let mut group1 = vec![1.0, 2.0, 3.0]; // small group
        let mut group2 = vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]; // larger group

        let engine = scalar_sequential();
        let mean_est = MeanEstimator;
        let std_est = StdEstimator::<f64>::new();
        let quantile_est = robust_quantile::estimators::harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);

        let cohen_d_pooled = CohenD::new();
        let cohen_d_welch = CohenD::new().with_welch_correction();

        let effect_pooled = cohen_d_pooled.compute(
            &mut group1.clone(),
            &mut group2.clone(),
            &mean_est,
            &std_est,
            &quantile_est,
            &cache
        ).unwrap();
        
        let effect_welch = cohen_d_welch.compute(
            &mut group1,
            &mut group2,
            &mean_est,
            &std_est,
            &quantile_est,
            &cache
        ).unwrap();

        // Welch correction should give different result
        assert_ne!(effect_pooled.magnitude, effect_welch.magnitude);
    }

    #[test]
    fn test_zero_difference() {
        let mut group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut group2 = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // identical

        let cohen_d = CohenD::new();
        let mean_est = MeanEstimator;
        let engine = scalar_sequential();
        let std_est = StdEstimator::<f64>::new();
        let quantile_est = robust_quantile::estimators::harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        
        let effect_size = cohen_d.compute(
            &mut group1,
            &mut group2,
            &mean_est,
            &std_est,
            &quantile_est,
            &cache
        ).unwrap();

        assert_abs_diff_eq!(effect_size.magnitude, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_empty_groups() {
        let mut empty_group = vec![];
        let mut group = vec![1.0, 2.0, 3.0];

        let cohen_d = CohenD::new();
        let mean_est = MeanEstimator;
        let engine = scalar_sequential();
        let std_est = StdEstimator::<f64>::new();
        let quantile_est = robust_quantile::estimators::harrell_davis(engine);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);

        assert!(cohen_d.compute(&mut empty_group, &mut group.clone(), &mean_est, &std_est, &quantile_est, &cache).is_err());
        assert!(cohen_d.compute(&mut group, &mut empty_group, &mean_est, &std_est, &quantile_est, &cache).is_err());
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
                Ok((v1 + v2) / NumCast::from(2.0).unwrap())
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
                Ok((v1 + v2) / NumCast::from(2.0).unwrap())
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