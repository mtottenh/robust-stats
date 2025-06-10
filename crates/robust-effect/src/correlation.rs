//! Correlation-based effect size measures

use crate::{EffectSize, EffectSizeEstimator, EffectSizeType, NonParametricEffectSize};
use robust_core::{Result, Numeric};
use num_traits::{Float, Zero, One};

/// Point-biserial correlation effect size
///
/// Point-biserial correlation measures the relationship between a binary variable
/// (group membership) and a continuous variable. It's equivalent to Pearson
/// correlation when one variable is binary (0/1).
#[derive(Debug, Clone, Copy)]
pub struct PointBiserialCorrelation;

impl PointBiserialCorrelation {
    /// Create a new point-biserial correlation estimator
    pub fn new() -> Self {
        Self
    }
}

impl Default for PointBiserialCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Numeric> NonParametricEffectSize<T> for PointBiserialCorrelation {
    fn compute(&self, group1: &[T], group2: &[T]) -> Result<EffectSize> {
        if group1.is_empty() || group2.is_empty() {
            return Err(robust_core::Error::InvalidInput(
                "Both groups must be non-empty".to_string(),
            ));
        }

        let values1 = group1;
        let values2 = group2;

        let n1 = values1.len();
        let n2 = values2.len();
        let n_total = n1 + n2;

        use num_traits::NumCast;
        
        // Create binary variable (0 for group1, 1 for group2)
        let mut binary_var = Vec::with_capacity(n_total);
        let mut continuous_var = Vec::with_capacity(n_total);

        // Group 1 coded as 0
        for &value in values1.iter() {
            binary_var.push(T::Float::zero());
            continuous_var.push(value.to_float());
        }

        // Group 2 coded as 1
        for &value in values2.iter() {
            binary_var.push(T::Float::one());
            continuous_var.push(value.to_float());
        }

        // Calculate Pearson correlation
        let r = pearson_correlation(&binary_var, &continuous_var)?;
        let r_f64: f64 = NumCast::from(r).unwrap();

        Ok(EffectSize::new(
            r_f64,
            EffectSizeType::Correlation,
            Some((n1, n2)),
        ))
    }

    fn compute_sorted(&self, sorted_group1: &[T], sorted_group2: &[T]) -> Result<EffectSize> {
        // Correlation doesn't benefit from sorted data
        self.compute(sorted_group1, sorted_group2)
    }
}

impl<T: Numeric> EffectSizeEstimator<T> for PointBiserialCorrelation {
    fn effect_size_type(&self) -> EffectSizeType {
        EffectSizeType::Correlation
    }

    fn supports_weighted_samples(&self) -> bool {
        false
    }
}

// TODO: RobustCorrelation needs to be redesigned to follow the parameterized pattern
// It should take location and scale estimators as parameters rather than storing them

/// Calculate Pearson correlation coefficient
fn pearson_correlation<T: Float + std::ops::AddAssign>(x: &[T], y: &[T]) -> Result<T> {
    if x.len() != y.len() {
        return Err(robust_core::Error::InvalidInput(
            "Arrays must have the same length".to_string(),
        ));
    }

    if x.len() < 2 {
        return Err(robust_core::Error::InvalidInput(
            "Need at least 2 observations".to_string(),
        ));
    }

    use num_traits::NumCast;
    let n: T = NumCast::from(x.len()).unwrap();

    // Calculate means
    let mean_x = x.iter().fold(T::zero(), |acc, &v| acc + v) / n;
    let mean_y = y.iter().fold(T::zero(), |acc, &v| acc + v) / n;

    // Calculate correlation components
    let mut numerator = T::zero();
    let mut sum_sq_x = T::zero();
    let mut sum_sq_y = T::zero();

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;

        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();

    if denominator == T::zero() {
        return Err(robust_core::Error::Computation(
            "Cannot compute correlation: zero variance".to_string(),
        ));
    }

    Ok(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use robust_core::execution::scalar_sequential;

    #[test]
    fn test_point_biserial_correlation() {
        // Groups with clear difference
        let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // mean = 3
        let group2 = vec![6.0, 7.0, 8.0, 9.0, 10.0]; // mean = 8

        let pb_corr = PointBiserialCorrelation::new();
        let effect_size = pb_corr.compute(&group1, &group2).unwrap();

        // Should have strong positive correlation (group2 > group1)
        assert!(effect_size.magnitude > 0.7);
        assert_eq!(effect_size.effect_type, EffectSizeType::Correlation);
        assert_eq!(effect_size.sample_sizes, Some((5, 5)));
    }

    #[test]
    fn test_point_biserial_no_difference() {
        // Identical groups
        let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let pb_corr = PointBiserialCorrelation::new();
        let effect_size = pb_corr.compute(&group1, &group2).unwrap();

        // Should have zero correlation
        assert_abs_diff_eq!(effect_size.magnitude, 0.0, epsilon = 1e-10);
    }

    // TODO: Re-enable this test once RobustCorrelation is implemented with new API
    // #[test]
    // fn test_robust_correlation() {
    //     // Groups with outliers
    //     let group1 = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // outlier
    //     let group2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];

    //     // Traditional point-biserial
    //     let pb_corr = PointBiserialCorrelation::new();
    //     let traditional_effect = pb_corr.effect_size(&group1, &group2).unwrap();

    //     // Robust correlation using median and MAD
    //     let median_estimator = QuantileAdapter::median(HarrellDavisQuantileEstimator::new());
    //     let robust_corr = RobustCorrelation::new(median_estimator, Mad);
    //     let robust_effect = robust_corr.effect_size(&group1, &group2).unwrap();

    //     // Robust version should be less affected by outlier
    //     assert!(robust_effect.magnitude > traditional_effect.magnitude);
    //     assert_eq!(robust_effect.effect_type, EffectSizeType::Correlation);
    // }

    #[test]
    fn test_pearson_correlation_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // perfect positive correlation

        let r = pearson_correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(r, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_correlation_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0]; // perfect negative correlation

        let r = pearson_correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(r, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_correlation_zero() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 5.0, 5.0, 5.0, 5.0]; // no variation in y

        // Should error due to zero variance
        assert!(pearson_correlation(&x, &y).is_err());
    }

    #[test]
    fn test_correlation_bounds() {
        let group1 = vec![1.0, 2.0, 3.0];
        let group2 = vec![4.0, 5.0, 6.0];

        let pb_corr = PointBiserialCorrelation::new();
        let effect_size = pb_corr.compute(&group1, &group2).unwrap();

        // Correlation should be between -1 and 1
        assert!(effect_size.magnitude >= -1.0 && effect_size.magnitude <= 1.0);
    }

    #[test]
    fn test_empty_groups() {
        let empty_group = vec![];
        let group = vec![1.0, 2.0, 3.0];

        let pb_corr = PointBiserialCorrelation::new();

        assert!(pb_corr.compute(&empty_group, &group).is_err());
        assert!(pb_corr.compute(&group, &empty_group).is_err());
    }
}
