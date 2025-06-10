//! Core traits for effect size estimation

use crate::types::EffectSize;
use robust_core::{Result, Numeric};

/// Base trait for effect size estimators between two groups
/// 
/// This trait follows the parameterized design pattern where estimators
/// and engines are passed as parameters rather than stored internally.
/// This ensures consistency with the rest of the codebase and allows
/// for generic programming over numeric types.
pub trait EffectSizeEstimator<T: Numeric = f64> {
    /// Get the type of effect size this estimator computes
    fn effect_size_type(&self) -> crate::EffectSizeType;

    /// Check if the estimator supports weighted samples
    fn supports_weighted_samples(&self) -> bool {
        false
    }

    /// Check if the estimator is symmetric (effect_size(A,B) = -effect_size(B,A))
    fn is_symmetric(&self) -> bool {
        true
    }
}

/// Trait for effect size estimators that use location and spread measures
/// 
/// This trait is for the Cohen's d family of effect sizes that need
/// both location (mean/median) and spread (sd/mad) estimators.
pub trait StandardizedEffectSize<T: Numeric = f64> {
    /// Compute effect size with provided estimators
    /// 
    /// This method allows full control over the estimation process,
    /// including cache reuse for optimal performance.
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
        L: robust_core::CentralTendencyEstimator<T>,
        S: robust_spread::SpreadEstimator<T, Q>,
        Q: robust_quantile::QuantileEstimator<T>;

    /// Compute from sorted data
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
        L: robust_core::CentralTendencyEstimator<T>,
        S: robust_spread::SpreadEstimator<T, Q>,
        Q: robust_quantile::QuantileEstimator<T>;
}

/// Trait for non-parametric effect size estimators
/// 
/// These estimators (like Cliff's Delta) don't need location or spread estimators
pub trait NonParametricEffectSize<T: Numeric = f64> {
    /// Compute effect size directly from the data
    fn compute(&self, group1: &[T], group2: &[T]) -> Result<EffectSize>;
    
    /// Compute from sorted data (may provide optimization opportunities)
    fn compute_sorted(&self, sorted_group1: &[T], sorted_group2: &[T]) -> Result<EffectSize>;
}

