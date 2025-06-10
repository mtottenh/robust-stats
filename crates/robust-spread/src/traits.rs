//! Core traits for spread estimation

use robust_core::{Result, StatisticalKernel, Numeric};
use robust_quantile::QuantileEstimator;

/// Intrinsic properties of a spread estimator that don't depend on implementation details
pub trait SpreadEstimatorProperties {
    /// Get the name of this spread estimator
    fn name(&self) -> &str;

    /// Check if this estimator is robust to outliers
    fn is_robust(&self) -> bool;

    /// Get the asymptotic breakdown point (0.0 to 0.5)
    fn breakdown_point(&self) -> f64;

    /// Get the efficiency of this estimator relative to the standard deviation
    /// for normal distributions (0.0 to 1.0)
    fn gaussian_efficiency(&self) -> f64;
}

/// Parameterized trait for spread/scale estimators
///
/// This trait follows the new architecture where estimators are parameterized
/// by their dependencies rather than storing them internally.
pub trait SpreadEstimator<T: Numeric = f64, Q: QuantileEstimator<T> = robust_quantile::estimators::Generic<T, robust_core::execution::SequentialEngine<T, robust_core::primitives::ScalarBackend>>>: SpreadEstimatorProperties {
    /// Estimate spread with provided quantile estimator
    fn estimate(&self, data: &mut [T], quantile_est: &Q, cache: &Q::State) -> Result<T::Float>;

    /// Compute spread from pre-sorted data (optimization)
    fn estimate_sorted(
        &self,
        sorted_data: &[T],
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<T::Float>;
}

/// Kernel trait for spread-specific computational patterns
pub trait SpreadKernel<T: Numeric = f64>: StatisticalKernel<T> {
    /// Compute absolute deviations from center
    fn compute_deviations(&self, data: &[T], center: T::Float) -> Vec<T::Float>;

    /// Apply spread-specific transform to deviations
    fn apply_transform(&self, deviations: &[T::Float]) -> T::Float;
}

/// Trait for robust scale estimators that can be used for standardization
pub trait RobustScale<T: Numeric = f64, Q: QuantileEstimator<T> = robust_quantile::estimators::Generic<T, robust_core::execution::SequentialEngine<T, robust_core::primitives::ScalarBackend>>>: SpreadEstimator<T, Q> {
    /// Standardize a value using this scale estimate
    fn standardize(
        &self,
        sample: &mut [T],
        value: T::Float,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<T::Float> {
        let scale = self.estimate(sample, quantile_est, cache)?;
        if scale == <T::Float as num_traits::Zero>::zero() {
            return Err(robust_core::Error::InvalidInput(
                "Scale estimate is zero".to_string(),
            ));
        }
        Ok(value / scale)
    }

    /// Standardize a value using pre-sorted data
    fn standardize_sorted(
        &self,
        sorted_sample: &[T],
        value: T::Float,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<T::Float> {
        let scale = self.estimate_sorted(sorted_sample, quantile_est, cache)?;
        if scale == <T::Float as num_traits::Zero>::zero() {
            return Err(robust_core::Error::InvalidInput(
                "Scale estimate is zero".to_string(),
            ));
        }
        Ok(value / scale)
    }
}
