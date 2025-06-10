//! Statistical window analyzer implementation

use crate::types::{StabilityParameters, StabilityStatus, UnstableReason};
use crate::window_traits::{WindowStabilityAnalyzer, WindowStabilityResult};
use num_traits::{Float, FromPrimitive, One, Zero};
use robust_core::{ExecutionEngine, Numeric, Result};
use robust_quantile::{QuantileEstimator, SqrtWidth, TrimmedHarrellDavis};
use robust_spread::{Mad, SpreadEstimator};
/// Compatibility wrapper for StandardizedMAD
#[derive(Clone, Debug)]
pub struct StandardizedMadWrapper<T: Numeric, S, Q> {
    spread_estimator: S,
    quantile_estimator: Q,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, S, Q> StandardizedMadWrapper<T, S, Q>
where
    S: SpreadEstimator<T, Q>,
    Q: QuantileEstimator<T>,
    T::Float: FromPrimitive,
{
    pub fn new(spread_estimator: S, quantile_estimator: Q) -> Self {
        Self {
            spread_estimator,
            quantile_estimator,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn estimate(&self, sample: &mut [T], cache: &Q::State) -> Result<T::Float> {
        let mad_value = self
            .spread_estimator
            .estimate(sample, &self.quantile_estimator, cache)?;
        Ok(mad_value * T::Float::from_f64(1.4826).unwrap()) // Apply standardization factor
    }
}

/// Statistical analyzer for single windows
///
/// Generic over:
/// - `T`: Numeric type
/// - `Q`: Quantile estimator for computing median
/// - `S`: Spread estimator for computing robust standard deviation
#[derive(Clone)]
pub struct StatisticalWindowAnalyzer<T: Numeric, S, Q>
where
    Q: QuantileEstimator<T> + Clone,
{
    params: StabilityParameters<T>,
    spread_estimator: S,
    quantile_estimator: Q,
    cache: Q::State,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, S, Q> StatisticalWindowAnalyzer<T, S, Q>
where
    S: SpreadEstimator<T, Q> + Clone,
    Q: QuantileEstimator<T> + Clone,
{
    pub fn new(
        params: StabilityParameters<T>,
        spread_estimator: S,
        quantile_estimator: Q,
        cache: Q::State,
    ) -> Self {
        Self {
            params,
            spread_estimator,
            quantile_estimator,
            cache,
            _phantom: std::marker::PhantomData,
        }
    }
}

// Type alias for common configuration using Mad and TrimmedHD with runtime-selected backend
pub type DefaultStatisticalAnalyzer = StatisticalWindowAnalyzer<
    f64,
    Mad<f64, robust_core::primitives::ScalarBackend>,
    TrimmedHarrellDavis<
        f64,
        robust_core::execution::SequentialEngine<f64, robust_core::primitives::ScalarBackend>,
        SqrtWidth,
    >,
>;

impl DefaultStatisticalAnalyzer {
    /// Create analyzer with default estimators
    pub fn with_default_estimators(params: StabilityParameters<f64>) -> Self {
        let engine = robust_core::scalar_sequential();
        let primitives = engine.primitives().clone();
        let spread_est = Mad::new(primitives);
        let quantile_est = robust_quantile::estimators::trimmed_harrell_davis(engine, SqrtWidth);
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::TrimmedHDWeightComputer::new(SqrtWidth),
            robust_core::CachePolicy::NoCache,
        );
        Self::new(params, spread_est, quantile_est, cache)
    }
}

impl<T: Numeric, S, Q> WindowStabilityAnalyzer<T> for StatisticalWindowAnalyzer<T, S, Q>
where
    S: SpreadEstimator<T, Q> + Clone + Send + Sync,
    Q: QuantileEstimator<T> + Clone + Send + Sync,
    Q::State: Clone + Send + Sync,
    T::Float: FromPrimitive,
{
    fn analyze_window(&self, window: &[T]) -> WindowStabilityResult<T> {
        // Handle empty window
        if window.is_empty() {
            return WindowStabilityResult {
                is_stable: false,
                cv: T::Float::zero(),
                trend_strength: T::Float::zero(),
                mean: T::Float::zero(),
                robust_std: T::Float::zero(),
                status: StabilityStatus::Unknown,
            };
        }

        let n = T::Float::from_f64(window.len() as f64).unwrap();

        // Check for NaN values
        let has_nan = window
            .iter()
            .any(|x| x.to_f64().is_nan());
        if has_nan {
            return WindowStabilityResult {
                is_stable: false,
                cv: T::Float::nan(),
                trend_strength: T::Float::nan(),
                mean: T::Float::nan(),
                robust_std: T::Float::nan(),
                status: StabilityStatus::Unknown,
            };
        }

        // Robust statistics using generic estimators
        let mut window_copy = window.to_vec();
        let median = self
            .quantile_estimator
            .quantile(&mut window_copy, 0.5, &self.cache)
            .unwrap_or(T::Float::zero());

        let mut window_for_spread = window.to_vec();
        let spread_value = self
            .spread_estimator
            .estimate(
                &mut window_for_spread,
                &self.quantile_estimator,
                &self.cache,
            )
            .unwrap_or(T::Float::zero());
        let robust_std = spread_value * T::Float::from_f64(1.4826).unwrap(); // Apply standardization factor

        // Inline trend calculation - no allocations
        let (sum_x, sum_y, sum_xy, sum_x2) = window.iter().enumerate().fold(
            (
                T::Float::zero(),
                T::Float::zero(),
                T::Float::zero(),
                T::Float::zero(),
            ),
            |(sx, sy, sxy, sx2), (i, y)| {
                let x = T::Float::from_f64(i as f64).unwrap();
                let y_float = y.to_float();
                (sx + x, sy + y_float, sxy + x * y_float, sx2 + x * x)
            },
        );

        let mean = sum_y / n;
        let trend_strength = if n > T::Float::one() {
            let denom = n * sum_x2 - sum_x * sum_x;
            if denom > T::Float::zero() {
                ((n * sum_xy - sum_x * sum_y) / denom).abs()
            } else {
                T::Float::zero()
            }
        } else {
            T::Float::zero()
        };

        let cv = if median.abs() > T::Float::from_f64(1e-10).unwrap() {
            robust_std / median.abs()
        } else if robust_std < T::Float::from_f64(1e-10).unwrap() {
            T::Float::zero() // Both are essentially zero
        } else {
            T::Float::infinity() // Zero median with non-zero spread
        };

        // Determine stability
        let is_stable = window.len() >= self.params.min_samples
            && cv <= self.params.max_cv
            && trend_strength <= self.params.max_trend
            && cv.is_finite();

        let status = if window.len() < self.params.min_samples {
            StabilityStatus::Unknown
        } else if is_stable {
            StabilityStatus::Stable
        } else if trend_strength > self.params.max_trend {
            StabilityStatus::Unstable {
                reason: UnstableReason::Trend {
                    slope: trend_strength,
                },
                severity: (trend_strength / self.params.max_trend).min(T::Float::one()),
            }
        } else if cv > self.params.max_cv || !cv.is_finite() {
            StabilityStatus::Unstable {
                reason: UnstableReason::VarianceChange { rate: cv },
                severity: if cv.is_finite() {
                    (cv / self.params.max_cv).min(T::Float::one())
                } else {
                    T::Float::one()
                },
            }
        } else {
            StabilityStatus::Unstable {
                reason: UnstableReason::NonStationary,
                severity: T::Float::from_f64(0.5).unwrap(),
            }
        };

        WindowStabilityResult {
            is_stable,
            cv,
            trend_strength,
            mean,
            robust_std,
            status,
        }
    }

    fn min_window_size(&self) -> usize {
        self.params.min_samples
    }
}
