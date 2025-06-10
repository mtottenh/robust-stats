//! Online (real-time) stability detection

use crate::traits::OnlineStabilityAnalyzer;
use crate::types::{
    StabilityMetrics, StabilityParameters, StabilityStatus, StationarityTests, UnstableReason,
};
use crate::visualization::{NullStabilityVisualizer, StabilityVisualizer};
use num_traits::{Float, FromPrimitive, One, Zero};
use robust_core::{ExecutionEngine, Numeric};
use robust_quantile::{estimators::trimmed_harrell_davis, QuantileEstimator, SqrtWidth};
use robust_spread::{Mad, SpreadEstimator};
use std::collections::VecDeque;
use std::marker::PhantomData;

/// Online stability detector that processes data incrementally
///
/// This detector maintains a sliding window of observations and uses
/// Welford's algorithm for efficient incremental statistics calculation.
/// It detects trends, variance changes, and transitions to stability.
///
/// # Type Parameters
///
/// - `T`: The numeric type
/// - `V`: The visualizer type that implements `StabilityVisualizer`
pub struct OnlineStabilityDetector<
    T: Numeric,
    V: StabilityVisualizer<T> = NullStabilityVisualizer<T>,
> {
    params: StabilityParameters<T>,
    buffer: VecDeque<T>,
    stable_count: usize,
    current_status: StabilityStatus<T>,

    // Incremental statistics
    count: usize,
    mean: T::Float,
    m2: T::Float, // For Welford's algorithm

    // Trend detection
    sum_x: T::Float,
    sum_y: T::Float,
    sum_xy: T::Float,
    sum_x2: T::Float,

    visualizer: V,
    _phantom: PhantomData<(T, V)>,
}

impl<T: Numeric> OnlineStabilityDetector<T, NullStabilityVisualizer<T>>
where
    T::Float: FromPrimitive,
{
    /// Create a new online stability detector with null visualizer
    pub fn new(params: StabilityParameters<T>) -> Self {
        Self {
            buffer: VecDeque::with_capacity(params.min_samples * 2),
            stable_count: 0,
            current_status: StabilityStatus::Unknown,
            count: 0,
            mean: T::Float::zero(),
            m2: T::Float::zero(),
            sum_x: T::Float::zero(),
            sum_y: T::Float::zero(),
            sum_xy: T::Float::zero(),
            sum_x2: T::Float::zero(),
            params,
            visualizer: NullStabilityVisualizer::new(),
            _phantom: PhantomData,
        }
    }

    /// Create with default parameters and null visualizer
    pub fn default() -> Self {
        Self::new(StabilityParameters::<T>::default())
    }
}

impl<T: Numeric, V: StabilityVisualizer<T>> OnlineStabilityDetector<T, V>
where
    T::Float: FromPrimitive,
{
    /// Create a new online stability detector with custom visualizer
    pub fn with_visualizer(visualizer: V, params: StabilityParameters<T>) -> Self {
        Self {
            buffer: VecDeque::with_capacity(params.min_samples * 2),
            stable_count: 0,
            current_status: StabilityStatus::Unknown,
            count: 0,
            mean: T::Float::zero(),
            m2: T::Float::zero(),
            sum_x: T::Float::zero(),
            sum_y: T::Float::zero(),
            sum_xy: T::Float::zero(),
            sum_x2: T::Float::zero(),
            params,
            visualizer,
            _phantom: PhantomData,
        }
    }

    /// Update incremental statistics using Welford's algorithm
    fn update_statistics(&mut self, value: T) {
        self.count += 1;
        let value_float = value.to_float();
        let delta = value_float - self.mean;
        self.mean = self.mean + delta / T::Float::from_f64(self.count as f64).unwrap();
        let delta2 = value_float - self.mean;
        self.m2 = self.m2 + delta * delta2;

        // Update trend statistics
        let x = T::Float::from_f64(self.count as f64).unwrap();
        self.sum_x = self.sum_x + x;
        self.sum_y = self.sum_y + value_float;
        self.sum_xy = self.sum_xy + x * value_float;
        self.sum_x2 = self.sum_x2 + x * x;
    }

    /// Check if current buffer is stable
    fn check_stability(&mut self) -> bool {
        if self.buffer.len() < self.params.min_samples {
            return false;
        }

        // Calculate coefficient of variation using incremental statistics
        let variance = if self.count > 1 {
            self.m2 / T::Float::from_f64((self.count - 1) as f64).unwrap()
        } else {
            T::Float::zero()
        };

        let std_dev = variance.sqrt();
        let cv = if self.mean.abs() > T::Float::from_f64(1e-10).unwrap() {
            std_dev / self.mean.abs()
        } else if std_dev < T::Float::from_f64(1e-10).unwrap() {
            T::Float::zero()
        } else {
            T::Float::infinity()
        };

        // Calculate trend using incremental statistics
        let n = T::Float::from_f64(self.count as f64).unwrap();
        let trend = if self.count > 1 {
            let denom = n * self.sum_x2 - self.sum_x * self.sum_x;
            if denom > T::Float::zero() {
                ((n * self.sum_xy - self.sum_x * self.sum_y) / denom).abs()
            } else {
                T::Float::zero()
            }
        } else {
            T::Float::zero()
        };

        cv <= self.params.max_cv && trend <= self.params.max_trend && cv.is_finite()
    }

    /// Update status based on recent observations
    fn update_status(&mut self, is_stable: bool) {
        if is_stable {
            self.stable_count += 1;

            if self.stable_count >= self.params.min_stable_windows {
                self.current_status = StabilityStatus::Stable;
            } else {
                let progress = T::Float::from_f64(self.stable_count as f64).unwrap()
                    / T::Float::from_f64(self.params.min_stable_windows as f64).unwrap();
                let samples_remaining =
                    (self.params.min_stable_windows - self.stable_count) * self.params.min_samples;
                self.current_status = StabilityStatus::Transitioning {
                    progress,
                    estimated_samples_remaining: Some(samples_remaining),
                };
            }
        } else {
            self.stable_count = 0;

            // Determine reason for instability
            let variance = if self.count > 1 {
                self.m2 / T::Float::from_f64((self.count - 1) as f64).unwrap()
            } else {
                T::Float::zero()
            };

            let std_dev = variance.sqrt();
            let cv = if self.mean.abs() > T::Float::from_f64(1e-10).unwrap() {
                std_dev / self.mean.abs()
            } else {
                T::Float::infinity()
            };

            let n = T::Float::from_f64(self.count as f64).unwrap();
            let trend = if self.count > 1 {
                let denom = n * self.sum_x2 - self.sum_x * self.sum_x;
                if denom > T::Float::zero() {
                    ((n * self.sum_xy - self.sum_x * self.sum_y) / denom).abs()
                } else {
                    T::Float::zero()
                }
            } else {
                T::Float::zero()
            };

            if trend > self.params.max_trend {
                self.current_status = StabilityStatus::Unstable {
                    reason: UnstableReason::Trend { slope: trend },
                    severity: if trend / self.params.max_trend < T::Float::one() {
                        trend / self.params.max_trend
                    } else {
                        T::Float::one()
                    },
                };
            } else if cv > self.params.max_cv || !cv.is_finite() {
                self.current_status = StabilityStatus::Unstable {
                    reason: UnstableReason::VarianceChange { rate: cv },
                    severity: if cv.is_finite() {
                        if cv / self.params.max_cv < T::Float::one() {
                            cv / self.params.max_cv
                        } else {
                            T::Float::one()
                        }
                    } else {
                        T::Float::one()
                    },
                };
            } else {
                self.current_status = StabilityStatus::Unknown;
            }
        }
    }
}

impl<T: Numeric, V: StabilityVisualizer<T>> OnlineStabilityAnalyzer<T>
    for OnlineStabilityDetector<T, V>
where
    T::Float: FromPrimitive,
{
    fn add_observation(&mut self, value: T) -> StabilityStatus<T> {
        // Add to buffer
        self.buffer.push_back(value);

        // Remove old values if buffer is too large
        while self.buffer.len() > self.params.min_samples * 2 {
            self.buffer.pop_front();
        }

        // Update statistics
        self.update_statistics(value);

        // Check stability
        let is_stable = self.check_stability();
        self.update_status(is_stable);

        // Update visualizer
        self.visualizer.update(value, &self.current_status);

        self.current_status.clone()
    }

    fn current_status(&self) -> StabilityStatus<T> {
        self.current_status.clone()
    }

    fn current_metrics(&self) -> StabilityMetrics<T> {
        let variance = if self.count > 1 {
            self.m2 / T::Float::from_f64((self.count - 1) as f64).unwrap()
        } else {
            T::Float::zero()
        };

        let std_dev = variance.sqrt();
        let cv = if self.mean.abs() > T::Float::from_f64(1e-10).unwrap() {
            std_dev / self.mean.abs()
        } else {
            T::Float::zero()
        };

        let n = T::Float::from_f64(self.count as f64).unwrap();
        let trend = if self.count > 1 {
            let denom = n * self.sum_x2 - self.sum_x * self.sum_x;
            if denom > T::Float::zero() {
                Some(((n * self.sum_xy - self.sum_x * self.sum_y) / denom).abs())
            } else {
                None
            }
        } else {
            None
        };

        let std_dev = variance.sqrt();

        StabilityMetrics {
            mean: self.mean,
            std_dev,
            cv,
            trend_strength: trend,
            oscillation_metrics: None,
            stationarity_tests: StationarityTests::default(),
            sample_count: self.buffer.len(),
        }
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.stable_count = 0;
        self.current_status = StabilityStatus::Unknown;
        self.count = 0;
        self.mean = T::Float::zero();
        self.m2 = T::Float::zero();
        self.sum_x = T::Float::zero();
        self.sum_y = T::Float::zero();
        self.sum_xy = T::Float::zero();
        self.sum_x2 = T::Float::zero();
        self.visualizer.reset();
    }
}
