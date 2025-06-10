//! Common types used in stability analysis

use std::fmt;
use robust_core::Numeric;
use num_traits::{FromPrimitive, NumCast};

/// The stability status of a system
#[derive(Clone, PartialEq)]
pub enum StabilityStatus<T: Numeric> {
    /// System is stable and ready for measurement
    Stable,
    
    /// System is unstable with specific reason
    Unstable {
        reason: UnstableReason<T>,
        severity: T::Float, // 0.0 to 1.0
    },
    
    /// Stability cannot be determined (insufficient data)
    Unknown,
    
    /// System is transitioning (e.g., warming up)
    Transitioning {
        progress: T::Float, // 0.0 to 1.0
        estimated_samples_remaining: Option<usize>,
    },
}

impl<T: Numeric> fmt::Debug for StabilityStatus<T>
where
    T::Float: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StabilityStatus::Stable => write!(f, "Stable"),
            StabilityStatus::Unstable { reason, severity } => f.debug_struct("Unstable")
                .field("reason", reason)
                .field("severity", severity)
                .finish(),
            StabilityStatus::Unknown => write!(f, "Unknown"),
            StabilityStatus::Transitioning { progress, estimated_samples_remaining } => {
                f.debug_struct("Transitioning")
                    .field("progress", progress)
                    .field("estimated_samples_remaining", estimated_samples_remaining)
                    .finish()
            }
        }
    }
}

/// Reasons for instability
#[derive(Clone, PartialEq)]
pub enum UnstableReason<T: Numeric> {
    /// Significant trend detected
    Trend { slope: T::Float },
    
    /// Oscillations detected
    Oscillations { frequency: T::Float, amplitude: T::Float },
    
    /// Variance is changing over time
    VarianceChange { rate: T::Float },
    
    /// Multiple changepoints detected
    ChangePoints { count: usize },
    
    /// General non-stationarity
    NonStationary,
    
    /// Custom reason
    Custom(String),
}

impl<T: Numeric> fmt::Debug for UnstableReason<T>
where
    T::Float: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnstableReason::Trend { slope } => f.debug_struct("Trend")
                .field("slope", slope)
                .finish(),
            UnstableReason::Oscillations { frequency, amplitude } => f.debug_struct("Oscillations")
                .field("frequency", frequency)
                .field("amplitude", amplitude)
                .finish(),
            UnstableReason::VarianceChange { rate } => f.debug_struct("VarianceChange")
                .field("rate", rate)
                .finish(),
            UnstableReason::ChangePoints { count } => f.debug_struct("ChangePoints")
                .field("count", count)
                .finish(),
            UnstableReason::NonStationary => write!(f, "NonStationary"),
            UnstableReason::Custom(s) => write!(f, "Custom({:?})", s),
        }
    }
}

/// Metrics collected during stability analysis
#[derive(Clone)]
pub struct StabilityMetrics<T: Numeric> {
    /// Mean of the signal
    pub mean: T::Float,
    
    /// Standard deviation
    pub std_dev: T::Float,
    
    /// Coefficient of variation (CV)
    pub cv: T::Float,
    
    /// Trend strength (if detected)
    pub trend_strength: Option<T::Float>,
    
    /// Oscillation metrics (if detected)
    pub oscillation_metrics: Option<OscillationMetrics<T>>,
    
    /// Stationarity test results
    pub stationarity_tests: StationarityTests<T>,
    
    /// Number of samples analyzed
    pub sample_count: usize,
}

impl<T: Numeric> fmt::Debug for StabilityMetrics<T>
where
    T::Float: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StabilityMetrics")
            .field("mean", &self.mean)
            .field("std_dev", &self.std_dev)
            .field("cv", &self.cv)
            .field("trend_strength", &self.trend_strength)
            .field("oscillation_metrics", &self.oscillation_metrics)
            .field("stationarity_tests", &self.stationarity_tests)
            .field("sample_count", &self.sample_count)
            .finish()
    }
}

/// Metrics related to oscillatory behavior
#[derive(Clone)]
pub struct OscillationMetrics<T: Numeric> {
    /// Phase coherence (0.0 to 1.0)
    pub phase_coherence: T::Float,
    
    /// Amplitude stability (0.0 to 1.0)
    pub amplitude_stability: T::Float,
    
    /// Frequency stability (0.0 to 1.0)
    pub frequency_stability: T::Float,
    
    /// Dominant oscillation frequencies
    pub dominant_frequencies: Vec<DominantOscillation<T>>,
}

impl<T: Numeric> fmt::Debug for OscillationMetrics<T>
where
    T::Float: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OscillationMetrics")
            .field("phase_coherence", &self.phase_coherence)
            .field("amplitude_stability", &self.amplitude_stability)
            .field("frequency_stability", &self.frequency_stability)
            .field("dominant_frequencies", &self.dominant_frequencies)
            .finish()
    }
}

/// Information about a dominant oscillation
#[derive(Clone)]
pub struct DominantOscillation<T: Numeric> {
    /// Frequency in Hz (or samples^-1)
    pub frequency: T::Float,
    
    /// Power/amplitude of this oscillation
    pub power: T::Float,
    
    /// Prominence compared to background
    pub prominence: T::Float,
}

impl<T: Numeric> fmt::Debug for DominantOscillation<T>
where
    T::Float: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DominantOscillation")
            .field("frequency", &self.frequency)
            .field("power", &self.power)
            .field("prominence", &self.prominence)
            .finish()
    }
}

/// Results from various stationarity tests
#[derive(Clone)]
pub struct StationarityTests<T: Numeric> {
    /// Augmented Dickey-Fuller test p-value
    pub adf_pvalue: Option<T::Float>,
    
    /// KPSS test p-value
    pub kpss_pvalue: Option<T::Float>,
    
    /// Mann-Kendall trend test p-value
    pub mann_kendall_pvalue: Option<T::Float>,
    
    /// Ljung-Box test p-value
    pub ljung_box_pvalue: Option<T::Float>,
}

impl<T: Numeric> fmt::Debug for StationarityTests<T>
where
    T::Float: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StationarityTests")
            .field("adf_pvalue", &self.adf_pvalue)
            .field("kpss_pvalue", &self.kpss_pvalue)
            .field("mann_kendall_pvalue", &self.mann_kendall_pvalue)
            .field("ljung_box_pvalue", &self.ljung_box_pvalue)
            .finish()
    }
}

impl<T: Numeric> Default for StationarityTests<T> {
    fn default() -> Self {
        Self {
            adf_pvalue: None,
            kpss_pvalue: None,
            mann_kendall_pvalue: None,
            ljung_box_pvalue: None,
        }
    }
}

/// Parameters for stability analysis
#[derive(Clone, Copy)]
pub struct StabilityParameters<T: Numeric> {
    /// Minimum number of samples required
    pub min_samples: usize,
    
    /// Maximum coefficient of variation for stability
    pub max_cv: T::Float,
    
    /// Maximum trend strength for stability
    pub max_trend: T::Float,
    
    /// Threshold for oscillation detection
    pub oscillation_threshold: T::Float,
    
    /// Threshold for phase coherence
    pub phase_coherence_threshold: T::Float,
    
    /// Minimum consecutive stable windows
    pub min_stable_windows: usize,
    
    /// Significance level for statistical tests
    pub significance_level: T::Float,
}

impl<T: Numeric> fmt::Debug for StabilityParameters<T>
where
    T::Float: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StabilityParameters")
            .field("min_samples", &self.min_samples)
            .field("max_cv", &self.max_cv)
            .field("max_trend", &self.max_trend)
            .field("oscillation_threshold", &self.oscillation_threshold)
            .field("phase_coherence_threshold", &self.phase_coherence_threshold)
            .field("min_stable_windows", &self.min_stable_windows)
            .field("significance_level", &self.significance_level)
            .finish()
    }
}

impl<T: Numeric> Default for StabilityParameters<T>
where
    T::Float: FromPrimitive,
{
    fn default() -> Self {
        Self {
            min_samples: 50,
            max_cv: T::Float::from_f64(0.25).unwrap(),  // Slightly higher for robust CV
            max_trend: T::Float::from_f64(0.01).unwrap(),
            oscillation_threshold: T::Float::from_f64(0.1).unwrap(),
            phase_coherence_threshold: T::Float::from_f64(0.3).unwrap(),
            min_stable_windows: 5,
            significance_level: T::Float::from_f64(0.05).unwrap(),
        }
    }
}

impl<T: Numeric> StabilityParameters<T>
where
    T::Float: FromPrimitive,
{
    /// Create parameters for strict stability requirements
    pub fn strict() -> Self {
        Self {
            min_samples: 100,
            max_cv: T::Float::from_f64(0.1).unwrap(),
            max_trend: T::Float::from_f64(0.005).unwrap(),
            oscillation_threshold: T::Float::from_f64(0.05).unwrap(),
            phase_coherence_threshold: T::Float::from_f64(0.2).unwrap(),
            min_stable_windows: 10,
            significance_level: T::Float::from_f64(0.01).unwrap(),
        }
    }
    
    /// Create parameters for relaxed stability requirements
    pub fn relaxed() -> Self {
        Self {
            min_samples: 30,
            max_cv: T::Float::from_f64(0.3).unwrap(),
            max_trend: T::Float::from_f64(0.02).unwrap(),
            oscillation_threshold: T::Float::from_f64(0.2).unwrap(),
            phase_coherence_threshold: T::Float::from_f64(0.4).unwrap(),
            min_stable_windows: 3,
            significance_level: T::Float::from_f64(0.1).unwrap(),
        }
    }
}

impl<T: Numeric> fmt::Display for StabilityStatus<T>
where
    T::Float: fmt::Display + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StabilityStatus::Stable => write!(f, "Stable"),
            StabilityStatus::Unstable { reason, severity } => {
                write!(f, "Unstable (severity: {:.2}): {:?}", severity, reason)
            }
            StabilityStatus::Unknown => write!(f, "Unknown"),
            StabilityStatus::Transitioning { progress, .. } => {
                write!(f, "Transitioning ({:.0}% complete)", NumCast::from(progress.clone()).unwrap_or(0.0) * 100.0)
            }
        }
    }
}