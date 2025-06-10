//! Visualization interface for stability analysis
//!
//! This module defines the interface that visualizers must implement
//! to receive algorithm data from stability analyzers. Each analyzer
//! (Statistical, Hilbert, Online) can call these methods during analysis
//! to enable real-time visualization and debugging.

use crate::types::{
    OscillationMetrics, StabilityStatus, StationarityTests,
    DominantOscillation,
};
use robust_core::{Result, Numeric};

/// Trait for visualizing stability analysis algorithm stages
///
/// Stability analyzers call these methods during algorithm execution to
/// allow visualizers to record important stages and data for visualization.
///
/// # Example Implementation
///
/// ```rust,ignore
/// use robust_stability::visualization::{StabilityVisualizer, NullStabilityVisualizer};
/// use robust_stability::HilbertStabilityAnalyzer;
///
/// // Use null visualizer (default - no overhead)
/// let analyzer = HilbertStabilityAnalyzer::with_visualizer(NullStabilityVisualizer::<f64>::default());
///
/// // Or use a custom visualizer
/// struct MyVisualizer { /* ... */ }
/// impl StabilityVisualizer for MyVisualizer { /* ... */ }
/// 
/// let analyzer = HilbertStabilityAnalyzer::with_visualizer(MyVisualizer::new());
/// ```
pub trait StabilityVisualizer<T: Numeric>: Send + Sync {
    /// Record the raw input signal before analysis
    fn record_signal(&self, signal: &[T], label: &str) -> Result<()>;
    
    /// Record basic statistics (mean, median, spread, CV)
    fn record_statistics(
        &self,
        mean: T::Float,
        median: T::Float,
        std_dev: T::Float,
        robust_spread: T::Float,
        cv: T::Float,
        robust_cv: T::Float,
    ) -> Result<()>;
    
    /// Record stationarity test results
    fn record_stationarity_tests(&self, tests: &StationarityTests<T>) -> Result<()>;
    
    /// Record trend analysis results
    fn record_trend_analysis(
        &self,
        signal: &[T],
        trend_slope: T::Float,
        trend_r_squared: T::Float,
        mk_pvalue: T::Float,
    ) -> Result<()>;
    
    // Hilbert-specific visualizations
    
    /// Record analytic signal components (amplitude and phase)
    fn record_analytic_signal(
        &self,
        original: &[T],
        amplitude: &[T::Float],
        phase: &[T::Float],
    ) -> Result<()>;
    
    /// Record instantaneous frequency analysis
    fn record_instantaneous_frequency(
        &self,
        frequency: &[T::Float],
        median_freq: T::Float,
        freq_spread: T::Float,
        freq_cv: T::Float,
    ) -> Result<()>;
    
    /// Record phase coherence analysis
    fn record_phase_coherence(
        &self,
        coherence: T::Float,
        frequency_cv: T::Float,
    ) -> Result<()>;
    
    /// Record detected oscillations
    fn record_oscillations(
        &self,
        oscillations: &[DominantOscillation<T>],
    ) -> Result<()>;
    
    /// Record complete oscillation metrics
    fn record_oscillation_metrics(&self, metrics: &OscillationMetrics<T>) -> Result<()>;
    
    // Online-specific visualizations
    
    /// Record a single observation in online analysis
    fn record_online_observation(
        &self,
        observation_index: usize,
        value: T,
        current_mean: T::Float,
        current_variance: T::Float,
        current_cv: T::Float,
        status: &StabilityStatus<T>,
    ) -> Result<()>;
    
    /// Record sliding window analysis
    fn record_window_analysis(
        &self,
        window: &[T],
        window_cv: T::Float,
        window_trend: T::Float,
        is_stable: bool,
    ) -> Result<()>;
    
    /// Record current window for online analysis
    fn record_window(
        &self,
        window: &[T],
        observation_count: usize,
    ) -> Result<()>;
    
    /// Record incremental statistics for online analysis
    fn record_incremental_stats(
        &self,
        count: usize,
        value: T,
        mean: T::Float,
        std_dev: T::Float,
        cv: T::Float,
        trend_slope: T::Float,
    ) -> Result<()>;
    
    /// Record status transition for online analysis
    fn record_status_transition(
        &self,
        new_status: &StabilityStatus<T>,
        stable_count: usize,
        required_stable_windows: usize,
    ) -> Result<()>;
    
    /// Record statistical test results
    fn record_statistical_tests(
        &self,
        mk_z: T::Float,
        mk_pvalue: T::Float,
        levene_pvalue: T::Float,
        stationarity_score: T::Float,
        trend_slope: T::Float,
        trend_r2: T::Float,
    ) -> Result<()>;
    
    // Final results
    
    /// Record the final stability decision
    fn record_final_decision(
        &self,
        is_stable: bool,
        cv: T::Float,
        confidence: T::Float,
        explanation: &str,
    ) -> Result<()>;
    
    /// Save all recorded visualizations to files
    ///
    /// The path_prefix will be used to generate multiple output files
    /// (e.g., "stability_analysis" -> "stability_analysis_signal.html", etc.)
    fn save_visualizations(&self, path_prefix: &str) -> Result<()>;
    
    /// Check if this visualizer actually produces output
    fn is_enabled(&self) -> bool {
        true
    }
    
    /// Update the visualizer with a new observation (for online analyzers)
    fn update(&mut self, value: T, status: &StabilityStatus<T>) -> () {
        // Default implementation does nothing
    }
    
    /// Reset the visualizer state
    fn reset(&mut self) -> () {
        // Default implementation does nothing
    }
}

/// Null visualizer that does nothing (for when visualization is disabled)
///
/// This is the default visualizer used when no specific visualizer is provided.
/// All methods are no-ops that compile away, ensuring zero overhead.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullStabilityVisualizer<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> NullStabilityVisualizer<T> {
    /// Create a new null stability visualizer
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Numeric> StabilityVisualizer<T> for NullStabilityVisualizer<T> {
    #[inline(always)]
    fn record_signal(&self, _signal: &[T], _label: &str) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn record_statistics(
        &self,
        _mean: T::Float,
        _median: T::Float,
        _std_dev: T::Float,
        _robust_spread: T::Float,
        _cv: T::Float,
        _robust_cv: T::Float,
    ) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn record_stationarity_tests(&self, _tests: &StationarityTests<T>) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn record_trend_analysis(
        &self,
        _signal: &[T],
        _trend_slope: T::Float,
        _trend_r_squared: T::Float,
        _mk_pvalue: T::Float,
    ) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn record_analytic_signal(
        &self,
        _original: &[T],
        _amplitude: &[T::Float],
        _phase: &[T::Float],
    ) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn record_instantaneous_frequency(
        &self,
        _frequency: &[T::Float],
        _median_freq: T::Float,
        _freq_spread: T::Float,
        _freq_cv: T::Float,
    ) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn record_phase_coherence(
        &self,
        _coherence: T::Float,
        _frequency_cv: T::Float,
    ) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn record_oscillations(
        &self,
        _oscillations: &[DominantOscillation<T>],
    ) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn record_oscillation_metrics(&self, _metrics: &OscillationMetrics<T>) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn record_online_observation(
        &self,
        _observation_index: usize,
        _value: T,
        _current_mean: T::Float,
        _current_variance: T::Float,
        _current_cv: T::Float,
        _status: &StabilityStatus<T>,
    ) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn record_window_analysis(
        &self,
        _window: &[T],
        _window_cv: T::Float,
        _window_trend: T::Float,
        _is_stable: bool,
    ) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn record_window(
        &self,
        _window: &[T],
        _observation_count: usize,
    ) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn record_incremental_stats(
        &self,
        _count: usize,
        _value: T,
        _mean: T::Float,
        _std_dev: T::Float,
        _cv: T::Float,
        _trend_slope: T::Float,
    ) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn record_status_transition(
        &self,
        _new_status: &StabilityStatus<T>,
        _stable_count: usize,
        _required_stable_windows: usize,
    ) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn record_statistical_tests(
        &self,
        _mk_z: T::Float,
        _mk_pvalue: T::Float,
        _levene_pvalue: T::Float,
        _stationarity_score: T::Float,
        _trend_slope: T::Float,
        _trend_r2: T::Float,
    ) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn record_final_decision(
        &self,
        _is_stable: bool,
        _cv: T::Float,
        _confidence: T::Float,
        _explanation: &str,
    ) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn save_visualizations(&self, _path_prefix: &str) -> Result<()> {
        Ok(())
    }
    
    #[inline(always)]
    fn is_enabled(&self) -> bool {
        false
    }
}