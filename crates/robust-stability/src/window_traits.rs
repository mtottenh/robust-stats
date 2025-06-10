//! Core traits for window-based stability analysis
//!
//! These traits enable a generic, zero-allocation approach to stability analysis
//! that can work with both traditional buffers and Polars' native windowing.

use crate::types::StabilityStatus;
use robust_core::Numeric;

/// Core trait for stability analysis on a single window
pub trait WindowStabilityAnalyzer<T: Numeric>: Send + Sync {
    /// Analyze a single window of data
    fn analyze_window(&self, window: &[T]) -> WindowStabilityResult<T>;
    
    /// Minimum window size required for analysis
    fn min_window_size(&self) -> usize;
}

/// Result from analyzing a single window
#[derive(Clone)]
pub struct WindowStabilityResult<T: Numeric> {
    /// Whether this window is considered stable
    pub is_stable: bool,
    
    /// Coefficient of variation (robust std / median)
    pub cv: T::Float,
    
    /// Strength of any trend in the window (0.0 = no trend)
    pub trend_strength: T::Float,
    
    /// Mean of the window
    pub mean: T::Float,
    
    /// Robust standard deviation (e.g., MAD-based)
    pub robust_std: T::Float,
    
    /// Detailed stability status
    pub status: StabilityStatus<T>,
}

impl<T: Numeric> std::fmt::Debug for WindowStabilityResult<T>
where
    T::Float: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowStabilityResult")
            .field("is_stable", &self.is_stable)
            .field("cv", &self.cv)
            .field("trend_strength", &self.trend_strength)
            .field("mean", &self.mean)
            .field("robust_std", &self.robust_std)
            .field("status", &self.status)
            .finish()
    }
}

/// Trait for incremental stability tracking across windows
pub trait StabilityTracker<T: Numeric>: Send + Sync {
    /// Update tracker with a new window result
    fn update(&mut self, window_result: &WindowStabilityResult<T>);
    
    /// Get current overall stability status
    fn current_status(&self) -> StabilityStatus<T>;
    
    /// Reset the tracker to initial state
    fn reset(&mut self);
    
    /// Get the window index when stability was first achieved (if any)
    fn stability_index(&self) -> Option<usize>;
}