//! Visualization interface for modality detection
//!
//! This module defines the interface that visualizers must implement
//! to receive algorithm data from the modality detector.

use crate::types::Mode;
use robust_core::{Result, Numeric};
use robust_histogram::Histogram;

/// Trait for visualizing modality detection algorithm stages
///
/// The detector calls these methods during algorithm execution to
/// allow visualizers to record important stages and data.
pub trait ModalityVisualizer<T: Numeric = f64> 
where
    T::Float: std::fmt::Debug + PartialEq + num_traits::ToPrimitive,
{
    /// Record the initial histogram before analysis begins
    fn record_histogram(&self, histogram: &Histogram<T>) -> Result<()>;

    /// Record detected peaks in the histogram
    fn record_peaks(&self, histogram: &Histogram<T>, peak_indices: &[usize]) -> Result<()>;

    /// Record a water level analysis attempt between two peaks
    ///
    /// This is called during the try_split algorithm to show the water level
    /// concept used to determine if there's a valid lowland between peaks.
    fn record_water_level_attempt(
        &self,
        histogram: &Histogram<T>,
        peak1_idx: usize,
        peak2_idx: usize,
        water_level: T::Float,
        underwater_left: usize,
        underwater_right: usize,
        is_lowland: bool,
    ) -> Result<()>;

    /// Record the final detected modes
    fn record_final_modes(
        &self,
        histogram: &Histogram<T>,
        modes: &[Mode<T>],
        lowland_indices: &[usize],
    ) -> Result<()>;

    /// Save all recorded visualizations to files
    ///
    /// The path_prefix will be used to generate multiple output files
    /// (e.g., "analysis" -> "analysis_histogram.html", "analysis_peaks.html", etc.)
    fn save_visualizations(&self, path_prefix: &str) -> Result<()>;

    /// Check if this visualizer actually produces output
    fn is_enabled(&self) -> bool {
        true
    }
}

/// Null visualizer that does nothing (for when visualization is disabled)
#[derive(Default)]
pub struct NullModalityVisualizer;

impl<T: Numeric> ModalityVisualizer<T> for NullModalityVisualizer 
where
    T::Float: std::fmt::Debug + PartialEq + num_traits::ToPrimitive,
{
    fn record_histogram(&self, _histogram: &Histogram<T>) -> Result<()> {
        Ok(())
    }

    fn record_peaks(&self, _histogram: &Histogram<T>, _peak_indices: &[usize]) -> Result<()> {
        Ok(())
    }

    fn record_water_level_attempt(
        &self,
        _histogram: &Histogram<T>,
        _peak1_idx: usize,
        _peak2_idx: usize,
        _water_level: T::Float,
        _underwater_left: usize,
        _underwater_right: usize,
        _is_lowland: bool,
    ) -> Result<()> {
        Ok(())
    }

    fn record_final_modes(
        &self,
        _histogram: &Histogram<T>,
        _modes: &[Mode<T>],
        _lowland_indices: &[usize],
    ) -> Result<()> {
        Ok(())
    }

    fn save_visualizations(&self, _path_prefix: &str) -> Result<()> {
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        false
    }
}
