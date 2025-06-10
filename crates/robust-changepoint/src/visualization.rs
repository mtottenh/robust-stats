//! Visualization interface for changepoint detection algorithms
//!
//! This module defines traits that allow changepoint detection algorithms to 
//! provide visualization hooks without depending on specific visualization libraries.
//! The design uses compile-time generics for zero-cost abstraction when visualization
//! is disabled.

use crate::types::ChangePointResult;
use robust_core::Result;

/// Trait for visualizing changepoint detection algorithm stages
///
/// This trait provides hooks at key points during changepoint detection,
/// allowing visualizers to record and display algorithm-specific data.
/// The trait is designed to be generic enough for all changepoint algorithms
/// while allowing algorithm-specific extensions.
pub trait ChangePointVisualizer {
    /// Record the input data before analysis begins
    fn record_data(&mut self, data: &[f64], sample_rate: Option<f64>) -> Result<()>;
    
    /// Record algorithm-specific detection scores/statistics
    /// 
    /// This is intentionally generic to support different algorithms:
    /// - CUSUM: cumulative sums
    /// - Polynomial slopes: slope values
    /// - EWMA: exponentially weighted statistics
    /// - Variance-based: running variance estimates
    fn record_detection_scores(
        &mut self,
        scores: &[f64],
        score_name: &str,
        score_description: Option<&str>,
    ) -> Result<()>;
    
    /// Record threshold information for the algorithm
    fn record_thresholds(
        &mut self,
        thresholds: &[(&str, f64)],  // (name, value) pairs
    ) -> Result<()>;
    
    /// Record intermediate algorithm state (optional)
    /// 
    /// This allows algorithms to provide additional diagnostic data.
    /// The data is provided as key-value pairs where values are f64 arrays.
    fn record_algorithm_state(
        &mut self,
        state_name: &str,
        state_data: &[(&str, &[f64])],
    ) -> Result<()>;
    
    /// Record a potential changepoint candidate during detection
    /// 
    /// Called when the algorithm identifies a potential changepoint,
    /// even if it might be filtered out later.
    fn record_candidate(
        &mut self,
        index: usize,
        score: f64,
        metadata: Option<&[(&str, f64)]>,
    ) -> Result<()>;
    
    /// Record the final detected changepoints
    fn record_final_changepoints(
        &mut self,
        result: &ChangePointResult,
    ) -> Result<()>;
    
    /// Generate and save visualizations
    /// 
    /// Returns paths to generated files (if any)
    fn save_visualizations(&self, output_prefix: &str) -> Result<Vec<String>>;
    
    /// Check if this visualizer is active
    fn is_enabled(&self) -> bool {
        true
    }
}

/// Extended trait for polynomial-based algorithms
pub trait PolynomialVisualizer: ChangePointVisualizer {
    /// Record polynomial fit results for each window
    fn record_polynomial_fits(
        &mut self,
        window_starts: &[usize],
        window_size: usize,
        coefficients: &[&[f64]],
        fit_quality: &[f64],  // e.g., R² values
    ) -> Result<()>;
    
    /// Record window analysis details
    fn record_window_analysis(
        &mut self,
        window_center: usize,
        data_slice: &[f64],
        fit_coefficients: &[f64],
        slope: f64,
        rmse: f64,
        r_squared: f64,
    ) -> Result<()>;
}

/// Null visualizer that performs no operations (zero-cost abstraction)
///
/// This visualizer compiles to no-ops and should have zero runtime overhead
/// when used with optimizations enabled. All methods are marked `#[inline(always)]`
/// to ensure they are eliminated during compilation.
#[derive(Default, Clone, Copy, Debug)]
pub struct NullChangePointVisualizer;

impl ChangePointVisualizer for NullChangePointVisualizer {
    #[inline(always)]
    fn record_data(&mut self, _: &[f64], _: Option<f64>) -> Result<()> { 
        Ok(()) 
    }
    
    #[inline(always)]
    fn record_detection_scores(&mut self, _: &[f64], _: &str, _: Option<&str>) -> Result<()> { 
        Ok(()) 
    }
    
    #[inline(always)]
    fn record_thresholds(&mut self, _: &[(&str, f64)]) -> Result<()> { 
        Ok(()) 
    }
    
    #[inline(always)]
    fn record_algorithm_state(&mut self, _: &str, _: &[(&str, &[f64])]) -> Result<()> { 
        Ok(()) 
    }
    
    #[inline(always)]
    fn record_candidate(&mut self, _: usize, _: f64, _: Option<&[(&str, f64)]>) -> Result<()> { 
        Ok(()) 
    }
    
    #[inline(always)]
    fn record_final_changepoints(&mut self, _: &ChangePointResult) -> Result<()> { 
        Ok(()) 
    }
    
    #[inline(always)]
    fn save_visualizations(&self, _: &str) -> Result<Vec<String>> { 
        Ok(vec![]) 
    }
    
    #[inline(always)]
    fn is_enabled(&self) -> bool { 
        false 
    }
}

// Blanket implementation for the null visualizer
impl PolynomialVisualizer for NullChangePointVisualizer {
    #[inline(always)]
    fn record_polynomial_fits(&mut self, _: &[usize], _: usize, _: &[&[f64]], _: &[f64]) -> Result<()> { 
        Ok(()) 
    }
    
    #[inline(always)]
    fn record_window_analysis(&mut self, _: usize, _: &[f64], _: &[f64], _: f64, _: f64, _: f64) -> Result<()> { 
        Ok(()) 
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_null_visualizer_zero_cost() {
        let mut viz = NullChangePointVisualizer;
        
        // All these operations should compile to no-ops
        let _ = viz.record_data(&[1.0, 2.0, 3.0], None);
        let _ = viz.record_detection_scores(&[0.1, 0.2], "test", None);
        let _ = viz.record_thresholds(&[("threshold", 0.5)]);
        let _ = viz.record_algorithm_state("state", &[("data", &[1.0, 2.0])]);
        let _ = viz.record_candidate(10, 0.8, None);
        
        // Test that it reports as disabled
        assert!(!viz.is_enabled());
        
        // Test save returns empty
        let paths = viz.save_visualizations("test").unwrap();
        assert!(paths.is_empty());
    }
}