//! Unit tests for window-based stability analysis
//! 
//! These tests verify the core window analysis functionality without
//! requiring the full Polars integration.

use robust_stability::*;

// Helper functions to generate test signals
fn generate_stable_signal(n: usize) -> Vec<f64> {
    (0..n).map(|_| 5.0 + 0.1 * rand::random::<f64>()).collect()
}

fn generate_trending_signal(n: usize, slope: f64) -> Vec<f64> {
    (0..n).map(|i| i as f64 * slope + 5.0 + 0.05 * rand::random::<f64>()).collect()
}

fn generate_oscillating_signal(n: usize, freq: f64, amp: f64) -> Vec<f64> {
    use std::f64::consts::PI;
    (0..n).map(|i| {
        5.0 + amp * (2.0 * PI * freq * i as f64).sin() + 0.05 * rand::random::<f64>()
    }).collect()
}

fn generate_high_variance_signal(n: usize, variance_factor: f64) -> Vec<f64> {
    (0..n).map(|_| 5.0 + variance_factor * (rand::random::<f64>() - 0.5)).collect()
}

#[cfg(test)]
mod window_analyzer_tests {
    use super::*;
    use robust_stability::window_traits::WindowStabilityAnalyzer;
    use robust_stability::algorithms::{StatisticalWindowAnalyzer, DefaultStatisticalAnalyzer};
    use robust_stability::types::{StabilityParameters, StabilityStatus, UnstableReason};
    use robust_quantile::estimators::harrell_davis;
    use robust_spread::Mad;
    use robust_core::{execution::scalar_sequential, UnifiedWeightCache, CachePolicy, ExecutionEngine};
    
    #[test]
    fn test_stable_window_detection() {
        let params = StabilityParameters::<f64>::default();
        let engine = scalar_sequential();
        let primitives = engine.primitives().clone();
        let quantile_est = harrell_davis(engine);
        let spread_est = Mad::new(primitives.clone());
        let cache = UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            CachePolicy::NoCache
        );
        let analyzer = StatisticalWindowAnalyzer::new(params, spread_est, quantile_est, cache);
        
        // Generate a stable signal
        let window = generate_stable_signal(100);
        let result = analyzer.analyze_window(&window);
        
        assert!(result.is_stable, "Stable signal should be detected as stable");
        assert!(result.cv < 0.1, "CV should be low for stable signal");
        assert!(result.trend_strength < 0.01, "Trend should be minimal");
        assert!(matches!(result.status, StabilityStatus::<f64>::Stable));
    }
    
    #[test]
    fn test_trending_window_detection() {
        let params = StabilityParameters::<f64>::default();
        let engine = scalar_sequential();
        let primitives = engine.primitives().clone();
        let quantile_est = harrell_davis(engine);
        let spread_est = Mad::new(primitives.clone());
        let cache = UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            CachePolicy::NoCache
        );
        let analyzer = StatisticalWindowAnalyzer::new(params, spread_est, quantile_est, cache);
        
        // Generate trending signal
        let window = generate_trending_signal(100, 0.05);
        let result = analyzer.analyze_window(&window);
        
        assert!(!result.is_stable, "Trending signal should not be stable");
        assert!(result.trend_strength > params.max_trend, "Trend should be detected");
        
        if let StabilityStatus::<f64>::Unstable { reason, .. } = &result.status {
            assert!(matches!(reason, UnstableReason::<f64>::Trend { .. }));
        } else {
            panic!("Expected unstable status with trend reason");
        }
    }
    
    #[test]
    fn test_high_variance_window_detection() {
        let params = StabilityParameters::<f64>::default();
        let engine = scalar_sequential();
        let primitives = engine.primitives().clone();
        let quantile_est = harrell_davis(engine);
        let spread_est = Mad::new(primitives.clone());
        let cache = UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            CachePolicy::NoCache
        );
        let analyzer = StatisticalWindowAnalyzer::new(params, spread_est, quantile_est, cache);
        
        // Generate high variance signal - need more variance to exceed 0.25 CV threshold
        let window = generate_high_variance_signal(100, 5.0);
        let result = analyzer.analyze_window(&window);
        
        println!("High variance test:");
        println!("  CV: {}, max_cv: {}", result.cv, params.max_cv);
        println!("  Trend: {}, max_trend: {}", result.trend_strength, params.max_trend);
        println!("  Is stable: {}", result.is_stable);
        println!("  Mean: {}, Robust STD: {}", result.mean, result.robust_std);
        
        assert!(!result.is_stable, "High variance signal should not be stable");
        assert!(result.cv > params.max_cv, "CV should exceed threshold: {} <= {}", result.cv, params.max_cv);
        
        if let StabilityStatus::<f64>::Unstable { reason, .. } = &result.status {
            assert!(matches!(reason, UnstableReason::<f64>::VarianceChange { .. }));
        } else {
            panic!("Expected unstable status with variance change reason");
        }
    }
    
    #[test]
    fn test_insufficient_data_window() {
        let params = StabilityParameters::<f64> {
            min_samples: 50,
            ..Default::default()
        };
        let engine = scalar_sequential();
        let primitives = engine.primitives().clone();
        let quantile_est = harrell_davis(engine);
        let spread_est = Mad::new(primitives.clone());
        let cache = UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            CachePolicy::NoCache
        );
        let analyzer = StatisticalWindowAnalyzer::new(params, spread_est, quantile_est, cache);
        
        // Window smaller than min_samples
        let window = generate_stable_signal(30);
        let result = analyzer.analyze_window(&window);
        
        assert!(!result.is_stable, "Small window should not be stable");
        assert!(matches!(result.status, StabilityStatus::<f64>::Unknown));
    }
    
    #[test]
    fn test_window_analyzer_min_size() {
        let params = StabilityParameters::<f64> {
            min_samples: 75,
            ..Default::default()
        };
        let engine = scalar_sequential();
        let primitives = engine.primitives().clone();
        let quantile_est = harrell_davis(engine);
        let spread_est = Mad::new(primitives.clone());
        let cache = UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            CachePolicy::NoCache
        );
        let analyzer = StatisticalWindowAnalyzer::new(params, spread_est, quantile_est, cache);
        
        assert_eq!(analyzer.min_window_size(), 75);
    }
    
    #[test]
    fn test_zero_mean_stability() {
        let params = StabilityParameters::<f64>::default();
        let engine = scalar_sequential();
        let primitives = engine.primitives().clone();
        let quantile_est = harrell_davis(engine);
        let spread_est = Mad::new(primitives.clone());
        let cache = UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            CachePolicy::NoCache
        );
        let analyzer = StatisticalWindowAnalyzer::new(params, spread_est, quantile_est, cache);
        
        // Signal centered around zero
        let window: Vec<f64> = (0..100).map(|_| 0.1 * (rand::random::<f64>() - 0.5)).collect();
        let result = analyzer.analyze_window(&window);
        
        // Should handle zero-mean signals correctly
        assert!(result.mean.abs() < 0.1);
        assert!(!result.cv.is_nan() && !result.cv.is_infinite());
    }
    
    #[test]
    fn test_constant_signal_stability() {
        let params = StabilityParameters::<f64>::default();
        let engine = scalar_sequential();
        let primitives = engine.primitives().clone();
        let quantile_est = harrell_davis(engine);
        let spread_est = Mad::new(primitives.clone());
        let cache = UnifiedWeightCache::new(
            robust_quantile::HDWeightComputer::new(),
            CachePolicy::NoCache
        );
        let analyzer = StatisticalWindowAnalyzer::new(params, spread_est, quantile_est, cache);
        
        // Perfectly constant signal
        let window = vec![5.0; 100];
        let result = analyzer.analyze_window(&window);
        
        assert!(result.is_stable);
        assert_eq!(result.cv, 0.0);
        assert_eq!(result.trend_strength, 0.0);
    }
    
    #[test]
    fn test_custom_estimators() {
        let params = StabilityParameters::<f64>::default();
        
        // Use simple quantile estimator and basic MAD
        let engine = robust_core::simd_sequential();
        let primitives = engine.primitives().clone();
        let analyzer = StatisticalWindowAnalyzer::new(
            params,
            Mad::new(primitives),
            harrell_davis(engine),
            UnifiedWeightCache::new(
                robust_quantile::HDWeightComputer::new(),
                CachePolicy::NoCache
            )
        );
        
        let window = generate_stable_signal(100);
        let result = analyzer.analyze_window(&window);
        
        assert!(result.is_stable, "Should detect stability with custom estimators");
        
        // Compare with default estimators
        let default_analyzer = DefaultStatisticalAnalyzer::with_default_estimators(params);
        let default_result = default_analyzer.analyze_window(&window);
        
        // Results should be similar but not necessarily identical
        assert_eq!(result.is_stable, default_result.is_stable);
        assert!((result.cv - default_result.cv).abs() < 0.1);
    }
}

#[cfg(test)]
mod stability_tracker_tests {
    use super::*;
    use robust_stability::window_traits::{StabilityTracker, WindowStabilityResult};
    use robust_stability::algorithms::ConsecutiveWindowTracker;
    use robust_stability::types::{StabilityStatus, UnstableReason};
    
    fn create_stable_result() -> WindowStabilityResult<f64> {
        WindowStabilityResult {
            is_stable: true,
            cv: 0.05,
            trend_strength: 0.001,
            mean: 5.0,
            robust_std: 0.25,
            status: StabilityStatus::<f64>::Stable,
        }
    }
    
    fn create_unstable_result() -> WindowStabilityResult<f64> {
        WindowStabilityResult {
            is_stable: false,
            cv: 0.35,
            trend_strength: 0.02,
            mean: 5.0,
            robust_std: 1.75,
            status: StabilityStatus::<f64>::Unstable {
                reason: UnstableReason::<f64>::VarianceChange { rate: 0.35 },
                severity: 0.7,
            },
        }
    }
    
    #[test]
    fn test_tracker_initialization() {
        let params = StabilityParameters::<f64>::default();
        let tracker = ConsecutiveWindowTracker::new(params);
        
        assert!(matches!(tracker.current_status(), StabilityStatus::<f64>::Unknown));
        assert_eq!(tracker.stability_index(), None);
    }
    
    #[test]
    fn test_tracker_consecutive_stable_windows() {
        let params = StabilityParameters::<f64> {
            min_stable_windows: 3,
            ..Default::default()
        };
        let mut tracker = ConsecutiveWindowTracker::new(params);
        
        // Add stable windows
        for i in 0..3 {
            tracker.update(&create_stable_result());
            
            if i < 2 {
                // Not enough consecutive windows yet
                assert!(tracker.stability_index().is_none());
            } else {
                // Should achieve stability after 3 windows
                assert_eq!(tracker.stability_index(), Some(3));
                assert!(matches!(tracker.current_status(), StabilityStatus::<f64>::Stable));
            }
        }
    }
    
    #[test]
    fn test_tracker_stability_reset_on_unstable() {
        let params = StabilityParameters::<f64> {
            min_stable_windows: 2,
            ..Default::default()
        };
        let mut tracker = ConsecutiveWindowTracker::new(params);
        
        // Achieve stability
        tracker.update(&create_stable_result());
        tracker.update(&create_stable_result());
        assert!(tracker.stability_index().is_some());
        
        // Then become unstable
        tracker.update(&create_unstable_result());
        
        // Stability should be reset
        assert!(tracker.stability_index().is_none());
        assert!(!matches!(tracker.current_status(), StabilityStatus::<f64>::Stable));
    }
    
    #[test]
    fn test_tracker_interrupted_stability() {
        let params = StabilityParameters::<f64> {
            min_stable_windows: 3,
            ..Default::default()
        };
        let mut tracker = ConsecutiveWindowTracker::new(params);
        
        // Two stable, then unstable, then stable again
        tracker.update(&create_stable_result());
        tracker.update(&create_stable_result());
        tracker.update(&create_unstable_result()); // Interrupts the sequence
        tracker.update(&create_stable_result());
        tracker.update(&create_stable_result());
        
        // Should not have achieved stability (only 2 consecutive after interruption)
        assert!(tracker.stability_index().is_none());
    }
    
    #[test]
    fn test_tracker_reset() {
        let params = StabilityParameters::<f64>::default();
        let mut tracker = ConsecutiveWindowTracker::new(params);
        
        // Add some windows
        tracker.update(&create_stable_result());
        tracker.update(&create_stable_result());
        
        // Reset
        tracker.reset();
        
        // Should be back to initial state
        assert!(matches!(tracker.current_status(), StabilityStatus::<f64>::Unknown));
        assert_eq!(tracker.stability_index(), None);
    }
}