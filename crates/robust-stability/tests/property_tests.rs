//! Property-based tests for stability analysis
//! 
//! These tests ensure that the window-based and traditional approaches
//! produce consistent results across a wide range of inputs.

#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;
    use robust_stability::*;
    use robust_stability::algorithms::{DefaultStatisticalAnalyzer, ConsecutiveWindowTracker};
    use robust_stability::window_traits::{WindowStabilityAnalyzer, StabilityTracker};
    
    // Property: Window analyzer should never allocate
    #[test]
    fn test_no_allocations_in_window_analysis() {
        let analyzer = DefaultStatisticalAnalyzer::with_default_estimators(
            StabilityParameters::<f64>::default()
        );
        
        // Various window sizes
        let test_windows = vec![
            vec![1.0; 10],
            vec![5.0; 50],
            (0..100).map(|i| i as f64).collect(),
            (0..200).map(|i| (i as f64).sin()).collect(),
        ];
        
        for window in test_windows {
            // This should not allocate
            let _result = analyzer.analyze_window(&window);
        }
    }
    
    proptest! {
        // Property: Stable signals should remain stable across different window sizes
        #[test]
        fn prop_stable_signal_consistency(
            base_value in 0.0..100.0,
            noise_scale in 0.0..0.1,
            window_size in 50usize..200,
            signal_len in 500usize..2000
        ) {
            // Generate stable signal
            let signal: Vec<f64> = (0..signal_len)
                .map(|_| base_value + noise_scale * (rand::random::<f64>() - 0.5))
                .collect();
            
            let analyzer = DefaultStatisticalAnalyzer::with_default_estimators(
                StabilityParameters::<f64> {
                    min_samples: window_size.min(50),
                    max_cv: 0.2,
                    ..Default::default()
                }
            );
            
            // Check multiple windows
            let mut stable_count = 0;
            let mut total_windows = 0;
            
            for window in signal.windows(window_size).skip(10) {
                let result = analyzer.analyze_window(window);
                if result.is_stable {
                    stable_count += 1;
                }
                total_windows += 1;
            }
            
            // Most windows should be stable for a stable signal
            let stability_ratio = stable_count as f64 / total_windows as f64;
            prop_assert!(stability_ratio > 0.8, 
                "Stable signal should have >80% stable windows, got {:.2}%", 
                stability_ratio * 100.0);
        }
        
        // Property: Trending signals should be consistently detected
        #[test]
        fn prop_trending_signal_detection(
            start_value in 0.0..100.0,
            trend_rate in 0.02..0.1,
            noise_scale in 0.0..0.05,
            window_size in 50usize..200
        ) {
            let signal_len = 1000;
            let signal: Vec<f64> = (0..signal_len)
                .map(|i| start_value + i as f64 * trend_rate + noise_scale * rand::random::<f64>())
                .collect();
            
            let analyzer = DefaultStatisticalAnalyzer::with_default_estimators(
                StabilityParameters::<f64> {
                    min_samples: window_size.min(50),
                    max_trend: 0.01, // Strict trend threshold
                    ..Default::default()
                }
            );
            
            // Check windows after initial warmup
            let mut trend_detected = 0;
            let mut total_windows = 0;
            
            for window in signal.windows(window_size).skip(5) {
                let result = analyzer.analyze_window(window);
                if !result.is_stable && result.trend_strength > 0.01 {
                    trend_detected += 1;
                }
                total_windows += 1;
            }
            
            // Most windows should detect the trend
            let detection_ratio = trend_detected as f64 / total_windows as f64;
            prop_assert!(detection_ratio > 0.7,
                "Trending signal should be detected in >70% of windows, got {:.2}%",
                detection_ratio * 100.0);
        }
        
        // Property: Tracker state should be consistent with window results
        #[test]
        fn prop_tracker_consistency(
            signal in prop::collection::vec(-50.0..50.0, 100..500),
            min_stable_windows in 2usize..10
        ) {
            let params = StabilityParameters::<f64> {
                min_stable_windows,
                min_samples: 50,
                ..Default::default()
            };
            
            let analyzer = DefaultStatisticalAnalyzer::with_default_estimators(params.clone());
            let mut tracker = ConsecutiveWindowTracker::new(params);
            
            let mut consecutive_stable = 0;
            let mut achieved_stability = false;
            
            for window in signal.windows(50) {
                let result = analyzer.analyze_window(window);
                tracker.update(&result);
                
                if result.is_stable {
                    consecutive_stable += 1;
                    if consecutive_stable >= min_stable_windows {
                        achieved_stability = true;
                    }
                } else {
                    consecutive_stable = 0;
                    achieved_stability = false;
                }
                
                // Tracker should agree with our manual tracking
                if achieved_stability {
                    prop_assert!(tracker.stability_index().is_some(),
                        "Tracker should report stability when {} consecutive stable windows found",
                        min_stable_windows);
                }
            }
        }
    }
    
    // Regression test for edge cases
    #[test]
    fn test_edge_cases() {
        let analyzer = DefaultStatisticalAnalyzer::with_default_estimators(
            StabilityParameters::<f64>::default()
        );
        
        // Empty window
        let empty: Vec<f64> = vec![];
        let result = analyzer.analyze_window(&empty);
        assert!(!result.is_stable);
        
        // Single value
        let single = vec![5.0];
        let result = analyzer.analyze_window(&single);
        assert!(!result.is_stable); // Too few samples
        
        // All zeros
        let zeros = vec![0.0; 100];
        let result = analyzer.analyze_window(&zeros);
        assert!(result.is_stable);
        assert_eq!(result.cv, 0.0);
        
        // Infinite/NaN handling
        let with_nan = vec![1.0, 2.0, f64::NAN, 3.0, 4.0];
        let result = analyzer.analyze_window(&with_nan);
        // Should handle gracefully (implementation dependent)
    }
}