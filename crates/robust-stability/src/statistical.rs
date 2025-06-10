//! Statistical methods for stability analysis

use crate::traits::{StabilityAnalyzer, StabilityAnalyzerProperties, StabilityAnalyzerWithEstimators, StabilityResult};
use crate::types::{
    StabilityMetrics, StabilityParameters, StabilityStatus, StationarityTests, UnstableReason,
};
use crate::visualization::{StabilityVisualizer, NullStabilityVisualizer};
use robust_core::{Error, Result, ExecutionEngine};
use robust_spread::{Mad, SpreadEstimator};
use robust_quantile::{estimators::trimmed_harrell_davis, SqrtWidth, QuantileEstimator};
use statrs::statistics::Statistics;
use std::marker::PhantomData;

/// Statistical stability analyzer using traditional tests
///
/// This analyzer uses classical statistical tests to detect stability issues:
/// - Mann-Kendall test for trend detection
/// - Levene's test for homoscedasticity (constant variance)
/// - Sliding window analysis for stationarity
/// - Robust statistics (median/MAD) for outlier resistance
///
/// # Type Parameters
/// 
/// - `V`: The visualizer type that implements `StabilityVisualizer`
pub struct StatisticalStabilityAnalyzer<V: StabilityVisualizer<f64> = NullStabilityVisualizer<f64>> {
    params: StabilityParameters<f64>,
    visualizer: V,
    _phantom: PhantomData<V>,
}

impl StatisticalStabilityAnalyzer<NullStabilityVisualizer<f64>> {
    /// Create a new statistical stability analyzer with null visualizer
    pub fn new(params: StabilityParameters<f64>) -> Self {
        Self { 
            params,
            visualizer: NullStabilityVisualizer::new(),
            _phantom: PhantomData,
        }
    }

    /// Create with default parameters and null visualizer
    pub fn default() -> Self {
        Self::new(StabilityParameters::<f64>::default())
    }
}

impl<V: StabilityVisualizer<f64>> StatisticalStabilityAnalyzer<V> {
    /// Create a new statistical stability analyzer with custom visualizer
    pub fn with_visualizer(visualizer: V, params: StabilityParameters<f64>) -> Self {
        Self {
            params,
            visualizer,
            _phantom: PhantomData,
        }
    }

    /// Perform Mann-Kendall trend test
    fn mann_kendall_test(&self, data: &[f64]) -> (f64, f64) {
        let n = data.len();
        let mut s = 0.0;

        // Calculate S statistic
        for i in 0..n - 1 {
            for j in i + 1..n {
                s += match data[j].partial_cmp(&data[i]) {
                    Some(std::cmp::Ordering::Greater) => 1.0,
                    Some(std::cmp::Ordering::Less) => -1.0,
                    _ => 0.0,
                };
            }
        }

        // Calculate variance
        let var_s = (n * (n - 1) * (2 * n + 5)) as f64 / 18.0;
        let std_s = var_s.sqrt();

        // Calculate z-score
        let z = if s > 0.0 {
            (s - 1.0) / std_s
        } else if s < 0.0 {
            (s + 1.0) / std_s
        } else {
            0.0
        };

        // Two-tailed p-value
        let p_value = 2.0 * (1.0 - statrs::function::erf::erf(z.abs() / std::f64::consts::SQRT_2));

        (z, p_value)
    }

    /// Test for homoscedasticity using Levene's test
    fn levene_test(&self, data: &[f64], num_groups: usize) -> f64 {
        if data.len() < num_groups * 2 {
            return 1.0; // Not enough data
        }

        let group_size = data.len() / num_groups;
        let mut groups = Vec::new();

        for i in 0..num_groups {
            let start = i * group_size;
            let end = if i == num_groups - 1 {
                data.len()
            } else {
                (i + 1) * group_size
            };
            groups.push(&data[start..end]);
        }

        // Calculate group medians
        let medians: Vec<f64> = groups
            .iter()
            .map(|group| {
                let mut sorted = group.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if sorted.len() % 2 == 0 {
                    (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
                } else {
                    sorted[sorted.len() / 2]
                }
            })
            .collect();

        // Calculate absolute deviations from median
        let mut all_deviations = Vec::new();
        for (i, group) in groups.iter().enumerate() {
            for &value in group.iter() {
                all_deviations.push((value - medians[i]).abs());
            }
        }

        // Perform ANOVA on deviations
        let overall_mean = all_deviations.iter().sum::<f64>() / all_deviations.len() as f64;

        // Between-group sum of squares
        let mut ss_between = 0.0;
        let mut idx = 0;
        for i in 0..num_groups {
            let group_size = groups[i].len();
            let group_deviations = &all_deviations[idx..idx + group_size];
            let group_mean = group_deviations.iter().sum::<f64>() / group_size as f64;
            ss_between += group_size as f64 * (group_mean - overall_mean).powi(2);
            idx += group_size;
        }

        // Within-group sum of squares
        let mut ss_within = 0.0;
        idx = 0;
        for i in 0..num_groups {
            let group_size = groups[i].len();
            let group_deviations = &all_deviations[idx..idx + group_size];
            let group_mean = group_deviations.iter().sum::<f64>() / group_size as f64;
            for &dev in group_deviations {
                ss_within += (dev - group_mean).powi(2);
            }
            idx += group_size;
        }

        // Calculate F-statistic
        let df_between = (num_groups - 1) as f64;
        let df_within = (data.len() - num_groups) as f64;
        let ms_between = ss_between / df_between;
        let ms_within = ss_within / df_within;
        let f_stat = ms_between / ms_within;

        // Approximate p-value using beta distribution
        // This is a simplification - in practice you'd use F-distribution
        let p_value = 1.0 - f_stat / (f_stat + df_within / df_between);

        p_value
    }

    /// Simple stationarity test using sliding windows
    fn simple_stationarity_test(&self, data: &[f64]) -> f64 {
        let window_size = data.len() / 4;
        if window_size < 10 {
            return 0.0; // Not enough data
        }

        let mut means = Vec::new();
        let mut variances = Vec::new();

        // Calculate statistics for sliding windows
        for i in 0..=data.len() - window_size {
            let window = &data[i..i + window_size];
            let mean = window.iter().sum::<f64>() / window_size as f64;
            let variance = window
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / window_size as f64;
            means.push(mean);
            variances.push(variance);
        }

        // Check if means are stable
        let mean_cv = self.coefficient_of_variation(&means);
        let var_cv = self.coefficient_of_variation(&variances);

        // Combined stationarity score
        let stationarity_score = 1.0 - (mean_cv + var_cv) / 2.0;
        stationarity_score.max(0.0).min(1.0)
    }

    /// Calculate coefficient of variation using robust estimators
    fn coefficient_of_variation(&self, data: &[f64]) -> f64 {
        let engine = robust_core::simd_sequential();
        let primitives = engine.primitives().clone();
        let quantile_estimator = robust_quantile::estimators::trimmed_harrell_davis(engine, SqrtWidth);
        let mad_est = Mad::new(primitives);
        // Create cache for weight reuse
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::TrimmedHDWeightComputer::new(SqrtWidth),
            robust_core::CachePolicy::NoCache
        );
        self.coefficient_of_variation_with_estimators(data, &mad_est, &quantile_estimator, &cache)
    }
    
    /// Calculate coefficient of variation using provided estimators for cache reuse
    fn coefficient_of_variation_with_estimators<S, Q>(
        &self, 
        data: &[f64], 
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> f64 
    where
        S: SpreadEstimator<f64, Q>,
        Q: QuantileEstimator<f64>,
    {
        if data.is_empty() {
            return 0.0;
        }

        let mut data_copy = data.to_vec();
        let median = match quantile_est.quantile(&mut data_copy, 0.5, cache) {
            Ok(m) => m,
            Err(_) => return 0.0,
        };
        
        if median.abs() < f64::EPSILON {
            return 0.0;
        }

        // Use standardized MAD for robust spread estimation
        let mut data_for_spread = data.to_vec();
        let spread = match spread_est.estimate(&mut data_for_spread, quantile_est, cache) {
            Ok(mad_value) => mad_value * 1.4826, // Apply standardization factor
            Err(_) => return 0.0,
        };

        spread / median.abs()
    }
    
    /// Calculate coefficient of variation using traditional statistics (for testing)
    #[allow(dead_code)]
    fn coefficient_of_variation_traditional(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        if mean.abs() < f64::EPSILON {
            return 0.0;
        }

        let variance = data
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;

        let std_dev = variance.sqrt();
        std_dev / mean.abs()
    }

    /// Detect trend using linear regression
    fn detect_trend(&self, data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = data.iter().sum::<f64>() / n;

        let mut num = 0.0;
        let mut den = 0.0;

        for (i, &y) in data.iter().enumerate() {
            let x = i as f64;
            num += (x - x_mean) * (y - y_mean);
            den += (x - x_mean).powi(2);
        }

        let slope = if den > 0.0 { num / den } else { 0.0 };

        // Calculate R-squared
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;

        for (i, &y) in data.iter().enumerate() {
            let y_pred = y_mean + slope * (i as f64 - x_mean);
            ss_res += (y - y_pred).powi(2);
            ss_tot += (y - y_mean).powi(2);
        }

        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        (slope, r_squared)
    }
}

impl<V: StabilityVisualizer<f64>> StabilityAnalyzerProperties for StatisticalStabilityAnalyzer<V> {
    fn minimum_samples(&self) -> usize {
        self.params.min_samples
    }

    fn method_name(&self) -> &str {
        "Statistical Analysis"
    }
}

impl<V, S, Q> StabilityAnalyzerWithEstimators<f64, S, Q> for StatisticalStabilityAnalyzer<V> 
where
    V: StabilityVisualizer<f64>,
    S: SpreadEstimator<f64, Q>,
    Q: QuantileEstimator<f64>,
{
    fn analyze_with_estimators(
        &self,
        data: &[f64],
        spread_est: &S,
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<StabilityResult<f64>> {
        if data.len() < self.params.min_samples {
            return Err(Error::InsufficientData {
                expected: self.params.min_samples,
                actual: data.len(),
            });
        }

        // Record the input signal
        let _ = self.visualizer.record_signal(data, "Input Signal");

        // Calculate basic statistics using provided estimators
        let mut data_for_median = data.to_vec();
        let median = quantile_est.quantile(&mut data_for_median, 0.5, cache)
            .map_err(|e| Error::Other(anyhow::anyhow!("Quantile estimation failed: {}", e)))?;
        
        let robust_spread = {
            let mut data_for_spread = data.to_vec();
            let spread_value = spread_est.estimate(&mut data_for_spread, quantile_est, cache)?;
            spread_value * 1.4826 // Apply standardization factor
        };
        
        // Use robust CV
        let cv = if median.abs() > f64::EPSILON {
            robust_spread / median.abs()
        } else {
            0.0
        };
        
        // Keep traditional mean/std_dev for metrics reporting
        let mean = data.mean();
        let std_dev = data.std_dev();
        
        // Traditional CV for visualization
        let traditional_cv = if mean.abs() > f64::EPSILON {
            std_dev / mean.abs()
        } else {
            0.0
        };

        // Record statistics
        let _ = self.visualizer.record_statistics(
            mean,
            median,
            std_dev,
            robust_spread,
            traditional_cv,
            cv,
        );
        
        // Debug output
        if std::env::var("DEBUG_STABILITY").is_ok() {
            println!("[StatisticalAnalyzer] Data samples: {}", data.len());
            println!("[StatisticalAnalyzer] Traditional mean: {:.4}, std_dev: {:.4}, CV: {:.4}", 
                mean, std_dev, if mean.abs() > f64::EPSILON { std_dev / mean.abs() } else { 0.0 });
            println!("[StatisticalAnalyzer] Robust median: {:.4}, spread: {:.4}, CV: {:.4}", 
                median, robust_spread, cv);
        }

        // Perform statistical tests
        let (mk_z, mk_pvalue) = self.mann_kendall_test(data);
        let levene_pvalue = self.levene_test(data, 3);
        let stationarity_score = self.simple_stationarity_test(data);
        let (trend_slope, trend_r2) = self.detect_trend(data);
        
        // Record statistical test results
        let _ = self.visualizer.record_statistical_tests(
            mk_z,
            mk_pvalue,
            levene_pvalue,
            stationarity_score,
            trend_slope,
            trend_r2,
        );
        
        if std::env::var("DEBUG_STABILITY").is_ok() {
            println!("[StatisticalAnalyzer] Mann-Kendall z: {:.4}, p-value: {:.4}", mk_z, mk_pvalue);
            println!("[StatisticalAnalyzer] Levene p-value: {:.4}", levene_pvalue);
            println!("[StatisticalAnalyzer] Stationarity score: {:.4}", stationarity_score);
            println!("[StatisticalAnalyzer] Trend slope: {:.4}, R²: {:.4}", trend_slope, trend_r2);
        }

        // Build stationarity tests
        let stationarity_tests = StationarityTests::<f64> {
            mann_kendall_pvalue: Some(mk_pvalue),
            adf_pvalue: None,
            kpss_pvalue: None,
            ljung_box_pvalue: None,
        };

        // Determine stability status with multiple criteria
        let trend_significant = mk_pvalue < 0.05;
        let variance_unstable = levene_pvalue < 0.05;
        let nonstationary = stationarity_score < 0.7;

        // Count unstable indicators
        let unstable_count = 
            (cv > self.params.max_cv) as i32 +
            trend_significant as i32 +
            variance_unstable as i32 +
            nonstationary as i32;

        // Confidence is based on evidence strength
        let confidence = if unstable_count == 0 {
            // All tests indicate stability
            0.95 - cv / self.params.max_cv * 0.2
        } else if unstable_count >= 3 {
            // Strong evidence of instability
            0.9 + mk_pvalue.min(levene_pvalue).min(stationarity_score) * 0.1
        } else {
            // Mixed evidence
            0.5 + (2 - unstable_count) as f64 * 0.2
        };

        // Determine status
        let status = if unstable_count == 0 {
            StabilityStatus::<f64>::Stable
        } else if unstable_count >= 3 {
            // Determine primary reason
            if cv > self.params.max_cv {
                StabilityStatus::<f64>::Unstable { 
                    reason: UnstableReason::<f64>::Custom(format!("High variability (CV={:.3})", cv)),
                    severity: cv / self.params.max_cv
                }
            } else if trend_significant {
                StabilityStatus::<f64>::Unstable { 
                    reason: UnstableReason::<f64>::Trend { slope: trend_slope },
                    severity: trend_slope.abs() / self.params.max_trend
                }
            } else if variance_unstable {
                StabilityStatus::<f64>::Unstable { 
                    reason: UnstableReason::<f64>::VarianceChange { rate: 1.0 - levene_pvalue },
                    severity: 1.0 - levene_pvalue
                }
            } else {
                StabilityStatus::<f64>::Unstable { 
                    reason: UnstableReason::<f64>::NonStationary,
                    severity: 1.0 - stationarity_score
                }
            }
        } else {
            StabilityStatus::<f64>::Unknown
        };
        
        if std::env::var("DEBUG_STABILITY").is_ok() {
            println!("[StatisticalAnalyzer] Unstable indicators: {}/4", unstable_count);
            println!("[StatisticalAnalyzer] Final status: {:?}, confidence: {:.4}", status, confidence);
            println!("[StatisticalAnalyzer] Tests - MK p-value: {:.4}, Levene p-value: {:.4}, Stationarity: {:.4}", 
                mk_pvalue, levene_pvalue, stationarity_score);
        }

        // Build metrics
        let metrics = StabilityMetrics::<f64> {
            mean,
            std_dev,
            cv,
            trend_strength: Some(trend_slope.abs() * trend_r2),
            oscillation_metrics: None,
            stationarity_tests,
            sample_count: data.len(),
        };

        // Build explanation
        let explanation = format!(
            "Statistical analysis (robust): CV={:.3}, median={:.3}, robust spread={:.3}, trend slope={:.3}, Mann-Kendall p={:.3}, Levene p={:.3}, stationarity={:.3}",
            cv, median, robust_spread, trend_slope, mk_pvalue, levene_pvalue, stationarity_score
        );

        // Record final decision
        let _ = self.visualizer.record_final_decision(
            matches!(status, StabilityStatus::<f64>::Stable),
            cv,
            confidence,
            &explanation,
        );

        Ok(StabilityResult::new(status, metrics, confidence, self.method_name().to_string())
            .with_explanation(explanation))
    }
}

impl<V: StabilityVisualizer<f64>> StabilityAnalyzer<f64> for StatisticalStabilityAnalyzer<V> {
    fn analyze(&self, data: &[f64]) -> Result<StabilityResult<f64>> {
        // Create default estimators and delegate to parameterized version
        let engine = robust_core::simd_sequential();
        let primitives = engine.primitives().clone();
        let quantile_estimator = robust_quantile::estimators::trimmed_harrell_davis(engine, SqrtWidth);
        let mad_est = Mad::new(primitives);
        // Create cache for weight reuse
        let cache = robust_core::UnifiedWeightCache::new(
            robust_quantile::TrimmedHDWeightComputer::new(SqrtWidth),
            robust_core::CachePolicy::NoCache
        );
        
        self.analyze_with_estimators(data, &mad_est, &quantile_estimator, &cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_signal() {
        let analyzer = StatisticalStabilityAnalyzer::default();

        // Generate stable signal
        let signal: Vec<f64> = (0..100).map(|_| 10.0 + rand::random::<f64>() * 0.5).collect();

        let result = analyzer.analyze(&signal).unwrap();
        assert!(result.is_stable());
    }

    #[test]
    fn test_trending_signal() {
        let analyzer = StatisticalStabilityAnalyzer::default();

        // Generate trending signal
        let signal: Vec<f64> = (0..100).map(|i| 10.0 + 0.1 * i as f64).collect();

        let result = analyzer.analyze(&signal).unwrap();
        assert!(result.is_unstable());

        if let StabilityStatus::<f64>::Unstable { reason, .. } = &result.status {
            match reason {
                UnstableReason::<f64>::Trend { .. } => (),
                _ => panic!("Expected trend detection"),
            }
        }
    }

    #[test]
    fn test_mann_kendall() {
        let analyzer = StatisticalStabilityAnalyzer::default();

        // Test with no trend
        let no_trend = vec![1.0, 2.0, 1.5, 2.5, 1.0, 2.0];
        let (_z, p) = analyzer.mann_kendall_test(&no_trend);
        assert!(p > 0.05);

        // Test with strong trend
        let trend = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (_z, p) = analyzer.mann_kendall_test(&trend);
        assert!(p < 0.05);
    }
}