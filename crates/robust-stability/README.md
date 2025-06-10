# robust-stability

Tools for determining when performance systems reach reliable, measurable steady states. Essential for performance testing, benchmarking, and system monitoring.

## Features

- **Statistical Analysis**: Mann-Kendall trend test, Levene's test for variance stability
- **Hilbert Transform**: Detect hidden oscillations and periodic instabilities
- **Online Detection**: Real-time stability monitoring with rolling windows
- **Offline Analysis**: Comprehensive retrospective analysis
- **Parameterized Design**: Flexible integration with robust estimators
- **Visualization Support**: Optional plotting of stability metrics

## Usage

### Basic Statistical Analysis

```rust
use robust_stability::{StatisticalStabilityAnalyzer, StabilityAnalyzer};

// Performance measurements over time
let data = vec![100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 101.0, 99.0];

// Create analyzer with default parameters
let analyzer = StatisticalStabilityAnalyzer::default();
let result = analyzer.analyze(&data)?;

match result.status {
    StabilityStatus::Stable => println!("System is stable"),
    StabilityStatus::Unstable(reasons) => {
        println!("System is unstable:");
        for reason in reasons {
            println!("  - {:?}", reason);
        }
    }
}

// Access detailed metrics
println!("Trend p-value: {:.4}", result.metrics.trend_p_value);
println!("Variance stable: {}", result.metrics.variance_stable);
```

### Custom Parameters

```rust
use robust_stability::{StabilityParameters, StatisticalStabilityAnalyzer};

let params = StabilityParameters {
    window_size: 50,              // Analysis window size
    significance_level: 0.01,     // Stricter significance (99% confidence)
    min_stable_windows: 5,        // Require 5 consecutive stable windows
    trend_threshold: 0.1,         // Maximum allowed trend coefficient
    variance_ratio_threshold: 2.0, // Maximum variance ratio between windows
};

let analyzer = StatisticalStabilityAnalyzer::new(params);
```

### Hilbert Transform Analysis

Detect oscillations and periodic instabilities:

```rust
use robust_stability::{HilbertStabilityAnalyzer, StabilityAnalyzer};

// Data with hidden oscillation
let t: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
let data: Vec<f64> = t.iter()
    .map(|&x| 100.0 + 5.0 * (2.0 * std::f64::consts::PI * x).sin())
    .collect();

let analyzer = HilbertStabilityAnalyzer::default();
let result = analyzer.analyze(&data)?;

if let Some(freq) = result.metrics.dominant_frequency {
    println!("Detected oscillation at {:.2} Hz", freq);
}
```

### Online Stability Detection

For real-time monitoring:

```rust
use robust_stability::{OnlineStabilityDetector, OnlineStabilityAnalyzer};

let mut detector = OnlineStabilityDetector::with_window_size(30);

// Process streaming data
for value in data_stream {
    detector.update(value);
    
    if detector.has_enough_data() {
        let result = detector.current_stability()?;
        if matches!(result.status, StabilityStatus::Stable) {
            println!("System reached steady state");
            break;
        }
    }
}
```

### With Robust Estimators

Using parameterized traits for custom estimators:

```rust
use robust_stability::{StatisticalStabilityAnalyzer, StabilityAnalyzerWithEstimators};
use robust_spread::Mad;
use robust_quantile::{harrell_davis_factory, EstimatorFactory};
use robust_core::auto_engine;

// Create estimators
let engine = auto_engine();
let factory = harrell_davis_factory();
let quantile_est = factory.create(&engine);
let cache = factory.create_cache();
let spread_est = Mad::new(engine.primitives().clone());

// Analyze with robust estimators
let analyzer = StatisticalStabilityAnalyzer::default();
let result = analyzer.analyze_with_estimators(
    &data,
    &spread_est,
    &quantile_est,
    &cache
)?;
```

### Offline Comprehensive Analysis

For detailed post-hoc analysis:

```rust
use robust_stability::{OfflineStabilityAnalyzer, StabilityAnalyzer};

let analyzer = OfflineStabilityAnalyzer::default()
    .with_bootstrap_samples(1000)
    .with_detailed_metrics(true);

let result = analyzer.analyze(&data)?;

// Access extended metrics
println!("Bootstrap CI: [{:.2}, {:.2}]", 
    result.metrics.bootstrap_ci_lower.unwrap(),
    result.metrics.bootstrap_ci_upper.unwrap()
);
```

## Stability Metrics

The analysis provides various metrics:

- **Trend Analysis**: Mann-Kendall test statistic and p-value
- **Variance Stability**: Levene's test for homoscedasticity
- **Oscillation Detection**: Dominant frequencies via Hilbert transform
- **Window Statistics**: Rolling mean, variance, and quantiles
- **Stationarity Tests**: Multiple statistical tests for time series stability

## Unstable Reasons

When instability is detected, specific reasons are provided:

```rust
use robust_stability::UnstableReason;

match reason {
    UnstableReason::SignificantTrend => {
        println!("Data shows significant upward or downward trend");
    }
    UnstableReason::VarianceChange => {
        println!("Variance is not constant over time");
    }
    UnstableReason::Oscillation => {
        println!("Periodic oscillations detected");
    }
    // ... other reasons
}
```

## Performance Considerations

1. **Window Size**: Larger windows provide more reliable results but need more data
2. **Significance Level**: Lower values (e.g., 0.01) are more conservative
3. **Minimum Samples**: Most algorithms need at least 20-30 data points
4. **Computation**: Hilbert transform is more expensive than statistical tests

## Use Cases

- **Benchmarking**: Determine when system performance stabilizes
- **CI/CD**: Automated performance regression detection
- **Load Testing**: Identify when system reaches steady state under load
- **Monitoring**: Real-time stability tracking for production systems