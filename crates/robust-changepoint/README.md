# robust-changepoint

Algorithms for detecting changepoints in time series data, with a focus on robust methods that work well in the presence of noise and outliers.

## Features

- **CUSUM**: Cumulative Sum for detecting changes in mean
- **EWMA**: Exponentially Weighted Moving Average for gradual changes
- **Slopes**: Detection based on linear trend changes
- **Polynomial Slopes**: Advanced polynomial fitting for complex trends
- **Parameterized Design**: Flexible integration with quantile/spread estimators
- **Kernel Architecture**: Efficient computation via specialized kernels

## Usage

### Simple Detectors (No External Dependencies)

Some detectors work directly on data without needing external estimators:

```rust
use robust_changepoint::{CusumDetector, SimpleDetector};
use robust_core::ScalarBackend;

// Create sample data with a changepoint
let data: Vec<f64> = (0..50)
    .map(|i| if i < 25 { 0.0 } else { 5.0 })
    .collect();

// Create CUSUM detector
let detector = CusumDetector::new(
    ScalarBackend,
    4.0,  // threshold
    0.5   // drift parameter
);

// Detect changepoints
let result = detector.detect_simple(&data)?;

for cp in result.changepoints() {
    println!("Changepoint at index {} (confidence: {:.2})", 
        cp.location, cp.confidence);
}
```

### EWMA Detection

For detecting gradual changes:

```rust
use robust_changepoint::{EwmaDetector, SimpleDetector};

// Create detector with smoothing factor
let detector = EwmaDetector::new(
    ScalarBackend,
    0.1,  // smoothing factor (alpha)
    3.0   // threshold in standard deviations
);

let result = detector.detect_simple(&data)?;
```

### Polynomial Slopes

Advanced detection using polynomial fitting:

```rust
use robust_changepoint::{
    PolynomialSlopesDetector, 
    PolynomialSlopesParameters,
    SimpleDetector
};

// Configure parameters
let params = PolynomialSlopesParameters {
    window_size: 20,
    degree: 2,  // quadratic polynomials
    threshold: 2.0,
    min_segment_size: 10,
};

let detector = PolynomialSlopesDetector::new(ScalarBackend, params);
let result = detector.detect_simple(&data)?;

// Access polynomial coefficients if needed
for window_result in &result.metadata {
    if let Some(coeffs) = window_result.get("coefficients") {
        println!("Polynomial coefficients: {:?}", coeffs);
    }
}
```

### Adaptive Polynomial Slopes

Automatically adjusts parameters based on data characteristics:

```rust
use robust_changepoint::{
    AdaptivePolynomialSlopesDetector,
    AdaptivePolynomialSlopesParameters
};

let params = AdaptivePolynomialSlopesParameters::default();
let detector = AdaptivePolynomialSlopesDetector::new(ScalarBackend, params);

// Detects optimal window size and polynomial degree
let result = detector.detect_simple(&data)?;
```

### With Spread Estimators

For detectors that use robust spread estimation:

```rust
use robust_changepoint::{SlopesDetector, ChangePointDetector};
use robust_spread::{Mad, SpreadEstimator};
use robust_quantile::{harrell_davis_factory, EstimatorFactory};
use robust_core::auto_engine;

// Create estimators
let engine = auto_engine();
let factory = harrell_davis_factory();
let quantile_est = factory.create(&engine);
let cache = factory.create_cache();
let spread_est = Mad::new(engine.primitives().clone());

// Create detector
let detector = SlopesDetector::new(engine.primitives().clone());

// Detect with robust spread estimation
let result = detector.detect(
    &data,
    &spread_est,
    &quantile_est,
    &cache
)?;
```

### Confidence Scoring

All detectors provide confidence scores for detected changepoints:

```rust
use robust_changepoint::ConfidenceScoring;

// After detection
let result = detector.detect_simple(&data)?;

for cp in result.changepoints() {
    match cp.confidence {
        c if c > 0.9 => println!("High confidence changepoint"),
        c if c > 0.7 => println!("Medium confidence changepoint"),
        _ => println!("Low confidence changepoint"),
    }
}
```

## Changepoint Types

Detected changepoints include:
- **Location**: Index where the change occurs
- **Change Type**: Mean shift, variance change, or trend change
- **Confidence**: Score between 0 and 1
- **Metadata**: Algorithm-specific information

```rust
use robust_changepoint::ChangeType;

match cp.change_type {
    ChangeType::MeanShift => println!("Mean level changed"),
    ChangeType::VarianceChange => println!("Variance changed"),
    ChangeType::TrendChange => println!("Trend/slope changed"),
}
```

## Performance Tips

1. **Pre-sort data** if using multiple algorithms on the same dataset
2. **Reuse caches** when analyzing sliding windows
3. **Use SIMD backends** for better performance:
   ```rust
   use robust_core::simd_sequential;
   let engine = simd_sequential();
   let detector = CusumDetector::new(engine.primitives().clone(), 4.0, 0.5);
   ```

## Algorithm Selection

- **CUSUM**: Best for abrupt changes in mean
- **EWMA**: Good for gradual changes and online detection
- **Slopes**: Effective for trend changes
- **Polynomial Slopes**: Best for complex patterns and acceleration detection

## References

- Page, E.S. (1954). Continuous inspection schemes. Biometrika.
- Roberts, S.W. (1959). Control chart tests based on geometric moving averages. Technometrics.