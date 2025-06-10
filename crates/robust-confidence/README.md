# robust-confidence

Confidence interval estimation for robust statistics, providing methods that work well with non-normal distributions and outliers.

## Features

- **Maritz-Jarrett intervals**: For quantile estimators with smooth CI construction
- **Bootstrap intervals**: Multiple methods (Percentile, BCa, Basic, Student)
- **Asymptotic intervals**: Based on standard errors
- **Two-sample comparisons**: Confidence intervals for shifts and ratios
- **Memory efficient**: Workspace pooling for bootstrap operations
- **Parallel execution**: Leverages hierarchical execution for performance

## Usage

### Maritz-Jarrett Confidence Intervals

For quantile estimators with linear interpolation properties:

```rust
use robust_confidence::{MaritzJarrettCI, ConfidenceIntervalEstimator};
use robust_quantile::{estimators::harrell_davis, QuantileAdapter, HDWeightComputer};
use robust_core::{scalar_sequential, UnifiedWeightCache, CachePolicy};

// Create Maritz-Jarrett CI estimator for median
let mj_ci = MaritzJarrettCI::new(
    0.5,  // median (probability)
    0.95  // 95% confidence level
);

// Setup quantile estimator
let engine = scalar_sequential();
let hd = harrell_davis(engine);
let estimator = QuantileAdapter::new(hd);
let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);

// Calculate CI
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let ci = mj_ci.confidence_interval(&data, &estimator, &cache)?;
println!("95% CI for median: [{:.2}, {:.2}]", ci.lower, ci.upper);
```

### Bootstrap Confidence Intervals

Using the high-level API for common bootstrap operations:

```rust
use robust_confidence::{
    bootstrap_confidence_intervals, 
    PercentileBootstrap,
    DEFAULT_RESAMPLES
};
use robust_quantile::harrell_davis_factory;
use robust_core::{
    QuantileShiftComparison, 
    execution::auto_budgeted_engine
};

// Two samples to compare
let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let sample2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];

// Define comparison (median shift)
let comparison = QuantileShiftComparison::new(vec![0.5])?;

// Setup execution engine and factory
let engine = auto_budgeted_engine();
let factory = harrell_davis_factory();

// Compute bootstrap CI
let result = bootstrap_confidence_intervals(
    &sample1,
    &sample2,
    &comparison,
    &factory,
    engine,
    PercentileBootstrap,
    0.95,  // 95% confidence
    DEFAULT_RESAMPLES
)?;

println!("Bootstrap CI: {:?}", result.intervals);
```

### Quantile-Specific Bootstrap

Specialized functions for common quantile comparisons:

```rust
use robust_confidence::{quantile_shift_confidence_intervals, PercentileBootstrap};
use robust_quantile::harrell_davis_factory;
use robust_core::execution::auto_budgeted_engine;

// Compare medians between two samples
let ci = quantile_shift_confidence_intervals(
    &sample1,
    &sample2,
    vec![0.5],  // median
    &harrell_davis_factory(),
    auto_budgeted_engine(),
    PercentileBootstrap,
    0.95,
    10000  // resamples
)?;
```

### Bootstrap Methods

Different bootstrap methods for various scenarios:

```rust
use robust_confidence::{Bootstrap, BCaBootstrap, BasicBootstrap, StudentBootstrap};

// BCa (Bias-Corrected and accelerated) - most accurate but slower
let bca_bootstrap = Bootstrap::new(engine, BCaBootstrap)
    .with_resamples(20000)
    .with_confidence_level(0.95);

// Basic bootstrap - simple and fast
let basic_bootstrap = Bootstrap::new(engine, BasicBootstrap)
    .with_resamples(10000);

// Student bootstrap - accounts for variance estimation
let student_bootstrap = Bootstrap::new(engine, StudentBootstrap)
    .with_resamples(15000);
```

### Asymptotic Confidence Intervals

For estimators with known standard errors:

```rust
use robust_confidence::{AsymptoticCI, StandardErrorEstimator, MeanCache};

// Example: CI for mean using t-distribution
struct MeanSE;
impl StandardErrorEstimator<f64> for MeanSE {
    type Cache = MeanCache;
    
    fn standard_error(&self, data: &[f64], cache: &MeanCache) -> Result<f64> {
        // Compute standard error of mean
        let n = data.len() as f64;
        Ok(cache.std_dev / n.sqrt())
    }
}

let asymptotic = AsymptoticCI::t_distribution(MeanSE, 0.95);
```

## Performance Considerations

1. **Hierarchical Execution**: Uses subordinate engines to prevent thread oversubscription
2. **Workspace Pooling**: Reuses memory across bootstrap iterations
3. **Batch Processing**: Leverages batch APIs for efficient resampling
4. **Parallel Bootstrap**: Automatically parallelizes across resamples

## Resampling Guidelines

- **Fast estimates**: 5,000-10,000 resamples
- **Standard precision**: 20,000 resamples (default)
- **High precision**: 30,000-50,000 resamples
- **Publication quality**: 100,000+ resamples

## References

- Maritz, J.S. and Jarrett, R.G. (1978). A note on estimating the variance of the sample median. JASA.
- Efron, B. and Tibshirani, R.J. (1993). An Introduction to the Bootstrap. Chapman & Hall.
- Davison, A.C. and Hinkley, D.V. (1997). Bootstrap Methods and their Application. Cambridge.