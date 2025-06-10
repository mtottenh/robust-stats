# robust-histogram

Flexible histogram construction with multiple binning strategies, including quantile-based methods that are robust to outliers.

## Features

- **Fixed-width bins**: Traditional equal-width histograms
- **Scott's rule**: Optimal bin width for normal distributions
- **Freedman-Diaconis rule**: IQR-based bin width selection
- **Quantile-based bins**: Robust to outliers and skewed data
- **QRDE**: Quantile-Respectful Density Estimation
- **Histogram operations**: Normalize, merge, compare histograms

## Usage

### Fixed-Width Histograms

```rust
use robust_histogram::{FixedWidthBuilder, HistogramBuilder};

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

// Create histogram with 5 equal-width bins
let builder = FixedWidthBuilder::new(5);
let histogram = builder.build(&data)?;

for bin in histogram.bins() {
    println!("[{:.1}, {:.1}): count={}, density={:.3}",
             bin.left, bin.right, bin.count, bin.density);
}
```

### Automatic Bin Width Selection

```rust
use robust_histogram::{ScottsRule, FreedmanDiaconisRule, HistogramBuilder};

// Scott's rule - optimal for normal distributions
let scott_hist = ScottsRule.build(&data)?;

// Freedman-Diaconis rule - uses IQR, more robust
let fd_hist = FreedmanDiaconisRule.build(&data)?;
```

### Quantile-Based Histograms (Robust)

Handle outliers gracefully by using quantiles for bin boundaries:

```rust
use robust_histogram::QuantileBuilder;
use robust_quantile::{harrell_davis_factory, QuantileAdapter};
use robust_core::{auto_engine, EstimatorFactory};

// Data with outliers
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0];

// Set up quantile estimator
let engine = auto_engine();
let factory = harrell_davis_factory();
let hd = factory.create(&engine);
let cache = factory.create_cache();
let estimator = QuantileAdapter::new(hd);

// Create quantile-based histogram
let builder = QuantileBuilder::new(5);
let histogram = builder.build(&data, &estimator, &cache)?;

// Bins adapt to data distribution, handling outliers well
```

### QRDE (Quantile-Respectful Density Estimation)

Advanced density estimation that adapts to data distribution:

```rust
use robust_histogram::{qrde, QRDEBuilderWithSteps};

// Simple QRDE with fixed number of bins
let density = qrde(&data, &estimator, &cache, 10)?;

// QRDE with custom quantile steps
use robust_histogram::qrde_with_steps;
let custom_density = qrde_with_steps(
    &data,
    &estimator,
    &cache,
    |_size| vec![0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
)?;

// Adaptive QRDE that adjusts to sample size
use robust_histogram::adaptive_qrde;
let adaptive = adaptive_qrde(&data, &estimator, &cache)?;
```

### QRDE Step Functions

Different strategies for placing bin boundaries:

```rust
use robust_histogram::QRDEBuilderWithSteps;

// Linear steps: evenly spaced quantiles
let linear = QRDEBuilderWithSteps::linear(20);

// Exponential steps: more bins in tails
let exp = QRDEBuilderWithSteps::exponential(20);

// Sqrt steps: balanced between linear and exponential
let sqrt = QRDEBuilderWithSteps::sqrt(20);

// Custom step function
let custom = QRDEBuilderWithSteps::custom(|size| {
    // Your custom logic here
    vec![0.0, 0.5, 0.9, 0.99, 1.0]
});
```

### Histogram Operations

```rust
use robust_histogram::{HistogramOps, fixed_histogram};

let hist1 = fixed_histogram(&data1, 10)?;
let hist2 = fixed_histogram(&data2, 10)?;

// Normalize to unit area
let norm1 = hist1.normalize();
let norm2 = hist2.normalize();

// Compare histograms
let wasserstein = norm1.wasserstein_distance(&norm2);
let chi_squared = norm1.chi_squared_distance(&norm2);

// Merge histograms
let merged = hist1.merge(&hist2)?;

// Access statistics
let total_count = histogram.total_count();
let mean = histogram.mean()?;
let variance = histogram.variance()?;
```

### Convenience Functions

```rust
use robust_histogram::{fixed_histogram, scott_histogram, quantile_histogram};

// Quick histogram creation
let fixed = fixed_histogram(&data, 10)?;
let scott = scott_histogram(&data)?;
let quantile = quantile_histogram(&data, &estimator, &cache, 10)?;
```

## Choosing a Binning Strategy

| Strategy | Use When |
|----------|----------|
| Fixed-width | Data is uniformly distributed |
| Scott's rule | Data is approximately normal |
| Freedman-Diaconis | Robust automatic bin selection needed |
| Quantile-based | Data has outliers or is heavily skewed |
| QRDE | Need accurate density estimation for complex distributions |

## Integration with Modality Detection

This crate is designed to work seamlessly with `robust-modality`:

```rust
use robust_modality::default_detector;
use robust_histogram::QRDEBuilderWithSteps;

// QRDE provides the density estimate for modality detection
let detector = default_detector()
    .histogram_builder(QRDEBuilderWithSteps::new());
```

## Performance Tips

1. **Pre-sort data** if building multiple histograms
2. **Reuse quantile caches** when using quantile-based methods
3. **Choose appropriate bin counts**: Too many bins = noisy, too few = loss of detail
4. **Use fixed-width** for speed when data is well-behaved

## References

- Scott, D.W. (1979). On optimal and data-based histograms. Biometrika.
- Freedman, D. and Diaconis, P. (1981). On the histogram as a density estimator. Z. Wahrscheinlichkeitstheorie verw. Gebiete.
- Akinshin, A. (2023). Quantile-respectful density estimation based on the Harrell-Davis quantile estimator. [arXiv:2404.03835](https://arxiv.org/pdf/2404.03835)