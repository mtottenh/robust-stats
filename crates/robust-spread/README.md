# robust-spread

Robust measures of statistical spread (scale/variability) that are resistant to outliers.

## Features

- **MAD** (Median Absolute Deviation): 50% breakdown point
- **QAD** (Quantile Absolute Deviation): Customizable robustness
- **IQR** (Interquartile Range): Classic robust spread measure
- **Trimmed/Winsorized Standard Deviation**: Adjustable outlier handling
- **Robust Moments**: Skewness and kurtosis measures

## Usage

All spread estimators are parameterized by the quantile estimator they use:

```rust
use robust_spread::{Mad, SpreadEstimator, SpreadEstimatorProperties};
use robust_quantile::{estimators::harrell_davis, HDWeightComputer};
use robust_core::{auto_engine, UnifiedWeightCache, CachePolicy, ScalarBackend};

// Create quantile estimator
let engine = auto_engine();
let quantile_est = harrell_davis(engine);
let cache = UnifiedWeightCache::new(
    HDWeightComputer::new(), 
    CachePolicy::NoCache
);

// Create MAD estimator
let mad = Mad::new(ScalarBackend::new());

// Analyze data with outliers
let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
let spread = mad.estimate(&mut data, &quantile_est, &cache)?;
println!("MAD: {:.2}", spread); // Robust to outlier
```

## Standardized MAD

For comparison with standard deviation on normal data:

```rust
use robust_spread::StandardizedMad;

// Scaled by consistency factor (1.4826)
let smad = StandardizedMad::new(ScalarBackend::new());
let spread = smad.estimate(&mut data, &quantile_est, &cache)?;
```

## QAD - Flexible Robustness

Choose your breakdown point with Quantile Absolute Deviation:

```rust
use robust_spread::Qad;

// QAD with p=0.25 (more efficient than MAD)
let qad = Qad::new(ScalarBackend::new(), 0.25)?;
let spread = qad.estimate(&mut data, &quantile_est, &cache)?;

// Standard QAD (p ≈ 0.318, optimal for normal data)
let standard_qad = Qad::standard(ScalarBackend::new());
```

## IQR - Interquartile Range

```rust
use robust_spread::Iqr;

let iqr = Iqr::new(ScalarBackend::new());
let range = iqr.estimate(&mut data, &quantile_est, &cache)?;
```

## Trimmed Standard Deviation

Remove extreme values before computing standard deviation:

```rust
use robust_spread::TrimmedStd;

// Trim 10% from each tail
let trimmed = TrimmedStd::symmetric(ScalarBackend::new(), 0.1)?;
let spread = trimmed.estimate(&mut data, &quantile_est, &cache)?;
```

## Factory Pattern

For integration with execution engines:

```rust
use robust_spread::mad_factory;
use robust_core::{EstimatorFactory, auto_engine};

let factory = mad_factory::<f64>();
let engine = auto_engine();
let estimator = factory.create(&engine);
let cache = factory.create_cache();

// Now use estimator.estimate_with_cache()
```

## Performance Tips

1. **Pre-sort data** when computing multiple statistics:
   ```rust
   data.sort_by(|a, b| a.partial_cmp(b).unwrap());
   let spread = mad.estimate_sorted(&data, &quantile_est, &cache)?;
   ```

2. **Reuse caches** across operations for better performance

3. **Choose appropriate primitives**: Use SIMD backends when available

## Estimator Properties

| Estimator | Breakdown Point | Gaussian Efficiency | Use When |
|-----------|----------------|-------------------|----------|
| MAD | 50% | 37% | Maximum robustness needed |
| QAD(0.25) | 25% | 82% | Balance robustness/efficiency |
| IQR | 25% | 65% | Simple robust spread |
| Trimmed Std | Variable | Variable | Known contamination level |

## References

- Rousseeuw, P.J. and Croux, C. (1993). Alternatives to the median absolute deviation. JASA.
- Akinshin, A. (2022). Quantile absolute deviation. [Blog post](https://aakinshin.net/posts/qad/)