# robust-quantile

High-performance quantile estimation algorithms with a focus on robustness and statistical rigor.

## Features

- **Harrell-Davis quantile estimator**: Smooth, differentiable quantile estimates using Beta distribution weights
- **Trimmed Harrell-Davis**: Robust variant that trims extreme weights for outlier resistance
- **Batch operations**: Efficient multi-quantile computation
- **SIMD acceleration**: Optional AVX2/AVX512 support for performance
- **Cache-friendly**: Reusable weight caches for repeated computations

## Usage

```rust
use robust_quantile::{estimators::harrell_davis, QuantileEstimator, HDWeightComputer};
use robust_core::{simd_sequential, CachePolicy, UnifiedWeightCache};

// Create estimator with SIMD engine
let engine = simd_sequential();
let hd = harrell_davis(engine);

// Create cache for weight reuse
let cache = UnifiedWeightCache::new(
    HDWeightComputer::new(), 
    CachePolicy::Lru { max_entries: 1024 }
);

// Sample data
let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];

// Compute quantiles
let median = hd.quantile(&mut data, 0.5, &cache)?;
let q90 = hd.quantile(&mut data, 0.9, &cache)?;

// Batch computation for efficiency
let quantiles = vec![0.25, 0.5, 0.75];
let results = hd.estimate_quantiles_with_cache(&mut data, &quantiles, &cache)?;
```

## Trimmed Harrell-Davis

For extreme outlier resistance:

```rust
use robust_quantile::{estimators::trimmed_harrell_davis, TrimmedHDWeightComputer};
use robust_quantile::estimators::LinearWidth;

// Create trimmed HD with linear width function
let estimator = trimmed_harrell_davis::<f64, LinearWidth>(engine);

// Create cache with trimmed weight computer
let computer = TrimmedHDWeightComputer::<f64, LinearWidth>::new();
let cache = UnifiedWeightCache::new(computer, CachePolicy::NoCache);

// Compute robust quantiles
let robust_median = estimator.quantile(&mut data, 0.5, &cache)?;
```

## Performance Tips

1. **Pre-sort data** when computing multiple quantiles:
   ```rust
   data.sort_by(|a, b| a.partial_cmp(b).unwrap());
   let median = hd.quantile_sorted(&data, 0.5, &cache)?;
   ```

2. **Use batch methods** for multiple quantiles:
   ```rust
   let results = hd.estimate_quantiles_with_cache(&mut data, &quantiles, &cache)?;
   ```

3. **Reuse caches** across operations:
   ```rust
   let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::Lru { max_entries: 100 });
   for window in data.windows(100) {
       let mut window_vec = window.to_vec();
       let median = hd.quantile(&mut window_vec, 0.5, &cache)?;
   }
   ```

## Algorithm Details

### Harrell-Davis Quantile Estimator

Uses weighted average with Beta distribution weights:
- Smooth, differentiable estimates
- Asymptotically normal distribution
- Suitable for confidence interval construction

### Trimmed Harrell-Davis

Reduces influence of extreme order statistics:
- Available width functions: `ConstantWidth`, `LinearWidth`, `SqrtWidth`
- Maintains smoothness while improving robustness
- Configurable trimming strategies

## References

- Harrell, F.E. and Davis, C.E. (1982). A new distribution-free quantile estimator. Biometrika, 69(3), 635-640.
- Akinshin, A. (2023). Trimmed Harrell-Davis quantile estimator. [Blog post](https://aakinshin.net/posts/trimmed-hd-quantile-estimator/)
