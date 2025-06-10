# robust-effect

Effect size measurements that work with robust estimators and are resistant to outliers. Effect sizes quantify the magnitude of differences between groups or relationships between variables.

## Features

### Location-based Measures
- **Cohen's d**: Standardized mean difference (with robust variants)
- **Glass's delta**: Uses control group spread only
- **Hedges' g**: Bias-corrected Cohen's d

### Non-parametric Measures
- **Cliff's delta**: Based on dominance probability
- **CLES**: Common Language Effect Size (probability of superiority)

### Correlation-based Measures
- **Point-biserial correlation**: Binary/continuous relationships
- **Eta-squared (η²)**: Proportion of variance explained
- **Omega-squared (ω²)**: Less biased variance measure

## Usage

### Cohen's d with Robust Estimators

```rust
use robust_effect::{CohenD, StandardizedEffectSize};
use robust_spread::Mad;
use robust_quantile::{harrell_davis_factory, QuantileAdapter};
use robust_core::{auto_engine, EstimatorFactory};

// Two groups to compare
let mut group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let mut group2 = vec![3.0, 4.0, 5.0, 6.0, 7.0];

// Set up estimators
let engine = auto_engine();
let factory = harrell_davis_factory();
let hd = factory.create(&engine);
let cache = factory.create_cache();

// Create location and spread estimators
let location_est = QuantileAdapter::median(hd.clone());
let spread_est = Mad::new(engine.primitives().clone());

// Compute robust Cohen's d
let cohen_d = CohenD::new();
let effect = cohen_d.compute_with_estimators(
    &mut group1,
    &mut group2,
    &location_est,    // Median for location
    &spread_est,      // MAD for spread
    &hd,              // Quantile estimator
    &cache
)?;

println!("Cohen's d: {:.3}", effect.magnitude);
println!("Interpretation: {:?}", effect.interpretation);
```

### Non-parametric Effect Sizes

No need for location/spread estimators:

```rust
use robust_effect::{CliffDelta, NonParametricEffectSize};

let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // With outlier
let group2 = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

// Cliff's delta is robust to outliers
let cliff = CliffDelta::new();
let effect = cliff.compute(&group1, &group2)?;

println!("Cliff's delta: {:.3}", effect.magnitude);

// Common Language Effect Size
use robust_effect::CommonLanguageEffectSize;
let cles = CommonLanguageEffectSize::new();
let prob = cles.compute(&group1, &group2)?;
println!("Probability that random value from group2 > group1: {:.1}%", 
         prob.magnitude * 100.0);
```

### Glass's Delta

When groups have different variances:

```rust
use robust_effect::{GlassDelta, StandardizedEffectSize};

// Control group with smaller variance
let mut control = vec![4.0, 5.0, 6.0, 5.0, 5.0];
let mut treatment = vec![7.0, 9.0, 11.0, 8.0, 13.0];

let glass = GlassDelta::new();
let effect = glass.compute_with_estimators(
    &mut control,
    &mut treatment,
    &location_est,
    &spread_est,
    &hd,
    &cache
)?;
```

### Hedges' g (Bias-Corrected)

For small sample sizes:

```rust
use robust_effect::{HedgesG, StandardizedEffectSize};

// Small samples
let mut small1 = vec![1.0, 2.0, 3.0];
let mut small2 = vec![4.0, 5.0, 6.0];

let hedges = HedgesG::new();
let effect = hedges.compute_with_estimators(
    &mut small1,
    &mut small2,
    &location_est,
    &spread_est,
    &hd,
    &cache
)?;
```

### Variance Explained

For ANOVA-like analyses:

```rust
use robust_effect::{EtaSquared, OmegaSquared};

// Multiple groups
let groups = vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
    vec![7.0, 8.0, 9.0],
];

let eta_squared = EtaSquared::compute_from_groups(&groups)?;
let omega_squared = OmegaSquared::compute_from_groups(&groups)?;

println!("η²: {:.3}", eta_squared);
println!("ω²: {:.3}", omega_squared);
```

## Effect Size Interpretation

All effect sizes include automatic interpretation:

```rust
use robust_effect::EffectSizeInterpretation;

match effect.interpretation {
    EffectSizeInterpretation::Negligible => println!("No practical difference"),
    EffectSizeInterpretation::Small => println!("Small effect"),
    EffectSizeInterpretation::Medium => println!("Medium effect"),
    EffectSizeInterpretation::Large => println!("Large effect"),
    EffectSizeInterpretation::VeryLarge => println!("Very large effect"),
}
```

## Choosing an Effect Size

| Measure | Use When |
|---------|----------|
| Cohen's d | General purpose, equal variances assumed |
| Glass's Δ | Control group is reference, unequal variances |
| Hedges' g | Small sample sizes (< 20 per group) |
| Cliff's δ | Non-normal data, ordinal scales |
| CLES | Intuitive probability interpretation needed |

## Performance Tips

1. **Pre-sort data** when computing multiple effect sizes
2. **Reuse caches** when using the same estimators repeatedly
3. **Use non-parametric measures** for heavily skewed data
4. **Batch computations** when comparing multiple groups

## References

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
- Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions.
- Hedges, L.V. (1981). Distribution theory for Glass's estimator of effect size.