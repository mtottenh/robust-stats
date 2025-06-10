# Lowlands: Robust Statistics in Rust

A comprehensive collection of high-performance robust statistical methods inspired by the work of [Andrey Akinshin](https://aakinshin.net/), particularly his [Perfolizer](https://github.com/AndreyAkinshin/perfolizer) library and research on robust statistics.

## Overview

Lowlands provides state-of-the-art robust statistical methods that are resistant to outliers and data contamination. The project is organized as a Cargo workspace with specialized crates for different statistical domains:

### Core Crates

- **[`robust-core`](crates/robust-core/README.md)** - Core traits, types, and computational primitives
- **[`robust-quantile`](crates/robust-quantile/README.md)** - Quantile estimation (Harrell-Davis, Trimmed HD)
- **[`robust-spread`](crates/robust-spread/README.md)** - Spread measures (MAD, QAD, IQR)
- **[`robust-confidence`](crates/robust-confidence/README.md)** - Confidence intervals (Maritz-Jarrett, Bootstrap)
- **[`robust-modality`](crates/robust-modality/README.md)** - Multimodality detection (Lowlands algorithm)

### Analysis Crates

- **[`robust-changepoint`](crates/robust-changepoint/README.md)** - Changepoint detection algorithms
- **[`robust-stability`](crates/robust-stability/README.md)** - Stability analysis methods
- **[`robust-effect`](crates/robust-effect/README.md)** - Effect size measurements
- **[`robust-histogram`](crates/robust-histogram/README.md)** - Histogram construction
- **[`robust-viz`](crates/robust-viz/README.md)** - Visualization utilities

### Integration

- **[`robust-polars`](crates/robust-polars/README.md)** - Polars DataFrame integration
- **[`statistical-analysis-pipeline`](crates/statistical-analysis-pipeline/README.md)** - High-level analysis pipeline

## Architecture

The crate ecosystem is built on a layered architecture that maximizes performance while maintaining composability:

### Layer 1: Compute Primitives
- **Purpose**: Provide SIMD-optimized operations (sum, dot product, sparse weighted sum)
- **Types**: `ScalarBackend`, `Avx2Backend`, `Avx512Backend`
- **Selection**: Compile-time dispatch with runtime validation

### Layer 2: Execution Engines
- **Purpose**: Control execution strategy (sequential/parallel) and primitive selection
- **Types**: `SequentialEngine`, `ParallelEngine`, `HierarchicalEngine`
- **Features**: Thread pool integration, nested operation control

### Layer 3: Statistical Components
- **Kernels**: Algorithm-specific operations (e.g., `WeightedSumKernel`, `MadKernel`)
- **Estimators**: User-facing APIs that compose kernels with engines
- **Caches**: Reusable computation results (e.g., `UnifiedWeightCache`)
- **Factories**: Zero-cost estimator construction with engine injection

This architecture enables:
- Zero-cost abstractions through compile-time optimization
- Efficient cache sharing across operations
- Prevention of thread oversubscription in nested parallel operations
- Clean separation between algorithms and execution strategies

## Quick Start

Add the crates you need to your `Cargo.toml`:

```toml
[dependencies]
robust-core = "0.1"
robust-quantile = "0.1"
robust-spread = "0.1"
```

Basic example:

```rust
use robust_quantile::{harrell_davis_factory, QuantileEstimator};
use robust_spread::{Mad, SpreadEstimator};
use robust_core::{auto_engine, EstimatorFactory, ScalarBackend};

// Create an execution engine (auto-selects best backend)
let engine = auto_engine();

// Create quantile estimator using factory pattern
let hd_factory = harrell_davis_factory::<f64>();
let hd = hd_factory.create(&engine);
let cache = hd_factory.create_cache();

// Create spread estimator
let mad = Mad::new(ScalarBackend::new());

// Analyze data with outliers
let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];

// Compute robust median (resistant to outlier)
let median = hd.quantile(&mut data, 0.5, &cache)?;
println!("Median: {:.2}", median); // ~3.5

// Compute robust spread
let spread = mad.estimate(&mut data, &hd, &cache)?;
println!("MAD: {:.2}", spread);
```

## Why Robust Statistics?

Traditional statistical methods can be heavily influenced by outliers:

```rust
use robust_core::utils::{mean, std_dev};

// Traditional mean is heavily affected by outliers
let clean = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let contaminated = vec![1.0, 2.0, 3.0, 4.0, 1000.0];

println!("Clean mean: {:.1}", mean(&clean));        // 3.0
println!("Contaminated mean: {:.1}", mean(&contaminated)); // 202.0

// Robust median remains stable
// (using the estimator from above)
let median_clean = hd.quantile(&mut clean.clone(), 0.5, &cache)?;    // ~3.0
let median_contam = hd.quantile(&mut contaminated.clone(), 0.5, &cache)?; // ~3.0
```

Our robust methods maintain high **breakdown points** - they can handle significant data contamination while still providing reliable estimates.

## Credits

This project is heavily inspired by the excellent work of [Andrey Akinshin](https://aakinshin.net/):
- Many algorithms are based on his [blog posts](https://aakinshin.net/posts/) and research
- The Lowlands multimodality detection algorithm is from his [paper](https://aakinshin.net/posts/lowland-multimodality-detection/)
- Implementation patterns follow his [Perfolizer](https://github.com/AndreyAkinshin/perfolizer) library

## License

This project is licensed under the MIT license.