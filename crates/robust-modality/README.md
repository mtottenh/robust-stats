# robust-modality

Multimodality detection using the Lowlands algorithm, which identifies modes in data distributions by finding areas of low density (lowlands) between peaks.

## Features

- **Lowlands Algorithm**: Robust mode detection based on density analysis
- **Configurable Sensitivity**: Adjust detection parameters for your use case
- **Integration**: Works seamlessly with robust-quantile estimators
- **Visualization Support**: Optional plotting capabilities
- **Generic Design**: Works with any numeric type and histogram builder

## Algorithm Overview

The Lowlands algorithm:
1. Creates a density histogram of your data
2. Identifies peaks (local maxima) in the histogram
3. Finds "lowlands" - areas of low density between peaks
4. Determines mode boundaries based on lowland positions

## Usage

### Basic Example

```rust
use robust_modality::{default_detector, ModalityDetector};
use robust_quantile::{estimators::harrell_davis, QuantileAdapter, HDWeightComputer};
use robust_core::{scalar_sequential, UnifiedWeightCache, CachePolicy};

// Create bimodal data
let mut data = Vec::new();
data.extend((0..50).map(|x| x as f64 / 10.0));  // Mode 1: 0-5
data.extend((80..130).map(|x| x as f64 / 10.0)); // Mode 2: 8-13

// Create detector with default settings
let detector = default_detector();

// Create quantile estimator and cache
let engine = scalar_sequential();
let hd = harrell_davis(engine);
let estimator = QuantileAdapter::new(hd);
let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);

// Detect modes
let result = detector.detect_modes_with_estimator(&data, &estimator, &cache)?;

println!("Found {} modes", result.mode_count());
for mode in result.modes() {
    println!("Mode at {:.2}, range: [{:.2}, {:.2}]", 
        mode.location, mode.left_bound, mode.right_bound);
}
```

### Custom Configuration

Adjust sensitivity and precision for your specific needs:

```rust
use robust_modality::detector_with_params;

// Lower sensitivity (0.3) = fewer modes detected
// Higher precision (0.02) = more accurate boundaries
let detector = detector_with_params(0.3, 0.02);

// Detect modes in challenging data
let data = vec![1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 9.0, 9.1, 9.2];
let result = detector.detect_modes_with_estimator(&data, &estimator, &cache)?;
```

### Working with Results

```rust
let result = detector.detect_modes_with_estimator(&data, &estimator, &cache)?;

// Check modality
match result.mode_count() {
    1 => println!("Unimodal distribution"),
    2 => println!("Bimodal distribution"),
    n => println!("Multimodal with {} modes", n),
}

// Analyze individual modes
for mode in result.modes() {
    println!("Mode width: {:.2}", mode.width());
    
    // Check if values belong to this mode
    if mode.contains(5.5) {
        println!("Value 5.5 belongs to this mode");
    }
}

// Access the underlying histogram
let histogram = result.histogram();
println!("Used {} bins for analysis", histogram.bins().len());
```

## Parameters

### Sensitivity (0.0 - 1.0)
- **Low (0.1-0.3)**: Detects only prominent modes
- **Medium (0.4-0.6)**: Balanced detection
- **High (0.7-0.9)**: Detects subtle modes

### Precision (0.001 - 0.1)
- **Low (0.05-0.1)**: Faster but less accurate boundaries
- **Medium (0.01-0.05)**: Good balance
- **High (0.001-0.01)**: Accurate boundaries but slower

## Builder Pattern

For fine-grained control:

```rust
use robust_modality::ModalityDetectorBuilder;
use robust_histogram::QRDEBuilderWithSteps;

let detector = ModalityDetectorBuilder::new()
    .sensitivity(0.5)
    .precision(0.01)
    .min_bin_count(50)
    .max_bin_count(500)
    .histogram_builder(QRDEBuilderWithSteps::new())
    .build();
```

## Visualization (Optional)

With the visualization feature:

```rust
use robust_modality::ModalityVisualizer;

// Implement custom visualizer
struct MyVisualizer;
impl ModalityVisualizer for MyVisualizer {
    fn visualize(&self, result: &ModalityResult<f64>) {
        // Custom plotting logic
    }
}

let detector = ModalityDetectorBuilder::new()
    .visualizer(MyVisualizer)
    .build();
```

## Performance Tips

1. **Pre-sort data** if using multiple times
2. **Adjust bin counts**: Fewer bins = faster but less precise
3. **Cache quantile computations** when analyzing multiple datasets

## References

- Akinshin, A. (2023). Lowland multimodality detection. [Blog post](https://aakinshin.net/posts/lowland-multimodality-detection/)