//! Demonstrates using QRDE with custom step functions for modality detection

use robust_modality::ModalityDetectorBuilder;
use robust_modality::visualization::NullModalityVisualizer;
use robust_quantile::{estimators::harrell_davis, QuantileAdapter, HDWeightComputer};
use robust_core::{execution::scalar_sequential, UnifiedWeightCache, CachePolicy};

#[cfg(feature = "test-utils")]
fn main() {
    use rand;
    
    // Generate test data with different characteristics
    let mut bimodal_data = Vec::new();
    
    // First mode around 10
    for _ in 0..1000 {
        bimodal_data.push(10.0 + rand::random::<f64>() * 2.0);
    }
    
    // Second mode around 20
    for _ in 0..1000 {
        bimodal_data.push(20.0 + rand::random::<f64>() * 2.0);
    }
    
    // Add some outliers
    bimodal_data.push(50.0);
    bimodal_data.push(55.0);
    
    // Create estimator and cache for all detections
    let engine = scalar_sequential();
    let hd = harrell_davis(engine);
    let estimator = QuantileAdapter::new(hd);
    let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
    
    println!("=== QRDE-based Modality Detection ===\n");
    
    // Example 1: Default uniform QRDE (via precision parameter)
    println!("1. Default Uniform QRDE:");
    let detector1 = ModalityDetectorBuilder::new(NullModalityVisualizer::default())
        .sensitivity(0.5)
        .precision(0.02) // Results in 50 uniform bins
        .build();
    
    let result1 = detector1.detect_modes_with_estimator(&bimodal_data, &estimator, &cache).unwrap();
    println!("   Modes detected: {}", result1.modes().len());
    println!("   Is multimodal: {}", result1.is_multimodal());
    
    // Example 2: Custom step function focusing on the middle range
    println!("\n2. Custom Step Function (Focus on Middle):");
    let detector2 = ModalityDetectorBuilder::new(NullModalityVisualizer::default())
        .sensitivity(0.5)
        .with_step_function(|_size| {
            // More bins in the middle range, fewer at extremes
            vec![
                0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68,
                0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0
            ]
        })
        .build();
    
    let result2 = detector2.detect_modes_with_estimator(&bimodal_data, &estimator, &cache).unwrap();
    println!("   Modes detected: {}", result2.modes().len());
    for (i, mode) in result2.modes().iter().enumerate() {
        println!("   Mode {}: location={:.1}, bounds=[{:.1}, {:.1}]",
            i + 1, mode.location, mode.left_bound, mode.right_bound);
    }
    
    // Example 3: Adaptive step function based on data size
    println!("\n3. Adaptive QRDE (sqrt(n) bins):");
    let detector3 = ModalityDetectorBuilder::new(NullModalityVisualizer::default())
        .sensitivity(0.5)
        .with_step_function(|size| {
            // Adaptive: use sqrt(n) bins
            let num_bins = (size as f64).sqrt().ceil() as usize;
            (0..=num_bins)
                .map(|i| i as f64 / num_bins as f64)
                .collect()
        })
        .build();
    
    let result3 = detector3.detect_modes_with_estimator(&bimodal_data, &estimator, &cache).unwrap();
    println!("   Data size: {}", bimodal_data.len());
    println!("   Bins used: ~{}", (bimodal_data.len() as f64).sqrt().ceil() as usize);
    println!("   Modes detected: {}", result3.modes().len());
    
    // Example 4: Exponential steps for better resolution at extremes
    println!("\n4. Exponential Step Function (Better Tail Resolution):");
    let detector4 = ModalityDetectorBuilder::new(NullModalityVisualizer::default())
        .sensitivity(0.5)
        .with_step_function(|_size| {
            let mut steps = vec![0.0];
            // Exponential spacing in first half
            for i in 1..10 {
                let t = i as f64 / 10.0;
                steps.push(0.5 * (1.0 - (-3.0 * t).exp()));
            }
            // Mirror for second half
            for i in 1..10 {
                steps.push(1.0 - 0.5 * (1.0 - (-3.0 * (10 - i) as f64 / 10.0).exp()));
            }
            steps.push(1.0);
            steps
        })
        .build();
    
    let result4 = detector4.detect_modes_with_estimator(&bimodal_data, &estimator, &cache).unwrap();
    println!("   Modes detected: {}", result4.modes().len());
    println!("   Better outlier handling due to tail resolution");
    
    // Compare histograms
    println!("\n=== Histogram Comparison ===");
    println!("Uniform bins: {} bins", result1.histogram().len());
    println!("Custom focus: {} bins", result2.histogram().len());
    println!("Adaptive:     {} bins", result3.histogram().len());
    println!("Exponential:  {} bins", result4.histogram().len());
}

#[cfg(not(feature = "test-utils"))]
fn main() {
    eprintln!("This example requires the 'test-utils' feature to be enabled.");
    eprintln!("Run with: cargo run --example qrde_modality --features test-utils");
}
