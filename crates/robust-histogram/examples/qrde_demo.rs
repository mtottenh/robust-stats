//! Demonstrates the QRDE (Quantile-Respectful Density Estimator) with step functions

use robust_histogram::{
    adaptive_qrde, qrde_with_steps, QRDEBuilderWithSteps,
};
use robust_quantile::{estimators::harrell_davis, QuantileAdapter, HDWeightComputer};
use robust_core::{
    execution::scalar_sequential, CachePolicy, UnifiedWeightCache,
};

fn main() {
    // Generate some example data with different characteristics
    let normal_data: Vec<f64> = (0..1000)
        .map(|i| {
            let x = i as f64 / 100.0 - 5.0;
            // Normal-like distribution
            (-x * x / 2.0).exp()
        })
        .collect();

    let skewed_data: Vec<f64> = (0..1000)
        .map(|i| {
            let x = i as f64 / 200.0;
            // Exponential-like distribution
            x * (-x).exp() * 10.0
        })
        .collect();

    // Create estimator and cache for all operations
    let engine = scalar_sequential();
    let hd = harrell_davis(engine);
    let estimator = QuantileAdapter::new(hd);
    let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);

    // Example 1: Adaptive QRDE that adjusts to sample size
    println!("=== Adaptive QRDE ===");
    let adaptive_hist = adaptive_qrde(&normal_data, &estimator, &cache).unwrap();
    println!("Adaptive bins for {} samples: {}", normal_data.len(), adaptive_hist.len());
    
    // Example 2: Custom step function for specific quantiles of interest
    println!("\n=== Custom Quantile Steps ===");
    let custom_hist = qrde_with_steps(
        &skewed_data,
        &estimator,
        &cache,
        |_size| {
            // Focus on specific percentiles
            vec![
                0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0
            ]
        },
    ).unwrap();
    
    println!("Custom quantile-based bins:");
    for (i, bin) in custom_hist.bins().iter().enumerate() {
        println!(
            "  Bin {}: [{:.3}, {:.3}) - count: {}, density: {:.3}",
            i, bin.left, bin.right, bin.count, bin.density
        );
    }

    // Example 3: Exponential steps for better tail resolution
    println!("\n=== Exponential Steps (Tail-Focused) ===");
    let exp_builder = QRDEBuilderWithSteps::exponential(10);
    let exp_hist = exp_builder.build(&skewed_data, &estimator, &cache).unwrap();
    
    println!("Exponential step bins (better tail resolution):");
    for (i, bin) in exp_hist.bins().iter().enumerate() {
        let width = bin.width();
        println!(
            "  Bin {}: width={:.3}, count={}, density={:.3}",
            i, width, bin.count, bin.density
        );
    }

    // Example 4: Custom logarithmic steps
    println!("\n=== Custom Logarithmic Steps ===");
    let log_hist = qrde_with_steps(
        &skewed_data,
        &estimator,
        &cache,
        |_size| {
            // Logarithmic spacing from 0.01 to 0.99
            let mut steps = vec![0.0];
            let n_inner = 20;
            for i in 0..n_inner {
                let t = (i + 1) as f64 / (n_inner + 1) as f64;
                // Map t from [0,1] to [0.01, 0.99] with log scaling
                let log_min = (0.01_f64).ln();
                let log_max = (0.99_f64).ln();
                let log_val = log_min + t * (log_max - log_min);
                steps.push(log_val.exp());
            }
            steps.push(1.0);
            steps
        },
    ).unwrap();
    
    println!("Logarithmic step bins: {}", log_hist.len());

    // Compare densities
    println!("\n=== Density Comparison ===");
    println!("Uniform 10 bins vs Adaptive vs Exponential:");
    
    let uniform_builder = QRDEBuilderWithSteps::uniform(10);
    let uniform_hist = uniform_builder.build(&skewed_data, &estimator, &cache).unwrap();
    
    println!("Uniform: {} bins", uniform_hist.len());
    println!("Adaptive: {} bins", adaptive_hist.len());
    println!("Exponential: {} bins", exp_hist.len());
}