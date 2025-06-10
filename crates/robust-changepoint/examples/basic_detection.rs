//! Basic changepoint detection example

use robust_changepoint::{CusumDetector, EwmaDetector, SlopesDetector};
use robust_changepoint::{ChangePointDetector, SimpleDetector};
use robust_core::execution::{simd_sequential};
use robust_core::{ExecutionEngine, UnifiedWeightCache, CachePolicy};
use robust_spread::StandardizedMad;
use robust_quantile::estimators::harrell_davis;
use robust_quantile::HDWeightComputer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Changepoint Detection Examples ===\n");

    // Set up shared components for the new API
    let engine = simd_sequential();
    let primitives = engine.primitives().clone();
    let quantile_est = harrell_davis(engine.clone());
    let spread_est = StandardizedMad::new(primitives.clone());
    let cache = UnifiedWeightCache::new(
        HDWeightComputer::new(),
        CachePolicy::NoCache,
    );

    // Example 1: CUSUM for mean shift detection
    println!("1. CUSUM - Mean Shift Detection");
    let data1: Vec<f64> = (0..100).map(|i| if i < 50 { 0.0 } else { 5.0 }).collect();

    // CUSUM with known target mean (0.0) for detecting shift to 5.0
    let cusum = CusumDetector::with_target_mean(primitives.clone(), 3.0, 0.5, 0.0);
    let result1 = cusum.detect_simple(&data1)?;

    println!("  Data: mean shift from 0 to 5 at index 50");
    println!("  Found {} changepoints", result1.changepoints().len());
    for cp in result1.changepoints().iter().take(3) {
        println!("    Changepoint at index {}, confidence: {:.3}", cp.index, cp.confidence);
    }

    // Example 2: Slopes for trend change detection
    println!("\n2. Slopes - Trend Change Detection");
    let data2: Vec<f64> = (0..100)
        .map(|i| if i < 50 { i as f64 } else { 100.0 - i as f64 })
        .collect();

    let slopes = SlopesDetector::new(primitives.clone(), 8, 3.0);
    let result2 = slopes.detect(&data2, &spread_est, &quantile_est, &cache)?;

    println!("  Data: increasing trend then decreasing trend");
    println!("  Found {} changepoints", result2.changepoints().len());
    for cp in result2.changepoints().iter().take(3) {
        println!("    Changepoint at index {}, confidence: {:.3}", cp.index, cp.confidence);
    }

    // Example 3: EWMA for gradual change detection
    println!("\n3. EWMA - Gradual Change Detection");
    let data3: Vec<f64> = (0..100)
        .map(|i| if i < 30 { 0.0 } else { (i - 30) as f64 * 0.2 })
        .collect();

    let ewma = EwmaDetector::new(primitives.clone(), 0.3, 2.5);
    let result3 = ewma.detect(&data3, &spread_est, &quantile_est, &cache)?;

    println!("  Data: gradual increase starting at index 30");
    println!("  Found {} changepoints", result3.changepoints().len());
    for cp in result3.changepoints().iter().take(3) {
        println!("    Changepoint at index {}, confidence: {:.3}", cp.index, cp.confidence);
    }

    // Example 4: Comparison on the same dataset
    println!("\n4. Algorithm Comparison");
    let data4: Vec<f64> = (0..80)
        .map(|i| match i {
            0..=19 => 2.0,
            20..=39 => 6.0,
            40..=59 => 3.0,
            _ => 8.0,
        })
        .collect();

    println!("  Data: multiple step changes (2→6→3→8)");

    let cusum_result = CusumDetector::new(primitives.clone(), 2.0, 0.5).detect_simple(&data4)?;
    println!("  CUSUM found {} changepoints", cusum_result.changepoints().len());

    let slopes_result = SlopesDetector::new(primitives.clone(), 5, 3.0)
        .detect(&data4, &spread_est, &quantile_est, &cache)?;
    println!("  Slopes found {} changepoints", slopes_result.changepoints().len());

    let ewma_result = EwmaDetector::new(primitives.clone(), 0.4, 2.0)
        .detect(&data4, &spread_est, &quantile_est, &cache)?;
    println!("  EWMA found {} changepoints", ewma_result.changepoints().len());

    Ok(())
}