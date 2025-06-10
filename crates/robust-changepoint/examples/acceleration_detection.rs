//! Acceleration-based changepoint detection example
//!
//! This example demonstrates how the new parameterized API can detect
//! different types of changes including acceleration changes.

use robust_changepoint::{AdaptivePolynomialSlopesDetector, AdaptivePolynomialSlopesParameters};
use robust_changepoint::{ChangePointDetector, types::ChangeType};
use robust_core::execution::{scalar_sequential, simd_sequential};
use robust_core::{ExecutionEngine, UnifiedWeightCache, CachePolicy};
use robust_spread::StandardizedMad;
use robust_quantile::estimators::harrell_davis;
use robust_quantile::HDWeightComputer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Acceleration-Based Changepoint Detection Demo ===\n");

    // Set up shared components for the new API
    let engine = simd_sequential();
    let quantile_est = harrell_davis(engine.clone());
    let spread_est = StandardizedMad::new(engine.primitives().clone());
    let cache = UnifiedWeightCache::new(
        HDWeightComputer::new(),
        CachePolicy::Lru { max_entries: 256 },
    );

    // Create different test signals
    let signals = vec![
        ("Linear Change", create_linear_change()),
        ("Quadratic Change", create_quadratic_change()),
        ("Exponential Change", create_exponential_change()),
        ("Step Change", create_step_change()),
    ];

    for (name, data) in signals {
        println!("\n--- {} ---", name);
        
        // Test basic polynomial detection
        test_basic_detection(&data, &engine, &spread_est, &quantile_est, &cache)?;
        
        // Test with acceleration detection enabled
        test_acceleration_detection(&data, &engine, &spread_est, &quantile_est, &cache)?;
    }

    Ok(())
}

fn test_basic_detection<E, S, Q>(
    data: &[f64],
    engine: &E,
    spread_est: &S,
    quantile_est: &Q,
    cache: &Q::State,
) -> Result<(), Box<dyn std::error::Error>>
where
    E: ExecutionEngine<f64>,
    S: robust_spread::SpreadEstimator<f64, Q>,
    Q: robust_quantile::QuantileEstimator<f64>,
{
    let params = AdaptivePolynomialSlopesParameters {
        base: robust_changepoint::PolynomialSlopesParameters {
            window_size: 20,
            polynomial_degree: 2, // Quadratic for better trend detection
            slope_threshold_multiplier: 3.0,
            steady_state_threshold_multiplier: 2.0,
            steady_state_windows: 3,
            min_confidence: 0.5,
            normalize_time: true,
        },
        use_acceleration: false, // Basic detection only
        acceleration_threshold_multiplier: 2.5,
        acceleration_weight: 0.3,
    };

    let detector = AdaptivePolynomialSlopesDetector::with_params(engine.primitives().clone(), params);
    let result = detector.detect(data, spread_est, quantile_est, cache)?;

    println!("  Basic polynomial detection:");
    println!("    Changepoints detected: {}", result.changepoints().len());
    for cp in result.changepoints() {
        println!(
            "      Index: {}, Confidence: {:.3}, Type: {:?}",
            cp.index, cp.confidence, cp.change_type
        );
    }

    Ok(())
}

fn test_acceleration_detection<E, S, Q>(
    data: &[f64],
    engine: &E,
    spread_est: &S,
    quantile_est: &Q,
    cache: &Q::State,
) -> Result<(), Box<dyn std::error::Error>>
where
    E: ExecutionEngine<f64>,
    S: robust_spread::SpreadEstimator<f64, Q>,
    Q: robust_quantile::QuantileEstimator<f64>,
{
    let params = AdaptivePolynomialSlopesParameters {
        base: robust_changepoint::PolynomialSlopesParameters {
            window_size: 20,
            polynomial_degree: 2, // Quadratic for better trend detection
            slope_threshold_multiplier: 3.0,
            steady_state_threshold_multiplier: 2.0,
            steady_state_windows: 3,
            min_confidence: 0.5,
            normalize_time: true,
        },
        use_acceleration: true, // Enable acceleration detection
        acceleration_threshold_multiplier: 2.0, // More sensitive to acceleration
        acceleration_weight: 0.4,
    };

    let detector = AdaptivePolynomialSlopesDetector::with_params(engine.primitives().clone(), params);
    let result = detector.detect(data, spread_est, quantile_est, cache)?;

    println!("  With acceleration detection:");
    println!("    Changepoints detected: {}", result.changepoints().len());
    for cp in result.changepoints() {
        println!(
            "      Index: {}, Confidence: {:.3}, Type: {:?}",
            cp.index, cp.confidence, cp.change_type
        );
    }

    // Count acceleration changes specifically
    let accel_changes = result.changepoints().iter()
        .filter(|cp| matches!(cp.change_type, Some(ChangeType::AccelerationChange)))
        .count();
    
    if accel_changes > 0 {
        println!("    Detected {} acceleration changes", accel_changes);
    } else {
        println!("    No acceleration changes detected");
    }

    Ok(())
}

// Signal generation functions
fn create_linear_change() -> Vec<f64> {
    let mut data = Vec::with_capacity(200);
    
    // Steady state
    for i in 0..50 {
        data.push(10.0 + (i as f64) * 0.01);
    }
    
    // Linear increase (constant slope, zero acceleration)
    for i in 50..150 {
        data.push(10.5 + (i as f64 - 50.0) * 0.1);
    }
    
    // Back to steady state
    for i in 150..200 {
        data.push(20.5 + (i as f64 - 150.0) * 0.01);
    }
    
    data
}

fn create_quadratic_change() -> Vec<f64> {
    let mut data = Vec::with_capacity(200);
    
    // Steady state
    for i in 0..50 {
        data.push(10.0 + (i as f64) * 0.01);
    }
    
    // Quadratic increase (accelerating slope)
    for i in 50..150 {
        let t = (i as f64 - 50.0) / 100.0;
        data.push(10.5 + 5.0 * t * t);
    }
    
    // Back to steady state
    for i in 150..200 {
        data.push(15.5 + (i as f64 - 150.0) * 0.01);
    }
    
    data
}

fn create_exponential_change() -> Vec<f64> {
    let mut data = Vec::with_capacity(200);
    
    // Steady state
    for i in 0..50 {
        data.push(10.0 + (i as f64) * 0.01);
    }
    
    // Exponential growth (increasing acceleration)
    for i in 50..150 {
        let t = (i as f64 - 50.0) / 100.0;
        data.push(10.5 + 2.0 * (t * 2.0).exp());
    }
    
    // Plateau
    for i in 150..200 {
        data.push(25.0 + (i as f64 - 150.0) * 0.01);
    }
    
    data
}

fn create_step_change() -> Vec<f64> {
    let mut data = Vec::with_capacity(200);
    
    // First level
    for i in 0..50 {
        data.push(10.0 + (i as f64) * 0.01);
    }
    
    // Sudden jump (infinite acceleration at the transition)
    for i in 50..150 {
        data.push(20.0 + (i as f64 - 50.0) * 0.01);
    }
    
    // Another level
    for i in 150..200 {
        data.push(21.0 + (i as f64 - 150.0) * 0.01);
    }
    
    data
}
