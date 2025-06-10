//! Realistic changepoint detection scenarios
//!
//! This example demonstrates changepoint detection on realistic time series data with:
//! 1. Warm-up periods
//! 2. Minor oscillations/noise
//! 3. Oscillations of increasing magnitude

use robust_changepoint::{CusumDetector, EwmaDetector, SlopesDetector};
use robust_changepoint::{ChangePointDetector, SimpleDetector};
use robust_core::execution::simd_sequential;
use robust_core::{ExecutionEngine, UnifiedWeightCache, CachePolicy};
use robust_spread::StandardizedMad;
use robust_quantile::estimators::harrell_davis;
use robust_quantile::HDWeightComputer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Realistic Changepoint Detection Scenarios ===\n");

    // Set up shared components for the new API (unused in main, used in analyze_scenario)
    // These were previously initialized here but are now created inside analyze_scenario

    // Scenario 1: System with warm-up period and minor noise
    println!("📊 Scenario 1: System with Warm-up + Minor Noise");
    let data1 = create_warmup_with_noise(200, 50, 5.0, 0.2);
    analyze_scenario("Warm-up + Minor Noise", &data1)?;

    // Scenario 2: System with warm-up and increasing oscillations
    println!("\n📊 Scenario 2: System with Warm-up + Increasing Oscillations");
    let data2 = create_warmup_with_increasing_oscillations(200, 40, 5.0);
    analyze_scenario("Warm-up + Increasing Oscillations", &data2)?;

    // Scenario 3: Complex realistic scenario
    println!("\n📊 Scenario 3: Complex Multi-Phase System");
    let data3 = create_complex_realistic_scenario(300);
    analyze_scenario("Complex Multi-Phase", &data3)?;

    // Scenario 4: Process monitoring scenario
    println!("\n📊 Scenario 4: Process Monitoring with Drift");
    let data4 = create_process_monitoring_scenario(250);
    analyze_scenario("Process Monitoring", &data4)?;

    Ok(())
}

/// Creates a time series with warm-up period followed by steady state with minor noise
fn create_warmup_with_noise(
    total_length: usize,
    warmup_length: usize,
    steady_level: f64,
    noise_std: f64,
) -> Vec<f64> {
    let mut data = Vec::with_capacity(total_length);

    // Warm-up period: exponential approach to steady state
    for i in 0..warmup_length {
        let progress = i as f64 / warmup_length as f64;
        let warmup_value = steady_level * (1.0 - (-3.0 * progress).exp());
        data.push(warmup_value);
    }

    // Steady state with minor noise
    for _ in warmup_length..total_length {
        let noise = (rand::random::<f64>() - 0.5) * 2.0 * noise_std;
        data.push(steady_level + noise);
    }

    data
}

/// Creates a time series with warm-up followed by increasing oscillations
fn create_warmup_with_increasing_oscillations(
    total_length: usize,
    warmup_length: usize,
    base_level: f64,
) -> Vec<f64> {
    let mut data = Vec::with_capacity(total_length);

    // Warm-up period
    for i in 0..warmup_length {
        let progress = i as f64 / warmup_length as f64;
        let warmup_value = base_level * (1.0 - (-2.5 * progress).exp());
        data.push(warmup_value);
    }

    // Oscillations with increasing magnitude
    for i in warmup_length..total_length {
        let phase = (i - warmup_length) as f64;
        let oscillation_magnitude = 0.1 + 0.01 * phase; // Increasing amplitude
        let oscillation = oscillation_magnitude * (0.2 * phase).sin();
        data.push(base_level + oscillation);
    }

    data
}

/// Creates a complex multi-phase realistic scenario
fn create_complex_realistic_scenario(total_length: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(total_length);
    
    for i in 0..total_length {
        let phase = i as f64;
        let value = match i {
            // Phase 1: Stable baseline with minor noise
            0..=79 => 3.0 + 0.1 * (rand::random::<f64>() - 0.5),
            
            // Phase 2: Gradual increase (trend change)
            80..=139 => 3.0 + 0.05 * (phase - 80.0) + 0.1 * (rand::random::<f64>() - 0.5),
            
            // Phase 3: High variability period
            140..=199 => {
                let base = 6.0;
                let variability = 0.8 * (0.1 * phase).sin();
                base + variability + 0.2 * (rand::random::<f64>() - 0.5)
            },
            
            // Phase 4: Abrupt shift to new level
            200..=249 => 9.0 + 0.15 * (rand::random::<f64>() - 0.5),
            
            // Phase 5: Decline with increasing noise
            _ => {
                let decline = 9.0 - 0.02 * (phase - 250.0);
                let noise_level = 0.1 + 0.005 * (phase - 250.0);
                decline + noise_level * (rand::random::<f64>() - 0.5)
            }
        };
        data.push(value);
    }
    
    data
}

/// Creates a process monitoring scenario with drift
fn create_process_monitoring_scenario(total_length: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(total_length);
    
    let mut baseline = 10.0;
    let mut _drift_rate = 0.0;
    
    for i in 0..total_length {
        let value = match i {
            // Normal operation
            0..=99 => baseline + 0.2 * (rand::random::<f64>() - 0.5),
            
            // Start of gradual drift
            100..=149 => {
                _drift_rate = 0.01;
                baseline += _drift_rate;
                baseline + 0.2 * (rand::random::<f64>() - 0.5)
            },
            
            // Acceleration of drift
            150..=199 => {
                _drift_rate = 0.03;
                baseline += _drift_rate;
                baseline + 0.3 * (rand::random::<f64>() - 0.5)
            },
            
            // Process correction (step change)
            _ => {
                if i == 200 {
                    baseline = 10.0; // Reset to target
                }
                baseline + 0.2 * (rand::random::<f64>() - 0.5)
            }
        };
        data.push(value);
    }
    
    data
}

/// Analyze a scenario using all available detectors
fn analyze_scenario(
    _name: &str, 
    data: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    // Set up components for analysis
    let engine = simd_sequential();
    let primitives = engine.primitives().clone();
    let quantile_est = harrell_davis(engine.clone());
    let spread_est = StandardizedMad::<f64, _>::new(primitives.clone());
    let cache = UnifiedWeightCache::new(
        HDWeightComputer::new(),
        CachePolicy::Lru { max_entries: 256 },
    );
    println!("  Data length: {} points", data.len());
    println!(
        "  Data range: [{:.3}, {:.3}]",
        data.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // CUSUM Analysis
    let cusum = CusumDetector::new(primitives.clone(), 2.0, 0.3);
    let cusum_result = cusum.detect_simple(data)?;
    println!(
        "  🎯 CUSUM (threshold=2.0): {} changepoints",
        cusum_result.changepoints().len()
    );
    if !cusum_result.changepoints().is_empty() {
        let confident_changes: Vec<_> = cusum_result.changepoints().iter()
            .filter(|cp| cp.confidence > 0.7)
            .collect();
        println!(
            "     High confidence (>0.7): {} changepoints",
            confident_changes.len()
        );
        if !confident_changes.is_empty() {
            let indices: Vec<String> = confident_changes
                .iter()
                .take(5) // Show first 5
                .map(|cp| format!("{}", cp.index))
                .collect();
            println!(
                "     Locations: [{}{}]",
                indices.join(", "),
                if confident_changes.len() > 5 {
                    ", ..."
                } else {
                    ""
                }
            );
        }
    }

    // Slopes Analysis
    let slopes = SlopesDetector::new(primitives.clone(), 15, 3.0);
    let slopes_result = slopes.detect(data, &spread_est, &quantile_est, &cache)?;
    println!(
        "  📈 Slopes (window=15): {} trend changes",
        slopes_result.changepoints().len()
    );
    if !slopes_result.changepoints().is_empty() {
        let indices: Vec<String> = slopes_result
            .changepoints()
            .iter()
            .take(5)
            .map(|cp| format!("{}(conf:{:.2})", cp.index, cp.confidence))
            .collect();
        println!(
            "     Locations: [{}{}]",
            indices.join(", "),
            if slopes_result.changepoints().len() > 5 {
                ", ..."
            } else {
                ""
            }
        );
    }

    // EWMA Analysis
    let ewma = EwmaDetector::new(primitives.clone(), 0.3, 2.5);
    let ewma_result = ewma.detect(data, &spread_est, &quantile_est, &cache)?;
    println!(
        "  📊 EWMA (λ=0.3): {} drift detections",
        ewma_result.changepoints().len()
    );
    if !ewma_result.changepoints().is_empty() {
        let indices: Vec<String> = ewma_result
            .changepoints()
            .iter()
            .take(5)
            .map(|cp| format!("{}(conf:{:.2})", cp.index, cp.confidence))
            .collect();
        println!(
            "     Locations: [{}{}]",
            indices.join(", "),
            if ewma_result.changepoints().len() > 5 {
                ", ..."
            } else {
                ""
            }
        );
    }

    Ok(())
}

// Simple pseudo-random number generator for consistent results
mod rand {
    static mut SEED: u64 = 1;
    
    pub fn random<T>() -> T 
    where 
        T: From<f64>
    {
        unsafe {
            SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
            let value = ((SEED / 65536) % 32768) as f64 / 32768.0;
            T::from(value)
        }
    }
}