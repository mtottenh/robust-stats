use std::f64::consts::PI;
use hilbert_tf::HilbertTransform;

fn main() {
    // Create the same oscillatory signal as in the example
    let n = 256;
    let frequency = 0.05;  // 0.05 cycles per sample
    let signal: Vec<f64> = (0..n)
        .map(|i| 10.0 + 2.0 * (2.0 * PI * frequency * i as f64).sin())
        .collect();
    
    println!("Signal info:");
    println!("- Length: {n}");
    println!("- True frequency: {frequency} cycles/sample");
    println!("- First 10 samples: {:?}", &signal[..10]);
    
    // Create Hilbert transformer
    let transformer = HilbertTransform::new(n);
    
    // Get instantaneous phase
    let phase = transformer.instantaneous_phase(&signal).unwrap();
    println!("\nPhase (first 10): {:?}", &phase[..10]);
    
    // Get instantaneous frequency (sampling rate = 1.0)
    let inst_freq = transformer.instantaneous_frequency(&signal, 1.0).unwrap();
    println!("\nInstantaneous frequency (first 10): {:?}", &inst_freq[..10]);
    
    // Calculate statistics
    let median_freq = median(&inst_freq);
    let mad_freq = mad(&inst_freq, median_freq);
    let cv = if median_freq.abs() > f64::EPSILON {
        mad_freq / median_freq.abs()
    } else {
        f64::INFINITY
    };
    
    println!("\nFrequency statistics:");
    println!("- Median: {median_freq:.6}");
    println!("- MAD: {mad_freq:.6}");
    println!("- CV: {cv:.6}");
    println!("- Expected frequency: {frequency:.6}");
    
    // Check the phase unwrapping by manually computing frequency
    println!("\nManual frequency calculation from phase:");
    for i in 1..10 {
        let phase_diff = phase[i] - phase[i-1];
        let adjusted_diff = if phase_diff > PI {
            phase_diff - 2.0 * PI
        } else if phase_diff < -PI {
            phase_diff + 2.0 * PI
        } else {
            phase_diff
        };
        let freq = adjusted_diff / (2.0 * PI);
        println!("  i={i}: phase_diff={phase_diff:.4}, adjusted={adjusted_diff:.4}, freq={freq:.4}");
    }
}

fn median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    }
}

fn mad(data: &[f64], median_val: f64) -> f64 {
    let deviations: Vec<f64> = data.iter().map(|&x| (x - median_val).abs()).collect();
    median(&deviations) * 1.4826  // Scale factor for consistency with std dev
}