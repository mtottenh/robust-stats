//! Validation example that generates outputs for comparison with Python

use anyhow::Result;
use hilbert_tf::HilbertTransform;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Save data to CSV file
fn save_to_csv(path: &Path, data: &[f64], header: &str) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "index,{header}")?;
    
    for (i, value) in data.iter().enumerate() {
        writeln!(file, "{i},{value}")?;
    }
    
    Ok(())
}

/// Load data from CSV file
fn load_from_csv(path: &Path) -> Result<Vec<f64>> {
    let content = std::fs::read_to_string(path)?;
    let mut data = Vec::new();
    
    for (i, line) in content.lines().enumerate() {
        if i == 0 { continue; } // Skip header
        
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            let value: f64 = parts[1].parse()?;
            data.push(value);
        }
    }
    
    Ok(data)
}

fn main() -> Result<()> {
    println!("Hilbert Transform Validation");
    println!("============================\n");
    
    // Create output directory
    std::fs::create_dir_all("validation_output")?;
    
    // Test signals to process
    let test_signals = vec![
        "sine", "cosine", "am_signal", "chirp", "noise", "step", "mixture"
    ];
    
    for signal_name in test_signals {
        println!("Processing {signal_name}...");
        
        // Load input signal
        let input_path = format!("test_data/{signal_name}_input.csv");
        let signal = load_from_csv(Path::new(&input_path))?;
        
        // Create transformer
        let transformer = HilbertTransform::new(signal.len());
        
        // Compute Hilbert transform
        let hilbert = transformer.transform(&signal)?;
        save_to_csv(
            Path::new(&format!("validation_output/{signal_name}_hilbert_transform.csv")),
            &hilbert,
            "value"
        )?;
        
        // Compute envelope
        let envelope = transformer.envelope(&signal)?;
        save_to_csv(
            Path::new(&format!("validation_output/{signal_name}_envelope.csv")),
            &envelope,
            "value"
        )?;
        
        // Compute instantaneous phase
        let phase = transformer.instantaneous_phase(&signal)?;
        save_to_csv(
            Path::new(&format!("validation_output/{signal_name}_phase.csv")),
            &phase,
            "value"
        )?;
        
        // Compute instantaneous frequency (derivative of unwrapped phase)
        let mut inst_freq = Vec::with_capacity(phase.len());
        
        // Unwrap phase first
        let mut unwrapped_phase = vec![phase[0]];
        for i in 1..phase.len() {
            let diff = phase[i] - phase[i-1];
            let adjusted_diff = if diff > std::f64::consts::PI {
                diff - 2.0 * std::f64::consts::PI
            } else if diff < -std::f64::consts::PI {
                diff + 2.0 * std::f64::consts::PI
            } else {
                diff
            };
            unwrapped_phase.push(unwrapped_phase[i-1] + adjusted_diff);
        }
        
        // Compute frequency (forward difference)
        for i in 0..unwrapped_phase.len()-1 {
            inst_freq.push((unwrapped_phase[i+1] - unwrapped_phase[i]) / (2.0 * std::f64::consts::PI));
        }
        // Duplicate last value to maintain length
        if let Some(&last) = inst_freq.last() {
            inst_freq.push(last);
        }
        
        save_to_csv(
            Path::new(&format!("validation_output/{signal_name}_instantaneous_frequency.csv")),
            &inst_freq,
            "value"
        )?;
    }
    
    println!("\nValidation outputs saved to validation_output/");
    println!("Run the Python validation script to compare results.");
    
    Ok(())
}