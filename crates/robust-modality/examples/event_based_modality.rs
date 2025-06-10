//! Example demonstrating event-based modality detection
//!
//! This example shows how the modality detector can emit events to an EventBus
//! for integration with pipeline visualization and monitoring.

use robust_core::pipeline::events::{EventBus, EventHandler, PipelineEvent};
use robust_core::pipeline::context::PipelineContext;
use robust_core::{
    UnifiedWeightCache, CachePolicy, execution::scalar_sequential
};
use robust_histogram::QRDEBuilderWithSteps;
use robust_modality::{
    detector::LowlandModalityDetector,
    visualization::NullModalityVisualizer,
};
use robust_quantile::{estimators::harrell_davis, QuantileAdapter, HDWeightComputer};
use std::sync::{Arc, Mutex};

/// Simple event handler that logs modality detection events
#[derive(Default)]
struct LoggingEventHandler {
    events: Arc<Mutex<Vec<String>>>,
}

impl EventHandler for LoggingEventHandler {
    fn handle_event(&self, event: &PipelineEvent, _context: &PipelineContext) {
        let mut events = self.events.lock().unwrap();
        
        match event {
            PipelineEvent::ModalityDetectionStarted { data_len, parameters, .. } => {
                events.push(format!("Modality detection started with {} data points", data_len));
                for (key, value) in parameters {
                    events.push(format!("  Parameter {}: {}", key, value));
                }
            }
            PipelineEvent::ModalityHistogramCreated { bin_count, min_value, max_value, .. } => {
                events.push(format!("Histogram created: {} bins, range [{:.2}, {:.2}]", 
                    bin_count, min_value, max_value));
            }
            PipelineEvent::ModalityPeaksDetected { peak_indices, peak_heights, .. } => {
                events.push(format!("Detected {} peaks", peak_indices.len()));
                for (idx, height) in peak_indices.iter().zip(peak_heights.iter()) {
                    events.push(format!("  Peak at bin {}: height {:.3}", idx, height));
                }
            }
            PipelineEvent::ModalityWaterLevelTest { peak1_idx, peak2_idx, water_level, is_lowland, .. } => {
                let result = if *is_lowland { "LOWLAND FOUND" } else { "no lowland" };
                events.push(format!("Water level test between peaks {} and {}: level={:.3}, result={}", 
                    peak1_idx, peak2_idx, water_level, result));
            }
            PipelineEvent::ModesDetected { modes, is_multimodal, .. } => {
                events.push(format!("Final result: {} modes detected (multimodal: {})", 
                    modes.len(), is_multimodal));
                for (i, mode) in modes.iter().enumerate() {
                    events.push(format!("  Mode {}: location={:.2}, height={:.3}, bounds=[{:.2}, {:.2}]",
                        i + 1, mode.location, mode.height, mode.left_bound, mode.right_bound));
                }
            }
            _ => {} // Ignore other events
        }
    }
}

#[cfg(feature = "test-utils")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rand;
    // Create event bus and logging handler
    let event_bus = EventBus::new();
    let logger = LoggingEventHandler::default();
    let events_ref = logger.events.clone();
    event_bus.register(logger)?;
    
    // Create pipeline context
    let context = PipelineContext::new();
    
    // Create modality detector with event bus
    let histogram_builder = QRDEBuilderWithSteps::uniform(50);
    let detector = LowlandModalityDetector::new(
        histogram_builder,
        NullModalityVisualizer::default(),
        0.5,  // sensitivity
        0.05  // precision
    )
    .with_event_bus(event_bus.clone())
    .with_context(context.clone());
    
    // Create quantile estimator for the detector
    let engine = scalar_sequential();
    let hd = harrell_davis(engine);
    let estimator = QuantileAdapter::new(hd);
    let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
    
    // Test 1: Unimodal distribution
    println!("=== Test 1: Unimodal Distribution ===");
    let unimodal_data: Vec<f64> = (0..1000)
        .map(|_| rand::random::<f64>() * 2.0 - 1.0) // Uniform [-1, 1]
        .collect();
    
    let _ = detector.detect_modes_with_estimator(&unimodal_data, &estimator, &cache)?;
    
    // Print captured events
    let mut events = events_ref.lock().unwrap();
    for event in events.iter() {
        println!("{}", event);
    }
    events.clear();
    drop(events);
    
    // Test 2: Bimodal distribution
    println!("\n=== Test 2: Bimodal Distribution ===");
    let mut bimodal_data = Vec::new();
    for _ in 0..500 {
        bimodal_data.push(rand::random::<f64>() - 2.0); // Mode around -1.5
    }
    for _ in 0..500 {
        bimodal_data.push(rand::random::<f64>() + 2.0);  // Mode around 2.5
    }
    
    let _ = detector.detect_modes_with_estimator(&bimodal_data, &estimator, &cache)?;
    
    // Print captured events
    let events = events_ref.lock().unwrap();
    for event in events.iter() {
        println!("{}", event);
    }
    
    println!("\n=== Event-based modality detection complete! ===");
    
    Ok(())
}

#[cfg(not(feature = "test-utils"))]
fn main() {
    eprintln!("This example requires the 'test-utils' feature to be enabled.");
    eprintln!("Run with: cargo run --example event_based_modality --features test-utils");
}