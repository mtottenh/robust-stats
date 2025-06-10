//! Event-based pipeline notification system
//!
//! This module provides an extensible event system for pipeline operations,
//! allowing multiple consumers (visualization, logging, metrics) to observe
//! pipeline execution without tight coupling.

use super::types::*;
use crate::error::Result;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Pipeline event that can be emitted during execution
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    /// Pipeline execution started
    PipelineStarted {
        trace_id: Uuid,
        timestamp: std::time::Instant,
    },
    
    /// Pipeline execution completed
    PipelineCompleted {
        trace_id: Uuid,
        duration: std::time::Duration,
    },
    
    /// Error occurred during pipeline execution
    PipelineError {
        trace_id: Uuid,
        stage: &'static str,
        error: String,
    },
    
    /// Change point detection started
    ChangePointDetectionStarted {
        trace_id: Uuid,
        data_len: usize,
    },
    
    /// Change point detected
    ChangePointDetected {
        trace_id: Uuid,
        range: TimeRange,
        confidence: f64,
        change_type: Option<String>,
        metadata: HashMap<String, serde_json::Value>,
    },
    
    /// Segment classification started
    SegmentClassificationStarted {
        trace_id: Uuid,
        segment: TimeRange,
    },
    
    /// Segment classified
    SegmentClassified {
        trace_id: Uuid,
        range: TimeRange,
        class: String,
        confidence: Option<f64>,
        features: HashMap<String, f64>,
    },
    
    /// Stability analysis started
    StabilityAnalysisStarted {
        trace_id: Uuid,
        segment: TimeRange,
    },
    
    /// Stability analysis complete
    StabilityAnalysisComplete {
        trace_id: Uuid,
        metrics: HashMap<String, f64>,
        overall_stability: f64,
    },
    
    /// Comparison started between two segments
    ComparisonStarted {
        trace_id: Uuid,
        segment_a: TimeRange,
        segment_b: TimeRange,
    },
    
    /// Segment comparison completed
    SegmentComparisonCompleted {
        trace_id: Uuid,
        pairing_id: Uuid,
        metric_type: String,
        estimate: f64,
        ci_lower: f64,
        ci_upper: f64,
        is_significant: bool,
    },
    
    /// Full comparison completed
    ComparisonCompleted {
        trace_id: Uuid,
        total_comparisons: usize,
        significant_differences: usize,
    },
    
    /// Significant difference detected
    SignificantDifference {
        trace_id: Uuid,
        segment_pair: (TimeRange, TimeRange),
        metric: String,
        p_value: f64,
    },
    
    /// Data processed at a specific stage
    DataProcessed {
        trace_id: Uuid,
        stage: String,
        data: Vec<f64>,
        metadata: HashMap<String, serde_json::Value>,
    },
    
    /// Diagnostic information from a stage
    DiagnosticInfo {
        trace_id: Uuid,
        stage: String,
        info: HashMap<String, serde_json::Value>,
    },
    
    /// Modality detection started
    ModalityDetectionStarted {
        trace_id: Uuid,
        data_len: usize,
        parameters: HashMap<String, f64>,
    },
    
    /// Histogram created for modality analysis
    ModalityHistogramCreated {
        trace_id: Uuid,
        bin_count: usize,
        min_value: f64,
        max_value: f64,
    },
    
    /// Peaks detected in histogram
    ModalityPeaksDetected {
        trace_id: Uuid,
        peak_indices: Vec<usize>,
        peak_heights: Vec<f64>,
    },
    
    /// Water level test performed
    ModalityWaterLevelTest {
        trace_id: Uuid,
        peak1_idx: usize,
        peak2_idx: usize,
        water_level: f64,
        is_lowland: bool,
        underwater_bins: (usize, usize),
    },
    
    /// Modes detected
    ModesDetected {
        trace_id: Uuid,
        modes: Vec<ModeInfo>,
        is_multimodal: bool,
    },
    
    /// Custom event for extensions
    Custom {
        trace_id: Uuid,
        event_type: String,
        data: Arc<dyn Any + Send + Sync>,
    },
}

/// Information about a detected mode
#[derive(Debug, Clone)]
pub struct ModeInfo {
    /// Mode location (peak position)
    pub location: f64,
    /// Mode height (density)
    pub height: f64,
    /// Left boundary of the mode
    pub left_bound: f64,
    /// Right boundary of the mode
    pub right_bound: f64,
}

/// Trait for handling pipeline events
pub trait EventHandler: Send + Sync {
    /// Handle a pipeline event
    fn handle_event(&self, event: &PipelineEvent, context: &crate::pipeline::context::PipelineContext);
    
    /// Check if this handler is interested in a particular event type
    fn is_interested(&self, event: &PipelineEvent) -> bool {
        // By default, handlers are interested in all events
        let _ = event;
        true
    }
    
    /// Get the name of this handler for debugging
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// Event bus for distributing events to multiple handlers
pub struct EventBus {
    handlers: Arc<Mutex<Vec<Box<dyn EventHandler>>>>,
    enabled: Arc<Mutex<bool>>,
}

impl EventBus {
    /// Create a new event bus
    pub fn new() -> Self {
        Self {
            handlers: Arc::new(Mutex::new(Vec::new())),
            enabled: Arc::new(Mutex::new(true)),
        }
    }
    
    /// Register an event handler
    pub fn register<H>(&self, handler: H) -> Result<()>
    where
        H: EventHandler + 'static,
    {
        let mut handlers = self.handlers.lock()
            .map_err(|e| crate::error::Error::Execution(format!("Failed to lock handlers: {e}")))?;
        handlers.push(Box::new(handler));
        Ok(())
    }
    
    /// Emit an event to all registered handlers
    pub fn emit(&self, event: PipelineEvent, context: &super::context::PipelineContext) -> Result<()> {
        let enabled = self.enabled.lock()
            .map_err(|e| crate::error::Error::Execution(format!("Failed to check enabled state: {e}")))?;
        
        if !*enabled {
            return Ok(());
        }
        
        let handlers = self.handlers.lock()
            .map_err(|e| crate::error::Error::Execution(format!("Failed to lock handlers: {e}")))?;
        
        for handler in handlers.iter() {
            if handler.is_interested(&event) {
                // Continue on error to ensure all handlers get a chance
                handler.handle_event(&event, context);
            }
        }
        
        Ok(())
    }
    
    /// Publish an event (alias for emit)
    pub fn publish(&self, event: &PipelineEvent, context: &super::context::PipelineContext) {
        let _ = self.emit(event.clone(), context);
    }
    
    /// Enable or disable event emission
    pub fn set_enabled(&self, enabled: bool) -> Result<()> {
        let mut state = self.enabled.lock()
            .map_err(|e| crate::error::Error::Execution(format!("Failed to lock enabled state: {e}")))?;
        *state = enabled;
        Ok(())
    }
    
    /// Check if the event bus is enabled
    pub fn is_enabled(&self) -> Result<bool> {
        let state = self.enabled.lock()
            .map_err(|e| crate::error::Error::Execution(format!("Failed to check enabled state: {e}")))?;
        Ok(*state)
    }
    
    /// Get the number of registered handlers
    pub fn handler_count(&self) -> Result<usize> {
        let handlers = self.handlers.lock()
            .map_err(|e| crate::error::Error::Execution(format!("Failed to lock handlers: {e}")))?;
        Ok(handlers.len())
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for EventBus {
    fn clone(&self) -> Self {
        Self {
            handlers: Arc::clone(&self.handlers),
            enabled: Arc::clone(&self.enabled),
        }
    }
}

/// Simple logging event handler
pub struct LoggingHandler {
    level: log::Level,
}

impl LoggingHandler {
    /// Create a new logging handler
    pub fn new(level: log::Level) -> Self {
        Self { level }
    }
}

impl EventHandler for LoggingHandler {
    fn handle_event(&self, event: &PipelineEvent, _context: &super::context::PipelineContext) {
        match event {
            PipelineEvent::PipelineStarted { trace_id, .. } => {
                log::log!(self.level, "Pipeline started: {trace_id}");
            }
            PipelineEvent::PipelineCompleted { trace_id, duration } => {
                log::log!(self.level, "Pipeline completed: {trace_id} in {duration:?}");
            }
            PipelineEvent::PipelineError { trace_id, stage, error } => {
                log::error!("Pipeline error in {stage}: {error} (trace: {trace_id})");
            }
            PipelineEvent::SignificantDifference { trace_id, segment_pair, metric, p_value } => {
                log::log!(self.level, "Significant difference found: {metric} (p={p_value:.4}) between {segment_pair:?} (trace: {trace_id})");
            }
            _ => {
                log::trace!("Pipeline event: {event:?}");
            }
        }
    }
}

/// Metrics collection handler
pub struct MetricsHandler {
    metrics: Arc<Mutex<PipelineMetrics>>,
}

#[derive(Default)]
pub struct PipelineMetrics {
    pub total_runs: usize,
    pub total_segments: usize,
    pub total_comparisons: usize,
    pub significant_differences: usize,
    pub errors: HashMap<String, usize>,
}

impl Default for MetricsHandler {
    fn default() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(PipelineMetrics::default())),
        }
    }
}

impl MetricsHandler {
    /// Create a new metrics handler
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Get a snapshot of current metrics
    pub fn snapshot(&self) -> Result<PipelineMetrics> {
        let metrics = self.metrics.lock()
            .map_err(|e| crate::error::Error::Execution(format!("Failed to lock metrics: {e}")))?;
        Ok(PipelineMetrics {
            total_runs: metrics.total_runs,
            total_segments: metrics.total_segments,
            total_comparisons: metrics.total_comparisons,
            significant_differences: metrics.significant_differences,
            errors: metrics.errors.clone(),
        })
    }
}

impl EventHandler for MetricsHandler {
    fn handle_event(&self, event: &PipelineEvent, _context: &super::context::PipelineContext) {
        let Ok(mut metrics) = self.metrics.lock() else {
            log::error!("Failed to lock metrics");
            return;
        };
        
        match event {
            PipelineEvent::PipelineStarted { .. } => {
                metrics.total_runs += 1;
            }
            PipelineEvent::ChangePointDetected { .. } => {
                metrics.total_segments += 1;
            }
            PipelineEvent::ComparisonCompleted { total_comparisons, significant_differences, .. } => {
                metrics.total_comparisons += total_comparisons;
                metrics.significant_differences += significant_differences;
            }
            PipelineEvent::PipelineError { stage, .. } => {
                *metrics.errors.entry(stage.to_string()).or_insert(0) += 1;
            }
            _ => {}
        }
    }
}

/// Null event handler that does nothing
#[derive(Default, Clone)]
pub struct NullEventHandler;

impl EventHandler for NullEventHandler {
    fn handle_event(&self, _event: &PipelineEvent, _context: &super::context::PipelineContext) {
        // Do nothing
    }
    
    fn is_interested(&self, _event: &PipelineEvent) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::context::PipelineContext;
    
    #[test]
    fn test_event_bus() {
        let bus = EventBus::new();
        
        // Register a handler
        let handler = LoggingHandler::new(log::Level::Debug);
        bus.register(handler).unwrap();
        
        assert_eq!(bus.handler_count().unwrap(), 1);
        
        // Emit an event
        let event = PipelineEvent::PipelineStarted {
            trace_id: Uuid::new_v4(),
            timestamp: std::time::Instant::now(),
        };
        
        let context = PipelineContext::new();
        bus.emit(event, &context).unwrap();
    }
    
    #[test]
    fn test_metrics_handler() {
        let handler = MetricsHandler::new();
        let context = PipelineContext::new();
        
        // Simulate pipeline events
        handler.handle_event(&PipelineEvent::PipelineStarted {
            trace_id: Uuid::new_v4(),
            timestamp: std::time::Instant::now(),
        }, &context);
        
        handler.handle_event(&PipelineEvent::ChangePointDetected {
            trace_id: Uuid::new_v4(),
            range: TimeRange::new(0, 100),
            confidence: 0.95,
            change_type: Some("ramp".to_string()),
            metadata: HashMap::new(),
        }, &context);
        
        let metrics = handler.snapshot().unwrap();
        assert_eq!(metrics.total_runs, 1);
        assert_eq!(metrics.total_segments, 1);
    }
}