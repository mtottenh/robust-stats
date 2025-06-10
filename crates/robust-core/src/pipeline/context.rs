//! Pipeline execution context
//!
//! The PipelineContext flows through all stages of the pipeline, accumulating
//! metadata, timing information, and cache hints.

use super::types::{CacheHint, CachePriority};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Dynamic value type for metadata
#[derive(Debug, Clone)]
pub enum Value {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Integer(i64),
    /// Floating point value
    Float(f64),
    /// String value
    String(String),
    /// Binary data
    Bytes(Vec<u8>),
    /// Duration value
    Duration(Duration),
}

impl Value {
    /// Try to get as boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
    
    /// Try to get as integer
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Value::Integer(i) => Some(*i),
            _ => None,
        }
    }
    
    /// Try to get as float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }
    
    /// Try to get as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Context that flows through the pipeline execution
#[derive(Debug, Clone)]
pub struct PipelineContext {
    /// Unique trace ID for this execution
    pub trace_id: Uuid,
    /// When the pipeline started
    pub start_time: Instant,
    /// Arbitrary metadata
    pub metadata: HashMap<String, Value>,
    /// Cache hints accumulated during execution
    pub cache_hints: Vec<CacheHint>,
    /// Stage timing information
    stage_timings: HashMap<String, Duration>,
    /// Parent context (for forked contexts)
    parent_trace_id: Option<Uuid>,
}

impl PipelineContext {
    /// Create a new pipeline context
    pub fn new() -> Self {
        Self {
            trace_id: Uuid::new_v4(),
            start_time: Instant::now(),
            metadata: HashMap::new(),
            cache_hints: Vec::new(),
            stage_timings: HashMap::new(),
            parent_trace_id: None,
        }
    }
    
    /// Create a context with a specific trace ID
    pub fn with_trace_id(trace_id: Uuid) -> Self {
        Self {
            trace_id,
            start_time: Instant::now(),
            metadata: HashMap::new(),
            cache_hints: Vec::new(),
            stage_timings: HashMap::new(),
            parent_trace_id: None,
        }
    }
    
    /// Fork this context for parallel operations
    pub fn fork(&self) -> Self {
        Self {
            trace_id: Uuid::new_v4(),
            start_time: Instant::now(),
            metadata: self.metadata.clone(),
            cache_hints: Vec::new(),
            stage_timings: HashMap::new(),
            parent_trace_id: Some(self.trace_id),
        }
    }
    
    /// Fork for comparison operations (returns two related contexts)
    pub fn fork_for_comparison(&self) -> (Self, Self) {
        let parent_id = self.trace_id;
        let mut ctx_a = self.fork();
        let mut ctx_b = self.fork();
        
        // Mark these as comparison contexts
        ctx_a.set_metadata("comparison_side", Value::String("A".to_string()));
        ctx_b.set_metadata("comparison_side", Value::String("B".to_string()));
        ctx_a.set_metadata("comparison_parent", Value::String(parent_id.to_string()));
        ctx_b.set_metadata("comparison_parent", Value::String(parent_id.to_string()));
        
        (ctx_a, ctx_b)
    }
    
    /// Set a metadata value
    pub fn set_metadata(&mut self, key: impl Into<String>, value: Value) {
        self.metadata.insert(key.into(), value);
    }
    
    /// Get a metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }
    
    /// Add a cache hint
    pub fn add_cache_hint(&mut self, hint: CacheHint) {
        self.cache_hints.push(hint);
    }
    
    /// Record timing for a stage
    pub fn record_stage_timing(&mut self, stage: impl Into<String>, duration: Duration) {
        self.stage_timings.insert(stage.into(), duration);
    }
    
    /// Time a stage execution
    pub fn time_stage<F, R>(&mut self, stage: impl Into<String>, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let stage_name = stage.into();
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        self.record_stage_timing(stage_name, duration);
        result
    }
    
    /// Get total elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    /// Get stage timings
    pub fn stage_timings(&self) -> &HashMap<String, Duration> {
        &self.stage_timings
    }
    
    /// Get parent trace ID if this is a forked context
    pub fn parent_trace_id(&self) -> Option<Uuid> {
        self.parent_trace_id
    }
    
    /// Check if this context suggests caching
    pub fn should_cache(&self) -> bool {
        // If any hint says to cache, we cache
        self.cache_hints.iter().any(|h| matches!(h, CacheHint::Cache))
    }
    
    /// Get the highest cache priority from hints
    pub fn cache_priority(&self) -> CachePriority {
        // For now, return Normal. Could be extended to track priorities
        CachePriority::Normal
    }
}

impl Default for PipelineContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for pipeline context
#[derive(Default)]
pub struct PipelineContextBuilder {
    context: PipelineContext,
}

impl PipelineContextBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set trace ID
    pub fn trace_id(mut self, trace_id: Uuid) -> Self {
        self.context.trace_id = trace_id;
        self
    }
    
    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.context.set_metadata(key, value);
        self
    }
    
    /// Add cache hint
    pub fn cache_hint(mut self, hint: CacheHint) -> Self {
        self.context.add_cache_hint(hint);
        self
    }
    
    /// Build the context
    pub fn build(self) -> PipelineContext {
        self.context
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_context_creation() {
        let ctx = PipelineContext::new();
        assert_eq!(ctx.parent_trace_id(), None);
        assert!(ctx.metadata.is_empty());
    }
    
    #[test]
    fn test_context_fork() {
        let mut parent = PipelineContext::new();
        parent.set_metadata("key", Value::String("value".to_string()));
        
        let child = parent.fork();
        assert_eq!(child.parent_trace_id(), Some(parent.trace_id));
        assert_eq!(
            child.get_metadata("key").and_then(|v| v.as_string()),
            Some("value")
        );
    }
    
    #[test]
    fn test_context_timing() {
        let mut ctx = PipelineContext::new();
        
        let result = ctx.time_stage("test_stage", || {
            std::thread::sleep(Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        assert!(ctx.stage_timings.contains_key("test_stage"));
        assert!(ctx.stage_timings["test_stage"] >= Duration::from_millis(10));
    }
    
    #[test]
    fn test_value_conversions() {
        assert_eq!(Value::Bool(true).as_bool(), Some(true));
        assert_eq!(Value::Integer(42).as_integer(), Some(42));
        assert_eq!(Value::Float(3.5).as_float(), Some(3.5));
        assert_eq!(Value::Integer(42).as_float(), Some(42.0));
        assert_eq!(Value::String("test".to_string()).as_string(), Some("test"));
    }
}