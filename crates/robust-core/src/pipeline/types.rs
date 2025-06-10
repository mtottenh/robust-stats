//! Core types for the statistical analysis pipeline
//!
//! These fundamental types are used across multiple crates and define
//! the basic data structures for pipeline operations.

use std::ops::Range;
use serde::{Deserialize, Serialize};

/// Time range representation - can be index-based or time-based
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeRange<T = usize> {
    /// Start of the range (inclusive)
    pub start: T,
    /// End of the range (exclusive)
    pub end: T,
    /// Start index in the original data
    pub start_idx: usize,
    /// End index in the original data (exclusive)
    pub end_idx: usize,
}

impl TimeRange<usize> {
    /// Create a new index-based time range
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            start_idx: start,
            end_idx: end,
        }
    }
    
    /// Get the length of this range
    pub fn len(&self) -> usize {
        self.end - self.start
    }
    
    /// Check if the range is empty
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
    
    /// Convert to a standard Range
    pub fn as_range(&self) -> Range<usize> {
        self.start_idx..self.end_idx
    }
}

/// Segment classification with embedded metadata
/// 
/// Note: We keep using f64 for statistical metrics as they represent
/// derived values (mean, std dev, etc.) that should be floating point
/// regardless of the input data type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SegmentClass {
    /// Linear trend segment
    Ramp {
        /// Slope of the linear fit
        slope: f64,
        /// R-squared value of the fit
        r_squared: f64,
    },
    /// Stable mean segment
    Plateau {
        /// Mean value
        mean: f64,
        /// Standard deviation
        std_dev: f64,
    },
    /// Periodic/oscillatory segment
    Oscillatory {
        /// Dominant frequency in Hz
        dominant_freq: f64,
        /// Amplitude of oscillation
        amplitude: f64,
        /// Damping ratio (0 = undamped, 1 = critically damped)
        damping_ratio: f64,
    },
    /// Short-lived disturbance
    Transient {
        /// Duration in samples
        duration: f64,
        /// Peak magnitude
        peak_magnitude: f64,
    },
    /// Unclassified segment
    Unknown,
}

/// Stability verdict for a segment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Stability {
    /// Segment is stable within tolerance
    Stable,
    /// Segment shows some instability but within bounds
    Marginal,
    /// Segment is unstable
    Unstable,
}

/// Stability metrics for a segment
/// 
/// Note: All metrics are f64 as they represent statistical
/// computations that should be floating point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMetrics {
    /// Relative variance (CV²)
    pub relative_variance: f64,
    /// Rate of change
    pub change_rate: f64,
    /// Stationarity test p-value
    pub stationarity_p_value: f64,
    /// Custom metrics from specific analyzers
    pub custom_metrics: Vec<(String, f64)>,
}

/// Result of analyzing a single segment
#[derive(Debug, Clone)]
pub struct AnalyzedSegment<T = usize> {
    /// The time range of this segment
    pub range: TimeRange<T>,
    /// Classification of the segment
    pub class: SegmentClass,
    /// Stability verdict
    pub stability: Stability,
    /// Detailed stability metrics
    pub metrics: StabilityMetrics,
}

/// Pairing of segments for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentPairing {
    /// Unique identifier for this pairing
    pub id: uuid::Uuid,
    /// Segment from first series
    pub segment_a: TimeRange,
    /// Segment from second series
    pub segment_b: TimeRange,
    /// Confidence score for the match (0.0 to 1.0)
    pub match_confidence: f64,
    /// Type of match
    pub match_type: MatchType,
}

/// Type of segment match
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchType {
    /// Exact temporal alignment
    Exact,
    /// Overlapping time ranges
    Overlapping,
    /// Similar characteristics but different times
    Similar,
    /// Matched by sequence order
    Sequential,
}

/// Effect size magnitude interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectMagnitude {
    /// Effect size is negligible
    Negligible,
    /// Small effect
    Small,
    /// Medium effect
    Medium,
    /// Large effect
    Large,
}

impl EffectMagnitude {
    /// Interpret Cohen's d effect size
    pub fn from_cohen_d(d: f64) -> Self {
        let abs_d = d.abs();
        if abs_d < 0.2 {
            Self::Negligible
        } else if abs_d < 0.5 {
            Self::Small
        } else if abs_d < 0.8 {
            Self::Medium
        } else {
            Self::Large
        }
    }
    
    /// Interpret Cliff's delta effect size
    pub fn from_cliff_delta(delta: f64) -> Self {
        let abs_delta = delta.abs();
        if abs_delta < 0.147 {
            Self::Negligible
        } else if abs_delta < 0.33 {
            Self::Small
        } else if abs_delta < 0.474 {
            Self::Medium
        } else {
            Self::Large
        }
    }
}

/// Cache hint for pipeline operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheHint {
    /// This computation should be cached
    Cache,
    /// This computation should not be cached
    NoCache,
    /// Use default caching policy
    Default,
}

/// Cache priority for eviction
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CachePriority {
    /// Low priority - evict first
    Low,
    /// Normal priority
    Normal,
    /// High priority - evict last
    High,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_time_range() {
        let range = TimeRange::new(10, 20);
        assert_eq!(range.len(), 10);
        assert!(!range.is_empty());
        assert_eq!(range.as_range(), 10..20);
        
        let empty = TimeRange::new(10, 10);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }
    
    #[test]
    fn test_effect_magnitude() {
        assert_eq!(EffectMagnitude::from_cohen_d(0.1), EffectMagnitude::Negligible);
        assert_eq!(EffectMagnitude::from_cohen_d(0.3), EffectMagnitude::Small);
        assert_eq!(EffectMagnitude::from_cohen_d(0.6), EffectMagnitude::Medium);
        assert_eq!(EffectMagnitude::from_cohen_d(1.0), EffectMagnitude::Large);
        
        assert_eq!(EffectMagnitude::from_cliff_delta(0.1), EffectMagnitude::Negligible);
        assert_eq!(EffectMagnitude::from_cliff_delta(0.2), EffectMagnitude::Small);
        assert_eq!(EffectMagnitude::from_cliff_delta(0.4), EffectMagnitude::Medium);
        assert_eq!(EffectMagnitude::from_cliff_delta(0.5), EffectMagnitude::Large);
    }
}