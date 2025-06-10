//! Types used for changepoint detection

use std::fmt;

/// Represents a detected changepoint
#[derive(Debug, Clone, PartialEq)]
pub struct ChangePoint {
    /// Index in the time series where the change occurred
    pub index: usize,
    /// Confidence score for this changepoint (0.0 to 1.0)
    pub confidence: f64,
    /// Optional description of the type of change detected
    pub change_type: Option<ChangeType>,
}

impl ChangePoint {
    /// Create a new changepoint
    pub fn new(index: usize, confidence: f64) -> Self {
        Self {
            index,
            confidence,
            change_type: None,
        }
    }

    /// Create a new changepoint with a specified change type
    pub fn with_type(index: usize, confidence: f64, change_type: ChangeType) -> Self {
        Self {
            index,
            confidence,
            change_type: Some(change_type),
        }
    }
}

impl fmt::Display for ChangePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.change_type {
            Some(change_type) => write!(
                f,
                "ChangePoint {{ index: {}, confidence: {:.3}, type: {} }}",
                self.index, self.confidence, change_type
            ),
            None => write!(
                f,
                "ChangePoint {{ index: {}, confidence: {:.3} }}",
                self.index, self.confidence
            ),
        }
    }
}

/// Types of changes that can be detected
#[derive(Debug, Clone, PartialEq)]
pub enum ChangeType {
    /// Change in mean level
    MeanShift,
    /// Change in variance
    VarianceChange,
    /// Change in trend/slope
    TrendChange,
    /// Gradual drift
    Drift,
    /// Steady state detected (low variability)
    SteadyState,
    /// Acceleration change (second derivative change)
    AccelerationChange,
    /// Unknown or unspecified change
    Unknown,
}

impl fmt::Display for ChangeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChangeType::MeanShift => write!(f, "Mean Shift"),
            ChangeType::VarianceChange => write!(f, "Variance Change"),
            ChangeType::TrendChange => write!(f, "Trend Change"),
            ChangeType::Drift => write!(f, "Drift"),
            ChangeType::SteadyState => write!(f, "Steady State"),
            ChangeType::AccelerationChange => write!(f, "Acceleration Change"),
            ChangeType::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Result of changepoint detection
#[derive(Debug, Clone)]
pub struct ChangePointResult {
    /// List of detected changepoints
    changepoints: Vec<ChangePoint>,
    /// Algorithm used for detection
    algorithm: String,
    /// Total number of data points analyzed
    sample_size: usize,
    /// Detection statistics (algorithm-specific)
    statistics: Vec<f64>,
}

impl ChangePointResult {
    /// Create a new changepoint result
    pub fn new(
        changepoints: Vec<ChangePoint>,
        algorithm: String,
        sample_size: usize,
        statistics: Vec<f64>,
    ) -> Self {
        Self {
            changepoints,
            algorithm,
            sample_size,
            statistics,
        }
    }

    /// Get the detected changepoints
    pub fn changepoints(&self) -> &[ChangePoint] {
        &self.changepoints
    }

    /// Get the number of detected changepoints
    pub fn count(&self) -> usize {
        self.changepoints.len()
    }

    /// Check if any changepoints were detected
    pub fn has_changepoints(&self) -> bool {
        !self.changepoints.is_empty()
    }

    /// Get the algorithm name used for detection
    pub fn algorithm(&self) -> &str {
        &self.algorithm
    }

    /// Get the sample size that was analyzed
    pub fn sample_size(&self) -> usize {
        self.sample_size
    }

    /// Get the detection statistics
    pub fn statistics(&self) -> &[f64] {
        &self.statistics
    }

    /// Get changepoints with confidence above a threshold
    pub fn confident_changepoints(&self, min_confidence: f64) -> Vec<&ChangePoint> {
        self.changepoints
            .iter()
            .filter(|cp| cp.confidence >= min_confidence)
            .collect()
    }

    /// Get the most confident changepoint
    pub fn most_confident(&self) -> Option<&ChangePoint> {
        self.changepoints
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
    }
}

impl fmt::Display for ChangePointResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ChangePoint Detection Result:")?;
        writeln!(f, "  Algorithm: {}", self.algorithm)?;
        writeln!(f, "  Sample size: {}", self.sample_size)?;
        writeln!(f, "  Changepoints detected: {}", self.count())?;

        if !self.changepoints.is_empty() {
            writeln!(f, "  Detected changepoints:")?;
            for cp in &self.changepoints {
                writeln!(f, "    {}", cp)?;
            }
        }

        Ok(())
    }
}
