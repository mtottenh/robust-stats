//! Error types for quantile estimation

use thiserror::Error;

/// Errors that can occur during quantile estimation
#[derive(Error, Debug)]
pub enum Error {
    /// Empty data provided
    #[error("Cannot compute quantile of empty data")]
    EmptyData,
    
    /// Invalid quantile probability
    #[error("Quantile probability {p} must be in [0, 1]")]
    InvalidProbability { p: f64 },
    
    /// Invalid trimming width
    #[error("Trimming width {width} must be in (0, 1]")]
    InvalidWidth { width: f64 },
    
    /// Invalid confidence level
    #[error("Confidence level {level} must be in (0, 1)")]
    InvalidConfidenceLevel { level: f64 },
    
    /// Numerical computation error
    #[error("Numerical error: {0}")]
    Numerical(String),
    
    /// Core computation error
    #[error("Core computation error: {0}")]
    Core(#[from] robust_core::Error),
}

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;

// Helper functions
impl Error {
    /// Check if probability is valid
    pub fn check_probability(p: f64) -> Result<()> {
        if !(0.0..=1.0).contains(&p) {
            return Err(Error::InvalidProbability { p });
        }
        Ok(())
    }
    
    /// Check if width is valid
    pub fn check_width(width: f64) -> Result<()> {
        if width <= 0.0 || width > 1.0 {
            return Err(Error::InvalidWidth { width });
        }
        Ok(())
    }
    
    /// Check if data is non-empty
    pub fn check_non_empty(data: &[f64]) -> Result<()> {
        if data.is_empty() {
            return Err(Error::EmptyData);
        }
        Ok(())
    }
}
