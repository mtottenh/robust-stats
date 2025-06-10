//! Error types for robust statistical analysis
//!
//! Provides a unified error type for all robust-stats crates.

use thiserror::Error;

/// Core error type for robust statistical operations
#[derive(Error, Debug)]
pub enum Error {
    /// Invalid parameter provided to a function
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Insufficient data for the requested operation
    #[error("Insufficient data: expected at least {expected} samples, got {actual}")]
    InsufficientData { expected: usize, actual: usize },

    /// Numerical computation error
    #[error("Computation error: {0}")]
    Computation(String),

    /// Memory allocation or workspace error
    #[error("Memory error: {0}")]
    Memory(String),

    /// Threading or parallelization error
    #[error("Execution error: {0}")]
    Execution(String),

    /// Feature not available
    #[error("Feature not available: {0}")]
    FeatureNotAvailable(String),

    /// Cache-related error
    #[error("Cache error: {0}")]
    Cache(String),

    /// IO error (for file operations)
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Other errors
    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

/// Result type alias using our Error type
pub type Result<T> = std::result::Result<T, Error>;

// Helper functions for common error patterns

impl Error {
    /// Create an error for empty input
    pub fn empty_input(_operation: &str) -> Self {
        Self::InsufficientData {
            expected: 1,
            actual: 0,
        }
    }
    
    /// Create an error for invalid quantile
    pub fn invalid_quantile(p: f64) -> Self {
        Self::InvalidParameter(format!("Quantile {p} must be in [0, 1]"))
    }
    
    /// Create an error for size mismatch
    pub fn size_mismatch(expected: usize, actual: usize, context: &str) -> Self {
        Self::InvalidInput(format!(
            "Size mismatch in {context}: expected {expected}, got {actual}"
        ))
    }
    
    /// Create an error for NaN/Inf values
    pub fn non_finite(context: &str) -> Self {
        Self::Computation(format!("{context} contains NaN or infinite values"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        // Test each error variant's display implementation
        let err = Error::InvalidParameter("alpha must be positive".to_string());
        assert_eq!(err.to_string(), "Invalid parameter: alpha must be positive");
        
        let err = Error::InvalidInput("data contains duplicates".to_string());
        assert_eq!(err.to_string(), "Invalid input: data contains duplicates");
        
        let err = Error::InsufficientData { expected: 10, actual: 5 };
        assert_eq!(err.to_string(), "Insufficient data: expected at least 10 samples, got 5");
        
        let err = Error::Computation("convergence failed".to_string());
        assert_eq!(err.to_string(), "Computation error: convergence failed");
        
        let err = Error::Memory("allocation failed".to_string());
        assert_eq!(err.to_string(), "Memory error: allocation failed");
        
        let err = Error::Execution("thread pool exhausted".to_string());
        assert_eq!(err.to_string(), "Execution error: thread pool exhausted");
        
        let err = Error::FeatureNotAvailable("AVX512 required".to_string());
        assert_eq!(err.to_string(), "Feature not available: AVX512 required");
        
        let err = Error::Cache("cache miss".to_string());
        assert_eq!(err.to_string(), "Cache error: cache miss");
    }
    
    #[test]
    fn test_error_helper_functions() {
        // Test empty_input
        let err = Error::empty_input("quantile estimation");
        match err {
            Error::InsufficientData { expected, actual } => {
                assert_eq!(expected, 1);
                assert_eq!(actual, 0);
            }
            _ => panic!("Wrong error type"),
        }
        
        // Test invalid_quantile
        let err = Error::invalid_quantile(1.5);
        assert_eq!(err.to_string(), "Invalid parameter: Quantile 1.5 must be in [0, 1]");
        
        let err = Error::invalid_quantile(-0.1);
        assert_eq!(err.to_string(), "Invalid parameter: Quantile -0.1 must be in [0, 1]");
        
        // Test size_mismatch
        let err = Error::size_mismatch(100, 50, "weight vector");
        assert_eq!(err.to_string(), "Invalid input: Size mismatch in weight vector: expected 100, got 50");
        
        // Test non_finite
        let err = Error::non_finite("input data");
        assert_eq!(err.to_string(), "Computation error: input data contains NaN or infinite values");
    }
    
    #[test]
    fn test_error_from_io_error() {
        use std::io;
        
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        
        match err {
            Error::Io(_) => {
                assert!(err.to_string().contains("file not found"));
            }
            _ => panic!("Wrong error type"),
        }
    }
    
    #[test]
    fn test_error_from_anyhow() {
        let anyhow_err = anyhow::anyhow!("custom error message");
        let err: Error = anyhow_err.into();
        
        match err {
            Error::Other(_) => {
                assert!(err.to_string().contains("custom error message"));
            }
            _ => panic!("Wrong error type"),
        }
    }
    
    #[test]
    fn test_result_type_alias() {
        // Test that Result<T> works as expected
        fn test_function(succeed: bool) -> Result<i32> {
            if succeed {
                Ok(42)
            } else {
                Err(Error::Computation("test failure".to_string()))
            }
        }
        
        assert_eq!(test_function(true).unwrap(), 42);
        assert!(test_function(false).is_err());
    }
    
    #[test]
    fn test_error_debug_impl() {
        // Test Debug implementation
        let err = Error::InvalidParameter("test".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InvalidParameter"));
        assert!(debug_str.contains("test"));
    }
    
    #[test]
    fn test_error_edge_cases() {
        // Test with empty strings
        let err = Error::InvalidParameter("".to_string());
        assert_eq!(err.to_string(), "Invalid parameter: ");
        
        // Test with very long strings
        let long_str = "a".repeat(1000);
        let err = Error::Computation(long_str.clone());
        assert_eq!(err.to_string(), format!("Computation error: {}", long_str));
        
        // Test InsufficientData with edge values
        let err = Error::InsufficientData { expected: 0, actual: 0 };
        assert_eq!(err.to_string(), "Insufficient data: expected at least 0 samples, got 0");
        
        let err = Error::InsufficientData { expected: usize::MAX, actual: 0 };
        assert_eq!(err.to_string(), format!("Insufficient data: expected at least {} samples, got 0", usize::MAX));
    }
    
    #[test]
    fn test_error_patterns() {
        // Common patterns in statistical code
        
        // Pattern 1: Check minimum sample size
        fn check_sample_size(data: &[f64], min_size: usize) -> Result<()> {
            if data.len() < min_size {
                return Err(Error::InsufficientData {
                    expected: min_size,
                    actual: data.len(),
                });
            }
            Ok(())
        }
        
        assert!(check_sample_size(&[1.0, 2.0], 5).is_err());
        assert!(check_sample_size(&[1.0, 2.0, 3.0, 4.0, 5.0], 5).is_ok());
        
        // Pattern 2: Validate parameters
        fn validate_probability(p: f64) -> Result<()> {
            if !(0.0..=1.0).contains(&p) {
                return Err(Error::invalid_quantile(p));
            }
            Ok(())
        }
        
        assert!(validate_probability(0.5).is_ok());
        assert!(validate_probability(1.1).is_err());
        assert!(validate_probability(-0.1).is_err());
        
        // Pattern 3: Check for finite values
        fn check_finite(data: &[f64]) -> Result<()> {
            if data.iter().any(|&x| !x.is_finite()) {
                return Err(Error::non_finite("data"));
            }
            Ok(())
        }
        
        assert!(check_finite(&[1.0, 2.0, 3.0]).is_ok());
        assert!(check_finite(&[1.0, f64::NAN, 3.0]).is_err());
        assert!(check_finite(&[1.0, f64::INFINITY, 3.0]).is_err());
    }
    
    #[test]
    fn test_error_chaining() {
        // Test that errors can be chained/wrapped
        fn inner_function() -> Result<()> {
            Err(Error::Computation("inner error".to_string()))
        }
        
        fn outer_function() -> Result<()> {
            inner_function().map_err(|e| {
                Error::Execution(format!("outer error: {}", e))
            })
        }
        
        let result = outer_function();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("outer error"));
        assert!(err.to_string().contains("inner error"));
    }
}