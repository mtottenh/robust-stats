//! Error types for robust-polars

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Polars error: {0}")]
    Polars(#[from] polars::error::PolarsError),
    
    #[error("Robust core error: {0}")]
    RobustCore(#[from] robust_core::Error),
    
    #[error("Invalid column: {0}")]
    InvalidColumn(String),
    
    #[error("Type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: String, got: String },
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Computation failed: {0}")]
    ComputationFailed(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, Error>;