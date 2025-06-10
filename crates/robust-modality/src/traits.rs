//! Core traits for modality detection

use robust_core::Numeric;

/// Trait for modality detection algorithms
pub trait ModalityDetector<T: Numeric = f64> 
where
    T::Float: std::fmt::Debug + PartialEq + num_traits::ToPrimitive,
{
    /// Get the sensitivity parameter (if applicable)
    fn sensitivity(&self) -> f64 {
        0.5 // Default sensitivity
    }

    /// Get the precision parameter (if applicable)  
    fn precision(&self) -> f64 {
        0.01 // Default precision
    }

    /// Check if this detector supports weighted samples
    fn supports_weighted_samples(&self) -> bool {
        false // Conservative default
    }
}
