//! Concrete quantile estimator implementations

// Generic implementation that eliminates duplication
mod generic;

// Specific estimators using the generic implementation
mod harrell_davis;
mod trimmed_hd;

// Export from generic implementations
pub use generic::Generic;
pub use harrell_davis::{harrell_davis, HarrellDavis, HDState};
pub use trimmed_hd::{
    trimmed_harrell_davis, ConstantWidth, LinearWidth, SqrtWidth,
    TrimmedHarrellDavis, TrimmedHDState, WidthFunction,
    TrimmedHDSqrt, TrimmedHDLinear, TrimmedHDConstant,
};
