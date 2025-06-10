//! Quantile-specific kernel implementations

mod weighted_sum;
mod tiled_weighted_sum;

pub use weighted_sum::WeightedSumKernel;
pub use tiled_weighted_sum::TiledWeightedSumKernel;

use robust_core::{ExecutionEngine, Numeric};

/// Create a kernel for the given execution engine
pub fn create_kernel<T: Numeric, E: ExecutionEngine<T>>(engine: &E) -> WeightedSumKernel<T, E> {
    WeightedSumKernel::new(engine.clone())
}
