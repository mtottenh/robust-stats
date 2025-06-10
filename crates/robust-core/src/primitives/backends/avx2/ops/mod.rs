//! Operation-centric modules for AVX2 compute primitives
//!
//! Each operation is organized in its own module with type-specific implementations

pub mod sparse_weighted_sum;
pub mod dot_product;
pub mod sum;
pub mod sparse_tile;

// Re-export the operation traits for convenience
pub use sparse_weighted_sum::SparseWeightedSum;
pub use dot_product::DotProduct;
pub use sum::Sum;
pub use sparse_tile::SparseTile;