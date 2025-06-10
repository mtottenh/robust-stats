//! Sparse data structures for efficient computation
//!
//! This module provides sparse representations that are useful across
//! various statistical algorithms. These are fundamental data structures
//! that enable efficient computation when dealing with mostly-zero data.

use crate::numeric::Numeric;
use num_traits::Float;

/// Sparse representation of weights or values for any numeric type
///
/// Many statistical algorithms work with weights where most values are near zero.
/// This representation only stores non-zero values and their indices, providing
/// significant memory and computational savings.
///
/// # Examples
///
/// ```
/// use robust_core::sparse::SparseWeights;
///
/// // Create from indices and weights (f64)
/// let indices = vec![1, 3, 7];
/// let weights = vec![0.2, 0.5, 0.3];
/// let sparse = SparseWeights::new(indices, weights, 10);
///
/// // Access weights
/// assert_eq!(sparse.get(1), 0.2);
/// assert_eq!(sparse.get(2), 0.0); // Not in sparse representation
/// 
/// // Also works with f32
/// let weights_f32 = vec![0.2f32, 0.5, 0.3];
/// let sparse_f32 = SparseWeights::new(vec![1, 3, 7], weights_f32, 10);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SparseWeights<T: Numeric = f64> {
    /// Indices of non-zero weights (sorted)
    pub indices: Vec<usize>,
    /// Values of non-zero weights (same length as indices)
    pub weights: Vec<T>,
    /// Total number of elements (for validation)
    pub n: usize,
}

impl<T: Numeric> SparseWeights<T> {
    /// Create new sparse weights
    ///
    /// # Panics
    /// - If indices and weights have different lengths
    /// - If any index is >= n
    pub fn new(indices: Vec<usize>, weights: Vec<T>, n: usize) -> Self {
        assert_eq!(
            indices.len(),
            weights.len(),
            "Indices and weights must have same length"
        );

        // Validate indices
        for &idx in &indices {
            assert!(idx < n, "Index {idx} out of bounds for n={n}");
        }

        let mut sw = Self { indices, weights, n };
        sw.sort_by_index();
        sw
    }

    /// Create from dense weights, filtering near-zero values
    ///
    /// # Arguments
    /// * `dense_weights` - Full weight vector
    /// * `threshold` - Values with absolute value <= threshold are considered zero
    pub fn from_dense(dense_weights: Vec<T>, threshold: T::Float) -> Self {
        let n = dense_weights.len();
        let mut indices = Vec::new();
        let mut weights = Vec::new();

        for (i, &w) in dense_weights.iter().enumerate() {
            let w_float = w.to_float();
            if Float::abs(w_float) > threshold {
                indices.push(i);
                weights.push(w);
            }
        }

        let mut sw = Self { indices, weights, n };
        sw.sort_by_index();
        sw
    }

    /// Convert to dense representation
    pub fn to_dense(&self) -> Vec<T> {
        let mut dense = vec![<T as Numeric>::zero(); self.n];
        for (&idx, &weight) in self.indices.iter().zip(self.weights.iter()) {
            dense[idx] = weight;
        }
        dense
    }

    /// Sort indices and corresponding weights
    fn sort_by_index(&mut self) {
        let mut combined: Vec<_> = self
            .indices
            .iter()
            .copied()
            .zip(self.weights.iter().copied())
            .collect();
        combined.sort_unstable_by_key(|(idx, _)| *idx);
        
        for (i, (idx, w)) in combined.into_iter().enumerate() {
            self.indices[i] = idx;
            self.weights[i] = w;
        }
    }

    /// Number of non-zero weights
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Check if empty (no non-zero weights)
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Total size (n)
    pub fn len(&self) -> usize {
        self.n
    }

    /// Sparsity ratio (fraction of zero weights)
    pub fn sparsity(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / self.n as f64)
        }
    }

    /// Total memory size in bytes
    pub fn memory_size(&self) -> usize {
        self.indices.len() * std::mem::size_of::<usize>()
            + self.weights.len() * std::mem::size_of::<f64>()
            + std::mem::size_of::<Self>()
    }

    /// Validate weights sum to approximately 1.0
    ///
    /// Useful for probability weights that should sum to 1
    pub fn validate_sum<P: crate::ComputePrimitives<T>>(
        &self, 
        primitives: &P,
        expected_sum: T::Float, 
        tolerance: T::Float
    ) -> bool {
        // Use Layer 1 (ComputePrimitives) to compute the sum
        let sum = primitives.sum(&self.weights);
        // Convert from T::Aggregate to f64, then use NumCast to convert to T::Float
        let sum_f64: f64 = sum.into();
        let sum_float = <T::Float as num_traits::NumCast>::from(sum_f64)
            .unwrap_or(<T::Float as num_traits::Zero>::zero());
        Float::abs(sum_float - expected_sum) < tolerance
    }

    /// Get weight at index (0.0 if not in sparse representation)
    pub fn get(&self, index: usize) -> T {
        // Binary search since indices are sorted
        match self.indices.binary_search(&index) {
            Ok(pos) => self.weights[pos],
            Err(_) => <T as Numeric>::zero(),
        }
    }

    /// Check if index has non-zero weight
    pub fn contains(&self, index: usize) -> bool {
        self.indices.binary_search(&index).is_ok()
    }

    /// Iterate over (index, weight) pairs
    pub fn iter(&self) -> impl Iterator<Item = (usize, T)> + '_ {
        self.indices.iter().copied()
            .zip(self.weights.iter().copied())
    }
    
    /// Apply sparse weights to compute weighted sum efficiently
    pub fn apply<P: crate::ComputePrimitives<T>>(&self, data: &[T], primitives: &P) -> T::Aggregate {
        assert_eq!(
            data.len(),
            self.n,
            "Data length must match sparse weights dimension"
        );

        // Use Layer 1 (ComputePrimitives) for the sparse weighted sum
        primitives.sparse_weighted_sum(data, &self.indices, &self.weights)
    }
}

/// Dense weight representation for any numeric type
///
/// For algorithms where most or all weights are non-zero, dense representation
/// is more efficient than sparse.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseWeights<T: Numeric = f64> {
    pub weights: Vec<T>,
}

impl<T: Numeric> DenseWeights<T> {
    /// Create dense weights
    pub fn new(weights: Vec<T>) -> Self {
        Self { weights }
    }

    /// Convert to sparse representation
    pub fn to_sparse(&self, threshold: T::Float) -> SparseWeights<T> {
        SparseWeights::from_dense(self.weights.clone(), threshold)
    }

    /// Number of weights
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Get weight at index
    pub fn get(&self, index: usize) -> Option<T> {
        self.weights.get(index).copied()
    }
    
    /// Apply dense weights to compute weighted sum
    pub fn apply<P: crate::ComputePrimitives<T>>(&self, data: &[T], primitives: &P) -> T::Aggregate {
        assert_eq!(
            data.len(),
            self.weights.len(),
            "Data and weights must have same length"
        );

        // Use Layer 1 (ComputePrimitives) for the dot product
        primitives.dot_product(data, &self.weights)
    }
}

/// Unified weight representation that can be either sparse or dense
#[derive(Clone, Debug, PartialEq)]
pub enum Weights<T: Numeric = f64> {
    Sparse(SparseWeights<T>),
    Dense(DenseWeights<T>),
}

impl<T: Numeric> Weights<T> {
    /// Choose optimal representation based on sparsity
    pub fn from_dense_auto(weights: Vec<T>, sparsity_threshold: f64, zero_threshold: T::Float) -> Self {
        // Count non-zero elements
        let nnz = weights.iter()
            .filter(|&&w| Float::abs(w.to_float()) > zero_threshold)
            .count();
        
        let sparsity = 1.0 - (nnz as f64 / weights.len() as f64);
        
        if sparsity >= sparsity_threshold {
            // Use sparse representation
            Self::Sparse(SparseWeights::from_dense(weights, zero_threshold))
        } else {
            // Use dense representation
            Self::Dense(DenseWeights::new(weights))
        }
    }

    /// Get the number of weights
    pub fn len(&self) -> usize {
        match self {
            Self::Sparse(s) => s.n,
            Self::Dense(d) => d.len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Sparse(s) => s.is_empty(),
            Self::Dense(d) => d.is_empty(),
        }
    }

    /// Apply weights to compute weighted sum
    pub fn apply<P: crate::ComputePrimitives<T>>(&self, data: &[T], primitives: &P) -> T::Aggregate {
        match self {
            Self::Sparse(s) => s.apply(data, primitives),
            Self::Dense(d) => d.apply(data, primitives),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_weights_creation_f64() {
        let indices = vec![1, 3, 7];
        let weights = vec![0.2, 0.5, 0.3];
        let sparse = SparseWeights::new(indices.clone(), weights.clone(), 10);

        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.len(), 10);
        assert_eq!(sparse.sparsity(), 0.7);
    }
    
    #[test]
    fn test_sparse_weights_creation_i32() {
        let indices = vec![0, 2, 4];
        let weights = vec![10i32, 20, 30];
        let sparse = SparseWeights::new(indices, weights, 5);

        assert_eq!(sparse.get(0), 10);
        assert_eq!(sparse.get(1), 0);
        assert_eq!(sparse.get(2), 20);
    }

    #[test]
    fn test_sparse_weights_sorting() {
        let indices = vec![7, 1, 3];
        let weights = vec![0.3, 0.2, 0.5];
        let sparse = SparseWeights::new(indices, weights, 10);

        // Should be sorted by index
        assert_eq!(sparse.indices, vec![1, 3, 7]);
        assert_eq!(sparse.weights, vec![0.2, 0.5, 0.3]);
    }

    #[test]
    fn test_sparse_from_dense() {
        let dense = vec![0.0, 0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0];
        let sparse = SparseWeights::from_dense(dense.clone(), 1e-10);

        assert_eq!(sparse.indices, vec![1, 3, 7]);
        assert_eq!(sparse.weights, vec![0.2, 0.5, 0.3]);
        assert_eq!(sparse.n, 10);
    }

    #[test]
    fn test_sparse_to_dense() {
        let indices = vec![1, 3, 7];
        let weights = vec![0.2, 0.5, 0.3];
        let sparse = SparseWeights::new(indices, weights, 10);
        
        let dense = sparse.to_dense();
        assert_eq!(dense.len(), 10);
        assert_eq!(dense[1], 0.2);
        assert_eq!(dense[3], 0.5);
        assert_eq!(dense[7], 0.3);
        assert_eq!(dense[0], 0.0);
    }

    #[test]
    fn test_sparse_get() {
        let indices = vec![1, 3, 7];
        let weights = vec![0.2, 0.5, 0.3];
        let sparse = SparseWeights::new(indices, weights, 10);

        assert_eq!(sparse.get(1), 0.2);
        assert_eq!(sparse.get(3), 0.5);
        assert_eq!(sparse.get(7), 0.3);
        assert_eq!(sparse.get(0), 0.0);
        assert_eq!(sparse.get(5), 0.0);
    }

    #[test]
    fn test_validate_sum() {
        use crate::ScalarBackend;
        
        let indices = vec![1, 3, 7];
        let weights = vec![0.2, 0.5, 0.3];
        let sparse = SparseWeights::new(indices, weights, 10);
        let primitives = ScalarBackend::new();

        assert!(sparse.validate_sum(&primitives, 1.0, 1e-10));
        assert!(!sparse.validate_sum(&primitives, 0.5, 1e-10));
    }

    #[test]
    #[should_panic(expected = "Index 10 out of bounds for n=10")]
    fn test_out_of_bounds_index() {
        let indices = vec![1, 3, 10];
        let weights = vec![0.2, 0.5, 0.3];
        SparseWeights::new(indices, weights, 10);
    }

    #[test]
    fn test_dense_weights_f64() {
        let weights = vec![0.1, 0.2, 0.3, 0.4];
        let dense = DenseWeights::new(weights.clone());

        assert_eq!(dense.len(), 4);
        assert_eq!(dense.get(1), Some(0.2));
        assert_eq!(dense.get(10), None);
    }
    
    #[test]
    fn test_unified_weights() {
        // Test sparse representation chosen
        let weights = vec![0.0, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0];
        let unified = Weights::from_dense_auto(weights, 0.5, 0.05);
        assert!(matches!(unified, Weights::Sparse(_)));

        // Test dense representation chosen
        let weights = vec![0.1, 0.2, 0.3, 0.4];
        let unified = Weights::from_dense_auto(weights, 0.8, 0.05);
        assert!(matches!(unified, Weights::Dense(_)));
    }

    #[test]
    fn test_dense_to_sparse() {
        let weights = vec![0.0, 0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0];
        let dense = DenseWeights::new(weights);
        let sparse = dense.to_sparse(1e-10);

        assert_eq!(sparse.indices, vec![1, 3, 7]);
        assert_eq!(sparse.weights, vec![0.2, 0.5, 0.3]);
    }
    
    #[test]
    fn test_apply_weights() {
        use crate::ScalarBackend;
        
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![0.1, 0.2, 0.3, 0.4];
        let primitives = ScalarBackend::new();
        
        // Test dense apply
        let dense = DenseWeights::new(weights.clone());
        let result = dense.apply(&data, &primitives);
        assert!(((result as f64) - 3.0).abs() < 1e-10); // 0.1*1 + 0.2*2 + 0.3*3 + 0.4*4 = 3.0
        
        // Test sparse apply
        let sparse = SparseWeights::new(vec![1, 3], vec![0.5, 0.5], 4);
        let result = sparse.apply(&data, &primitives);
        assert!(((result as f64) - 3.0).abs() < 1e-10); // 0.5*2 + 0.5*4 = 3.0
    }
}