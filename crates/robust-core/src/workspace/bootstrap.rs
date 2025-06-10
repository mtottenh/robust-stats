//! Bootstrap-specific workspace components for efficient resampling operations
//!
//! This module provides specialized workspace types for bootstrap operations:
//! - Index generation and storage
//! - Generic resampling operations
//! - Sorting workspaces for order statistics
//! - Thread-local workspace management

use crate::{AlignedBuffer, BufferPool, CheckedOutBuffer};
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// Component for index generation and storage
///
/// Manages buffers for bootstrap indices with proper alignment
pub struct IndexWorkspace {
    pool: Arc<BufferPool<usize>>,
}

impl IndexWorkspace {
    /// Create a new index workspace with specified alignment
    pub fn new(alignment: usize) -> Self {
        Self {
            pool: Arc::new(BufferPool::new(alignment, 16)), // Keep up to 16 index buffers
        }
    }

    /// Get a buffer for indices
    pub fn get_indices(&self, size: usize) -> CheckedOutBuffer<usize> {
        self.pool.checkout(size)
    }

    /// Generate bootstrap indices into provided buffer
    ///
    /// Fills the buffer with random indices in range [0, n_samples)
    #[cfg(feature = "rand")]
    pub fn generate_indices<R: rand::Rng>(
        &self,
        rng: &mut R,
        n_samples: usize,
        buffer: &mut AlignedBuffer<usize>,
    ) {
        use rand::distributions::{Distribution, Uniform};

        buffer.resize(n_samples);
        let indices = buffer.as_mut_slice();
        let dist = Uniform::new(0, n_samples);

        for idx in indices.iter_mut() {
            *idx = dist.sample(rng);
        }
    }

    /// Generate bootstrap indices for a specific iteration
    ///
    /// Uses a deterministic seed based on the iteration number
    #[cfg(feature = "rand")]
    pub fn generate_for_iteration(
        &self,
        iteration: usize,
        base_seed: u64,
        n_samples: usize,
    ) -> CheckedOutBuffer<usize> {
        use rand::{rngs::StdRng, SeedableRng};

        let mut buffer = self.get_indices(n_samples);
        let seed = base_seed.wrapping_add(iteration as u64);
        let mut rng = StdRng::seed_from_u64(seed);
        self.generate_indices(&mut rng, n_samples, &mut buffer);
        buffer
    }
}

/// Generic resampling workspace for any type
///
/// Manages buffers for resampled data with proper alignment
pub struct ResampleWorkspace<T> {
    pool: Arc<BufferPool<T>>,
}

impl<T: Send> ResampleWorkspace<T> {
    /// Create a new resample workspace with specified alignment
    pub fn new(alignment: usize) -> Self {
        Self {
            pool: Arc::new(BufferPool::new(alignment, 16)), // Keep up to 16 buffers
        }
    }

    /// Get a buffer for resampled data
    pub fn get_buffer(&self, size: usize) -> CheckedOutBuffer<T> {
        self.pool.checkout(size)
    }

    /// Resample data using provided indices
    ///
    /// Copies elements from source at positions specified by indices into dest
    pub fn resample(&self, source: &[T], indices: &[usize], dest: &mut AlignedBuffer<T>)
    where
        T: Copy,
    {
        dest.resize(indices.len());
        let output = dest.as_mut_slice();

        for (out, &idx) in output.iter_mut().zip(indices.iter()) {
            debug_assert!(idx < source.len(), "Index {idx} out of bounds");
            *out = source[idx];
        }
    }
}

/// Sorting workspace for sortable types
///
/// Manages buffers for sorting operations with proper alignment
pub struct SortWorkspace<T> {
    pool: Arc<BufferPool<T>>,
}

impl<T: Send + PartialOrd> SortWorkspace<T> {
    /// Create a new sort workspace with specified alignment
    pub fn new(alignment: usize) -> Self {
        Self {
            pool: Arc::new(BufferPool::new(alignment, 8)), // Keep up to 8 sort buffers
        }
    }

    /// Sort data in-place
    pub fn sort(&self, data: &mut [T]) {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Get a buffer and copy-sort into it
    pub fn sort_into(&self, source: &[T]) -> CheckedOutBuffer<T>
    where
        T: Copy,
    {
        let mut buffer = self.pool.checkout(source.len());
        buffer.resize(source.len());
        buffer.as_mut_slice().copy_from_slice(source);
        self.sort(buffer.as_mut_slice());
        buffer
    }
}

/// Composable bootstrap workspace for a specific type T
///
/// Combines index, resample, and optional sort workspaces
pub struct BootstrapWorkspace<T> {
    pub indices: Arc<IndexWorkspace>,
    pub resample: Arc<ResampleWorkspace<T>>,
    pub sort: Option<Arc<SortWorkspace<T>>>,
}

impl<T: Send + 'static> BootstrapWorkspace<T> {
    /// Create a new bootstrap workspace with specified alignment
    pub fn new(alignment: usize) -> Self {
        Self {
            indices: Arc::new(IndexWorkspace::new(alignment)),
            resample: Arc::new(ResampleWorkspace::new(alignment)),
            sort: None,
        }
    }

    /// Add sorting capability to the workspace
    pub fn with_sorting(mut self) -> Self
    where
        T: PartialOrd,
    {
        self.sort = Some(Arc::new(SortWorkspace::new(64)));
        self
    }
}

// Thread-local workspace storage
thread_local! {
    static WORKSPACES: RefCell<HashMap<TypeId, Box<dyn Any>>> = RefCell::new(HashMap::new());
}

/// Get or create a workspace for type T in the current thread
///
/// This function provides thread-local workspace management to avoid
/// allocation overhead and contention between threads.
///
/// # Example
/// ```rust,ignore
/// use robust_core::workspace::bootstrap::with_bootstrap_workspace;
///
/// with_bootstrap_workspace::<f64, _, _>(|workspace| {
///     let indices = workspace.indices.get_indices(1000);
///     let buffer = workspace.resample.get_buffer(1000);
///     // Use workspace components...
/// });
/// ```
pub fn with_bootstrap_workspace<T, F, R>(f: F) -> R
where
    T: Send + 'static,
    F: FnOnce(&BootstrapWorkspace<T>) -> R,
{
    WORKSPACES.with(|workspaces| {
        let mut ws_map = workspaces.borrow_mut();
        let type_id = TypeId::of::<T>();

        let workspace = ws_map
            .entry(type_id)
            .or_insert_with(|| Box::new(BootstrapWorkspace::<T>::new(64)));

        let ws = workspace
            .downcast_ref::<BootstrapWorkspace<T>>()
            .expect("workspace type mismatch");

        f(ws)
    })
}

/// Specialized version for f64 with sorting enabled
///
/// This is the most common use case for bootstrap operations
pub fn with_f64_bootstrap_workspace<F, R>(f: F) -> R
where
    F: FnOnce(&BootstrapWorkspace<f64>) -> R,
{
    WORKSPACES.with(|workspaces| {
        let mut ws_map = workspaces.borrow_mut();
        let type_id = TypeId::of::<f64>();

        let workspace = ws_map
            .entry(type_id)
            .or_insert_with(|| Box::new(BootstrapWorkspace::<f64>::new(64).with_sorting()));

        let ws = workspace
            .downcast_ref::<BootstrapWorkspace<f64>>()
            .expect("workspace type mismatch");

        f(ws)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_workspace() {
        let workspace = IndexWorkspace::new(64);

        // Get indices buffer
        let indices = workspace.get_indices(100);
        assert_eq!(indices.capacity(), 100);
        assert_eq!(indices.len(), 0);
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_index_generation() {
        use rand::thread_rng;

        let workspace = IndexWorkspace::new(64);
        let mut buffer = workspace.get_indices(100);
        let mut rng = thread_rng();

        workspace.generate_indices(&mut rng, 100, &mut buffer);
        assert_eq!(buffer.len(), 100);

        // All indices should be in range [0, 100)
        for &idx in buffer.as_slice() {
            assert!(idx < 100);
        }
    }

    #[test]
    fn test_resample_workspace() {
        let workspace = ResampleWorkspace::<f64>::new(64);
        let source = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = vec![4, 2, 2, 0, 1];

        let mut buffer = workspace.get_buffer(5);
        workspace.resample(&source, &indices, &mut buffer);

        let result = buffer.as_slice();
        assert_eq!(result, &[5.0, 3.0, 3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sort_workspace() {
        let workspace = SortWorkspace::<f64>::new(64);
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];

        let sorted = workspace.sort_into(&data);
        assert_eq!(sorted.as_slice(), &[1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 9.0]);
    }

    #[test]
    fn test_bootstrap_workspace() {
        let workspace = BootstrapWorkspace::<f64>::new(64).with_sorting();

        assert!(workspace.indices.get_indices(10).capacity() >= 10);
        assert!(workspace.resample.get_buffer(10).capacity() >= 10);
        assert!(workspace.sort.is_some());
    }

    #[test]
    fn test_thread_local_workspace() {
        // First call creates workspace
        with_bootstrap_workspace::<f64, _, _>(|workspace1| {
            let buffer1 = workspace1.resample.get_buffer(100);
            assert_eq!(buffer1.capacity(), 100);
        });

        // Second call reuses workspace
        with_bootstrap_workspace::<f64, _, _>(|workspace2| {
            let buffer2 = workspace2.resample.get_buffer(50);
            // May get a larger buffer from pool
            assert!(buffer2.capacity() >= 50);
        });
    }

    #[test]
    fn test_f64_bootstrap_workspace() {
        with_f64_bootstrap_workspace(|workspace| {
            // Should have sorting enabled
            assert!(workspace.sort.is_some());

            let indices = workspace.indices.get_indices(10);
            assert_eq!(indices.capacity(), 10);
        });
    }

    #[test]
    fn test_different_types() {
        // Workspace for f64
        with_bootstrap_workspace::<f64, _, _>(|workspace| {
            let buffer = workspace.resample.get_buffer(10);
            assert_eq!(buffer.capacity(), 10);
        });

        // Workspace for i32
        with_bootstrap_workspace::<i32, _, _>(|workspace| {
            let buffer = workspace.resample.get_buffer(20);
            assert_eq!(buffer.capacity(), 20);
        });
    }
}
