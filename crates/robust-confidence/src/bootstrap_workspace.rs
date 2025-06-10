//! Memory-efficient bootstrap workspace management
//!
//! This module provides workspace pools for bootstrap operations that minimize
//! memory allocation by reusing buffers across bootstrap iterations.

use robust_core::{
    ResampleWorkspace, IndexWorkspace, Numeric,
};
use std::sync::Arc;
use std::marker::PhantomData;

/// Thread-safe workspace pool for bootstrap operations
#[derive(Clone)]
pub struct BootstrapWorkspacePool<T: Numeric = f64> {
    index_workspace: Arc<IndexWorkspace>,
    resample_workspace: Arc<ResampleWorkspace<T>>,
    _phantom: PhantomData<T>,
}

impl<T: Numeric> BootstrapWorkspacePool<T> {
    /// Create a new workspace pool
    pub fn new() -> Self {
        Self {
            index_workspace: Arc::new(IndexWorkspace::new(64)),
            resample_workspace: Arc::new(ResampleWorkspace::new(64)),
            _phantom: PhantomData,
        }
    }
    
    /// Get a workspace handle for resampling operations
    pub fn get_workspace(&self) -> BootstrapWorkspace<T> {
        BootstrapWorkspace {
            pool: self.clone(),
        }
    }
}

impl<T: Numeric> Default for BootstrapWorkspacePool<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Workspace handle for bootstrap operations
///
/// This struct provides convenient methods for bootstrap-specific operations
/// using the underlying workspace pool.
pub struct BootstrapWorkspace<T: Numeric = f64> {
    pool: BootstrapWorkspacePool<T>,
}

impl<T: Numeric> BootstrapWorkspace<T> {
    /// Resample a slice using the given indices
    ///
    /// This method gets a buffer from the pool, performs the resampling,
    /// and returns a new Vec with the resampled data.
    pub fn resample_slice(&self, source: &[T], indices: &[usize]) -> Vec<T> {
        let mut buffer = self.pool.resample_workspace.get_buffer(indices.len());
        self.pool.resample_workspace.resample(source, indices, &mut buffer);
        buffer.as_slice().to_vec()
    }
    
    /// Get a buffer for indices
    pub fn get_index_buffer(&self, size: usize) -> robust_core::CheckedOutBuffer<usize> {
        self.pool.index_workspace.get_indices(size)
    }
    
    /// Get a buffer for resampled data
    pub fn get_resample_buffer(&self, size: usize) -> robust_core::CheckedOutBuffer<T> {
        self.pool.resample_workspace.get_buffer(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_workspace_pool() {
        let pool = BootstrapWorkspacePool::new();
        let workspace = pool.get_workspace();
        
        let source = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = vec![0, 2, 4, 1, 3];
        
        let resampled = workspace.resample_slice(&source, &indices);
        assert_eq!(resampled, vec![1.0, 3.0, 5.0, 2.0, 4.0]);
    }
    
    #[test]
    fn test_multiple_workspaces() {
        let pool = BootstrapWorkspacePool::new();
        
        // Create multiple workspaces from the same pool
        let ws1 = pool.get_workspace();
        let ws2 = pool.get_workspace();
        
        // They should work independently
        let data = vec![1.0, 2.0, 3.0];
        let indices = vec![2, 1, 0];
        
        let r1 = ws1.resample_slice(&data, &indices);
        let r2 = ws2.resample_slice(&data, &indices);
        
        assert_eq!(r1, r2);
        assert_eq!(r1, vec![3.0, 2.0, 1.0]);
    }
}