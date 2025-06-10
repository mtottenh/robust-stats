//! Execution engines for controlling computation strategy
//!
//! This module provides the execution engine abstraction that unifies
//! primitive selection (SIMD vs scalar) with execution strategy
//! (sequential vs parallel).
//!
//! # Design Philosophy
//!
//! - **Unified Control**: Single type parameter controls both SIMD and parallelism
//! - **Zero-Cost**: All decisions made at compile time
//! - **Thread Pool Integration**: Works with Rayon, Polars, or custom pools
//! - **Composable**: Engines can be mixed and matched with algorithms
//! - **Hierarchical Control**: Prevents thread oversubscription in nested operations

mod hierarchical;

pub use hierarchical::{
    BudgetedEngine,
    BudgetedSubordinate,
    budgeted_engine, auto_budgeted_engine,
    HierarchicalExecution,
    ParallelismBudget,
};

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
pub use hierarchical::budgeted_simd_engine;

use crate::numeric::Numeric;
use crate::primitives::ComputePrimitives;
#[cfg(feature = "parallel")]
use crate::Result;

/// Execution strategy for batch operations
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExecutionStrategy {
    /// Process items sequentially
    Sequential,
    /// Process items in parallel
    Parallel,
    /// Automatically choose based on workload
    Auto,
}

/// Marker trait for execution engine mode properties
///
/// This trait provides compile-time constants that enable zero-cost
/// specialization for different execution patterns.
pub trait ExecutionMode {
    /// Whether this engine executes tasks sequentially
    const IS_SEQUENTIAL: bool;

    /// Whether this engine supports direct (non-closure) processing
    /// This is true for sequential engines where we can avoid closure overhead
    const SUPPORTS_DIRECT: bool;

    /// Optimal chunk size for this execution mode
    fn chunk_size(n_items: usize, n_threads: usize) -> usize;
}

/// Trait for execution engines that control how computations are performed
///
/// An execution engine combines:
/// - Primitive operations (scalar vs SIMD)
/// - Execution strategy (sequential vs parallel)
/// - Thread pool selection (Rayon vs Polars vs custom)
pub trait ExecutionEngine<T: Numeric>: Clone + Send + Sync + ExecutionMode {
    /// The type of primitives used by this engine
    type Primitives: ComputePrimitives<T>;

    /// Get the primitives for low-level operations
    fn primitives(&self) -> &Self::Primitives;

    /// Execute a function in the engine's execution context
    fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send;

    /// Map a function over chunks of data
    fn map_chunks<'a, U, F, R>(&self, data: &'a [U], chunk_size: usize, f: F) -> Vec<R>
    where
        U: Sync,
        F: Fn(&'a [U]) -> R + Sync + Send,
        R: Send;

    /// Execute operations on multiple datasets
    fn execute_batch<F, R>(&self, count: usize, f: F) -> Vec<R>
    where
        F: Fn(usize) -> R + Sync + Send,
        R: Send;

    /// Get the execution strategy
    fn strategy(&self) -> ExecutionStrategy;

    /// Check if parallel execution is available
    fn is_parallel(&self) -> bool {
        matches!(
            self.strategy(),
            ExecutionStrategy::Parallel | ExecutionStrategy::Auto
        )
    }

    /// Get the number of threads available
    fn num_threads(&self) -> usize;
}

/// Sequential execution engine
///
/// Executes all operations sequentially in the current thread.
#[derive(Clone, Debug)]
pub struct SequentialEngine<T: Numeric, P: ComputePrimitives<T>> {
    primitives: P,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> SequentialEngine<T, P> {
    /// Create a new sequential engine with the given primitives
    pub fn new(primitives: P) -> Self {
        Self {
            primitives,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> ExecutionMode for SequentialEngine<T, P> {
    const IS_SEQUENTIAL: bool = true;
    const SUPPORTS_DIRECT: bool = true;

    fn chunk_size(_n_items: usize, _n_threads: usize) -> usize {
        // Process all items in one "chunk" for sequential
        usize::MAX
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> ExecutionEngine<T> for SequentialEngine<T, P> {
    type Primitives = P;

    fn primitives(&self) -> &Self::Primitives {
        &self.primitives
    }

    fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        f()
    }

    fn map_chunks<'a, U, F, R>(&self, data: &'a [U], chunk_size: usize, f: F) -> Vec<R>
    where
        U: Sync,
        F: Fn(&'a [U]) -> R + Sync + Send,
        R: Send,
    {
        data.chunks(chunk_size).map(f).collect()
    }

    fn execute_batch<F, R>(&self, count: usize, f: F) -> Vec<R>
    where
        F: Fn(usize) -> R + Sync + Send,
        R: Send,
    {
        (0..count).map(f).collect()
    }

    fn strategy(&self) -> ExecutionStrategy {
        ExecutionStrategy::Sequential
    }

    fn num_threads(&self) -> usize {
        1
    }
}

/// Parallel execution engine using Rayon
///
/// Executes operations in parallel using Rayon's thread pool.
#[cfg(feature = "parallel")]
#[derive(Clone, Debug)]
pub struct ParallelEngine<T: Numeric, P: ComputePrimitives<T>> {
    primitives: P,
    thread_pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "parallel")]
impl<T: Numeric, P: ComputePrimitives<T>> ParallelEngine<T, P> {
    /// Create a new parallel engine with default thread pool
    pub fn new(primitives: P) -> Self {
        Self {
            primitives,
            thread_pool: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new parallel engine with a custom thread pool
    pub fn with_thread_pool(primitives: P, pool: std::sync::Arc<rayon::ThreadPool>) -> Self {
        Self {
            primitives,
            thread_pool: Some(pool),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create with a specific number of threads
    pub fn with_num_threads(primitives: P, num_threads: usize) -> Result<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| crate::Error::Execution(format!("Failed to create thread pool: {e}")))?;

        Ok(Self {
            primitives,
            thread_pool: Some(std::sync::Arc::new(pool)),
            _phantom: std::marker::PhantomData,
        })
    }
}

#[cfg(feature = "parallel")]
impl<T: Numeric, P: ComputePrimitives<T>> ExecutionMode for ParallelEngine<T, P> {
    const IS_SEQUENTIAL: bool = false;
    const SUPPORTS_DIRECT: bool = false;

    fn chunk_size(n_items: usize, n_threads: usize) -> usize {
        // Delegate to the batch processor which has better context
        // For now, provide a reasonable default
        let target_chunks = n_threads * 6;
        let chunk_size = n_items.div_ceil(target_chunks);
        chunk_size.max(4).min(n_items)
    }
}

#[cfg(feature = "parallel")]
impl<T: Numeric, P: ComputePrimitives<T>> ExecutionEngine<T> for ParallelEngine<T, P> {
    type Primitives = P;

    fn primitives(&self) -> &Self::Primitives {
        &self.primitives
    }

    fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        if let Some(pool) = &self.thread_pool {
            pool.install(f)
        } else {
            rayon::scope(|_| f())
        }
    }

    fn map_chunks<'a, U, F, R>(&self, data: &'a [U], chunk_size: usize, f: F) -> Vec<R>
    where
        U: Sync,
        F: Fn(&'a [U]) -> R + Sync + Send,
        R: Send,
    {
        use rayon::prelude::*;

        if let Some(pool) = &self.thread_pool {
            pool.install(|| data.par_chunks(chunk_size).map(f).collect())
        } else {
            data.par_chunks(chunk_size).map(f).collect()
        }
    }

    fn execute_batch<F, R>(&self, count: usize, f: F) -> Vec<R>
    where
        F: Fn(usize) -> R + Sync + Send,
        R: Send,
    {
        use rayon::prelude::*;

        if let Some(pool) = &self.thread_pool {
            pool.install(|| (0..count).into_par_iter().map(f).collect())
        } else {
            (0..count).into_par_iter().map(f).collect()
        }
    }

    fn strategy(&self) -> ExecutionStrategy {
        ExecutionStrategy::Parallel
    }

    fn num_threads(&self) -> usize {
        if let Some(pool) = &self.thread_pool {
            pool.current_num_threads()
        } else {
            rayon::current_num_threads()
        }
    }
}


// TODO: Update convenience functions once migration is complete
// These will need to be generic over T: Numeric

/// Create a sequential scalar engine for f64
pub fn scalar_sequential() -> SequentialEngine<f64, crate::primitives::ScalarBackend> {
    SequentialEngine::new(crate::primitives::ScalarBackend)
}

/// Create a sequential SIMD engine for f64
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
pub fn simd_sequential() -> SequentialEngine<f64, crate::primitives::Avx2Backend> {
    SequentialEngine::new(crate::primitives::Avx2Backend::new())
}

/// Create a sequential SIMD engine for f64 (fallback to scalar if AVX2 not available)
#[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
pub fn simd_sequential() -> SequentialEngine<f64, crate::primitives::ScalarBackend> {
    SequentialEngine::new(crate::primitives::ScalarBackend)
}

/// Create a parallel scalar engine for f64
#[cfg(feature = "parallel")]
pub fn scalar_parallel() -> ParallelEngine<f64, crate::primitives::ScalarBackend> {
    ParallelEngine::new(crate::primitives::ScalarBackend)
}

/// Create a parallel SIMD engine for f64
#[cfg(all(feature = "parallel", target_arch = "x86_64", feature = "avx2"))]
pub fn simd_parallel() -> ParallelEngine<f64, crate::primitives::Avx2Backend> {
    ParallelEngine::new(crate::primitives::Avx2Backend::new())
}

/// Create a parallel SIMD engine for f64 (fallback to scalar if AVX2 not available)
#[cfg(all(feature = "parallel", not(all(target_arch = "x86_64", feature = "avx2"))))]
pub fn simd_parallel() -> ParallelEngine<f64, crate::primitives::ScalarBackend> {
    ParallelEngine::new(crate::primitives::ScalarBackend)
}

/// Create an auto-selected engine based on available features
/// 
/// This uses the hierarchical execution system with automatic parallelism budgeting,
/// which prevents thread oversubscription in nested parallel operations.
pub fn auto_engine() -> impl ExecutionEngine<f64> {
    #[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "parallel"))]
    {
        budgeted_simd_engine(ParallelismBudget::auto())
    }
    #[cfg(all(not(all(target_arch = "x86_64", feature = "avx2")), feature = "parallel"))]
    {
        budgeted_engine(ParallelismBudget::auto())
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx2", not(feature = "parallel")))]
    {
        simd_sequential()
    }
    #[cfg(not(any(all(target_arch = "x86_64", feature = "avx2"), feature = "parallel")))]
    {
        scalar_sequential()
    }
}

/// Create a parallel engine that prevents nested parallelism
/// 
/// This is useful for algorithms that have internal parallelism (like bootstrap)
/// to ensure subordinate operations run sequentially.
#[cfg(feature = "parallel")]
pub fn parallel_no_nested() -> impl ExecutionEngine<f64> {
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        budgeted_simd_engine(ParallelismBudget::no_nested())
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    {
        budgeted_engine(ParallelismBudget::no_nested())
    }
}

/// Create a parallel engine with custom thread distribution
/// 
/// # Arguments
/// * `outer_threads` - Threads for outer parallel operations
/// * `inner_threads` - Threads available for nested operations
#[cfg(feature = "parallel")]
pub fn parallel_with_budget(outer_threads: usize, inner_threads: usize) -> impl ExecutionEngine<f64> {
    let budget = ParallelismBudget::custom(outer_threads, inner_threads);
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        budgeted_simd_engine(budget)
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    {
        budgeted_engine(budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_engine() {
        let engine = scalar_sequential();

        // Test execute
        let result = engine.execute(|| 42);
        assert_eq!(result, 42);

        // Test map_chunks
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sums = engine.map_chunks(&data, 2, |chunk| chunk.iter().sum::<f64>());
        assert_eq!(sums, vec![3.0, 7.0, 11.0]);

        // Test execute_batch
        let squares = engine.execute_batch(5, |i| i * i);
        assert_eq!(squares, vec![0, 1, 4, 9, 16]);

        assert_eq!(engine.strategy(), ExecutionStrategy::Sequential);
        assert_eq!(engine.num_threads(), 1);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_engine() {
        let engine = scalar_parallel();

        // Test parallel execution
        let data: Vec<i32> = (0..1000).collect();
        let sum = engine.execute(|| {
            use rayon::prelude::*;
            data.par_iter().sum::<i32>()
        });
        assert_eq!(sum, 499500);

        // Test map_chunks
        let data = vec![1.0; 100];
        let sums = engine.map_chunks(&data, 25, |chunk| chunk.iter().sum::<f64>());
        assert_eq!(sums, vec![25.0, 25.0, 25.0, 25.0]);

        assert_eq!(engine.strategy(), ExecutionStrategy::Parallel);
        assert!(engine.num_threads() > 0);
    }

    #[test]
    fn test_auto_engine() {
        let engine = auto_engine();
        assert!(engine.num_threads() > 0);
    }

    #[test]
    fn test_simd_sequential() {
        let engine = simd_sequential();
        assert_eq!(engine.strategy(), ExecutionStrategy::Sequential);
        
        // Test that SIMD operations work
        let data = [1.0, 2.0, 3.0, 4.0];
        let result = engine.execute(|| data.iter().sum::<f64>());
        assert_eq!(result, 10.0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_simd_parallel() {
        let engine = simd_parallel();
        assert_eq!(engine.strategy(), ExecutionStrategy::Parallel);
        assert!(engine.num_threads() > 0);
    }
}
