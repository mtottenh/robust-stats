//! Hierarchical execution control for nested parallel operations
//!
//! This module extends the execution engine concept to handle nested parallelism
//! gracefully, preventing thread oversubscription in hierarchical computations.

use super::{ExecutionEngine, ExecutionStrategy, ExecutionMode, SequentialEngine};
#[cfg(feature = "parallel")]
use super::ParallelEngine;
use crate::primitives::ComputePrimitives;
use crate::numeric::Numeric;

/// Extension trait for hierarchical execution control
///
/// This trait allows execution engines to create subordinate engines with
/// constrained parallelism, enabling proper thread management in nested
/// parallel operations.
pub trait HierarchicalExecution<T: Numeric>: ExecutionEngine<T> {
    /// The type of subordinate engine this engine creates
    type SubordinateEngine: ExecutionEngine<T, Primitives = Self::Primitives>;
    
    /// Create a subordinate engine for nested operations
    ///
    /// The subordinate engine should have reduced or no parallelism to prevent
    /// thread oversubscription when used within already-parallel operations.
    ///
    /// # Example
    /// ```rust,ignore
    /// // Bootstrap uses parallel execution
    /// let outer_engine = simd_parallel();
    /// 
    /// // But creates sequential engines for inner operations
    /// let inner_engine = outer_engine.subordinate();
    /// 
    /// // Now inner operations won't create additional threads
    /// quantile_estimator.estimate_with_engine(&data, &inner_engine);
    /// ```
    fn subordinate(&self) -> Self::SubordinateEngine;
    
    /// Create a subordinate engine with specific thread budget
    ///
    /// More fine-grained control for cases where some parallelism is acceptable
    /// in nested operations.
    fn subordinate_with_threads(&self, max_threads: usize) -> Self::SubordinateEngine;
}

/// Parallelism budget for hierarchical execution
///
/// This type represents how available threads should be distributed between
/// outer and inner parallel operations.
#[derive(Debug, Clone, Copy)]
pub struct ParallelismBudget {
    /// Total threads available
    pub total_threads: usize,
    /// Threads reserved for outer operations (e.g., bootstrap)
    pub outer_threads: usize,
    /// Threads available for inner operations (e.g., quantile)
    pub inner_threads: usize,
}

impl ParallelismBudget {
    /// Create an automatic budget based on available cores
    pub fn auto() -> Self {
        #[cfg(feature = "parallel")]
        let total = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let total = 1;
        
        // Heuristic: Give most threads to outer operation (coarse-grained parallelism)
        // Reserve a few for inner operations if needed
        let outer_threads = (total * 3 / 4).max(1);
        let inner_threads = (total / 4).max(1);
        
        Self {
            total_threads: total,
            outer_threads,
            inner_threads,
        }
    }
    
    /// Create a budget that prevents all nested parallelism
    pub fn no_nested() -> Self {
        #[cfg(feature = "parallel")]
        let total = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let total = 1;
        Self {
            total_threads: total,
            outer_threads: total,
            inner_threads: 1, // Force sequential inner operations
        }
    }
    
    /// Create a custom budget
    pub fn custom(outer_threads: usize, inner_threads: usize) -> Self {
        Self {
            total_threads: outer_threads + inner_threads,
            outer_threads,
            inner_threads,
        }
    }
}

/// Budgeted execution engine that respects parallelism constraints
#[derive(Clone)]
pub struct BudgetedEngine<T: Numeric, P: ComputePrimitives<T>> {
    primitives: P,
    budget: ParallelismBudget,
    is_subordinate: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> BudgetedEngine<T, P> {
    /// Create a new budgeted engine
    pub fn new(primitives: P, budget: ParallelismBudget) -> Self {
        Self {
            primitives,
            budget,
            is_subordinate: false,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Create a subordinate budgeted engine
    fn new_subordinate(primitives: P, parent_budget: ParallelismBudget) -> Self {
        Self {
            primitives,
            budget: parent_budget,
            is_subordinate: true,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> ExecutionMode for BudgetedEngine<T, P> {
    const IS_SEQUENTIAL: bool = false; // Dynamic based on budget
    const SUPPORTS_DIRECT: bool = false;
    
    fn chunk_size(n_items: usize, n_threads: usize) -> usize {
        let target_chunks = n_threads * 6;
        let chunk_size = n_items.div_ceil(target_chunks);
        chunk_size.max(4).min(n_items)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> ExecutionEngine<T> for BudgetedEngine<T, P> {
    type Primitives = P;
    
    fn primitives(&self) -> &Self::Primitives {
        &self.primitives
    }
    
    fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        #[cfg(feature = "parallel")]
        {
            if self.strategy() == ExecutionStrategy::Parallel {
                rayon::scope(|_| f())
            } else {
                f()
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            f()
        }
    }
    
    fn map_chunks<'a, U, F, R>(&self, data: &'a [U], chunk_size: usize, f: F) -> Vec<R>
    where
        U: Sync,
        F: Fn(&'a [U]) -> R + Sync + Send,
        R: Send,
    {
        #[cfg(feature = "parallel")]
        {
            if self.strategy() == ExecutionStrategy::Parallel {
                use rayon::prelude::*;
                data.par_chunks(chunk_size).map(f).collect()
            } else {
                data.chunks(chunk_size).map(f).collect()
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            data.chunks(chunk_size).map(f).collect()
        }
    }
    
    fn execute_batch<F, R>(&self, count: usize, f: F) -> Vec<R>
    where
        F: Fn(usize) -> R + Sync + Send,
        R: Send,
    {
        #[cfg(feature = "parallel")]
        {
            if self.strategy() == ExecutionStrategy::Parallel {
                use rayon::prelude::*;
                (0..count).into_par_iter().map(f).collect()
            } else {
                (0..count).map(f).collect()
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..count).map(f).collect()
        }
    }
    
    fn strategy(&self) -> ExecutionStrategy {
        if self.is_subordinate && self.budget.inner_threads <= 1 {
            ExecutionStrategy::Sequential
        } else if self.is_subordinate || self.budget.outer_threads > 1 {
            ExecutionStrategy::Parallel
        } else {
            ExecutionStrategy::Sequential
        }
    }
    
    fn num_threads(&self) -> usize {
        if self.is_subordinate {
            self.budget.inner_threads
        } else {
            self.budget.outer_threads
        }
    }
}

/// Enum to represent subordinate engines from BudgetedEngine
#[derive(Clone)]
pub enum BudgetedSubordinate<T: Numeric, P: ComputePrimitives<T>> {
    Sequential(SequentialEngine<T, P>),
    Budgeted(BudgetedEngine<T, P>),
}

impl<T: Numeric, P: ComputePrimitives<T>> ExecutionMode for BudgetedSubordinate<T, P> {
    const IS_SEQUENTIAL: bool = false; // Dynamic
    const SUPPORTS_DIRECT: bool = false;
    
    fn chunk_size(n_items: usize, n_threads: usize) -> usize {
        match Self::IS_SEQUENTIAL {
            true => SequentialEngine::<T, P>::chunk_size(n_items, n_threads),
            false => BudgetedEngine::<T, P>::chunk_size(n_items, n_threads),
        }
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> ExecutionEngine<T> for BudgetedSubordinate<T, P> {
    type Primitives = P;
    
    fn primitives(&self) -> &Self::Primitives {
        match self {
            BudgetedSubordinate::Sequential(e) => e.primitives(),
            BudgetedSubordinate::Budgeted(e) => e.primitives(),
        }
    }
    
    fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        match self {
            BudgetedSubordinate::Sequential(e) => e.execute(f),
            BudgetedSubordinate::Budgeted(e) => e.execute(f),
        }
    }
    
    fn map_chunks<'a, U, F, R>(&self, data: &'a [U], chunk_size: usize, f: F) -> Vec<R>
    where
        U: Sync,
        F: Fn(&'a [U]) -> R + Sync + Send,
        R: Send,
    {
        match self {
            BudgetedSubordinate::Sequential(e) => e.map_chunks(data, chunk_size, f),
            BudgetedSubordinate::Budgeted(e) => e.map_chunks(data, chunk_size, f),
        }
    }
    
    fn execute_batch<F, R>(&self, count: usize, f: F) -> Vec<R>
    where
        F: Fn(usize) -> R + Sync + Send,
        R: Send,
    {
        match self {
            BudgetedSubordinate::Sequential(e) => e.execute_batch(count, f),
            BudgetedSubordinate::Budgeted(e) => e.execute_batch(count, f),
        }
    }
    
    fn strategy(&self) -> ExecutionStrategy {
        match self {
            BudgetedSubordinate::Sequential(e) => e.strategy(),
            BudgetedSubordinate::Budgeted(e) => e.strategy(),
        }
    }
    
    fn num_threads(&self) -> usize {
        match self {
            BudgetedSubordinate::Sequential(e) => e.num_threads(),
            BudgetedSubordinate::Budgeted(e) => e.num_threads(),
        }
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> HierarchicalExecution<T> for BudgetedEngine<T, P> {
    type SubordinateEngine = BudgetedSubordinate<T, P>;
    
    fn subordinate(&self) -> Self::SubordinateEngine {
        // Subordinate engines get the inner thread budget
        if self.budget.inner_threads <= 1 {
            // Force sequential execution for inner operations
            BudgetedSubordinate::Sequential(SequentialEngine::new(self.primitives.clone()))
        } else {
            // Create a subordinate with limited parallelism
            BudgetedSubordinate::Budgeted(BudgetedEngine::new_subordinate(self.primitives.clone(), self.budget))
        }
    }
    
    fn subordinate_with_threads(&self, max_threads: usize) -> Self::SubordinateEngine {
        let actual_threads = max_threads.min(self.budget.inner_threads);
        if actual_threads <= 1 {
            BudgetedSubordinate::Sequential(SequentialEngine::new(self.primitives.clone()))
        } else {
            let subordinate_budget = ParallelismBudget {
                total_threads: actual_threads,
                outer_threads: 0,
                inner_threads: actual_threads,
            };
            BudgetedSubordinate::Budgeted(BudgetedEngine::new_subordinate(self.primitives.clone(), subordinate_budget))
        }
    }
}

// Implement HierarchicalExecution for existing engines

impl<T: Numeric, P: ComputePrimitives<T>> HierarchicalExecution<T> for SequentialEngine<T, P> {
    type SubordinateEngine = SequentialEngine<T, P>;
    
    fn subordinate(&self) -> Self::SubordinateEngine {
        // Sequential engine always creates sequential subordinates
        self.clone()
    }
    
    fn subordinate_with_threads(&self, _max_threads: usize) -> Self::SubordinateEngine {
        // Sequential engine ignores thread budget
        self.clone()
    }
}

#[cfg(feature = "parallel")]
impl<T: Numeric, P: ComputePrimitives<T>> HierarchicalExecution<T> for ParallelEngine<T, P> {
    type SubordinateEngine = SequentialEngine<T, P>;
    
    fn subordinate(&self) -> Self::SubordinateEngine {
        // Parallel engine creates sequential subordinates to prevent oversubscription
        SequentialEngine::new(self.primitives().clone())
    }
    
    fn subordinate_with_threads(&self, _max_threads: usize) -> Self::SubordinateEngine {
        // For simplicity, always return sequential to prevent oversubscription
        // In the future, could return a BudgetedEngine with limited threads
        SequentialEngine::new(self.primitives().clone())
    }
}

/// Create a budgeted execution engine with scalar backend for f64
pub fn budgeted_engine(budget: ParallelismBudget) -> BudgetedEngine<f64, crate::primitives::ScalarBackend> {
    BudgetedEngine::new(crate::primitives::ScalarBackend, budget)
}

/// Create a budgeted execution engine with SIMD backend for f64
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
pub fn budgeted_simd_engine(budget: ParallelismBudget) -> BudgetedEngine<f64, crate::primitives::Avx2Backend> {
    BudgetedEngine::new(crate::primitives::Avx2Backend::new(), budget)
}

/// Create an execution engine with automatic budgeting
pub fn auto_budgeted_engine() -> BudgetedEngine<f64, crate::primitives::ScalarBackend> {
    BudgetedEngine::new(crate::primitives::ScalarBackend, ParallelismBudget::auto())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parallelism_budget() {
        let budget = ParallelismBudget::auto();
        assert!(budget.total_threads > 0);
        assert!(budget.outer_threads > 0);
        assert!(budget.inner_threads > 0);
        assert!(budget.outer_threads + budget.inner_threads <= budget.total_threads * 2);
    }
    
    #[test]
    fn test_no_nested_budget() {
        let budget = ParallelismBudget::no_nested();
        assert_eq!(budget.inner_threads, 1);
        assert_eq!(budget.outer_threads, budget.total_threads);
    }
    
    #[test]
    fn test_budgeted_engine_subordinate() {
        let budget = ParallelismBudget::no_nested();
        let engine = BudgetedEngine::<f64, _>::new(crate::primitives::ScalarBackend, budget);
        
        // Check that outer engine is parallel if we have more than 1 thread
        if budget.total_threads > 1 {
            assert_eq!(engine.strategy(), ExecutionStrategy::Parallel);
        }
        
        // Check that subordinate is sequential
        let subordinate = engine.subordinate();
        assert_eq!(subordinate.strategy(), ExecutionStrategy::Sequential);
    }
    
    #[test]
    fn test_hierarchical_execution_trait() {
        // Test that existing engines implement the trait
        let seq_engine = SequentialEngine::<f64, _>::new(crate::primitives::ScalarBackend);
        let sub = seq_engine.subordinate();
        assert_eq!(sub.strategy(), ExecutionStrategy::Sequential);
        
        #[cfg(feature = "parallel")]
        {
            let par_engine = ParallelEngine::<f64, _>::new(crate::primitives::ScalarBackend);
            let sub = par_engine.subordinate();
            assert_eq!(sub.strategy(), ExecutionStrategy::Sequential);
        }
    }
    
    #[test]
    fn test_custom_parallelism_budget() {
        // Test custom budget creation
        let budget = ParallelismBudget::custom(4, 2);
        assert_eq!(budget.outer_threads, 4);
        assert_eq!(budget.inner_threads, 2);
        assert_eq!(budget.total_threads, 6);
    }
    
    #[test]
    fn test_budgeted_engine_thread_counts() {
        let budget = ParallelismBudget::custom(8, 4);
        let engine = BudgetedEngine::<f64, _>::new(crate::primitives::ScalarBackend, budget);
        
        // Outer engine should report outer thread count
        assert_eq!(engine.num_threads(), 8);
        
        // Subordinate should report inner thread count
        let subordinate = engine.subordinate();
        match subordinate {
            BudgetedSubordinate::Budgeted(ref e) => {
                assert_eq!(e.num_threads(), 4);
            }
            _ => {} // Sequential doesn't track threads
        }
    }
    
    #[test]
    fn test_subordinate_with_threads() {
        let budget = ParallelismBudget::custom(8, 4);
        let engine = BudgetedEngine::<f64, _>::new(crate::primitives::ScalarBackend, budget);
        
        // Request more threads than available - should be capped
        let sub1 = engine.subordinate_with_threads(10);
        match sub1 {
            BudgetedSubordinate::Budgeted(e) => {
                assert_eq!(e.num_threads(), 4); // Capped at inner_threads
            }
            _ => panic!("Expected Budgeted subordinate"),
        }
        
        // Request fewer threads
        let sub2 = engine.subordinate_with_threads(2);
        match sub2 {
            BudgetedSubordinate::Budgeted(e) => {
                assert_eq!(e.num_threads(), 2);
            }
            _ => panic!("Expected Budgeted subordinate"),
        }
        
        // Request 1 thread - should become sequential
        let sub3 = engine.subordinate_with_threads(1);
        match sub3 {
            BudgetedSubordinate::Sequential(_) => {
                // Good - forced sequential
            }
            _ => panic!("Expected Sequential subordinate"),
        }
    }
    
    #[test]
    fn test_budgeted_engine_execution() {
        let budget = ParallelismBudget::custom(2, 1);
        let engine = BudgetedEngine::<f64, _>::new(crate::primitives::ScalarBackend, budget);
        
        // Test execute
        let result = engine.execute(|| 42);
        assert_eq!(result, 42);
        
        // Test map_chunks
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let sums = engine.map_chunks(&data, 3, |chunk| chunk.iter().sum::<f64>());
        let total: f64 = sums.iter().sum();
        assert_eq!(total, 36.0); // 1+2+3+4+5+6+7+8
        
        // Test execute_batch
        let squares = engine.execute_batch(5, |i| i * i);
        assert_eq!(squares, vec![0, 1, 4, 9, 16]);
    }
    
    #[test]
    fn test_budgeted_subordinate_execution() {
        let budget = ParallelismBudget::custom(4, 2);
        let engine = BudgetedEngine::<f64, _>::new(crate::primitives::ScalarBackend, budget);
        let subordinate = engine.subordinate();
        
        // Test execute
        let result = subordinate.execute(|| "hello");
        assert_eq!(result, "hello");
        
        // Test map_chunks
        let data = vec![1, 2, 3, 4, 5, 6];
        let doubled = subordinate.map_chunks(&data, 2, |chunk| {
            chunk.iter().map(|&x| x * 2).collect::<Vec<_>>()
        });
        let flattened: Vec<i32> = doubled.into_iter().flatten().collect();
        assert_eq!(flattened, vec![2, 4, 6, 8, 10, 12]);
        
        // Test execute_batch
        let cubes = subordinate.execute_batch(4, |i| i * i * i);
        assert_eq!(cubes, vec![0, 1, 8, 27]);
    }
    
    #[test]
    fn test_execution_mode_chunk_size() {
        // Test chunk size calculation
        let chunk_size = BudgetedEngine::<f64, crate::primitives::ScalarBackend>::chunk_size(100, 4);
        assert!(chunk_size >= 4); // Minimum chunk size
        assert!(chunk_size <= 100); // Can't exceed total items
        
        // Edge cases
        assert_eq!(BudgetedEngine::<f64, crate::primitives::ScalarBackend>::chunk_size(3, 4), 3);
        assert_eq!(BudgetedEngine::<f64, crate::primitives::ScalarBackend>::chunk_size(0, 4), 0);
    }
    
    #[test]
    fn test_subordinate_creation_edge_cases() {
        // Test with minimal budget
        let budget = ParallelismBudget::custom(1, 0);
        let engine = BudgetedEngine::<f64, _>::new(crate::primitives::ScalarBackend, budget);
        
        // Should create sequential subordinate even though inner_threads is 0
        let sub = engine.subordinate();
        assert_eq!(sub.strategy(), ExecutionStrategy::Sequential);
        
        // Test subordinate_with_threads with 0
        let sub_zero = engine.subordinate_with_threads(0);
        match sub_zero {
            BudgetedSubordinate::Sequential(_) => {
                // Good - forced sequential
            }
            _ => panic!("Expected Sequential subordinate for 0 threads"),
        }
    }
    
    #[test] 
    fn test_nested_subordinates() {
        // Test creating subordinates from subordinates
        let budget = ParallelismBudget::custom(8, 4);
        let _engine = BudgetedEngine::<f64, _>::new(crate::primitives::ScalarBackend, budget);
        
        // Sequential subordinate should produce itself
        let seq_engine = SequentialEngine::<f64, _>::new(crate::primitives::ScalarBackend);
        let seq_sub = seq_engine.subordinate();
        let seq_sub_sub = seq_sub.subordinate();
        assert_eq!(seq_sub_sub.strategy(), ExecutionStrategy::Sequential);
    }
    
    #[test]
    fn test_budgeted_constructor_functions() {
        // Test the convenience constructors
        let engine = auto_budgeted_engine();
        assert!(engine.num_threads() > 0);
        
        let custom_budget = ParallelismBudget::custom(2, 2);
        let custom_engine = budgeted_engine(custom_budget);
        assert_eq!(custom_engine.num_threads(), 2);
        
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        {
            let simd_engine = budgeted_simd_engine(custom_budget);
            assert_eq!(simd_engine.num_threads(), 2);
        }
    }
}