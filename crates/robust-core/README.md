# robust-core

Core traits, types, and computational primitives that form the foundation of the robust statistics ecosystem.

## Architecture

The crate implements a three-layer architecture for maximum performance and composability:

### Layer 1: Compute Primitives
Low-level SIMD-optimized operations with compile-time dispatch:
- `ScalarBackend` - Portable fallback implementation
- `Avx2Backend` - AVX2 SIMD operations (x86_64)
- `Avx512Backend` - AVX512 SIMD operations (x86_64)
- `SseBackend` - SSE SIMD operations (x86_64)

### Layer 2: Execution Engines
Control execution strategy and primitive selection:
- `SequentialEngine` - Single-threaded execution
- `ParallelEngine` - Multi-threaded with Rayon (requires `parallel` feature)
- `BudgetedEngine` - Hierarchical execution with thread budget control

### Layer 3: Statistical Components
- **Kernels**: Algorithm-specific operations (`StatisticalKernel`)
- **Estimators**: User-facing APIs (`StatefulEstimator`, `CentralTendencyEstimator`)
- **Caches**: Computation reuse (`UnifiedWeightCache`, `ComputationCache`)
- **Factories**: Zero-cost construction (`EstimatorFactory`)

## Core Concepts

### Compute Primitives

```rust
use robust_core::{ComputePrimitives, ScalarBackend};

// Create a scalar backend
let backend = ScalarBackend;

// Use primitives
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let sum = backend.sum(&data);
let mean = backend.mean(&data);

// Sparse operations
let indices = vec![0, 2, 4];
let weights = vec![0.3, 0.4, 0.3];
let weighted_sum = backend.sparse_weighted_sum(&data, &indices, &weights);
```

### Execution Engines

```rust
use robust_core::{auto_engine, scalar_sequential, simd_sequential};
use robust_core::ExecutionEngine;

// Create engines with convenience functions
let auto = auto_engine();        // Auto-selects best configuration
let scalar = scalar_sequential(); // Scalar operations, sequential
let simd = simd_sequential();    // SIMD operations if available, sequential

// With parallel execution (requires "parallel" feature)
#[cfg(feature = "parallel")]
{
    use robust_core::{scalar_parallel, simd_parallel};
    let parallel = simd_parallel(); // SIMD operations, parallel
}

// Access primitives from engine
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let sum = auto.primitives().sum(&data);
```

### Factory Pattern

Enable zero-cost estimator construction with engine injection:

```rust
use robust_core::{EstimatorFactory, StatefulEstimator, ExecutionEngine, Numeric};

// Example factory implementation
#[derive(Clone)]
struct MyEstimatorFactory<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, E: ExecutionEngine<T>> EstimatorFactory<T, E> for MyEstimatorFactory<T> {
    type Estimator = MyEstimator<T, E>;
    
    fn create(&self, engine: E) -> Self::Estimator {
        MyEstimator::new(engine)
    }
    
    fn create_cache(&self) -> <Self::Estimator as StatefulEstimator<T>>::State {
        () // or your cache type
    }
}
```

### Hierarchical Execution

Prevent thread oversubscription in nested operations:

```rust
use robust_core::execution::{auto_budgeted_engine, HierarchicalExecution};

// Create a budgeted engine that manages thread allocation
let engine = auto_budgeted_engine();

// Get subordinate engine for nested operations
let subordinate = engine.subordinate();
// subordinate automatically uses appropriate execution strategy
```

### Weight Caching

Efficient caching for expensive computations:

```rust
use robust_core::{UnifiedWeightCache, WeightComputer, CachePolicy, SparseWeights};

// Example weight computer
#[derive(Clone)]
struct MyWeightComputer;

impl WeightComputer<f64> for MyWeightComputer {
    fn compute_sparse(&self, n: usize, p: f64) -> SparseWeights<f64> {
        // Expensive computation
        SparseWeights::new(vec![0], vec![1.0])
    }
}

// Create cache with LRU policy
let cache = UnifiedWeightCache::new(
    MyWeightComputer,
    CachePolicy::Lru { max_entries: 1024 }
);
```

### Workspace Management

Memory pools for efficient allocation:

```rust
use robust_core::{AlignedBuffer, with_bootstrap_workspace};

// Create aligned buffer for SIMD (default alignment)
let mut buffer = AlignedBuffer::<f64>::new(1000);
buffer.resize(500); // Set active length
let slice = buffer.as_mut_slice();

// Bootstrap workspace for resampling operations
with_bootstrap_workspace(1000, 100, |workspace| {
    // Use pre-allocated workspace for bootstrap operations
    let indices = workspace.indices();
    let resample_buffer = workspace.resample_buffer();
});
```

## Numeric Types

The crate is generic over numeric types through the `Numeric` trait:

```rust
use robust_core::Numeric;

fn compute_something<T: Numeric>(data: &[T]) -> T::Float {
    // T can be f64, f32, i32, etc.
    // T::Float is the floating-point type for results (f64 or f32)
    // T::Aggregate is the type for accumulation
    let sum: T::Aggregate = data.iter()
        .map(|&x| T::Aggregate::from(x))
        .sum();
    // Convert aggregate to float
    sum.into()
}
```

## Error Handling

Unified error types for the ecosystem:

```rust
use robust_core::{Error, Result};

fn my_function(data: &[f64]) -> Result<f64> {
    if data.is_empty() {
        return Err(Error::empty_input("my_function"));
    }
    Ok(42.0)
}
```

## Features

- `parallel` - Enable parallel execution engines
- `avx2` - Enable AVX2 SIMD backend
- `avx512` - Enable AVX512 SIMD backend 
- `sse` - Enable SSE SIMD backend
- `nightly` - Enable unstable features

## Performance Tips

1. **Backend Selection**: Use `auto_engine()` for automatic optimization
2. **Cache Reuse**: Share `UnifiedWeightCache` instances across operations
3. **Workspace Pools**: Use `with_bootstrap_workspace` for repeated allocations
4. **Hierarchical Execution**: Use `auto_budgeted_engine()` and subordinate engines in nested operations
5. **SIMD Alignment**: `AlignedBuffer::new()` uses appropriate alignment automatically