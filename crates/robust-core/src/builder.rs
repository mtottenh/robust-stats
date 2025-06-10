//! Type-safe builder pattern for constructing estimators
//!
//! This module provides a type-state builder pattern that ensures
//! estimators are constructed with all required components at compile time.
//!

use crate::{ExecutionEngine, CachePolicy};
use crate::numeric::Numeric;
use std::marker::PhantomData;

/// Type-state markers for builder pattern
pub mod state {
    /// Marker for builder that needs an execution engine
    pub struct NeedsEngine;
    
    /// Marker for builder that needs cache configuration
    pub struct NeedsCache;
    
    /// Marker for builder that is ready to build
    pub struct Ready;
}

pub use state::{NeedsEngine, NeedsCache, Ready};

/// Builder state trait
pub trait BuilderState {}

impl BuilderState for NeedsEngine {}
impl BuilderState for NeedsCache {}
impl BuilderState for Ready {}

/// Type-safe builder for constructing estimators
///
/// This builder uses the type-state pattern to ensure all required
/// components are provided before building.
pub struct EstimatorBuilder<T, E, C, State> {
    pub(crate) engine: E,
    pub(crate) cache_policy: C,
    pub(crate) _state: PhantomData<(T, State)>,
}

impl<T: Numeric> EstimatorBuilder<T, (), (), NeedsEngine> {
    /// Create a new estimator builder
    pub fn new() -> Self {
        Self {
            engine: (),
            cache_policy: (),
            _state: PhantomData,
        }
    }
}

impl<T: Numeric> Default for EstimatorBuilder<T, (), (), NeedsEngine> {
    fn default() -> Self {
        Self::new()
    }
}

// Methods available when needing an engine
impl<T: Numeric, C> EstimatorBuilder<T, (), C, NeedsEngine> {
    /// Set the execution engine
    pub fn with_engine<E: ExecutionEngine<T>>(
        self,
        engine: E,
    ) -> EstimatorBuilder<T, E, C, NeedsCache> {
        EstimatorBuilder {
            engine,
            cache_policy: self.cache_policy,
            _state: PhantomData,
        }
    }
}

// Methods available when needing cache configuration
impl<T: Numeric, E: ExecutionEngine<T>> EstimatorBuilder<T, E, (), NeedsCache> {
    /// Set the cache policy
    pub fn with_cache(
        self,
        policy: CachePolicy,
    ) -> EstimatorBuilder<T, E, CachePolicy, Ready> {
        EstimatorBuilder {
            engine: self.engine,
            cache_policy: policy,
            _state: PhantomData,
        }
    }
    
    /// Use no caching
    pub fn without_cache(self) -> EstimatorBuilder<T, E, CachePolicy, Ready> {
        self.with_cache(CachePolicy::NoCache)
    }
}

// Methods available when ready to build
impl<T: Numeric, E: ExecutionEngine<T>> EstimatorBuilder<T, E, CachePolicy, Ready> {
    /// Get the configured engine
    pub fn engine(&self) -> &E {
        &self.engine
    }
    
    /// Get the configured cache policy
    pub fn cache_policy(&self) -> &CachePolicy {
        &self.cache_policy
    }
    
    /// Consume the builder and return its components
    pub fn into_parts(self) -> (E, CachePolicy) {
        (self.engine, self.cache_policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_builder_type_safety() {
        // This is a compile-time test - these lines should not compile:
        // let bad1 = EstimatorBuilder::new().with_cache(CachePolicy::NoCache);
        // let bad2 = EstimatorBuilder::new().into_parts();
        
        // This should compile:
        let builder = EstimatorBuilder::new()
            .with_engine(crate::execution::scalar_sequential())
            .with_cache(CachePolicy::NoCache);
        
        let (engine, cache) = builder.into_parts();
        assert!(matches!(cache, CachePolicy::NoCache));
        assert_eq!(engine.num_threads(), 1);
    }
    
    #[test]
    fn test_simd_builder() {
        let builder = EstimatorBuilder::<f64, _, _, _>::new()
            .with_engine(crate::execution::simd_sequential())
            .without_cache();
        let (engine, _) = builder.into_parts();
        assert_eq!(engine.num_threads(), 1);
    }
}
