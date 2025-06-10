//! Factory implementations for common estimators
//!
//! This module provides zero-cost factory implementations for creating
//! estimators with the appropriate execution engine.

use robust_core::{EstimatorFactory, ExecutionEngine, Numeric};





/// Generic factory that wraps a closure
///
/// This allows creating custom factories for any estimator type.
#[derive(Clone)]
pub struct ClosureFactory<F> {
    factory_fn: F,
}

impl<F> ClosureFactory<F> {
    pub fn new(factory_fn: F) -> Self {
        Self { factory_fn }
    }
}

impl<T: Numeric, E, Est, F> EstimatorFactory<T, E> for ClosureFactory<F>
where
    E: ExecutionEngine<T>,
    Est: robust_core::StatefulEstimator<T>,
    F: Fn(E) -> Est + Clone + Send + Sync,
{
    type Estimator = Est;
    
    fn create(&self, engine: E) -> Self::Estimator {
        (self.factory_fn)(engine)
    }
    
    fn create_cache(&self) -> <Self::Estimator as robust_core::StatefulEstimator<T>>::State {
        // For closure factories, we'll use a no-cache approach
        // Users who need custom caching should implement a full factory type
        panic!("ClosureFactory doesn't support automatic cache creation. Please implement a custom factory type.");
    }
}

/// Factory for estimators that don't need an execution engine
///
/// This is useful for simple estimators that don't benefit from
/// SIMD or parallelization.
#[derive(Clone, Debug)]
pub struct StaticFactory<Est> {
    estimator: Est,
}

impl<Est: Clone> StaticFactory<Est> {
    pub fn new(estimator: Est) -> Self {
        Self { estimator }
    }
}

impl<T: Numeric, E, Est> EstimatorFactory<T, E> for StaticFactory<Est>
where
    E: ExecutionEngine<T>,
    Est: robust_core::StatefulEstimator<T> + Clone + Send + Sync,
{
    type Estimator = Est;
    
    fn create(&self, _engine: E) -> Self::Estimator {
        self.estimator.clone()
    }
    
    fn create_cache(&self) -> <Self::Estimator as robust_core::StatefulEstimator<T>>::State {
        // Static factories also don't support automatic cache creation
        panic!("StaticFactory doesn't support automatic cache creation. Please implement a custom factory type.");
    }
}

/// Convenience functions for creating factories



pub fn closure_factory<F>(f: F) -> ClosureFactory<F> {
    ClosureFactory::new(f)
}

pub fn static_factory<Est>(estimator: Est) -> StaticFactory<Est> 
where
    Est: Clone,
{
    StaticFactory::new(estimator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use robust_core::{execution::scalar_sequential, StatefulEstimator};


    #[test]
    fn test_closure_factory() {
        use robust_core::adapters::NoCache;
        
        // Simple mock estimator
        #[derive(Clone)]
        struct MockEstimator;
        
        impl robust_core::StatefulEstimator for MockEstimator {
            type State = NoCache;
            
            fn estimate_with_cache(&self, data: &[f64], _cache: &Self::State) -> robust_core::Result<f64> {
                Ok(data.iter().sum::<f64>() / data.len() as f64)
            }
        }
        
        let factory = closure_factory(|_engine| MockEstimator);
        let engine = scalar_sequential();
        let estimator = factory.create(engine);
        
        // Verify it works by using it
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cache = NoCache;
        let result = estimator.estimate_with_cache(&data, &cache).unwrap();
        assert_eq!(result, 3.0);
    }
}