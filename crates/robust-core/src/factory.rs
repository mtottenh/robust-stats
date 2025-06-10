//! Factory pattern for creating estimators with execution engines
//!
//! This module provides the core factory trait that enables zero-cost
//! estimator construction with the appropriate execution engine.

use crate::{StatefulEstimator, Numeric};

/// Factory for creating estimators with the appropriate engine
///
/// This trait enables zero-cost estimator construction with the
/// correct execution engine for hierarchical operations. It's designed
/// to work with any estimator type and any execution engine.
///
/// # Type Parameters
/// * `T` - The numeric type (f64, f32, etc.)
/// * `Engine` - The execution engine type to be injected into estimators
///
/// # Example
/// ```rust,ignore
/// struct MyEstimatorFactory<T>;
/// 
/// impl<T: Numeric, E: ExecutionEngine<T>> EstimatorFactory<T, E> for MyEstimatorFactory<T> {
///     type Estimator = MyEstimator<T, E>;
///     
///     fn create(&self, engine: E) -> Self::Estimator {
///         MyEstimator::new(engine)
///     }
///     
///     fn create_cache(&self) -> CacheType {
///         CacheType::new()
///     }
/// }
/// ```
pub trait EstimatorFactory<T: Numeric, Engine>: Clone + Send + Sync {
    /// The type of estimator this factory creates
    type Estimator: StatefulEstimator<T>;
    
    /// Create an estimator with the given engine
    ///
    /// This method should construct the estimator with the provided
    /// execution engine, enabling hierarchical execution control.
    fn create(&self, engine: Engine) -> Self::Estimator;
    
    /// Create the cache/state for the estimator
    ///
    /// This method creates the appropriate cache or state object
    /// for the estimator type. The cache can be shared across
    /// multiple estimator instances for efficiency.
    fn create_cache(&self) -> <Self::Estimator as StatefulEstimator<T>>::State;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionEngine, Error, Result, SequentialEngine, ScalarBackend};
    use std::marker::PhantomData;
    
    // Mock state/cache type
    #[derive(Clone, Debug)]
    struct MockCache {
        size: usize,
    }
    
    impl MockCache {
        fn new(size: usize) -> Self {
            Self { size }
        }
    }
    
    // Mock estimator type
    struct MockEstimator<T: Numeric, E> {
        engine: E,
        _phantom: PhantomData<T>,
    }
    
    impl<T: Numeric, E> MockEstimator<T, E> {
        fn new(engine: E) -> Self {
            Self {
                engine,
                _phantom: PhantomData,
            }
        }
    }
    
    impl<T: Numeric, E: ExecutionEngine<T>> StatefulEstimator<T> for MockEstimator<T, E> {
        type State = MockCache;
        
        fn estimate_with_cache(&self, sample: &[T], cache: &Self::State) -> Result<T::Float> {
            if sample.is_empty() {
                return Err(Error::empty_input("estimate"));
            }
            // Return cache size as the "estimate"
            Ok(num_traits::NumCast::from(cache.size).unwrap())
        }
        
        fn estimate_sorted_with_cache(&self, sorted_sample: &[T], cache: &Self::State) -> Result<T::Float> {
            self.estimate_with_cache(sorted_sample, cache)
        }
    }
    
    // Mock factory
    #[derive(Clone)]
    struct MockFactory<T> {
        cache_size: usize,
        _phantom: PhantomData<T>,
    }
    
    impl<T> MockFactory<T> {
        fn new(cache_size: usize) -> Self {
            Self {
                cache_size,
                _phantom: PhantomData,
            }
        }
    }
    
    impl<T: Numeric, E: ExecutionEngine<T>> EstimatorFactory<T, E> for MockFactory<T> {
        type Estimator = MockEstimator<T, E>;
        
        fn create(&self, engine: E) -> Self::Estimator {
            MockEstimator::new(engine)
        }
        
        fn create_cache(&self) -> MockCache {
            MockCache::new(self.cache_size)
        }
    }
    
    #[test]
    fn test_factory_basic_usage() {
        let factory = MockFactory::<f64>::new(100);
        let engine = SequentialEngine::new(ScalarBackend);
        
        // Create estimator
        let estimator = factory.create(engine);
        
        // Create cache
        let cache = <MockFactory<f64> as EstimatorFactory<f64, SequentialEngine<f64, ScalarBackend>>>::create_cache(&factory);
        assert_eq!(cache.size, 100);
        
        // Use estimator with cache
        let data = vec![1.0, 2.0, 3.0];
        let result = estimator.estimate_with_cache(&data, &cache).unwrap();
        assert_eq!(result, 100.0); // Returns cache size
    }
    
    #[test]
    fn test_factory_clone() {
        let factory1 = MockFactory::<f64>::new(50);
        let factory2 = factory1.clone();
        
        assert_eq!(factory1.cache_size, factory2.cache_size);
        
        // Both factories should create equivalent estimators
        let engine1 = SequentialEngine::new(ScalarBackend);
        let engine2 = SequentialEngine::new(ScalarBackend);
        
        let _est1 = factory1.create(engine1);
        let _est2 = factory2.create(engine2);
        
        // Both estimators are of the same type
    }
    
    #[test]
    fn test_factory_different_engines() {
        let factory = MockFactory::<f64>::new(75);
        
        // Create with sequential engine
        let seq_engine = SequentialEngine::new(ScalarBackend);
        let _seq_est = factory.create(seq_engine);
        
        // Create with parallel engine (if available)
        #[cfg(feature = "parallel")]
        {
            use crate::ParallelEngine;
            let par_engine = ParallelEngine::new(ScalarBackend);
            let _par_est = factory.create(par_engine);
        }
    }
    
    #[test]
    fn test_factory_cache_sharing() {
        let factory = MockFactory::<f64>::new(200);
        
        // Create multiple estimators but share cache
        let engine1 = SequentialEngine::new(ScalarBackend);
        let engine2 = SequentialEngine::new(ScalarBackend);
        
        let est1 = factory.create(engine1);
        let est2 = factory.create(engine2);
        
        // Single shared cache
        let cache = <MockFactory<f64> as EstimatorFactory<f64, SequentialEngine<f64, ScalarBackend>>>::create_cache(&factory);
        
        let data = vec![1.0, 2.0, 3.0];
        let result1 = est1.estimate_with_cache(&data, &cache).unwrap();
        let result2 = est2.estimate_with_cache(&data, &cache).unwrap();
        
        // Both should use the same cache
        assert_eq!(result1, result2);
        assert_eq!(result1, 200.0);
    }
    
    // Test with different numeric types
    struct GenericFactory<T: Numeric> {
        multiplier: T::Float,
        _phantom: PhantomData<T>,
    }
    
    impl<T: Numeric> GenericFactory<T> {
        fn new(multiplier: T::Float) -> Self {
            Self {
                multiplier,
                _phantom: PhantomData,
            }
        }
    }
    
    impl<T: Numeric> Clone for GenericFactory<T> {
        fn clone(&self) -> Self {
            Self {
                multiplier: self.multiplier,
                _phantom: PhantomData,
            }
        }
    }
    
    struct GenericEstimator<T: Numeric, E> {
        multiplier: T::Float,
        _engine: E,
        _phantom: PhantomData<T>,
    }
    
    impl<T: Numeric, E> StatefulEstimator<T> for GenericEstimator<T, E> {
        type State = ();
        
        fn estimate_with_cache(&self, sample: &[T], _cache: &Self::State) -> Result<T::Float> {
            if sample.is_empty() {
                return Err(Error::empty_input("estimate"));
            }
            Ok(sample[0].to_float() * self.multiplier)
        }
        
        fn estimate_sorted_with_cache(&self, sorted_sample: &[T], cache: &Self::State) -> Result<T::Float> {
            self.estimate_with_cache(sorted_sample, cache)
        }
    }
    
    impl<T: Numeric, E: ExecutionEngine<T>> EstimatorFactory<T, E> for GenericFactory<T> {
        type Estimator = GenericEstimator<T, E>;
        
        fn create(&self, engine: E) -> Self::Estimator {
            GenericEstimator {
                multiplier: self.multiplier,
                _engine: engine,
                _phantom: PhantomData,
            }
        }
        
        fn create_cache(&self) -> () {}
    }
    
    #[test]
    fn test_factory_with_f32() {
        let factory = GenericFactory::<f32>::new(2.5);
        let engine = SequentialEngine::new(ScalarBackend);
        let estimator = factory.create(engine);
        
        let data = vec![4.0f32];
        let result = estimator.estimate_with_cache(&data, &()).unwrap();
        assert!((result - 10.0).abs() < 1e-6); // 4.0 * 2.5
    }
    
    #[test]
    fn test_factory_with_i32() {
        // Test that factory works with integer types
        let factory = GenericFactory::<i32>::new(3.0);
        let engine = SequentialEngine::new(ScalarBackend);
        let estimator = factory.create(engine);
        
        let data = vec![5i32];
        let result = estimator.estimate_with_cache(&data, &()).unwrap();
        assert!((result - 15.0).abs() < 1e-6); // 5 * 3.0
    }
    
    #[test]
    fn test_factory_thread_safety() {
        use std::thread;
        use std::sync::Arc;
        
        let factory = Arc::new(MockFactory::<f64>::new(42));
        let mut handles = vec![];
        
        // Spawn threads that use the factory
        for i in 0..4 {
            let factory_clone = Arc::clone(&factory);
            let handle = thread::spawn(move || {
                let engine = SequentialEngine::new(ScalarBackend);
                let estimator = factory_clone.create(engine);
                let cache = <MockFactory<f64> as EstimatorFactory<f64, SequentialEngine<f64, ScalarBackend>>>::create_cache(&factory_clone);
                
                let data = vec![i as f64];
                estimator.estimate_with_cache(&data, &cache).unwrap()
            });
            handles.push(handle);
        }
        
        // All threads should get the same result (cache size)
        for handle in handles {
            let result = handle.join().unwrap();
            assert_eq!(result, 42.0);
        }
    }
}