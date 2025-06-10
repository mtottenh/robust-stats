//! Factory implementations for quantile estimators
//!
//! This module provides zero-cost factory implementations for creating
//! quantile estimators with the appropriate execution engine.

use crate::{
    estimators::{
        harrell_davis, trimmed_harrell_davis, ConstantWidth, HarrellDavis, LinearWidth, SqrtWidth,
        TrimmedHarrellDavis,
    },
    weights::{ConstantWidthFn, LinearWidthFn, SqrtWidthFn},
    HDWeightComputer, QuantileAdapter, TrimmedHDWeightComputer,
};
use robust_core::{CachePolicy, UnifiedWeightCache};
use robust_core::{EstimatorFactory, ExecutionEngine, Numeric};

/// Factory for Harrell-Davis quantile estimator
#[derive(Clone, Debug)]
pub struct HarrellDavisFactory<T: Numeric> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric> Default for HarrellDavisFactory<T> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Factory for Trimmed Harrell-Davis quantile estimator with constant width
#[derive(Clone, Debug)]
pub struct TrimmedHDConstantFactory<T: Numeric> {
    pub width: f64,
    _phantom: std::marker::PhantomData<T>,
}

/// Factory for Trimmed Harrell-Davis quantile estimator with linear width
#[derive(Clone, Debug)]
pub struct TrimmedHDLinearFactory<T: Numeric> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric> Default for TrimmedHDLinearFactory<T> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Factory for Trimmed Harrell-Davis quantile estimator with sqrt width
#[derive(Clone, Debug)]
pub struct TrimmedHDSqrtFactory<T: Numeric> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric> Default for TrimmedHDSqrtFactory<T> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Numeric + num_traits::NumCast, E: ExecutionEngine<T>> EstimatorFactory<T, E>
    for HarrellDavisFactory<T>
{
    type Estimator = QuantileAdapter<T, HarrellDavis<T, E>>;

    fn create(&self, engine: E) -> Self::Estimator {
        let hd = harrell_davis(engine);
        QuantileAdapter::new(hd)
    }

    fn create_cache(&self) -> <Self::Estimator as robust_core::StatefulEstimator<T>>::State {
        UnifiedWeightCache::new(
            HDWeightComputer::new(),
            CachePolicy::Lru {
                max_entries: 1024 * 1024,
            },
        )
    }
}

impl<T: Numeric + num_traits::NumCast, E: ExecutionEngine<T>> EstimatorFactory<T, E>
    for TrimmedHDConstantFactory<T>
{
    type Estimator = QuantileAdapter<T, TrimmedHarrellDavis<T, E, ConstantWidth>>;

    fn create(&self, engine: E) -> Self::Estimator {
        let width_fn = ConstantWidth::new(self.width);
        let thd = trimmed_harrell_davis(engine, width_fn);
        QuantileAdapter::new(thd)
    }

    fn create_cache(&self) -> <Self::Estimator as robust_core::StatefulEstimator<T>>::State {
        let computer = TrimmedHDWeightComputer::new(ConstantWidthFn::new(self.width));
        UnifiedWeightCache::new(
            computer,
            CachePolicy::Lru {
                max_entries: 1024 * 1024,
            },
        )
    }
}

impl<T: Numeric + num_traits::NumCast, E: ExecutionEngine<T>> EstimatorFactory<T, E>
    for TrimmedHDLinearFactory<T>
{
    type Estimator = QuantileAdapter<T, TrimmedHarrellDavis<T, E, LinearWidth>>;

    fn create(&self, engine: E) -> Self::Estimator {
        let thd = trimmed_harrell_davis(engine, LinearWidth);
        QuantileAdapter::new(thd)
    }

    fn create_cache(&self) -> <Self::Estimator as robust_core::StatefulEstimator<T>>::State {
        let computer = TrimmedHDWeightComputer::new(LinearWidthFn);
        UnifiedWeightCache::new(
            computer,
            CachePolicy::Lru {
                max_entries: 1024 * 1024,
            },
        )
    }
}

impl<T: Numeric + num_traits::NumCast, E: ExecutionEngine<T>> EstimatorFactory<T, E>
    for TrimmedHDSqrtFactory<T>
{
    type Estimator = QuantileAdapter<T, TrimmedHarrellDavis<T, E, SqrtWidth>>;

    fn create(&self, engine: E) -> Self::Estimator {
        let thd = trimmed_harrell_davis(engine, SqrtWidth);
        QuantileAdapter::new(thd)
    }

    fn create_cache(&self) -> <Self::Estimator as robust_core::StatefulEstimator<T>>::State {
        let computer = TrimmedHDWeightComputer::new(SqrtWidthFn);
        UnifiedWeightCache::new(
            computer,
            CachePolicy::Lru {
                max_entries: 1024 * 1024,
            },
        )
    }
}

/// Convenience functions for creating factories
pub fn harrell_davis_factory<T: Numeric>() -> HarrellDavisFactory<T> {
    HarrellDavisFactory::default()
}

pub fn trimmed_hd_constant_factory<T: Numeric>(width: f64) -> TrimmedHDConstantFactory<T> {
    TrimmedHDConstantFactory {
        width,
        _phantom: std::marker::PhantomData,
    }
}

pub fn trimmed_hd_linear_factory<T: Numeric>() -> TrimmedHDLinearFactory<T> {
    TrimmedHDLinearFactory::default()
}

pub fn trimmed_hd_sqrt_factory<T: Numeric>() -> TrimmedHDSqrtFactory<T> {
    TrimmedHDSqrtFactory::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use robust_core::{execution::scalar_sequential, StatefulEstimator};

    #[test]
    fn test_harrell_davis_factory() {
        use robust_core::primitives::ScalarBackend;
        use robust_core::SequentialEngine;

        let factory = harrell_davis_factory::<f64>();
        let engine = scalar_sequential();
        let estimator = factory.create(engine);

        // Just verify it creates successfully
        let cache = <HarrellDavisFactory<f64> as EstimatorFactory<
            f64,
            SequentialEngine<f64, ScalarBackend>,
        >>::create_cache(&factory);

        // Test it works
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = estimator.estimate_with_cache(&data, &cache).unwrap();
        assert!(result > 0.0); // Simple sanity check
    }
}
