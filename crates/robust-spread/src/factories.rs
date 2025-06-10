//! Factory implementations for spread estimators
//!
//! This module provides zero-cost factory implementations for creating
//! spread estimators with the appropriate execution engine.

use robust_core::{EstimatorFactory, ExecutionEngine, StatefulEstimator, Numeric};
use crate::{Mad, Qad, Iqr, SpreadAdapter};

/// Factory for Median Absolute Deviation (MAD) estimator with Harrell-Davis quantile estimation
#[cfg(feature = "quantile")]
#[derive(Clone, Debug)]
pub struct MADFactory<T: Numeric> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "quantile")]
impl<T: Numeric> Default for MADFactory<T> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "quantile")]
impl<T: Numeric + num_traits::NumCast, E: ExecutionEngine<T>> EstimatorFactory<T, E> for MADFactory<T> {
    type Estimator = SpreadAdapter<T, Mad<T, E::Primitives>, robust_quantile::HarrellDavis<T, E>>;
    
    fn create(&self, engine: E) -> Self::Estimator {
        let mad = Mad::new(engine.primitives().clone());
        let hd = robust_quantile::estimators::harrell_davis(engine);
        SpreadAdapter::new(mad, hd)
    }
    
    fn create_cache(&self) -> <Self::Estimator as StatefulEstimator<T>>::State {
        use robust_quantile::HDWeightComputer;
        robust_core::UnifiedWeightCache::new(HDWeightComputer::new(), robust_core::CachePolicy::Lru { max_entries: 1024 * 1024 })
    }
}

/// Factory for Quantile Absolute Deviation (QAD) estimator with Harrell-Davis quantile estimation
#[derive(Clone, Debug)]
pub struct QADFactory<T: Numeric> {
    pub probability: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric> QADFactory<T> {
    pub fn new(probability: f64) -> Self {
        assert!(probability > 0.0 && probability < 1.0, "Probability must be in (0, 1)");
        Self { 
            probability,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "quantile")]
impl<T: Numeric + num_traits::NumCast, E: ExecutionEngine<T>> EstimatorFactory<T, E> for QADFactory<T> {
    type Estimator = SpreadAdapter<T, Qad<T, E::Primitives>, robust_quantile::HarrellDavis<T, E>>;
    
    fn create(&self, engine: E) -> Self::Estimator {
        let qad = Qad::new(engine.primitives().clone(), self.probability)
            .expect("Invalid QAD probability");
        let hd = robust_quantile::estimators::harrell_davis(engine);
        SpreadAdapter::new(qad, hd)
    }
    
    fn create_cache(&self) -> <Self::Estimator as StatefulEstimator<T>>::State {
        use robust_quantile::HDWeightComputer;
        robust_core::UnifiedWeightCache::new(HDWeightComputer::new(), robust_core::CachePolicy::Lru { max_entries: 1024 * 1024 })
    }
}

/// Factory for Interquartile Range (IQR) estimator with Harrell-Davis quantile estimation
#[cfg(feature = "quantile")]
#[derive(Clone, Debug)]
pub struct IqrFactory<T: Numeric> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "quantile")]
impl<T: Numeric> Default for IqrFactory<T> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "quantile")]
impl<T: Numeric + num_traits::NumCast, E: ExecutionEngine<T>> EstimatorFactory<T, E> for IqrFactory<T> {
    type Estimator = SpreadAdapter<T, Iqr<T, E::Primitives>, robust_quantile::HarrellDavis<T, E>>;
    
    fn create(&self, engine: E) -> Self::Estimator {
        let iqr = Iqr::new(engine.primitives().clone());
        let hd = robust_quantile::estimators::harrell_davis(engine);
        SpreadAdapter::new(iqr, hd)
    }
    
    fn create_cache(&self) -> <Self::Estimator as StatefulEstimator<T>>::State {
        use robust_quantile::HDWeightComputer;
        robust_core::UnifiedWeightCache::new(HDWeightComputer::new(), robust_core::CachePolicy::Lru { max_entries: 1024 * 1024 })
    }
}

/// Convenience functions for creating factories

#[cfg(feature = "quantile")]
pub fn mad_factory<T: Numeric>() -> MADFactory<T> {
    MADFactory::default()
}

pub fn qad_factory<T: Numeric>(probability: f64) -> QADFactory<T> {
    QADFactory::new(probability)
}

#[cfg(feature = "quantile")]
pub fn iqr_factory<T: Numeric>() -> IqrFactory<T> {
    IqrFactory::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use robust_core::{execution::scalar_sequential, StatefulEstimator};

    #[test]
    #[cfg(feature = "quantile")]
    fn test_mad_factory() {
        use robust_core::SequentialEngine;
        use robust_core::primitives::ScalarBackend;
        
        let factory = mad_factory::<f64>();
        let engine = robust_core::scalar_sequential();
        let estimator = factory.create(engine);
        
        // Just verify it creates successfully
        let cache = <MADFactory<f64> as EstimatorFactory<f64, robust_core::execution::SequentialEngine<f64, robust_core::primitives::ScalarBackend>>>::create_cache(&factory);
        
        // Test it works
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = estimator.estimate_with_cache(&data, &cache).unwrap();
        assert!(result > 0.0); // Simple sanity check
    }

    #[test]
    fn test_qad_factory() {
        let factory = qad_factory::<f64>(0.25);
        assert_eq!(factory.probability, 0.25);
    }
}