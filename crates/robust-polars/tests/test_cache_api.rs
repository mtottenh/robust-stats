//! Tests for the cache API functionality

mod common;

use polars::prelude::*;
use robust_polars::{RobustStatsExt, QuantileMethod, TrimWidth, QuantileCache, SpreadMethod};
use robust_core::{UnifiedWeightCache, CachePolicy};
use robust_quantile::{HDWeightComputer, TrimmedHDWeightComputer, weights::{SqrtWidthFn, ConstantWidthFn}};
use std::sync::Arc;
use std::time::Instant;
use common::create_test_df;

#[test]
fn test_quantiles_cached_with_matching_cache() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    // Create HD cache
    let cache = QuantileCache::HarrellDavis(Arc::new(
        UnifiedWeightCache::new(
            HDWeightComputer::<f64>::new(),
            CachePolicy::Lru { max_entries: 100 }
        )
    ));
    
    let result = df.robust_quantiles_cached(
        &["values"],
        &[0.25, 0.5, 0.75],
        QuantileMethod::HarrellDavis,
        &cache,
    ).unwrap();
    
    // Should work fine with matching cache
    assert_eq!(result.shape(), (1, 3));
}

#[test]
fn test_quantiles_cached_with_mismatched_cache() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    // Create HD cache but try to use with Trimmed method
    let cache = QuantileCache::HarrellDavis(Arc::new(
        UnifiedWeightCache::new(
            HDWeightComputer::<f64>::new(),
            CachePolicy::Lru { max_entries: 100 }
        )
    ));
    
    let result = df.robust_quantiles_cached(
        &["values"],
        &[0.5],
        QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Sqrt },
        &cache,
    );
    
    // Should error due to cache mismatch
    assert!(result.is_err());
}

#[test]
fn test_trimmed_cache_variants() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    // Test Constant width cache
    let const_cache = QuantileCache::TrimmedConstant { 
        cache: Arc::new(UnifiedWeightCache::new(
            TrimmedHDWeightComputer::<f64, _>::new(ConstantWidthFn::new(0.1)),
            CachePolicy::Lru { max_entries: 100 }
        )),
        width: 0.1,
    };
    
    let const_result = df.robust_quantiles_cached(
        &["values"],
        &[0.5],
        QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Constant(0.1) },
        &const_cache,
    ).unwrap();
    
    assert!(const_result.column("values_q0.50").is_ok());
    
    // Test Sqrt width cache
    let sqrt_cache = QuantileCache::TrimmedSqrt(Arc::new(
        UnifiedWeightCache::new(
            TrimmedHDWeightComputer::<f64, _>::new(SqrtWidthFn),
            CachePolicy::Lru { max_entries: 100 }
        )
    ));
    
    let sqrt_result = df.robust_quantiles_cached(
        &["values"],
        &[0.5],
        QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Sqrt },
        &sqrt_cache,
    ).unwrap();
    
    assert!(sqrt_result.column("values_q0.50").is_ok());
}

#[test]
fn test_cache_type_mismatch_f32() {
    let values_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let df = DataFrame::new(vec![
        Series::new(PlSmallStr::from("values"), values_f32).into()
    ]).unwrap();
    
    // Create f64 cache but use with f32 data
    let cache = QuantileCache::HarrellDavis(Arc::new(
        UnifiedWeightCache::new(
            HDWeightComputer::<f64>::new(),
            CachePolicy::NoCache
        )
    ));
    
    let result = df.robust_quantiles_cached(
        &["values"],
        &[0.5],
        QuantileMethod::HarrellDavis,
        &cache,
    );
    
    // This should error because cache type (f64) doesn't match data type (f32)
    assert!(result.is_err());
}

#[test]
fn test_cache_performance_benefit() {
    // Create a larger dataset for performance testing
    let values: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let df = create_test_df(&values);
    
    // Many quantiles to stress test caching
    let quantiles: Vec<f64> = (1..=99).map(|i| i as f64 / 100.0).collect();
    
    // Run once to warm up
    let _ = df.robust_quantiles(
        &["values"],
        &quantiles,
        QuantileMethod::HarrellDavis,
    ).unwrap();
    
    // Time with global cache (should be fast on second run due to caching)
    let start = Instant::now();
    let result1 = df.robust_quantiles(
        &["values"],
        &quantiles,
        QuantileMethod::HarrellDavis,
    ).unwrap();
    let global_time = start.elapsed();
    
    // Run same computation again - should be even faster due to cache hits
    let start = Instant::now();
    let result2 = df.robust_quantiles(
        &["values"],
        &quantiles,
        QuantileMethod::HarrellDavis,
    ).unwrap();
    let cached_time = start.elapsed();
    
    // Verify results are identical
    for q in &quantiles {
        let col_name = format!("values_q{:.2}", q);
        let val1 = result1.column(&col_name).unwrap().f64().unwrap().get(0).unwrap();
        let val2 = result2.column(&col_name).unwrap().f64().unwrap().get(0).unwrap();
        assert!((val1 - val2).abs() < 1e-10, "Results should be identical");
    }
    
    // Cache should make second run faster (or at least not slower)
    eprintln!("First run: {:?}, Second run (cached): {:?}", global_time, cached_time);
    assert!(cached_time <= global_time.saturating_add(std::time::Duration::from_millis(10)), 
            "Cached run should not be significantly slower");
}

#[test]
fn test_spread_cached_with_hd_cache() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    // Create HD cache
    let cache = QuantileCache::HarrellDavis(Arc::new(
        UnifiedWeightCache::new(
            HDWeightComputer::<f64>::new(),
            CachePolicy::Lru { max_entries: 100 }
        )
    ));
    
    let result = df.robust_spread_cached(
        &["values"],
        SpreadMethod::Mad,
        &cache,
    ).unwrap();
    
    // Should produce same results as non-cached version
    let mad_value = result.column("values_mad").unwrap()
        .f64().unwrap().get(0).unwrap();
    assert!(mad_value > 2.0 && mad_value < 3.5);
}

#[test]
fn test_spread_cached_with_trimmed_cache() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    // Create trimmed cache
    let cache = QuantileCache::TrimmedSqrt(Arc::new(
        UnifiedWeightCache::new(
            TrimmedHDWeightComputer::<f64, _>::new(SqrtWidthFn),
            CachePolicy::Lru { max_entries: 100 }
        )
    ));
    
    let result = df.robust_spread_cached(
        &["values"],
        SpreadMethod::Mad,
        &cache,
    ).unwrap();
    
    // Trimmed estimator should produce different (often smaller) spread estimate
    let mad_value = result.column("values_mad").unwrap()
        .f64().unwrap().get(0).unwrap();
    assert!(mad_value > 0.0, "Trimmed MAD should be positive");
}

#[test]
fn test_spread_cache_reuse_from_quantiles() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    // Create shared cache
    let cache = QuantileCache::HarrellDavis(Arc::new(
        UnifiedWeightCache::new(
            HDWeightComputer::<f64>::new(),
            CachePolicy::Lru { max_entries: 100 }
        )
    ));
    
    // First compute quantiles to warm cache
    let _ = df.robust_quantiles_cached(
        &["values"],
        &[0.25, 0.5, 0.75],
        QuantileMethod::HarrellDavis,
        &cache,
    ).unwrap();
    
    // Then compute spread - should benefit from warmed cache
    let result = df.robust_spread_cached(
        &["values"],
        SpreadMethod::Iqr,  // IQR uses Q1 and Q3
        &cache,
    ).unwrap();
    
    let iqr_value = result.column("values_iqr").unwrap()
        .f64().unwrap().get(0).unwrap();
    assert!(iqr_value > 4.0 && iqr_value < 5.5);
}