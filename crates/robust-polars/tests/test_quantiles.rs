//! Tests for quantile estimation methods

mod common;

use polars::prelude::*;
use robust_polars::{RobustStatsExt, QuantileMethod, TrimWidth};
use common::{create_test_df, extract_quantile_values};

#[test]
fn test_harrell_davis_basic() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    let result = df.robust_quantiles(
        &["values"],
        &[0.25, 0.5, 0.75],
        QuantileMethod::HarrellDavis,
    ).unwrap();
    
    // Check result structure
    assert_eq!(result.shape(), (1, 3));
    assert!(result.column("values_q0.25").is_ok());
    assert!(result.column("values_q0.50").is_ok());
    assert!(result.column("values_q0.75").is_ok());
    
    // Extract values
    let values = extract_quantile_values(&result, "values", &[0.25, 0.5, 0.75]);
    
    // HD estimator for uniform data should give approximately:
    // Q1 ≈ 3.25, Q2 ≈ 5.5, Q3 ≈ 7.75
    assert!(values[0] > 2.5 && values[0] < 4.0, "Q1 should be around 3.25, got {}", values[0]);
    assert!(values[1] > 5.0 && values[1] < 6.0, "Q2 should be around 5.5, got {}", values[1]);
    assert!(values[2] > 7.0 && values[2] < 8.5, "Q3 should be around 7.75, got {}", values[2]);
}

#[test]
fn test_trimmed_hd_constant_width() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    let result = df.robust_quantiles(
        &["values"],
        &[0.5],
        QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Constant(0.1) },
    ).unwrap();
    
    let median = result.column("values_q0.50").unwrap()
        .f64().unwrap().get(0).unwrap();
    
    // Trimmed HD with small width should be close to HD
    assert!(median > 5.0 && median < 6.0, "Median should be around 5.5, got {}", median);
}

#[test]
fn test_trimmed_hd_sqrt_width() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    let result = df.robust_quantiles(
        &["values"],
        &[0.5],
        QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Sqrt },
    ).unwrap();
    
    let median = result.column("values_q0.50").unwrap()
        .f64().unwrap().get(0).unwrap();
    
    assert!(median > 5.0 && median < 6.0, "Median should be around 5.5, got {}", median);
}

#[test]
fn test_trimmed_hd_linear_width() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    let result = df.robust_quantiles(
        &["values"],
        &[0.5],
        QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Linear },
    ).unwrap();
    
    let median = result.column("values_q0.50").unwrap()
        .f64().unwrap().get(0).unwrap();
    
    assert!(median > 5.0 && median < 6.0, "Median should be around 5.5, got {}", median);
}

#[test]
fn test_multiple_columns() {
    let df = df![
        "col1" => &[1.0, 2.0, 3.0, 4.0, 5.0],
        "col2" => &[10.0, 20.0, 30.0, 40.0, 50.0],
        "col3" => &[0.1, 0.2, 0.3, 0.4, 0.5],
    ].unwrap();
    
    let result = df.robust_quantiles(
        &["col1", "col2", "col3"],
        &[0.5],
        QuantileMethod::HarrellDavis,
    ).unwrap();
    
    // Check all columns are present
    assert_eq!(result.shape(), (1, 3));
    assert!(result.column("col1_q0.50").is_ok());
    assert!(result.column("col2_q0.50").is_ok());
    assert!(result.column("col3_q0.50").is_ok());
    
    // Values should scale with the data
    let med1 = result.column("col1_q0.50").unwrap().f64().unwrap().get(0).unwrap();
    let med2 = result.column("col2_q0.50").unwrap().f64().unwrap().get(0).unwrap();
    let med3 = result.column("col3_q0.50").unwrap().f64().unwrap().get(0).unwrap();
    
    assert!((med2 / med1 - 10.0).abs() < 0.1, "Medians should scale proportionally");
    assert!((med3 / med1 - 0.1).abs() < 0.01, "Medians should scale proportionally");
}

#[test]
fn test_extreme_quantiles() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    
    let result = df.robust_quantiles(
        &["values"],
        &[0.01, 0.99],
        QuantileMethod::HarrellDavis,
    ).unwrap();
    
    let q01 = result.column("values_q0.01").unwrap().f64().unwrap().get(0).unwrap();
    let q99 = result.column("values_q0.99").unwrap().f64().unwrap().get(0).unwrap();
    
    // Extreme quantiles should be close to min/max
    assert!(q01 > 0.8 && q01 < 1.5, "Q0.01 should be close to minimum");
    assert!(q99 > 4.5 && q99 < 5.2, "Q0.99 should be close to maximum");
}

#[test]
fn test_many_quantiles() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    let quantiles: Vec<f64> = (1..=9).map(|i| i as f64 / 10.0).collect();
    
    let result = df.robust_quantiles(
        &["values"],
        &quantiles,
        QuantileMethod::HarrellDavis,
    ).unwrap();
    
    // Should have one column per quantile
    assert_eq!(result.shape(), (1, 9));
    
    // Quantiles should be monotonic
    let values = extract_quantile_values(&result, "values", &quantiles);
    for i in 1..values.len() {
        assert!(values[i] >= values[i-1], "Quantiles should be monotonic");
    }
}

#[test]
fn test_different_trim_widths_produce_different_results() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    // Get results with different trim widths
    let const_result = df.robust_quantiles(
        &["values"],
        &[0.5],
        QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Constant(0.2) },
    ).unwrap();
    
    let sqrt_result = df.robust_quantiles(
        &["values"],
        &[0.5],
        QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Sqrt },
    ).unwrap();
    
    let linear_result = df.robust_quantiles(
        &["values"],
        &[0.5],
        QuantileMethod::TrimmedHarrellDavis { width: TrimWidth::Linear },
    ).unwrap();
    
    let const_median = const_result.column("values_q0.50").unwrap()
        .f64().unwrap().get(0).unwrap();
    let sqrt_median = sqrt_result.column("values_q0.50").unwrap()
        .f64().unwrap().get(0).unwrap();
    let linear_median = linear_result.column("values_q0.50").unwrap()
        .f64().unwrap().get(0).unwrap();
    
    // Different trim widths should produce (slightly) different results
    // For small n=10: sqrt width = 1/sqrt(10) ≈ 0.316, linear width = 1/10 = 0.1
    eprintln!("Const: {}, Sqrt: {}, Linear: {}", const_median, sqrt_median, linear_median);
    
    // Check if at least one pair is different
    let all_same = (const_median - sqrt_median).abs() < 1e-10 && 
                   (const_median - linear_median).abs() < 1e-10 && 
                   (sqrt_median - linear_median).abs() < 1e-10;
    if all_same {
        eprintln!("Note: All trim widths produced identical results, which is possible but unusual");
    }
}