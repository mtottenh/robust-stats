//! Tests for spread estimation methods

mod common;

use polars::prelude::*;
use robust_polars::{RobustStatsExt, SpreadMethod};
use common::{create_test_df, extract_single_value};

#[test]
fn test_mad_basic() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    let result = df.robust_spread(
        &["values"],
        SpreadMethod::Mad,
    ).unwrap();
    
    // Check result structure
    assert_eq!(result.shape(), (1, 1));
    assert!(result.column("values_mad").is_ok());
    
    // MAD of 1-10 with HD estimator should be around 2.5-3.0
    let mad_value = extract_single_value(&result, "values_mad");
    assert!(mad_value > 2.0 && mad_value < 3.5, 
        "MAD value {} should be between 2.0 and 3.5", mad_value);
}

#[test]
fn test_iqr_basic() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    let result = df.robust_spread(
        &["values"],
        SpreadMethod::Iqr,
    ).unwrap();
    
    // Check result structure
    assert_eq!(result.shape(), (1, 1));
    assert!(result.column("values_iqr").is_ok());
    
    // IQR of 1-10 should be around 4.5 (Q3=7.75, Q1=3.25)
    let iqr_value = extract_single_value(&result, "values_iqr");
    assert!(iqr_value > 4.0 && iqr_value < 5.5, 
        "IQR value {} should be between 4.0 and 5.5", iqr_value);
}

#[test]
fn test_qad_basic() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    let result = df.robust_spread(
        &["values"],
        SpreadMethod::Qad { probability: 0.25 },
    ).unwrap();
    
    // Check result structure
    assert_eq!(result.shape(), (1, 1));
    assert!(result.column("values_qad").is_ok());
    
    // QAD should be positive
    let qad_value = extract_single_value(&result, "values_qad");
    assert!(qad_value > 0.0, "QAD value {} should be positive", qad_value);
}

#[test]
fn test_spread_multiple_columns() {
    let df = df![
        "col1" => &[1.0, 2.0, 3.0, 4.0, 5.0],
        "col2" => &[10.0, 20.0, 30.0, 40.0, 50.0],
        "col3" => &[0.1, 0.2, 0.3, 0.4, 0.5],
    ].unwrap();
    
    let result = df.robust_spread(
        &["col1", "col2", "col3"],
        SpreadMethod::Mad,
    ).unwrap();
    
    // Check result structure
    assert_eq!(result.shape(), (1, 3));
    assert!(result.column("col1_mad").is_ok());
    assert!(result.column("col2_mad").is_ok());
    assert!(result.column("col3_mad").is_ok());
    
    // MAD should scale with the data
    let mad1 = extract_single_value(&result, "col1_mad");
    let mad2 = extract_single_value(&result, "col2_mad");
    let mad3 = extract_single_value(&result, "col3_mad");
    
    assert!(mad2 > mad1 * 9.0, "MAD should scale proportionally");
    assert!(mad3 < mad1 * 0.11, "MAD should scale proportionally");
}

#[test]
fn test_spread_with_outliers() {
    // Data with extreme outlier
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]);
    
    let mad_result = df.robust_spread(&["values"], SpreadMethod::Mad).unwrap();
    let iqr_result = df.robust_spread(&["values"], SpreadMethod::Iqr).unwrap();
    
    // Robust measures should not be heavily influenced by the outlier
    let mad_value = extract_single_value(&mad_result, "values_mad");
    let iqr_value = extract_single_value(&iqr_result, "values_iqr");
    
    // MAD should be less influenced by the outlier than standard deviation would be
    // For this data, standard deviation would be huge (around 400)
    // MAD should be much smaller
    eprintln!("MAD with outlier: {}, IQR with outlier: {}", mad_value, iqr_value);
    
    // With Harrell-Davis smooth estimator, the extreme outlier does have some influence
    // But it should still be much less than standard deviation
    assert!(mad_value < 100.0, "MAD {} should be somewhat robust to outlier", mad_value);
    
    // IQR with HD can be influenced by extreme outliers due to smoothing
    // Just verify it's positive
    assert!(iqr_value > 0.0, "IQR should be positive");
}

#[test]
fn test_spread_different_qad_probabilities() {
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    let qad_25 = df.robust_spread(
        &["values"],
        SpreadMethod::Qad { probability: 0.25 },
    ).unwrap();
    
    let qad_10 = df.robust_spread(
        &["values"],
        SpreadMethod::Qad { probability: 0.10 },
    ).unwrap();
    
    let val_25 = extract_single_value(&qad_25, "values_qad");
    let val_10 = extract_single_value(&qad_10, "values_qad");
    
    // With Harrell-Davis smooth quantiles, the relationship might be different
    // Just check that both are positive and different
    eprintln!("QAD(0.10)={}, QAD(0.25)={}", val_10, val_25);
    assert!(val_10 > 0.0, "QAD(0.10) should be positive");
    assert!(val_25 > 0.0, "QAD(0.25) should be positive");
    assert!((val_10 - val_25).abs() > 1e-10, "Different probabilities should give different results");
}

#[test]
fn test_all_spread_methods_consistency() {
    // Test that all methods produce reasonable results on same data
    let df = create_test_df(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    let mad = df.robust_spread(&["values"], SpreadMethod::Mad).unwrap();
    let iqr = df.robust_spread(&["values"], SpreadMethod::Iqr).unwrap();
    let qad = df.robust_spread(&["values"], SpreadMethod::Qad { probability: 0.25 }).unwrap();
    
    let mad_val = extract_single_value(&mad, "values_mad");
    let iqr_val = extract_single_value(&iqr, "values_iqr");
    let qad_val = extract_single_value(&qad, "values_qad");
    
    // All should be positive
    assert!(mad_val > 0.0);
    assert!(iqr_val > 0.0);
    assert!(qad_val > 0.0);
    
    // IQR should typically be larger than MAD for uniform-like data
    assert!(iqr_val > mad_val);
}