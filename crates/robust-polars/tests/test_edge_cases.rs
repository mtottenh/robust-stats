//! Tests for edge cases and error handling

mod common;

use polars::prelude::*;
use robust_polars::{RobustStatsExt, QuantileMethod, SpreadMethod};

#[test]
fn test_empty_data() {
    let df = DataFrame::new(vec![
        Series::new(PlSmallStr::from("values"), Vec::<f64>::new()).into()
    ]).unwrap();
    
    // Test quantiles with empty data
    let q_result = df.robust_quantiles(&["values"], &[0.5], QuantileMethod::HarrellDavis).unwrap();
    let q_value = q_result.column("values_q0.50").unwrap().f64().unwrap().get(0).unwrap();
    assert!(q_value.is_nan());
    
    // Test spread with empty data
    let s_result = df.robust_spread(&["values"], SpreadMethod::Mad).unwrap();
    let s_value = s_result.column("values_mad").unwrap().f64().unwrap().get(0).unwrap();
    assert!(s_value.is_nan());
}

#[test]
fn test_single_value() {
    let df = DataFrame::new(vec![
        Series::new(PlSmallStr::from("values"), vec![42.0]).into()
    ]).unwrap();
    
    // Test quantiles with single value
    let q_result = df.robust_quantiles(&["values"], &[0.5], QuantileMethod::HarrellDavis).unwrap();
    let q_value = q_result.column("values_q0.50").unwrap().f64().unwrap().get(0).unwrap();
    // Single value should return that value
    assert!((q_value - 42.0).abs() < 1e-10);
    
    // Test spread with single value
    let s_result = df.robust_spread(&["values"], SpreadMethod::Mad).unwrap();
    let s_value = s_result.column("values_mad").unwrap().f64().unwrap().get(0).unwrap();
    // MAD of single value should be 0 or NaN
    assert!(s_value == 0.0 || s_value.is_nan());
}

#[test]
fn test_invalid_column_error() {
    let df = DataFrame::new(vec![
        Series::new(PlSmallStr::from("values"), vec![1.0, 2.0, 3.0]).into()
    ]).unwrap();
    
    // Test invalid column for quantiles
    let q_result = df.robust_quantiles(
        &["nonexistent"],
        &[0.5],
        QuantileMethod::HarrellDavis,
    );
    assert!(q_result.is_err());
    
    // Test invalid column for spread
    let s_result = df.robust_spread(&["nonexistent"], SpreadMethod::Mad);
    assert!(s_result.is_err());
}

#[test]
fn test_non_numeric_column_error() {
    let df = DataFrame::new(vec![
        Series::new(PlSmallStr::from("strings"), vec!["a", "b", "c"]).into()
    ]).unwrap();
    
    // Test non-numeric column for quantiles
    let q_result = df.robust_quantiles(&["strings"], &[0.5], QuantileMethod::HarrellDavis);
    assert!(q_result.is_err());
    
    // Test non-numeric column for spread
    let s_result = df.robust_spread(&["strings"], SpreadMethod::Mad);
    assert!(s_result.is_err());
}