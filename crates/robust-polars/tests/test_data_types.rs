//! Tests for different data type support

mod common;

use polars::prelude::*;
use robust_polars::{RobustStatsExt, QuantileMethod, SpreadMethod};

#[test]
fn test_float32_column() {
    let values_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let df = DataFrame::new(vec![
        Series::new(PlSmallStr::from("values"), values_f32).into()
    ]).unwrap();
    
    // Test quantiles with f32
    let q_result = df.robust_quantiles(
        &["values"],
        &[0.5],
        QuantileMethod::HarrellDavis,
    ).unwrap();
    
    assert_eq!(q_result.shape(), (1, 1));
    let median = q_result.column("values_q0.50").unwrap()
        .f64().unwrap().get(0).unwrap();
    assert!(median > 2.5 && median < 3.5);
    
    // Test spread with f32
    let s_result = df.robust_spread(&["values"], SpreadMethod::Mad).unwrap();
    assert_eq!(s_result.shape(), (1, 1));
    let mad_value = s_result.column("values_mad").unwrap()
        .f64().unwrap().get(0).unwrap();
    assert!(mad_value > 0.0);
}

#[test]
fn test_integer_columns() {
    let df = DataFrame::new(vec![
        Series::new(PlSmallStr::from("i64"), vec![1i64, 2, 3, 4, 5]).into(),
        Series::new(PlSmallStr::from("i32"), vec![1i32, 2, 3, 4, 5]).into(),
        Series::new(PlSmallStr::from("i16"), vec![1i16, 2, 3, 4, 5]).into(),
    ]).unwrap();
    
    // Test quantiles with integers
    let q_result = df.robust_quantiles(
        &["i64", "i32", "i16"],
        &[0.5],
        QuantileMethod::HarrellDavis,
    ).unwrap();
    
    assert_eq!(q_result.shape(), (1, 3));
    
    let med_i64 = q_result.column("i64_q0.50").unwrap()
        .f64().unwrap().get(0).unwrap();
    let med_i32 = q_result.column("i32_q0.50").unwrap()
        .f64().unwrap().get(0).unwrap();
    let med_i16 = q_result.column("i16_q0.50").unwrap()
        .f64().unwrap().get(0).unwrap();
    
    // All should produce the same median
    assert!((med_i64 - med_i32).abs() < 1e-10);
    assert!((med_i64 - med_i16).abs() < 1e-10);
    
    // Test spread with integers
    let s_result = df.robust_spread(&["i64", "i32"], SpreadMethod::Mad).unwrap();
    assert_eq!(s_result.shape(), (1, 2));
    
    let mad_i64 = s_result.column("i64_mad").unwrap()
        .f64().unwrap().get(0).unwrap();
    let mad_i32 = s_result.column("i32_mad").unwrap()
        .f64().unwrap().get(0).unwrap();
    
    // Both should produce similar results
    assert!((mad_i64 - mad_i32).abs() < 1e-10);
}