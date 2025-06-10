//! Common test utilities for robust-polars tests

use polars::prelude::*;

/// Helper function to create a test DataFrame with specific values
pub fn create_test_df(values: &[f64]) -> DataFrame {
    df!["values" => values].unwrap()
}

/// Helper function to extract quantile results
pub fn extract_quantile_values(df: &DataFrame, col_prefix: &str, quantiles: &[f64]) -> Vec<f64> {
    let mut values = Vec::new();
    for q in quantiles {
        let col_name = format!("{}_q{:.2}", col_prefix, q);
        let series = df.column(&col_name).unwrap();
        values.push(series.f64().unwrap().get(0).unwrap());
    }
    values
}

/// Helper function to extract a single value from a result DataFrame
pub fn extract_single_value(df: &DataFrame, col_name: &str) -> f64 {
    df.column(col_name)
        .unwrap()
        .f64()
        .unwrap()
        .get(0)
        .unwrap()
}