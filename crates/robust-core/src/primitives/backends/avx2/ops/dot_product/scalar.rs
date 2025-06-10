//! Scalar fallback implementation for dot product

use crate::Numeric;

/// Scalar implementation of dot product
pub fn dot_product_scalar<T: Numeric>(
    a: &[T],
    b: &[T],
) -> T {
    let n = a.len().min(b.len());
    let mut sum = <T as Numeric>::zero();
    
    for i in 0..n {
        sum = sum + a[i] * b[i];
    }
    
    sum
}