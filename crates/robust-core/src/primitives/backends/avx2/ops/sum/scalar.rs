//! Scalar fallback implementation for sum

use crate::Numeric;

/// Scalar implementation of sum
pub fn sum_scalar<T: Numeric>(
    data: &[T],
) -> T {
    let mut sum = <T as Numeric>::zero();
    
    for &val in data {
        sum = sum + val;
    }
    
    sum
}