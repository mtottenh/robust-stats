//! Generic numeric trait hierarchy for type-safe statistical computing
//!
//! This module provides the type foundation for generic numeric computing across different
//! numeric types (f64, f32, i32, etc.) without imposing any computational infrastructure.
//!
//! # Design Philosophy
//!
//! - **Pure type constraints**: Defines relationships between numeric types
//! - **No computational layer**: All computation happens through Layer 1 (ComputePrimitives)
//! - **Type safety**: Can't accidentally mix numeric types
//! - **Extensible**: Easy to add new numeric types

use bytemuck::Pod;
use num_traits::{Num, Float};
use std::fmt::Debug;
use std::ops::{Add, Mul};

/// Wrapper type for aggregates that can be constructed from large integers
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct F64Aggregate(pub f64);

impl From<i64> for F64Aggregate {
    fn from(val: i64) -> Self {
        F64Aggregate(val as f64)
    }
}

impl From<u64> for F64Aggregate {
    fn from(val: u64) -> Self {
        F64Aggregate(val as f64)
    }
}

impl From<f64> for F64Aggregate {
    fn from(val: f64) -> Self {
        F64Aggregate(val)
    }
}

impl From<F64Aggregate> for f64 {
    fn from(val: F64Aggregate) -> Self {
        val.0
    }
}

impl Add for F64Aggregate {
    type Output = Self;
    
    fn add(self, rhs: Self) -> Self::Output {
        F64Aggregate(self.0 + rhs.0)
    }
}

impl Mul for F64Aggregate {
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self::Output {
        F64Aggregate(self.0 * rhs.0)
    }
}

impl Float for F64Aggregate {
    fn nan() -> Self { F64Aggregate(f64::nan()) }
    fn infinity() -> Self { F64Aggregate(f64::infinity()) }
    fn neg_infinity() -> Self { F64Aggregate(f64::neg_infinity()) }
    fn neg_zero() -> Self { F64Aggregate(-0.0) }
    fn min_value() -> Self { F64Aggregate(f64::MIN) }
    fn min_positive_value() -> Self { F64Aggregate(f64::MIN_POSITIVE) }
    fn max_value() -> Self { F64Aggregate(f64::MAX) }
    fn is_nan(self) -> bool { self.0.is_nan() }
    fn is_infinite(self) -> bool { self.0.is_infinite() }
    fn is_finite(self) -> bool { self.0.is_finite() }
    fn is_normal(self) -> bool { self.0.is_normal() }
    fn classify(self) -> std::num::FpCategory { self.0.classify() }
    fn floor(self) -> Self { F64Aggregate(self.0.floor()) }
    fn ceil(self) -> Self { F64Aggregate(self.0.ceil()) }
    fn round(self) -> Self { F64Aggregate(self.0.round()) }
    fn trunc(self) -> Self { F64Aggregate(self.0.trunc()) }
    fn fract(self) -> Self { F64Aggregate(self.0.fract()) }
    fn abs(self) -> Self { F64Aggregate(self.0.abs()) }
    fn signum(self) -> Self { F64Aggregate(self.0.signum()) }
    fn is_sign_positive(self) -> bool { self.0.is_sign_positive() }
    fn is_sign_negative(self) -> bool { self.0.is_sign_negative() }
    fn recip(self) -> Self { F64Aggregate(self.0.recip()) }
    fn powi(self, n: i32) -> Self { F64Aggregate(self.0.powi(n)) }
    fn powf(self, n: Self) -> Self { F64Aggregate(self.0.powf(n.0)) }
    fn sqrt(self) -> Self { F64Aggregate(self.0.sqrt()) }
    fn exp(self) -> Self { F64Aggregate(self.0.exp()) }
    fn exp2(self) -> Self { F64Aggregate(self.0.exp2()) }
    fn ln(self) -> Self { F64Aggregate(self.0.ln()) }
    fn log(self, base: Self) -> Self { F64Aggregate(self.0.log(base.0)) }
    fn log2(self) -> Self { F64Aggregate(self.0.log2()) }
    fn log10(self) -> Self { F64Aggregate(self.0.log10()) }
    fn max(self, other: Self) -> Self { F64Aggregate(self.0.max(other.0)) }
    fn min(self, other: Self) -> Self { F64Aggregate(self.0.min(other.0)) }
    fn abs_sub(self, other: Self) -> Self { F64Aggregate((self.0 - other.0).abs()) }
    fn cbrt(self) -> Self { F64Aggregate(self.0.cbrt()) }
    fn hypot(self, other: Self) -> Self { F64Aggregate(self.0.hypot(other.0)) }
    fn sin(self) -> Self { F64Aggregate(self.0.sin()) }
    fn cos(self) -> Self { F64Aggregate(self.0.cos()) }
    fn tan(self) -> Self { F64Aggregate(self.0.tan()) }
    fn asin(self) -> Self { F64Aggregate(self.0.asin()) }
    fn acos(self) -> Self { F64Aggregate(self.0.acos()) }
    fn atan(self) -> Self { F64Aggregate(self.0.atan()) }
    fn atan2(self, other: Self) -> Self { F64Aggregate(self.0.atan2(other.0)) }
    fn sin_cos(self) -> (Self, Self) { 
        let (s, c) = self.0.sin_cos();
        (F64Aggregate(s), F64Aggregate(c))
    }
    fn exp_m1(self) -> Self { F64Aggregate(self.0.exp_m1()) }
    fn ln_1p(self) -> Self { F64Aggregate(self.0.ln_1p()) }
    fn sinh(self) -> Self { F64Aggregate(self.0.sinh()) }
    fn cosh(self) -> Self { F64Aggregate(self.0.cosh()) }
    fn tanh(self) -> Self { F64Aggregate(self.0.tanh()) }
    fn asinh(self) -> Self { F64Aggregate(self.0.asinh()) }
    fn acosh(self) -> Self { F64Aggregate(self.0.acosh()) }
    fn atanh(self) -> Self { F64Aggregate(self.0.atanh()) }
    fn integer_decode(self) -> (u64, i16, i8) { self.0.integer_decode() }
    fn epsilon() -> Self { F64Aggregate(f64::EPSILON) }
    fn to_degrees(self) -> Self { F64Aggregate(self.0.to_degrees()) }
    fn to_radians(self) -> Self { F64Aggregate(self.0.to_radians()) }
    
    fn mul_add(self, a: Self, b: Self) -> Self {
        F64Aggregate(self.0.mul_add(a.0, b.0))
    }
}

impl num_traits::Num for F64Aggregate {
    type FromStrRadixErr = <f64 as num_traits::Num>::FromStrRadixErr;
    
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f64::from_str_radix(str, radix).map(F64Aggregate)
    }
}

impl num_traits::Zero for F64Aggregate {
    fn zero() -> Self { F64Aggregate(0.0) }
    fn is_zero(&self) -> bool { self.0 == 0.0 }
}

impl num_traits::One for F64Aggregate {
    fn one() -> Self { F64Aggregate(1.0) }
}

impl std::ops::Sub for F64Aggregate {
    type Output = Self;
    
    fn sub(self, rhs: Self) -> Self::Output {
        F64Aggregate(self.0 - rhs.0)
    }
}

impl std::ops::Div for F64Aggregate {
    type Output = Self;
    
    fn div(self, rhs: Self) -> Self::Output {
        F64Aggregate(self.0 / rhs.0)
    }
}

impl std::ops::Rem for F64Aggregate {
    type Output = Self;
    
    fn rem(self, rhs: Self) -> Self::Output {
        F64Aggregate(self.0 % rhs.0)
    }
}

impl std::ops::Neg for F64Aggregate {
    type Output = Self;
    
    fn neg(self) -> Self::Output {
        F64Aggregate(-self.0)
    }
}

impl std::ops::AddAssign for F64Aggregate {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl num_traits::NumCast for F64Aggregate {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        n.to_f64().map(F64Aggregate)
    }
}

impl num_traits::ToPrimitive for F64Aggregate {
    fn to_i64(&self) -> Option<i64> { self.0.to_i64() }
    fn to_u64(&self) -> Option<u64> { self.0.to_u64() }
    fn to_f64(&self) -> Option<f64> { Some(self.0) }
}

/// Base trait for numeric types that can be used in statistical computations
pub trait Numeric: Pod + Num + Copy + PartialOrd + Debug + Send + Sync {
    /// Type used for aggregate operations (sum, mean, variance)
    /// This allows integer types to use f64 for aggregates to prevent overflow
    type Aggregate: Float + From<Self> + Into<f64> + Send + Sync + std::ops::AddAssign;
    
    /// Type used for floating-point operations (std dev, correlations)
    type Float: Float + From<Self> + Into<f64> + num_traits::Zero + num_traits::NumCast + Send + Sync + std::ops::AddAssign;
    
    /// Convert to floating point for statistical operations
    fn to_float(self) -> Self::Float;
    
    /// Check if value is finite (always true for integers)
    fn is_finite(&self) -> bool;
    
    /// Get the zero value
    fn zero() -> Self;
    
    /// Get the one value
    fn one() -> Self;
    
    /// Convert from f64 (for creating constants)
    fn from_f64(val: f64) -> Self;
    
    /// Convert to f64 (for operations that need f64)
    fn to_f64(&self) -> f64;
}

// Removed ComputeOps trait and ComputeExt trait
// All computation now happens through Layer 1 (ComputePrimitives)

// =============================================================================
// Numeric implementations for concrete types
// =============================================================================

impl Numeric for f64 {
    type Aggregate = f64;
    type Float = f64;
    
    fn to_float(self) -> f64 {
        self
    }
    
    fn is_finite(&self) -> bool {
        f64::is_finite(*self)
    }
    
    fn zero() -> Self {
        0.0
    }
    
    fn one() -> Self {
        1.0
    }
    
    fn from_f64(val: f64) -> Self {
        val
    }
    
    fn to_f64(&self) -> f64 {
        *self
    }
}

impl Numeric for f32 {
    type Aggregate = f64; // Use f64 for better precision in aggregates
    type Float = f32;
    
    fn to_float(self) -> f32 {
        self
    }
    
    fn is_finite(&self) -> bool {
        f32::is_finite(*self)
    }
    
    fn zero() -> Self {
        0.0
    }
    
    fn one() -> Self {
        1.0
    }
    
    fn from_f64(val: f64) -> Self {
        val as f32
    }
    
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

impl Numeric for i32 {
    type Aggregate = f64; // Use f64 to prevent overflow
    type Float = f64;
    
    fn to_float(self) -> f64 {
        self as f64
    }
    
    fn is_finite(&self) -> bool {
        true // Integers are always finite
    }
    
    fn zero() -> Self {
        0
    }
    
    fn one() -> Self {
        1
    }
    
    fn from_f64(val: f64) -> Self {
        val as i32
    }
    
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

impl Numeric for u32 {
    type Aggregate = f64; // Use f64 to prevent overflow
    type Float = f64;
    
    fn to_float(self) -> f64 {
        self as f64
    }
    
    fn is_finite(&self) -> bool {
        true // Integers are always finite
    }
    
    fn zero() -> Self {
        0
    }
    
    fn one() -> Self {
        1
    }
    
    fn from_f64(val: f64) -> Self {
        val as u32
    }
    
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

impl Numeric for i64 {
    type Aggregate = F64Aggregate;
    type Float = F64Aggregate;
    
    fn to_float(self) -> F64Aggregate {
        F64Aggregate(self as f64)
    }
    
    fn is_finite(&self) -> bool {
        true
    }
    
    fn zero() -> Self {
        0
    }
    
    fn one() -> Self {
        1
    }
    
    fn from_f64(val: f64) -> Self {
        val as i64
    }
    
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

impl Numeric for u64 {
    type Aggregate = F64Aggregate;
    type Float = F64Aggregate;
    
    fn to_float(self) -> F64Aggregate {
        F64Aggregate(self as f64)
    }
    
    fn is_finite(&self) -> bool {
        true
    }
    
    fn zero() -> Self {
        0
    }
    
    fn one() -> Self {
        1
    }
    
    fn from_f64(val: f64) -> Self {
        val as u64
    }
    
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_numeric_trait() {
        // Test f64
        assert_eq!(f64::zero(), 0.0);
        assert_eq!(f64::one(), 1.0);
        assert!(5.0f64.is_finite());
        
        // Test i32
        assert_eq!(i32::zero(), 0);
        assert_eq!(i32::one(), 1);
        assert!(42i32.is_finite());
        
        // Test aggregate types
        let x: f64 = 5.0;
        let agg = <f64 as Numeric>::Aggregate::from(x);
        // For f64, Aggregate is f64, so this is a no-op
        assert_eq!(agg, 5.0);
    }
}