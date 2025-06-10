//! Changepoint detection kernels (Layer 2)
//!
//! This module provides domain-specific computational kernels for changepoint detection.
//! Kernels express algorithms in terms of primitive operations to enable SIMD acceleration.

use robust_core::{ComputePrimitives, Result, StatisticalKernel, Numeric};
use num_traits::{One, Zero, NumCast, FromPrimitive};
use nalgebra::{DMatrix, DVector};

/// Kernel for CUSUM computations
/// 
/// Expresses CUSUM algorithm in terms of primitive operations:
/// - Vector subtraction for deviations
/// - Element-wise max for cumulative sums
#[derive(Clone, Debug)]
pub struct CusumKernel<T: Numeric, P: ComputePrimitives<T>> {
    primitives: P,
    drift: T::Float,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> CusumKernel<T, P> {
    pub fn new(primitives: P, drift: T::Float) -> Self {
        Self { primitives, drift, _phantom: std::marker::PhantomData }
    }
    
    /// Compute CUSUM statistics
    pub fn compute_cusum(&self, data: &[T], reference: T::Float) -> (Vec<T::Float>, Vec<T::Float>) 
    where
        T::Float: PartialOrd,
    {
        use num_traits::Zero;
        
        // Convert data to float and compute deviations
        let mut positive_cusum = Vec::with_capacity(data.len());
        let mut negative_cusum = Vec::with_capacity(data.len());
        
        let mut pos_sum = T::Float::zero();
        let mut neg_sum = T::Float::zero();
        
        for &x in data {
            let x_float = x.to_float();
            let deviation = x_float - reference;
            
            // Positive CUSUM: detect increases
            let pos_adjusted = deviation - self.drift;
            let zero = T::Float::zero();
            pos_sum = if pos_sum + pos_adjusted > zero {
                pos_sum + pos_adjusted
            } else {
                zero
            };
            positive_cusum.push(pos_sum);
            
            // Negative CUSUM: detect decreases
            let neg_adjusted = -deviation - self.drift;
            neg_sum = if neg_sum + neg_adjusted > zero {
                neg_sum + neg_adjusted
            } else {
                zero
            };
            negative_cusum.push(neg_sum);
        }
        
        (positive_cusum, negative_cusum)
    }
    
    /// Batch CUSUM computation for multiple reference values
    pub fn compute_cusum_batch(&self, data: &[T], references: &[T::Float]) -> Vec<(Vec<T::Float>, Vec<T::Float>)> {
        references.iter()
            .map(|&ref_val| self.compute_cusum(data, ref_val))
            .collect()
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> StatisticalKernel<T> for CusumKernel<T, P> {
    type Primitives = P;
    
    fn primitives(&self) -> &Self::Primitives {
        &self.primitives
    }
}

/// Kernel for polynomial fitting operations
/// 
/// Decomposes polynomial fitting into primitive operations:
/// - Matrix multiplication via dot products
/// - Vector operations for residuals
#[derive(Clone, Debug)]
pub struct PolynomialKernel<T: Numeric, P: ComputePrimitives<T>> {
    primitives: P,
    degree: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> PolynomialKernel<T, P> {
    pub fn new(primitives: P, degree: usize) -> Self {
        Self { primitives, degree, _phantom: std::marker::PhantomData }
    }
    
    /// Build design matrix for polynomial regression
    /// X[i,j] = x[i]^j for j in 0..=degree
    pub fn build_design_matrix(&self, x: &[T]) -> DMatrix<T::Float> 
    where
        T::Float: nalgebra::Scalar,
    {
        let n = x.len();
        let mut matrix = DMatrix::zeros(n, self.degree + 1);
        
        // Column 0: all ones (x^0)
        for i in 0..n {
            matrix[(i, 0)] = T::Float::one();
        }
        
        // Use primitives to compute powers efficiently
        let mut current_power: Vec<T::Float> = x.iter().map(|&v| v.to_float()).collect();
        
        for col in 1..=self.degree {
            // Copy current power to matrix column
            for i in 0..n {
                matrix[(i, col)] = current_power[i];
            }
            
            // For next iteration: multiply by x again
            if col < self.degree {
                // Element-wise multiplication
                for i in 0..n {
                    current_power[i] = current_power[i] * x[i].to_float();
                }
            }
        }
        
        matrix
    }
    
    /// Solve normal equations X^T X β = X^T y using primitives
    pub fn fit_polynomial(&self, x: &[T], y: &[T]) -> Result<Vec<T::Float>> 
    where
        T::Float: nalgebra::Scalar + nalgebra::RealField + FromPrimitive,
    {
        if x.len() != y.len() || x.is_empty() {
            return Err(robust_core::Error::InvalidInput(
                "Input vectors must have same non-zero length".to_string()
            ));
        }
        
        let design = self.build_design_matrix(x);
        let y_float: Vec<T::Float> = y.iter().map(|&v| v.to_float()).collect();
        let _y_vec = DVector::from_row_slice(&y_float);
        
        // Compute X^T X using dot products
        let _xt = design.transpose();
        let mut xtx = DMatrix::zeros(self.degree + 1, self.degree + 1);
        
        // Use primitives for matrix multiplication via dot products
        for i in 0..=self.degree {
            for j in 0..=self.degree {
                let col_i: Vec<T> = (0..x.len())
                    .map(|k| T::from_f64(NumCast::from(design[(k, i)]).unwrap()))
                    .collect();
                let col_j: Vec<T> = (0..x.len())
                    .map(|k| T::from_f64(NumCast::from(design[(k, j)]).unwrap()))
                    .collect();
                xtx[(i, j)] = NumCast::from(self.primitives.dot_product(&col_i, &col_j)).unwrap();
            }
        }
        
        // Compute X^T y using dot products
        let mut xty = DVector::zeros(self.degree + 1);
        for i in 0..=self.degree {
            let col_i: Vec<T> = (0..x.len())
                .map(|k| T::from_f64(NumCast::from(design[(k, i)]).unwrap()))
                .collect();
            xty[i] = NumCast::from(self.primitives.dot_product(&col_i, y)).unwrap();
        }
        
        // Solve using Cholesky decomposition (could also use primitives here)
        match xtx.clone().cholesky() {
            Some(chol) => {
                let coeffs = chol.solve(&xty);
                Ok(coeffs.as_slice().to_vec())
            }
            None => {
                // Fallback to SVD for rank-deficient cases
                let svd = xtx.svd(true, true);
                let coeffs = svd.solve(&xty, T::Float::from_f64(1e-10).unwrap())
                    .map_err(|_| robust_core::Error::Computation(
                        "Failed to solve polynomial system".to_string()
                    ))?;
                Ok(coeffs.as_slice().to_vec())
            }
        }
    }
    
    /// Compute residuals using primitives
    pub fn compute_residuals(&self, x: &[T], y: &[T], coeffs: &[T::Float]) -> Vec<T::Float> 
    where
        T::Float: nalgebra::Scalar,
    {
        let design = self.build_design_matrix(x);
        let mut predictions = vec![T::Float::zero(); x.len()];
        
        // Compute predictions using matrix-vector multiplication
        for i in 0..x.len() {
            let row: Vec<T> = (0..=self.degree)
                .map(|j| T::from_f64(NumCast::from(design[(i, j)]).unwrap()))
                .collect();
            let coeffs_t: Vec<T> = coeffs.iter()
                .map(|&c| T::from_f64(NumCast::from(c).unwrap()))
                .collect();
            predictions[i] = NumCast::from(self.primitives.dot_product(&row, &coeffs_t)).unwrap();
        }
        
        // Compute residuals = y - predictions
        y.iter()
            .zip(predictions.iter())
            .map(|(&y_i, &pred_i)| y_i.to_float() - pred_i)
            .collect()
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> StatisticalKernel<T> for PolynomialKernel<T, P> {
    type Primitives = P;
    
    fn primitives(&self) -> &Self::Primitives {
        &self.primitives
    }
}

/// Kernel for EWMA (Exponentially Weighted Moving Average) computations
#[derive(Clone, Debug)]
pub struct EwmaKernel<T: Numeric, P: ComputePrimitives<T>> {
    primitives: P,
    alpha: T::Float,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> EwmaKernel<T, P> {
    pub fn new(primitives: P, alpha: T::Float) -> Self {
        Self { primitives, alpha, _phantom: std::marker::PhantomData }
    }
    
    /// Compute EWMA using primitives for vector operations
    pub fn compute_ewma(&self, data: &[T], initial: Option<T::Float>) -> Vec<T::Float> {
        if data.is_empty() {
            return vec![];
        }
        
        let mut ewma = Vec::with_capacity(data.len());
        let mut current = initial.unwrap_or_else(|| data[0].to_float());
        
        // EWMA formula: S_t = α * X_t + (1 - α) * S_{t-1}
        let one_minus_alpha = T::Float::one() - self.alpha;
        
        for &value in data {
            current = self.alpha * value.to_float() + one_minus_alpha * current;
            ewma.push(current);
        }
        
        ewma
    }
    
    /// Compute EWMA deviations for changepoint detection
    pub fn compute_deviations(&self, data: &[T], ewma: &[T::Float]) -> Vec<T::Float> {
        // Compute |data - ewma| 
        data.iter()
            .zip(ewma.iter())
            .map(|(&d, &e)| {
                let diff = d.to_float() - e;
                if diff < num_traits::Zero::zero() {
                    -diff
                } else {
                    diff
                }
            })
            .collect()
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> StatisticalKernel<T> for EwmaKernel<T, P> {
    type Primitives = P;
    
    fn primitives(&self) -> &Self::Primitives {
        &self.primitives
    }
}

/// Kernel for sliding window operations
/// 
/// Provides efficient computation over rolling windows using primitives
#[derive(Clone, Debug)]
pub struct WindowKernel<T: Numeric, P: ComputePrimitives<T>> {
    primitives: P,
    window_size: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> WindowKernel<T, P> {
    pub fn new(primitives: P, window_size: usize) -> Self {
        Self { primitives, window_size, _phantom: std::marker::PhantomData }
    }
    
    /// Compute statistics for all windows efficiently
    pub fn compute_window_stats<F>(&self, data: &[T], stat_fn: F) -> Vec<T::Float>
    where
        F: Fn(&[T], &P) -> T::Float,
    {
        if data.len() < self.window_size {
            return vec![];
        }
        
        let n_windows = data.len() - self.window_size + 1;
        let mut results = Vec::with_capacity(n_windows);
        
        for i in 0..n_windows {
            let window = &data[i..i + self.window_size];
            results.push(stat_fn(window, &self.primitives));
        }
        
        results
    }
    
    /// Compute multiple statistics in a single pass for efficiency
    pub fn compute_window_stats_multi<F>(&self, data: &[T], stat_fns: &[F]) -> Vec<Vec<T::Float>>
    where
        F: Fn(&[T], &P) -> T::Float,
    {
        if data.len() < self.window_size {
            return vec![vec![]; stat_fns.len()];
        }
        
        let n_windows = data.len() - self.window_size + 1;
        let mut results = vec![Vec::with_capacity(n_windows); stat_fns.len()];
        
        // Single pass through windows
        for i in 0..n_windows {
            let window = &data[i..i + self.window_size];
            
            // Compute all statistics for this window
            for (j, stat_fn) in stat_fns.iter().enumerate() {
                results[j].push(stat_fn(window, &self.primitives));
            }
        }
        
        results
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> StatisticalKernel<T> for WindowKernel<T, P> {
    type Primitives = P;
    
    fn primitives(&self) -> &Self::Primitives {
        &self.primitives
    }
}