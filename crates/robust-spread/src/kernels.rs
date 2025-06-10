//! Spread estimation kernels using the simplified primitives architecture

use crate::traits::SpreadKernel;
use robust_core::{ComputePrimitives, Result, StatisticalKernel, Numeric};
use robust_quantile::QuantileEstimator;
use num_traits::{NumCast, Zero, Float};

/// Kernel for computing Quantile Absolute Deviation (QAD)
///
/// QAD(p) = K * quantile_p(|X - median(X)|)
/// MAD is a special case with p = 0.5
#[derive(Clone, Debug)]
pub struct QadKernel<T: Numeric = f64, P: ComputePrimitives<T> = robust_core::primitives::ScalarBackend> {
    pub(crate) primitives: P,
    pub(crate) p: f64,
    pub(crate) constant: f64,
    _phantom: std::marker::PhantomData<T>,
}

/// Alias for MAD kernel for backward compatibility
pub type MadKernel<T, P> = QadKernel<T, P>;

impl<T: Numeric, P: ComputePrimitives<T>> QadKernel<T, P> {
    /// Create a new QAD kernel with the given primitives, quantile p, and constant
    pub fn new(primitives: P, p: f64, constant: f64) -> Self {
        Self {
            primitives,
            p,
            constant,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a MAD kernel (QAD with p=0.5 and constant=1.0)
    pub fn mad(primitives: P) -> Self {
        Self {
            primitives,
            p: 0.5,
            constant: 1.0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a standardized MAD kernel (QAD with p=0.5 and constant=1.4826)
    pub fn standardized_mad(primitives: P) -> Self {
        Self {
            primitives,
            p: 0.5,
            constant: 1.4826,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute QAD using the kernel's primitives for optimal performance
    pub fn compute_qad<Q: QuantileEstimator<T>>(
        &self,
        data: &mut [T],
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<T::Float> {
        if data.is_empty() {
            return Err(robust_core::Error::InvalidInput(
                "Cannot compute QAD of empty sample".to_string(),
            ));
        }

        // Use quantile estimator to find median
        let median = quantile_est
            .quantile(data, 0.5, cache)
            .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;

        // Compute absolute deviations manually
        let mut deviations: Vec<T> = data.iter()
            .map(|&x| {
                let x_float = x.to_float();
                let diff = if x_float >= median {
                    x_float - median
                } else {
                    median - x_float
                };
                // Convert back to T - this might lose precision for integer types
                T::from_f64(diff.into())
            })
            .collect();

        // Use quantile estimator to find p-th quantile of deviations
        let qad = quantile_est
            .quantile(&mut deviations, self.p, cache)
            .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;

        // Apply constant
        Ok(qad * <T::Float as NumCast>::from(self.constant).unwrap())
    }

    /// Compute QAD using sorted data
    pub fn compute_qad_sorted<Q: QuantileEstimator<T>>(
        &self,
        sorted_data: &[T],
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<T::Float> {
        if sorted_data.is_empty() {
            return Err(robust_core::Error::InvalidInput(
                "Cannot compute QAD of empty sample".to_string(),
            ));
        }

        // Use quantile estimator to find median
        let median = quantile_est
            .quantile_sorted(sorted_data, 0.5, cache)
            .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;

        // Compute absolute deviations manually
        let mut deviations: Vec<T> = sorted_data.iter()
            .map(|&x| {
                let x_float = x.to_float();
                let diff = if x_float >= median {
                    x_float - median
                } else {
                    median - x_float
                };
                // Convert back to T - this might lose precision for integer types
                T::from_f64(diff.into())
            })
            .collect();

        // Use quantile estimator to find p-th quantile of deviations
        let qad = quantile_est
            .quantile(&mut deviations, self.p, cache)
            .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;

        // Apply constant
        Ok(qad * <T::Float as NumCast>::from(self.constant).unwrap())
    }
}

// Extension methods for MadKernel backward compatibility
impl<T: Numeric, P: ComputePrimitives<T>> QadKernel<T, P> {
    /// Compute MAD (alias for compute_qad with p=0.5)
    pub fn compute_mad<Q: QuantileEstimator<T>>(
        &self,
        data: &mut [T],
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<T::Float> {
        self.compute_qad(data, quantile_est, cache)
    }

    /// Compute MAD using sorted data (alias for compute_qad_sorted)
    pub fn compute_mad_sorted<Q: QuantileEstimator<T>>(
        &self,
        sorted_data: &[T],
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<T::Float> {
        self.compute_qad_sorted(sorted_data, quantile_est, cache)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> StatisticalKernel<T> for QadKernel<T, P> {
    type Primitives = P;

    fn primitives(&self) -> &Self::Primitives {
        &self.primitives
    }

    fn name(&self) -> &'static str {
        "QAD Kernel"
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> SpreadKernel<T> for QadKernel<T, P> {
    fn compute_deviations(&self, data: &[T], center: T::Float) -> Vec<T::Float> {
        // Compute absolute deviations manually
        data.iter()
            .map(|&x| {
                let x_float = x.to_float();
                if x_float >= center {
                    x_float - center
                } else {
                    center - x_float
                }
            })
            .collect()
    }

    fn apply_transform(&self, deviations: &[T::Float]) -> T::Float {
        // For MAD, the transform is just finding the median
        // This would use a separate quantile estimator in practice
        let mut deviations = deviations.to_vec();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = deviations.len();
        if n % 2 == 0 {
            let mid1 = deviations[n / 2 - 1];
            let mid2 = deviations[n / 2];
            // Average of two middle values
            (mid1 + mid2) / <T::Float as NumCast>::from(2.0).unwrap()
        } else {
            deviations[n / 2]
        }
    }
}

/// Kernel for computing Interquartile Range (IQR)
#[derive(Clone, Debug)]
pub struct IqrKernel<T: Numeric, P: ComputePrimitives<T>> {
    primitives: P,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> IqrKernel<T, P> {
    /// Create a new IQR kernel with the given primitives
    pub fn new(primitives: P) -> Self {
        Self {
            primitives,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute IQR using the kernel's primitives
    pub fn compute_iqr<Q: QuantileEstimator<T>>(
        &self,
        data: &mut [T],
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<T::Float> {
        if data.len() < 4 {
            return Err(robust_core::Error::InvalidInput(
                "Need at least 4 observations for IQR".to_string(),
            ));
        }

        let q1 = quantile_est
            .quantile(data, 0.25, cache)
            .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
        let q3 = quantile_est
            .quantile(data, 0.75, cache)
            .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;

        Ok(q3 - q1)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> StatisticalKernel<T> for IqrKernel<T, P> {
    type Primitives = P;

    fn primitives(&self) -> &Self::Primitives {
        &self.primitives
    }

    fn name(&self) -> &'static str {
        "IQR Kernel"
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> SpreadKernel<T> for IqrKernel<T, P> {
    fn compute_deviations(&self, data: &[T], center: T::Float) -> Vec<T::Float> {
        // Compute absolute deviations manually
        data.iter()
            .map(|&x| {
                let x_float = x.to_float();
                if x_float >= center {
                    x_float - center
                } else {
                    center - x_float
                }
            })
            .collect()
    }

    fn apply_transform(&self, _deviations: &[T::Float]) -> T::Float {
        // IQR doesn't use deviations from center - it uses quantiles directly
        <T::Float as Zero>::zero() // Placeholder
    }
}

/// Kernel for computing trimmed statistics
#[derive(Clone, Debug)]
pub struct TrimmedKernel<T: Numeric, P: ComputePrimitives<T>> {
    primitives: P,
    pub(crate) trim_proportion: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> TrimmedKernel<T, P> {
    /// Create a new trimmed kernel with the given primitives and trim proportion
    pub fn new(primitives: P, trim_proportion: f64) -> Self {
        Self {
            primitives,
            trim_proportion,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute trimmed standard deviation
    pub fn compute_trimmed_std(&self, sorted_data: &[T]) -> Result<T::Float> {
        let n = sorted_data.len();
        
        if n < 3 {
            return Err(robust_core::Error::InvalidInput(
                "Need at least 3 observations for trimmed std".to_string(),
            ));
        }

        // Calculate number to trim from each tail
        let trim_count = (n as f64 * self.trim_proportion).floor() as usize;

        if n - 2 * trim_count < 2 {
            return Err(robust_core::Error::InvalidInput(
                "Too much trimming for sample size".to_string(),
            ));
        }

        // Get trimmed values
        let trimmed = &sorted_data[trim_count..n - trim_count];

        // Calculate trimmed mean using primitives
        let sum = self.primitives.sum(trimmed);
        let n_float = <T::Float as NumCast>::from(trimmed.len()).unwrap();
        let trimmed_mean = <T::Float as NumCast>::from(sum.into()).unwrap() / n_float;

        // Calculate deviations manually (sub_scalar not in generic primitives)
        let mut squared_sum = <T::Aggregate as Zero>::zero();
        for &val in trimmed {
            let val_float = val.to_float();
            let diff = val_float - trimmed_mean;
            let squared = diff * diff;
            squared_sum = squared_sum + <T::Aggregate as NumCast>::from(squared.into()).unwrap();
        }
        
        let n_minus_one = <T::Float as NumCast>::from(trimmed.len() - 1).unwrap();
        let variance = <T::Float as NumCast>::from(squared_sum.into()).unwrap() / n_minus_one;

        Ok(<T::Float as Float>::sqrt(variance))
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> StatisticalKernel<T> for TrimmedKernel<T, P> {
    type Primitives = P;

    fn primitives(&self) -> &Self::Primitives {
        &self.primitives
    }

    fn name(&self) -> &'static str {
        "Trimmed Kernel"
    }
}

/// Kernel for computing winsorized statistics
#[derive(Clone, Debug)]
pub struct WinsorizedKernel<T: Numeric, P: ComputePrimitives<T>> {
    primitives: P,
    pub(crate) winsor_proportion: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> WinsorizedKernel<T, P> {
    /// Create a new winsorized kernel with the given primitives and winsor proportion
    pub fn new(primitives: P, winsor_proportion: f64) -> Self {
        Self {
            primitives,
            winsor_proportion,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute winsorized standard deviation
    pub fn compute_winsorized_std(&self, sorted_data: &[T]) -> Result<T::Float> {
        let n = sorted_data.len();
        
        if n < 3 {
            return Err(robust_core::Error::InvalidInput(
                "Need at least 3 observations for winsorized std".to_string(),
            ));
        }

        // Calculate number to winsorize from each tail
        let winsor_count = (n as f64 * self.winsor_proportion).floor() as usize;

        if n - 2 * winsor_count < 1 {
            return Err(robust_core::Error::InvalidInput(
                "Too much winsorization for sample size".to_string(),
            ));
        }

        // Create winsorized data by replacing extreme values
        let mut winsorized = sorted_data.to_vec();
        let lower_value = sorted_data[winsor_count];
        let upper_value = sorted_data[n - winsor_count - 1];
        
        for i in 0..winsor_count {
            winsorized[i] = lower_value;
            winsorized[n - i - 1] = upper_value;
        }

        // Calculate mean using primitives
        let sum = self.primitives.sum(&winsorized);
        let n_float = <T::Float as NumCast>::from(n).unwrap();
        let mean = <T::Float as NumCast>::from(sum.into()).unwrap() / n_float;

        // Calculate variance manually
        let mut squared_sum = <T::Aggregate as Zero>::zero();
        for &val in &winsorized {
            let val_float = val.to_float();
            let diff = val_float - mean;
            let squared = diff * diff;
            squared_sum = squared_sum + <T::Aggregate as NumCast>::from(squared.into()).unwrap();
        }
        
        let n_minus_one = <T::Float as NumCast>::from(n - 1).unwrap();
        let variance = <T::Float as NumCast>::from(squared_sum.into()).unwrap() / n_minus_one;

        Ok(<T::Float as Float>::sqrt(variance))
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> StatisticalKernel<T> for WinsorizedKernel<T, P> {
    type Primitives = P;

    fn primitives(&self) -> &Self::Primitives {
        &self.primitives
    }

    fn name(&self) -> &'static str {
        "Winsorized Kernel"
    }
}

/// Kernel for computing skewness measures
#[derive(Clone, Debug)]
pub struct SkewnessKernel<T: Numeric, P: ComputePrimitives<T>> {
    primitives: P,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> SkewnessKernel<T, P> {
    /// Create a new skewness kernel with the given primitives
    pub fn new(primitives: P) -> Self {
        Self { 
            primitives,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute robust skewness using quantiles: (Q3 + Q1 - 2*Q2) / (Q3 - Q1)
    pub fn compute_quantile_skewness<Q: QuantileEstimator<T>>(
        &self,
        data: &mut [T],
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<T::Float> {
        let q1 = quantile_est
            .quantile(data, 0.25, cache)
            .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
        let q2 = quantile_est
            .quantile(data, 0.5, cache)
            .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
        let q3 = quantile_est
            .quantile(data, 0.75, cache)
            .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;

        let iqr = q3 - q1;
        let epsilon = <T::Float as NumCast>::from(f64::EPSILON).unwrap();
        if iqr.abs() < epsilon {
            return Ok(<T::Float as NumCast>::from(0.0).unwrap());
        }

        let two = <T::Float as NumCast>::from(2.0).unwrap();
        Ok((q3 + q1 - two * q2) / iqr)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> StatisticalKernel<T> for SkewnessKernel<T, P> {
    type Primitives = P;

    fn primitives(&self) -> &Self::Primitives {
        &self.primitives
    }

    fn name(&self) -> &'static str {
        "Skewness Kernel"
    }
}

/// Kernel for computing kurtosis measures
#[derive(Clone, Debug)]
pub struct KurtosisKernel<T: Numeric, P: ComputePrimitives<T>> {
    primitives: P,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Numeric, P: ComputePrimitives<T>> KurtosisKernel<T, P> {
    /// Create a new kurtosis kernel with the given primitives
    pub fn new(primitives: P) -> Self {
        Self { 
            primitives,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute robust kurtosis using Moors' method
    pub fn compute_moors_kurtosis<Q: QuantileEstimator<T>>(
        &self,
        data: &mut [T],
        quantile_est: &Q,
        cache: &Q::State,
    ) -> Result<T::Float> {
        // Moors' kurtosis: ((Q7 - Q5) + (Q3 - Q1)) / (Q6 - Q2)
        let quantiles = [
            (1.0 / 8.0, 0.0),  // Q1
            (2.0 / 8.0, 0.0),  // Q2  
            (3.0 / 8.0, 0.0),  // Q3
            (5.0 / 8.0, 0.0),  // Q5
            (6.0 / 8.0, 0.0),  // Q6
            (7.0 / 8.0, 0.0),  // Q7
        ];
        
        let mut q_values = vec![<T::Float as Zero>::zero(); 6];
        for (i, &(p, _)) in quantiles.iter().enumerate() {
            q_values[i] = quantile_est
                .quantile(data, p, cache)
                .map_err(|e| robust_core::Error::Computation(format!("Quantile error: {:?}", e)))?;
        }

        let denominator = q_values[4] - q_values[1]; // Q6 - Q2
        let epsilon = <T::Float as NumCast>::from(f64::EPSILON).unwrap();
        if denominator.abs() < epsilon {
            return Ok(<T::Float as NumCast>::from(3.0).unwrap()); // Return normal kurtosis if no spread
        }

        let numerator = (q_values[5] - q_values[3]) + (q_values[2] - q_values[0]); // (Q7 - Q5) + (Q3 - Q1)
        Ok(numerator / denominator)
    }
}

impl<T: Numeric, P: ComputePrimitives<T>> StatisticalKernel<T> for KurtosisKernel<T, P> {
    type Primitives = P;

    fn primitives(&self) -> &Self::Primitives {
        &self.primitives
    }

    fn name(&self) -> &'static str {
        "Kurtosis Kernel"
    }
}
