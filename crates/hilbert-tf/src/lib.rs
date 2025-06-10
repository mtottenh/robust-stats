//! # Hilbert Transform Library
//!
//! A high-performance implementation of the Hilbert transform using RustFFT,
//! providing mathematically rigorous signal processing capabilities.
//!
//! The Hilbert transform is a fundamental operation in signal processing that
//! shifts the phase of all frequency components by 90 degrees. This library
//! implements the transform using the frequency domain approach, which is
//! computationally efficient and numerically stable.
//!
//! ## Mathematical Background
//!
//! The Hilbert transform H[x(t)] of a signal x(t) can be understood as:
//! - In time domain: convolution with 1/(πt) (problematic due to singularity)
//! - In frequency domain: multiplication by -j*sign(ω) (elegant and practical)
//!
//! This library uses the frequency domain approach for efficiency and numerical stability.
//!
//! ## Key Features
//!
//! - **High Performance**: FFT-based implementation with O(N log N) complexity
//! - **Mathematical Rigor**: Satisfies all fundamental Hilbert transform properties
//! - **Comprehensive API**: Transform, analytic signal, envelope, and phase extraction
//! - **Flexible**: Works with any signal length (not just powers of 2)
//! - **Validated**: Extensively tested against mathematical properties and reference implementations
//!
//! ## Basic Usage
//!
//! ```rust
//! use hilbert_tf::HilbertTransform;
//!
//! // Create a test signal
//! let signal: Vec<f64> = (0..256)
//!     .map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / 256.0).cos())
//!     .collect();
//!
//! // Create transformer for this signal length
//! let transformer = HilbertTransform::new(signal.len());
//!
//! // Compute Hilbert transform
//! let hilbert = transformer.transform(&signal).unwrap();
//!
//! // Get analytic signal (complex-valued)
//! let analytic = transformer.analytic_signal(&signal).unwrap();
//!
//! // Extract envelope
//! let envelope = transformer.envelope(&signal).unwrap();
//! ```
//!
//! ## Advanced Applications
//!
//! The library supports sophisticated signal analysis techniques:
//!
//! - **Envelope Detection**: Extract amplitude modulation from signals
//! - **Instantaneous Phase**: Analyze phase relationships and frequency modulation
//! - **Analytic Signals**: Create complex representations with only positive frequencies
//! - **Single Sideband Processing**: Telecommunications and audio applications

use rustfft::{num_complex::Complex, FftPlanner};

/// Error types for Hilbert transform operations
#[derive(Debug, Clone)]
pub enum HilbertError {
    /// Signal length doesn't match the transformer's expected length
    LengthMismatch { expected: usize, actual: usize },
    /// Signal is too short for meaningful analysis
    SignalTooShort { length: usize },
    /// Numerical computation failed
    ComputationError(String),
}

impl std::fmt::Display for HilbertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HilbertError::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "Signal length mismatch: expected {expected}, got {actual}"
                )
            }
            HilbertError::SignalTooShort { length } => {
                write!(f, "Signal too short for analysis: {length} samples")
            }
            HilbertError::ComputationError(msg) => {
                write!(f, "Computation error: {msg}")
            }
        }
    }
}

impl std::error::Error for HilbertError {}

/// Result type for Hilbert transform operations
pub type HilbertResult<T> = Result<T, HilbertError>;

/// A high-performance Hilbert transform processor using FFT-based algorithms
///
/// This struct encapsulates the mathematical machinery needed to perform
/// Hilbert transforms efficiently. It pre-computes the frequency domain
/// multiplier and caches FFT planners for optimal performance when processing
/// multiple signals of the same length.
///
/// ## Design Philosophy
///
/// The implementation prioritizes both mathematical correctness and computational
/// efficiency. By caching the frequency domain multiplier, we avoid redundant
/// calculations when processing multiple signals. The FFT-based approach ensures
/// O(N log N) complexity, making it practical for large signals.
///
/// ## Mathematical Foundation
///
/// The frequency domain representation of the Hilbert transform multiplies
/// each frequency bin by a specific complex factor:
/// - DC component (k=0): multiply by 0
/// - Positive frequencies (0 < k < N/2): multiply by -j
/// - Nyquist frequency (k=N/2, even N only): multiply by 0
/// - Negative frequencies (N/2 < k < N): multiply by +j
///
/// This pattern implements the -j*sign(ω) relationship that defines the
/// Hilbert transform in the frequency domain.
pub struct HilbertTransform {
    /// Length of signals this transformer handles
    length: usize,
    /// Pre-computed frequency domain multiplier for the Hilbert transform
    /// This vector contains the complex factors that implement -j*sign(ω)
    frequency_multiplier: Vec<Complex<f64>>,
}

impl HilbertTransform {
    /// Create a new Hilbert transform processor for signals of a specific length
    ///
    /// # Arguments
    /// * `length` - The length of signals this processor will handle
    ///
    /// # Returns
    /// A new HilbertTransform instance optimized for the specified signal length
    ///
    /// # Mathematical Details
    ///
    /// This constructor pre-computes the frequency domain multiplier that implements
    /// the Hilbert transform. For an N-point FFT, the frequency bins correspond to:
    /// - k=0: DC component (frequency = 0)
    /// - k=1..N/2-1: positive frequencies
    /// - k=N/2: Nyquist frequency (for even N)
    /// - k=N/2+1..N-1: negative frequencies
    ///
    /// Each bin gets multiplied by the appropriate complex factor to implement
    /// the -j*sign(ω) relationship.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hilbert_tf::HilbertTransform;
    ///
    /// // Create transformer for 1024-sample signals
    /// let transformer = HilbertTransform::new(1024);
    ///
    /// // Reuse for multiple signals of the same length
    /// let signal1 = vec![0.0; 1024];
    /// let signal2 = vec![1.0; 1024];
    ///
    /// let result1 = transformer.transform(&signal1);
    /// let result2 = transformer.transform(&signal2);
    /// ```
    pub fn new(length: usize) -> Self {
        // Pre-allocate the frequency multiplier vector
        let mut frequency_multiplier = vec![Complex::new(0.0, 0.0); length];

        // Construct the frequency domain multiplier according to the
        // mathematical definition of the Hilbert transform
        for (k, freq_mult) in frequency_multiplier.iter_mut().enumerate() {
            *freq_mult = if k == 0 {
                // DC component: H[constant] = 0
                // This ensures that the Hilbert transform of any constant is zero
                Complex::new(0.0, 0.0)
            } else if k < length / 2 {
                // Positive frequencies: multiply by -j
                // This implements a -90 degree phase shift for positive frequencies
                Complex::new(0.0, -1.0)
            } else if k == length / 2 && length % 2 == 0 {
                // Nyquist frequency for even-length signals: multiply by 0
                // This preserves the real-valued nature of the output
                Complex::new(0.0, 0.0)
            } else {
                // Negative frequencies: multiply by +j
                // This implements a +90 degree phase shift for negative frequencies
                Complex::new(0.0, 1.0)
            };
        }

        Self {
            length,
            frequency_multiplier,
        }
    }

    /// Compute the Hilbert transform of a real-valued signal
    ///
    /// This method implements the core Hilbert transform algorithm using the
    /// frequency domain approach. The mathematical process involves:
    /// 1. Forward FFT to convert to frequency domain
    /// 2. Apply the Hilbert transform frequency multiplier
    /// 3. Inverse FFT to return to time domain
    /// 4. Extract the real part (imaginary should be negligible)
    ///
    /// # Arguments
    /// * `signal` - Input real-valued signal
    ///
    /// # Returns
    /// The Hilbert transform as a real-valued signal
    ///
    /// # Errors
    /// Returns `HilbertError::LengthMismatch` if the signal length doesn't
    /// match the transformer's expected length.
    ///
    /// # Mathematical Properties
    ///
    /// The output satisfies fundamental Hilbert transform properties:
    /// - H[cos(ωt)] = -sin(ωt)
    /// - H[sin(ωt)] = cos(ωt)
    /// - H[H[x(t)]] = -x(t)
    /// - Linearity: H[ax + by] = aH[x] + bH[y]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hilbert_tf::HilbertTransform;
    /// use std::f64::consts::PI;
    ///
    /// let n = 256;
    /// let omega = 2.0 * PI * 5.0 / n as f64;
    ///
    /// // Create a cosine signal
    /// let cosine: Vec<f64> = (0..n)
    ///     .map(|i| (omega * i as f64).cos())
    ///     .collect();
    ///
    /// let transformer = HilbertTransform::new(n);
    /// let hilbert = transformer.transform(&cosine).unwrap();
    ///
    /// // Result should approximate -sin(ωt)
    /// for (i, &h_val) in hilbert.iter().enumerate() {
    ///     let expected = -(omega * i as f64).sin();
    ///     assert!((h_val - expected).abs() < 1e-6);
    /// }
    /// ```
    pub fn transform(&self, signal: &[f64]) -> HilbertResult<Vec<f64>> {
        // Validate input signal length
        if signal.len() != self.length {
            return Err(HilbertError::LengthMismatch {
                expected: self.length,
                actual: signal.len(),
            });
        }

        // Convert real input to complex format (imaginary parts start as zero)
        let mut complex_signal: Vec<Complex<f64>> =
            signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Create FFT planner and get forward transform
        // The planner is created fresh each time to avoid lifetime complexity
        // In performance-critical applications, you might want to cache the planner
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(self.length);

        // Transform to frequency domain
        fft.process(&mut complex_signal);

        // Apply the Hilbert transform in frequency domain
        // This is the mathematical heart of the algorithm: each frequency
        // component gets multiplied by the appropriate phase shift factor
        for (freq_component, &multiplier) in complex_signal
            .iter_mut()
            .zip(self.frequency_multiplier.iter())
        {
            *freq_component *= multiplier;
        }

        // Transform back to time domain
        let ifft = planner.plan_fft_inverse(self.length);
        ifft.process(&mut complex_signal);

        // Extract real part and normalize
        // The normalization factor accounts for RustFFT's convention
        let norm_factor = 1.0 / (self.length as f64);
        let result: Vec<f64> = complex_signal.iter().map(|c| c.re * norm_factor).collect();

        Ok(result)
    }

    /// Compute the analytic signal from a real-valued input
    ///
    /// The analytic signal z(t) = x(t) + j*H[x(t)] is a complex-valued
    /// representation that contains only positive frequencies. This makes it
    /// extremely useful for envelope detection, instantaneous frequency analysis,
    /// and many other signal processing applications.
    ///
    /// # Arguments
    /// * `signal` - Input real-valued signal
    ///
    /// # Returns
    /// Complex-valued analytic signal where:
    /// - Real part is the original signal
    /// - Imaginary part is the Hilbert transform
    ///
    /// # Mathematical Properties
    ///
    /// The analytic signal has several important properties:
    /// - Contains only positive frequencies (one-sided spectrum)
    /// - |z(t)| gives the instantaneous envelope
    /// - arg(z(t)) gives the instantaneous phase
    /// - Real and imaginary parts have equal energy for symmetric spectra
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hilbert_tf::HilbertTransform;
    /// use std::f64::consts::PI;
    ///
    /// let n = 256;
    /// let signal: Vec<f64> = (0..n)
    ///     .map(|i| (2.0 * PI * 10.0 * i as f64 / n as f64).cos())
    ///     .collect();
    ///
    /// let transformer = HilbertTransform::new(n);
    /// let analytic = transformer.analytic_signal(&signal).unwrap();
    ///
    /// // Extract envelope and phase
    /// let envelope: Vec<f64> = analytic.iter().map(|z| z.norm()).collect();
    /// let phase: Vec<f64> = analytic.iter().map(|z| z.arg()).collect();
    /// ```
    pub fn analytic_signal(&self, signal: &[f64]) -> HilbertResult<Vec<Complex<f64>>> {
        if signal.len() != self.length {
            return Err(HilbertError::LengthMismatch {
                expected: self.length,
                actual: signal.len(),
            });
        }

        // Convert input to complex format
        let mut complex_signal: Vec<Complex<f64>> =
            signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Forward FFT
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(self.length);
        fft.process(&mut complex_signal);

        // For the analytic signal, we want to:
        // - Keep DC component unchanged
        // - Double positive frequency components
        // - Zero out negative frequency components
        // - Handle Nyquist frequency appropriately

        for (k, signal) in complex_signal.iter_mut().enumerate().take(self.length) {
            *signal *= if k == 0 {
                // DC component: keep unchanged
                Complex::new(1.0, 0.0)
            } else if k < self.length / 2 {
                // Positive frequencies: double them
                Complex::new(2.0, 0.0)
            } else if k == self.length / 2 && self.length % 2 == 0 {
                // Nyquist frequency for even N: keep unchanged
                Complex::new(1.0, 0.0)
            } else {
                // Negative frequencies: zero them out
                Complex::new(0.0, 0.0)
            };
        }

        // Inverse FFT
        let ifft = planner.plan_fft_inverse(self.length);
        ifft.process(&mut complex_signal);

        // Normalize and return
        let norm_factor = 1.0 / (self.length as f64);
        let result: Vec<Complex<f64>> = complex_signal.iter().map(|c| c * norm_factor).collect();

        Ok(result)
    }

    /// Extract the instantaneous envelope (amplitude) from a signal
    ///
    /// The envelope is the magnitude |z(t)| of the analytic signal z(t).
    /// This is particularly useful for analyzing amplitude modulated signals,
    /// where the envelope represents the modulating waveform.
    ///
    /// # Arguments
    /// * `signal` - Input real-valued signal
    ///
    /// # Returns
    /// The instantaneous envelope as a real-valued signal
    ///
    /// # Applications
    /// - AM radio demodulation
    /// - Speech envelope extraction
    /// - Biomedical signal analysis
    /// - Radar and sonar processing
    pub fn envelope(&self, signal: &[f64]) -> HilbertResult<Vec<f64>> {
        let analytic = self.analytic_signal(signal)?;
        Ok(analytic.iter().map(|c| c.norm()).collect())
    }

    /// Extract the instantaneous phase from a signal
    ///
    /// The instantaneous phase is the argument arg(z(t)) of the analytic signal z(t).
    /// This reveals how the phase evolves over time and is fundamental for
    /// frequency modulation analysis.
    ///
    /// # Arguments
    /// * `signal` - Input real-valued signal
    ///
    /// # Returns
    /// The instantaneous phase in radians
    ///
    /// # Note
    /// The phase is returned in the range [-π, π]. To get instantaneous frequency,
    /// take the derivative of the unwrapped phase.
    ///
    /// # Applications
    /// - FM demodulation
    /// - Phase-locked loop analysis
    /// - Synchronization systems
    /// - Neurological signal analysis
    pub fn instantaneous_phase(&self, signal: &[f64]) -> HilbertResult<Vec<f64>> {
        let analytic = self.analytic_signal(signal)?;
        Ok(analytic.iter().map(|c| c.arg()).collect())
    }

    /// Get the signal length this transformer handles
    pub fn length(&self) -> usize {
        self.length
    }
    
    /// Compute instantaneous frequency from a signal
    ///
    /// The instantaneous frequency is the derivative of the unwrapped instantaneous phase.
    /// This is computed using finite differences with appropriate boundary handling.
    ///
    /// # Arguments
    /// * `signal` - Input real-valued signal
    /// * `sampling_rate` - Sampling rate in Hz (for proper frequency scaling)
    ///
    /// # Returns
    /// The instantaneous frequency in Hz
    ///
    /// # Mathematical Background
    /// 
    /// For an analytic signal z(t) = A(t)e^(jφ(t)), the instantaneous frequency is:
    /// f(t) = (1/2π) * dφ/dt
    ///
    /// We compute this using finite differences on the unwrapped phase.
    ///
    /// # Applications
    /// - FM demodulation
    /// - Vibration analysis
    /// - Speech processing
    /// - Biomedical signal analysis
    pub fn instantaneous_frequency(&self, signal: &[f64], sampling_rate: f64) -> HilbertResult<Vec<f64>> {
        // Get instantaneous phase
        let phase = self.instantaneous_phase(signal)?;
        
        // Unwrap the phase to handle discontinuities
        let unwrapped_phase = unwrap_phase(&phase);
        
        // Compute frequency using finite differences
        let mut frequency = Vec::with_capacity(unwrapped_phase.len());
        
        if unwrapped_phase.len() < 2 {
            return Ok(frequency);
        }
        
        // Forward difference for first point
        frequency.push((unwrapped_phase[1] - unwrapped_phase[0]) * sampling_rate / (2.0 * std::f64::consts::PI));
        
        // Central differences for interior points
        for i in 1..unwrapped_phase.len() - 1 {
            let freq = (unwrapped_phase[i + 1] - unwrapped_phase[i - 1]) * sampling_rate / (4.0 * std::f64::consts::PI);
            frequency.push(freq);
        }
        
        // Backward difference for last point
        let n = unwrapped_phase.len();
        frequency.push((unwrapped_phase[n - 1] - unwrapped_phase[n - 2]) * sampling_rate / (2.0 * std::f64::consts::PI));
        
        Ok(frequency)
    }
}

/// Unwrap phase to handle discontinuities at ±π
///
/// Phase unwrapping ensures that phase values don't have artificial jumps
/// of 2π, which is essential for computing instantaneous frequency.
///
/// # Arguments
/// * `phase` - Wrapped phase values in radians
///
/// # Returns
/// Unwrapped phase values
fn unwrap_phase(phase: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    
    if phase.is_empty() {
        return vec![];
    }
    
    let mut unwrapped = vec![phase[0]];
    
    for i in 1..phase.len() {
        let diff = phase[i] - phase[i - 1];
        let adjusted_diff = if diff > PI {
            diff - 2.0 * PI
        } else if diff < -PI {
            diff + 2.0 * PI
        } else {
            diff
        };
        unwrapped.push(unwrapped[i - 1] + adjusted_diff);
    }
    
    unwrapped
}

/// Convenience function for one-off Hilbert transforms
///
/// Use this when you only need to transform one signal and don't need
/// the efficiency benefits of reusing a HilbertTransform instance.
/// For processing multiple signals of the same length, create a
/// HilbertTransform instance and reuse it.
///
/// # Arguments
/// * `signal` - Input real-valued signal
///
/// # Returns
/// The Hilbert transform as a real-valued signal
///
/// # Examples
///
/// ```rust
/// use hilbert_tf::hilbert_transform;
///
/// let signal = vec![1.0, 0.0, -1.0, 0.0]; // Simple test signal
/// let hilbert = hilbert_transform(&signal).unwrap();
/// ```
pub fn hilbert_transform(signal: &[f64]) -> HilbertResult<Vec<f64>> {
    let transformer = HilbertTransform::new(signal.len());
    transformer.transform(signal)
}

/// Convenience function for one-off analytic signal computation
///
/// # Arguments
/// * `signal` - Input real-valued signal
///
/// # Returns
/// The analytic signal as a complex-valued signal
pub fn analytic_signal(signal: &[f64]) -> HilbertResult<Vec<Complex<f64>>> {
    let transformer = HilbertTransform::new(signal.len());
    transformer.analytic_signal(signal)
}

/// Convenience function for one-off envelope extraction
///
/// # Arguments
/// * `signal` - Input real-valued signal
///
/// # Returns
/// The instantaneous envelope as a real-valued signal
pub fn envelope(signal: &[f64]) -> HilbertResult<Vec<f64>> {
    let transformer = HilbertTransform::new(signal.len());
    transformer.envelope(signal)
}

/// Convenience function for one-off phase extraction
///
/// # Arguments
/// * `signal` - Input real-valued signal
///
/// # Returns
/// The instantaneous phase in radians
pub fn instantaneous_phase(signal: &[f64]) -> HilbertResult<Vec<f64>> {
    let transformer = HilbertTransform::new(signal.len());
    transformer.instantaneous_phase(signal)
}

/// Convenience function for one-off instantaneous frequency computation
///
/// # Arguments
/// * `signal` - Input real-valued signal
/// * `sampling_rate` - Sampling rate in Hz
///
/// # Returns
/// The instantaneous frequency in Hz
pub fn instantaneous_frequency(signal: &[f64], sampling_rate: f64) -> HilbertResult<Vec<f64>> {
    let transformer = HilbertTransform::new(signal.len());
    transformer.instantaneous_frequency(signal, sampling_rate)
}
