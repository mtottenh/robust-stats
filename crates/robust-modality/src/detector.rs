use crate::traits::ModalityDetector;
use crate::types::{ModalityResult, Mode};
use crate::visualization::ModalityVisualizer;
use robust_core::{Result, Numeric};
use robust_core::pipeline::events::{EventBus, PipelineEvent, ModeInfo};
use robust_core::pipeline::context::PipelineContext;
use robust_histogram::{Histogram, HistogramBuilder};
use std::collections::HashMap;
use uuid::Uuid;
use std::fmt;
use num_traits::{Zero, NumCast};


/// The Lowland modality detector implementation.
///
/// This detector identifies modes by finding "lowlands" - regions of low density
/// that separate higher density regions (modes). The algorithm works by:
/// 1. Building a histogram of the data
/// 2. Identifying potential modes as local maxima in density
/// 3. Validating modes by checking for sufficient "lowlands" (valleys) between them
///
/// This implementation follows the algorithm from:
/// https://aakinshin.net/posts/lowland-multimodality-detection/
pub struct LowlandModalityDetector<T: Numeric = f64, H = (), V = ()> 
where
    T::Float: fmt::Debug + PartialEq + num_traits::ToPrimitive,
{
    histogram_builder: H,
    visualizer: V,
    sensitivity: f64,
    precision: f64,
    /// Optional event bus for pipeline integration
    event_bus: Option<EventBus>,
    /// Optional pipeline context
    context: Option<PipelineContext>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, H, V> LowlandModalityDetector<T, H, V> 
where
    T: Numeric,
    T::Float: fmt::Debug + PartialEq + num_traits::ToPrimitive,
{
    /// Creates a new Lowland modality detector.
    ///
    /// # Arguments
    /// * `histogram_builder` - The histogram builder to use for density estimation
    /// * `visualizer` - The visualizer for optional plotting
    /// * `sensitivity` - Controls how sensitive the detector is to lowlands (0.0-1.0)
    ///                   Lower values make it easier to split modes at shallow valleys
    /// * `precision` - Controls the resolution of the analysis (0.0-1.0).
    ///                  The number of histogram bins is calculated as round(1/precision)
    pub fn new(histogram_builder: H, visualizer: V, sensitivity: f64, precision: f64) -> Self {
        Self {
            histogram_builder,
            visualizer,
            sensitivity,
            precision,
            event_bus: None,
            context: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Sets the sensitivity parameter.
    pub fn with_sensitivity(mut self, sensitivity: f64) -> Self {
        self.sensitivity = sensitivity.clamp(0.0, 1.0);
        self
    }

    /// Sets the precision parameter.
    pub fn with_precision(mut self, precision: f64) -> Self {
        self.precision = precision.clamp(0.001, 1.0);
        self
    }

    /// Get a reference to the visualizer
    pub fn visualizer(&self) -> &V {
        &self.visualizer
    }
    
    /// Set the event bus for pipeline integration
    pub fn with_event_bus(mut self, event_bus: EventBus) -> Self {
        self.event_bus = Some(event_bus);
        self
    }
    
    /// Set the pipeline context
    pub fn with_context(mut self, context: PipelineContext) -> Self {
        self.context = Some(context);
        self
    }
    
    /// Emit an event if event bus is configured
    fn emit_event(&self, event: PipelineEvent) {
        if let (Some(bus), Some(ctx)) = (&self.event_bus, &self.context) {
            let _ = bus.emit(event, ctx);
        }
    }
}

impl<T, H, V> ModalityDetector<T> for LowlandModalityDetector<T, H, V>
where
    T: Numeric,
    T::Float: fmt::Debug + PartialEq + num_traits::ToPrimitive,
    H: HistogramBuilder<T>,
    V: ModalityVisualizer<T>,
{
}

// Add a trait for builders that can work with quantile estimators
pub trait QuantileAwareBuilder<T: Numeric, Q> {
    fn build_with_estimator(&self, sample: &[T], estimator: &Q, cache: &Q::State) -> Result<Histogram<T>>
    where
        Q: robust_core::BatchQuantileEstimator<T>;
}

// Helper trait to identify types that have a build method with estimator
pub trait HasQuantileBuild<T: Numeric, Q> {
    fn build(&self, sample: &[T], estimator: &Q, cache: &Q::State) -> Result<Histogram<T>>
    where
        Q: robust_core::BatchQuantileEstimator<T>;
}

// Implement for our quantile builders
impl<T, Q> HasQuantileBuild<T, Q> for robust_histogram::QRDEBuilderWithSteps
where
    T: Numeric,
{
    fn build(&self, sample: &[T], estimator: &Q, cache: &Q::State) -> Result<Histogram<T>>
    where
        Q: robust_core::BatchQuantileEstimator<T>,
    {
        self.build(sample, estimator, cache)
    }
}

impl<T, Q> HasQuantileBuild<T, Q> for robust_histogram::QuantileBuilder
where
    T: Numeric,
{
    fn build(&self, sample: &[T], estimator: &Q, cache: &Q::State) -> Result<Histogram<T>>
    where
        Q: robust_core::BatchQuantileEstimator<T>,
    {
        self.build(sample, estimator, cache)
    }
}

// Blanket implementation for any type that has a build method with estimator signature
impl<T, Q, H> QuantileAwareBuilder<T, Q> for H
where
    T: Numeric,
    H: HasQuantileBuild<T, Q>,
{
    fn build_with_estimator(&self, sample: &[T], estimator: &Q, cache: &Q::State) -> Result<Histogram<T>>
    where
        Q: robust_core::BatchQuantileEstimator<T>,
    {
        self.build(sample, estimator, cache)
    }
}

impl<T, H, V> LowlandModalityDetector<T, H, V>
where
    T: Numeric,
    T::Float: fmt::Debug + PartialEq + fmt::Display + num_traits::ToPrimitive,
    V: ModalityVisualizer<T>,
{
    /// Detect modes using the provided quantile estimator
    /// 
    /// This is the primary method that enables batch optimization.
    pub fn detect_modes_with_estimator<Q>(
        &self, 
        sample: &[T], 
        estimator: &Q,
        cache: &Q::State
    ) -> Result<ModalityResult<T>>
    where
        Q: robust_core::BatchQuantileEstimator<T>,
        H: QuantileAwareBuilder<T, Q>,
    {
        if sample.len() < 2 {
            // Create empty histogram for minimal data
            use num_traits::Zero;
            let zero = T::Float::zero();
            let empty_hist = Histogram::new(vec![], 0, zero, zero);
            return Ok(ModalityResult::new(vec![], empty_hist, vec![]));
        }

        // Check if all values are the same (within epsilon)
        use num_traits::NumCast;
        let first = sample[0].to_float();
        let (min_val, max_val) = sample.iter().skip(1).fold((first, first), |(min, max), &val| {
            let v = val.to_float();
            (if v < min { v } else { min }, if v > max { v } else { max })
        });

        let epsilon: T::Float = NumCast::from(1e-9).unwrap();
        if (max_val - min_val) < epsilon {
            return Err(robust_core::Error::InvalidInput(
                "Sample should contain at least two different elements".to_string(),
            ));
        }

        // Emit modality detection started event
        let trace_id = self.context.as_ref().map(|c| c.trace_id).unwrap_or_else(Uuid::new_v4);
        self.emit_event(PipelineEvent::ModalityDetectionStarted {
            trace_id,
            data_len: sample.len(),
            parameters: {
                let mut params = HashMap::new();
                params.insert("sensitivity".to_string(), self.sensitivity);
                params.insert("precision".to_string(), self.precision);
                params
            },
        });

        // Build histogram with the provided estimator using the quantile-aware interface
        let histogram = self.histogram_builder.build_with_estimator(sample, estimator, cache)?;

        // Record initial histogram for visualization
        let _ = self.visualizer.record_histogram(&histogram);
        
        // Emit histogram created event
        use num_traits::ToPrimitive;
        self.emit_event(PipelineEvent::ModalityHistogramCreated {
            trace_id,
            bin_count: histogram.bins().len(),
            min_value: histogram.min().to_f64().unwrap_or(0.0),
            max_value: histogram.max().to_f64().unwrap_or(0.0),
        });

        // Detect modes using the lowland algorithm
        self.detect_modes_algorithm(&histogram, sample)
    }
    /// Implements the lowland algorithm for mode detection.
    fn detect_modes_algorithm(
        &self,
        histogram: &Histogram<T>,
        _sample: &[T],
    ) -> Result<ModalityResult<T>> {
        let bins = histogram.bins();
        let bin_count = bins.len();
        let bin_area = 1.0 / bin_count as f64;
        let bin_heights: Vec<T::Float> = bins.iter().map(|bin| bin.density).collect();

        // Find peaks (local maxima)
        let mut peaks = Vec::new();
        for i in 1..bin_count - 1 {
            if bin_heights[i] > bin_heights[i - 1] && bin_heights[i] >= bin_heights[i + 1] {
                peaks.push(i);
            }
        }

        // Record peaks for visualization
        let _ = self.visualizer.record_peaks(histogram, &peaks);
        
        // Emit peaks detected event
        let trace_id = self.context.as_ref().map(|c| c.trace_id).unwrap_or_else(Uuid::new_v4);
        use num_traits::cast::ToPrimitive;
        let peak_heights: Vec<f64> = peaks.iter().map(|&idx| bin_heights[idx].to_f64().unwrap_or(0.0)).collect();
        self.emit_event(PipelineEvent::ModalityPeaksDetected {
            trace_id,
            peak_indices: peaks.clone(),
            peak_heights,
        });

        // Helper function to create a global mode
        let global_mode = |location: T::Float| -> Mode<T> {
            Mode {
                location,
                left_bound: histogram.min(),
                right_bound: histogram.max(),
                height: bin_heights[peaks.first().copied().unwrap_or(0)],
            }
        };

        // Helper function to create a local mode
        let local_mode = |location: T::Float, left: T::Float, right: T::Float| -> Mode<T> {
            // Find the height at the location
            let height = self.interpolate_height_at_location(&bins, location);
            Mode {
                location,
                left_bound: left,
                right_bound: right,
                height,
            }
        };

        match peaks.len() {
            0 => {
                // No peaks found, use bin with maximum height
                let max_idx = bin_heights
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let two = NumCast::from(2.0).unwrap();
                let location = (bins[max_idx].left + bins[max_idx].right) / two;
                let modes = vec![global_mode(location)];
                let _ = self.visualizer.record_final_modes(histogram, &modes, &[]);
                
                // Emit modes detected event
                self.emit_event(PipelineEvent::ModesDetected {
                    trace_id,
                    modes: modes.iter().map(|m| ModeInfo {
                        location: m.location.to_f64().unwrap_or(0.0),
                        height: m.height.to_f64().unwrap_or(0.0),
                        left_bound: m.left_bound.to_f64().unwrap_or(0.0),
                        right_bound: m.right_bound.to_f64().unwrap_or(0.0),
                    }).collect(),
                    is_multimodal: false,
                });
                
                Ok(ModalityResult::new(modes, histogram.clone(), vec![]))
            }
            1 => {
                // Single peak
                let peak_idx = peaks[0];
                let two = NumCast::from(2.0).unwrap();
                let location = (bins[peak_idx].left + bins[peak_idx].right) / two;
                let modes = vec![global_mode(location)];
                let _ = self.visualizer.record_final_modes(histogram, &modes, &[]);
                
                // Emit modes detected event
                self.emit_event(PipelineEvent::ModesDetected {
                    trace_id,
                    modes: modes.iter().map(|m| ModeInfo {
                        location: m.location.to_f64().unwrap_or(0.0),
                        height: m.height.to_f64().unwrap_or(0.0),
                        left_bound: m.left_bound.to_f64().unwrap_or(0.0),
                        right_bound: m.right_bound.to_f64().unwrap_or(0.0),
                    }).collect(),
                    is_multimodal: false,
                });
                
                Ok(ModalityResult::new(modes, histogram.clone(), vec![]))
            }
            _ => {
                // Multiple peaks - apply the reference algorithm
                let mut mode_locations = Vec::new();
                let mut cut_points = Vec::new();
                let mut lowland_indices = Vec::new();

                let mut previous_peaks = vec![peaks[0]];

                for i in 1..peaks.len() {
                    let current_peak = peaks[i];

                    // Process peaks according to lowland algorithm
                    while !previous_peaks.is_empty()
                        && bin_heights[*previous_peaks.last().unwrap()] < bin_heights[current_peak]
                    {
                        if let Some((new_mode, cut_point, lowland_bins)) = self.try_split(
                            histogram,
                            &bins,
                            &bin_heights,
                            bin_area,
                            previous_peaks.first().copied().unwrap(),
                            *previous_peaks.last().unwrap(),
                            current_peak,
                        )? {
                            mode_locations.push(new_mode);
                            cut_points.push(cut_point);
                            lowland_indices.extend(lowland_bins);
                            previous_peaks.clear();
                        } else {
                            previous_peaks.pop();
                        }
                    }

                    if !previous_peaks.is_empty()
                        && bin_heights[*previous_peaks.last().unwrap()] > bin_heights[current_peak]
                    {
                        if let Some((new_mode, cut_point, lowland_bins)) = self.try_split(
                            histogram,
                            &bins,
                            &bin_heights,
                            bin_area,
                            previous_peaks.first().copied().unwrap(),
                            *previous_peaks.last().unwrap(),
                            current_peak,
                        )? {
                            mode_locations.push(new_mode);
                            cut_points.push(cut_point);
                            lowland_indices.extend(lowland_bins);
                            previous_peaks.clear();
                        }
                    }

                    previous_peaks.push(current_peak);
                }

                // Add the final mode
                let final_peak = previous_peaks.first().copied().unwrap();
                let two = NumCast::from(2.0).unwrap();
                let final_location = (bins[final_peak].left + bins[final_peak].right) / two;
                mode_locations.push(final_location);

                // Create modes based on the number of mode locations
                let modes = match mode_locations.len() {
                    0 => {
                        let max_idx = bin_heights
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        let two = NumCast::from(2.0).unwrap();
                let location = (bins[max_idx].left + bins[max_idx].right) / two;
                        vec![global_mode(location)]
                    }
                    1 => {
                        vec![global_mode(mode_locations[0])]
                    }
                    _ => {
                        let mut modes = Vec::new();

                        // First mode: from global lower to first cut point
                        modes.push(local_mode(
                            mode_locations[0],
                            histogram.min(),
                            cut_points[0],
                        ));

                        // Middle modes: between cut points
                        for i in 1..mode_locations.len() - 1 {
                            modes.push(local_mode(
                                mode_locations[i],
                                cut_points[i - 1],
                                cut_points[i],
                            ));
                        }

                        // Last mode: from last cut point to global upper
                        modes.push(local_mode(
                            *mode_locations.last().unwrap(),
                            *cut_points.last().unwrap(),
                            histogram.max(),
                        ));

                        modes
                    }
                };

                // Record final modes for visualization
                let _ = self
                    .visualizer
                    .record_final_modes(histogram, &modes, &lowland_indices);
                
                // Emit modes detected event
                self.emit_event(PipelineEvent::ModesDetected {
                    trace_id,
                    modes: modes.iter().map(|m| ModeInfo {
                        location: m.location.to_f64().unwrap_or(0.0),
                        height: m.height.to_f64().unwrap_or(0.0),
                        left_bound: m.left_bound.to_f64().unwrap_or(0.0),
                        right_bound: m.right_bound.to_f64().unwrap_or(0.0),
                    }).collect(),
                    is_multimodal: modes.len() > 1,
                });

                Ok(ModalityResult::new(
                    modes,
                    histogram.clone(),
                    lowland_indices,
                ))
            }
        }
    }

    /// Attempts to split between peaks using the lowland algorithm.
    /// Returns (mode_location, cut_point, lowland_indices) if successful.
    fn try_split(
        &self,
        histogram: &Histogram<T>,
        bins: &[robust_histogram::HistogramBin<T>],
        bin_heights: &[T::Float],
        bin_area: f64,
        peak0: usize,
        peak1: usize,
        peak2: usize,
    ) -> Result<Option<(T::Float, T::Float, Vec<usize>)>> {
        let mut left = peak1;
        let mut right = peak2;
        let height = if bin_heights[peak1] < bin_heights[peak2] {
            bin_heights[peak1]
        } else {
            bin_heights[peak2]
        };

        // Find the lowland region
        while left < right && bin_heights[left] > height {
            left += 1;
        }
        while left < right && bin_heights[right] > height {
            right -= 1;
        }

        use num_traits::{NumCast, ToPrimitive};
        let width = bins[right].right - bins[left].left;
        let total_area_f64 = width.to_f64().unwrap_or(0.0) * height.to_f64().unwrap_or(0.0);
        let total_bin_count = right - left + 1;
        let total_bin_area = total_bin_count as f64 * bin_area;
        let bin_proportion = if total_area_f64 > 0.0 {
            total_bin_area / total_area_f64
        } else {
            0.0
        };

        let is_lowland = bin_proportion < self.sensitivity;

        // Record water level attempt for visualization
        let _ = self
            .visualizer
            .record_water_level_attempt(histogram, peak1, peak2, height, left, right, is_lowland);
            
        // Emit water level test event
        let trace_id = self.context.as_ref().map(|c| c.trace_id).unwrap_or_else(Uuid::new_v4);
        self.emit_event(PipelineEvent::ModalityWaterLevelTest {
            trace_id,
            peak1_idx: peak1,
            peak2_idx: peak2,
            water_level: height.to_f64().unwrap_or(0.0),
            is_lowland,
            underwater_bins: (left, right),
        });

        if is_lowland {
            // Split is valid
            let two = NumCast::from(2.0).unwrap();
            let mode_location = (bins[peak0].left + bins[peak0].right) / two;

            // Find the minimum between peak1 and peak2 for the cut point
            let min_idx = (peak1..=peak2)
                .min_by(|&a, &b| bin_heights[a].partial_cmp(&bin_heights[b]).unwrap())
                .unwrap_or(peak1);
            let cut_point = (bins[min_idx].left + bins[min_idx].right) / two;

            // Collect lowland indices
            let lowland_indices: Vec<usize> = (left..=right).collect();

            Ok(Some((mode_location, cut_point, lowland_indices)))
        } else {
            Ok(None)
        }
    }

    /// Interpolates the height at a specific location in the histogram.
    fn interpolate_height_at_location(
        &self,
        bins: &[robust_histogram::HistogramBin<T>],
        location: T::Float,
    ) -> T::Float {
        for bin in bins {
            if location >= bin.left && location <= bin.right {
                return bin.density;
            }
        }
        T::Float::zero() // Default if location is outside histogram bounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_data::{TestDistributions, TestParameters};
    use crate::visualization::NullModalityVisualizer;
    use robust_histogram::QuantileBuilder;
    use robust_quantile::{estimators::harrell_davis, QuantileAdapter, HDWeightComputer};
    use robust_core::{execution::scalar_sequential, UnifiedWeightCache, CachePolicy};

    fn create_detector_with_params<T: Numeric>(sensitivity: f64, precision: f64) -> LowlandModalityDetector<T, QuantileBuilder, NullModalityVisualizer> 
    where
        T::Float: fmt::Debug + PartialEq + fmt::Display + num_traits::ToPrimitive,
    {
        let histogram_builder = QuantileBuilder::new((1.0 / precision) as usize);
        LowlandModalityDetector::new(histogram_builder, NullModalityVisualizer::default(), sensitivity, precision)
    }

    #[test]
    fn test_unimodal_detection() {
        // Use standardized test data and parameters
        let data = TestDistributions::unimodal_normal();
        let (sensitivity, precision) = TestParameters::DEFAULT;
        let detector = create_detector_with_params::<f64>(sensitivity, precision);
        
        // Create estimator and cache
        let engine = scalar_sequential();
        let hd = harrell_davis(engine);
        let estimator = QuantileAdapter::new(hd);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);

        let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();
        
        // Should detect as unimodal
        assert!(!result.is_multimodal(), "Normal distribution should be unimodal");
        assert_eq!(result.mode_count(), 1, "Should detect exactly one mode");
    }

    #[test]
    fn test_bimodal_detection() {
        // Use standardized bimodal test data
        let data = TestDistributions::bimodal_symmetric();
        let (sensitivity, precision) = TestParameters::DEFAULT;
        let detector = create_detector_with_params::<f64>(sensitivity, precision);
        
        // Create estimator and cache
        let engine = scalar_sequential();
        let hd = harrell_davis(engine);
        let estimator = QuantileAdapter::new(hd);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);

        let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();

        // Should detect as bimodal
        assert!(result.is_multimodal(), "Should detect multimodality");
        assert_eq!(result.mode_count(), 2, "Should detect exactly two modes");
    }

    #[test]
    fn test_insufficient_data() {
        let (sensitivity, precision) = TestParameters::DEFAULT;
        let detector = create_detector_with_params::<f64>(sensitivity, precision);
        
        // Create estimator and cache
        let engine = scalar_sequential();
        let hd = harrell_davis(engine);
        let estimator = QuantileAdapter::new(hd);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);
        
        let data = vec![1.0];

        let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();
        assert!(!result.is_multimodal());
        assert!(result.modes().is_empty());
    }

    #[test]
    fn test_generic_f32() {
        // Test with f32 data
        // Note: harrell_davis currently creates f64 estimators, so we test the type conversion
        // Create more data points to ensure mode detection works
        let mut data_f32: Vec<f32> = Vec::new();
        // Add points around 2.0
        for _ in 0..20 {
            data_f32.push(1.0);
            data_f32.push(2.0);
            data_f32.push(3.0);
        }
        // Add points around 9.0  
        for _ in 0..20 {
            data_f32.push(8.0);
            data_f32.push(9.0);
            data_f32.push(10.0);
        }
        
        let data: Vec<f64> = data_f32.iter().map(|&x| x as f64).collect();
        let (sensitivity, precision) = TestParameters::SENSITIVE;
        let detector = create_detector_with_params::<f64>(sensitivity, precision);
        
        // Create estimator and cache
        let engine = scalar_sequential();
        let hd = harrell_davis(engine);
        let estimator = QuantileAdapter::new(hd);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);

        let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();
        
        // Should detect multimodality (may detect more than 2 modes due to discrete data)
        assert!(result.is_multimodal(), "Should detect multimodality in bimodal data");
        assert!(result.mode_count() >= 2, "Should detect at least two modes in bimodal data (f32 converted to f64)");
    }

    #[test]
    fn test_generic_i32() {
        // Test with integer data
        // Note: We convert to f64 for now since harrell_davis creates f64 estimators
        let data_i32: Vec<i32> = vec![1, 1, 2, 2, 3, 8, 8, 9, 9, 10];
        let data: Vec<f64> = data_i32.iter().map(|&x| x as f64).collect();
        let (sensitivity, precision) = TestParameters::SENSITIVE;
        let detector = create_detector_with_params::<f64>(sensitivity, precision);
        
        // Create estimator and cache
        let engine = scalar_sequential();
        let hd = harrell_davis(engine);
        let estimator = QuantileAdapter::new(hd);
        let cache = UnifiedWeightCache::new(HDWeightComputer::new(), CachePolicy::NoCache);

        let result = detector.detect_modes_with_estimator(&data, &estimator, &cache).unwrap();
        
        // Integer data should still be detectable
        assert!(result.mode_count() > 0, "Should detect modes in integer data (converted to f64)");
    }
}
