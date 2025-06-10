//! Stability tracking implementation

use crate::window_traits::{StabilityTracker, WindowStabilityResult};
use crate::types::{StabilityParameters, StabilityStatus};
use robust_core::Numeric;
use num_traits::FromPrimitive;

/// Tracks stability across multiple windows by counting consecutive stable windows
#[derive(Clone)]
pub struct ConsecutiveWindowTracker<T: Numeric> {
    params: StabilityParameters<T>,
    consecutive_stable: usize,
    window_count: usize,
    stability_achieved_at: Option<usize>,
    current_status: StabilityStatus<T>,
}

impl<T: Numeric> ConsecutiveWindowTracker<T> {
    pub fn new(params: StabilityParameters<T>) -> Self {
        Self {
            params,
            consecutive_stable: 0,
            window_count: 0,
            stability_achieved_at: None,
            current_status: StabilityStatus::Unknown,
        }
    }
}

impl<T: Numeric> StabilityTracker<T> for ConsecutiveWindowTracker<T>
where
    T::Float: FromPrimitive,
{
    fn update(&mut self, window_result: &WindowStabilityResult<T>) {
        self.window_count += 1;
        
        if window_result.is_stable {
            self.consecutive_stable += 1;
            
            // Check if we've achieved stability
            if self.consecutive_stable >= self.params.min_stable_windows 
                && self.stability_achieved_at.is_none() {
                self.stability_achieved_at = Some(self.window_count);
                self.current_status = StabilityStatus::Stable;
            } else if self.consecutive_stable < self.params.min_stable_windows {
                // Still transitioning
                let progress = T::Float::from_f64(self.consecutive_stable as f64).unwrap() / T::Float::from_f64(self.params.min_stable_windows as f64).unwrap();
                self.current_status = StabilityStatus::Transitioning {
                    progress,
                    estimated_samples_remaining: Some(
                        (self.params.min_stable_windows - self.consecutive_stable) * self.params.min_samples
                    ),
                };
            }
        } else {
            // Reset consecutive count
            self.consecutive_stable = 0;
            self.current_status = window_result.status.clone();
            
            // Reset stability achievement if we become unstable after being stable
            if self.stability_achieved_at.is_some() {
                self.stability_achieved_at = None;
            }
        }
    }
    
    fn current_status(&self) -> StabilityStatus<T> {
        self.current_status.clone()
    }
    
    fn reset(&mut self) {
        self.consecutive_stable = 0;
        self.window_count = 0;
        self.stability_achieved_at = None;
        self.current_status = StabilityStatus::Unknown;
    }
    
    fn stability_index(&self) -> Option<usize> {
        self.stability_achieved_at
    }
}