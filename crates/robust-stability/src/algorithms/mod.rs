//! Algorithm implementations for stability analysis

mod statistical_window;
mod stability_tracker;

pub use statistical_window::{StatisticalWindowAnalyzer, DefaultStatisticalAnalyzer};
pub use stability_tracker::ConsecutiveWindowTracker;