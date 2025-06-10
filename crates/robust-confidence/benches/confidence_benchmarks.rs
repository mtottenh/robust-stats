use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use rand_distr::Normal;
use robust_confidence::{
    MaritzJarrettCI, AsymptoticCI,
    PercentileBootstrap, BCaBootstrap,
    ConfidenceIntervalEstimator, MeanWithSE,
    Bootstrap,
};
use robust_core::{
    execution::{simd_sequential, auto_budgeted_engine},
    UnifiedWeightCache, CachePolicy,
};

// Import quantile estimators only when the feature is enabled
#[cfg(feature = "quantile")]
use robust_quantile::{
    estimators::harrell_davis, QuantileAdapter, HDWeightComputer,
};

// Import spread features when available
#[cfg(all(feature = "quantile", feature = "spread"))]
use robust_confidence::mad_factory;

/// Generate normal data
fn generate_normal_data(size: usize, mean: f64, std: f64, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(mean, std).unwrap();
    (0..size).map(|_| normal.sample(&mut rng)).collect()
}

#[cfg(feature = "quantile")]
fn bench_maritz_jarrett(c: &mut Criterion) {
    let mut group = c.benchmark_group("MaritzJarrett");
    let sizes = [50, 100, 500, 1000];
    let data_cache: Vec<_> = sizes.iter()
        .map(|&size| generate_normal_data(size, 100.0, 15.0, 42))
        .collect();
    
    // Create HD estimator and cache once
    let engine = simd_sequential();
    let hd = harrell_davis(engine);
    let estimator = QuantileAdapter::new(hd);
    let cache = UnifiedWeightCache::new(HDWeightComputer, CachePolicy::Lru { max_entries: 1024 });
    
    for (i, &size) in sizes.iter().enumerate() {
        let data = &data_cache[i];
        
        // Maritz-Jarrett CI for median
        let mj_ci = MaritzJarrettCI::new(0.5, 0.95);
        
        group.bench_with_input(
            BenchmarkId::new("median", size),
            data,
            |b, data| b.iter(|| mj_ci.confidence_interval(black_box(data), &estimator, &cache))
        );
        
        // Maritz-Jarrett CI for quartiles
        let mj_q1 = MaritzJarrettCI::new(0.25, 0.95);
        
        group.bench_with_input(
            BenchmarkId::new("q1", size),
            data,
            |b, data| b.iter(|| mj_q1.confidence_interval(black_box(data), &estimator, &cache))
        );
    }
    
    group.finish();
}

#[cfg(all(feature = "quantile", feature = "spread"))]
fn bench_bootstrap(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bootstrap");
    let n_resamples = [100, 500, 1000];
    
    let data1 = generate_normal_data(100, 100.0, 15.0, 42);
    let data2 = generate_normal_data(100, 105.0, 15.0, 43);
    
    // Use simple shift comparison - MAD will be the estimator, not the comparison
    use robust_core::ShiftComparison;
    let comparison = ShiftComparison;
    
    // Use MAD factory
    let factory = mad_factory();
    
    for &n_resample in &n_resamples {
        // Using the low-level Bootstrap API with Percentile method
        let engine = auto_budgeted_engine();
        let bootstrap = Bootstrap::new(engine.clone(), PercentileBootstrap)
            .with_resamples(n_resample)
            .with_confidence_level(0.95)
            .with_seed(42);
        
        group.bench_with_input(
            BenchmarkId::new("percentile", n_resample),
            &(&data1, &data2),
            |b, (data1, data2)| {
                b.iter(|| {
                    bootstrap.confidence_intervals(
                        black_box(*data1),
                        black_box(*data2),
                        &comparison,
                        &factory,
                    )
                })
            }
        );
        
        // BCa bootstrap
        let bootstrap_bca = Bootstrap::new(engine, BCaBootstrap)
            .with_resamples(n_resample)
            .with_confidence_level(0.95)
            .with_seed(42);
        
        group.bench_with_input(
            BenchmarkId::new("bca", n_resample),
            &(&data1, &data2),
            |b, (data1, data2)| {
                b.iter(|| {
                    bootstrap_bca.confidence_intervals(
                        black_box(*data1),
                        black_box(*data2),
                        &comparison,
                        &factory,
                    )
                })
            }
        );
    }
    
    group.finish();
}

fn bench_asymptotic(c: &mut Criterion) {
    let mut group = c.benchmark_group("Asymptotic");
    let sizes = [50, 100, 500, 1000, 5000];
    
    for &size in &sizes {
        let data = generate_normal_data(size, 100.0, 15.0, 42);
        
        // T-distribution based CI
        let t_ci = AsymptoticCI::students_t(0.95);
        let mean_se = MeanWithSE;
        
        let cache = robust_confidence::MeanCache::default();
        
        group.bench_with_input(
            BenchmarkId::new("T_distribution", size),
            &data,
            |b, data| {
                b.iter(|| {
                    t_ci.confidence_interval(black_box(data), &mean_se, &cache)
                })
            }
        );
        
        // Normal approximation
        let normal_ci = AsymptoticCI::normal(0.95);
        
        group.bench_with_input(
            BenchmarkId::new("Normal", size),
            &data,
            |b, data| {
                b.iter(|| {
                    normal_ci.confidence_interval(black_box(data), &mean_se, &cache)
                })
            }
        );
    }
    
    group.finish();
}

// Alternative bootstrap benchmark using quantile features
#[cfg(feature = "quantile")]
fn bench_bootstrap_quantile(c: &mut Criterion) {
    let mut group = c.benchmark_group("BootstrapQuantile");
    
    use robust_confidence::{
        quantile_shift_confidence_intervals,
        FAST_RESAMPLES,
    };
    use robust_quantile::harrell_davis_factory;
    
    let sizes = [50, 100, 200];
    
    for &size in &sizes {
        let data1 = generate_normal_data(size, 100.0, 15.0, 42);
        let data2 = generate_normal_data(size, 105.0, 15.0, 43);
        
        let factory = harrell_davis_factory();
        let engine = auto_budgeted_engine();
        
        // Benchmark median shift CI
        group.bench_with_input(
            BenchmarkId::new("median_shift", size),
            &(&data1, &data2),
            |b, (data1, data2)| {
                b.iter(|| {
                    quantile_shift_confidence_intervals(
                        black_box(*data1),
                        black_box(*data2),
                        &[0.5],
                        &factory,
                        engine.clone(),
                        PercentileBootstrap,
                        0.95,
                        FAST_RESAMPLES,
                    )
                })
            }
        );
        
        // Benchmark quartile shift CIs
        group.bench_with_input(
            BenchmarkId::new("quartile_shifts", size),
            &(&data1, &data2),
            |b, (data1, data2)| {
                b.iter(|| {
                    quantile_shift_confidence_intervals(
                        black_box(*data1),
                        black_box(*data2),
                        &[0.25, 0.5, 0.75],
                        &factory,
                        engine.clone(),
                        PercentileBootstrap,
                        0.95,
                        FAST_RESAMPLES,
                    )
                })
            }
        );
    }
    
    group.finish();
}

// Define which benchmarks to run based on features
#[cfg(all(feature = "quantile", feature = "spread"))]
criterion_group!(
    benches,
    bench_maritz_jarrett,
    bench_bootstrap,
    bench_bootstrap_quantile,
    bench_asymptotic
);

#[cfg(all(feature = "quantile", not(feature = "spread")))]
criterion_group!(
    benches,
    bench_maritz_jarrett,
    bench_bootstrap_quantile,
    bench_asymptotic
);

#[cfg(not(feature = "quantile"))]
criterion_group!(
    benches,
    bench_asymptotic
);

criterion_main!(benches);