//! Benchmarks comparing scalar vs SIMD implementations of compute primitives

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use robust_core::{
    ComputePrimitives, ScalarBackend, SparseTile, TileEntry,
};

#[cfg(feature = "avx2")]
use robust_core::Avx2Backend;

/// Generate test data with specific patterns
fn generate_test_data(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| (i as f64 * 0.1).sin() * 100.0)
        .collect()
}

/// Generate random indices for sparse operations
fn generate_sparse_indices(n: usize, data_size: usize) -> Vec<usize> {
    (0..n)
        .map(|i| (i * 7 + 3) % data_size)
        .collect()
}

/// Generate weights for sparse operations
fn generate_weights(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.1 + (i as f64 * 0.01).cos().abs())
        .collect()
}

/// Benchmark sparse_weighted_sum implementations
fn bench_sparse_weighted_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_weighted_sum");
    
    let data_sizes = vec![1000, 10000, 100000];
    let sparse_sizes = vec![10, 50, 100, 500];
    
    for &data_size in &data_sizes {
        let data = generate_test_data(data_size);
        
        for &sparse_size in &sparse_sizes {
            let indices = generate_sparse_indices(sparse_size, data_size);
            let weights = generate_weights(sparse_size);
            
            // Benchmark scalar implementation
            let scalar_backend = ScalarBackend::new();
            group.bench_with_input(
                BenchmarkId::new("scalar", format!("{data_size}_data_{sparse_size}_sparse")),
                &(&data, &indices, &weights),
                |b, (data, indices, weights)| {
                    b.iter(|| {
                        black_box(scalar_backend.sparse_weighted_sum(data, indices, weights))
                    });
                },
            );
            
            // Benchmark AVX2 implementation
            #[cfg(feature = "avx2")]
            if Avx2Backend::is_available() {
                let avx2_backend = Avx2Backend::new();
                group.bench_with_input(
                    BenchmarkId::new("avx2", format!("{data_size}_data_{sparse_size}_sparse")),
                    &(&data, &indices, &weights),
                    |b, (data, indices, weights)| {
                        b.iter(|| {
                            black_box(avx2_backend.sparse_weighted_sum(data, indices, weights))
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

/// Create a test tile with specific density
fn create_test_tile(rows: usize, cols: usize, density: f64) -> SparseTile {
    let total_entries = rows * cols;
    let n_entries = (total_entries as f64 * density) as usize;
    
    let mut entries = Vec::with_capacity(n_entries);
    for i in 0..n_entries {
        // Use a pattern that distributes entries across the tile
        let row = (i * 7) % rows;
        let col = (i * 13) % cols;
        let weight = 0.1 + (i as f64 * 0.01).sin().abs();
        
        entries.push(TileEntry {
            local_row: row as u16,
            local_col: col as u16,
            weight,
        });
    }
    
    SparseTile::new(0, 0, 0, rows, 0, cols, entries)
}

/// Benchmark apply_sparse_tile implementations
fn bench_apply_sparse_tile(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_sparse_tile");
    
    let tile_configs = vec![
        (16, 16, 0.5),    // Small tile, 50% density
        (32, 32, 0.25),   // Medium tile, 25% density
        (64, 64, 0.1),    // Large tile, 10% density
        (128, 128, 0.05), // Very large tile, 5% density
    ];
    
    for (rows, cols, density) in tile_configs {
        let tile = create_test_tile(rows, cols, density);
        let tile_data: Vec<f64> = (0..cols).map(|i| i as f64 * 0.1).collect();
        let mut result = vec![0.0; rows];
        
        let config_name = format!("{}x{}_{:.0}%", rows, cols, density * 100.0);
        
        // Benchmark scalar implementation
        let scalar_backend = ScalarBackend::new();
        group.bench_with_input(
            BenchmarkId::new("scalar", &config_name),
            &(&tile_data, &tile),
            |b, (data, tile)| {
                b.iter(|| {
                    result.fill(0.0);
                    scalar_backend.apply_sparse_tile(data, *tile, &mut result);
                    black_box(&result);
                });
            },
        );
        
        // Benchmark scalar backend implementation (unchecked)
        group.bench_with_input(
            BenchmarkId::new("scalar_unchecked", &config_name),
            &(&tile_data, &tile),
            |b, (data, tile)| {
                b.iter(|| {
                    result.fill(0.0);
                    unsafe {
                        scalar_backend.apply_sparse_tile_unchecked(data, &**tile, &mut result);
                    }
                    black_box(&result);
                });
            },
        );
        
        // Benchmark AVX2 implementation
        #[cfg(feature = "avx2")]
        if Avx2Backend::is_available() {
            let avx2_backend = Avx2Backend::new();
            group.bench_with_input(
                BenchmarkId::new("avx2", &config_name),
                &(&tile_data, &tile),
                |b, (data, tile)| {
                    b.iter(|| {
                        result.fill(0.0);
                        avx2_backend.apply_sparse_tile(data, *tile, &mut result);
                        black_box(&result);
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark dot product for comparison
fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    
    let sizes = vec![100, 1000, 10000, 100000];
    
    for &size in &sizes {
        let a = generate_test_data(size);
        let b = generate_test_data(size);
        
        // Benchmark scalar implementation
        let scalar_backend = ScalarBackend::new();
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    black_box(scalar_backend.dot_product(a, b))
                });
            },
        );
        
        // Benchmark AVX2 implementation
        #[cfg(feature = "avx2")]
        if Avx2Backend::is_available() {
            let avx2_backend = Avx2Backend::new();
            group.bench_with_input(
                BenchmarkId::new("avx2", size),
                &(&a, &b),
                |bench, (a, b)| {
                    bench.iter(|| {
                        black_box(avx2_backend.dot_product(a, b))
                    });
                },
            );
        }
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_sparse_weighted_sum,
    bench_apply_sparse_tile,
    bench_dot_product
);
criterion_main!(benches);