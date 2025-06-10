//! Benchmarks comparing different memory layouts for sparse tiles
//!
//! This benchmark compares:
//! - AoS (Array of Structs) - Original implementation
//! - Vec SoA (Structure of Arrays with separate Vecs)
//! - Optimized SoA (Single contiguous buffer with alignment)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use robust_core::{ScalarBackend, SparseTile, TileEntry};

#[cfg(feature = "benchmark-variants")]
use robust_core::tiled::benchmark_variants::{SparseTileAoS, SparseTileVecSoA};

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
use robust_core::Avx2Backend;

/// Create test entries for benchmarking
fn create_test_entries(rows: usize, cols: usize, density: f64) -> Vec<TileEntry> {
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

    entries
}


#[cfg(feature = "benchmark-variants")]
fn bench_memory_layouts(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_layouts");

    let tile_configs = vec![
        (16, 16, 0.5, "small_dense"),      // Small tile, 50% density
        (32, 32, 0.25, "medium_medium"),   // Medium tile, 25% density
        (64, 64, 0.1, "large_sparse"),     // Large tile, 10% density
        (128, 128, 0.05, "xlarge_sparse"), // Very large tile, 5% density
    ];

    for (rows, cols, density, name) in tile_configs {
        let entries = create_test_entries(rows, cols, density);
        let tile_data: Vec<f64> = (0..cols).map(|i| i as f64 * 0.1).collect();
        let mut result = vec![0.0; rows];

        // Benchmark AoS with bounds checks
        let aos_tile = SparseTileAoS::new(0, 0, 0, rows, 0, cols, entries.clone());
        group.bench_with_input(
            BenchmarkId::new("aos_bounds", name),
            &(&tile_data, &aos_tile),
            |b, (data, tile)| {
                b.iter(|| {
                    result.fill(0.0);
                    tile.apply(data, &mut result);
                    black_box(&result);
                });
            },
        );

        // Benchmark AoS without bounds checks
        group.bench_with_input(
            BenchmarkId::new("aos_unchecked", name),
            &(&tile_data, &aos_tile),
            |b, (data, tile)| {
                b.iter(|| {
                    result.fill(0.0);
                    unsafe {
                        tile.apply_unchecked(data, &mut result);
                    }
                    black_box(&result);
                });
            },
        );

        // Benchmark Vec SoA with bounds checks
        let vec_soa_tile = SparseTileVecSoA::new(0, 0, 0, rows, 0, cols, entries.clone());
        group.bench_with_input(
            BenchmarkId::new("vec_soa_bounds", name),
            &(&tile_data, &vec_soa_tile),
            |b, (data, tile)| {
                b.iter(|| {
                    result.fill(0.0);
                    tile.apply(data, &mut result);
                    black_box(&result);
                });
            },
        );

        // Benchmark Vec SoA without bounds checks
        group.bench_with_input(
            BenchmarkId::new("vec_soa_unchecked", name),
            &(&tile_data, &vec_soa_tile),
            |b, (data, tile)| {
                b.iter(|| {
                    result.fill(0.0);
                    unsafe {
                        tile.apply_unchecked(data, &mut result);
                    }
                    black_box(&result);
                });
            },
        );

        // Benchmark Optimized SoA with scalar backend
        let opt_tile = SparseTile::new(0, 0, 0, rows, 0, cols, entries.clone());
        let scalar_backend = ScalarBackend::new();
        group.bench_with_input(
            BenchmarkId::new("opt_soa_scalar", name),
            &(&tile_data, &opt_tile),
            |b, (data, tile)| {
                b.iter(|| {
                    result.fill(0.0);
                    tile.apply(data, &mut result, &scalar_backend);
                    black_box(&result);
                });
            },
        );

        // Benchmark Optimized SoA with AVX2 backend
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        if is_x86_feature_detected!("avx2") {
            let avx2_backend = Avx2Backend::new();
            group.bench_with_input(
                BenchmarkId::new("opt_soa_avx2", name),
                &(&tile_data, &opt_tile),
                |b, (data, tile)| {
                    b.iter(|| {
                        result.fill(0.0);
                        tile.apply(data, &mut result, &avx2_backend);
                        black_box(&result);
                    });
                },
            );
        }
    }

    group.finish();
}

#[cfg(feature = "benchmark-variants")]
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    let sizes = vec![
        (100, "100_entries"),
        (1000, "1k_entries"),
        (10000, "10k_entries"),
    ];

    for (n_entries, name) in sizes {
        let entries: Vec<TileEntry> = (0..n_entries)
            .map(|i| TileEntry {
                local_row: (i % 128) as u16,
                local_col: ((i * 7) % 128) as u16,
                weight: 0.1 + (i as f64 * 0.01).sin().abs(),
            })
            .collect();

        // Benchmark AoS allocation
        group.bench_with_input(BenchmarkId::new("aos", name), &entries, |b, entries| {
            b.iter(|| {
                let tile = SparseTileAoS::new(0, 0, 0, 128, 0, 128, entries.clone());
                black_box(tile);
            });
        });

        // Benchmark Vec SoA allocation
        group.bench_with_input(BenchmarkId::new("vec_soa", name), &entries, |b, entries| {
            b.iter(|| {
                let tile = SparseTileVecSoA::new(0, 0, 0, 128, 0, 128, entries.clone());
                black_box(tile);
            });
        });

        // Benchmark Optimized SoA allocation
        group.bench_with_input(BenchmarkId::new("opt_soa", name), &entries, |b, entries| {
            b.iter(|| {
                let tile = SparseTile::new(0, 0, 0, 128, 0, 128, entries.clone());
                black_box(tile);
            });
        });
    }

    group.finish();
}


#[cfg(feature = "benchmark-variants")]
criterion_group!(
    benches,
    bench_memory_layouts,
    bench_memory_allocation
);

#[cfg(feature = "benchmark-variants")]
criterion_main!(benches);

#[cfg(not(feature = "benchmark-variants"))]
fn main() {
    eprintln!("This benchmark requires the 'benchmark-variants' feature");
    eprintln!("Run with: cargo bench -p robust-core --bench memory_layout_bench --features benchmark-variants");
}
