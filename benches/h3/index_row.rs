use criterion::{BenchmarkId, Criterion, Throughput};
use h3o::{GeographicRow, LatLngIndexer, Resolution, RowPath};
use std::hint::black_box;

/// Pixels per row.
const WIDTH: usize = 4000;

/// Row origins (latitude, first longitude) across the CONUS, in degrees.
const ROWS: [(f64, f64); 8] = [
    (25.3, -81.2),
    (29.9, -97.6),
    (33.4, -117.1),
    (36.8, -89.4),
    (39.7, -105.0),
    (42.2, -71.9),
    (45.5, -122.6),
    (48.7, -100.3),
];

/// Indexes whole raster rows at one or several resolutions: per-pixel
/// `to_cell`, a reused `LatLngIndexer::to_cells`, and
/// `LatLngIndexer::index_row`. Throughput is in pixels.
pub fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("latLngIndexRow");
    group.sample_size(20);
    group.throughput(Throughput::Elements((ROWS.len() * WIDTH) as u64));

    // 1 and 1/3 arc-second pixels (about 30 m and 10 m).
    for (spacing, step) in [("30m", 1. / 3600.), ("10m", 1. / 10800.)] {
        let rows = ROWS
            .iter()
            .map(|&(lat, lng0)| GeographicRow::new(lat, lng0, step))
            .collect::<Vec<_>>();
        for resolutions in [&[6][..], &[10], &[12], &[6, 8, 10]] {
            let resolutions = resolutions
                .iter()
                .map(|&res| Resolution::try_from(res).expect("resolution"))
                .collect::<Vec<_>>();
            let id = format!("{spacing} {resolutions:?}");

            group.bench_with_input(
                BenchmarkId::new("h3o/to_cell", &id),
                &resolutions,
                |b, resolutions| {
                    let mut rows = rows.clone();
                    b.iter(|| {
                        for row in &mut rows {
                            for col in 0..WIDTH {
                                let ll = row.point(black_box(col));
                                for &res in resolutions {
                                    black_box(ll.to_cell(res));
                                }
                            }
                        }
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("h3o/LatLngIndexer", &id),
                &resolutions,
                |b, resolutions| {
                    let mut rows = rows.clone();
                    b.iter(|| {
                        let mut indexer = LatLngIndexer::new();
                        for row in &mut rows {
                            for col in 0..WIDTH {
                                let ll = row.point(black_box(col));
                                for cell in indexer
                                    .to_cells(ll, resolutions.iter().copied())
                                {
                                    black_box(cell);
                                }
                            }
                        }
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("h3o/index_row", &id),
                &resolutions,
                |b, resolutions| {
                    let mut rows = rows.clone();
                    b.iter(|| {
                        let mut indexer = LatLngIndexer::new();
                        for row in &mut rows {
                            black_box(indexer.index_row(
                                row,
                                0..black_box(WIDTH),
                                resolutions,
                                |index, cols, cell| {
                                    black_box((index, cols, cell));
                                },
                            ));
                        }
                    });
                },
            );
        }
    }
    group.finish();
}
