use super::*;
use crate::{CellIndex, LatLng, LatLngIndexer, Resolution, math::mul_add};

/// Deterministic PRNG in [0, 1).
fn rng(mut state: u64) -> impl FnMut() -> f64 {
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[expect(clippy::cast_precision_loss, reason = "test")]
        let v = (state >> 11) as f64 / (1_u64 << 53) as f64;
        v
    }
}

// The proof relies on `CoordIJK::from(Vec2d)` returning the hexagon that
// contains the point (lattice Voronoi cell), which `hex_margin` measures.
#[test]
fn hex_rounding_returns_containing_hexagon() {
    let mut next = rng(42);
    for scale in [1., 10., 1e3, 1e5, 3e6] {
        for _ in 0..200_000 {
            let q = Vec2d::new(
                mul_add(2., next(), -1.) * scale,
                mul_add(2., next(), -1.) * scale,
            );
            let ijk = CoordIJK::from(q);
            let delta = hex_margin(q, ijk);
            assert!(
                delta >= -1e-12 * (1. + scale),
                "{q:?} rounded to {ijk:?}, {delta} outside"
            );
        }
    }
}

/// Indexes the row and checks every column against `to_cell`.
#[expect(clippy::needless_range_loop, reason = "column-major check")]
fn check_row(
    lat: f64,
    lng0: f64,
    step: f64,
    width: usize,
    resolutions: &[Resolution],
) -> RowStats {
    let mut cells = vec![vec![None::<CellIndex>; width]; resolutions.len()];
    let mut path = GeographicRow::new(lat, lng0, step);
    let stats = LatLngIndexer::new().index_row(
        &mut path,
        0..width,
        resolutions,
        |index, cols, cell| {
            for col in cols {
                assert!(cells[index][col].is_none(), "column {col} twice");
                cells[index][col] = Some(cell);
            }
        },
    );
    for col in 0..width {
        let ll = path.point(col);
        for (index, &res) in resolutions.iter().enumerate() {
            assert_eq!(
                cells[index][col],
                Some(ll.to_cell(res)),
                "lat {lat} lng0 {lng0} step {step} col {col} res {res:?} ({ll})"
            );
        }
    }
    stats
}

#[test]
fn rows_match_to_cell_on_hard_cases() {
    let all = Resolution::range(Resolution::Zero, Resolution::Fifteen)
        .collect::<Vec<_>>();
    let face_center = LatLng::from(Vec3d::new(
        face::CENTER_POINT[0].x,
        face::CENTER_POINT[0].y,
        face::CENTER_POINT[0].z,
    ));
    let mut origins = vec![
        (39.7, -105.0),
        (0.0, 179.9),  // antimeridian
        (89.99, 30.0), // pole
        (-89.9, -10.0),
        (face_center.lat(), face_center.lng() - 0.05), // through a face center
        (58.282_525_588_538_995, 11.25), // icosahedron vertex (pentagon)
    ];
    origins.extend(
        CellIndex::base_cells()
            .filter(|cell| cell.is_pentagon())
            .map(|cell| {
                let ll = LatLng::from(cell);
                (ll.lat(), ll.lng() - 0.05)
            }),
    );
    for &(lat, lng0) in &origins {
        for step in [9e-5, 2.7e-4, 9e-3] {
            check_row(lat, lng0, step, 2000, &all);
        }
    }
}

#[test]
fn rows_match_to_cell_scattered() {
    let resolutions = [
        Resolution::Five,
        Resolution::Seven,
        Resolution::Nine,
        Resolution::Ten,
        Resolution::Twelve,
    ];
    let mut next = rng(7);
    for _ in 0..300 {
        let lat = mul_add(2.0, next(), -1.0).asin().to_degrees();
        let lng0 = mul_add(360.0, next(), -180.0);
        let step = 10_f64.powf(mul_add(3., next(), -5.)); // ~1 m .. ~1 km
        check_row(lat, lng0, step, 3000, &resolutions);
    }
}

/// Test path: `count` samples along a segment that is linear in latitude and
/// longitude (radians). Unlike `GeographicRow`, latitude varies and the path can
/// cross faces at any angle, which exercises the general `RowPath` machinery.
///
/// Sample `i` (for `i` in `0..=count`) is computed exactly as:
///
/// ```text
/// lat = (start.lat * (n - i)).mul_add(1/n, end.lat * i * (1/n))
/// lng = (start.lng * (n - i)).mul_add(1/n, end.lng * i * (1/n))
/// ```
///
/// with `n = count as f64`, so sample 0 is `start` and sample `count` is
/// `end`.
#[derive(Debug, Clone, Copy)]
struct LatLngSegment {
    start: (f64, f64),
    end: (f64, f64),
    count: f64,
    inv_count: f64,
    speed: f64,
    accel: f64,
}

impl LatLngSegment {
    /// Initializes `count` samples from `start` towards `end`.
    ///
    /// # Panics
    ///
    /// If `count` is zero or above 2^53.
    fn new(start: LatLng, end: LatLng, count: u64) -> Self {
        assert!(count > 0 && count <= 1 << f64::MANTISSA_DIGITS);
        #[expect(clippy::cast_precision_loss, reason = "checked above")]
        let count = count as f64;
        let start = (start.lat_radians(), start.lng_radians());
        let end = (end.lat_radians(), end.lng_radians());
        // With φ and λ linear in the sample index:
        //   |P'| = sqrt(φ'² + cos²φ λ'²) ≤ |φ'| + |λ'|
        //   |P''| ≤ φ'² + 2|φ'λ'| + λ'² = (|φ'| + |λ'|)²
        // Slightly inflated to absorb rounding.
        let rate = (abs(end.0 - start.0) + abs(end.1 - start.1)) / count;
        let speed = mul_add(rate, 1. + 1e-9, 1e-300);
        Self {
            start,
            end,
            count,
            inv_count: 1. / count,
            speed,
            accel: speed * speed,
        }
    }
}

impl RowPath for LatLngSegment {
    #[expect(clippy::cast_precision_loss, reason = "i ≤ count ≤ 2^53")]
    fn point(&mut self, i: usize) -> LatLng {
        let i = i as f64;
        let lat = mul_add(
            self.start.0 * (self.count - i),
            self.inv_count,
            self.end.0 * i * self.inv_count,
        );
        let lng = mul_add(
            self.start.1 * (self.count - i),
            self.inv_count,
            self.end.1 * i * self.inv_count,
        );
        LatLng::from_radians(lat, lng).expect("finite coordinate")
    }

    fn speed_bound(&self) -> f64 {
        self.speed
    }

    fn accel_bound(&self) -> f64 {
        self.accel
    }
}

/// Indexes the segment samples and checks each against `to_cell`.
fn check_segment(start: LatLng, end: LatLng, count: u64, res: &[Resolution]) {
    let width = usize::try_from(count).expect("count");
    let mut cells = vec![vec![None::<CellIndex>; width]; res.len()];
    let mut path = LatLngSegment::new(start, end, count);
    LatLngIndexer::new().index_row(
        &mut path,
        0..width,
        res,
        |i, cols, cell| {
            for col in cols {
                assert!(cells[i][col].is_none(), "sample {col} twice");
                cells[i][col] = Some(cell);
            }
        },
    );
    for (i, &r) in res.iter().enumerate() {
        for (col, cell) in cells[i].iter().enumerate() {
            let ll = path.point(col);
            assert_eq!(
                *cell,
                Some(ll.to_cell(r)),
                "{start} -> {end} ({count}) sample {col} res {r:?}"
            );
        }
    }
}

#[test]
fn segments_match_to_cell() {
    let resolutions = [
        Resolution::Four,
        Resolution::Seven,
        Resolution::Nine,
        Resolution::Eleven,
        Resolution::Thirteen,
    ];
    let face_center = LatLng::from(Vec3d::new(
        face::CENTER_POINT[3].x,
        face::CENTER_POINT[3].y,
        face::CENTER_POINT[3].z,
    ));
    let ll = |lat: f64, lng: f64| LatLng::new(lat, lng).expect("valid");
    let hard = [
        // Through a face center, across the antimeridian, near a pole, and
        // through an icosahedron vertex.
        (
            ll(face_center.lat() - 0.3, face_center.lng() - 0.4),
            ll(face_center.lat() + 0.3, face_center.lng() + 0.4),
        ),
        (ll(-10.0, 179.0), ll(-10.5, 181.0)),
        (ll(89.0, -30.0), ll(89.9, 150.0)),
        (ll(58.2, 11.0), ll(58.4, 11.5)),
    ];
    for (start, end) in hard {
        for count in [8, 100, 5000] {
            check_segment(start, end, count, &resolutions);
        }
    }
    let mut next = rng(11);
    for _ in 0..400 {
        let lat = mul_add(2.0, next(), -1.0).asin().to_degrees();
        let lng = mul_add(360.0, next(), -180.0);
        let len = 10_f64.powf(mul_add(3., next(), -3.)); // 0.001° .. 1°
        let angle = next() * core::f64::consts::TAU;
        let start = ll(lat, lng);
        let end = ll(
            (lat + len * angle.sin()).clamp(-90., 90.),
            lng + len * angle.cos(),
        );
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "test"
        )]
        let count = 10_f64.powf(mul_add(3., next(), 1.)) as u64; // 10 .. 10k
        check_segment(start, end, count, &resolutions);
    }
}

/// A geographic row whose points jitter around the smooth row by up to
/// `sigma` radians (deterministic per index, like an iterative inverse
/// projection's tolerance), reporting `reported` as its noise bound.
struct JitteredRow {
    row: GeographicRow,
    sigma: f64,
    reported: f64,
    seed: u64,
}

impl JitteredRow {
    /// Uniform in [-1, 1), from the index.
    fn unit(&self, i: usize, salt: u64) -> f64 {
        let mut z =
            self.seed ^ salt ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        #[expect(clippy::cast_precision_loss, reason = "test")]
        let v = (z >> 11) as f64 / (1_u64 << 52) as f64 - 1.;
        v
    }
}

impl RowPath for JitteredRow {
    fn point(&mut self, i: usize) -> LatLng {
        let ll = self.row.point(i);
        let (lat, lng) = (ll.lat_radians(), ll.lng_radians());
        // Offsets of ≤ σ/2 north-south and east-west: ≤ σ/√2 in angle.
        let half = 0.5 * self.sigma;
        LatLng::from_radians(
            mul_add(half, self.unit(i, 1), lat),
            mul_add(half / cos(lat), self.unit(i, 2), lng),
        )
        .expect("finite coordinate")
    }

    fn speed_bound(&self) -> f64 {
        self.row.speed_bound()
    }

    fn accel_bound(&self) -> f64 {
        self.row.accel_bound()
    }

    fn noise_bound(&self) -> f64 {
        self.reported
    }
}

/// Indexes `0..width` and returns the stats and the number of (location,
/// resolution) pairs that differ from `to_cell`.
#[expect(clippy::needless_range_loop, reason = "column-major check")]
fn count_mismatches(
    path: &mut impl RowPath,
    width: usize,
    res: &[Resolution],
) -> (RowStats, usize) {
    let mut cells = vec![vec![None::<CellIndex>; width]; res.len()];
    let stats =
        LatLngIndexer::new().index_row(path, 0..width, res, |i, cols, cell| {
            for col in cols {
                assert!(cells[i][col].is_none(), "column {col} twice");
                cells[i][col] = Some(cell);
            }
        });
    let mut mismatches = 0;
    for col in 0..width {
        let ll = path.point(col);
        for (i, &r) in res.iter().enumerate() {
            let cell = cells[i][col].expect("every column indexed");
            mismatches += usize::from(cell != ll.to_cell(r));
        }
    }
    (stats, mismatches)
}

fn jittered_rows(reported_as_sigma: bool) -> impl Iterator<Item = JitteredRow> {
    let mut next = rng(23);
    (0..80_u64).map(move |seed| {
        let lat = mul_add(140.0, next(), -70.0);
        let lng0 = mul_add(360.0, next(), -180.0);
        let step = 10_f64.powf(mul_add(1.5, next(), -5.)); // ~1 m .. ~35 m
        let sigma = 10_f64.powf(mul_add(1., next(), -7.5)); // 3e-8 .. 3e-7 rad
        JitteredRow {
            row: GeographicRow::new(lat, lng0, step),
            sigma,
            reported: if reported_as_sigma { sigma } else { 0. },
            seed,
        }
    })
}

const JITTER_RESOLUTIONS: [Resolution; 4] = [
    Resolution::Nine,
    Resolution::Eleven,
    Resolution::Twelve,
    Resolution::Thirteen,
];

#[test]
fn jittered_rows_match_to_cell_when_noise_is_reported() {
    for mut path in jittered_rows(true) {
        let (stats, mismatches) =
            count_mismatches(&mut path, 3000, &JITTER_RESOLUTIONS);
        assert_eq!(mismatches, 0, "sigma {}", path.sigma);
        assert!(stats.decisions > 0, "interpolation still used");
    }
}

// Control: the jitter is large enough to matter, so the test above is
// meaningful.
#[test]
fn jittered_rows_mismatch_when_noise_is_not_reported() {
    let total = jittered_rows(false)
        .map(|mut path| {
            count_mismatches(&mut path, 3000, &JITTER_RESOLUTIONS).1
        })
        .sum::<usize>();
    assert!(total > 0, "jitter never changed a cell");
}

#[test]
fn invalid_or_large_noise_falls_back_to_to_cell() {
    let width = 500;
    for reported in [f64::NAN, -1e-9, 2e-3] {
        let mut path = JitteredRow {
            row: GeographicRow::new(39.7, -105.0, 2.7e-4),
            sigma: 1e-7,
            reported,
            seed: 5,
        };
        let (stats, mismatches) =
            count_mismatches(&mut path, width, &JITTER_RESOLUTIONS);
        assert_eq!(mismatches, 0, "{reported}");
        assert_eq!(stats.decisions, 0, "{reported}");
        assert_eq!(stats.fallbacks, width * JITTER_RESOLUTIONS.len());
    }
}


