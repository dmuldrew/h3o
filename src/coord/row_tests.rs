use super::super::FaceProjection;
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
    /// Whether to report bounds for each span (see `span_bounds`).
    per_span: bool,
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
            per_span: false,
        }
    }

    /// Reports bounds for each span instead of path-wide ones.
    fn per_span(mut self) -> Self {
        self.per_span = true;
        self
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

    // With φ' = a and λ' = b constant:
    //   |P'| = sqrt(a² + b² cos²φ) ≤ |a| + |b| cos φ
    //   |P''| ≤ a² |∂φφP| + 2|ab| |∂φλP| + b² |∂λλP|
    //         = a² + 2|ab| |sin φ| + b² cos φ ≤ a² + 2|ab| + b² cos φ
    // with the largest cos φ over the span: φ is linear in the index, so at
    // an end, or 1 if the span crosses the equator. Near a pole, these are
    // much tighter than the path-wide (|a| + |b|) and (|a| + |b|)².
    fn span_bounds(&mut self, first: usize, last: usize) -> Option<SpanBounds> {
        if !self.per_span {
            return Some(SpanBounds {
                speed: self.speed,
                accel: self.accel,
                noise: 0.,
            });
        }
        let (lat0, lat1) = (
            self.point(first).lat_radians(),
            self.point(last).lat_radians(),
        );
        let cos_max = if lat0.signum() == lat1.signum() {
            cos(lat0).max(cos(lat1))
        } else {
            1.
        };
        let cos_max = mul_add(cos_max, 1. + 1e-9, 1e-12).min(1.);
        let a = abs(self.end.0 - self.start.0) * self.inv_count;
        let b = abs(self.end.1 - self.start.1) * self.inv_count;
        Some(SpanBounds {
            speed: mul_add(mul_add(b, cos_max, a), 1. + 1e-9, 1e-300),
            accel: mul_add(
                mul_add(b * b, cos_max, mul_add(a, a, 2. * a * b)),
                1. + 1e-9,
                1e-300,
            ),
            noise: 0.,
        })
    }
}

/// Indexes the segment samples and checks each against `to_cell`, with
/// path-wide bounds and with per-span bounds.
fn check_segment(start: LatLng, end: LatLng, count: u64, res: &[Resolution]) {
    check_segment_path(LatLngSegment::new(start, end, count), res);
    check_segment_path(LatLngSegment::new(start, end, count).per_span(), res);
}

fn check_segment_path(mut path: LatLngSegment, res: &[Resolution]) {
    let count = path.count;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "count is an integer ≤ 2^53"
    )]
    let width = count as usize;
    let (start, end) = (path.point(0), path.point(width));
    let mut cells = vec![vec![None::<CellIndex>; width]; res.len()];
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

// Segments heading towards a pole with a large change in longitude: per-span
// bounds shrink near the pole, so the same exact cells take less work.
#[test]
fn per_span_bounds_stay_exact_and_save_work() {
    let resolutions = [Resolution::Nine, Resolution::Eleven];
    let ll = |lat: f64, lng: f64| LatLng::new(lat, lng).expect("valid");
    for (start, end, count) in [
        (ll(60., 0.), ll(89.5, 180.), 20_000),
        (ll(-50., 20.), ll(-89., -160.), 10_000),
        (ll(70., -170.), ll(85., 170.), 20_000),
        (ll(70., -170.), ll(85., 170.), 400_000),
    ] {
        let width = usize::try_from(count).expect("count");
        let (global, mismatches) = count_mismatches(
            &mut LatLngSegment::new(start, end, count),
            width,
            &resolutions,
        );
        assert_eq!(mismatches, 0, "path-wide bounds, {start} -> {end}");
        let (per_span, mismatches) = count_mismatches(
            &mut LatLngSegment::new(start, end, count).per_span(),
            width,
            &resolutions,
        );
        assert_eq!(mismatches, 0, "per-span bounds, {start} -> {end}");
        assert!(
            per_span.samples + per_span.fallbacks
                < global.samples + global.fallbacks,
            "{start} -> {end}: {per_span:?} vs {global:?}"
        );
    }
}

/// A geographic row whose locations in `hole` lie on no smooth path at all
/// (scattered over the globe). Spans that touch the hole report no bounds.
struct HoleRow {
    row: GeographicRow,
    hole: Range<usize>,
}

impl RowPath for HoleRow {
    #[expect(clippy::cast_precision_loss, reason = "test")]
    fn point(&mut self, i: usize) -> LatLng {
        if self.hole.contains(&i) {
            let lat = ((i * 7919) % 1700) as f64 / 10. - 85.;
            let lng = ((i * 104_729) % 3600) as f64 / 10. - 180.;
            LatLng::new(lat, lng).expect("valid")
        } else {
            self.row.point(i)
        }
    }

    // Wrong inside the hole: only valid through `span_bounds`.
    fn speed_bound(&self) -> f64 {
        self.row.speed_bound()
    }

    fn accel_bound(&self) -> f64 {
        self.row.accel_bound()
    }

    fn span_bounds(&mut self, first: usize, last: usize) -> Option<SpanBounds> {
        (last < self.hole.start || first >= self.hole.end).then(|| SpanBounds {
            speed: self.row.speed_bound(),
            accel: self.row.accel_bound(),
            noise: 0.,
        })
    }
}

#[test]
fn spans_without_bounds_use_to_cell() {
    let width = 3000;
    for (hole, step) in
        [(1000..1200, 2.7e-4), (0..10, 9e-5), (2990..3000, 9e-3)]
    {
        let len = hole.len();
        let mut path = HoleRow {
            row: GeographicRow::new(39.7, -105.0, step),
            hole,
        };
        let (stats, mismatches) =
            count_mismatches(&mut path, width, &JITTER_RESOLUTIONS);
        assert_eq!(mismatches, 0, "{:?}", path.hole);
        assert!(stats.decisions > 0, "the rest still interpolates");
        assert!(stats.fallbacks >= len * JITTER_RESOLUTIONS.len());
    }
}

// Dumps `to_cell`'s floating-point hex-plane positions for the locations in
// `$H3O_NOISE_IN` (one `lat lng` pair per line, as hexadecimal f64 bits in
// radians) to `$H3O_NOISE_OUT`, one line per location: face, then x and y at
// resolutions 0, 14 and 15 (hexadecimal f64 bits). Driven by
// `tools/to_cell_noise.py`, which measures them against a high-precision
// evaluation to check `TO_CELL_NOISE`.
#[test]
#[ignore = "run through tools/to_cell_noise.py"]
fn dump_to_cell_positions() {
    use std::{fmt::Write as _, io::Write as _};

    let parse = |s: &str| f64::from_bits(u64::from_str_radix(s, 16).unwrap());
    let input = std::env::var("H3O_NOISE_IN").unwrap();
    let output = std::env::var("H3O_NOISE_OUT").unwrap();
    let mut out =
        std::io::BufWriter::new(std::fs::File::create(output).unwrap());
    for line in std::fs::read_to_string(input).unwrap().lines() {
        let (lat, lng) = line.split_once(' ').unwrap();
        let ll = LatLng::from_radians(parse(lat), parse(lng)).unwrap();
        let p = Vec3d::from(ll);
        let (face, distance) = p.closest_face();
        let projection = FaceProjection::new(p, face, distance);
        let mut row = std::format!("{}", usize::from(face));
        for q in [
            projection.res0_xy(),
            projection.to_vec2d(Resolution::Fourteen),
            projection.to_vec2d(Resolution::Fifteen),
        ] {
            write!(row, " {:x} {:x}", q.x.to_bits(), q.y.to_bits()).unwrap();
        }
        writeln!(out, "{row}").unwrap();
    }
}

// -----------------------------------------------------------------------------
// Large randomized equivalence run.
//
//     cargo test --release --lib index_row_equivalence -- --ignored --nocapture
//
// Environment: `H3O_EQUIV_ROWS` (default 10⁸), `H3O_EQUIV_SEED` (default 1),
// `H3O_EQUIV_THREADS` (default: all cores), `H3O_EQUIV_CONTROL=1` to
// understate every bound 10× (the run must then find mismatches), and
// `H3O_EQUIV_ROW=<n>` to rerun a single row verbosely. Each row is generated
// from `(seed, row number)` alone, so a reported row can be reproduced.
#[expect(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::use_debug,
    reason = "progress and failure report of a manual run"
)]
mod equivalence {
    use super::*;

    /// Divides a path's speed and acceleration bounds by 10 (controls).
    struct Understated<P>(P);

    impl<P: RowPath> RowPath for Understated<P> {
        fn point(&mut self, i: usize) -> LatLng {
            self.0.point(i)
        }

        fn speed_bound(&self) -> f64 {
            self.0.speed_bound() / 10.
        }

        fn accel_bound(&self) -> f64 {
            self.0.accel_bound() / 10.
        }

        fn noise_bound(&self) -> f64 {
            self.0.noise_bound()
        }

        fn span_bounds(
            &mut self,
            first: usize,
            last: usize,
        ) -> Option<SpanBounds> {
            self.0.span_bounds(first, last).map(|bounds| SpanBounds {
                speed: bounds.speed / 10.,
                accel: bounds.accel / 10.,
                noise: bounds.noise,
            })
        }
    }

    fn splitmix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// The location `r` radians from `ll`, at azimuth `az`.
    fn offset(ll: LatLng, r: f64, az: f64) -> LatLng {
        let (lat, lng) = (ll.lat_radians(), ll.lng_radians());
        let (sin_lat, cos_lat) = sin_cos(lat);
        let (sin_r, cos_r) = sin_cos(r);
        let (sin_az, cos_az) = sin_cos(az);
        let lat2 =
            libm::asin(mul_add(sin_lat, cos_r, cos_lat * sin_r * cos_az));
        let lng2 = lng
            + libm::atan2(
                sin_az * sin_r * cos_lat,
                mul_add(-sin_lat, libm::sin(lat2), cos_r),
            );
        LatLng::from_radians(lat2, lng2).expect("finite coordinate")
    }

    /// Kind of path in the equivalence run.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum EquivKind {
        Parallel,
        Segment,
        SegmentPerSpan,
        Jittered,
    }

    /// One randomly generated row of the equivalence run.
    #[derive(Debug, Clone)]
    struct EquivRow {
        kind: EquivKind,
        start: LatLng,
        /// Step (radians per sample) along the path.
        step: f64,
        /// Heading (radians, from north) for segments.
        heading: f64,
        sigma: f64,
        width: usize,
        resolutions: Vec<Resolution>,
    }

    impl EquivRow {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            clippy::cast_sign_loss,
            reason = "small positive values"
        )]
        fn generate(seed: u64, row: u64, specials: &[LatLng]) -> Self {
            let mut next = rng(splitmix(seed ^ splitmix(row)));
            let tau = core::f64::consts::TAU;
            // Start: uniform, or close to a face center, a pentagon (icosahedron
            // vertex: three faces meet), the antimeridian or a pole.
            let uniform = LatLng::from(Vec3d::new(
                mul_add(2., next(), -1.),
                mul_add(2., next(), -1.),
                mul_add(2., next(), -1.),
            ));
            let pick = next();
            let near = 10_f64.powf(mul_add(7., next(), -8.)); // 1e-8 .. 0.1 rad
            let start = if pick < 0.4 {
                uniform
            } else if pick < 0.95 {
                let anchor =
                    specials[(next() * specials.len() as f64) as usize];
                offset(anchor, near, tau * next())
            } else {
                LatLng::from_radians(
                    uniform.lat_radians(),
                    core::f64::consts::PI - near * mul_add(2., next(), -1.),
                )
                .expect("finite")
            };
            // 1 to 3 resolutions; steps from 1/300 to 3 edges of the finest.
            let mut resolutions = (0..=((next() * 3.) as usize))
                .map(|_| {
                    Resolution::try_from((next() * 16.) as u8)
                        .expect("in range")
                })
                .collect::<Vec<_>>();
            resolutions.dedup();
            let finest = resolutions.iter().copied().max().expect("not empty");
            let edge = 0.1738 / SQRT7_POWERS[usize::from(finest)];
            let step = edge * 10_f64.powf(mul_add(3., next(), -2.5));
            let width = 10_f64.powf(mul_add(1.7, next(), 1.)) as usize; // 10..500
            let kind = match (next() * 10.) as u8 {
                0..=3 => EquivKind::Parallel,
                4..=5 => EquivKind::Segment,
                6..=7 => EquivKind::SegmentPerSpan,
                _ => EquivKind::Jittered,
            };
            Self {
                kind,
                start,
                step,
                heading: tau * next(),
                sigma: step * 10_f64.powf(mul_add(3., next(), -5.)),
                width,
                resolutions,
            }
        }

        fn parallel(&self) -> GeographicRow {
            // Longitude step for an angular step `step` along the parallel.
            let lat = self.start.lat_radians();
            let dlng = self.step / cos(lat).max(1e-3);
            GeographicRow::new(
                lat.to_degrees(),
                self.start.lng(),
                dlng.to_degrees(),
            )
        }

        /// Runs the row, returns the stats, the number of checked pairs and the
        /// first mismatch (column, resolution, got, expected) with their count.
        fn check(
            &self,
            control: bool,
            runs: &mut Vec<(usize, Range<usize>, CellIndex)>,
        ) -> (RowStats, usize, usize, Option<String>) {
            match self.kind {
                EquivKind::Parallel => {
                    self.check_path(self.parallel(), control, runs)
                }
                EquivKind::Jittered => self.check_path(
                    JitteredRow {
                        row: self.parallel(),
                        sigma: self.sigma,
                        reported: self.sigma,
                        seed: self.width as u64,
                    },
                    control,
                    runs,
                ),
                EquivKind::Segment | EquivKind::SegmentPerSpan => {
                    #[expect(clippy::cast_precision_loss, reason = "≤ 500")]
                    let length = self.step * self.width as f64;
                    let end = offset(self.start, length, self.heading);
                    // Keep the longitude difference below π (no wrap-around).
                    let mut end_lng = end.lng_radians();
                    let tau = core::f64::consts::TAU;
                    while end_lng - self.start.lng_radians()
                        > core::f64::consts::PI
                    {
                        end_lng -= tau;
                    }
                    while self.start.lng_radians() - end_lng
                        > core::f64::consts::PI
                    {
                        end_lng += tau;
                    }
                    let end = LatLng::from_radians(end.lat_radians(), end_lng)
                        .expect("finite");
                    let path =
                        LatLngSegment::new(self.start, end, self.width as u64);
                    if self.kind == EquivKind::SegmentPerSpan {
                        self.check_path(path.per_span(), control, runs)
                    } else {
                        self.check_path(path, control, runs)
                    }
                }
            }
        }

        fn check_path(
            &self,
            path: impl RowPath,
            control: bool,
            runs: &mut Vec<(usize, Range<usize>, CellIndex)>,
        ) -> (RowStats, usize, usize, Option<String>) {
            if control {
                self.check_inner(Understated(path), runs)
            } else {
                self.check_inner(path, runs)
            }
        }

        fn check_inner(
            &self,
            mut path: impl RowPath,
            runs: &mut Vec<(usize, Range<usize>, CellIndex)>,
        ) -> (RowStats, usize, usize, Option<String>) {
            runs.clear();
            let stats = LatLngIndexer::new().index_row(
                &mut path,
                0..self.width,
                &self.resolutions,
                |i, cols, cell| runs.push((i, cols, cell)),
            );
            // Runs must tile `0..width` at each resolution, in column order.
            let mut next_col = vec![0; self.resolutions.len()];
            let mut checked = 0;
            let mut mismatches = 0;
            let mut first = None;
            for (i, cols, cell) in runs.iter() {
                assert_eq!(
                    cols.start, next_col[*i],
                    "{self:?}: runs out of order"
                );
                assert!(!cols.is_empty(), "{self:?}: empty run");
                next_col[*i] = cols.end;
                let res = self.resolutions[*i];
                for col in cols.clone() {
                    let expected = path.point(col).to_cell(res);
                    checked += 1;
                    if expected != *cell {
                        mismatches += 1;
                        first.get_or_insert_with(|| {
                            std::format!(
                                "column {col} res {res}: {cell} instead of {expected}"
                            )
                        });
                    }
                }
            }
            assert!(
                next_col.iter().all(|&end| end == self.width),
                "{self:?}: runs don't cover the row"
            );
            (stats, checked, mismatches, first)
        }
    }

    #[test]
    #[ignore = "long: run with --release, see the comment above"]
    #[expect(clippy::cast_precision_loss, reason = "statistics")]
    #[expect(clippy::too_many_lines, reason = "one self-contained driver")]
    fn index_row_equivalence() {
        use std::sync::{
            Mutex,
            atomic::{AtomicU64, Ordering},
        };

        const CHUNK: u64 = 10_000;

        let var = |name: &str| std::env::var(name).ok();
        let seed = var("H3O_EQUIV_SEED").map_or(1, |v| v.parse().unwrap());
        let control = var("H3O_EQUIV_CONTROL").is_some_and(|v| v == "1");
        let specials = (0..20)
            .map(|f| {
                let c = face::CENTER_POINT[f];
                LatLng::from(Vec3d::new(c.x, c.y, c.z))
            })
            .chain(
                CellIndex::base_cells()
                    .filter(|cell| cell.is_pentagon())
                    .map(LatLng::from),
            )
            .chain([
                LatLng::new(90., 0.).unwrap(),
                LatLng::new(-90., 0.).unwrap(),
            ])
            .collect::<Vec<_>>();

        if let Some(row) = var("H3O_EQUIV_ROW") {
            let row = EquivRow::generate(seed, row.parse().unwrap(), &specials);
            let result = row.check(control, &mut Vec::new());
            std::println!("{row:#?}\n{result:?}");
            return;
        }

        let rows: u64 =
            var("H3O_EQUIV_ROWS").map_or(100_000_000, |v| v.parse().unwrap());
        let threads = var("H3O_EQUIV_THREADS").map_or_else(
            || std::thread::available_parallelism().map_or(1, usize::from),
            |v| v.parse().unwrap(),
        );
        let cursor = AtomicU64::new(0);
        let done = AtomicU64::new(0);
        let total = Mutex::new((RowStats::default(), 0_u64, 0_u64, Vec::new()));
        let started = std::time::Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    let mut runs = Vec::new();
                    loop {
                        let first = cursor.fetch_add(CHUNK, Ordering::Relaxed);
                        if first >= rows {
                            break;
                        }
                        let mut stats = RowStats::default();
                        let (mut checked, mut mismatches) = (0, 0);
                        let mut failures = Vec::new();
                        for row in first..(first + CHUNK).min(rows) {
                            let equiv =
                                EquivRow::generate(seed, row, &specials);
                            let (s, c, m, f) = equiv.check(control, &mut runs);
                            stats.samples += s.samples;
                            stats.fallbacks += s.fallbacks;
                            stats.decisions += s.decisions;
                            checked += c as u64;
                            mismatches += m as u64;
                            if let Some(f) = f {
                                failures.push((row, f));
                            }
                        }
                        let mut total = total.lock().unwrap();
                        total.0.samples += stats.samples;
                        total.0.fallbacks += stats.fallbacks;
                        total.0.decisions += stats.decisions;
                        total.1 += checked;
                        total.2 += mismatches;
                        for failure in failures {
                            if total.3.len() < 20 {
                                std::eprintln!(
                                    "row {}: {}",
                                    failure.0,
                                    failure.1
                                );
                            }
                            total.3.push(failure.0);
                        }
                        drop(total);
                        let finished =
                            done.fetch_add(CHUNK, Ordering::Relaxed) + CHUNK;
                        if finished % (rows / 100).max(CHUNK) < CHUNK {
                            std::eprintln!(
                                "{:5.1}% ({finished} rows, {:.0} s)",
                                100. * finished.min(rows) as f64 / rows as f64,
                                started.elapsed().as_secs_f64()
                            );
                        }
                    }
                });
            }
        });
        let (stats, checked, mismatches, failed_rows) =
            total.into_inner().unwrap();
        std::println!(
            "seed {seed}, control {control}: {rows} rows, {checked} (location, \
             resolution) pairs checked against to_cell, {mismatches} mismatches \
             in {} rows; {} samples, {} fallbacks, {} decisions; {:.0} s",
            failed_rows.len(),
            stats.samples,
            stats.fallbacks,
            stats.decisions,
            started.elapsed().as_secs_f64()
        );
        if control {
            assert!(mismatches > 0, "understated bounds never changed a cell");
        } else {
            assert_eq!(mismatches, 0, "rows {failed_rows:?}");
        }
    }
}

// One indexer reused across rows, as raster callers do: cells remembered from
// earlier rows (other faces, resolutions and sample paths, and slots reused by
// colliding addresses) must never leak into later ones.
#[test]
#[expect(clippy::needless_range_loop, reason = "column-major check")]
fn reused_indexer_matches_to_cell() {
    let mut next = rng(7);
    let mut indexer = LatLngIndexer::new();
    let sets: [&[Resolution]; 4] = [
        &[Resolution::Nine],
        &[Resolution::Five, Resolution::Nine],
        &[Resolution::Twelve, Resolution::Seven, Resolution::Zero],
        &[Resolution::Fifteen],
    ];
    for round in 0..400 {
        let resolutions = sets[round % sets.len()];
        let lat = mul_add(170., next(), -85.);
        let lng0 = mul_add(360., next(), -180.);
        // Several nearby rows (like the sample paths of a raster row), then
        // jump elsewhere.
        let step = mul_add(1e-3, next(), 1e-5);
        let width = 300;
        for offset in [0., 1e-4, 2e-4] {
            let mut path = GeographicRow::new(lat + offset, lng0, step);
            let mut cells =
                vec![vec![None::<CellIndex>; width]; resolutions.len()];
            indexer.index_row(
                &mut path,
                0..width,
                resolutions,
                |index, cols, cell| {
                    for col in cols {
                        assert!(
                            cells[index][col].is_none(),
                            "column {col} twice"
                        );
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
                        "round {round} lat {lat} col {col} res {res:?}"
                    );
                }
            }
        }
    }
}

// The same `IJK` coordinates on every face and at every resolution: the 512
// slots of the recent-cell table then collide, and a hit must still match the
// face and resolution, not only the coordinates.
#[test]
fn recent_cells_key_on_face_and_resolution() {
    let mut indexer = LatLngIndexer::new();
    let coords = [
        CoordIJK::new(0, 0, 0),
        CoordIJK::new(1, 0, 0),
        CoordIJK::new(0, 1, 0),
        CoordIJK::new(0, 0, 1),
        CoordIJK::new(1, 1, 0),
    ];
    for _ in 0..2 {
        for coord in coords {
            for face in 0..20 {
                for res in
                    Resolution::range(Resolution::One, Resolution::Fifteen)
                {
                    let fijk = FaceIJK::new(Face::new_unchecked(face), coord);
                    let mut cache = EncodeCache::default();
                    assert_eq!(
                        indexer.encode_recent(fijk, res, &mut cache),
                        fijk.to_cell(res),
                        "{fijk:?} at {res:?}"
                    );
                }
            }
        }
    }
}

/// `(U, V)` of face `f`: the resolution 0 gnomonic position of `p` is
/// `(U·p, V·p) / (c·p)`, from `FaceProjection`'s polar form.
fn gnomonic_axes(f: usize) -> (Vec3d, Vec3d) {
    let (north, east) = face::CENTER_POINT[f].tangent_basis();
    let (sin_a, cos_a) = sin_cos(face::AXES_AZ_RADS_CII[f]);
    let k = INV_RES0_U_GNOMONIC;
    let u = Vec3d::new(
        k * mul_add(cos_a, north.x, sin_a * east.x),
        k * mul_add(cos_a, north.y, sin_a * east.y),
        k * mul_add(cos_a, north.z, sin_a * east.z),
    );
    let v = Vec3d::new(
        k * mul_add(sin_a, north.x, -cos_a * east.x),
        k * mul_add(sin_a, north.y, -cos_a * east.y),
        k * mul_add(sin_a, north.z, -cos_a * east.z),
    );
    (u, v)
}

#[test]
fn gnomonic_axes_match_construction() {
    for (f, &(u, v)) in GNOMONIC_AXES.iter().enumerate() {
        let (eu, ev) = gnomonic_axes(f);
        for (got, want) in [
            (u.x, eu.x),
            (u.y, eu.y),
            (u.z, eu.z),
            (v.x, ev.x),
            (v.y, ev.y),
            (v.z, ev.z),
        ] {
            assert!(
                abs(got - want) <= 4. * f64::EPSILON * INV_RES0_U_GNOMONIC,
                "face {f}: {got} vs {want}"
            );
        }
    }
}

// Samples skip `FaceProjection`'s trigonometry: same face whenever the face
// margin clears `face_ok`'s allowance, the same position within a few dozen
// ε, and a radius that never exceeds the angular distance to the face
// center.
#[test]
fn samples_match_face_projection() {
    let mut next = rng(11);
    let mut worst = 0_f64;
    for i in 0..400_000 {
        // Uniform on the sphere, plus points near face centers.
        let ll = if i % 4 == 0 {
            let c = face::CENTER_POINT[i % 20];
            let e = 1e-3 * next();
            let ll = LatLng::from(Vec3d::new(c.x, c.y, c.z));
            LatLng::from_radians(
                ll.lat_radians() + e * mul_add(2., next(), -1.),
                ll.lng_radians() + e * mul_add(2., next(), -1.),
            )
            .unwrap()
        } else {
            LatLng::from_radians(
                libm::asin(mul_add(2., next(), -1.)),
                core::f64::consts::PI * mul_add(2., next(), -1.),
            )
            .unwrap()
        };
        let sample = Sample::new(ll);
        let p = Vec3d::from(ll);
        let (face, distance) = p.closest_face();
        if sample.face_margin > 1e-13 {
            assert_eq!(sample.face, face, "{ll}");
        }
        if sample.face != face {
            continue;
        }
        let r = libm::acos(mul_add(distance, -0.5, 1.));
        // `acos`'s own error near the face center is ~ε / r.
        assert!(
            sample.r <= r + f64::EPSILON / r.max(1e-300),
            "{ll}: {} > {r}",
            sample.r
        );
        assert!(
            sample.r
                >= libm::sin(r) - 1e-13 - 2. * f64::EPSILON / r.max(1e-300),
            "{ll}: {} ≪ {r}",
            sample.r
        );
        if r > 1e-4 {
            let q = FaceProjection::new(p, face, distance).res0_xy();
            let err = abs(sample.q.x - q.x).max(abs(sample.q.y - q.y));
            if r > 0.1 {
                worst = worst.max(err / f64::EPSILON);
            }
            // The reference has `to_cell`'s ~0.65 k ε / r error (see
            // `TO_CELL_NOISE`) on top of the sample's own.
            let allowed = mul_add(
                INV_RES0_U_GNOMONIC,
                f64::EPSILON / r,
                64. * f64::EPSILON,
            );
            assert!(err <= allowed, "{ll}: {:?} vs {q:?}", sample.q);
        }
    }
    // Where the reference is accurate, both agree to ~10ε (measured).
    assert!(worst <= 32., "worst difference {worst} ε");
}
