use super::LatLngSegment;
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
