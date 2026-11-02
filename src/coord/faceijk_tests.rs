use super::*;
use crate::{CellIndex, LatLng, math::mul_add};

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

/// Encodes `ll` through `cache` and checks it against the uncached encoding.
fn check(ll: LatLng, resolution: Resolution, cache: &mut EncodeCache) {
    let fijk = FaceIJK::from_vec3d(Vec3d::from(ll), resolution);
    assert_eq!(
        fijk.to_cell_cached(resolution, cache),
        fijk.to_cell(resolution),
        "{ll} at {resolution:?} ({fijk:?})"
    );
}

// Random walks at every resolution: mostly steps of about one cell, so that
// consecutive cells share most of their ancestors, with occasional jumps
// anywhere (other faces, other base cells), and walks starting next to every
// pentagon and around a face center.
#[test]
fn cached_encoding_matches_to_cell() {
    let mut next = rng(3);
    let mut starts = CellIndex::base_cells()
        .filter(|cell| cell.is_pentagon())
        .map(LatLng::from)
        .collect::<Vec<_>>();
    starts.push(LatLng::from(Vec3d::new(
        face::CENTER_POINT[7].x,
        face::CENTER_POINT[7].y,
        face::CENTER_POINT[7].z,
    )));
    for _ in 0..8 {
        let lat = mul_add(2., next(), -1.).asin().to_degrees();
        let lng = mul_add(360., next(), -180.);
        starts.push(LatLng::new(lat, lng).expect("valid"));
    }

    for resolution in Resolution::range(Resolution::Zero, Resolution::Fifteen) {
        // About one cell, in degrees of latitude.
        let step = resolution.edge_length_km() / 111.;
        for &start in &starts {
            let mut cache = EncodeCache::default();
            let (mut lat, mut lng) = (start.lat(), start.lng());
            for _ in 0..1500 {
                if next() < 0.02 {
                    lat = mul_add(2., next(), -1.).asin().to_degrees();
                    lng = mul_add(360., next(), -180.);
                } else {
                    let angle = next() * core::f64::consts::TAU;
                    let distance = 1.5 * step * next();
                    lat = (lat + distance * angle.sin()).clamp(-89.9, 89.9);
                    lng += distance * angle.cos() / lat.to_radians().cos();
                }
                check(
                    LatLng::new(lat, lng).expect("valid"),
                    resolution,
                    &mut cache,
                );
            }
        }
    }
}
