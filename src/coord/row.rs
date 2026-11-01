//! Experimental: exact run-length indexing of raster rows.
//!
//! Along a row of a raster, consecutive pixel centers move smoothly over the
//! sphere. Instead of projecting every pixel, the indexer projects a few
//! samples exactly, linearly interpolates the hex-plane coordinates in
//! between, and only trusts the interpolated position when it is farther from
//! any hexagon edge than a proven bound on the interpolation error plus the
//! floating-point noise of `to_cell` itself. Pixels that don't clear the
//! margin are indexed with `to_cell`, so every pixel gets exactly the cell
//! `to_cell` would return.
//!
//! The cost is then proportional to the number of cell crossings along the
//! row instead of the number of pixels.

use super::{
    AP7_ROT_RADS, CoordIJK, FaceIJK, FaceProjection, INV_RES0_U_GNOMONIC,
    LatLng, LatLngIndexer, SQRT3_2, SQRT7_POWERS, Vec2d, Vec3d,
};
use crate::{
    CellIndex, Face, Resolution, face,
    math::{abs, acos, cos, mul_add, sin_cos, sqrt},
};
use alloc::vec::Vec;
use core::ops::Range;

/// A sequence of locations sampled along a smooth path, such as the pixel
/// centers of a raster row.
pub trait RowPath {
    /// Returns location `i`.
    ///
    /// This must be computed exactly the way the caller would compute it
    /// before calling [`LatLng::to_cell`]: the indexer guarantees
    /// `cell(i) == self.point(i).to_cell(res)`. It is called for indexes in
    /// `cols.start..=cols.end` (the end is used as a span endpoint).
    fn point(&mut self, i: usize) -> LatLng;

    /// An upper bound on `|dP/di|`, where `P(i)` is the unit vector of
    /// location `i` (radians per sample).
    fn speed_bound(&self) -> f64;

    /// An upper bound on `|d²P/di²|` (radians per sample²).
    fn accel_bound(&self) -> f64;
}

/// A row of a geographic (equirectangular) raster: constant latitude,
/// longitude growing linearly with the column.
///
/// Location `i` is `LatLng::new(lat, step.mul_add(i as f64, lng0))`, in
/// degrees.
#[derive(Debug, Clone, Copy)]
pub struct GeographicRow {
    lat: f64,
    lng0: f64,
    step: f64,
    speed: f64,
    accel: f64,
}

impl GeographicRow {
    /// Initializes a row at latitude `lat`, starting at longitude `lng0` and
    /// advancing by `step` degrees per column.
    #[must_use]
    pub fn new(lat: f64, lng0: f64, step: f64) -> Self {
        // A parallel is a circle of radius cos(lat): |P'| = cos(lat) |dλ| and
        // |P''| = cos(lat) dλ². Slightly inflated to absorb rounding.
        let dlambda = abs(step.to_radians());
        let radius = mul_add(abs(cos(lat.to_radians())), 1. + 1e-9, 1e-15);
        Self {
            lat,
            lng0,
            step,
            speed: radius * dlambda,
            accel: radius * dlambda * dlambda,
        }
    }
}

impl RowPath for GeographicRow {
    #[expect(clippy::cast_precision_loss, reason = "raster column index")]
    fn point(&mut self, i: usize) -> LatLng {
        LatLng::new(self.lat, mul_add(self.step, i as f64, self.lng0))
            .expect("finite coordinate")
    }

    fn speed_bound(&self) -> f64 {
        self.speed
    }

    fn accel_bound(&self) -> f64 {
        self.accel
    }
}

/// `count` samples along a segment that is linear in latitude and longitude
/// (radians), such as a densified polygon edge.
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
pub struct LatLngSegment {
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
    #[must_use]
    pub fn new(start: LatLng, end: LatLng, count: u64) -> Self {
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

// -----------------------------------------------------------------------------

/// Lower bound of `P · n` for any point whose closest face center is `n`: the
/// farthest such point is an icosahedron vertex, 37.38° away (cos = 0.7947).
const W_MIN: f64 = 0.79;

/// Estimated bound of `to_cell`'s own floating-point error on the resolution 0
/// hex-plane position, as a multiple of `f64::EPSILON * (1 + 1/r)`, where `r`
/// is the angular distance to the face center (`acos` near 1 and the azimuth
/// are ill-conditioned close to the face center). A handful of libm calls with
/// ≤ 2 ulp error each are involved; this is deliberately loose.
const FLOAT_NOISE: f64 = 256. * f64::EPSILON;

/// Margin, in hex units at the finest resolution, that spans are sized for.
/// About `4 * TARGET_MARGIN` of the samples fall back to `to_cell`.
const TARGET_MARGIN: f64 = 0.002;

/// Longest span between two exact samples.
const MAX_SPAN: usize = 1024;
#[expect(clippy::cast_precision_loss, reason = "small constant")]
const MAX_SPAN_F64: f64 = MAX_SPAN as f64;

/// Spans closer than this (radians) to a face center are split down to
/// per-pixel indexing.
const MIN_FACE_CENTER_DISTANCE: f64 = 1e-6;

/// An exactly projected sample.
#[derive(Debug, Clone, Copy)]
struct Sample {
    face: Face,
    /// Resolution 0 hex-plane coordinates on `face`.
    q: Vec2d,
    /// Angular distance to the face center.
    r: f64,
    /// `min_g (P·c_face - P·c_g)`: how far the sample is from switching face.
    face_margin: f64,
}

impl Sample {
    fn new(ll: LatLng) -> Self {
        let p = Vec3d::from(ll);
        let (face, distance) = p.closest_face();
        let q = FaceProjection::new(p, face, distance).res0_xy();
        let r = acos(mul_add(distance, -0.5, 1.));
        let center = face::CENTER_POINT[usize::from(face)];
        let own = dot(&p, &center);
        let face_margin = face::CENTER_POINT
            .iter()
            .enumerate()
            .filter(|&(g, _)| g != usize::from(face))
            .map(|(_, c)| own - dot(&p, c))
            .fold(f64::INFINITY, f64::min);
        Self {
            face,
            q,
            r,
            face_margin,
        }
    }
}

fn dot(a: &Vec3d, b: &Vec3d) -> f64 {
    mul_add(a.x, b.x, mul_add(a.y, b.y, a.z * b.z))
}

/// Per-resolution precomputed transform from the resolution 0 plane, and the
/// run being built.
struct Level {
    resolution: Resolution,
    scale: f64,
    /// Rotation applied after scaling (Class III).
    cos: f64,
    sin: f64,
    run: Option<(usize, CellIndex)>,
}

impl Level {
    fn new(resolution: Resolution) -> Self {
        let scale = SQRT7_POWERS[usize::from(resolution)];
        let (sin, cos) = if resolution.is_class3() {
            sin_cos(-AP7_ROT_RADS)
        } else {
            (0., 1.)
        };
        Self {
            resolution,
            scale,
            cos,
            sin,
            run: None,
        }
    }

    /// Maps resolution 0 plane coordinates to this resolution.
    fn map(&self, x: f64, y: f64) -> Vec2d {
        let (x, y) = (x * self.scale, y * self.scale);
        Vec2d::new(
            mul_add(x, self.cos, -y * self.sin),
            mul_add(x, self.sin, y * self.cos),
        )
    }

    fn push(
        &mut self,
        index: usize,
        col: usize,
        cell: CellIndex,
        emit: &mut impl FnMut(usize, Range<usize>, CellIndex),
    ) {
        match self.run {
            Some((_, current)) if current == cell => {}
            Some((start, current)) => {
                emit(index, start..col, current);
                self.run = Some((col, cell));
            }
            None => self.run = Some((col, cell)),
        }
    }
}

/// Distance from `q` to the boundary of the hexagon `ijk` (unit spacing
/// between hex centers, so the apothem is 0.5); negative if outside.
fn hex_margin(q: Vec2d, ijk: CoordIJK) -> f64 {
    let center = Vec2d::from(ijk);
    let (ox, oy) = (q.x - center.x, q.y - center.y);
    let a = abs(ox);
    let b = abs(mul_add(0.5, ox, SQRT3_2 * oy));
    let c = abs(mul_add(-0.5, ox, SQRT3_2 * oy));
    0.5 - a.max(b).max(c)
}

/// Largest real `k ≥ 0` such that `(o + k d)·n ≤ limit` for each of the 6 hexagon
/// edge normals `±n`, given that it holds for `k = 0`.
fn exit_steps(o: Vec2d, d: Vec2d, limit: f64) -> f64 {
    let normals = [(1., 0.), (0.5, SQRT3_2), (-0.5, SQRT3_2)];
    let mut k = f64::INFINITY;
    for (nx, ny) in normals {
        let on = mul_add(o.x, nx, o.y * ny);
        let dn = mul_add(d.x, nx, d.y * ny);
        if dn > 0. {
            k = k.min((limit - on) / dn);
        } else if dn < 0. {
            k = k.min((limit + on) / -dn);
        }
    }
    // Callers truncate to an integer, which floors a non-negative value.
    k.max(0.)
}

/// Statistics about one `index_row` call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RowStats {
    /// Number of exactly projected span endpoints.
    pub samples: usize,
    /// Number of (pixel, resolution) pairs indexed with `to_cell`.
    pub fallbacks: usize,
    /// Number of (pixel, resolution) pairs decided by interpolation, each
    /// possibly covering several pixels.
    pub decisions: usize,
}

struct Row<'a, P, E> {
    indexer: &'a mut LatLngIndexer,
    path: &'a mut P,
    emit: E,
    levels: Vec<Level>,
    speed: f64,
    accel: f64,
    /// `max |Q''|` of the gnomonic plane position along the path, per sample²
    /// (tangent-plane units).
    curvature: f64,
    /// Last exact location, for fallbacks at several resolutions.
    last_point: Option<(usize, LatLng)>,
    stats: RowStats,
}

/// See [`LatLngIndexer::index_row`].
pub(super) fn index_row(
    indexer: &mut LatLngIndexer,
    path: &mut impl RowPath,
    cols: Range<usize>,
    resolutions: &[Resolution],
    emit: impl FnMut(usize, Range<usize>, CellIndex),
) -> RowStats {
    if cols.is_empty() || resolutions.is_empty() {
        return RowStats::default();
    }
    let speed = path.speed_bound();
    let accel = path.accel_bound();
    // Q = P_perp / w with |P_perp| ≤ 1, w ≥ W_MIN, |P'|, |w'| ≤ V and
    // |P''|, |w''| ≤ A:
    //   |Q''| ≤ A/w + 2V²/w² + A/w² + 2V²/w³.
    let v2 = speed * speed;
    let curvature = accel / W_MIN
        + 2. * v2 / (W_MIN * W_MIN)
        + accel / (W_MIN * W_MIN)
        + 2. * v2 / (W_MIN * W_MIN * W_MIN);

    let finest = resolutions
        .iter()
        .map(|&res| SQRT7_POWERS[usize::from(res)])
        .fold(0., f64::max);
    // Interpolation error over `h` samples: INV_RES0_U * h²/8 * |Q''|.
    let span = if curvature > 0. {
        let h = sqrt(
            8. * TARGET_MARGIN / (finest * INV_RES0_U_GNOMONIC * curvature),
        );
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped below"
        )]
        let h = h.min(MAX_SPAN_F64) as usize;
        h.max(1)
    } else {
        MAX_SPAN
    };

    let mut row = Row {
        indexer,
        path,
        emit,
        levels: resolutions.iter().copied().map(Level::new).collect(),
        speed,
        accel,
        curvature,
        last_point: None,
        stats: RowStats::default(),
    };

    let mut a = cols.start;
    let mut sa = row.sample(a);
    while a < cols.end {
        let b = (a + span).min(cols.end);
        let sb = row.sample(b);
        row.span(a, sa, b, sb);
        a = b;
        sa = sb;
    }
    for (index, level) in row.levels.iter_mut().enumerate() {
        if let Some((start, cell)) = level.run.take() {
            (row.emit)(index, start..cols.end, cell);
        }
    }
    row.stats
}

impl<P: RowPath, E: FnMut(usize, Range<usize>, CellIndex)> Row<'_, P, E> {
    fn point(&mut self, i: usize) -> LatLng {
        match self.last_point {
            Some((j, ll)) if j == i => ll,
            _ => {
                let ll = self.path.point(i);
                self.last_point = Some((i, ll));
                ll
            }
        }
    }

    fn sample(&mut self, i: usize) -> Sample {
        self.stats.samples += 1;
        let ll = self.point(i);
        Sample::new(ll)
    }

    /// Indexes columns `a..b`, given exact samples at `a` and `b`.
    #[expect(clippy::cast_precision_loss, reason = "span length ≤ MAX_SPAN")]
    #[expect(clippy::many_single_char_names, reason = "math notation")]
    fn span(&mut self, a: usize, sa: Sample, b: usize, sb: Sample) {
        let h = (b - a) as f64;

        // Every location of the span must have `sa.face` as closest face:
        // `s_g = P·(c_f - c_g)` is linear in P, so its deviation from the
        // chord is ≤ h²/8 * |c_f - c_g| * A ≤ h²/4 * A.
        let face_ok = sa.face == sb.face
            && sa.face_margin.min(sb.face_margin)
                > mul_add(h * h * 0.25, self.accel, 1e-13);
        // ... and stay away from the face center, where `to_cell` is noisy.
        let r_lo = mul_add(-0.5 * h, self.speed, sa.r.min(sb.r));
        if !face_ok || r_lo < MIN_FACE_CENTER_DISTANCE {
            if b - a >= 2 {
                let m = a + (b - a) / 2;
                let sm = self.sample(m);
                self.span(a, sa, m, sm);
                self.span(m, sm, b, sb);
            } else {
                self.exact(a);
            }
            return;
        }

        // Bound on |true - interpolated| + 2 * float noise, at resolution 0.
        let interpolation =
            INV_RES0_U_GNOMONIC * h * h * 0.125 * self.curvature;
        let noise = FLOAT_NOISE * (1. + 1. / r_lo);
        let error0 = mul_add(2., noise, interpolation);

        let (dx, dy) = ((sb.q.x - sa.q.x) / h, (sb.q.y - sa.q.y) / h);
        for index in 0..self.levels.len() {
            let level = &self.levels[index];
            let resolution = level.resolution;
            // Per-column displacement at this resolution.
            let d = level.map(dx, dy);
            let travel = h * sqrt(mul_add(d.x, d.x, d.y * d.y));
            let margin = level.scale * error0;

            let mut col = a;
            while col < b {
                let t = (col - a) as f64;
                let q = self.levels[index]
                    .map(mul_add(t, dx, sa.q.x), mul_add(t, dy, sa.q.y));
                let ijk = CoordIJK::from(q);
                // Margin, plus rounding of the interpolation and plane mapping
                // for any column of the span.
                let required = mul_add(
                    16. * f64::EPSILON,
                    1. + abs(q.x) + abs(q.y) + travel,
                    margin,
                );
                let center = Vec2d::from(ijk);
                let o = Vec2d::new(q.x - center.x, q.y - center.y);
                if hex_margin(q, ijk) > required {
                    self.stats.decisions += 1;
                    let cell = self
                        .indexer
                        .encode(FaceIJK::new(sa.face, ijk), resolution);
                    self.levels[index].push(index, col, cell, &mut self.emit);
                    // Columns `col + k` stay in the hexagon with margin as
                    // long as `(o + k d)·n ≤ 0.5 - required` for each of the
                    // 6 edge normals ±n.
                    let skip = exit_steps(o, d, 0.5 - required).min(h);
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "non-negative, clamped to the span"
                    )]
                    let skip = skip as usize;
                    col += skip + 1;
                } else {
                    self.stats.fallbacks += 1;
                    let ll = self.point(col);
                    let cell = self.indexer.to_cell(ll, resolution);
                    self.levels[index].push(index, col, cell, &mut self.emit);
                    col += 1;
                }
            }
        }
    }

    /// Indexes column `col` with `to_cell` at every resolution.
    fn exact(&mut self, col: usize) {
        let ll = self.point(col);
        for index in 0..self.levels.len() {
            self.stats.fallbacks += 1;
            let resolution = self.levels[index].resolution;
            let cell = self.indexer.to_cell(ll, resolution);
            self.levels[index].push(index, col, cell, &mut self.emit);
        }
    }
}

#[cfg(test)]
#[path = "./row_tests.rs"]
mod tests;
