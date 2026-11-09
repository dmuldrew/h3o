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
    AP7_ROT_RADS, CoordIJK, EncodeCache, FaceIJK, FaceProjection,
    INV_RES0_U_GNOMONIC, LatLng, LatLngIndexer, SQRT3_2, SQRT7_POWERS, Vec2d,
    Vec3d,
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
    /// location `i` on the smooth path (radians per sample).
    fn speed_bound(&self) -> f64;

    /// An upper bound on `|d²P/di²|` on the smooth path (radians per
    /// sample²).
    fn accel_bound(&self) -> f64;

    /// An upper bound on the angular distance (radians) between `point(i)`
    /// and the smooth path described by the speed and acceleration bounds.
    ///
    /// The indexer already allows `64 * f64::EPSILON` rad (≈ 1.4e-14 rad, or
    /// ~0.1 nm) on top of this, which covers the rounding of a handful of
    /// closed-form operations on coordinates in radians or degrees (e.g. a
    /// longitude computed as `lng0 + step * i`, ≈ 2 ulp of π): the default is
    /// 0. Coordinates computed numerically, such as an iterative inverse
    /// projection, or through a long chain of operations, deviate from the
    /// smooth path by more: report it here. The margin grows accordingly, so
    /// results stay exact; values above 1e-3 rad (or NaN/negative values)
    /// make every location use `to_cell`.
    fn noise_bound(&self) -> f64 {
        0.
    }

    /// Bounds for the locations `first..=last` only, or `None` to index them
    /// with [`LatLng::to_cell`].
    ///
    /// The speed, acceleration and noise bounds above must hold for the whole
    /// call. Paths whose motion varies a lot along the path can report tighter
    /// bounds per span here: spans then get a smaller error margin, can be
    /// longer, and fall back less. Returning `None` covers stretches where no
    /// bound exists, such as near a pole or a projection singularity, without
    /// giving up on the rest of the path.
    ///
    /// The returned bounds must hold for every location in `first..=last` and
    /// for the smooth path between them; the indexer may use them for any
    /// part of that range. It asks once per span, before computing the span's
    /// samples. The default returns the path-wide bounds.
    fn span_bounds(&mut self, first: usize, last: usize) -> Option<SpanBounds> {
        let _ = (first, last);
        Some(SpanBounds {
            speed: self.speed_bound(),
            accel: self.accel_bound(),
            noise: self.noise_bound(),
        })
    }
}

/// Bounds on how a [`RowPath`] moves over part of its locations (see
/// [`RowPath::span_bounds`]), with the same meaning as
/// [`RowPath::speed_bound`], [`RowPath::accel_bound`] and
/// [`RowPath::noise_bound`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpanBounds {
    /// An upper bound on `|dP/di|` (radians per sample).
    pub speed: f64,
    /// An upper bound on `|d²P/di²|` (radians per sample²).
    pub accel: f64,
    /// An upper bound on the distance between a location and the smooth path
    /// (radians).
    pub noise: f64,
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

// -----------------------------------------------------------------------------

// Constants that exactness depends on.

/// `cos` of the angular radius of a face: the farthest point whose closest
/// face center is `n` is an icosahedron vertex, `acos(√((5 + 2√5) / 15))` ≈
/// 37.38° away.
const COS_FACE_RADIUS: f64 = 0.794_654_472_291_766;

/// Lower bound of `P · n` used by the plane-error bounds, for points within
/// `MAX_NOISE` of a point whose closest face center is `n`.
const W_MIN: f64 = 0.79;

/// Largest supported `RowPath::noise_bound` (radians); larger noise falls
/// back to `to_cell`. A point within `MAX_NOISE` of a location whose closest
/// face center is `n` has `P · n ≥ COS_FACE_RADIUS - MAX_NOISE ≥ W_MIN`.
const MAX_NOISE: f64 = 1e-3;
const _: () = assert!(MAX_NOISE + ROUNDING_NOISE <= COS_FACE_RADIUS - W_MIN);

/// Bound on `to_cell`'s own floating-point error on the resolution 0
/// hex-plane position (and, scaled by `√7ʳ`, at resolution `r`), as a
/// multiple of `f64::EPSILON * (1 + 1/r)`, where `r` is the angular distance
/// to the face center.
///
/// The error is `a ε + b ε / r`:
/// - `b ε / r` comes from rounding `1 - d²/2` (≤ ε/4) before `acos`, whose
///   slope is `1 / sin r`: `b = INV_RES0_U_GNOMONIC / 4 ≈ 0.65`. Nothing else
///   is ill-conditioned near the face center: the azimuth's angle error
///   (~ε / sin r) is multiplied by the gnomonic radius (~tan r).
/// - `a ε` collects the rest (n-vector, `tan`, azimuth, `sin_cos`, Class III
///   rotation, scaling), each a few ulps amplified by at most
///   `PLANE_PER_RADIAN` ≈ 7.5: a worst-case sum is around `a ≈ 50`.
///
/// Measured against a 40-digit evaluation of the same projection on 100,000
/// points (uniform, and log-spaced down to 3e-7 rad from face centers), at
/// resolutions 0, 14 and 15 (`tools/to_cell_noise.py`): `a ≤ 3.75` and
/// `b = 0.65`. 256 is far above both, and still costs nothing: at resolution
/// 15 the resulting margin is ~5e-7 hex units against `TARGET_MARGIN`.
const TO_CELL_NOISE: f64 = 256. * f64::EPSILON;

/// Point noise (radians) allowed on top of `RowPath::noise_bound`, so that
/// closed-form paths can report 0: rounding of a few operations on
/// coordinates, e.g. ≈ 2 ulp of π for `lng0 + step * i` and the degree to
/// radian conversion.
const ROUNDING_NOISE: f64 = 64. * f64::EPSILON;

// Performance-only constants: the error margin of a span is computed from its
// actual length and distance to the face center, so any positive value keeps
// results exact.

/// Margin, in hex units at the finest resolution, that spans are sized for.
/// About `4 * TARGET_MARGIN` of the samples fall back to `to_cell`.
const TARGET_MARGIN: f64 = 0.002;

/// Longest span between two exact samples.
const MAX_SPAN: usize = 1024;
#[expect(clippy::cast_precision_loss, reason = "small constant")]
const MAX_SPAN_F64: f64 = MAX_SPAN as f64;

/// Spans closer than this (radians) to a face center are split down to
/// per-pixel indexing (the `1/r` term of `TO_CELL_NOISE` would leave little
/// room anyway).
const MIN_FACE_CENTER_DISTANCE: f64 = 1e-6;

/// Bound on the resolution 0 hex-plane displacement per radian of point
/// displacement: with `Q = P_perp / w`, `|δQ| ≤ |δP|/w + |P_perp| |δw|/w²`.
const PLANE_PER_RADIAN: f64 =
    INV_RES0_U_GNOMONIC * (1. / W_MIN + 1. / (W_MIN * W_MIN));

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
    /// Ancestors of the last cell encoded at this resolution, so that a new
    /// cell only recomputes the digits it doesn't share with it.
    encode_cache: EncodeCache,
    /// Likely hexagon of the next decision: the neighbour across the edge
    /// the previous run left through, or the hexagon itself where the next
    /// decision is the next span's first column. Only a guess: it is used
    /// when it contains the position with margin, and the position is
    /// rounded to its hexagon otherwise.
    hint: Option<FaceIJK>,
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
            encode_cache: EncodeCache::default(),
            hint: None,
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
///
/// # Hex rounding
///
/// Exactness needs `CoordIJK::from(Vec2d)` to return the hexagon containing
/// `to_cell`'s position whenever that position is inside it by more than the
/// rounding slack. (On the interpolated side it is checked: a wrong hexagon
/// would give a negative `hex_margin` and fall back.) In exact arithmetic it
/// returns the nearest lattice point:
///
/// - It works on `|x|, |y|` in axial coordinates `x2 = |y| / sin 60°`,
///   `x1 = |x| + x2 / 2`, where hexagon `(i, j, 0)` is centered at
///   `(x1, x2) = (i, j)`, and splits them into integer parts `(m1, m2)` and
///   fractional parts `(r1, r2)`. The nearest center is then one of the
///   corners `(0,0)`, `(1,0)`, `(0,1)`, `(1,1)` of the unit cell (relative to
///   `(m1, m2)`): the squared distance to `(i, j)` is
///   `(r1-i)² - (r1-i)(r2-j) + (r2-j)²`, and the corners' Voronoi cells
///   cover the unit cell.
/// - The bisectors between corners are `r2 = 2 r1 - 1` for (0,0)|(1,0),
///   `r2 = (1 + r1) / 2` for (0,0)|(0,1), `r1 + r2 = 1` for (0,0)|(1,1),
///   `r2 = r1 / 2` for (1,0)|(1,1) and `r2 = 2 r1` for (0,1)|(1,1);
///   (1,0) and (0,1) are not neighbors.
/// - `r1 < 1/3`: (1,0) would need `r2 < 2 r1 - 1 < 0`, and (1,1) would need
///   `r2 > 1 - r1 > 2/3 > 2 r1`: the code picks between (0,0) and (0,1) by
///   `r2 ≥ (1 + r1) / 2`.
/// - `r1 ≥ 2/3`: (0,0) would need `2 r1 - 1 < r2 < 1 - r1`, impossible, and
///   (0,1) `r2 > 2 r1 ≥ 4/3`: the code picks between (1,0) and (1,1) by
///   `r2 ≥ r1 / 2`.
/// - `1/3 ≤ r1 < 2/3`: `r2 < 1 - r1` selects the lower pair; there (0,1)
///   never wins since `1 - r1 ≤ (1 + r1) / 2`, and the code splits (0,0)
///   from (1,0) by `r2 ≤ 2 r1 - 1` (which is never true for `r1 < 1/2`).
///   Otherwise (1,0) never wins since `1 - r1 > r1 / 2`, and the code splits
///   (0,1) from (1,1) by `r2 < 2 r1` (always true for `r1 ≥ 1/2`).
/// - Folding back the signs applies the lattice reflections: `x → -x` maps
///   `(i, j)` to `(j - i, j)` and `y → -y` maps it to `(i - j, -j)`.
///
/// In floating point, the integer and fractional parts are exact, the
/// thresholds round by at most ε, and `x1, x2` by ~3 ulp of `|q|`. A move of
/// `(δ1, δ2)` in axial coordinates moves the point by at most `|δ1| + |δ2|`
/// relative to any bisector (the bisector distances are `|2 r1 - r2 - 1| / 2`
/// and similar), so the slack is about `4ε |q| + 2ε`, within the `16ε (1 + |q|)` term of the margin (itself
/// far below the scaled `TO_CELL_NOISE`).
fn hex_margin(q: Vec2d, ijk: CoordIJK) -> f64 {
    let center = Vec2d::from(ijk);
    let (ox, oy) = (q.x - center.x, q.y - center.y);
    let a = abs(ox);
    let b = abs(mul_add(0.5, ox, SQRT3_2 * oy));
    let c = abs(mul_add(-0.5, ox, SQRT3_2 * oy));
    0.5 - a.max(b).max(c)
}

/// The edges a position moving by `d` per column can leave a hexagon
/// through: for each pair of opposite edges, the outward normal facing `d`,
/// `1 / (d·n)`, and the offset to the hexagon across that edge.
#[derive(Debug, Clone, Copy)]
struct ExitEdges([(Vec2d, f64, CoordIJK); 3]);

impl ExitEdges {
    fn new(d: Vec2d) -> Self {
        // Edge normals, and the neighbours across the `+n` and `-n` edges.
        let normals = [
            (
                Vec2d::new(1., 0.),
                CoordIJK::new(1, 0, 0),
                CoordIJK::new(0, 1, 1),
            ),
            (
                Vec2d::new(0.5, SQRT3_2),
                CoordIJK::new(1, 1, 0),
                CoordIJK::new(0, 0, 1),
            ),
            (
                Vec2d::new(-0.5, SQRT3_2),
                CoordIJK::new(0, 1, 0),
                CoordIJK::new(1, 0, 1),
            ),
        ];
        Self(normals.map(|(n, plus, minus)| {
            let dn = mul_add(d.x, n.x, d.y * n.y);
            if dn < 0. {
                (Vec2d::new(-n.x, -n.y), -1. / dn, minus)
            } else {
                // `dn = ±0` gives an infinite (or NaN, never smaller) count.
                (n, 1. / abs(dn), plus)
            }
        }))
    }

    /// Largest real `k ≥ 0` such that `(o + k d)·n ≤ limit` for each of the
    /// 6 edge normals `n`, given that it holds for `k = 0`, and the offset to
    /// the hexagon across the edge that limits `k` (`None` if none does).
    ///
    /// Multiplying by the precomputed `1 / (d·n)` rounds once more than
    /// dividing: `k` is within ~2ε of its exact value relatively, so a skipped
    /// column passes `limit` by at most ~2ε·`travel` hex units, which the
    /// `16ε · travel` term of the span's `required` margin covers.
    #[expect(clippy::float_cmp, reason = "`k` is one of the `steps`")]
    fn exit_steps(&self, o: Vec2d, limit: f64) -> (f64, Option<CoordIJK>) {
        let steps = self.0.map(|(n, inv_dn, _)| {
            (limit - mul_add(o.x, n.x, o.y * n.y)) * inv_dn
        });
        // `min` skips NaN (`0 * ∞` for an edge parallel to `d`).
        let k = steps[0].min(steps[1]).min(steps[2]);
        let exit = if steps[1] == k {
            self.0[1].2
        } else if steps[2] == k {
            self.0[2].2
        } else {
            self.0[0].2
        };
        // Callers truncate to an integer, which floors a non-negative value.
        (k.max(0.), (k < f64::INFINITY).then_some(exit))
    }
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

/// Validated bounds for a span, and what the indexer derives from them.
#[derive(Debug, Clone, Copy)]
struct Bounds {
    speed: f64,
    accel: f64,
    /// The reported noise bound plus `ROUNDING_NOISE`.
    noise: f64,
    /// `max |Q''|` of the gnomonic plane position along the path, per sample²
    /// (tangent-plane units).
    curvature: f64,
}

impl Bounds {
    /// `None` for NaN, negative or infinite bounds, or noise above
    /// `MAX_NOISE`: those locations use `to_cell`.
    fn new(bounds: SpanBounds) -> Option<Self> {
        let SpanBounds {
            speed,
            accel,
            noise,
        } = bounds;
        let is_bound = |value: f64| (0. ..f64::INFINITY).contains(&value);
        if !is_bound(speed)
            || !is_bound(accel)
            || !(0. ..=MAX_NOISE).contains(&noise)
        {
            return None;
        }
        // Q = P_perp / w with |P_perp| ≤ 1, w ≥ W_MIN, |P'|, |w'| ≤ V and
        // |P''|, |w''| ≤ A:
        //   |Q''| ≤ A/w + 2V²/w² + A/w² + 2V²/w³.
        let v2 = speed * speed;
        let curvature = accel / W_MIN
            + 2. * v2 / (W_MIN * W_MIN)
            + accel / (W_MIN * W_MIN)
            + 2. * v2 / (W_MIN * W_MIN * W_MIN);
        Some(Self {
            speed,
            accel,
            noise: noise + ROUNDING_NOISE,
            curvature,
        })
    }

    /// Longest span whose interpolation error stays near `TARGET_MARGIN` at
    /// the finest requested resolution (scale `finest`).
    fn span_len(&self, finest: f64) -> usize {
        if self.curvature <= 0. {
            return MAX_SPAN;
        }
        // Interpolation error over `h` samples: INV_RES0_U * h²/8 * |Q''|.
        let h = sqrt(
            8. * TARGET_MARGIN
                / (finest * INV_RES0_U_GNOMONIC * self.curvature),
        );
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped below"
        )]
        let h = h.min(MAX_SPAN_F64) as usize;
        h.max(1)
    }
}

struct Row<'a, P, E> {
    indexer: &'a mut LatLngIndexer,
    path: &'a mut P,
    emit: E,
    levels: Vec<Level>,
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
    let finest = resolutions
        .iter()
        .map(|&res| SQRT7_POWERS[usize::from(res)])
        .fold(0., f64::max);
    // First span length from the path-wide bounds; afterwards, from the
    // previous span's bounds.
    let mut len = Bounds::new(SpanBounds {
        speed: path.speed_bound(),
        accel: path.accel_bound(),
        noise: path.noise_bound(),
    })
    .map_or(MAX_SPAN, |bounds| bounds.span_len(finest));

    let mut row = Row {
        indexer,
        path,
        emit,
        levels: resolutions.iter().copied().map(Level::new).collect(),
        last_point: None,
        stats: RowStats::default(),
    };

    let mut a = cols.start;
    // Exact sample at `a`, if the previous span computed it.
    let mut sa = None;
    while a < cols.end {
        let b = (a + len).min(cols.end);
        let bounds = row.path.span_bounds(a, b).and_then(Bounds::new);
        if let Some(bounds) = bounds {
            // The bounds hold on `a..=b`, hence on any shorter span from `a`.
            len = bounds.span_len(finest);
            let end = b.min(a + len);
            let sa_ = sa.unwrap_or_else(|| row.sample(a));
            let sb = row.sample(end);
            row.span(a, sa_, end, sb, bounds);
            a = end;
            sa = Some(sb);
        } else {
            for col in a..b {
                row.exact(col);
            }
            a = b;
            sa = None;
        }
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
    fn span(
        &mut self,
        a: usize,
        sa: Sample,
        b: usize,
        sb: Sample,
        bounds: Bounds,
    ) {
        let h = (b - a) as f64;

        // Every location of the span must have `sa.face` as closest face:
        // `s_g = P·(c_f - c_g)` is linear in P, so on the smooth path its
        // deviation from the chord is ≤ h²/8 * |c_f - c_g| * A ≤ h²/4 * A.
        // Point noise σ shifts each `s_g` by ≤ 2σ: at both samples (moving
        // the chord) and at the location itself, hence 4σ.
        let face_ok = sa.face == sb.face
            && sa.face_margin.min(sb.face_margin)
                > mul_add(
                    h * h * 0.25,
                    bounds.accel,
                    mul_add(4., bounds.noise, 1e-13),
                );
        // ... and stay away from the face center, where `to_cell` is noisy.
        let r_lo = mul_add(
            -0.5 * h,
            bounds.speed,
            mul_add(-2., bounds.noise, sa.r.min(sb.r)),
        );
        if !face_ok || r_lo < MIN_FACE_CENTER_DISTANCE {
            if b - a >= 2 {
                let m = a + (b - a) / 2;
                let sm = self.sample(m);
                self.span(a, sa, m, sm, bounds);
                self.span(m, sm, b, sb, bounds);
            } else {
                self.exact(a);
            }
            return;
        }

        // Bound on |true - interpolated| + 2 * float noise, at resolution 0.
        // Point noise σ moves the interpolated chord (through the samples)
        // and the location itself by ≤ σ * PLANE_PER_RADIAN each.
        let interpolation =
            INV_RES0_U_GNOMONIC * h * h * 0.125 * bounds.curvature;
        let float_noise = TO_CELL_NOISE * (1. + 1. / r_lo);
        let point_noise = 2. * PLANE_PER_RADIAN * bounds.noise;
        let error0 = mul_add(2., float_noise, interpolation + point_noise);

        let (dx, dy) = ((sb.q.x - sa.q.x) / h, (sb.q.y - sa.q.y) / h);
        for index in 0..self.levels.len() {
            let level = &self.levels[index];
            let resolution = level.resolution;
            // Per-column displacement at this resolution.
            let d = level.map(dx, dy);
            let travel = h * sqrt(mul_add(d.x, d.x, d.y * d.y));
            let edges = ExitEdges::new(d);
            let margin = level.scale * error0;

            let mut col = a;
            while col < b {
                let t = (col - a) as f64;
                let q = self.levels[index]
                    .map(mul_add(t, dx, sa.q.x), mul_add(t, dy, sa.q.y));
                // Margin, plus rounding of the interpolation and plane mapping
                // for any column of the span.
                let required = mul_add(
                    16. * f64::EPSILON,
                    1. + abs(q.x) + abs(q.y) + travel,
                    margin,
                );
                // A hexagon containing `q` with margin is the one
                // `CoordIJK::from(q)` returns (see `hex_margin`), so the
                // hint, when it passes, decides exactly like rounding.
                let ijk = match self.levels[index].hint.take() {
                    Some(hint)
                        if hint.face == sa.face
                            && hex_margin(q, hint.coord) > required =>
                    {
                        hint.coord
                    }
                    _ => CoordIJK::from(q),
                };
                let center = Vec2d::from(ijk);
                let o = Vec2d::new(q.x - center.x, q.y - center.y);
                if hex_margin(q, ijk) > required {
                    self.stats.decisions += 1;
                    let cell = self.indexer.encode_recent(
                        FaceIJK::new(sa.face, ijk),
                        resolution,
                        &mut self.levels[index].encode_cache,
                    );
                    self.levels[index].push(index, col, cell, &mut self.emit);
                    // Columns `col + k` stay in the hexagon with margin as
                    // long as `(o + k d)·n ≤ 0.5 - required` for each of the
                    // 6 edge normals ±n.
                    let (skip, exit) = edges.exit_steps(o, 0.5 - required);
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "non-negative, clamped to the span"
                    )]
                    let skip = skip.min(h) as usize;
                    col += skip + 1;
                    // The next decision is at `col` if `col <= b` (here or
                    // as the next span's first column), past the edge that
                    // stopped the skip; otherwise the next span starts
                    // inside this hexagon.
                    let next = match exit {
                        Some(offset) if col <= b => (ijk + offset).normalize(),
                        _ => ijk,
                    };
                    self.levels[index].hint = Some(FaceIJK::new(sa.face, next));
                } else {
                    self.stats.fallbacks += 1;
                    let p = Vec3d::from(self.point(col));
                    let cell = self.exact_cell(p, index);
                    self.levels[index].push(index, col, cell, &mut self.emit);
                    col += 1;
                }
            }
        }
    }

    /// `LatLng::to_cell` at level `index` of the location whose n-vector is
    /// `p`: the same `FaceIJK` address, encoded through the recent cells.
    fn exact_cell(&mut self, p: Vec3d, index: usize) -> CellIndex {
        let level = &mut self.levels[index];
        let fijk = FaceIJK::from_vec3d(p, level.resolution);
        self.indexer.encode_recent(
            fijk,
            level.resolution,
            &mut level.encode_cache,
        )
    }

    /// Indexes column `col` with `to_cell` at every resolution.
    fn exact(&mut self, col: usize) {
        let p = Vec3d::from(self.point(col));
        for index in 0..self.levels.len() {
            self.stats.fallbacks += 1;
            let cell = self.exact_cell(p, index);
            self.levels[index].push(index, col, cell, &mut self.emit);
        }
    }
}

#[cfg(test)]
#[path = "./row_tests.rs"]
mod tests;
