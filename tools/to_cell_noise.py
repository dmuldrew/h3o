#!/usr/bin/env python3
"""Measure `to_cell`'s floating-point error on the hex-plane position.

`LatLngIndexer::index_row` (src/coord/row.rs) bounds this error by
`TO_CELL_NOISE * (1 + 1/r)`, where `r` is the angular distance to the face
center. This script checks that bound empirically:

1. generates locations: uniform on the sphere, and log-spaced in `r` around
   the face centers, where the `1/r` term dominates;
2. runs the ignored `dump_to_cell_positions` test, which writes h3o's float
   positions at resolutions 0, 14 and 15;
3. evaluates the same projection (same constants: face centers, axes
   azimuths, scaling, Class III rotation) at high precision, from the exact
   unit vector of each float `LatLng`;
4. reports `|float - exact| / (scale * EPSILON * (1 + 1/r))` per decade of
   `r`, and exits with status 1 if it exceeds `TO_CELL_NOISE / EPSILON`.

The constants are read from the sources, so this keeps up with them.

Requires mpmath (`pip install mpmath`). Usage, from anywhere:

    python3 tools/to_cell_noise.py [--points 100000] [--seed 1] [--dps 40]
"""

import argparse
import math
import os
import random
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from mpmath import acos, atan2, cos, mp, mpf, sin, sqrt, tan

ROOT = Path(__file__).resolve().parent.parent
EPSILON = 2.0**-52
RESOLUTIONS = (0, 14, 15)


def rust_float(text):
    return float(text.replace("_f64", "").replace("_", ""))


def read_constants():
    face = (ROOT / "src/face.rs").read_text()
    coord = (ROOT / "src/coord/mod.rs").read_text()
    row = (ROOT / "src/coord/row.rs").read_text()

    def block(source, name):
        start = source.index(f"static {name}")
        return source[start : source.index("];", start)]

    number = r"-?[0-9][0-9_.eE+-]*(?:_f64)?"
    centers = [
        tuple(rust_float(v) for v in m)
        for m in re.findall(
            rf"Vec3d::new\(\s*({number}),\s*({number}),\s*({number})\)",
            block(face, "CENTER_POINT"),
        )
    ]
    axes = [
        rust_float(v)
        for v in re.findall(
            rf"^\s*({number}),", block(face, "AXES_AZ_RADS_CII"), re.M
        )
    ]
    assert len(centers) == 20 and len(axes) == 20, "failed to parse face.rs"

    def const(source, name):
        m = re.search(rf"const {name}: f64 = ({number});", source)
        return rust_float(m.group(1))

    start = coord.index("const SQRT7_POWERS")
    powers = [
        rust_float(v)
        for v in re.findall(
            rf"^\s*({number}),", coord[start : coord.index("];", start)], re.M
        )
    ]
    noise = re.search(r"const TO_CELL_NOISE: f64 = ([0-9.]+) \* f64::EPSILON;", row)
    return {
        "centers": centers,
        "axes": axes,
        "inv_res0_u": const(coord, "INV_RES0_U_GNOMONIC"),
        "ap7_rot": const(coord, "AP7_ROT_RADS"),
        "sqrt7_powers": powers,
        "to_cell_noise": float(noise.group(1)),
    }


def lat_lng(v):
    x, y, z = v
    n = math.sqrt(x * x + y * y + z * z)
    return math.asin(max(-1.0, min(1.0, z / n))), math.atan2(y, x)


def generate(count, centers, seed):
    rng = random.Random(seed)
    points = []
    for _ in range(count * 2 // 5):
        points.append(lat_lng([rng.gauss(0, 1) for _ in range(3)]))
    while len(points) < count:
        c = centers[rng.randrange(20)]
        r = 10 ** rng.uniform(-6.5, -0.3)
        a = [rng.gauss(0, 1) for _ in range(3)]
        d = sum(a[i] * c[i] for i in range(3))
        t = [a[i] - d * c[i] for i in range(3)]
        tn = math.sqrt(sum(x * x for x in t))
        points.append(
            lat_lng([c[i] * math.cos(r) + t[i] / tn * math.sin(r) for i in range(3)])
        )
    return points


def bits(x):
    return f"{struct.unpack('<Q', struct.pack('<d', x))[0]:x}"


def from_bits(s):
    return struct.unpack("<d", struct.pack("<Q", int(s, 16)))[0]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def exact_positions(lat, lng, face, k):
    """Hex-plane positions at RESOLUTIONS, and `r`, in exact arithmetic."""
    p = (cos(mpf(lng)) * cos(mpf(lat)), sin(mpf(lng)) * cos(mpf(lat)), sin(mpf(lat)))
    c = tuple(mpf(x) for x in k["centers"][face])
    # Same azimuth reference as `Vec3d::azimuth`: north is the pole projected
    # on the tangent plane at the face center.
    pole = (mpf(0), mpf(0), mpf(1))
    pc = dot(pole, c)
    north = tuple(pole[i] - pc * c[i] for i in range(3))
    norm = sqrt(dot(north, north))
    north = tuple(x / norm for x in north)
    east = cross(north, c)
    pc = dot(p, c)
    proj = tuple(p[i] - pc * c[i] for i in range(3))
    d2 = sum((p[i] - c[i]) ** 2 for i in range(3))
    r = acos(1 - d2 / 2)
    theta = mpf(k["axes"][face]) - atan2(dot(proj, east), dot(proj, north))
    r0 = tan(r) * mpf(k["inv_res0_u"])
    positions = []
    for res in RESOLUTIONS:
        scale = mpf(k["sqrt7_powers"][res])
        t = theta - mpf(k["ap7_rot"]) if res % 2 else theta
        positions.append((r0 * scale * cos(t), r0 * scale * sin(t), scale))
    return positions, float(r)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dps", type=int, default=40, help="mpmath digits")
    args = parser.parse_args()
    mp.dps = args.dps

    k = read_constants()
    points = generate(args.points, k["centers"], args.seed)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "points.txt", Path(tmp) / "float.txt"
        src.write_text("".join(f"{bits(la)} {bits(lo)}\n" for la, lo in points))
        subprocess.run(
            ["cargo", "test", "-q", "--lib", "dump_to_cell_positions", "--",
             "--ignored", "--exact", "coord::row::tests::dump_to_cell_positions"],
            cwd=ROOT,
            env={**os.environ, "H3O_NOISE_IN": str(src), "H3O_NOISE_OUT": str(dst)},
            check=True,
            stdout=subprocess.DEVNULL,
        )
        lines = dst.read_text().splitlines()
    assert len(lines) == len(points), "dump is missing locations"

    worst = [0.0] * len(RESOLUTIONS)
    decades = {}
    for (lat, lng), line in zip(points, lines):
        fields = line.split()
        face, floats = int(fields[0]), [from_bits(s) for s in fields[1:]]
        exact, r = exact_positions(lat, lng, face, k)
        unit = EPSILON * (1 + 1 / r)
        decade = decades.setdefault(math.floor(math.log10(r)), [0] + [0.0] * len(RESOLUTIONS))
        decade[0] += 1
        for i, (x, y, scale) in enumerate(exact):
            fx, fy = floats[2 * i], floats[2 * i + 1]
            ratio = float(sqrt((fx - x) ** 2 + (fy - y) ** 2) / scale) / unit
            worst[i] = max(worst[i], ratio)
            decade[i + 1] = max(decade[i + 1], ratio)

    header = "  ".join(f"res {res:>2}" for res in RESOLUTIONS)
    print("max |float - exact| / (scale * EPSILON * (1 + 1/r))")
    print(f"{'r':>8} {'points':>7}  {header}")
    for d in sorted(decades):
        n, *ratios = decades[d]
        print(f"{'1e' + str(d):>8} {n:>7}  " + "  ".join(f"{v:6.2f}" for v in ratios))
    bound = k["to_cell_noise"]
    print(f"{'all':>8} {len(points):>7}  " + "  ".join(f"{v:6.2f}" for v in worst))
    print(f"TO_CELL_NOISE / EPSILON = {bound:g}")
    if max(worst) > bound:
        print("bound exceeded", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
