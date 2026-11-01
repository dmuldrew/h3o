"""Schematic figures for h3o PR #2, in H3's hex-plane convention:
unit spacing between hex centers, apothem 0.5, pointy-top hexagons,
centers at (i - j/2, j*sqrt(3)/2). Distances are exaggerated for legibility."""
import math

S3 = math.sqrt(3)
R = 1 / S3  # circumradius
NORMALS = [(1, 0), (0.5, S3 / 2), (-0.5, S3 / 2)]
COL = dict(edge="#9aa0a6", shrunk="#c5cad0", path="#1a73e8", chord="#3c4043",
           decided="#188038", skipped="#81c995", fallback="#e8710a",
           wrong="#d93025", sample="#202124", text="#202124", band="#d2e3fc")


def center(i, j):
    return (i - j / 2, j * S3 / 2)


def nearest_center(p):
    best = None
    jj = round(p[1] / (S3 / 2))
    for j in range(jj - 2, jj + 3):
        ii = round(p[0] + j / 2)
        for i in range(ii - 2, ii + 3):
            c = center(i, j)
            d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
            if best is None or d < best[0]:
                best = (d, (i, j), c)
    return best[1], best[2]


def hex_margin(p):
    ij, c = nearest_center(p)
    o = (p[0] - c[0], p[1] - c[1])
    return 0.5 - max(abs(o[0] * nx + o[1] * ny) for nx, ny in NORMALS), ij, o


def exit_steps(o, d, limit):
    k = math.inf
    for nx, ny in NORMALS:
        on = o[0] * nx + o[1] * ny
        dn = d[0] * nx + d[1] * ny
        if dn > 0:
            k = min(k, (limit - on) / dn)
        elif dn < 0:
            k = min(k, (limit + on) / -dn)
    return max(0, math.floor(k))


class Svg:
    def __init__(self, w, h, x0, x1, y0, y1, pad=10):
        self.w, self.h = w, h
        self.sx = (w - 2 * pad) / (x1 - x0)
        self.sy = self.sx
        self.x0, self.y1, self.pad = x0, y1, pad
        self.items = []

    def t(self, p):
        return (self.pad + (p[0] - self.x0) * self.sx, self.pad + (self.y1 - p[1]) * self.sy)

    def poly(self, pts, stroke, width=1.5, dash=None, fill="none", close=True, opacity=1):
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(self.t, pts))
        tag = "polygon" if close else "polyline"
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<{tag} points="{d}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{width}"{da} stroke-linejoin="round"/>')

    def dot(self, p, r, fill, stroke="none"):
        x, y = self.t(p)
        self.items.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')

    def text(self, p, s, size=15, anchor="start", color=None, weight="normal", px=False):
        x, y = p if px else self.t(p)
        self.items.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color or COL["text"]}" font-weight="{weight}" font-family="Helvetica, Arial, sans-serif">{s}</text>')

    def panel(self, x, y, w, h):
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" fill-opacity="0.93"/>')

    def raw(self, s):
        self.items.append(s)

    def hexagon(self, c, r=R, **kw):
        pts = [(c[0] + r * math.cos(math.radians(30 + 60 * k)), c[1] + r * math.sin(math.radians(30 + 60 * k))) for k in range(6)]
        self.poly(pts, **kw)

    def save(self, path):
        body = "\n".join(self.items)
        open(path, "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}"><rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>\n')


def grid(svg, cells, margin=None):
    for i, j in cells:
        c = center(i, j)
        svg.hexagon(c, stroke=COL["edge"], width=1.6)
        if margin:
            svg.hexagon(c, r=(0.5 - margin) / (S3 / 2), stroke=COL["shrunk"], width=1.2, dash="5,4")


def legend(svg, entries, x, y, step=24):
    for k, (kind, color, label) in enumerate(entries):
        yy = y + k * step
        if kind == "dot":
            svg.raw(f'<circle cx="{x + 8}" cy="{yy - 5}" r="6" fill="{color}"/>')
        elif kind == "small":
            svg.raw(f'<circle cx="{x + 8}" cy="{yy - 5}" r="3.5" fill="{color}"/>')
        elif kind == "ring":
            svg.raw(f'<circle cx="{x + 8}" cy="{yy - 5}" r="7" fill="white" stroke="{color}" stroke-width="2.5"/>')
        elif kind == "line":
            svg.raw(f'<line x1="{x}" y1="{yy - 5}" x2="{x + 18}" y2="{yy - 5}" stroke="{color}" stroke-width="2.5"/>')
        elif kind == "dash":
            svg.raw(f'<line x1="{x}" y1="{yy - 5}" x2="{x + 18}" y2="{yy - 5}" stroke="{color}" stroke-width="2" stroke-dasharray="5,4"/>')
        elif kind == "x":
            svg.raw(f'<text x="{x + 8}" y="{yy}" font-size="16" text-anchor="middle" fill="{color}" font-weight="bold" font-family="Helvetica, Arial, sans-serif">✕</text>')
        svg.text((x + 26, yy), label, size=14, px=True)


CELLS = [(i, j) for j in (-1, 0, 1) for i in range(-2, 5)]

# ---------------------------------------------------------------- figure 1
def fig_margin():
    m, bulge = 0.10, 0.05
    a, b = (-0.62, -0.10), (2.55, 0.16)
    svg = Svg(900, 470, -1.05, 3.05, -0.95, 0.95)
    grid(svg, CELLS, margin=m)
    d = (b[0] - a[0], b[1] - a[1])
    L = math.hypot(*d)
    nrm = (-d[1] / L, d[0] / L)
    # error band around the chord
    band = [(a[0] + nrm[0] * bulge, a[1] + nrm[1] * bulge), (b[0] + nrm[0] * bulge, b[1] + nrm[1] * bulge),
            (b[0] - nrm[0] * bulge, b[1] - nrm[1] * bulge), (a[0] - nrm[0] * bulge, a[1] - nrm[1] * bulge)]
    svg.poly(band, stroke="none", fill=COL["band"], opacity=0.8)
    true = [(a[0] + d[0] * t + nrm[0] * bulge * 4 * t * (1 - t), a[1] + d[1] * t + nrm[1] * bulge * 4 * t * (1 - t)) for t in [k / 100 for k in range(101)]]
    svg.poly(true, stroke=COL["path"], width=2.5, close=False)
    svg.poly([a, b], stroke=COL["chord"], width=1.6, dash="7,5", close=False)
    n = 30
    for k in range(1, n):
        q = (a[0] + d[0] * k / n, a[1] + d[1] * k / n)
        delta, _, _ = hex_margin(q)
        svg.dot(q, 5.5, COL["decided"] if delta > m else COL["fallback"])
    for p in (a, b):
        svg.dot(p, 8, "white", stroke=COL["sample"])
    svg.panel(0, 0, 900, 40)
    svg.panel(0, 338, 900, 132)
    svg.text((14, 26), "Margin test, in the resolution-r hex plane of one face", size=17, weight="bold", px=True)
    legend(svg, [("ring", COL["sample"], "exact sample (full projection, span end)"),
                 ("line", COL["path"], "true path of the locations"),
                 ("dash", COL["chord"], "chord: linear interpolation between samples"),
                 ("dot", COL["decided"], "interpolated location trusted: cell decided without projecting"),
                 ("dot", COL["fallback"], "within the margin of an edge: falls back to to_cell")], 14, 362)
    svg.text((560, 362), "dashed hexagons: each cell shrunk by the margin", size=14, px=True)
    svg.text((560, 386), "margin ≥ interpolation error (blue band)", size=14, px=True)
    svg.text((560, 410), "        + 2 × to_cell float noise + rounding", size=14, px=True)
    svg.text((560, 440), "(distances exaggerated)", size=13, color="#5f6368", px=True)
    svg.save("/tmp/figs/margin.svg")

# ---------------------------------------------------------------- figure 2
def fig_skip():
    m = 0.035
    a, b = (-0.62, -0.06), (2.55, 0.12)
    svg = Svg(900, 430, -1.05, 3.05, -0.95, 0.95)
    grid(svg, CELLS, margin=m)
    n = 96
    d = ((b[0] - a[0]) / n, (b[1] - a[1]) / n)
    svg.poly([a, b], stroke=COL["chord"], width=1.4, dash="7,5", close=False)
    col, decided, skipped, fallback, jumps = 0, [], [], [], []
    while col < n:
        q = (a[0] + d[0] * col, a[1] + d[1] * col)
        delta, ij, o = hex_margin(q)
        if delta > m:
            k = min(exit_steps(o, d, 0.5 - m), n - col - 1)
            decided.append(q)
            skipped += [(a[0] + d[0] * c, a[1] + d[1] * c) for c in range(col + 1, col + k + 1)]
            if k >= 3:
                jumps.append((q, (a[0] + d[0] * (col + k), a[1] + d[1] * (col + k))))
            col += k + 1
        else:
            fallback.append(q)
            col += 1
    for p in skipped:
        svg.dot(p, 3.2, COL["skipped"])
    for p, qq in jumps:
        (x1, y1), (x2, y2) = svg.t(p), svg.t(qq)
        h = max(18, (x2 - x1) * 0.35)
        svg.raw(f'<path d="M {x1:.1f} {y1 - 7:.1f} Q {(x1 + x2) / 2:.1f} {y1 - h:.1f} {x2:.1f} {y2 - 7:.1f}" fill="none" stroke="{COL["decided"]}" stroke-width="1.6" marker-end="url(#arr)"/>')
    for p in decided:
        svg.dot(p, 6, COL["decided"])
    for p in fallback:
        svg.dot(p, 5.5, COL["fallback"])
    svg.raw(f'<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{COL["decided"]}"/></marker></defs>')
    svg.panel(0, 0, 900, 40)
    svg.panel(0, 328, 900, 102)
    svg.text((14, 26), f"Skipping: {n} locations, {len(decided)} decisions, {len(fallback)} fallbacks", size=17, weight="bold", px=True)
    legend(svg, [("dot", COL["decided"], "decision: hexagon found, jump to where the chord leaves the shrunk hexagon"),
                 ("small", COL["skipped"], "covered by the jump: same cell, no work"),
                 ("dot", COL["fallback"], "close to an edge: to_cell")], 14, 350)
    svg.text((14, 420), "Work follows cell crossings, not locations. (Distances exaggerated.)", size=13, color="#5f6368", px=True)
    svg.save("/tmp/figs/skipping.svg")

# ---------------------------------------------------------------- figure 3
def fig_reentry():
    cells = [(i, j) for j in (-1, 0, 1, 2) for i in range(-2, 4)]
    n_vec = (0.5, S3 / 2)          # normal of the home cell's top-right edge
    u = (S3 / 2, -0.5)             # direction along that edge
    inset, half, bulge = 0.14, 0.24, 0.24
    c0 = (n_vec[0] * (0.5 - inset), n_vec[1] * (0.5 - inset))
    a = (c0[0] - half * u[0], c0[1] - half * u[1])
    b = (c0[0] + half * u[0], c0[1] + half * u[1])

    def chord_pt(t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    def true_pt(t):
        q, w = chord_pt(t), bulge * 4 * t * (1 - t)
        return (q[0] + n_vec[0] * w, q[1] + n_vec[1] * w)

    n = 14
    home = (0, 0)
    margin = bulge + 0.02
    halves = []
    for panel in ("lookahead", "index_row"):
        svg = Svg(450, 430, -0.85, 1.15, -0.55, 1.35)
        grid(svg, cells)
        svg.hexagon(center(0, 0), stroke=COL["sample"], width=2.4)
        if panel == "index_row":
            nv = n_vec
            band = [(a[0] + nv[0] * bulge, a[1] + nv[1] * bulge), (b[0] + nv[0] * bulge, b[1] + nv[1] * bulge),
                    (b[0] - nv[0] * bulge, b[1] - nv[1] * bulge), (a[0] - nv[0] * bulge, a[1] - nv[1] * bulge)]
            svg.poly(band, stroke="none", fill=COL["band"], opacity=0.8)
            svg.hexagon(center(0, 0), r=(0.5 - margin) / (S3 / 2), stroke=COL["shrunk"], width=1.4, dash="5,4")
        svg.poly([true_pt(k / 100) for k in range(101)], stroke=COL["path"], width=2.5, close=False)
        if panel == "index_row":
            svg.poly([a, b], stroke=COL["chord"], width=1.4, dash="7,5", close=False)
        for k in range(n + 1):
            t = k / n
            if panel == "lookahead":
                p = true_pt(t)
                if k in (0, n):
                    svg.dot(p, 7, "white", stroke=COL["sample"])
                elif nearest_center(p)[0] == home:
                    svg.dot(p, 5, COL["decided"])
                else:
                    x, y = svg.t(p)
                    svg.raw(f'<text x="{x:.1f}" y="{y + 6:.1f}" font-size="17" text-anchor="middle" fill="{COL["wrong"]}" font-weight="bold" font-family="Helvetica, Arial, sans-serif">✕</text>')
            else:
                q = chord_pt(t)
                if k in (0, n):
                    svg.dot(q, 7, "white", stroke=COL["sample"])
                else:
                    svg.dot(q, 5, COL["decided"] if hex_margin(q)[0] > margin else COL["fallback"])
        svg.panel(0, 0, 450, 74)
        if panel == "lookahead":
            svg.text((12, 24), "Endpoint lookahead (removed from raster_h3)", size=16, weight="bold", px=True)
            svg.text((12, 46), "Both ends are in the outlined cell, so every", size=13, px=True)
            svg.text((12, 63), "location in between is assumed to be in it too.", size=13, px=True)
        else:
            svg.text((12, 24), "index_row", size=16, weight="bold", px=True)
            svg.text((12, 46), "The margin covers how far the path can bulge, and", size=13, px=True)
            svg.text((12, 63), "the chord is closer to the edge than that: to_cell decides.", size=13, px=True)
        halves.append(svg)
    parts = ['<defs><clipPath id="half"><rect x="0" y="0" width="450" height="430"/></clipPath></defs>']
    parts += [f'<g transform="translate({k * 450},0)"><g clip-path="url(#half)">' + "\n".join(h.items) + "</g></g>" for k, h in enumerate(halves)]
    leg = Svg(900, 100, 0, 1, 0, 1)
    leg.panel(0, 0, 900, 100)
    legend(leg, [("line", COL["path"], "true path, leaving and re-entering the outlined cell"),
                 ("x", COL["wrong"], "wrong: assigned the outlined cell, but it is in the neighbour"),
                 ("dot", COL["decided"], "correctly in the outlined cell")], 14, 26)
    legend(leg, [("dash", COL["chord"], "chord between exact samples; blue band: possible bulge"),
                 ("dot", COL["fallback"], "to_cell: correct, including the neighbour")], 470, 26)
    body = "\n".join(parts) + '\n<g transform="translate(0,430)">' + "\n".join(leg.items) + "</g>"
    open("/tmp/figs/reentry.svg", "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="530" viewBox="0 0 900 530"><rect width="100%" height="100%" fill="white"/>\n{body}\n<line x1="450" y1="0" x2="450" y2="430" stroke="#5f6368" stroke-width="1.5"/>\n</svg>\n')


fig_margin()
fig_skip()
fig_reentry()
print("ok")
