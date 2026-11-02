//! Renders an H3 layer for one Web Mercator (slippy map) tile with
//! [`LatLngIndexer::index_row`].
//!
//! ```text
//! cargo run --release --example web_mercator_tile -- [Z X Y [RES...]] [--size N] [--out FILE] [--check]
//! ```
//!
//! - `Z X Y`: the tile (defaults to 11/426/777, around Denver).
//! - `RES...`: H3 resolutions. The first one fills the tile and is outlined;
//!   further ones are outlined in black, e.g. a coarser parent level. By
//!   default, the finest resolution whose edges span at least 8 pixels.
//! - `--size N`: tile width and height in pixels (default 256; 512 for
//!   high-DPI tiles).
//! - `--out FILE`: writes the tile as a binary PPM image.
//! - `--check`: compares every pixel with `LatLng::to_cell`.
//!
//! # Why rows of a Web Mercator tile are cheap to index
//!
//! In Web Mercator (EPSG:3857), x is proportional to longitude and y depends
//! on latitude only. Every screen row of a tile is therefore a parallel: one
//! latitude, with longitudes spaced evenly along the row. That's exactly a
//! [`GeographicRow`], whose speed and acceleration bounds h3o derives itself.
//! No projection-specific bounds are needed.
//!
//! `index_row` then reports maximal runs of pixels that share a cell, exactly
//! as `to_cell` would assign them, while projecting only a few pixels per row.
//! A renderer can paint each run at once instead of handling each pixel.

use h3o::{
    CellIndex, GeographicRow, LatLng, LatLngIndexer, Resolution, RowPath,
    RowStats,
};
use std::{
    f64::consts::PI, fs::File, io::Write, process::ExitCode, time::Instant,
};

/// Earth's circumference at the equator in Web Mercator, in metres.
const EQUATOR_M: f64 = 40_075_016.685_578_5;

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let tile = Tile::new(args.z, args.x, args.y, args.size);
    let resolutions = if args.resolutions.is_empty() {
        vec![tile.default_resolution()]
    } else {
        args.resolutions.clone()
    };

    // 1. Index the tile: one `index_row` call per screen row, all resolutions
    //    at once. `cells[r][row * size + col]` is the cell of that pixel at
    //    `resolutions[r]`.
    let start = Instant::now();
    let (cells, runs, stats) = tile.index(&resolutions);
    let elapsed = start.elapsed();

    let pixels = args.size * args.size;
    println!(
        "tile {}/{}/{} ({}×{} px), resolutions {:?}",
        args.z, args.x, args.y, args.size, args.size, resolutions
    );
    println!(
        "indexed in {:.2} ms: {runs} runs, {} exact samples, {} decisions, \
         {} to_cell fallbacks ({:.2}% of pixel × resolution pairs)",
        elapsed.as_secs_f64() * 1e3,
        stats.samples,
        stats.decisions,
        stats.fallbacks,
        100. * stats.fallbacks as f64 / (pixels * resolutions.len()) as f64,
    );

    // 2. Optionally, check every pixel against `to_cell`.
    if args.check {
        let start = Instant::now();
        let mut mismatches = 0;
        for row in 0..args.size {
            let mut path = tile.row(row);
            for col in 0..args.size {
                let ll = path.point(col);
                for (r, &res) in resolutions.iter().enumerate() {
                    mismatches += usize::from(
                        cells[r][row * args.size + col] != ll.to_cell(res),
                    );
                }
            }
        }
        println!(
            "check: {mismatches} mismatches with to_cell (per-pixel to_cell \
             took {:.2} ms)",
            start.elapsed().as_secs_f64() * 1e3
        );
        if mismatches != 0 {
            return ExitCode::FAILURE;
        }
    }

    // 3. Optionally, render: fill with the first resolution, outline every
    //    resolution where the cell changes between neighbouring pixels.
    if let Some(path) = &args.out {
        if let Err(err) = write_ppm(path, args.size, &cells) {
            eprintln!("error: {path}: {err}");
            return ExitCode::FAILURE;
        }
        println!("wrote {path}");
    }
    ExitCode::SUCCESS
}

/// A Web Mercator tile and its pixel centers.
struct Tile {
    z: u32,
    x: u32,
    y: u32,
    size: usize,
}

impl Tile {
    fn new(z: u32, x: u32, y: u32, size: usize) -> Self {
        Self { z, x, y, size }
    }

    /// Number of pixels across the whole world at this zoom.
    fn world_px(&self) -> f64 {
        f64::from(1_u32 << self.z) * self.size as f64
    }

    /// Latitude of the centers of screen row `row`, in degrees: the inverse
    /// Mercator of the row's y (rows grow southwards).
    fn row_lat(&self, row: usize) -> f64 {
        let y_px = f64::from(self.y) * self.size as f64 + row as f64 + 0.5;
        let y = PI * (1. - 2. * y_px / self.world_px());
        y.sinh().atan().to_degrees()
    }

    /// Screen row `row` as a `RowPath`: constant latitude, pixel `col` at
    /// longitude `lng0 + col * step`. The bounds come with `GeographicRow`.
    fn row(&self, row: usize) -> GeographicRow {
        let step = 360. / self.world_px();
        let lng0 =
            (f64::from(self.x) * self.size as f64 + 0.5).mul_add(step, -180.);
        GeographicRow::new(self.row_lat(row), lng0, step)
    }

    /// The finest resolution whose average hexagon edge spans at least 8
    /// pixels at the tile's center.
    fn default_resolution(&self) -> Resolution {
        let lat = self.row_lat(self.size / 2).to_radians();
        let pixel_m = EQUATOR_M * lat.cos() / self.world_px();
        Resolution::range(Resolution::Zero, Resolution::Fifteen)
            .take_while(|res| res.edge_length_m() >= 8. * pixel_m)
            .last()
            .unwrap_or(Resolution::Zero)
    }

    /// Indexes every pixel at every resolution. Returns the cell of each
    /// pixel per resolution, the number of runs, and the summed `RowStats`.
    fn index(
        &self,
        resolutions: &[Resolution],
    ) -> (Vec<Vec<CellIndex>>, usize, RowStats) {
        let size = self.size;
        // Placeholder until every pixel is written; each one is, exactly once.
        let placeholder = LatLng::new(0., 0.)
            .expect("valid")
            .to_cell(Resolution::Zero);
        let mut cells = vec![vec![placeholder; size * size]; resolutions.len()];
        let mut runs = 0;
        let mut total = RowStats::default();
        // Reused across rows: neighbouring rows share most cells, so #1's
        // encoding cache keeps working across row boundaries.
        let mut indexer = LatLngIndexer::new();
        for row in 0..size {
            let mut path = self.row(row);
            let stats = indexer.index_row(
                &mut path,
                0..size,
                resolutions,
                |r, cols, cell| {
                    // `cols` is a maximal run of pixels of this row that all
                    // fall in `cell` at `resolutions[r]`. A renderer would
                    // paint the span directly; here it is stored per pixel.
                    runs += 1;
                    cells[r][row * size + cols.start..row * size + cols.end]
                        .fill(cell);
                },
            );
            total.samples += stats.samples;
            total.decisions += stats.decisions;
            total.fallbacks += stats.fallbacks;
        }
        (cells, runs, total)
    }
}

/// Writes the tile as a binary PPM: a pastel color per cell of the first
/// resolution, darker outlines where that cell changes, and black outlines
/// where a cell of any further resolution changes.
fn write_ppm(
    path: &str,
    size: usize,
    cells: &[Vec<CellIndex>],
) -> std::io::Result<()> {
    let boundary = |r: usize, row: usize, col: usize| {
        let cell = cells[r][row * size + col];
        (col > 0 && cells[r][row * size + col - 1] != cell)
            || (row > 0 && cells[r][(row - 1) * size + col] != cell)
    };
    let mut image = Vec::with_capacity(size * size * 3);
    for row in 0..size {
        for col in 0..size {
            let mut rgb = color(cells[0][row * size + col]);
            if boundary(0, row, col) {
                rgb = rgb.map(|c| c / 2);
            }
            if (1..cells.len()).any(|r| boundary(r, row, col)) {
                rgb = [0, 0, 0];
            }
            image.extend_from_slice(&rgb);
        }
    }
    let mut file = File::create(path)?;
    write!(file, "P6\n{size} {size}\n255\n")?;
    file.write_all(&image)
}

/// A stable pastel color for a cell.
fn color(cell: CellIndex) -> [u8; 3] {
    let mut h = u64::from(cell).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^= h >> 29;
    let channel = |shift: u32| 128 + ((h >> shift) & 0x7f) as u8;
    [channel(0), channel(16), channel(32)]
}

struct Args {
    z: u32,
    x: u32,
    y: u32,
    resolutions: Vec<Resolution>,
    size: usize,
    out: Option<String>,
    check: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut positional = Vec::new();
        let (mut size, mut out, mut check) = (256, None, false);
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--check" => check = true,
                "--out" => out = Some(args.next().ok_or("--out needs a path")?),
                "--size" => {
                    size = args
                        .next()
                        .and_then(|s| s.parse().ok())
                        .filter(|&n| n > 0)
                        .ok_or("--size needs a positive integer")?;
                }
                _ => positional.push(
                    arg.parse::<u32>()
                        .map_err(|_| format!("unexpected argument `{arg}`"))?,
                ),
            }
        }
        let (z, x, y) = match positional[..] {
            [] => (11, 426, 777),
            [z, x, y, ..] => (z, x, y),
            _ => return Err("expected Z X Y [RES...]".into()),
        };
        if z > 24 || x >= 1 << z || y >= 1 << z {
            return Err(format!("invalid tile {z}/{x}/{y}"));
        }
        let resolutions = positional
            .iter()
            .skip(3)
            .map(|&res| {
                u8::try_from(res)
                    .ok()
                    .and_then(|res| Resolution::try_from(res).ok())
                    .ok_or(format!("invalid resolution {res}"))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            z,
            x,
            y,
            resolutions,
            size,
            out,
            check,
        })
    }
}
