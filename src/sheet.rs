//! Contact sheet: one iterated location × N palettes, composited into a grid.
//!
//! The whole point is separability: the location is iterated **once** into a
//! [`SampleBuffer`], then each palette re-shades that same buffer via the pure
//! [`shade_and_downsample`] stage — the expensive iteration is amortized over
//! every palette. This module never touches a backend.
//!
//! Each tile is the shaded location with a thin **gradient swatch strip** burned
//! along the bottom (the palette across `t∈[0,1)`) and a **numeric index** in the
//! top-left (a hand-rolled 5×7 bitmap font — no font crate). Tiles paste into a
//! grid PNG; the caller prints a `tile N → <name>` legend.

use clap::Args;
use image::RgbImage;

use crate::cli::{LocationArgs, ShadeArgs};
use crate::coloring::ColorParams;
use crate::palette::{linear_to_srgb, Palette};
use crate::render::{shade_and_downsample, SampleBuffer};

/// Background/padding color of the grid (dark gray).
const PAD_RGB: [u8; 3] = [24, 24, 24];
/// Padding between tiles, in pixels.
const PAD: u32 = 6;
/// Swatch strip height, in pixels.
const SWATCH_H: u32 = 18;

/// Render the contact sheet. `buf` is the once-iterated supersampled location;
/// `palettes` re-shade it. Returns the composed grid plus the tile→name legend.
pub fn render_contact_sheet(
    buf: &SampleBuffer,
    palettes: &[Palette],
    params: &ColorParams,
    pixel_spacing: f64,
    cols: Option<usize>,
) -> (RgbImage, Vec<String>) {
    assert!(!palettes.is_empty(), "contact sheet needs ≥1 palette");
    let tile_w = buf.out_width;
    let tile_h = buf.out_height;

    // One shaded tile per palette (the only per-palette cost — no re-iteration).
    let tiles: Vec<RgbImage> = palettes
        .iter()
        .enumerate()
        .map(|(i, pal)| {
            let mut img = shade_and_downsample(
                &buf.samples,
                tile_w,
                tile_h,
                buf.ss,
                pal,
                params,
                pixel_spacing,
            );
            draw_swatch(&mut img, pal);
            draw_index(&mut img, i);
            img
        })
        .collect();

    let grid = compose_grid(&tiles, cols);

    let legend = palettes
        .iter()
        .enumerate()
        .map(|(i, p)| format!("tile {i} → {}", p.name()))
        .collect();
    (grid, legend)
}

/// Compose equal-sized `tiles` into a padded grid (`PAD`-px gutters, `PAD_RGB`
/// background), `cols` columns (default ≈ √N). Shared by the palette contact
/// sheet and the `search` top-N diversity sheet — the only difference between
/// them is how the tiles are produced (one buffer × N palettes vs. N distinct
/// already-rendered locations), so the grid layout itself lives here once.
pub fn compose_grid(tiles: &[RgbImage], cols: Option<usize>) -> RgbImage {
    assert!(!tiles.is_empty(), "compose_grid needs ≥1 tile");
    let tile_w = tiles[0].width();
    let tile_h = tiles[0].height();
    let n = tiles.len();
    let cols = cols.unwrap_or_else(|| (n as f64).sqrt().ceil() as usize).max(1);
    let rows = n.div_ceil(cols);

    let grid_w = cols as u32 * tile_w + (cols as u32 + 1) * PAD;
    let grid_h = rows as u32 * tile_h + (rows as u32 + 1) * PAD;
    let mut grid = RgbImage::from_pixel(grid_w, grid_h, image::Rgb(PAD_RGB));

    for (i, tile) in tiles.iter().enumerate() {
        let col = (i % cols) as u32;
        let row = (i / cols) as u32;
        let x0 = PAD + col * (tile_w + PAD);
        let y0 = PAD + row * (tile_h + PAD);
        for (tx, ty, px) in tile.enumerate_pixels() {
            if x0 + tx < grid_w && y0 + ty < grid_h {
                grid.put_pixel(x0 + tx, y0 + ty, *px);
            }
        }
    }
    grid
}

/// Burn the palette's gradient (`t∈[0,1)`) across the bottom `SWATCH_H` rows.
fn draw_swatch(img: &mut RgbImage, pal: &Palette) {
    let w = img.width();
    let h = img.height();
    if h <= SWATCH_H {
        return;
    }
    let y0 = h - SWATCH_H;
    for x in 0..w {
        let t = x as f64 / w as f64;
        let lin = pal.lookup_linear(t);
        let rgb = [
            (linear_to_srgb(lin[0]) * 255.0 + 0.5) as u8,
            (linear_to_srgb(lin[1]) * 255.0 + 0.5) as u8,
            (linear_to_srgb(lin[2]) * 255.0 + 0.5) as u8,
        ];
        for y in y0..h {
            img.put_pixel(x, y, image::Rgb(rgb));
        }
    }
    // A 1px dark rule above the swatch separates it from the fractal.
    for x in 0..w {
        img.put_pixel(x, y0, image::Rgb([0, 0, 0]));
    }
}

/// Burn the decimal index into the top-left over a dark plate for legibility.
fn draw_index(img: &mut RgbImage, index: usize) {
    let digits: Vec<u8> = index
        .to_string()
        .bytes()
        .map(|b| b - b'0')
        .collect();
    const SCALE: u32 = 2;
    const PADP: u32 = 2;
    let gw = GLYPH_W * SCALE;
    let gh = GLYPH_H * SCALE;
    let plate_w = PADP * 2 + digits.len() as u32 * (gw + SCALE);
    let plate_h = PADP * 2 + gh;
    // Dark translucent-looking plate (solid dark).
    for y in 0..plate_h.min(img.height()) {
        for x in 0..plate_w.min(img.width()) {
            img.put_pixel(x, y, image::Rgb([0, 0, 0]));
        }
    }
    let mut cx = PADP;
    let cy = PADP;
    for d in digits {
        draw_glyph(img, DIGITS[d as usize], cx, cy, SCALE);
        cx += gw + SCALE;
    }
}

const GLYPH_W: u32 = 5;
const GLYPH_H: u32 = 7;

/// Draw a 5×7 glyph (row-major bits, MSB = leftmost of 5) at `(x0,y0)`, scaled.
fn draw_glyph(img: &mut RgbImage, rows: [u8; 7], x0: u32, y0: u32, scale: u32) {
    let white = image::Rgb([240, 240, 240]);
    for (ry, bits) in rows.iter().enumerate() {
        for rx in 0..GLYPH_W {
            // bit 4 is the leftmost column.
            if (bits >> (GLYPH_W - 1 - rx)) & 1 == 1 {
                for sy in 0..scale {
                    for sx in 0..scale {
                        let px = x0 + rx * scale + sx;
                        let py = y0 + ry as u32 * scale + sy;
                        if px < img.width() && py < img.height() {
                            img.put_pixel(px, py, white);
                        }
                    }
                }
            }
        }
    }
}

/// 5×7 bitmaps for digits 0–9 (each row is 5 low bits).
const DIGITS: [[u8; 7]; 10] = [
    // 0
    [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    // 1
    [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    // 2
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    // 3
    [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
    // 4
    [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    // 5
    [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    // 6
    [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    // 7
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    // 8
    [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    // 9
    [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
];

// ===== Args structs relocated from cli.rs (P0 cli decomposition) =====
/// `sheet` subcommand: same location + shading, multiple palettes.
#[derive(Args, Debug)]
pub struct SheetArgs {
    #[command(flatten)]
    pub location: LocationArgs,

    #[command(flatten)]
    pub shade: ShadeArgs,

    /// Palette file paths (`.ugr`/`.map`). For multi-block `.ugr`, every block
    /// is included as its own tile.
    #[arg(long, num_args = 0.., value_delimiter = ' ')]
    pub palettes: Vec<String>,

    /// Built-in palette names (`default`, `cubehelix`, `viridis`).
    #[arg(long, num_args = 0.., value_delimiter = ' ')]
    pub builtins: Vec<String>,

    /// Grid columns (default: ≈ √N).
    #[arg(long)]
    pub cols: Option<usize>,

    /// Per-tile width in pixels (height follows aspect). Modest, e.g. 320–512.
    #[arg(long, default_value_t = 384)]
    pub tile_width: u32,

    /// Reverse every palette's gradient direction.
    #[arg(long, default_value_t = false)]
    pub palette_reverse: bool,

    /// Output PNG path.
    #[arg(long, default_value = "out/strips/sheet.png")]
    pub output: String,
}
