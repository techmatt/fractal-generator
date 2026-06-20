//! A tiny hand-rolled 5×7 bitmap font for burning short status labels onto a
//! render — no font crate, no external assets.
//!
//! This extends the digit-only table the contact sheet uses ([`crate::sheet`])
//! to the reduced set `[A-Z 0-9 . = + - / e space]` the descend probe needs for
//! its per-row labels (e.g. `L07 M=4.7E5 IT=11500 PERT INT=18 SIG=OK`). Glyphs
//! are row-major: each `u8` row holds the 5 columns in its low bits, bit 4
//! (`0b10000`) being the leftmost column. Unknown characters render blank.

use image::{Rgb, RgbImage};

pub const GLYPH_W: u32 = 5;
pub const GLYPH_H: u32 = 7;

/// Look up the 7-row, 5-bit bitmap for a character. Lowercase letters fold to
/// uppercase; `e` keeps a dedicated lowercase glyph for scientific notation
/// (though labels emit uppercase `E`, both resolve). Unknown → blank.
pub fn glyph(c: char) -> [u8; 7] {
    match c {
        ' ' => [0; 7],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        'A' | 'a' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' | 'b' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' | 'c' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' | 'd' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' | 'f' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' | 'g' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' | 'h' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' | 'i' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' | 'j' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        'K' | 'k' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' | 'l' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' | 'm' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' | 'n' => [0b10001, 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001],
        'O' | 'o' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' | 'p' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' | 'q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' | 'r' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' | 's' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' | 't' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' | 'u' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' | 'v' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' | 'w' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' | 'x' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' | 'y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' | 'z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        'e' => [0b00000, 0b00000, 0b01110, 0b10001, 0b11110, 0b10000, 0b01110],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        '=' => [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        ':' => [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
        '%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        _ => [0; 7],
    }
}

/// Advance (in unscaled glyph cells, including the 1px inter-glyph gap) for one
/// character — used to size the backing plate before drawing.
pub const ADVANCE: u32 = GLYPH_W + 1;

/// Pixel width of `text` at `scale` (without the trailing inter-glyph gap).
pub fn text_width(text: &str, scale: u32) -> u32 {
    let n = text.chars().count() as u32;
    if n == 0 {
        0
    } else {
        (n * ADVANCE - 1) * scale
    }
}

/// Draw a single 5×7 glyph at `(x0, y0)`, scaled, in `color`. Clips at edges.
pub fn draw_glyph(img: &mut RgbImage, rows: [u8; 7], x0: u32, y0: u32, scale: u32, color: Rgb<u8>) {
    for (ry, bits) in rows.iter().enumerate() {
        for rx in 0..GLYPH_W {
            if (bits >> (GLYPH_W - 1 - rx)) & 1 == 1 {
                for sy in 0..scale {
                    for sx in 0..scale {
                        let px = x0 + rx * scale + sx;
                        let py = y0 + ry as u32 * scale + sy;
                        if px < img.width() && py < img.height() {
                            img.put_pixel(px, py, color);
                        }
                    }
                }
            }
        }
    }
}

/// Draw `text` at `(x0, y0)` scaled in `color`. When `plate` is set, a solid
/// dark rectangle is painted behind the text first (legibility over busy
/// regions). Returns the pixel size of the drawn (plated) box.
pub fn draw_text(
    img: &mut RgbImage,
    text: &str,
    x0: u32,
    y0: u32,
    scale: u32,
    color: Rgb<u8>,
    plate: bool,
) -> (u32, u32) {
    const PAD: u32 = 2;
    let tw = text_width(text, scale);
    let th = GLYPH_H * scale;
    let (box_w, box_h) = (tw + 2 * PAD, th + 2 * PAD);
    if plate {
        for y in 0..box_h {
            for x in 0..box_w {
                let px = x0 + x;
                let py = y0 + y;
                if px < img.width() && py < img.height() {
                    img.put_pixel(px, py, Rgb([0, 0, 0]));
                }
            }
        }
    }
    let mut cx = x0 + PAD;
    let cy = y0 + PAD;
    for ch in text.chars() {
        draw_glyph(img, glyph(ch), cx, cy, scale, color);
        cx += ADVANCE * scale;
    }
    (box_w, box_h)
}
