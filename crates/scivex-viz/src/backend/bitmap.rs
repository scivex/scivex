//! Software rasterizer and PNG rendering backend.
//!
//! Renders [`Element`]s to an RGBA pixel buffer and encodes the result as PNG.
//! Everything is implemented from scratch — no external image or compression
//! crates are used.

use super::Renderer;
use crate::color::Color;
use crate::element::{Element, TextAnchor};
use crate::error::Result;

// ---------------------------------------------------------------------------
// Bitmap font – minimal 5x7 glyphs
// ---------------------------------------------------------------------------

/// Width of each glyph in the built-in bitmap font.
const GLYPH_W: usize = 5;
/// Height of each glyph in the built-in bitmap font.
const GLYPH_H: usize = 7;
/// Horizontal spacing between glyphs.
const GLYPH_SPACING: usize = 1;

/// Returns a 5×7 bitmap for a character, or `None` if the character is not in
/// the built-in font.  Each `u8` encodes one row of 5 pixels (MSB-first, i.e.
/// bit 4 is the leftmost pixel).
#[allow(clippy::too_many_lines)]
fn glyph(ch: char) -> Option<[u8; GLYPH_H]> {
    #[allow(clippy::match_same_arms)]
    let g = match ch {
        'A' | 'a' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' | 'b' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
        'C' | 'c' => [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
        'D' | 'd' => [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'E' | 'e' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'F' | 'f' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'G' | 'g' => [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
        ],
        'H' | 'h' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' | 'i' => [
            0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        'J' | 'j' => [
            0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100,
        ],
        'K' | 'k' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'L' | 'l' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' | 'm' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' | 'n' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'O' | 'o' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'P' | 'p' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'Q' | 'q' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
        'R' | 'r' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' | 's' => [
            0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110,
        ],
        'T' | 't' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'U' | 'u' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'V' | 'v' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
        'W' | 'w' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
        ],
        'X' | 'x' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
        'Y' | 'y' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'Z' | 'z' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111,
        ],
        '3' => [
            0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
        '6' => [
            0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110,
        ],
        '.' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100,
        ],
        '-' => [
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ],
        ' ' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000,
        ],
        '+' => [
            0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000,
        ],
        ':' => [
            0b00000, 0b00000, 0b00100, 0b00000, 0b00100, 0b00000, 0b00000,
        ],
        '(' => [
            0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010,
        ],
        ')' => [
            0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000,
        ],
        ',' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b01000,
        ],
        '=' => [
            0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000,
        ],
        _ => return None,
    };
    Some(g)
}

// ---------------------------------------------------------------------------
// Framebuffer
// ---------------------------------------------------------------------------

/// RGBA pixel buffer used during rasterization.
struct Framebuffer {
    width: usize,
    height: usize,
    /// Row-major RGBA, 4 bytes per pixel.
    pixels: Vec<u8>,
}

impl Framebuffer {
    /// Create a new framebuffer filled with `bg`.
    fn new(width: usize, height: usize, bg: Color) -> Self {
        let len = width.checked_mul(height).unwrap_or(0) * 4;
        let mut pixels = vec![0u8; len];
        for chunk in pixels.chunks_exact_mut(4) {
            chunk[0] = bg.r;
            chunk[1] = bg.g;
            chunk[2] = bg.b;
            chunk[3] = bg.a;
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Alpha-blend `color` onto the pixel at `(x, y)`.
    fn set_pixel(&mut self, x: usize, y: usize, color: Color) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = (y * self.width + x) * 4;
        if color.a == 255 {
            self.pixels[idx] = color.r;
            self.pixels[idx + 1] = color.g;
            self.pixels[idx + 2] = color.b;
            self.pixels[idx + 3] = 255;
        } else if color.a > 0 {
            let sa = u32::from(color.a);
            let da = u32::from(self.pixels[idx + 3]);
            let out_a = sa + da * (255 - sa) / 255;
            if out_a == 0 {
                return;
            }
            let blend = |src: u8, dst: u8| -> u8 {
                let s = u32::from(src);
                let d = u32::from(dst);
                ((s * sa + d * da * (255 - sa) / 255) / out_a).min(255) as u8
            };
            self.pixels[idx] = blend(color.r, self.pixels[idx]);
            self.pixels[idx + 1] = blend(color.g, self.pixels[idx + 1]);
            self.pixels[idx + 2] = blend(color.b, self.pixels[idx + 2]);
            self.pixels[idx + 3] = out_a.min(255) as u8;
        }
    }

    /// Read the color of a pixel (for testing).
    #[cfg(test)]
    fn get_pixel(&self, x: usize, y: usize) -> Color {
        if x >= self.width || y >= self.height {
            return Color::TRANSPARENT;
        }
        let idx = (y * self.width + x) * 4;
        Color::rgba(
            self.pixels[idx],
            self.pixels[idx + 1],
            self.pixels[idx + 2],
            self.pixels[idx + 3],
        )
    }
}

// ---------------------------------------------------------------------------
// Drawing primitives
// ---------------------------------------------------------------------------

/// Draw a line using Bresenham's algorithm with variable thickness.
fn draw_line(fb: &mut Framebuffer, x1: f64, y1: f64, x2: f64, y2: f64, color: Color, width: f64) {
    if width <= 1.0 {
        draw_line_thin(fb, x1, y1, x2, y2, color);
    } else {
        draw_line_thick(fb, x1, y1, x2, y2, color, width);
    }
}

/// Bresenham's line (1-pixel wide).
fn draw_line_thin(fb: &mut Framebuffer, x1: f64, y1: f64, x2: f64, y2: f64, color: Color) {
    let mut ix1 = x1.round() as i64;
    let mut iy1 = y1.round() as i64;
    let ix2 = x2.round() as i64;
    let iy2 = y2.round() as i64;

    let dx = (ix2 - ix1).abs();
    let dy = -(iy2 - iy1).abs();
    let sx: i64 = if ix1 < ix2 { 1 } else { -1 };
    let sy: i64 = if iy1 < iy2 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if ix1 >= 0 && iy1 >= 0 {
            fb.set_pixel(ix1 as usize, iy1 as usize, color);
        }
        if ix1 == ix2 && iy1 == iy2 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            ix1 += sx;
        }
        if e2 <= dx {
            err += dx;
            iy1 += sy;
        }
    }
}

/// Thick line: draw a 1-pixel Bresenham and stamp a filled circle at each pixel.
fn draw_line_thick(
    fb: &mut Framebuffer,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    color: Color,
    width: f64,
) {
    let half = (width / 2.0).max(0.5);
    let mut ix1 = x1.round() as i64;
    let mut iy1 = y1.round() as i64;
    let ix2 = x2.round() as i64;
    let iy2 = y2.round() as i64;

    let dx = (ix2 - ix1).abs();
    let dy = -(iy2 - iy1).abs();
    let sx: i64 = if ix1 < ix2 { 1 } else { -1 };
    let sy: i64 = if iy1 < iy2 { 1 } else { -1 };
    let mut err = dx + dy;

    let r_sq = half * half;
    let r_ceil = half.ceil() as i64;

    loop {
        // Stamp a filled disc centered at (ix1, iy1).
        for oy in -r_ceil..=r_ceil {
            for ox in -r_ceil..=r_ceil {
                if (ox * ox + oy * oy) as f64 <= r_sq {
                    let px = ix1 + ox;
                    let py = iy1 + oy;
                    if px >= 0 && py >= 0 {
                        fb.set_pixel(px as usize, py as usize, color);
                    }
                }
            }
        }
        if ix1 == ix2 && iy1 == iy2 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            ix1 += sx;
        }
        if e2 <= dx {
            err += dx;
            iy1 += sy;
        }
    }
}

/// Fill an axis-aligned rectangle.
fn fill_rect(fb: &mut Framebuffer, x: f64, y: f64, w: f64, h: f64, color: Color) {
    let x0 = x.round().max(0.0) as usize;
    let y0 = y.round().max(0.0) as usize;
    let x1 = (x + w).round().min(fb.width as f64) as usize;
    let y1 = (y + h).round().min(fb.height as f64) as usize;
    for py in y0..y1 {
        for px in x0..x1 {
            fb.set_pixel(px, py, color);
        }
    }
}

/// Stroke a rectangle outline.
fn stroke_rect(fb: &mut Framebuffer, x: f64, y: f64, w: f64, h: f64, color: Color, width: f64) {
    let x2 = x + w;
    let y2 = y + h;
    draw_line(fb, x, y, x2, y, color, width);
    draw_line(fb, x2, y, x2, y2, color, width);
    draw_line(fb, x2, y2, x, y2, color, width);
    draw_line(fb, x, y2, x, y, color, width);
}

/// Fill a circle using the midpoint algorithm.
fn fill_circle(fb: &mut Framebuffer, cx: f64, cy: f64, r: f64, color: Color) {
    let icx = cx.round() as i64;
    let icy = cy.round() as i64;
    let ir = r.round() as i64;
    if ir <= 0 {
        #[allow(clippy::collapsible_if)]
        if ir == 0 {
            if icx >= 0 && icy >= 0 {
                fb.set_pixel(icx as usize, icy as usize, color);
            }
        }
        return;
    }

    let mut x: i64 = 0;
    let mut y: i64 = ir;
    let mut d: i64 = 1 - ir;

    #[allow(clippy::cast_possible_wrap)]
    let scanline = |fb: &mut Framebuffer, py: i64, px_left: i64, px_right: i64| {
        if py < 0 || py >= fb.height as i64 {
            return;
        }
        let left = px_left.max(0) as usize;
        let right = (px_right + 1).min(fb.width as i64) as usize;
        for px in left..right {
            fb.set_pixel(px, py as usize, color);
        }
    };

    while x <= y {
        scanline(fb, icy + y, icx - x, icx + x);
        scanline(fb, icy - y, icx - x, icx + x);
        scanline(fb, icy + x, icx - y, icx + y);
        scanline(fb, icy - x, icx - y, icx + y);

        x += 1;
        if d < 0 {
            d += 2 * x + 1;
        } else {
            y -= 1;
            d += 2 * (x - y) + 1;
        }
    }
}

/// Stroke a circle outline.
fn stroke_circle(fb: &mut Framebuffer, cx: f64, cy: f64, r: f64, color: Color, width: f64) {
    let outer = r + width / 2.0;
    let inner = (r - width / 2.0).max(0.0);
    let outer_sq = outer * outer;
    let inner_sq = inner * inner;
    let icx = cx.round() as i64;
    let icy = cy.round() as i64;
    let r_ceil = outer.ceil() as i64;

    for oy in -r_ceil..=r_ceil {
        for ox in -r_ceil..=r_ceil {
            let dist_sq = (ox * ox + oy * oy) as f64;
            if dist_sq <= outer_sq && dist_sq >= inner_sq {
                let px = icx + ox;
                let py = icy + oy;
                if px >= 0 && py >= 0 {
                    fb.set_pixel(px as usize, py as usize, color);
                }
            }
        }
    }
}

/// Fill a polygon using scanline rasterization.
fn fill_polygon(fb: &mut Framebuffer, points: &[(f64, f64)], color: Color) {
    if points.len() < 3 {
        return;
    }
    // Find bounding box.
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    for &(_x, y) in points {
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    }
    let y_start = min_y.floor().max(0.0) as i64;
    let y_end = max_y.ceil().min(fb.height as f64) as i64;

    let n = points.len();
    let mut intersections = Vec::with_capacity(16);

    for scan_y in y_start..y_end {
        let sy = scan_y as f64 + 0.5;
        intersections.clear();

        for i in 0..n {
            let j = (i + 1) % n;
            let (_, y0) = points[i];
            let (_, y1) = points[j];
            if (y0 <= sy && y1 > sy) || (y1 <= sy && y0 > sy) {
                let (x0, _) = points[i];
                let (x1, _) = points[j];
                let t = (sy - y0) / (y1 - y0);
                intersections.push(x0 + t * (x1 - x0));
            }
        }

        intersections
            .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

        let mut k = 0;
        while k + 1 < intersections.len() {
            let left = intersections[k].max(0.0) as usize;
            let right = intersections[k + 1].min(fb.width as f64) as usize;
            for px in left..right {
                fb.set_pixel(px, scan_y as usize, color);
            }
            k += 2;
        }
    }
}

/// Draw connected line segments.
fn draw_polyline(fb: &mut Framebuffer, points: &[(f64, f64)], color: Color, width: f64) {
    for pair in points.windows(2) {
        draw_line(fb, pair[0].0, pair[0].1, pair[1].0, pair[1].1, color, width);
    }
}

/// Draw text using the built-in bitmap font.
fn draw_text(
    fb: &mut Framebuffer,
    x: f64,
    y: f64,
    text: &str,
    font_size: f64,
    anchor: TextAnchor,
    color: Color,
) {
    let scale = (font_size / GLYPH_H as f64).max(1.0).round() as usize;
    let char_w = (GLYPH_W + GLYPH_SPACING) * scale;
    let total_w = text.len() * char_w;

    let start_x = match anchor {
        TextAnchor::Start => x.round() as i64,
        TextAnchor::Middle => (x - total_w as f64 / 2.0).round() as i64,
        TextAnchor::End => (x - total_w as f64).round() as i64,
    };
    // Vertically center on the given y coordinate.
    let start_y = (y - (GLYPH_H * scale) as f64 / 2.0).round() as i64;

    #[allow(clippy::cast_possible_wrap)]
    for (ci, ch) in text.chars().enumerate() {
        let Some(rows) = glyph(ch) else {
            continue;
        };
        let gx = start_x + (ci * char_w) as i64;
        for (row_idx, &row) in rows.iter().enumerate() {
            for col in 0..GLYPH_W {
                if row & (1 << (GLYPH_W - 1 - col)) != 0 {
                    let px_base = gx + (col * scale) as i64;
                    let py_base = start_y + (row_idx * scale) as i64;
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = px_base + sx as i64;
                            let py = py_base + sy as i64;
                            if px >= 0 && py >= 0 {
                                fb.set_pixel(px as usize, py as usize, color);
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Element rendering
// ---------------------------------------------------------------------------

fn render_element(fb: &mut Framebuffer, element: &Element) {
    match element {
        Element::Line {
            x1,
            y1,
            x2,
            y2,
            stroke,
        } => {
            draw_line(fb, *x1, *y1, *x2, *y2, stroke.color, stroke.width);
        }
        Element::Rect {
            x,
            y,
            w,
            h,
            fill,
            stroke,
        } => {
            if let Some(f) = fill {
                fill_rect(fb, *x, *y, *w, *h, f.color);
            }
            if let Some(s) = stroke {
                stroke_rect(fb, *x, *y, *w, *h, s.color, s.width);
            }
        }
        Element::Circle {
            cx,
            cy,
            r,
            fill,
            stroke,
        } => {
            if let Some(f) = fill {
                fill_circle(fb, *cx, *cy, *r, f.color);
            }
            if let Some(s) = stroke {
                stroke_circle(fb, *cx, *cy, *r, s.color, s.width);
            }
        }
        Element::Text {
            x,
            y,
            text,
            font,
            anchor,
        } => {
            draw_text(fb, *x, *y, text, font.size, *anchor, font.color);
        }
        Element::Polyline {
            points,
            stroke,
            fill,
        } => {
            if let Some(f) = fill {
                fill_polygon(fb, points, f.color);
            }
            draw_polyline(fb, points, stroke.color, stroke.width);
        }
        Element::Group { elements } => {
            for child in elements {
                render_element(fb, child);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CRC-32 (ISO/HDLC)
// ---------------------------------------------------------------------------

/// Build a standard CRC-32 lookup table.
const fn make_crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut n = 0usize;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            if c & 1 != 0 {
                c = 0xEDB8_8320 ^ (c >> 1);
            } else {
                c >>= 1;
            }
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
}

static CRC_TABLE: [u32; 256] = make_crc_table();

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc = CRC_TABLE[((crc ^ u32::from(b)) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

// ---------------------------------------------------------------------------
// Adler-32
// ---------------------------------------------------------------------------

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

// ---------------------------------------------------------------------------
// PNG encoder
// ---------------------------------------------------------------------------

/// Write a PNG chunk to `out`.
fn write_chunk(out: &mut Vec<u8>, chunk_type: [u8; 4], data: &[u8]) {
    let len = data.len() as u32;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(&chunk_type);
    out.extend_from_slice(data);
    // CRC covers type + data.
    let mut crc_data = Vec::with_capacity(4 + data.len());
    crc_data.extend_from_slice(&chunk_type);
    crc_data.extend_from_slice(data);
    let crc = crc32(&crc_data);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// Encode raw RGBA data to PNG.
fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba.len() + 1024);

    // PNG signature.
    out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);

    // IHDR chunk.
    {
        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.push(8); // bit depth
        ihdr.push(6); // color type RGBA
        ihdr.push(0); // compression method
        ihdr.push(0); // filter method
        ihdr.push(0); // interlace method
        write_chunk(&mut out, *b"IHDR", &ihdr);
    }

    // Prepare raw scanline data (filter byte 0 + RGBA row).
    let row_bytes = width as usize * 4;
    let raw_len = height as usize * (1 + row_bytes);
    let mut raw = Vec::with_capacity(raw_len);
    for y in 0..height as usize {
        raw.push(0); // filter type: None
        let start = y * row_bytes;
        raw.extend_from_slice(&rgba[start..start + row_bytes]);
    }

    // Wrap in zlib: header + stored deflate blocks + adler32.
    let adler = adler32(&raw);
    let mut zlib = Vec::with_capacity(raw.len() + 64);
    // zlib header: CM=8 (deflate), CINFO=7 (32K window), FCHECK so header%31==0
    zlib.push(0x78);
    zlib.push(0x01);

    // Stored deflate blocks (max 65535 bytes each).
    let max_block: usize = 65535;
    let mut offset = 0;
    while offset < raw.len() {
        let remaining = raw.len() - offset;
        let block_len = remaining.min(max_block);
        let is_final = offset + block_len >= raw.len();
        zlib.push(u8::from(is_final));
        let len16 = block_len as u16;
        zlib.extend_from_slice(&len16.to_le_bytes());
        zlib.extend_from_slice(&(!len16).to_le_bytes());
        zlib.extend_from_slice(&raw[offset..offset + block_len]);
        offset += block_len;
    }

    zlib.extend_from_slice(&adler.to_be_bytes());

    write_chunk(&mut out, *b"IDAT", &zlib);

    // IEND chunk.
    write_chunk(&mut out, *b"IEND", &[]);

    out
}

// ---------------------------------------------------------------------------
// Base64 encoder
// ---------------------------------------------------------------------------

const BASE64_CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 2 < data.len() {
        let b0 = data[i];
        let b1 = data[i + 1];
        let b2 = data[i + 2];
        out.push(BASE64_CHARS[(b0 >> 2) as usize] as char);
        out.push(BASE64_CHARS[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        out.push(BASE64_CHARS[(((b1 & 0x0F) << 2) | (b2 >> 6)) as usize] as char);
        out.push(BASE64_CHARS[(b2 & 0x3F) as usize] as char);
        i += 3;
    }
    let remaining = data.len() - i;
    if remaining == 2 {
        let b0 = data[i];
        let b1 = data[i + 1];
        out.push(BASE64_CHARS[(b0 >> 2) as usize] as char);
        out.push(BASE64_CHARS[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        out.push(BASE64_CHARS[((b1 & 0x0F) << 2) as usize] as char);
        out.push('=');
    } else if remaining == 1 {
        let b0 = data[i];
        out.push(BASE64_CHARS[(b0 >> 2) as usize] as char);
        out.push(BASE64_CHARS[((b0 & 0x03) << 4) as usize] as char);
        out.push('=');
        out.push('=');
    }
    out
}

// ---------------------------------------------------------------------------
// BitmapBackend
// ---------------------------------------------------------------------------

/// Software-rasterizer backend that renders [`Element`]s to an RGBA pixel
/// buffer and encodes the result as PNG.
///
/// The rasterizer is entirely self-contained: CRC-32, Adler-32, stored-deflate
/// PNG encoding, base64, and a 5×7 bitmap font are all implemented from
/// scratch with zero external dependencies.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct BitmapBackend {
    background: Color,
    antialias: bool,
}

impl BitmapBackend {
    /// Create a new bitmap backend with a white background and antialiasing
    /// enabled.
    #[must_use]
    pub fn new() -> Self {
        Self {
            background: Color::WHITE,
            antialias: true,
        }
    }

    /// Set the background color.
    #[must_use]
    pub fn background(mut self, c: Color) -> Self {
        self.background = c;
        self
    }

    /// Enable or disable antialiasing (currently a hint; the software
    /// rasterizer uses basic Bresenham drawing).
    #[must_use]
    pub fn antialias(mut self, v: bool) -> Self {
        self.antialias = v;
        self
    }

    /// Render elements to raw RGBA pixel data.
    #[must_use]
    pub fn render_rgba(&self, elements: &[Element], width: u32, height: u32) -> Vec<u8> {
        let mut fb = Framebuffer::new(width as usize, height as usize, self.background);
        for elem in elements {
            render_element(&mut fb, elem);
        }
        fb.pixels
    }

    /// Render elements to PNG bytes.
    ///
    /// # Errors
    ///
    /// Returns `VizError::InvalidParameter` if width or height is zero.
    pub fn render_png(&self, elements: &[Element], width: u32, height: u32) -> Result<Vec<u8>> {
        if width == 0 || height == 0 {
            return Err(crate::error::VizError::InvalidParameter {
                name: "dimensions",
                reason: "width and height must be greater than zero",
            });
        }
        let rgba = self.render_rgba(elements, width, height);
        Ok(encode_png(width, height, &rgba))
    }

    /// Render elements and write the result to a PNG file.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimensions are invalid or the file cannot be
    /// written.
    pub fn to_file(&self, elements: &[Element], width: u32, height: u32, path: &str) -> Result<()> {
        let png = self.render_png(elements, width, height)?;
        std::fs::write(path, png)?;
        Ok(())
    }
}

impl Default for BitmapBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer for BitmapBackend {
    fn render(&self, elements: &[Element], width: f64, height: f64) -> Result<String> {
        let png = self.render_png(elements, width as u32, height as u32)?;
        Ok(base64_encode(&png))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::{Fill, Font, Stroke};

    #[test]
    fn framebuffer_set_pixel() {
        let mut fb = Framebuffer::new(10, 10, Color::WHITE);
        fb.set_pixel(3, 5, Color::RED);
        let c = fb.get_pixel(3, 5);
        assert_eq!(c, Color::RED);
        // Surrounding pixels unchanged.
        assert_eq!(fb.get_pixel(0, 0), Color::WHITE);
    }

    #[test]
    fn framebuffer_out_of_bounds() {
        let mut fb = Framebuffer::new(5, 5, Color::BLACK);
        // Should not panic.
        fb.set_pixel(100, 100, Color::RED);
        let c = fb.get_pixel(100, 100);
        assert_eq!(c, Color::TRANSPARENT);
    }

    #[test]
    fn draw_line_horizontal() {
        let mut fb = Framebuffer::new(20, 10, Color::WHITE);
        draw_line(&mut fb, 2.0, 5.0, 18.0, 5.0, Color::BLACK, 1.0);
        // All pixels on row 5 between x=2 and x=18 should be black.
        for x in 2..=18 {
            assert_eq!(fb.get_pixel(x, 5), Color::BLACK, "pixel ({x}, 5)");
        }
        // Pixel outside line should be white.
        assert_eq!(fb.get_pixel(0, 5), Color::WHITE);
    }

    #[test]
    fn fill_rect_basic() {
        let mut fb = Framebuffer::new(20, 20, Color::WHITE);
        fill_rect(&mut fb, 5.0, 5.0, 10.0, 10.0, Color::BLUE);
        assert_eq!(fb.get_pixel(5, 5), Color::BLUE);
        assert_eq!(fb.get_pixel(14, 14), Color::BLUE);
        assert_eq!(fb.get_pixel(4, 4), Color::WHITE);
    }

    #[test]
    fn fill_circle_basic() {
        let mut fb = Framebuffer::new(30, 30, Color::WHITE);
        fill_circle(&mut fb, 15.0, 15.0, 8.0, Color::RED);
        assert_eq!(fb.get_pixel(15, 15), Color::RED);
        // A corner far away should be untouched.
        assert_eq!(fb.get_pixel(0, 0), Color::WHITE);
    }

    #[test]
    fn png_signature() {
        let backend = BitmapBackend::new();
        let png = backend.render_png(&[], 4, 4).unwrap();
        assert_eq!(&png[..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn render_returns_base64() {
        let backend = BitmapBackend::new();
        let result = backend.render(&[], 4.0, 4.0).unwrap();
        // Base64 string should start with the encoded PNG signature.
        assert!(!result.is_empty());
        // Must only contain valid base64 characters.
        assert!(
            result
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=')
        );
    }

    #[test]
    fn draw_elements() {
        let elements = vec![
            Element::Rect {
                x: 10.0,
                y: 10.0,
                w: 80.0,
                h: 60.0,
                fill: Some(Fill::new(Color::BLUE)),
                stroke: Some(Stroke::new(Color::BLACK, 1.0)),
            },
            Element::Circle {
                cx: 50.0,
                cy: 40.0,
                r: 15.0,
                fill: Some(Fill::new(Color::RED)),
                stroke: None,
            },
            Element::Line {
                x1: 0.0,
                y1: 0.0,
                x2: 100.0,
                y2: 80.0,
                stroke: Stroke::new(Color::GREEN, 2.0),
            },
            Element::Text {
                x: 50.0,
                y: 70.0,
                text: "HELLO".to_string(),
                font: Font::default(),
                anchor: TextAnchor::Middle,
            },
            Element::Polyline {
                points: vec![(0.0, 0.0), (50.0, 80.0), (100.0, 0.0)],
                stroke: Stroke::new(Color::BLACK, 1.0),
                fill: Some(Fill::new(Color::rgba(255, 255, 0, 128))),
            },
            Element::Group {
                elements: vec![Element::Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 5.0,
                    h: 5.0,
                    fill: Some(Fill::new(Color::WHITE)),
                    stroke: None,
                }],
            },
        ];
        let backend = BitmapBackend::new();
        let png = backend.render_png(&elements, 100, 80).unwrap();
        assert!(png.len() > 8);
        assert_eq!(&png[..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn alpha_blending() {
        let mut fb = Framebuffer::new(10, 10, Color::WHITE);
        fb.set_pixel(5, 5, Color::rgba(0, 0, 0, 128));
        let c = fb.get_pixel(5, 5);
        // Should be a grayish pixel (not pure black, not pure white).
        assert!(c.r > 0 && c.r < 255);
    }

    #[test]
    fn builder_pattern() {
        let b = BitmapBackend::new()
            .background(Color::BLACK)
            .antialias(false);
        assert_eq!(b.background, Color::BLACK);
        assert!(!b.antialias);
    }

    #[test]
    fn zero_dimensions_error() {
        let backend = BitmapBackend::new();
        assert!(backend.render_png(&[], 0, 10).is_err());
        assert!(backend.render_png(&[], 10, 0).is_err());
    }

    #[test]
    fn crc32_known_value() {
        // CRC-32 of "IEND" is a well-known constant.
        let crc = crc32(b"IEND");
        assert_eq!(crc, 0xAE42_6082);
    }

    #[test]
    fn base64_encode_empty() {
        assert_eq!(base64_encode(&[]), "");
    }

    #[test]
    fn base64_encode_known() {
        assert_eq!(base64_encode(b"Hello"), "SGVsbG8=");
    }

    #[test]
    fn stroke_circle_draws() {
        let mut fb = Framebuffer::new(40, 40, Color::WHITE);
        stroke_circle(&mut fb, 20.0, 20.0, 10.0, Color::BLACK, 1.0);
        // Center should still be white (hollow circle).
        assert_eq!(fb.get_pixel(20, 20), Color::WHITE);
        // A point on the circumference should be black.
        assert_eq!(fb.get_pixel(30, 20), Color::BLACK);
    }

    #[test]
    fn polygon_fill() {
        let mut fb = Framebuffer::new(20, 20, Color::WHITE);
        let triangle = [(10.0, 2.0), (2.0, 18.0), (18.0, 18.0)];
        fill_polygon(&mut fb, &triangle, Color::RED);
        // Centroid (10, 12-ish) should be filled.
        assert_eq!(fb.get_pixel(10, 12), Color::RED);
        // Corner far from triangle should be white.
        assert_eq!(fb.get_pixel(0, 0), Color::WHITE);
    }
}
