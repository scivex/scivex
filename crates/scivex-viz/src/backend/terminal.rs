use super::Renderer;
use crate::element::Element;
use crate::error::Result;

/// Renders elements to terminal using Unicode braille characters (U+2800..U+28FF).
///
/// Each character cell represents a 2-wide × 4-tall dot grid.
///
/// Braille dot bit mapping per character cell:
/// ```text
/// Dot 0 (0x01)  Dot 3 (0x08)
/// Dot 1 (0x02)  Dot 4 (0x10)
/// Dot 2 (0x04)  Dot 5 (0x20)
/// Dot 6 (0x40)  Dot 7 (0x80)
/// ```
#[derive(Debug, Clone)]
pub struct TerminalBackend {
    /// Terminal width in character columns.
    pub cols: usize,
    /// Terminal height in character rows.
    pub rows: usize,
}

impl TerminalBackend {
    /// Create a backend with the given terminal dimensions.
    #[must_use]
    pub fn new(cols: usize, rows: usize) -> Self {
        Self { cols, rows }
    }
}

impl Default for TerminalBackend {
    fn default() -> Self {
        Self::new(80, 24)
    }
}

/// Bit canvas: a 2-D grid of booleans, where each `(x, y)` is a dot.
struct BrailleCanvas {
    width: usize,
    height: usize,
    dots: Vec<bool>,
}

impl BrailleCanvas {
    fn new(char_cols: usize, char_rows: usize) -> Self {
        let width = char_cols * 2;
        let height = char_rows * 4;
        Self {
            width,
            height,
            dots: vec![false; width * height],
        }
    }

    fn set(&mut self, x: i64, y: i64) {
        if x >= 0 && y >= 0 {
            let (x, y) = (x as usize, y as usize);
            if x < self.width && y < self.height {
                self.dots[y * self.width + x] = true;
            }
        }
    }

    /// Draw a line using Bresenham's algorithm.
    fn line(&mut self, mut x0: i64, mut y0: i64, x1: i64, y1: i64) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx: i64 = if x0 < x1 { 1 } else { -1 };
        let sy: i64 = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            self.set(x0, y0);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    /// Draw a filled rectangle.
    fn fill_rect(&mut self, x: i64, y: i64, w: i64, h: i64) {
        for dy in 0..h {
            for dx in 0..w {
                self.set(x + dx, y + dy);
            }
        }
    }

    /// Draw a circle outline using the midpoint algorithm.
    fn circle(&mut self, cx: i64, cy: i64, r: i64) {
        let mut x = r;
        let mut y: i64 = 0;
        let mut d = 1 - r;
        while x >= y {
            self.set(cx + x, cy + y);
            self.set(cx - x, cy + y);
            self.set(cx + x, cy - y);
            self.set(cx - x, cy - y);
            self.set(cx + y, cy + x);
            self.set(cx - y, cy + x);
            self.set(cx + y, cy - x);
            self.set(cx - y, cy - x);
            y += 1;
            if d <= 0 {
                d += 2 * y + 1;
            } else {
                x -= 1;
                d += 2 * (y - x) + 1;
            }
        }
    }

    /// Convert the dot grid to a string of braille characters.
    fn render(&self) -> String {
        let char_cols = self.width / 2;
        let char_rows = self.height / 4;
        let mut output = String::with_capacity(char_rows * (char_cols + 1));

        for cr in 0..char_rows {
            for cc in 0..char_cols {
                let px = cc * 2;
                let py = cr * 4;
                let mut code: u32 = 0x2800;

                // Map dots to braille bit pattern.
                if self.dot(px, py) {
                    code |= 0x01;
                }
                if self.dot(px, py + 1) {
                    code |= 0x02;
                }
                if self.dot(px, py + 2) {
                    code |= 0x04;
                }
                if self.dot(px + 1, py) {
                    code |= 0x08;
                }
                if self.dot(px + 1, py + 1) {
                    code |= 0x10;
                }
                if self.dot(px + 1, py + 2) {
                    code |= 0x20;
                }
                if self.dot(px, py + 3) {
                    code |= 0x40;
                }
                if self.dot(px + 1, py + 3) {
                    code |= 0x80;
                }

                if let Some(ch) = char::from_u32(code) {
                    output.push(ch);
                }
            }
            output.push('\n');
        }
        output
    }

    fn dot(&self, x: usize, y: usize) -> bool {
        if x < self.width && y < self.height {
            self.dots[y * self.width + x]
        } else {
            false
        }
    }
}

impl Renderer for TerminalBackend {
    fn render(&self, elements: &[Element], width: f64, height: f64) -> Result<String> {
        let mut canvas = BrailleCanvas::new(self.cols, self.rows);

        let sx = (self.cols * 2) as f64 / width;
        let sy = (self.rows * 4) as f64 / height;

        for elem in elements {
            render_element(&mut canvas, elem, sx, sy);
        }

        Ok(canvas.render())
    }
}

fn render_element(canvas: &mut BrailleCanvas, elem: &Element, sx: f64, sy: f64) {
    match elem {
        Element::Line { x1, y1, x2, y2, .. } => {
            canvas.line(
                (x1 * sx) as i64,
                (y1 * sy) as i64,
                (x2 * sx) as i64,
                (y2 * sy) as i64,
            );
        }
        Element::Rect { x, y, w, h, .. } => {
            canvas.fill_rect(
                (x * sx) as i64,
                (y * sy) as i64,
                (w * sx) as i64,
                (h * sy) as i64,
            );
        }
        Element::Circle { cx, cy, r, .. } => {
            canvas.circle((cx * sx) as i64, (cy * sy) as i64, (r * sx) as i64);
        }
        Element::Polyline { points, .. } => {
            for pair in points.windows(2) {
                canvas.line(
                    (pair[0].0 * sx) as i64,
                    (pair[0].1 * sy) as i64,
                    (pair[1].0 * sx) as i64,
                    (pair[1].1 * sy) as i64,
                );
            }
        }
        Element::Text { .. } => {
            // Text is not rendered in braille mode — too coarse.
        }
        Element::Group { elements } => {
            for child in elements {
                render_element(canvas, child, sx, sy);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::style::Stroke;

    #[test]
    fn terminal_backend_default() {
        let tb = TerminalBackend::default();
        assert_eq!(tb.cols, 80);
        assert_eq!(tb.rows, 24);
    }

    #[test]
    fn terminal_backend_custom_size() {
        let tb = TerminalBackend::new(120, 40);
        assert_eq!(tb.cols, 120);
        assert_eq!(tb.rows, 40);
    }

    #[test]
    fn render_empty_elements() {
        let tb = TerminalBackend::default();
        let output = tb.render(&[], 100.0, 100.0).unwrap();
        // All blank braille (U+2800)
        assert!(output.chars().all(|c| c == '\u{2800}' || c == '\n'));
    }

    #[test]
    fn render_line_has_braille() {
        let elements = vec![Element::Line {
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 100.0,
            stroke: Stroke::new(Color::BLACK, 1.0),
        }];
        let backend = TerminalBackend::new(40, 12);
        let output = backend.render(&elements, 100.0, 100.0).unwrap();
        // Output should contain braille characters (U+2800..U+28FF).
        assert!(
            output
                .chars()
                .any(|c| ('\u{2800}'..='\u{28FF}').contains(&c))
        );
    }
}
