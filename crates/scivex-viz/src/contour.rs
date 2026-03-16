use crate::color::ColorMap;
use crate::element::Element;
use crate::layout::Rect;
use crate::plot::{AxisRange, PlotBuilder};
use crate::scale::Scale;
use crate::style::Stroke;

/// A contour plot using marching squares to extract iso-lines from 2-D data.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ContourPlot {
    data: Vec<Vec<f64>>,
    n_levels: usize,
    colormap: ColorMap,
    stroke_width: f64,
    plot_label: Option<String>,
}

impl ContourPlot {
    /// Create a contour plot from a 2-D grid and a number of contour levels.
    #[must_use]
    pub fn new(data: Vec<Vec<f64>>, n_levels: usize) -> Self {
        Self {
            data,
            n_levels,
            colormap: ColorMap::viridis(),
            stroke_width: 1.0,
            plot_label: None,
        }
    }

    /// Set the colormap used to color contour levels.
    #[must_use]
    pub fn colormap(mut self, cm: ColorMap) -> Self {
        self.colormap = cm;
        self
    }

    /// Set the contour line width.
    #[must_use]
    pub fn stroke_width(mut self, w: f64) -> Self {
        self.stroke_width = w;
        self
    }

    /// Set the legend label for this contour plot.
    #[must_use]
    pub fn label(mut self, l: &str) -> Self {
        self.plot_label = Some(l.to_string());
        self
    }

    /// Return `(min, max)` of all data values.
    fn data_bounds(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for row in &self.data {
            for &v in row {
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
        }
        if !lo.is_finite() {
            lo = 0.0;
        }
        if !hi.is_finite() {
            hi = 1.0;
        }
        if (hi - lo).abs() < f64::EPSILON {
            hi = lo + 1.0;
        }
        (lo, hi)
    }
}

/// Linearly interpolate the position where `threshold` crosses between `v0` and `v1`.
/// Returns a value in `[0.0, 1.0]`.
fn lerp_pos(v0: f64, v1: f64, threshold: f64) -> f64 {
    let denom = v1 - v0;
    if denom.abs() < f64::EPSILON {
        0.5
    } else {
        ((threshold - v0) / denom).clamp(0.0, 1.0)
    }
}

/// Run marching squares for a single contour level on the grid.
/// Returns a list of line segments as `((x0, y0), (x1, y1))` in grid coordinates.
#[allow(clippy::too_many_lines)]
fn marching_squares(data: &[Vec<f64>], threshold: f64) -> Vec<((f64, f64), (f64, f64))> {
    let n_rows = data.len();
    if n_rows < 2 {
        return vec![];
    }
    let n_cols = data.first().map_or(0, Vec::len);
    if n_cols < 2 {
        return vec![];
    }

    let mut segments = Vec::new();

    for r in 0..n_rows - 1 {
        for c in 0..n_cols - 1 {
            let tl = data[r][c];
            let tr = data[r][c + 1];
            let br = data[r + 1][c + 1];
            let bl = data[r + 1][c];

            // Classify corners: 1 if above threshold, 0 if below
            let case = (u8::from(tl >= threshold) << 3)
                | (u8::from(tr >= threshold) << 2)
                | (u8::from(br >= threshold) << 1)
                | u8::from(bl >= threshold);

            if case == 0 || case == 15 {
                continue; // all above or all below
            }

            let x = c as f64;
            let y = r as f64;

            // Edge midpoints with interpolation
            let top = (x + lerp_pos(tl, tr, threshold), y);
            let right = (x + 1.0, y + lerp_pos(tr, br, threshold));
            let bottom = (x + lerp_pos(bl, br, threshold), y + 1.0);
            let left = (x, y + lerp_pos(tl, bl, threshold));

            match case {
                1 | 14 => segments.push((left, bottom)),
                2 | 13 => segments.push((bottom, right)),
                3 | 12 => segments.push((left, right)),
                4 | 11 => segments.push((top, right)),
                5 => {
                    // Saddle point — ambiguous, use simple resolution
                    segments.push((top, left));
                    segments.push((bottom, right));
                }
                6 | 9 => segments.push((top, bottom)),
                7 | 8 => segments.push((top, left)),
                10 => {
                    // Saddle point — ambiguous, use simple resolution
                    segments.push((top, right));
                    segments.push((bottom, left));
                }
                _ => {}
            }
        }
    }

    segments
}

impl PlotBuilder for ContourPlot {
    fn data_range(&self) -> (AxisRange, AxisRange) {
        let n_rows = self.data.len();
        let n_cols = self.data.first().map_or(0, Vec::len);
        let x_range = if n_cols > 1 {
            Some((0.0, (n_cols - 1) as f64))
        } else {
            None
        };
        let y_range = if n_rows > 1 {
            Some((0.0, (n_rows - 1) as f64))
        } else {
            None
        };
        (x_range, y_range)
    }

    fn build_elements(&self, x_scale: &dyn Scale, y_scale: &dyn Scale, area: Rect) -> Vec<Element> {
        let n_rows = self.data.len();
        if n_rows < 2 {
            return vec![];
        }
        let n_cols = self.data.first().map_or(0, Vec::len);
        if n_cols < 2 {
            return vec![];
        }
        if self.n_levels == 0 {
            return vec![];
        }

        let (lo, hi) = self.data_bounds();
        let mut elements = Vec::new();

        for level_idx in 0..self.n_levels {
            let t = (level_idx as f64 + 0.5) / self.n_levels as f64;
            let threshold = lo + t * (hi - lo);
            let color = self.colormap.sample(t);

            let segments = marching_squares(&self.data, threshold);
            for ((x0, y0), (x1, y1)) in &segments {
                // Note: y in grid coords grows downward, but y_scale inverts
                let px0 = area.x + x_scale.transform(*x0) * area.w;
                let py0 = area.y + area.h - y_scale.transform(*y0) * area.h;
                let px1 = area.x + x_scale.transform(*x1) * area.w;
                let py1 = area.y + area.h - y_scale.transform(*y1) * area.h;

                elements.push(Element::Line {
                    x1: px0,
                    y1: py0,
                    x2: px1,
                    y2: py1,
                    stroke: Stroke::new(color, self.stroke_width),
                });
            }
        }

        elements
    }

    fn label(&self) -> Option<&str> {
        self.plot_label.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Rect;
    use crate::plot::PlotBuilder;
    use crate::scale::LinearScale;

    fn test_area() -> Rect {
        Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 200.0,
        }
    }

    #[test]
    fn contour_data_range() {
        let data = vec![
            vec![0.0, 1.0, 2.0],
            vec![3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0],
        ];
        let c = ContourPlot::new(data, 3);
        let (xr, yr) = c.data_range();
        assert_eq!(xr, Some((0.0, 2.0)));
        assert_eq!(yr, Some((0.0, 2.0)));
    }

    #[test]
    fn contour_produces_elements() {
        let data = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        let c = ContourPlot::new(data, 2);
        let xs = LinearScale::new(0.0, 2.0);
        let ys = LinearScale::new(0.0, 2.0);
        let elems = c.build_elements(&xs, &ys, test_area());
        // Should produce some line segments for the contours around the peak
        assert!(!elems.is_empty());
        assert!(elems.iter().all(|e| matches!(e, Element::Line { .. })));
    }

    #[test]
    fn contour_empty_grid() {
        let c = ContourPlot::new(vec![], 3);
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        assert!(c.build_elements(&xs, &ys, test_area()).is_empty());
    }

    #[test]
    fn contour_uniform_grid_no_lines() {
        // All same value — no contours should appear
        let data = vec![vec![5.0, 5.0], vec![5.0, 5.0]];
        let c = ContourPlot::new(data, 3);
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        let elems = c.build_elements(&xs, &ys, test_area());
        assert!(elems.is_empty());
    }

    #[test]
    fn contour_builder_methods() {
        let c = ContourPlot::new(vec![vec![0.0]], 1)
            .colormap(ColorMap::plasma())
            .stroke_width(2.5)
            .label("contour");
        assert!((c.stroke_width - 2.5).abs() < f64::EPSILON);
        assert_eq!(PlotBuilder::label(&c), Some("contour"));
    }

    #[test]
    fn marching_squares_simple() {
        // 2x2 grid with one corner high
        let data = vec![vec![0.0, 0.0], vec![0.0, 1.0]];
        let segs = marching_squares(&data, 0.5);
        assert!(!segs.is_empty());
    }
}
