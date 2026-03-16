use crate::color::Color;
use crate::element::Element;
use crate::layout::Rect;
use crate::plot::{AxisRange, PlotBuilder};
use crate::scale::Scale;
use crate::style::{Fill, Stroke};

/// A violin plot showing the distribution shape via mirrored KDE curves,
/// with an optional inner box plot overlay.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ViolinPlot {
    datasets: Vec<Vec<f64>>,
    show_box: bool,
    bandwidth: f64,
    n_points: usize,
    fill: Option<Fill>,
    stroke: Stroke,
}

impl ViolinPlot {
    /// Create a violin plot from one or more datasets.
    #[must_use]
    pub fn new(datasets: Vec<Vec<f64>>) -> Self {
        Self {
            datasets,
            show_box: true,
            bandwidth: 0.5,
            n_points: 50,
            fill: Some(Fill::new(Color::rgb(31, 119, 180))),
            stroke: Stroke::new(Color::BLACK, 1.0),
        }
    }

    /// Whether to show an inner box plot (median line + IQR box).
    #[must_use]
    pub fn show_box(mut self, show: bool) -> Self {
        self.show_box = show;
        self
    }

    /// Set the KDE bandwidth (Gaussian kernel standard deviation).
    #[must_use]
    pub fn bandwidth(mut self, bw: f64) -> Self {
        self.bandwidth = bw;
        self
    }

    /// Set the fill color for the violin shapes.
    #[must_use]
    pub fn fill_color(mut self, c: Color) -> Self {
        self.fill = Some(Fill::new(c));
        self
    }

    /// Remove the fill (outline only).
    #[must_use]
    pub fn no_fill(mut self) -> Self {
        self.fill = None;
        self
    }
}

/// Compute Gaussian KDE at the given evaluation points.
fn gaussian_kde(data: &[f64], eval_points: &[f64], bandwidth: f64) -> Vec<f64> {
    let n = data.len() as f64;
    let inv_bw = 1.0 / bandwidth;
    let norm = 1.0 / (n * bandwidth * (2.0 * std::f64::consts::PI).sqrt());

    eval_points
        .iter()
        .map(|&x| {
            let sum: f64 = data
                .iter()
                .map(|&d| {
                    let z = (x - d) * inv_bw;
                    (-0.5 * z * z).exp()
                })
                .sum();
            sum * norm
        })
        .collect()
}

/// Compute the percentile of a sorted slice using linear interpolation.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    let frac = idx - lo as f64;
    if hi >= n {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

impl PlotBuilder for ViolinPlot {
    fn data_range(&self) -> (AxisRange, AxisRange) {
        let n = self.datasets.len();
        let x_range = if n > 0 {
            Some((-0.5, n as f64 - 0.5))
        } else {
            None
        };

        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for ds in &self.datasets {
            for &v in ds {
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
        }
        let y_range = if lo.is_finite() && hi.is_finite() {
            Some((lo, hi))
        } else {
            None
        };
        (x_range, y_range)
    }

    fn build_elements(&self, x_scale: &dyn Scale, y_scale: &dyn Scale, area: Rect) -> Vec<Element> {
        let mut elements = Vec::new();
        let n = self.datasets.len();
        if n == 0 {
            return elements;
        }

        let half_width_data = 0.4;

        for (i, ds) in self.datasets.iter().enumerate() {
            if ds.is_empty() {
                continue;
            }

            let mut sorted = ds.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let data_min = sorted[0];
            let data_max = sorted[sorted.len() - 1];

            if (data_max - data_min).abs() < f64::EPSILON {
                // All values are the same; draw a simple line
                let cx = area.x + x_scale.transform(i as f64) * area.w;
                let cy = area.y + area.h - y_scale.transform(data_min) * area.h;
                elements.push(Element::Line {
                    x1: cx - 5.0,
                    y1: cy,
                    x2: cx + 5.0,
                    y2: cy,
                    stroke: self.stroke.clone(),
                });
                continue;
            }

            // Evaluate KDE at n_points evenly spaced along the data range
            let step = (data_max - data_min) / (self.n_points - 1) as f64;
            let eval_points: Vec<f64> = (0..self.n_points)
                .map(|j| data_min + j as f64 * step)
                .collect();
            let densities = gaussian_kde(&sorted, &eval_points, self.bandwidth);

            // Find max density to normalize width
            let max_density = densities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if max_density <= 0.0 {
                continue;
            }

            let cx = area.x + x_scale.transform(i as f64) * area.w;

            // Compute the pixel half-width at full density
            let pixel_half_width =
                (x_scale.transform(half_width_data) - x_scale.transform(-half_width_data)) * area.w
                    / 2.0;

            // Build right-side points (top to bottom in data, but y is inverted)
            let mut right_points: Vec<(f64, f64)> = Vec::with_capacity(self.n_points);
            let mut left_points: Vec<(f64, f64)> = Vec::with_capacity(self.n_points);

            for (j, &density) in densities.iter().enumerate() {
                let y_val = eval_points[j];
                let py = area.y + area.h - y_scale.transform(y_val) * area.h;
                let width = (density / max_density) * pixel_half_width;
                right_points.push((cx + width, py));
                left_points.push((cx - width, py));
            }

            // Build the violin outline: right side top-to-bottom, then left side bottom-to-top
            let mut outline = Vec::with_capacity(self.n_points * 2 + 1);
            outline.extend_from_slice(&right_points);
            left_points.reverse();
            outline.extend_from_slice(&left_points);
            // Close the shape
            if let Some(&first) = outline.first() {
                outline.push(first);
            }

            elements.push(Element::Polyline {
                points: outline,
                stroke: self.stroke.clone(),
                fill: self.fill,
            });

            // Optional inner box plot
            if self.show_box {
                let median = percentile(&sorted, 0.5);
                let q1 = percentile(&sorted, 0.25);
                let q3 = percentile(&sorted, 0.75);

                let y_med = area.y + area.h - y_scale.transform(median) * area.h;
                let y_q1 = area.y + area.h - y_scale.transform(q1) * area.h;
                let y_q3 = area.y + area.h - y_scale.transform(q3) * area.h;

                let box_half = pixel_half_width * 0.15;

                // IQR box
                elements.push(Element::Rect {
                    x: cx - box_half,
                    y: y_q3,
                    w: box_half * 2.0,
                    h: y_q1 - y_q3,
                    fill: Some(Fill::new(Color::WHITE)),
                    stroke: Some(Stroke::new(Color::BLACK, 1.0)),
                });

                // Median line
                elements.push(Element::Line {
                    x1: cx - box_half,
                    y1: y_med,
                    x2: cx + box_half,
                    y2: y_med,
                    stroke: Stroke::new(Color::BLACK, 2.0),
                });
            }
        }

        elements
    }

    fn label(&self) -> Option<&str> {
        None
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
            w: 400.0,
            h: 300.0,
        }
    }

    #[test]
    fn violin_data_range() {
        let v = ViolinPlot::new(vec![vec![1.0, 5.0], vec![2.0, 8.0]]);
        let (xr, yr) = v.data_range();
        assert_eq!(xr, Some((-0.5, 1.5)));
        assert_eq!(yr, Some((1.0, 8.0)));
    }

    #[test]
    fn violin_empty_datasets() {
        let v = ViolinPlot::new(vec![]);
        let (xr, yr) = v.data_range();
        assert!(xr.is_none());
        assert!(yr.is_none());
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        assert!(v.build_elements(&xs, &ys, test_area()).is_empty());
    }

    #[test]
    fn violin_produces_elements() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let v = ViolinPlot::new(vec![data]);
        let xs = LinearScale::new(-0.5, 0.5);
        let ys = LinearScale::new(1.0, 10.0);
        let elems = v.build_elements(&xs, &ys, test_area());
        // Should have at least a polyline for the violin shape + box elements
        assert!(!elems.is_empty());
        // First element should be the polyline shape
        assert!(matches!(elems[0], Element::Polyline { .. }));
    }

    #[test]
    fn violin_no_box() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = ViolinPlot::new(vec![data]).show_box(false);
        let xs = LinearScale::new(-0.5, 0.5);
        let ys = LinearScale::new(1.0, 5.0);
        let elems = v.build_elements(&xs, &ys, test_area());
        // Without box, should have just the polyline
        assert_eq!(elems.len(), 1);
        assert!(matches!(elems[0], Element::Polyline { .. }));
    }

    #[test]
    fn violin_builder_methods() {
        let v = ViolinPlot::new(vec![vec![1.0]])
            .bandwidth(1.0)
            .fill_color(Color::RED)
            .show_box(false);
        assert!(!v.show_box);
        assert!((v.bandwidth - 1.0).abs() < f64::EPSILON);
        assert_eq!(v.fill.unwrap().color, Color::RED);
    }

    #[test]
    fn violin_no_fill() {
        let v = ViolinPlot::new(vec![vec![1.0]]).no_fill();
        assert!(v.fill.is_none());
    }

    #[test]
    fn violin_label_is_none() {
        let v = ViolinPlot::new(vec![vec![1.0]]);
        assert_eq!(PlotBuilder::label(&v), None);
    }
}
