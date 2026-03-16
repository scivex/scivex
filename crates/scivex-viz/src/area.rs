use crate::color::Color;
use crate::element::Element;
use crate::layout::Rect;
use crate::plot::{AxisRange, PlotBuilder};
use crate::scale::Scale;
use crate::style::{Fill, Stroke};

/// An area chart that fills the region between a baseline and the data curve.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct AreaPlot {
    x: Vec<f64>,
    y: Vec<f64>,
    fill: Fill,
    stroke: Option<Stroke>,
    baseline: f64,
    plot_label: Option<String>,
}

impl AreaPlot {
    /// Create an area plot from x and y data vectors.
    #[must_use]
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        Self {
            x,
            y,
            fill: Fill::new(Color::rgba(31, 119, 180, 128)),
            stroke: Some(Stroke::new(Color::rgb(31, 119, 180), 1.5)),
            baseline: 0.0,
            plot_label: None,
        }
    }

    /// Set the fill and stroke color.
    #[must_use]
    pub fn color(mut self, c: Color) -> Self {
        self.fill = Fill::new(Color::rgba(c.r, c.g, c.b, 128));
        self.stroke = Some(Stroke::new(c, 1.5));
        self
    }

    /// Set the stroke color (the line along the top of the area).
    #[must_use]
    pub fn stroke_color(mut self, c: Color) -> Self {
        self.stroke = Some(Stroke::new(c, 1.5));
        self
    }

    /// Set the baseline y-value (default is 0.0).
    #[must_use]
    pub fn baseline(mut self, b: f64) -> Self {
        self.baseline = b;
        self
    }

    /// Set the legend label for this area plot.
    #[must_use]
    pub fn label(mut self, l: &str) -> Self {
        self.plot_label = Some(l.to_string());
        self
    }
}

/// Compute the `(min, max)` of a slice, or `None` if empty.
fn min_max(values: &[f64]) -> Option<(f64, f64)> {
    if values.is_empty() {
        return None;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in values {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    if lo.is_finite() && hi.is_finite() {
        Some((lo, hi))
    } else {
        None
    }
}

impl PlotBuilder for AreaPlot {
    fn data_range(&self) -> (AxisRange, AxisRange) {
        let x_range = min_max(&self.x);
        let y_range = match min_max(&self.y) {
            Some((lo, hi)) => {
                let lo = lo.min(self.baseline);
                let hi = hi.max(self.baseline);
                Some((lo, hi))
            }
            None => None,
        };
        (x_range, y_range)
    }

    fn build_elements(&self, x_scale: &dyn Scale, y_scale: &dyn Scale, area: Rect) -> Vec<Element> {
        let n = self.x.len().min(self.y.len());
        if n == 0 {
            return vec![];
        }

        let baseline_py = area.y + area.h - y_scale.transform(self.baseline) * area.h;

        // Build the filled area: data points left-to-right, then baseline right-to-left
        let mut points = Vec::with_capacity(n * 2 + 1);

        // Forward along data curve
        for i in 0..n {
            let px = area.x + x_scale.transform(self.x[i]) * area.w;
            let py = area.y + area.h - y_scale.transform(self.y[i]) * area.h;
            points.push((px, py));
        }

        // Return along baseline (right to left)
        let x_last = area.x + x_scale.transform(self.x[n - 1]) * area.w;
        let x_first = area.x + x_scale.transform(self.x[0]) * area.w;
        points.push((x_last, baseline_py));
        points.push((x_first, baseline_py));

        // Close shape
        points.push(points[0]);

        let mut elements = vec![Element::Polyline {
            points: points.clone(),
            stroke: self
                .stroke
                .clone()
                .unwrap_or_else(|| Stroke::new(Color::TRANSPARENT, 0.0)),
            fill: Some(self.fill),
        }];

        // Draw the top line separately if a stroke is set, for crispness
        if let Some(ref stroke) = self.stroke {
            let top_points: Vec<(f64, f64)> = (0..n)
                .map(|i| {
                    let px = area.x + x_scale.transform(self.x[i]) * area.w;
                    let py = area.y + area.h - y_scale.transform(self.y[i]) * area.h;
                    (px, py)
                })
                .collect();
            elements.push(Element::Polyline {
                points: top_points,
                stroke: stroke.clone(),
                fill: None,
            });
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
            h: 150.0,
        }
    }

    #[test]
    fn area_data_range_includes_baseline() {
        let a = AreaPlot::new(vec![0.0, 1.0, 2.0], vec![3.0, 5.0, 4.0]);
        let (xr, yr) = a.data_range();
        assert_eq!(xr, Some((0.0, 2.0)));
        // baseline is 0.0, so y range should be (0.0, 5.0)
        assert_eq!(yr, Some((0.0, 5.0)));
    }

    #[test]
    fn area_produces_elements() {
        let a = AreaPlot::new(vec![0.0, 1.0, 2.0], vec![0.0, 2.0, 1.0]);
        let xs = LinearScale::new(0.0, 2.0);
        let ys = LinearScale::new(0.0, 2.0);
        let elems = a.build_elements(&xs, &ys, test_area());
        // Should have filled polyline + top stroke polyline
        assert_eq!(elems.len(), 2);
        assert!(matches!(elems[0], Element::Polyline { .. }));
    }

    #[test]
    fn area_empty_data() {
        let a = AreaPlot::new(vec![], vec![]);
        let (xr, yr) = a.data_range();
        assert!(xr.is_none());
        assert!(yr.is_none());
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        assert!(a.build_elements(&xs, &ys, test_area()).is_empty());
    }

    #[test]
    fn area_builder_methods() {
        let a = AreaPlot::new(vec![0.0], vec![1.0])
            .color(Color::RED)
            .baseline(1.0)
            .label("area");
        assert!((a.baseline - 1.0).abs() < f64::EPSILON);
        assert_eq!(PlotBuilder::label(&a), Some("area"));
    }

    #[test]
    fn area_stroke_color() {
        let a = AreaPlot::new(vec![0.0], vec![1.0]).stroke_color(Color::GREEN);
        assert_eq!(a.stroke.unwrap().color, Color::GREEN);
    }
}
