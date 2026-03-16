use crate::color::Color;
use crate::element::Element;
use crate::layout::Rect;
use crate::plot::{AxisRange, PlotBuilder};
use crate::scale::Scale;
use crate::style::{Fill, Stroke};

/// Error bars rendered as vertical lines with caps at each data point.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ErrorBarPlot {
    x: Vec<f64>,
    y: Vec<f64>,
    y_err_low: Vec<f64>,
    y_err_high: Vec<f64>,
    stroke: Stroke,
    cap_width: f64,
    plot_label: Option<String>,
}

impl ErrorBarPlot {
    /// Create an error bar plot from x, y, lower errors, and upper errors.
    ///
    /// The error values are absolute offsets: the bar extends from
    /// `y - y_err_low` to `y + y_err_high`.
    #[must_use]
    pub fn new(x: Vec<f64>, y: Vec<f64>, y_err_low: Vec<f64>, y_err_high: Vec<f64>) -> Self {
        Self {
            x,
            y,
            y_err_low,
            y_err_high,
            stroke: Stroke::new(Color::BLACK, 1.0),
            cap_width: 6.0,
            plot_label: None,
        }
    }

    /// Set the stroke color.
    #[must_use]
    pub fn color(mut self, c: Color) -> Self {
        self.stroke = Stroke::new(c, self.stroke.width);
        self
    }

    /// Set the width of the error bar caps in pixels.
    #[must_use]
    pub fn cap_width(mut self, w: f64) -> Self {
        self.cap_width = w;
        self
    }

    /// Set the legend label.
    #[must_use]
    pub fn label(mut self, l: &str) -> Self {
        self.plot_label = Some(l.to_string());
        self
    }
}

/// Compute `(min, max)` of a slice, or `None` if empty.
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

impl PlotBuilder for ErrorBarPlot {
    fn data_range(&self) -> (AxisRange, AxisRange) {
        let x_range = min_max(&self.x);

        let n = self
            .x
            .len()
            .min(self.y.len())
            .min(self.y_err_low.len())
            .min(self.y_err_high.len());
        if n == 0 {
            return (x_range, None);
        }

        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..n {
            let low = self.y[i] - self.y_err_low[i];
            let high = self.y[i] + self.y_err_high[i];
            if low < lo {
                lo = low;
            }
            if high > hi {
                hi = high;
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
        let n = self
            .x
            .len()
            .min(self.y.len())
            .min(self.y_err_low.len())
            .min(self.y_err_high.len());
        if n == 0 {
            return vec![];
        }

        let half_cap = self.cap_width / 2.0;
        let mut elements = Vec::with_capacity(n * 3);

        for i in 0..n {
            let px = area.x + x_scale.transform(self.x[i]) * area.w;
            let py_lo = area.y + area.h - y_scale.transform(self.y[i] - self.y_err_low[i]) * area.h;
            let py_hi =
                area.y + area.h - y_scale.transform(self.y[i] + self.y_err_high[i]) * area.h;

            // Vertical bar
            elements.push(Element::Line {
                x1: px,
                y1: py_lo,
                x2: px,
                y2: py_hi,
                stroke: self.stroke.clone(),
            });

            // Lower cap
            elements.push(Element::Line {
                x1: px - half_cap,
                y1: py_lo,
                x2: px + half_cap,
                y2: py_lo,
                stroke: self.stroke.clone(),
            });

            // Upper cap
            elements.push(Element::Line {
                x1: px - half_cap,
                y1: py_hi,
                x2: px + half_cap,
                y2: py_hi,
                stroke: self.stroke.clone(),
            });
        }

        elements
    }

    fn label(&self) -> Option<&str> {
        self.plot_label.as_deref()
    }
}

// ---------------------------------------------------------------------------
// ConfidenceBand
// ---------------------------------------------------------------------------

/// A filled confidence band between lower and upper curves.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ConfidenceBand {
    x: Vec<f64>,
    y_lower: Vec<f64>,
    y_upper: Vec<f64>,
    fill: Fill,
    plot_label: Option<String>,
}

impl ConfidenceBand {
    /// Create a confidence band from x values and lower/upper y boundaries.
    #[must_use]
    pub fn new(x: Vec<f64>, y_lower: Vec<f64>, y_upper: Vec<f64>) -> Self {
        Self {
            x,
            y_lower,
            y_upper,
            fill: Fill::new(Color::rgba(31, 119, 180, 64)),
            plot_label: None,
        }
    }

    /// Set the fill color.
    #[must_use]
    pub fn color(mut self, c: Color) -> Self {
        self.fill = Fill::new(c);
        self
    }

    /// Set the legend label.
    #[must_use]
    pub fn label(mut self, l: &str) -> Self {
        self.plot_label = Some(l.to_string());
        self
    }
}

impl PlotBuilder for ConfidenceBand {
    fn data_range(&self) -> (AxisRange, AxisRange) {
        let x_range = min_max(&self.x);

        let n = self.x.len().min(self.y_lower.len()).min(self.y_upper.len());
        if n == 0 {
            return (x_range, None);
        }

        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..n {
            if self.y_lower[i] < lo {
                lo = self.y_lower[i];
            }
            if self.y_upper[i] > hi {
                hi = self.y_upper[i];
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
        let n = self.x.len().min(self.y_lower.len()).min(self.y_upper.len());
        if n == 0 {
            return vec![];
        }

        // Build polygon: upper curve left-to-right, then lower curve right-to-left
        let mut points = Vec::with_capacity(n * 2 + 1);

        for i in 0..n {
            let px = area.x + x_scale.transform(self.x[i]) * area.w;
            let py = area.y + area.h - y_scale.transform(self.y_upper[i]) * area.h;
            points.push((px, py));
        }
        for i in (0..n).rev() {
            let px = area.x + x_scale.transform(self.x[i]) * area.w;
            let py = area.y + area.h - y_scale.transform(self.y_lower[i]) * area.h;
            points.push((px, py));
        }
        // Close
        if let Some(&first) = points.first() {
            points.push(first);
        }

        vec![Element::Polyline {
            points,
            stroke: Stroke::new(Color::TRANSPARENT, 0.0),
            fill: Some(self.fill),
        }]
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
    fn error_bar_data_range() {
        let e = ErrorBarPlot::new(
            vec![1.0, 2.0, 3.0],
            vec![5.0, 6.0, 7.0],
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
        );
        let (xr, yr) = e.data_range();
        assert_eq!(xr, Some((1.0, 3.0)));
        // y range: min(5-1, 6-1, 7-1)=4, max(5+2, 6+2, 7+2)=9
        assert_eq!(yr, Some((4.0, 9.0)));
    }

    #[test]
    fn error_bar_produces_elements() {
        let e = ErrorBarPlot::new(
            vec![1.0, 2.0],
            vec![5.0, 6.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        );
        let xs = LinearScale::new(1.0, 2.0);
        let ys = LinearScale::new(4.0, 7.0);
        let elems = e.build_elements(&xs, &ys, test_area());
        // 2 points × 3 elements each (bar + lower cap + upper cap) = 6
        assert_eq!(elems.len(), 6);
        assert!(elems.iter().all(|e| matches!(e, Element::Line { .. })));
    }

    #[test]
    fn error_bar_empty() {
        let e = ErrorBarPlot::new(vec![], vec![], vec![], vec![]);
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        assert!(e.build_elements(&xs, &ys, test_area()).is_empty());
    }

    #[test]
    fn error_bar_builder() {
        let e = ErrorBarPlot::new(vec![1.0], vec![2.0], vec![0.5], vec![0.5])
            .color(Color::RED)
            .cap_width(10.0)
            .label("err");
        assert_eq!(e.stroke.color, Color::RED);
        assert!((e.cap_width - 10.0).abs() < f64::EPSILON);
        assert_eq!(PlotBuilder::label(&e), Some("err"));
    }

    #[test]
    fn confidence_band_data_range() {
        let cb = ConfidenceBand::new(
            vec![0.0, 1.0, 2.0],
            vec![1.0, 2.0, 1.5],
            vec![3.0, 4.0, 3.5],
        );
        let (xr, yr) = cb.data_range();
        assert_eq!(xr, Some((0.0, 2.0)));
        assert_eq!(yr, Some((1.0, 4.0)));
    }

    #[test]
    fn confidence_band_produces_polyline() {
        let cb = ConfidenceBand::new(vec![0.0, 1.0], vec![1.0, 2.0], vec![3.0, 4.0]);
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(1.0, 4.0);
        let elems = cb.build_elements(&xs, &ys, test_area());
        assert_eq!(elems.len(), 1);
        assert!(matches!(elems[0], Element::Polyline { .. }));
    }

    #[test]
    fn confidence_band_empty() {
        let cb = ConfidenceBand::new(vec![], vec![], vec![]);
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        assert!(cb.build_elements(&xs, &ys, test_area()).is_empty());
    }

    #[test]
    fn confidence_band_builder() {
        let cb = ConfidenceBand::new(vec![0.0], vec![1.0], vec![2.0])
            .color(Color::BLUE)
            .label("band");
        assert_eq!(cb.fill.color, Color::BLUE);
        assert_eq!(PlotBuilder::label(&cb), Some("band"));
    }
}
