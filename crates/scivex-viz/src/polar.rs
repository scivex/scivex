use crate::color::Color;
use crate::element::Element;
use crate::layout::Rect;
use crate::plot::{AxisRange, PlotBuilder};
use crate::scale::Scale;
use crate::style::{Fill, Stroke};

/// A polar (radar / spider) chart plotting values on radial axes.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct PolarPlot {
    categories: Vec<String>,
    values: Vec<f64>,
    fill: Option<Fill>,
    stroke: Stroke,
    plot_label: Option<String>,
}

impl PolarPlot {
    /// Create a polar plot from category names and values.
    #[must_use]
    pub fn new(categories: Vec<String>, values: Vec<f64>) -> Self {
        Self {
            categories,
            values,
            fill: Some(Fill::new(Color::rgba(31, 119, 180, 80))),
            stroke: Stroke::new(Color::rgb(31, 119, 180), 1.5),
            plot_label: None,
        }
    }

    /// Set the outline color.
    #[must_use]
    pub fn color(mut self, c: Color) -> Self {
        self.stroke = Stroke::new(c, self.stroke.width);
        self
    }

    /// Set the fill color for the polygon area.
    #[must_use]
    pub fn fill_color(mut self, c: Color) -> Self {
        self.fill = Some(Fill::new(c));
        self
    }

    /// Set the legend label for this polar plot.
    #[must_use]
    pub fn label(mut self, l: &str) -> Self {
        self.plot_label = Some(l.to_string());
        self
    }
}

impl PlotBuilder for PolarPlot {
    fn data_range(&self) -> (AxisRange, AxisRange) {
        // Polar charts use their own coordinate system within the plot area.
        (Some((0.0, 1.0)), Some((0.0, 1.0)))
    }

    fn build_elements(
        &self,
        _x_scale: &dyn Scale,
        _y_scale: &dyn Scale,
        area: Rect,
    ) -> Vec<Element> {
        let n = self.categories.len().min(self.values.len());
        if n == 0 {
            return vec![];
        }

        let cx = area.x + area.w / 2.0;
        let cy = area.y + area.h / 2.0;
        let radius = area.w.min(area.h) / 2.0 * 0.85; // slight padding

        // Find max value for normalization
        let max_val = self
            .values
            .iter()
            .take(n)
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if !max_val.is_finite() || max_val <= 0.0 {
            return vec![];
        }

        let mut elements = Vec::new();
        let angle_step = 2.0 * std::f64::consts::PI / n as f64;
        let start_angle = -std::f64::consts::FRAC_PI_2; // start at top

        // Draw axis lines from center to each vertex
        for j in 0..n {
            let angle = start_angle + j as f64 * angle_step;
            let x_end = cx + radius * angle.cos();
            let y_end = cy + radius * angle.sin();
            elements.push(Element::Line {
                x1: cx,
                y1: cy,
                x2: x_end,
                y2: y_end,
                stroke: Stroke::new(Color::LIGHT_GRAY, 0.5),
            });
        }

        // Build the data polygon
        let mut points = Vec::with_capacity(n + 1);
        for j in 0..n {
            let angle = start_angle + j as f64 * angle_step;
            let r = (self.values[j] / max_val).clamp(0.0, 1.0) * radius;
            points.push((cx + r * angle.cos(), cy + r * angle.sin()));
        }
        // Close the polygon
        if let Some(&first) = points.first() {
            points.push(first);
        }

        elements.push(Element::Polyline {
            points,
            stroke: self.stroke.clone(),
            fill: self.fill,
        });

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
            w: 300.0,
            h: 300.0,
        }
    }

    #[test]
    fn polar_data_range_is_normalized() {
        let p = PolarPlot::new(vec!["A".into(), "B".into()], vec![1.0, 2.0]);
        let (xr, yr) = p.data_range();
        assert_eq!(xr, Some((0.0, 1.0)));
        assert_eq!(yr, Some((0.0, 1.0)));
    }

    #[test]
    fn polar_produces_elements() {
        let p = PolarPlot::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![3.0, 5.0, 4.0],
        );
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        let elems = p.build_elements(&xs, &ys, test_area());
        // 3 axis lines + 1 data polygon
        assert_eq!(elems.len(), 4);
        assert!(matches!(elems[3], Element::Polyline { .. }));
    }

    #[test]
    fn polar_empty() {
        let p = PolarPlot::new(vec![], vec![]);
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        assert!(p.build_elements(&xs, &ys, test_area()).is_empty());
    }

    #[test]
    fn polar_builder_methods() {
        let p = PolarPlot::new(vec!["A".into()], vec![1.0])
            .color(Color::RED)
            .fill_color(Color::BLUE)
            .label("radar");
        assert_eq!(p.stroke.color, Color::RED);
        assert_eq!(p.fill.unwrap().color, Color::BLUE);
        assert_eq!(PlotBuilder::label(&p), Some("radar"));
    }
}
