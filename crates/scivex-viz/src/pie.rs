use crate::color::{self, Color};
use crate::element::Element;
use crate::layout::Rect;
use crate::plot::{AxisRange, PlotBuilder};
use crate::scale::Scale;
use crate::style::{Fill, Stroke};

/// A pie or donut chart.
///
/// Set `inner_radius` > 0 for a donut chart appearance.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct PieChart {
    values: Vec<f64>,
    labels: Option<Vec<String>>,
    colors: Option<Vec<Color>>,
    inner_radius: f64,
    plot_label: Option<String>,
}

/// Number of line segments used to approximate each arc wedge.
const ARC_SEGMENTS: usize = 20;

impl PieChart {
    /// Create a pie chart from a vector of values.
    #[must_use]
    pub fn new(values: Vec<f64>) -> Self {
        Self {
            values,
            labels: None,
            colors: None,
            inner_radius: 0.0,
            plot_label: None,
        }
    }

    /// Set category labels for each wedge.
    #[must_use]
    pub fn labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Set custom colors for the wedges.
    #[must_use]
    pub fn colors(mut self, colors: Vec<Color>) -> Self {
        self.colors = Some(colors);
        self
    }

    /// Set the inner radius fraction (0.0 = full pie, 0.5 = donut with half-radius hole).
    ///
    /// The value is clamped to `[0.0, 0.99]`.
    #[must_use]
    pub fn inner_radius(mut self, r: f64) -> Self {
        self.inner_radius = r.clamp(0.0, 0.99);
        self
    }

    /// Set the legend label for this chart.
    #[must_use]
    pub fn label(mut self, l: &str) -> Self {
        self.plot_label = Some(l.to_string());
        self
    }
}

impl PlotBuilder for PieChart {
    fn data_range(&self) -> (AxisRange, AxisRange) {
        // Pie charts use normalized space; the plot area is the circle.
        (Some((0.0, 1.0)), Some((0.0, 1.0)))
    }

    fn build_elements(
        &self,
        _x_scale: &dyn Scale,
        _y_scale: &dyn Scale,
        area: Rect,
    ) -> Vec<Element> {
        if self.values.is_empty() {
            return vec![];
        }

        let total: f64 = self.values.iter().filter(|v| **v > 0.0).sum();
        if total <= 0.0 {
            return vec![];
        }

        let palette = self.colors.clone().unwrap_or_else(color::default_palette);

        let cx = area.x + area.w / 2.0;
        let cy = area.y + area.h / 2.0;
        let outer_r = area.w.min(area.h) / 2.0;
        let inner_r = outer_r * self.inner_radius;

        let mut elements = Vec::new();
        let mut start_angle: f64 = -std::f64::consts::FRAC_PI_2; // start at top

        for (i, &val) in self.values.iter().enumerate() {
            if val <= 0.0 {
                continue;
            }
            let sweep = (val / total) * 2.0 * std::f64::consts::PI;
            let end_angle = start_angle + sweep;

            let color = palette[i % palette.len()];

            // Build wedge as a polyline approximation
            let mut points = Vec::with_capacity(ARC_SEGMENTS * 2 + 3);

            if inner_r > 0.0 {
                // Donut: outer arc forward, then inner arc backward
                for j in 0..=ARC_SEGMENTS {
                    let t = j as f64 / ARC_SEGMENTS as f64;
                    let angle = start_angle + t * sweep;
                    points.push((cx + outer_r * angle.cos(), cy + outer_r * angle.sin()));
                }
                for j in (0..=ARC_SEGMENTS).rev() {
                    let t = j as f64 / ARC_SEGMENTS as f64;
                    let angle = start_angle + t * sweep;
                    points.push((cx + inner_r * angle.cos(), cy + inner_r * angle.sin()));
                }
            } else {
                // Full pie: center, outer arc, back to center
                points.push((cx, cy));
                for j in 0..=ARC_SEGMENTS {
                    let t = j as f64 / ARC_SEGMENTS as f64;
                    let angle = start_angle + t * sweep;
                    points.push((cx + outer_r * angle.cos(), cy + outer_r * angle.sin()));
                }
                points.push((cx, cy));
            }

            elements.push(Element::Polyline {
                points,
                stroke: Stroke::new(Color::WHITE, 1.0),
                fill: Some(Fill::new(color)),
            });

            start_angle = end_angle;
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
    fn pie_data_range_is_normalized() {
        let p = PieChart::new(vec![1.0, 2.0, 3.0]);
        let (xr, yr) = p.data_range();
        assert_eq!(xr, Some((0.0, 1.0)));
        assert_eq!(yr, Some((0.0, 1.0)));
    }

    #[test]
    fn pie_produces_wedges() {
        let p = PieChart::new(vec![1.0, 2.0, 3.0]);
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        let elems = p.build_elements(&xs, &ys, test_area());
        assert_eq!(elems.len(), 3);
        assert!(elems.iter().all(|e| matches!(e, Element::Polyline { .. })));
    }

    #[test]
    fn pie_empty_values() {
        let p = PieChart::new(vec![]);
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        assert!(p.build_elements(&xs, &ys, test_area()).is_empty());
    }

    #[test]
    fn donut_chart() {
        let p = PieChart::new(vec![1.0, 1.0]).inner_radius(0.5);
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        let elems = p.build_elements(&xs, &ys, test_area());
        assert_eq!(elems.len(), 2);
        // Donut wedges should not go through center
        if let Element::Polyline { points, .. } = &elems[0] {
            // Should have outer arc + inner arc points
            assert!(points.len() > ARC_SEGMENTS);
        }
    }

    #[test]
    fn pie_label() {
        let p = PieChart::new(vec![1.0]).label("pie");
        assert_eq!(PlotBuilder::label(&p), Some("pie"));
    }

    #[test]
    fn pie_custom_colors() {
        let p = PieChart::new(vec![1.0, 2.0]).colors(vec![Color::RED, Color::BLUE]);
        assert_eq!(p.colors.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn pie_labels_set() {
        let p = PieChart::new(vec![1.0]).labels(vec!["Slice A".into()]);
        assert_eq!(p.labels, Some(vec!["Slice A".to_string()]));
    }
}
