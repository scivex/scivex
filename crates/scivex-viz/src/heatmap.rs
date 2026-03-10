use crate::color::ColorMap;
use crate::element::{Element, TextAnchor};
use crate::layout::Rect;
use crate::plot::{AxisRange, PlotBuilder};
use crate::scale::Scale;
use crate::style::{Fill, Font};

/// A heatmap rendered as a grid of colored rectangles.
pub struct HeatmapBuilder {
    data: Vec<Vec<f64>>,
    colormap: ColorMap,
    show_values: bool,
    x_labels: Option<Vec<String>>,
    y_labels: Option<Vec<String>>,
}

impl HeatmapBuilder {
    /// Create a heatmap from a 2-D grid of values.
    #[must_use]
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        Self {
            data,
            colormap: ColorMap::viridis(),
            show_values: false,
            x_labels: None,
            y_labels: None,
        }
    }

    /// Set the colormap used to map values to colors.
    #[must_use]
    pub fn colormap(mut self, cm: ColorMap) -> Self {
        self.colormap = cm;
        self
    }

    /// Show numeric values inside each cell.
    #[must_use]
    pub fn show_values(mut self, show: bool) -> Self {
        self.show_values = show;
        self
    }

    /// Set column labels.
    #[must_use]
    pub fn x_labels(mut self, labels: Vec<String>) -> Self {
        self.x_labels = Some(labels);
        self
    }

    /// Set row labels.
    #[must_use]
    pub fn y_labels(mut self, labels: Vec<String>) -> Self {
        self.y_labels = Some(labels);
        self
    }

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

impl PlotBuilder for HeatmapBuilder {
    fn data_range(&self) -> (AxisRange, AxisRange) {
        let n_rows = self.data.len();
        let n_cols = self.data.first().map_or(0, Vec::len);
        let x_range = if n_cols > 0 {
            Some((-0.5, n_cols as f64 - 0.5))
        } else {
            None
        };
        let y_range = if n_rows > 0 {
            Some((-0.5, n_rows as f64 - 0.5))
        } else {
            None
        };
        (x_range, y_range)
    }

    fn build_elements(
        &self,
        _x_scale: &dyn Scale,
        _y_scale: &dyn Scale,
        area: Rect,
    ) -> Vec<Element> {
        let n_rows = self.data.len();
        if n_rows == 0 {
            return vec![];
        }
        let n_cols = self.data.first().map_or(0, Vec::len);
        if n_cols == 0 {
            return vec![];
        }

        let (lo, hi) = self.data_bounds();
        let cell_w = area.w / n_cols as f64;
        let cell_h = area.h / n_rows as f64;

        let mut elements = Vec::with_capacity(n_rows * n_cols * 2);
        for (r, row) in self.data.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                let t = (val - lo) / (hi - lo);
                let color = self.colormap.sample(t);
                let x = area.x + c as f64 * cell_w;
                let y = area.y + r as f64 * cell_h;
                elements.push(Element::Rect {
                    x,
                    y,
                    w: cell_w,
                    h: cell_h,
                    fill: Some(Fill::new(color)),
                    stroke: None,
                });
                if self.show_values {
                    elements.push(Element::Text {
                        x: x + cell_w / 2.0,
                        y: y + cell_h / 2.0 + 4.0,
                        text: format!("{val:.1}"),
                        font: Font {
                            size: 10.0,
                            ..Font::default()
                        },
                        anchor: TextAnchor::Middle,
                    });
                }
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
    use crate::color::ColorMap;
    use crate::layout::Rect;
    use crate::plot::PlotBuilder;
    use crate::scale::LinearScale;

    #[test]
    fn heatmap_colormap() {
        let h = HeatmapBuilder::new(vec![vec![1.0]]).colormap(ColorMap::plasma());
        // Just verify it doesn't panic and is set
        let xs = LinearScale::new(-0.5, 0.5);
        let ys = LinearScale::new(-0.5, 0.5);
        let area = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 100.0,
        };
        let elems = h.build_elements(&xs, &ys, area);
        assert_eq!(elems.len(), 1);
    }

    #[test]
    fn heatmap_labels() {
        let h = HeatmapBuilder::new(vec![vec![1.0, 2.0]])
            .x_labels(vec!["A".into(), "B".into()])
            .y_labels(vec!["Row1".into()]);
        assert_eq!(h.x_labels, Some(vec!["A".into(), "B".into()]));
        assert_eq!(h.y_labels, Some(vec!["Row1".into()]));
    }

    #[test]
    fn heatmap_empty_data() {
        let h = HeatmapBuilder::new(vec![]);
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        let area = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 100.0,
        };
        assert!(h.build_elements(&xs, &ys, area).is_empty());
    }

    #[test]
    fn heatmap_data_range() {
        let h = HeatmapBuilder::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let (xr, yr) = PlotBuilder::data_range(&h);
        assert_eq!(xr, Some((-0.5, 1.5)));
        assert_eq!(yr, Some((-0.5, 1.5)));
    }

    #[test]
    fn heatmap_label_is_none() {
        let h = HeatmapBuilder::new(vec![vec![1.0]]);
        assert_eq!(PlotBuilder::label(&h), None);
    }

    #[test]
    fn heatmap_element_count() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let h = HeatmapBuilder::new(data);
        let xs = LinearScale::new(-0.5, 2.5);
        let ys = LinearScale::new(-0.5, 1.5);
        let area = Rect {
            x: 0.0,
            y: 0.0,
            w: 300.0,
            h: 200.0,
        };
        let elems = h.build_elements(&xs, &ys, area);
        // 2 rows × 3 cols = 6 rectangles (no text overlay by default)
        assert_eq!(elems.len(), 6);
    }

    #[test]
    fn heatmap_with_values() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let h = HeatmapBuilder::new(data).show_values(true);
        let xs = LinearScale::new(-0.5, 1.5);
        let ys = LinearScale::new(-0.5, 1.5);
        let area = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 200.0,
        };
        let elems = h.build_elements(&xs, &ys, area);
        // 4 rects + 4 text labels = 8
        assert_eq!(elems.len(), 8);
    }
}
