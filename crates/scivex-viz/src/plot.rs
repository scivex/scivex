use crate::color::Color;
use crate::element::Element;
use crate::layout::Rect;
use crate::scale::Scale;
use crate::style::{Fill, Marker, MarkerShape, Stroke};

/// A range for a single axis: `(min, max)`.
pub type AxisRange = Option<(f64, f64)>;

/// Trait for plot builders that can produce drawing elements.
pub trait PlotBuilder {
    /// Return the data range `(x_range, y_range)`, or `None` per axis if unknown.
    fn data_range(&self) -> (AxisRange, AxisRange);

    /// Produce drawing elements in pixel space using the given scales and area.
    fn build_elements(&self, x_scale: &dyn Scale, y_scale: &dyn Scale, area: Rect) -> Vec<Element>;

    /// Optional label for the legend.
    fn label(&self) -> Option<&str>;
}

// ---------------------------------------------------------------------------
// LinePlot
// ---------------------------------------------------------------------------

/// A line plot connecting `(x, y)` data points.
pub struct LinePlot {
    x: Vec<f64>,
    y: Vec<f64>,
    stroke: Stroke,
    plot_label: Option<String>,
}

impl LinePlot {
    /// Create a line plot from x and y data vectors.
    #[must_use]
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        Self {
            x,
            y,
            stroke: Stroke::new(Color::rgb(31, 119, 180), 1.5),
            plot_label: None,
        }
    }

    /// Set the line color.
    #[must_use]
    pub fn color(mut self, c: Color) -> Self {
        self.stroke.color = c;
        self
    }

    /// Set the line width in pixels.
    #[must_use]
    pub fn width(mut self, w: f64) -> Self {
        self.stroke.width = w;
        self
    }

    /// Set a dash pattern for the line.
    #[must_use]
    pub fn dash(mut self, d: Vec<f64>) -> Self {
        self.stroke.dash = Some(d);
        self
    }

    /// Set the legend label for this line plot.
    #[must_use]
    pub fn label(mut self, l: &str) -> Self {
        self.plot_label = Some(l.to_string());
        self
    }
}

impl PlotBuilder for LinePlot {
    fn data_range(&self) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
        (min_max(&self.x), min_max(&self.y))
    }

    fn build_elements(&self, x_scale: &dyn Scale, y_scale: &dyn Scale, area: Rect) -> Vec<Element> {
        let points: Vec<(f64, f64)> = self
            .x
            .iter()
            .zip(&self.y)
            .map(|(&xv, &yv)| {
                let px = area.x + x_scale.transform(xv) * area.w;
                let py = area.y + area.h - y_scale.transform(yv) * area.h;
                (px, py)
            })
            .collect();

        vec![Element::Polyline {
            points,
            stroke: self.stroke.clone(),
            fill: None,
        }]
    }

    fn label(&self) -> Option<&str> {
        self.plot_label.as_deref()
    }
}

// ---------------------------------------------------------------------------
// ScatterPlot
// ---------------------------------------------------------------------------

/// A scatter plot drawing markers at `(x, y)` data points.
pub struct ScatterPlot {
    x: Vec<f64>,
    y: Vec<f64>,
    marker: Marker,
    plot_label: Option<String>,
}

impl ScatterPlot {
    /// Create a scatter plot from x and y data vectors.
    #[must_use]
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        Self {
            x,
            y,
            marker: Marker::default(),
            plot_label: None,
        }
    }

    /// Set the marker color.
    #[must_use]
    pub fn color(mut self, c: Color) -> Self {
        self.marker.color = c;
        self
    }

    /// Set the marker size (radius in pixels).
    #[must_use]
    pub fn size(mut self, s: f64) -> Self {
        self.marker.size = s;
        self
    }

    /// Set the marker shape.
    #[must_use]
    pub fn shape(mut self, s: MarkerShape) -> Self {
        self.marker.shape = s;
        self
    }

    /// Set the legend label for this scatter plot.
    #[must_use]
    pub fn label(mut self, l: &str) -> Self {
        self.plot_label = Some(l.to_string());
        self
    }
}

impl PlotBuilder for ScatterPlot {
    fn data_range(&self) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
        (min_max(&self.x), min_max(&self.y))
    }

    fn build_elements(&self, x_scale: &dyn Scale, y_scale: &dyn Scale, area: Rect) -> Vec<Element> {
        self.x
            .iter()
            .zip(&self.y)
            .map(|(&xv, &yv)| {
                let px = area.x + x_scale.transform(xv) * area.w;
                let py = area.y + area.h - y_scale.transform(yv) * area.h;
                Element::Circle {
                    cx: px,
                    cy: py,
                    r: self.marker.size,
                    fill: Some(Fill::new(self.marker.color)),
                    stroke: None,
                }
            })
            .collect()
    }

    fn label(&self) -> Option<&str> {
        self.plot_label.as_deref()
    }
}

// ---------------------------------------------------------------------------
// BarPlot
// ---------------------------------------------------------------------------

/// A bar chart for categorical data.
pub struct BarPlot {
    categories: Vec<String>,
    values: Vec<f64>,
    fill: Fill,
    stroke: Option<Stroke>,
    bar_width: f64,
    plot_label: Option<String>,
}

impl BarPlot {
    /// Create a bar plot from category names and corresponding values.
    #[must_use]
    pub fn new(categories: Vec<String>, values: Vec<f64>) -> Self {
        Self {
            categories,
            values,
            fill: Fill::new(Color::rgb(31, 119, 180)),
            stroke: Some(Stroke::new(Color::BLACK, 0.5)),
            bar_width: 0.8,
            plot_label: None,
        }
    }

    /// Set the bar fill color.
    #[must_use]
    pub fn color(mut self, c: Color) -> Self {
        self.fill = Fill::new(c);
        self
    }

    /// Set the bar width as a fraction of the category spacing (0.0--1.0).
    #[must_use]
    pub fn bar_width(mut self, w: f64) -> Self {
        self.bar_width = w;
        self
    }

    /// Set the legend label for this bar plot.
    #[must_use]
    pub fn label(mut self, l: &str) -> Self {
        self.plot_label = Some(l.to_string());
        self
    }
}

impl PlotBuilder for BarPlot {
    fn data_range(&self) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
        let n = self.categories.len();
        let x_range = if n > 0 {
            Some((-0.5, n as f64 - 0.5))
        } else {
            None
        };
        let y_max = self
            .values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let y_range = if y_max.is_finite() {
            Some((0.0, y_max))
        } else {
            None
        };
        (x_range, y_range)
    }

    fn build_elements(&self, x_scale: &dyn Scale, y_scale: &dyn Scale, area: Rect) -> Vec<Element> {
        let n = self.categories.len();
        if n == 0 {
            return vec![];
        }
        let bar_pixel_width = (x_scale.transform(self.bar_width / 2.0)
            - x_scale.transform(-self.bar_width / 2.0))
            * area.w;

        self.values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let cx = area.x + x_scale.transform(i as f64) * area.w;
                let top = area.y + area.h - y_scale.transform(v) * area.h;
                let bottom = area.y + area.h - y_scale.transform(0.0) * area.h;
                Element::Rect {
                    x: cx - bar_pixel_width / 2.0,
                    y: top,
                    w: bar_pixel_width,
                    h: bottom - top,
                    fill: Some(self.fill),
                    stroke: self.stroke.clone(),
                }
            })
            .collect()
    }

    fn label(&self) -> Option<&str> {
        self.plot_label.as_deref()
    }
}

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------

/// A histogram that bins continuous data.
pub struct Histogram {
    data: Vec<f64>,
    n_bins: usize,
    fill: Fill,
    stroke: Option<Stroke>,
    plot_label: Option<String>,
}

impl Histogram {
    /// Create a histogram from raw data with the given number of bins.
    #[must_use]
    pub fn new(data: Vec<f64>, n_bins: usize) -> Self {
        Self {
            data,
            n_bins,
            fill: Fill::new(Color::rgb(31, 119, 180)),
            stroke: Some(Stroke::new(Color::BLACK, 0.5)),
            plot_label: None,
        }
    }

    /// Set the bar fill color.
    #[must_use]
    pub fn color(mut self, c: Color) -> Self {
        self.fill = Fill::new(c);
        self
    }

    /// Set the legend label for this histogram.
    #[must_use]
    pub fn label(mut self, l: &str) -> Self {
        self.plot_label = Some(l.to_string());
        self
    }

    /// Compute bin edges and counts.
    fn compute_bins(&self) -> (Vec<f64>, Vec<usize>) {
        if self.data.is_empty() || self.n_bins == 0 {
            return (vec![], vec![]);
        }
        let min = self.data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        let bin_width = if range.abs() < f64::EPSILON {
            1.0
        } else {
            range / self.n_bins as f64
        };

        let mut edges = Vec::with_capacity(self.n_bins + 1);
        for i in 0..=self.n_bins {
            edges.push(min + i as f64 * bin_width);
        }

        let mut counts = vec![0usize; self.n_bins];
        for &v in &self.data {
            let idx = ((v - min) / bin_width) as usize;
            let idx = idx.min(self.n_bins - 1);
            counts[idx] += 1;
        }
        (edges, counts)
    }
}

impl PlotBuilder for Histogram {
    fn data_range(&self) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
        let (edges, counts) = self.compute_bins();
        let x_range = if edges.len() >= 2 {
            Some((edges[0], edges[edges.len() - 1]))
        } else {
            None
        };
        let y_max = counts.iter().copied().max().unwrap_or(0);
        let y_range = if y_max > 0 {
            Some((0.0, y_max as f64))
        } else {
            None
        };
        (x_range, y_range)
    }

    fn build_elements(&self, x_scale: &dyn Scale, y_scale: &dyn Scale, area: Rect) -> Vec<Element> {
        let (edges, counts) = self.compute_bins();
        if edges.len() < 2 {
            return vec![];
        }

        counts
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let x_left = area.x + x_scale.transform(edges[i]) * area.w;
                let x_right = area.x + x_scale.transform(edges[i + 1]) * area.w;
                let top = area.y + area.h - y_scale.transform(count as f64) * area.h;
                let bottom = area.y + area.h - y_scale.transform(0.0) * area.h;
                Element::Rect {
                    x: x_left,
                    y: top,
                    w: x_right - x_left,
                    h: bottom - top,
                    fill: Some(self.fill),
                    stroke: self.stroke.clone(),
                }
            })
            .collect()
    }

    fn label(&self) -> Option<&str> {
        self.plot_label.as_deref()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Rect;
    use crate::scale::LinearScale;

    #[test]
    fn line_plot_data_range() {
        let p = LinePlot::new(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]);
        let (xr, yr) = p.data_range();
        assert_eq!(xr, Some((1.0, 3.0)));
        assert_eq!(yr, Some((4.0, 6.0)));
    }

    #[test]
    fn line_plot_produces_polyline() {
        let p = LinePlot::new(vec![0.0, 1.0], vec![0.0, 1.0]);
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        let area = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 100.0,
        };
        let elems = p.build_elements(&xs, &ys, area);
        assert_eq!(elems.len(), 1);
        assert!(matches!(elems[0], Element::Polyline { .. }));
    }

    #[test]
    fn scatter_plot_element_count() {
        let p = ScatterPlot::new(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]);
        let xs = LinearScale::new(1.0, 3.0);
        let ys = LinearScale::new(4.0, 6.0);
        let area = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 100.0,
        };
        let elems = p.build_elements(&xs, &ys, area);
        assert_eq!(elems.len(), 3);
        assert!(elems.iter().all(|e| matches!(e, Element::Circle { .. })));
    }

    #[test]
    fn bar_plot_element_count() {
        let p = BarPlot::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![10.0, 20.0, 30.0],
        );
        let (xr, yr) = p.data_range();
        let xs = LinearScale::new(xr.unwrap().0, xr.unwrap().1);
        let ys = LinearScale::new(yr.unwrap().0, yr.unwrap().1);
        let area = Rect {
            x: 0.0,
            y: 0.0,
            w: 300.0,
            h: 200.0,
        };
        let elems = p.build_elements(&xs, &ys, area);
        assert_eq!(elems.len(), 3);
    }

    #[test]
    fn line_plot_color() {
        let p = LinePlot::new(vec![0.0], vec![0.0]).color(Color::RED);
        assert_eq!(p.stroke.color, Color::RED);
    }

    #[test]
    fn line_plot_width() {
        let p = LinePlot::new(vec![0.0], vec![0.0]).width(3.0);
        assert!((p.stroke.width - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn line_plot_dash() {
        let p = LinePlot::new(vec![0.0], vec![0.0]).dash(vec![4.0, 2.0]);
        assert_eq!(p.stroke.dash, Some(vec![4.0, 2.0]));
    }

    #[test]
    fn line_plot_label() {
        let p = LinePlot::new(vec![0.0], vec![0.0]).label("test");
        assert_eq!(PlotBuilder::label(&p), Some("test"));
    }

    #[test]
    fn line_plot_no_label() {
        let p = LinePlot::new(vec![0.0], vec![0.0]);
        assert_eq!(PlotBuilder::label(&p), None);
    }

    #[test]
    fn line_plot_empty_data_range() {
        let p = LinePlot::new(vec![], vec![]);
        let (xr, yr) = p.data_range();
        assert!(xr.is_none());
        assert!(yr.is_none());
    }

    #[test]
    fn scatter_plot_color() {
        let p = ScatterPlot::new(vec![0.0], vec![0.0]).color(Color::GREEN);
        assert_eq!(p.marker.color, Color::GREEN);
    }

    #[test]
    fn scatter_plot_size() {
        let p = ScatterPlot::new(vec![0.0], vec![0.0]).size(8.0);
        assert!((p.marker.size - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn scatter_plot_shape() {
        let p = ScatterPlot::new(vec![0.0], vec![0.0]).shape(MarkerShape::Diamond);
        assert_eq!(p.marker.shape, MarkerShape::Diamond);
    }

    #[test]
    fn scatter_plot_label() {
        let p = ScatterPlot::new(vec![0.0], vec![0.0]).label("pts");
        assert_eq!(PlotBuilder::label(&p), Some("pts"));
    }

    #[test]
    fn scatter_plot_data_range() {
        let p = ScatterPlot::new(vec![1.0, 5.0], vec![2.0, 8.0]);
        let (xr, yr) = p.data_range();
        assert_eq!(xr, Some((1.0, 5.0)));
        assert_eq!(yr, Some((2.0, 8.0)));
    }

    #[test]
    fn bar_plot_color() {
        let p = BarPlot::new(vec!["A".into()], vec![1.0]).color(Color::ORANGE);
        assert_eq!(p.fill.color, Color::ORANGE);
    }

    #[test]
    fn bar_plot_bar_width() {
        let p = BarPlot::new(vec!["A".into()], vec![1.0]).bar_width(0.5);
        assert!((p.bar_width - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn bar_plot_label() {
        let p = BarPlot::new(vec!["A".into()], vec![1.0]).label("bars");
        assert_eq!(PlotBuilder::label(&p), Some("bars"));
    }

    #[test]
    fn bar_plot_empty() {
        let p = BarPlot::new(vec![], vec![]);
        let (xr, _yr) = p.data_range();
        assert!(xr.is_none());
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        let area = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 100.0,
        };
        assert!(p.build_elements(&xs, &ys, area).is_empty());
    }

    #[test]
    fn histogram_color() {
        let h = Histogram::new(vec![1.0, 2.0], 2).color(Color::PURPLE);
        assert_eq!(h.fill.color, Color::PURPLE);
    }

    #[test]
    fn histogram_label() {
        let h = Histogram::new(vec![1.0, 2.0], 2).label("hist");
        assert_eq!(PlotBuilder::label(&h), Some("hist"));
    }

    #[test]
    fn histogram_empty_data() {
        let h = Histogram::new(vec![], 5);
        let (xr, yr) = h.data_range();
        assert!(xr.is_none());
        assert!(yr.is_none());
    }

    #[test]
    fn min_max_basic() {
        assert_eq!(min_max(&[3.0, 1.0, 4.0, 1.5, 9.2]), Some((1.0, 9.2)));
        assert_eq!(min_max(&[]), None);
    }

    #[test]
    fn histogram_bins() {
        let data = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let h = Histogram::new(data, 4);
        let (xr, yr) = h.data_range();
        assert!(xr.is_some());
        assert!(yr.is_some());
        let xs = LinearScale::new(xr.unwrap().0, xr.unwrap().1);
        let ys = LinearScale::new(yr.unwrap().0, yr.unwrap().1);
        let area = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 150.0,
        };
        let elems = h.build_elements(&xs, &ys, area);
        assert_eq!(elems.len(), 4);
    }
}
