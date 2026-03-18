//! Joint plot: central scatter plot with marginal histograms.
//!
//! Uses a weighted grid layout with a large central scatter panel and smaller
//! marginal panels on the top (x-distribution) and right (y-distribution).

use crate::axes::Axes;
use crate::color::Color;
use crate::figure::Figure;
use crate::layout::Layout;
use crate::plot::{Histogram, ScatterPlot};

/// Builder for a joint plot (scatter + marginal histograms).
///
/// # Example
///
/// ```rust,no_run
/// use scivex_viz::joint_plot::JointPlot;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
/// let fig = JointPlot::new(x, y).build();
/// ```
pub struct JointPlot {
    x: Vec<f64>,
    y: Vec<f64>,
    point_color: Color,
    point_size: f64,
    hist_bins: usize,
    hist_color: Color,
    x_label: Option<String>,
    y_label: Option<String>,
    fig_size: (f64, f64),
    marginal_ratio: f64,
}

impl JointPlot {
    /// Create a new joint plot from x and y data.
    #[must_use]
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        Self {
            x,
            y,
            point_color: Color::rgba(70, 130, 180, 160),
            point_size: 4.0,
            hist_bins: 20,
            hist_color: Color::rgba(70, 130, 180, 180),
            x_label: None,
            y_label: None,
            fig_size: (700.0, 700.0),
            marginal_ratio: 0.2,
        }
    }

    /// Set the scatter point color.
    #[must_use]
    pub fn point_color(mut self, c: Color) -> Self {
        self.point_color = c;
        self
    }

    /// Set the scatter point size.
    #[must_use]
    pub fn point_size(mut self, s: f64) -> Self {
        self.point_size = s;
        self
    }

    /// Set the number of histogram bins.
    #[must_use]
    pub fn hist_bins(mut self, n: usize) -> Self {
        self.hist_bins = n;
        self
    }

    /// Set the histogram fill color.
    #[must_use]
    pub fn hist_color(mut self, c: Color) -> Self {
        self.hist_color = c;
        self
    }

    /// Set the x-axis label.
    #[must_use]
    pub fn x_label(mut self, l: &str) -> Self {
        self.x_label = Some(l.to_string());
        self
    }

    /// Set the y-axis label.
    #[must_use]
    pub fn y_label(mut self, l: &str) -> Self {
        self.y_label = Some(l.to_string());
        self
    }

    /// Set the overall figure size.
    #[must_use]
    pub fn fig_size(mut self, w: f64, h: f64) -> Self {
        self.fig_size = (w, h);
        self
    }

    /// Set the marginal panel ratio (fraction of figure width/height).
    ///
    /// Default is 0.2 (marginal panels take 20% of the figure dimension).
    #[must_use]
    pub fn marginal_ratio(mut self, r: f64) -> Self {
        self.marginal_ratio = r.clamp(0.05, 0.5);
        self
    }

    /// Build the joint plot into a [`Figure`].
    ///
    /// Layout:
    /// ```text
    /// ┌─────────┬───┐
    /// │ x-hist  │   │  row 0 (small)
    /// ├─────────┼───┤
    /// │ scatter │ y │  row 1 (large)
    /// │         │hist│
    /// └─────────┴───┘
    ///   col 0     col 1
    ///  (large)   (small)
    /// ```
    #[must_use]
    pub fn build(&self) -> Figure {
        let m = self.marginal_ratio;
        let main = 1.0 - m;

        let layout = Layout::weighted_grid(vec![m, main], vec![main, m]);

        // Top-left: x marginal histogram.
        let x_marginal = Axes::new()
            .grid(false)
            .hide_x_ticks(true)
            .hide_y_ticks(true)
            .add_plot(Histogram::new(self.x.clone(), self.hist_bins).color(self.hist_color));

        // Bottom-right: y marginal histogram (plotted as horizontal via scatter of y).
        // We approximate by rotating: use y as x-data for the histogram.
        let y_marginal = Axes::new()
            .grid(false)
            .hide_x_ticks(true)
            .hide_y_ticks(true)
            .add_plot(Histogram::new(self.y.clone(), self.hist_bins).color(self.hist_color));

        // Center: scatter plot.
        let mut scatter_axes = Axes::new().grid(true).add_plot(
            ScatterPlot::new(self.x.clone(), self.y.clone())
                .color(self.point_color)
                .size(self.point_size),
        );

        if let Some(ref l) = self.x_label {
            scatter_axes = scatter_axes.x_label(l);
        }
        if let Some(ref l) = self.y_label {
            scatter_axes = scatter_axes.y_label(l);
        }

        // Top-right: empty cell.
        let empty = Axes::new()
            .grid(false)
            .hide_x_ticks(true)
            .hide_y_ticks(true);

        Figure::new()
            .size(self.fig_size.0, self.fig_size.1)
            .layout(layout)
            .add_axes(0, 0, x_marginal)
            .add_axes(0, 1, empty)
            .add_axes(1, 0, scatter_axes)
            .add_axes(1, 1, y_marginal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_plot_builds() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let fig = JointPlot::new(x, y).build();
        let svg = fig.to_svg().unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn joint_plot_with_labels() {
        let fig = JointPlot::new(vec![1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0])
            .x_label("Feature A")
            .y_label("Feature B")
            .build();
        let svg = fig.to_svg().unwrap();
        assert!(svg.contains("Feature A"));
        assert!(svg.contains("Feature B"));
    }

    #[test]
    fn joint_plot_builder_methods() {
        let jp = JointPlot::new(vec![1.0], vec![1.0])
            .point_color(Color::RED)
            .point_size(6.0)
            .hist_bins(30)
            .hist_color(Color::GREEN)
            .fig_size(900.0, 900.0)
            .marginal_ratio(0.3);
        assert_eq!(jp.point_color, Color::RED);
        assert!((jp.point_size - 6.0).abs() < f64::EPSILON);
        assert_eq!(jp.hist_bins, 30);
        assert!((jp.marginal_ratio - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn joint_plot_marginal_clamped() {
        let jp = JointPlot::new(vec![1.0], vec![1.0]).marginal_ratio(0.8);
        assert!((jp.marginal_ratio - 0.5).abs() < f64::EPSILON);
        let jp2 = JointPlot::new(vec![1.0], vec![1.0]).marginal_ratio(0.01);
        assert!((jp2.marginal_ratio - 0.05).abs() < f64::EPSILON);
    }
}
