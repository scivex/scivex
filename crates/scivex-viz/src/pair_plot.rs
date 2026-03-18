//! Pair plot: an n×n grid of scatter plots with diagonal histograms.
//!
//! Visualizes pairwise relationships between multiple variables. Diagonal
//! cells show histograms, off-diagonal cells show scatter plots.

use crate::axes::Axes;
use crate::color::Color;
use crate::figure::Figure;
use crate::layout::Layout;
use crate::plot::{Histogram, LinePlot, ScatterPlot};

/// Builder for a pair plot (scatter matrix).
///
/// # Example
///
/// ```rust,no_run
/// use scivex_viz::pair_plot::PairPlot;
///
/// let data = vec![
///     vec![1.0, 2.0, 3.0, 4.0],
///     vec![4.0, 3.0, 2.0, 1.0],
///     vec![1.0, 4.0, 9.0, 16.0],
/// ];
/// let fig = PairPlot::new(data).build();
/// ```
pub struct PairPlot {
    /// Each inner Vec is one variable's data.
    columns: Vec<Vec<f64>>,
    labels: Option<Vec<String>>,
    point_color: Color,
    point_size: f64,
    hist_bins: usize,
    hist_color: Color,
    fig_size: (f64, f64),
    diag_mode: DiagMode,
}

/// What to show on the diagonal cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagMode {
    /// Histogram of each variable.
    Histogram,
    /// Kernel density estimate (approximated via histogram outline).
    Kde,
}

impl PairPlot {
    /// Create a new pair plot from column data.
    ///
    /// Each element of `columns` is a vector of observations for one variable.
    /// All columns must have the same length.
    #[must_use]
    pub fn new(columns: Vec<Vec<f64>>) -> Self {
        Self {
            columns,
            labels: None,
            point_color: Color::rgba(70, 130, 180, 160),
            point_size: 3.0,
            hist_bins: 20,
            hist_color: Color::rgba(70, 130, 180, 180),
            fig_size: (800.0, 800.0),
            diag_mode: DiagMode::Histogram,
        }
    }

    /// Set variable labels (one per column).
    #[must_use]
    pub fn labels(mut self, labels: &[&str]) -> Self {
        self.labels = Some(labels.iter().map(|s| (*s).to_string()).collect());
        self
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

    /// Set the overall figure size.
    #[must_use]
    pub fn fig_size(mut self, w: f64, h: f64) -> Self {
        self.fig_size = (w, h);
        self
    }

    /// Set the diagonal cell mode.
    #[must_use]
    pub fn diag(mut self, mode: DiagMode) -> Self {
        self.diag_mode = mode;
        self
    }

    /// Build the pair plot into a [`Figure`].
    #[must_use]
    pub fn build(&self) -> Figure {
        let n = self.columns.len();
        if n == 0 {
            return Figure::new();
        }

        let mut fig = Figure::new()
            .size(self.fig_size.0, self.fig_size.1)
            .layout(Layout::grid(n, n))
            .share_x(true)
            .share_y(false);

        for row in 0..n {
            for col in 0..n {
                let axes = if row == col {
                    self.build_diagonal(row)
                } else {
                    self.build_scatter(col, row)
                };
                fig = fig.add_axes(row, col, axes);
            }
        }

        fig
    }

    fn build_diagonal(&self, idx: usize) -> Axes {
        let data = &self.columns[idx];
        let label = self
            .labels
            .as_ref()
            .and_then(|l| l.get(idx))
            .map(String::as_str);

        let mut axes = Axes::new().grid(false);
        if let Some(l) = label {
            axes = axes.title(l);
        }

        match self.diag_mode {
            DiagMode::Histogram => {
                axes = axes
                    .add_plot(Histogram::new(data.clone(), self.hist_bins).color(self.hist_color));
            }
            DiagMode::Kde => {
                // Approximate KDE with a smoothed histogram outline.
                let (xs, ys) = kde_estimate(data, 64);
                axes = axes.add_plot(LinePlot::new(xs, ys).color(self.hist_color));
            }
        }

        axes
    }

    fn build_scatter(&self, x_idx: usize, y_idx: usize) -> Axes {
        let x_data = self.columns[x_idx].clone();
        let y_data = self.columns[y_idx].clone();

        let x_label = self
            .labels
            .as_ref()
            .and_then(|l| l.get(x_idx))
            .map(String::as_str);
        let y_label = self
            .labels
            .as_ref()
            .and_then(|l| l.get(y_idx))
            .map(String::as_str);

        let mut axes = Axes::new().grid(true);
        if x_label.is_some_and(|_| y_idx == self.columns.len() - 1) {
            axes = axes.x_label(x_label.unwrap());
        }
        if y_label.is_some_and(|_| x_idx == 0) {
            axes = axes.y_label(y_label.unwrap());
        }

        axes = axes.add_plot(
            ScatterPlot::new(x_data, y_data)
                .color(self.point_color)
                .size(self.point_size),
        );

        axes
    }
}

/// Simple Gaussian KDE approximation using histogram-based evaluation.
fn kde_estimate(data: &[f64], n_points: usize) -> (Vec<f64>, Vec<f64>) {
    if data.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in data {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    if (hi - lo).abs() < f64::EPSILON {
        lo -= 1.0;
        hi += 1.0;
    }

    // Silverman's rule of thumb for bandwidth.
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let bw = 1.06 * std * n.powf(-0.2);
    let bw = if bw < 1e-10 { 1.0 } else { bw };

    let margin = 3.0 * bw;
    let x_lo = lo - margin;
    let x_hi = hi + margin;
    let step = (x_hi - x_lo) / (n_points - 1) as f64;

    let mut xs = Vec::with_capacity(n_points);
    let mut ys = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let x = x_lo + i as f64 * step;
        let mut density = 0.0;
        for &d in data {
            let u = (x - d) / bw;
            density += (-0.5 * u * u).exp();
        }
        density /= n * bw * (2.0 * std::f64::consts::PI).sqrt();
        xs.push(x);
        ys.push(density);
    }

    (xs, ys)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_columns() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
            vec![1.0, 4.0, 9.0, 16.0, 25.0],
        ]
    }

    #[test]
    fn pair_plot_builds_figure() {
        let fig = PairPlot::new(sample_columns()).build();
        let svg = fig.to_svg().unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn pair_plot_with_labels() {
        let fig = PairPlot::new(sample_columns())
            .labels(&["x", "y", "z"])
            .build();
        let svg = fig.to_svg().unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn pair_plot_kde_diagonal() {
        let fig = PairPlot::new(sample_columns()).diag(DiagMode::Kde).build();
        let svg = fig.to_svg().unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn pair_plot_empty_data() {
        let fig = PairPlot::new(vec![]).build();
        let svg = fig.to_svg().unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn pair_plot_builder_methods() {
        let pp = PairPlot::new(sample_columns())
            .point_color(Color::RED)
            .point_size(5.0)
            .hist_bins(30)
            .hist_color(Color::BLUE)
            .fig_size(1200.0, 1200.0);
        assert_eq!(pp.point_color, Color::RED);
        assert!((pp.point_size - 5.0).abs() < f64::EPSILON);
        assert_eq!(pp.hist_bins, 30);
    }

    #[test]
    fn kde_estimate_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (xs, ys) = kde_estimate(&data, 32);
        assert_eq!(xs.len(), 32);
        assert_eq!(ys.len(), 32);
        // All densities should be non-negative.
        for &y in &ys {
            assert!(y >= 0.0);
        }
    }

    #[test]
    fn kde_estimate_empty() {
        let (xs, ys) = kde_estimate(&[], 16);
        assert!(xs.is_empty());
        assert!(ys.is_empty());
    }
}
