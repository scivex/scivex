//! Statistical visualization plot builders: regression, residual, QQ, and
//! correlation heatmap.

use crate::color::{Color, ColorMap};
use crate::element::{Element, TextAnchor};
use crate::layout::Rect;
use crate::plot::{AxisRange, PlotBuilder};
use crate::scale::Scale;
use crate::style::{Fill, Font, Stroke};

// ---------------------------------------------------------------------------
// Private OLS helpers
// ---------------------------------------------------------------------------

/// Ordinary least-squares fit returning `(slope, intercept)`.
fn ols_fit(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let n = x.len();
    if n < 2 || n != y.len() {
        return None;
    }
    let n_f = n as f64;
    let x_bar: f64 = x.iter().sum::<f64>() / n_f;
    let y_bar: f64 = y.iter().sum::<f64>() / n_f;

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    for i in 0..n {
        let dx = x[i] - x_bar;
        ss_xy += dx * (y[i] - y_bar);
        ss_xx += dx * dx;
    }
    if ss_xx.abs() < f64::EPSILON {
        return None;
    }
    let slope = ss_xy / ss_xx;
    let intercept = y_bar - slope * x_bar;
    Some((slope, intercept))
}

/// Standard error of the regression and of predictions.
/// Returns `(se_residual, ss_xx, x_bar)`.
fn ols_diagnostics(x: &[f64], y: &[f64], slope: f64, intercept: f64) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let x_bar = x.iter().sum::<f64>() / n;
    let ss_xx: f64 = x.iter().map(|&xi| (xi - x_bar).powi(2)).sum();
    let ss_res: f64 = x
        .iter()
        .zip(y)
        .map(|(&xi, &yi)| {
            let pred = slope * xi + intercept;
            (yi - pred).powi(2)
        })
        .sum();
    let dof = (n - 2.0).max(1.0);
    let se = (ss_res / dof).sqrt();
    (se, ss_xx, x_bar)
}

/// Min/max of a slice, returning `None` for empty or non-finite data.
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

// ---------------------------------------------------------------------------
// Inverse normal CDF (rational approximation — Abramowitz & Stegun 26.2.23)
// ---------------------------------------------------------------------------

/// Approximate inverse of the standard normal CDF for `p` in (0, 1).
/// Accuracy ~4.5e-4 using the Abramowitz & Stegun rational approximation.
fn inv_normal_cdf(p: f64) -> f64 {
    // Constants from A&S 26.2.23
    const C0: f64 = 2.515_517;
    const C1: f64 = 0.802_853;
    const C2: f64 = 0.010_328;
    const D1: f64 = 1.432_788;
    const D2: f64 = 0.189_269;
    const D3: f64 = 0.001_308;

    let p_clamped = p.clamp(1e-12, 1.0 - 1e-12);
    let sign = if p_clamped < 0.5 { -1.0 } else { 1.0 };
    let q = if p_clamped < 0.5 {
        p_clamped
    } else {
        1.0 - p_clamped
    };
    let t = (-2.0 * q.ln()).sqrt();
    let num = C0 + C1 * t + C2 * t * t;
    let den = 1.0 + D1 * t + D2 * t * t + D3 * t * t * t;
    sign * (t - num / den)
}

// ---------------------------------------------------------------------------
// RegressionPlot
// ---------------------------------------------------------------------------

/// Scatter plot with a fitted OLS regression line and optional confidence band.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct RegressionPlot {
    x: Vec<f64>,
    y: Vec<f64>,
    show_ci: bool,
    ci_level: f64,
    scatter_color: Color,
    line_color: Color,
    band_color: Color,
    plot_label: Option<String>,
}

impl RegressionPlot {
    /// Create a regression plot from x and y data vectors.
    #[must_use]
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        Self {
            x,
            y,
            show_ci: false,
            ci_level: 0.95,
            scatter_color: Color::rgb(31, 119, 180),
            line_color: Color::RED,
            band_color: Color::rgba(255, 0, 0, 50),
            plot_label: None,
        }
    }

    /// Show or hide the confidence interval band.
    #[must_use]
    pub fn show_ci(mut self, show: bool) -> Self {
        self.show_ci = show;
        self
    }

    /// Set the confidence level (e.g. 0.95 for 95%).
    #[must_use]
    pub fn ci_level(mut self, level: f64) -> Self {
        self.ci_level = level.clamp(0.01, 0.999);
        self
    }

    /// Set the scatter point color.
    #[must_use]
    pub fn scatter_color(mut self, c: Color) -> Self {
        self.scatter_color = c;
        self
    }

    /// Set the regression line color.
    #[must_use]
    pub fn line_color(mut self, c: Color) -> Self {
        self.line_color = c;
        self
    }

    /// Set the legend label.
    #[must_use]
    pub fn label(mut self, l: &str) -> Self {
        self.plot_label = Some(l.to_string());
        self
    }
}

impl PlotBuilder for RegressionPlot {
    fn data_range(&self) -> (AxisRange, AxisRange) {
        (min_max(&self.x), min_max(&self.y))
    }

    #[allow(clippy::collapsible_if)]
    fn build_elements(&self, x_scale: &dyn Scale, y_scale: &dyn Scale, area: Rect) -> Vec<Element> {
        if self.x.is_empty() || self.x.len() != self.y.len() {
            return vec![];
        }

        let mut elements = Vec::new();

        // Scatter points
        for (&xv, &yv) in self.x.iter().zip(&self.y) {
            let px = area.x + x_scale.transform(xv) * area.w;
            let py = area.y + area.h - y_scale.transform(yv) * area.h;
            elements.push(Element::Circle {
                cx: px,
                cy: py,
                r: 3.0,
                fill: Some(Fill::new(self.scatter_color)),
                stroke: None,
            });
        }

        // OLS fit
        let Some((slope, intercept)) = ols_fit(&self.x, &self.y) else {
            return elements;
        };

        // Regression line — draw across x data range
        let x_min = self.x.iter().copied().fold(f64::INFINITY, f64::min);
        let x_max = self.x.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let n_line = 50;
        let step = (x_max - x_min) / n_line as f64;
        let line_points: Vec<(f64, f64)> = (0..=n_line)
            .map(|i| {
                let xv = x_min + i as f64 * step;
                let yv = slope * xv + intercept;
                let px = area.x + x_scale.transform(xv) * area.w;
                let py = area.y + area.h - y_scale.transform(yv) * area.h;
                (px, py)
            })
            .collect();

        elements.push(Element::Polyline {
            points: line_points,
            stroke: Stroke::new(self.line_color, 2.0),
            fill: None,
        });

        // Confidence band
        if self.show_ci {
            let (se, ss_xx, x_bar) = ols_diagnostics(&self.x, &self.y, slope, intercept);
            let n_f = self.x.len() as f64;
            let alpha = 1.0 - self.ci_level;
            let z = inv_normal_cdf(1.0 - alpha / 2.0);

            let mut upper_points = Vec::with_capacity(n_line + 1);
            let mut lower_points = Vec::with_capacity(n_line + 1);

            for i in 0..=n_line {
                let xv = x_min + i as f64 * step;
                let yhat = slope * xv + intercept;
                let se_pred = se * (1.0 / n_f + (xv - x_bar).powi(2) / ss_xx).sqrt();
                let margin = z * se_pred;

                let y_upper = yhat + margin;
                let y_lower = yhat - margin;

                let px = area.x + x_scale.transform(xv) * area.w;
                let py_upper = area.y + area.h - y_scale.transform(y_upper) * area.h;
                let py_lower = area.y + area.h - y_scale.transform(y_lower) * area.h;

                upper_points.push((px, py_upper));
                lower_points.push((px, py_lower));
            }

            // Build a closed polygon: upper forward, lower reversed
            let mut band_points = upper_points;
            lower_points.reverse();
            band_points.extend(lower_points);

            elements.push(Element::Polyline {
                points: band_points,
                stroke: Stroke::new(self.band_color, 0.0),
                fill: Some(Fill::new(self.band_color)),
            });
        }

        elements
    }

    fn label(&self) -> Option<&str> {
        self.plot_label.as_deref()
    }
}

// ---------------------------------------------------------------------------
// ResidualPlot
// ---------------------------------------------------------------------------

/// Residuals vs fitted values plot.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ResidualPlot {
    x: Vec<f64>,
    y: Vec<f64>,
    marker_color: Color,
    zero_line_color: Color,
    plot_label: Option<String>,
}

impl ResidualPlot {
    /// Create a residual plot from x and y data vectors.
    #[must_use]
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        Self {
            x,
            y,
            marker_color: Color::rgb(31, 119, 180),
            zero_line_color: Color::GRAY,
            plot_label: None,
        }
    }

    /// Set the marker color.
    #[must_use]
    pub fn marker_color(mut self, c: Color) -> Self {
        self.marker_color = c;
        self
    }

    /// Set the legend label.
    #[must_use]
    pub fn label(mut self, l: &str) -> Self {
        self.plot_label = Some(l.to_string());
        self
    }
}

impl PlotBuilder for ResidualPlot {
    fn data_range(&self) -> (AxisRange, AxisRange) {
        if self.x.is_empty() || self.x.len() != self.y.len() {
            return (None, None);
        }
        let Some((slope, intercept)) = ols_fit(&self.x, &self.y) else {
            return (None, None);
        };

        let fitted: Vec<f64> = self.x.iter().map(|&xi| slope * xi + intercept).collect();
        let residuals: Vec<f64> = self
            .y
            .iter()
            .zip(&fitted)
            .map(|(&yi, &fi)| yi - fi)
            .collect();
        (min_max(&fitted), min_max(&residuals))
    }

    fn build_elements(&self, x_scale: &dyn Scale, y_scale: &dyn Scale, area: Rect) -> Vec<Element> {
        if self.x.is_empty() || self.x.len() != self.y.len() {
            return vec![];
        }
        let Some((slope, intercept)) = ols_fit(&self.x, &self.y) else {
            return vec![];
        };

        let mut elements = Vec::new();

        // Zero line
        let px_left = area.x;
        let px_right = area.x + area.w;
        let py_zero = area.y + area.h - y_scale.transform(0.0) * area.h;
        elements.push(Element::Line {
            x1: px_left,
            y1: py_zero,
            x2: px_right,
            y2: py_zero,
            stroke: Stroke::new(self.zero_line_color, 1.0).dashed(vec![5.0, 3.0]),
        });

        // Scatter: fitted vs residuals
        for (&xi, &yi) in self.x.iter().zip(&self.y) {
            let fitted = slope * xi + intercept;
            let residual = yi - fitted;
            let px = area.x + x_scale.transform(fitted) * area.w;
            let py = area.y + area.h - y_scale.transform(residual) * area.h;
            elements.push(Element::Circle {
                cx: px,
                cy: py,
                r: 3.0,
                fill: Some(Fill::new(self.marker_color)),
                stroke: None,
            });
        }

        elements
    }

    fn label(&self) -> Option<&str> {
        self.plot_label.as_deref()
    }
}

// ---------------------------------------------------------------------------
// QQPlot
// ---------------------------------------------------------------------------

/// Quantile-quantile plot against the standard normal distribution.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct QQPlot {
    data: Vec<f64>,
    marker_color: Color,
    reference_color: Color,
    plot_label: Option<String>,
}

impl QQPlot {
    /// Create a QQ plot from a data vector.
    #[must_use]
    pub fn new(data: Vec<f64>) -> Self {
        Self {
            data,
            marker_color: Color::rgb(31, 119, 180),
            reference_color: Color::RED,
            plot_label: None,
        }
    }

    /// Set the marker color.
    #[must_use]
    pub fn marker_color(mut self, c: Color) -> Self {
        self.marker_color = c;
        self
    }

    /// Set the 45-degree reference line color.
    #[must_use]
    pub fn reference_color(mut self, c: Color) -> Self {
        self.reference_color = c;
        self
    }

    /// Set the legend label.
    #[must_use]
    pub fn label(mut self, l: &str) -> Self {
        self.plot_label = Some(l.to_string());
        self
    }

    /// Compute theoretical and sample quantiles.
    fn compute_quantiles(&self) -> (Vec<f64>, Vec<f64>) {
        if self.data.is_empty() {
            return (vec![], vec![]);
        }
        let mut sorted = self.data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();

        // Standardize data
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let var: f64 = sorted.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = var.sqrt().max(f64::EPSILON);
        let sample_quantiles: Vec<f64> = sorted.iter().map(|&v| (v - mean) / std_dev).collect();

        // Theoretical quantiles using inverse normal CDF with plotting position
        let theoretical: Vec<f64> = (0..n)
            .map(|i| {
                let p = (i as f64 + 0.5) / n as f64;
                inv_normal_cdf(p)
            })
            .collect();

        (theoretical, sample_quantiles)
    }
}

impl PlotBuilder for QQPlot {
    fn data_range(&self) -> (AxisRange, AxisRange) {
        let (theoretical, sample) = self.compute_quantiles();
        if theoretical.is_empty() {
            return (None, None);
        }
        // Use same range for both axes so the 45° line is meaningful
        let all_vals: Vec<f64> = theoretical.iter().chain(sample.iter()).copied().collect();
        let range = min_max(&all_vals);
        (range, range)
    }

    fn build_elements(&self, x_scale: &dyn Scale, y_scale: &dyn Scale, area: Rect) -> Vec<Element> {
        let (theoretical, sample) = self.compute_quantiles();
        if theoretical.is_empty() {
            return vec![];
        }

        let mut elements = Vec::new();

        // Compute extent for the reference line
        let all_vals: Vec<f64> = theoretical.iter().chain(sample.iter()).copied().collect();
        let (lo, hi) = min_max(&all_vals).unwrap_or((-3.0, 3.0));

        // 45° reference line (y = x)
        let px1 = area.x + x_scale.transform(lo) * area.w;
        let py1 = area.y + area.h - y_scale.transform(lo) * area.h;
        let px2 = area.x + x_scale.transform(hi) * area.w;
        let py2 = area.y + area.h - y_scale.transform(hi) * area.h;
        elements.push(Element::Line {
            x1: px1,
            y1: py1,
            x2: px2,
            y2: py2,
            stroke: Stroke::new(self.reference_color, 1.5).dashed(vec![6.0, 3.0]),
        });

        // Scatter: theoretical (x) vs sample (y)
        for (&tq, &sq) in theoretical.iter().zip(&sample) {
            let px = area.x + x_scale.transform(tq) * area.w;
            let py = area.y + area.h - y_scale.transform(sq) * area.h;
            elements.push(Element::Circle {
                cx: px,
                cy: py,
                r: 3.0,
                fill: Some(Fill::new(self.marker_color)),
                stroke: None,
            });
        }

        elements
    }

    fn label(&self) -> Option<&str> {
        self.plot_label.as_deref()
    }
}

// ---------------------------------------------------------------------------
// CorrelationHeatmap
// ---------------------------------------------------------------------------

/// A heatmap for visualizing a correlation matrix with values in \[-1, 1\].
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct CorrelationHeatmap {
    matrix: Vec<Vec<f64>>,
    labels: Option<Vec<String>>,
    colormap: ColorMap,
    show_values: bool,
    plot_label: Option<String>,
}

impl CorrelationHeatmap {
    /// Create a correlation heatmap from an `n x n` matrix with values in \[-1, 1\].
    #[must_use]
    pub fn new(matrix: Vec<Vec<f64>>) -> Self {
        Self {
            matrix,
            labels: None,
            colormap: ColorMap::coolwarm(),
            show_values: true,
            plot_label: None,
        }
    }

    /// Set axis labels.
    #[must_use]
    pub fn labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Set the colormap.
    #[must_use]
    pub fn colormap(mut self, cm: ColorMap) -> Self {
        self.colormap = cm;
        self
    }

    /// Show or hide numeric values inside each cell.
    #[must_use]
    pub fn show_values(mut self, show: bool) -> Self {
        self.show_values = show;
        self
    }

    /// Set the legend label.
    #[must_use]
    pub fn label(mut self, l: &str) -> Self {
        self.plot_label = Some(l.to_string());
        self
    }
}

impl PlotBuilder for CorrelationHeatmap {
    fn data_range(&self) -> (AxisRange, AxisRange) {
        let n = self.matrix.len();
        if n == 0 {
            return (None, None);
        }
        let range = Some((-0.5, n as f64 - 0.5));
        (range, range)
    }

    fn build_elements(
        &self,
        _x_scale: &dyn Scale,
        _y_scale: &dyn Scale,
        area: Rect,
    ) -> Vec<Element> {
        let n_rows = self.matrix.len();
        if n_rows == 0 {
            return vec![];
        }
        let n_cols = self.matrix.first().map_or(0, Vec::len);
        if n_cols == 0 {
            return vec![];
        }

        let cell_w = area.w / n_cols as f64;
        let cell_h = area.h / n_rows as f64;

        let mut elements = Vec::with_capacity(n_rows * n_cols * 2);
        for (r, row) in self.matrix.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                // Map [-1, 1] → [0, 1]
                let t = f64::midpoint(val.clamp(-1.0, 1.0), 1.0);
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
                        text: format!("{val:.2}"),
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
        self.plot_label.as_deref()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // RegressionPlot tests
    // -----------------------------------------------------------------------

    #[test]
    fn regression_plot_data_range() {
        let p = RegressionPlot::new(vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]);
        let (xr, yr) = p.data_range();
        assert_eq!(xr, Some((1.0, 3.0)));
        assert_eq!(yr, Some((2.0, 6.0)));
    }

    #[test]
    fn regression_plot_empty() {
        let p = RegressionPlot::new(vec![], vec![]);
        let (xr, yr) = p.data_range();
        assert!(xr.is_none());
        assert!(yr.is_none());
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        assert!(p.build_elements(&xs, &ys, test_area()).is_empty());
    }

    #[test]
    fn regression_plot_produces_scatter_and_line() {
        let p = RegressionPlot::new(vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 4.0, 5.0, 8.0]);
        let xs = LinearScale::new(1.0, 4.0);
        let ys = LinearScale::new(2.0, 8.0);
        let elems = p.build_elements(&xs, &ys, test_area());
        // 4 circles + 1 polyline (regression line)
        let circles = elems
            .iter()
            .filter(|e| matches!(e, Element::Circle { .. }))
            .count();
        let polylines = elems
            .iter()
            .filter(|e| matches!(e, Element::Polyline { .. }))
            .count();
        assert_eq!(circles, 4);
        assert_eq!(polylines, 1);
    }

    #[test]
    fn regression_plot_with_ci_produces_band() {
        let p = RegressionPlot::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 4.0, 5.0, 8.0, 10.0],
        )
        .show_ci(true);
        let xs = LinearScale::new(1.0, 5.0);
        let ys = LinearScale::new(0.0, 12.0);
        let elems = p.build_elements(&xs, &ys, test_area());
        // Should have at least one polyline for the band (filled)
        let filled_polylines = elems
            .iter()
            .filter(|e| matches!(e, Element::Polyline { fill: Some(_), .. }))
            .count();
        assert!(filled_polylines >= 1);
    }

    #[test]
    fn regression_plot_builder_methods() {
        let p = RegressionPlot::new(vec![1.0], vec![1.0])
            .show_ci(true)
            .ci_level(0.99)
            .scatter_color(Color::GREEN)
            .line_color(Color::BLUE)
            .label("regression");
        assert!(p.show_ci);
        assert!((p.ci_level - 0.99).abs() < f64::EPSILON);
        assert_eq!(p.scatter_color, Color::GREEN);
        assert_eq!(p.line_color, Color::BLUE);
        assert_eq!(PlotBuilder::label(&p), Some("regression"));
    }

    #[test]
    fn regression_plot_label_none() {
        let p = RegressionPlot::new(vec![1.0], vec![1.0]);
        assert_eq!(PlotBuilder::label(&p), None);
    }

    // -----------------------------------------------------------------------
    // ResidualPlot tests
    // -----------------------------------------------------------------------

    #[test]
    fn residual_plot_data_range() {
        let p = ResidualPlot::new(vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]);
        let (xr, yr) = p.data_range();
        // Perfect fit → all residuals are 0
        assert!(xr.is_some());
        assert!(yr.is_some());
    }

    #[test]
    fn residual_plot_empty() {
        let p = ResidualPlot::new(vec![], vec![]);
        let (xr, yr) = p.data_range();
        assert!(xr.is_none());
        assert!(yr.is_none());
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        assert!(p.build_elements(&xs, &ys, test_area()).is_empty());
    }

    #[test]
    fn residual_plot_produces_zero_line_and_scatter() {
        let p = ResidualPlot::new(vec![1.0, 2.0, 3.0, 4.0], vec![2.5, 3.8, 6.1, 7.9]);
        let (xr, yr) = p.data_range();
        let xs = LinearScale::new(xr.unwrap().0, xr.unwrap().1);
        let ys = LinearScale::new(yr.unwrap().0, yr.unwrap().1);
        let elems = p.build_elements(&xs, &ys, test_area());
        let lines = elems
            .iter()
            .filter(|e| matches!(e, Element::Line { .. }))
            .count();
        let circles = elems
            .iter()
            .filter(|e| matches!(e, Element::Circle { .. }))
            .count();
        assert_eq!(lines, 1); // zero line
        assert_eq!(circles, 4);
    }

    #[test]
    fn residual_plot_builder_methods() {
        let p = ResidualPlot::new(vec![1.0], vec![1.0])
            .marker_color(Color::ORANGE)
            .label("resid");
        assert_eq!(p.marker_color, Color::ORANGE);
        assert_eq!(PlotBuilder::label(&p), Some("resid"));
    }

    // -----------------------------------------------------------------------
    // QQPlot tests
    // -----------------------------------------------------------------------

    #[test]
    fn qq_plot_data_range() {
        let data: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
        let p = QQPlot::new(data);
        let (xr, yr) = p.data_range();
        assert!(xr.is_some());
        assert!(yr.is_some());
    }

    #[test]
    fn qq_plot_empty() {
        let p = QQPlot::new(vec![]);
        let (xr, yr) = p.data_range();
        assert!(xr.is_none());
        assert!(yr.is_none());
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        assert!(p.build_elements(&xs, &ys, test_area()).is_empty());
    }

    #[test]
    fn qq_plot_produces_reference_line_and_scatter() {
        let data: Vec<f64> = (1..=20).map(f64::from).collect();
        let p = QQPlot::new(data);
        let (xr, yr) = p.data_range();
        let xs = LinearScale::new(xr.unwrap().0, xr.unwrap().1);
        let ys = LinearScale::new(yr.unwrap().0, yr.unwrap().1);
        let elems = p.build_elements(&xs, &ys, test_area());
        let lines = elems
            .iter()
            .filter(|e| matches!(e, Element::Line { .. }))
            .count();
        let circles = elems
            .iter()
            .filter(|e| matches!(e, Element::Circle { .. }))
            .count();
        assert_eq!(lines, 1);
        assert_eq!(circles, 20);
    }

    #[test]
    fn qq_plot_builder_methods() {
        let p = QQPlot::new(vec![1.0, 2.0, 3.0])
            .marker_color(Color::GREEN)
            .reference_color(Color::BLUE)
            .label("qq");
        assert_eq!(p.marker_color, Color::GREEN);
        assert_eq!(p.reference_color, Color::BLUE);
        assert_eq!(PlotBuilder::label(&p), Some("qq"));
    }

    // -----------------------------------------------------------------------
    // CorrelationHeatmap tests
    // -----------------------------------------------------------------------

    #[test]
    fn correlation_heatmap_data_range() {
        let matrix = vec![
            vec![1.0, 0.5, -0.3],
            vec![0.5, 1.0, 0.2],
            vec![-0.3, 0.2, 1.0],
        ];
        let h = CorrelationHeatmap::new(matrix);
        let (xr, yr) = h.data_range();
        assert_eq!(xr, Some((-0.5, 2.5)));
        assert_eq!(yr, Some((-0.5, 2.5)));
    }

    #[test]
    fn correlation_heatmap_empty() {
        let h = CorrelationHeatmap::new(vec![]);
        let (xr, yr) = h.data_range();
        assert!(xr.is_none());
        assert!(yr.is_none());
        let xs = LinearScale::new(0.0, 1.0);
        let ys = LinearScale::new(0.0, 1.0);
        assert!(h.build_elements(&xs, &ys, test_area()).is_empty());
    }

    #[test]
    fn correlation_heatmap_element_count() {
        let matrix = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let h = CorrelationHeatmap::new(matrix);
        let xs = LinearScale::new(-0.5, 1.5);
        let ys = LinearScale::new(-0.5, 1.5);
        let elems = h.build_elements(&xs, &ys, test_area());
        // 4 rects + 4 text labels (show_values defaults to true)
        assert_eq!(elems.len(), 8);
    }

    #[test]
    fn correlation_heatmap_no_values() {
        let matrix = vec![vec![1.0, -1.0], vec![-1.0, 1.0]];
        let h = CorrelationHeatmap::new(matrix).show_values(false);
        let xs = LinearScale::new(-0.5, 1.5);
        let ys = LinearScale::new(-0.5, 1.5);
        let elems = h.build_elements(&xs, &ys, test_area());
        // 4 rects only
        assert_eq!(elems.len(), 4);
    }

    #[test]
    fn correlation_heatmap_builder_methods() {
        let h = CorrelationHeatmap::new(vec![vec![1.0]])
            .labels(vec!["A".into()])
            .colormap(ColorMap::viridis())
            .show_values(false)
            .label("corr");
        assert_eq!(h.labels, Some(vec!["A".into()]));
        assert!(!h.show_values);
        assert_eq!(PlotBuilder::label(&h), Some("corr"));
    }

    // -----------------------------------------------------------------------
    // OLS helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn ols_fit_perfect_line() {
        // y = 2x + 1
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let (slope, intercept) = ols_fit(&x, &y).unwrap();
        assert!((slope - 2.0).abs() < 1e-10);
        assert!((intercept - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ols_fit_returns_none_for_single_point() {
        assert!(ols_fit(&[1.0], &[1.0]).is_none());
    }

    #[test]
    fn ols_fit_returns_none_for_constant_x() {
        assert!(ols_fit(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_none());
    }

    // -----------------------------------------------------------------------
    // inv_normal_cdf tests
    // -----------------------------------------------------------------------

    #[test]
    fn inv_normal_cdf_symmetry() {
        let z1 = inv_normal_cdf(0.5);
        assert!(z1.abs() < 0.01, "median should be near 0, got {z1}");

        let z_low = inv_normal_cdf(0.025);
        let z_high = inv_normal_cdf(0.975);
        assert!((z_low + z_high).abs() < 0.01, "should be symmetric");
    }

    #[test]
    fn inv_normal_cdf_known_values() {
        // z(0.975) ≈ 1.96
        let z = inv_normal_cdf(0.975);
        assert!((z - 1.96).abs() < 0.01, "expected ~1.96, got {z}");
    }
}
