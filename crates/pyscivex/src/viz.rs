//! Python bindings for `scivex-viz` — visualization and plotting.
//!
//! All plot data is stored as plain Rust types and reconstructed into
//! `scivex_viz` objects at render time, because `Axes` contains non-Send
//! trait objects.

use pyo3::prelude::*;
use scivex_viz::{
    Axes, BarPlot, BoxPlotBuilder, Color, ConfidenceBand, ContourPlot, CorrelationHeatmap,
    ErrorBarPlot, Figure, HeatmapBuilder, Histogram, LinePlot, PieChart, PolarPlot, QQPlot,
    RegressionPlot, ResidualPlot, ScatterPlot, Theme, ViolinPlot,
};

#[allow(clippy::needless_pass_by_value)]
fn viz_err(e: scivex_viz::VizError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Stored plot variants — mirrors all supported plot types
// ---------------------------------------------------------------------------

enum StoredPlot {
    Line {
        x: Vec<f64>,
        y: Vec<f64>,
        label: Option<String>,
        color: Option<String>,
        width: Option<f64>,
    },
    Scatter {
        x: Vec<f64>,
        y: Vec<f64>,
        label: Option<String>,
        color: Option<String>,
        size: Option<f64>,
    },
    Bar {
        categories: Vec<String>,
        values: Vec<f64>,
        label: Option<String>,
        color: Option<String>,
    },
    Hist {
        data: Vec<f64>,
        bins: usize,
        label: Option<String>,
        color: Option<String>,
    },
    BoxPlot {
        datasets: Vec<Vec<f64>>,
        labels: Option<Vec<String>>,
    },
    Violin {
        datasets: Vec<Vec<f64>>,
    },
    Heatmap {
        data: Vec<Vec<f64>>,
        x_labels: Option<Vec<String>>,
        y_labels: Option<Vec<String>>,
        show_values: bool,
    },
    Pie {
        values: Vec<f64>,
        labels: Option<Vec<String>>,
    },
    Area {
        x: Vec<f64>,
        y: Vec<f64>,
        label: Option<String>,
        color: Option<String>,
    },
    ErrorBar {
        x: Vec<f64>,
        y: Vec<f64>,
        y_err_low: Vec<f64>,
        y_err_high: Vec<f64>,
        label: Option<String>,
    },
    Contour {
        data: Vec<Vec<f64>>,
        n_levels: usize,
    },
    Polar {
        categories: Vec<String>,
        values: Vec<f64>,
        label: Option<String>,
    },
    Regression {
        x: Vec<f64>,
        y: Vec<f64>,
        show_ci: bool,
    },
    Residual {
        x: Vec<f64>,
        y: Vec<f64>,
    },
    QQ {
        data: Vec<f64>,
    },
    CorrHeatmap {
        matrix: Vec<Vec<f64>>,
        labels: Option<Vec<String>>,
        show_values: bool,
    },
    ConfBand {
        x: Vec<f64>,
        y_lower: Vec<f64>,
        y_upper: Vec<f64>,
        label: Option<String>,
    },
}

fn parse_color(s: &str) -> Color {
    Color::from_hex(s).unwrap_or(Color::BLUE)
}

// ---------------------------------------------------------------------------
// PyFigure
// ---------------------------------------------------------------------------

#[pyclass(name = "Figure", unsendable)]
pub struct PyFigure {
    plots: Vec<StoredPlot>,
    fig_title: Option<String>,
    fig_x_label: Option<String>,
    fig_y_label: Option<String>,
    fig_width: f64,
    fig_height: f64,
    fig_theme: Option<String>,
    fig_grid: Option<bool>,
}

#[pymethods]
#[allow(clippy::needless_pass_by_value)]
impl PyFigure {
    #[new]
    #[pyo3(signature = (width=800.0, height=600.0))]
    fn new(width: f64, height: f64) -> Self {
        Self {
            plots: Vec::new(),
            fig_title: None,
            fig_x_label: None,
            fig_y_label: None,
            fig_width: width,
            fig_height: height,
            fig_theme: None,
            fig_grid: None,
        }
    }

    // -- Plot adders --

    #[pyo3(signature = (x, y, label=None, color=None, width=None))]
    fn line_plot(
        &mut self,
        x: Vec<f64>,
        y: Vec<f64>,
        label: Option<String>,
        color: Option<String>,
        width: Option<f64>,
    ) {
        self.plots.push(StoredPlot::Line {
            x,
            y,
            label,
            color,
            width,
        });
    }

    #[pyo3(signature = (x, y, label=None, color=None, size=None))]
    fn scatter_plot(
        &mut self,
        x: Vec<f64>,
        y: Vec<f64>,
        label: Option<String>,
        color: Option<String>,
        size: Option<f64>,
    ) {
        self.plots.push(StoredPlot::Scatter {
            x,
            y,
            label,
            color,
            size,
        });
    }

    #[pyo3(signature = (categories, values, label=None, color=None))]
    fn bar_plot(
        &mut self,
        categories: Vec<String>,
        values: Vec<f64>,
        label: Option<String>,
        color: Option<String>,
    ) {
        self.plots.push(StoredPlot::Bar {
            categories,
            values,
            label,
            color,
        });
    }

    #[pyo3(signature = (data, bins=20, label=None, color=None))]
    fn histogram(
        &mut self,
        data: Vec<f64>,
        bins: usize,
        label: Option<String>,
        color: Option<String>,
    ) {
        self.plots.push(StoredPlot::Hist {
            data,
            bins,
            label,
            color,
        });
    }

    #[pyo3(signature = (datasets, labels=None))]
    fn boxplot(&mut self, datasets: Vec<Vec<f64>>, labels: Option<Vec<String>>) {
        self.plots.push(StoredPlot::BoxPlot { datasets, labels });
    }

    fn violin_plot(&mut self, datasets: Vec<Vec<f64>>) {
        self.plots.push(StoredPlot::Violin { datasets });
    }

    #[pyo3(signature = (data, x_labels=None, y_labels=None, show_values=false))]
    fn heatmap(
        &mut self,
        data: Vec<Vec<f64>>,
        x_labels: Option<Vec<String>>,
        y_labels: Option<Vec<String>>,
        show_values: bool,
    ) {
        self.plots.push(StoredPlot::Heatmap {
            data,
            x_labels,
            y_labels,
            show_values,
        });
    }

    #[pyo3(signature = (values, labels=None))]
    fn pie_chart(&mut self, values: Vec<f64>, labels: Option<Vec<String>>) {
        self.plots.push(StoredPlot::Pie { values, labels });
    }

    #[pyo3(signature = (x, y, label=None, color=None))]
    fn area_plot(
        &mut self,
        x: Vec<f64>,
        y: Vec<f64>,
        label: Option<String>,
        color: Option<String>,
    ) {
        self.plots.push(StoredPlot::Area { x, y, label, color });
    }

    #[pyo3(signature = (x, y, y_err_low, y_err_high, label=None))]
    fn error_bar_plot(
        &mut self,
        x: Vec<f64>,
        y: Vec<f64>,
        y_err_low: Vec<f64>,
        y_err_high: Vec<f64>,
        label: Option<String>,
    ) {
        self.plots.push(StoredPlot::ErrorBar {
            x,
            y,
            y_err_low,
            y_err_high,
            label,
        });
    }

    #[pyo3(signature = (data, n_levels=10))]
    fn contour_plot(&mut self, data: Vec<Vec<f64>>, n_levels: usize) {
        self.plots.push(StoredPlot::Contour { data, n_levels });
    }

    #[pyo3(signature = (categories, values, label=None))]
    fn polar_plot(&mut self, categories: Vec<String>, values: Vec<f64>, label: Option<String>) {
        self.plots.push(StoredPlot::Polar {
            categories,
            values,
            label,
        });
    }

    #[pyo3(signature = (x, y, show_ci=true))]
    fn regression_plot(&mut self, x: Vec<f64>, y: Vec<f64>, show_ci: bool) {
        self.plots.push(StoredPlot::Regression { x, y, show_ci });
    }

    fn residual_plot(&mut self, x: Vec<f64>, y: Vec<f64>) {
        self.plots.push(StoredPlot::Residual { x, y });
    }

    fn qq_plot(&mut self, data: Vec<f64>) {
        self.plots.push(StoredPlot::QQ { data });
    }

    #[pyo3(signature = (matrix, labels=None, show_values=true))]
    fn correlation_heatmap(
        &mut self,
        matrix: Vec<Vec<f64>>,
        labels: Option<Vec<String>>,
        show_values: bool,
    ) {
        self.plots.push(StoredPlot::CorrHeatmap {
            matrix,
            labels,
            show_values,
        });
    }

    #[pyo3(signature = (x, y_lower, y_upper, label=None))]
    fn confidence_band(
        &mut self,
        x: Vec<f64>,
        y_lower: Vec<f64>,
        y_upper: Vec<f64>,
        label: Option<String>,
    ) {
        self.plots.push(StoredPlot::ConfBand {
            x,
            y_lower,
            y_upper,
            label,
        });
    }

    // -- Figure config --

    fn title(&mut self, t: String) {
        self.fig_title = Some(t);
    }

    fn x_label(&mut self, l: String) {
        self.fig_x_label = Some(l);
    }

    fn y_label(&mut self, l: String) {
        self.fig_y_label = Some(l);
    }

    fn grid(&mut self, on: bool) {
        self.fig_grid = Some(on);
    }

    fn theme(&mut self, name: String) {
        self.fig_theme = Some(name);
    }

    // -- Rendering --

    fn to_svg(&self) -> PyResult<String> {
        self.build_figure().to_svg().map_err(viz_err)
    }

    fn save_svg(&self, path: &str) -> PyResult<()> {
        self.build_figure().save_svg(path).map_err(viz_err)
    }

    fn to_terminal(&self) -> PyResult<String> {
        self.build_figure().to_terminal().map_err(viz_err)
    }

    /// Jupyter SVG rendering.
    fn _repr_svg_(&self) -> PyResult<String> {
        self.to_svg()
    }

    /// Generate a matplotlib Python script string that reproduces this figure.
    ///
    /// The returned string is valid Python that uses matplotlib to create a
    /// figure equivalent to the current PyFigure state.
    fn to_matplotlib_script(&self) -> String {
        self.generate_matplotlib_script()
    }

    fn __repr__(&self) -> String {
        format!(
            "Figure(plots={}, {}x{})",
            self.plots.len(),
            self.fig_width,
            self.fig_height
        )
    }
}

impl PyFigure {
    fn build_figure(&self) -> Figure {
        let theme = match self.fig_theme.as_deref() {
            Some("dark") => Theme::default_dark(),
            Some("light") => Theme::default_light(),
            _ => Theme::default(),
        };

        let mut axes = Axes::new().theme(theme.clone());

        if let Some(ref t) = self.fig_title {
            axes = axes.title(t);
        }
        if let Some(ref l) = self.fig_x_label {
            axes = axes.x_label(l);
        }
        if let Some(ref l) = self.fig_y_label {
            axes = axes.y_label(l);
        }
        if let Some(g) = self.fig_grid {
            axes = axes.grid(g);
        }

        for plot in &self.plots {
            match plot {
                StoredPlot::Line {
                    x,
                    y,
                    label,
                    color,
                    width,
                } => {
                    let mut p = LinePlot::new(x.clone(), y.clone());
                    if let Some(ref l) = *label {
                        p = p.label(l);
                    }
                    if let Some(ref c) = *color {
                        p = p.color(parse_color(c));
                    }
                    if let Some(w) = *width {
                        p = p.width(w);
                    }
                    axes = axes.add_plot(p);
                }
                StoredPlot::Scatter {
                    x,
                    y,
                    label,
                    color,
                    size,
                } => {
                    let mut p = ScatterPlot::new(x.clone(), y.clone());
                    if let Some(ref l) = *label {
                        p = p.label(l);
                    }
                    if let Some(ref c) = *color {
                        p = p.color(parse_color(c));
                    }
                    if let Some(s) = *size {
                        p = p.size(s);
                    }
                    axes = axes.add_plot(p);
                }
                StoredPlot::Bar {
                    categories,
                    values,
                    label,
                    color,
                } => {
                    let mut p = BarPlot::new(categories.clone(), values.clone());
                    if let Some(ref l) = *label {
                        p = p.label(l);
                    }
                    if let Some(ref c) = *color {
                        p = p.color(parse_color(c));
                    }
                    axes = axes.add_plot(p);
                }
                StoredPlot::Hist {
                    data,
                    bins,
                    label,
                    color,
                } => {
                    let mut p = Histogram::new(data.clone(), *bins);
                    if let Some(ref l) = *label {
                        p = p.label(l);
                    }
                    if let Some(ref c) = *color {
                        p = p.color(parse_color(c));
                    }
                    axes = axes.add_plot(p);
                }
                StoredPlot::BoxPlot { datasets, labels } => {
                    let mut p = BoxPlotBuilder::new(datasets.clone());
                    if let Some(ref l) = *labels {
                        p = p.labels(l.clone());
                    }
                    axes = axes.add_plot(p);
                }
                StoredPlot::Violin { datasets } => {
                    let p = ViolinPlot::new(datasets.clone());
                    axes = axes.add_plot(p);
                }
                StoredPlot::Heatmap {
                    data,
                    x_labels,
                    y_labels,
                    show_values,
                } => {
                    let mut p = HeatmapBuilder::new(data.clone());
                    if let Some(ref xl) = *x_labels {
                        p = p.x_labels(xl.clone());
                    }
                    if let Some(ref yl) = *y_labels {
                        p = p.y_labels(yl.clone());
                    }
                    p = p.show_values(*show_values);
                    axes = axes.add_plot(p);
                }
                StoredPlot::Pie { values, labels } => {
                    let mut p = PieChart::new(values.clone());
                    if let Some(ref l) = *labels {
                        p = p.labels(l.clone());
                    }
                    axes = axes.add_plot(p);
                }
                StoredPlot::Area { x, y, label, color } => {
                    let mut p = scivex_viz::AreaPlot::new(x.clone(), y.clone());
                    if let Some(ref l) = *label {
                        p = p.label(l);
                    }
                    if let Some(ref c) = *color {
                        p = p.color(parse_color(c));
                    }
                    axes = axes.add_plot(p);
                }
                StoredPlot::ErrorBar {
                    x,
                    y,
                    y_err_low,
                    y_err_high,
                    label,
                } => {
                    let mut p = ErrorBarPlot::new(
                        x.clone(),
                        y.clone(),
                        y_err_low.clone(),
                        y_err_high.clone(),
                    );
                    if let Some(ref l) = *label {
                        p = p.label(l);
                    }
                    axes = axes.add_plot(p);
                }
                StoredPlot::Contour { data, n_levels } => {
                    let p = ContourPlot::new(data.clone(), *n_levels);
                    axes = axes.add_plot(p);
                }
                StoredPlot::Polar {
                    categories,
                    values,
                    label,
                } => {
                    let mut p = PolarPlot::new(categories.clone(), values.clone());
                    if let Some(ref l) = *label {
                        p = p.label(l);
                    }
                    axes = axes.add_plot(p);
                }
                StoredPlot::Regression { x, y, show_ci } => {
                    let p = RegressionPlot::new(x.clone(), y.clone()).show_ci(*show_ci);
                    axes = axes.add_plot(p);
                }
                StoredPlot::Residual { x, y } => {
                    let p = ResidualPlot::new(x.clone(), y.clone());
                    axes = axes.add_plot(p);
                }
                StoredPlot::QQ { data } => {
                    let p = QQPlot::new(data.clone());
                    axes = axes.add_plot(p);
                }
                StoredPlot::CorrHeatmap {
                    matrix,
                    labels,
                    show_values,
                } => {
                    let mut p = CorrelationHeatmap::new(matrix.clone());
                    if let Some(ref l) = *labels {
                        p = p.labels(l.clone());
                    }
                    p = p.show_values(*show_values);
                    axes = axes.add_plot(p);
                }
                StoredPlot::ConfBand {
                    x,
                    y_lower,
                    y_upper,
                    label,
                } => {
                    let mut p = ConfidenceBand::new(x.clone(), y_lower.clone(), y_upper.clone());
                    if let Some(ref l) = *label {
                        p = p.label(l);
                    }
                    axes = axes.add_plot(p);
                }
            }
        }

        Figure::new()
            .size(self.fig_width, self.fig_height)
            .theme(theme)
            .plot(axes)
    }
}

// ===========================================================================
// PyChart — declarative grammar-of-graphics API
// ===========================================================================

/// Declarative chart builder inspired by Vega-Lite.
///
/// Build a chart by chaining `.mark_point()`, `.encode_x()`, `.encode_y()`,
/// etc., then render with `.to_svg()` or `.save_svg()`.
#[pyclass(name = "Chart", unsendable)]
pub struct PyChart {
    inner: scivex_viz::Chart,
}

#[pymethods]
impl PyChart {
    /// Create a chart from column-oriented data.
    ///
    /// `columns` is a list of `(name, values)` tuples.
    #[new]
    fn new(columns: Vec<(String, Vec<f64>)>) -> Self {
        let cols: Vec<(&str, Vec<f64>)> = columns
            .iter()
            .map(|(n, v)| (n.as_str(), v.clone()))
            .collect();
        Self {
            inner: scivex_viz::Chart::from_columns(cols),
        }
    }

    /// Set the mark type to point (scatter).
    fn mark_point(&mut self) {
        self.inner = std::mem::replace(&mut self.inner, scivex_viz::Chart::new(vec![]))
            .mark(scivex_viz::Mark::Point);
    }

    /// Set the mark type to line.
    fn mark_line(&mut self) {
        self.inner = std::mem::replace(&mut self.inner, scivex_viz::Chart::new(vec![]))
            .mark(scivex_viz::Mark::Line);
    }

    /// Set the mark type to bar.
    fn mark_bar(&mut self) {
        self.inner = std::mem::replace(&mut self.inner, scivex_viz::Chart::new(vec![]))
            .mark(scivex_viz::Mark::Bar);
    }

    /// Encode the x channel with the given field name and scale type.
    ///
    /// `scale` can be `"linear"`, `"log"`, `"ordinal"`, or `"temporal"`.
    #[pyo3(signature = (field, scale="linear"))]
    fn encode_x(&mut self, field: &str, scale: &str) {
        let s = parse_scale_type(scale);
        self.inner =
            std::mem::replace(&mut self.inner, scivex_viz::Chart::new(vec![])).encode_x(field, s);
    }

    /// Encode the y channel with the given field name and scale type.
    #[pyo3(signature = (field, scale="linear"))]
    fn encode_y(&mut self, field: &str, scale: &str) {
        let s = parse_scale_type(scale);
        self.inner =
            std::mem::replace(&mut self.inner, scivex_viz::Chart::new(vec![])).encode_y(field, s);
    }

    /// Encode the color channel with the given field name and scale type.
    #[pyo3(signature = (field, scale="ordinal"))]
    fn encode_color(&mut self, field: &str, scale: &str) {
        let s = parse_scale_type(scale);
        self.inner = std::mem::replace(&mut self.inner, scivex_viz::Chart::new(vec![]))
            .encode_color(field, s);
    }

    /// Encode the size channel with the given field name and scale type.
    #[pyo3(signature = (field, scale="linear"))]
    fn encode_size(&mut self, field: &str, scale: &str) {
        let s = parse_scale_type(scale);
        self.inner = std::mem::replace(&mut self.inner, scivex_viz::Chart::new(vec![]))
            .encode_size(field, s);
    }

    /// Set the chart title.
    fn title(&mut self, title: &str) {
        self.inner =
            std::mem::replace(&mut self.inner, scivex_viz::Chart::new(vec![])).title(title);
    }

    /// Set the chart width in pixels.
    fn width(&mut self, w: usize) {
        self.inner = std::mem::replace(&mut self.inner, scivex_viz::Chart::new(vec![])).width(w);
    }

    /// Set the chart height in pixels.
    fn height(&mut self, h: usize) {
        self.inner = std::mem::replace(&mut self.inner, scivex_viz::Chart::new(vec![])).height(h);
    }

    /// Render the chart to an SVG string.
    fn to_svg(&self) -> PyResult<String> {
        self.inner.to_svg().map_err(viz_err)
    }

    /// Render the chart to an SVG file.
    fn save_svg(&self, path: &str) -> PyResult<()> {
        let svg = self.inner.to_svg().map_err(viz_err)?;
        std::fs::write(path, svg).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Jupyter SVG rendering.
    fn _repr_svg_(&self) -> PyResult<String> {
        self.to_svg()
    }

    fn __repr__(&self) -> String {
        "Chart(...)".to_string()
    }
}

fn parse_scale_type(s: &str) -> scivex_viz::ScaleType {
    match s.to_lowercase().as_str() {
        "log" => scivex_viz::ScaleType::Log,
        "ordinal" => scivex_viz::ScaleType::Ordinal,
        "temporal" => scivex_viz::ScaleType::Temporal,
        _ => scivex_viz::ScaleType::Linear,
    }
}

// ===========================================================================
// PyAnimation — animated GIF sequences
// ===========================================================================

/// Animation builder for creating GIF animations from plot frames.
///
/// Add frames as SVG strings (from Figure/Chart renders) with a per-frame
/// delay, then encode to GIF.
#[pyclass(name = "Animation", unsendable)]
pub struct PyAnimation {
    inner: scivex_viz::Animation,
}

#[pymethods]
impl PyAnimation {
    /// Create a new animation with the given pixel dimensions.
    #[new]
    fn new(width: u32, height: u32) -> Self {
        Self {
            inner: scivex_viz::Animation::new(width, height),
        }
    }

    /// Add a frame consisting of drawing elements with the given delay in milliseconds.
    ///
    /// `elements_json` is currently unused — frames are added as blank canvases.
    /// For practical use, build frames via the bitmap backend.
    fn add_frame(&mut self, delay_ms: u16) {
        self.inner = std::mem::replace(&mut self.inner, scivex_viz::Animation::new(1, 1))
            .add_frame(vec![], delay_ms);
    }

    /// Return the number of frames added so far.
    fn frame_count(&self) -> usize {
        self.inner.frame_count()
    }

    /// Set the loop count (0 = infinite loop).
    fn set_loop_count(&mut self, n: u16) {
        self.inner =
            std::mem::replace(&mut self.inner, scivex_viz::Animation::new(1, 1)).loop_count(n);
    }

    /// Encode the animation as GIF bytes.
    fn to_gif(&self) -> PyResult<Vec<u8>> {
        self.inner.to_gif().map_err(viz_err)
    }

    /// Write the animation to a GIF file.
    fn save_gif(&self, path: &str) -> PyResult<()> {
        self.inner.save_gif(path).map_err(viz_err)
    }

    fn __repr__(&self) -> String {
        format!("Animation(frames={})", self.inner.frame_count())
    }
}

// ===========================================================================
// to_matplotlib_script — generate matplotlib Python code from a PyFigure
// ===========================================================================

impl PyFigure {
    /// Generate a matplotlib Python script string that reproduces this figure.
    ///
    /// The returned string is valid Python that uses matplotlib to create a
    /// figure equivalent to the current PyFigure state.
    pub fn generate_matplotlib_script(&self) -> String {
        let mut lines = Vec::new();
        lines.push("import matplotlib.pyplot as plt".to_string());
        lines.push(String::new());
        lines.push(format!(
            "fig, ax = plt.subplots(figsize=({}, {}))",
            self.fig_width / 100.0,
            self.fig_height / 100.0,
        ));

        for plot in &self.plots {
            match plot {
                StoredPlot::Line {
                    x,
                    y,
                    label,
                    color,
                    width,
                } => {
                    let mut args = format!("ax.plot({:?}, {:?}", x, y);
                    if let Some(ref l) = *label {
                        args.push_str(&format!(", label={l:?}"));
                    }
                    if let Some(ref c) = *color {
                        args.push_str(&format!(", color={c:?}"));
                    }
                    if let Some(w) = *width {
                        args.push_str(&format!(", linewidth={w}"));
                    }
                    args.push(')');
                    lines.push(args);
                }
                StoredPlot::Scatter {
                    x,
                    y,
                    label,
                    color,
                    size,
                } => {
                    let mut args = format!("ax.scatter({:?}, {:?}", x, y);
                    if let Some(ref l) = *label {
                        args.push_str(&format!(", label={l:?}"));
                    }
                    if let Some(ref c) = *color {
                        args.push_str(&format!(", color={c:?}"));
                    }
                    if let Some(s) = *size {
                        args.push_str(&format!(", s={s}"));
                    }
                    args.push(')');
                    lines.push(args);
                }
                StoredPlot::Bar {
                    categories,
                    values,
                    label,
                    color,
                } => {
                    let mut args = format!("ax.bar({:?}, {:?}", categories, values);
                    if let Some(ref l) = *label {
                        args.push_str(&format!(", label={l:?}"));
                    }
                    if let Some(ref c) = *color {
                        args.push_str(&format!(", color={c:?}"));
                    }
                    args.push(')');
                    lines.push(args);
                }
                StoredPlot::Hist {
                    data,
                    bins,
                    label,
                    color,
                } => {
                    let mut args = format!("ax.hist({:?}, bins={bins}", data);
                    if let Some(ref l) = *label {
                        args.push_str(&format!(", label={l:?}"));
                    }
                    if let Some(ref c) = *color {
                        args.push_str(&format!(", color={c:?}"));
                    }
                    args.push(')');
                    lines.push(args);
                }
                StoredPlot::BoxPlot { datasets, .. } => {
                    lines.push(format!("ax.boxplot({:?})", datasets));
                }
                StoredPlot::Violin { datasets } => {
                    lines.push(format!("ax.violinplot({:?})", datasets));
                }
                StoredPlot::Heatmap { data, .. } => {
                    lines.push(format!("ax.imshow({:?}, aspect='auto')", data));
                }
                StoredPlot::Pie { values, labels } => {
                    if let Some(ref l) = *labels {
                        lines.push(format!("ax.pie({:?}, labels={:?})", values, l));
                    } else {
                        lines.push(format!("ax.pie({:?})", values));
                    }
                }
                StoredPlot::Area { x, y, label, color } => {
                    let mut args = format!("ax.fill_between({:?}, {:?}", x, y);
                    if let Some(ref l) = *label {
                        args.push_str(&format!(", label={l:?}"));
                    }
                    if let Some(ref c) = *color {
                        args.push_str(&format!(", color={c:?}"));
                    }
                    args.push(')');
                    lines.push(args);
                }
                StoredPlot::ErrorBar {
                    x,
                    y,
                    y_err_low,
                    y_err_high,
                    label,
                } => {
                    let mut args = format!(
                        "ax.errorbar({:?}, {:?}, yerr=[{:?}, {:?}]",
                        x, y, y_err_low, y_err_high,
                    );
                    if let Some(ref l) = *label {
                        args.push_str(&format!(", label={l:?}"));
                    }
                    args.push(')');
                    lines.push(args);
                }
                StoredPlot::Contour { data, n_levels } => {
                    lines.push(format!("ax.contour({:?}, levels={n_levels})", data));
                }
                StoredPlot::Polar {
                    categories,
                    values,
                    label,
                } => {
                    let _ = (categories, values, label);
                    lines.push("# Polar plot (use plt.subplot(projection='polar'))".to_string());
                }
                StoredPlot::Regression { x, y, .. } => {
                    lines.push(format!("ax.scatter({:?}, {:?})", x, y));
                    lines.push("# Add regression line manually".to_string());
                }
                StoredPlot::Residual { x, y } => {
                    lines.push(format!("ax.scatter({:?}, {:?})", x, y));
                    lines.push("ax.axhline(0, color='gray', linestyle='--')".to_string());
                }
                StoredPlot::QQ { data } => {
                    lines.push("from scipy import stats".to_string());
                    lines.push(format!("stats.probplot({:?}, plot=ax)", data));
                }
                StoredPlot::CorrHeatmap { matrix, .. } => {
                    lines.push(format!(
                        "ax.imshow({:?}, cmap='coolwarm', aspect='auto')",
                        matrix
                    ));
                }
                StoredPlot::ConfBand {
                    x,
                    y_lower,
                    y_upper,
                    label,
                } => {
                    let mut args = format!(
                        "ax.fill_between({:?}, {:?}, {:?}, alpha=0.3",
                        x, y_lower, y_upper,
                    );
                    if let Some(ref l) = *label {
                        args.push_str(&format!(", label={l:?}"));
                    }
                    args.push(')');
                    lines.push(args);
                }
            }
        }

        if let Some(ref t) = self.fig_title {
            lines.push(format!("ax.set_title({t:?})"));
        }
        if let Some(ref l) = self.fig_x_label {
            lines.push(format!("ax.set_xlabel({l:?})"));
        }
        if let Some(ref l) = self.fig_y_label {
            lines.push(format!("ax.set_ylabel({l:?})"));
        }
        if let Some(g) = self.fig_grid {
            lines.push(format!("ax.grid({g})"));
        }

        // Add legend if any plot has a label
        let has_label = self.plots.iter().any(|p| {
            matches!(
                p,
                StoredPlot::Line { label: Some(_), .. }
                    | StoredPlot::Scatter { label: Some(_), .. }
                    | StoredPlot::Bar { label: Some(_), .. }
                    | StoredPlot::Hist { label: Some(_), .. }
                    | StoredPlot::Area { label: Some(_), .. }
                    | StoredPlot::ErrorBar { label: Some(_), .. }
                    | StoredPlot::ConfBand { label: Some(_), .. }
            )
        });
        if has_label {
            lines.push("ax.legend()".to_string());
        }

        lines.push("plt.tight_layout()".to_string());
        lines.push("plt.show()".to_string());

        lines.join("\n")
    }
}

// ===========================================================================
// Submodule with stat plot convenience functions
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (x, y, show_ci=true))]
fn regplot(x: Vec<f64>, y: Vec<f64>, show_ci: bool) -> PyResult<PyFigure> {
    let mut fig = PyFigure::new(800.0, 600.0);
    fig.regression_plot(x, y, show_ci);
    Ok(fig)
}

#[pyfunction]
fn residplot(x: Vec<f64>, y: Vec<f64>) -> PyResult<PyFigure> {
    let mut fig = PyFigure::new(800.0, 600.0);
    fig.residual_plot(x, y);
    Ok(fig)
}

#[pyfunction]
fn qqplot(data: Vec<f64>) -> PyResult<PyFigure> {
    let mut fig = PyFigure::new(800.0, 600.0);
    fig.qq_plot(data);
    Ok(fig)
}

#[pyfunction]
#[pyo3(signature = (matrix, labels=None, show_values=true))]
fn corrplot(
    matrix: Vec<Vec<f64>>,
    labels: Option<Vec<String>>,
    show_values: bool,
) -> PyResult<PyFigure> {
    let mut fig = PyFigure::new(800.0, 600.0);
    fig.correlation_heatmap(matrix, labels, show_values);
    Ok(fig)
}

// ===========================================================================
// Submodule registration
// ===========================================================================

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let viz = PyModule::new(py, "viz")?;

    viz.add_class::<PyFigure>()?;
    viz.add_class::<PyChart>()?;
    viz.add_class::<PyAnimation>()?;

    // Convenience stat plot functions
    viz.add_function(wrap_pyfunction!(regplot, &viz)?)?;
    viz.add_function(wrap_pyfunction!(residplot, &viz)?)?;
    viz.add_function(wrap_pyfunction!(qqplot, &viz)?)?;
    viz.add_function(wrap_pyfunction!(corrplot, &viz)?)?;

    parent.add_submodule(&viz)?;
    Ok(())
}
