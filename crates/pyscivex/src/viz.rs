//! Python bindings for [`scivex_viz`] visualization.

use pyo3::prelude::*;
use scivex_viz::{Axes, Figure, LinePlot, ScatterPlot};

/// Convert a `VizError` into a Python `ValueError`.
#[allow(clippy::needless_pass_by_value)]
fn viz_err(e: scivex_viz::VizError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Internal representation of a stored plot.
enum StoredPlot {
    Line {
        x: Vec<f64>,
        y: Vec<f64>,
        label: Option<String>,
    },
    Scatter {
        x: Vec<f64>,
        y: Vec<f64>,
        label: Option<String>,
    },
}

/// A figure that accumulates plots and can be rendered to SVG.
///
/// Because `scivex_viz::Axes` contains non-Send trait objects, we store
/// the raw plot data and reconstruct `Axes` at render time.
#[pyclass(name = "Figure", unsendable)]
pub struct PyFigure {
    plots: Vec<StoredPlot>,
    fig_title: Option<String>,
    fig_x_label: Option<String>,
    fig_y_label: Option<String>,
}

#[pymethods]
#[allow(clippy::needless_pass_by_value)]
impl PyFigure {
    /// Create a new empty figure.
    #[new]
    fn new() -> Self {
        Self {
            plots: Vec::new(),
            fig_title: None,
            fig_x_label: None,
            fig_y_label: None,
        }
    }

    /// Add a line plot to the figure.
    #[pyo3(signature = (x, y, label=None))]
    fn line_plot(&mut self, x: Vec<f64>, y: Vec<f64>, label: Option<String>) {
        self.plots.push(StoredPlot::Line { x, y, label });
    }

    /// Add a scatter plot to the figure.
    #[pyo3(signature = (x, y, label=None))]
    fn scatter_plot(&mut self, x: Vec<f64>, y: Vec<f64>, label: Option<String>) {
        self.plots.push(StoredPlot::Scatter { x, y, label });
    }

    /// Set the title for the figure.
    fn title(&mut self, t: String) {
        self.fig_title = Some(t);
    }

    /// Set the x-axis label for the figure.
    fn x_label(&mut self, l: String) {
        self.fig_x_label = Some(l);
    }

    /// Set the y-axis label for the figure.
    fn y_label(&mut self, l: String) {
        self.fig_y_label = Some(l);
    }

    /// Render the figure to an SVG string.
    fn to_svg(&self) -> PyResult<String> {
        self.build_figure().to_svg().map_err(viz_err)
    }

    /// Save the figure as an SVG file.
    fn save_svg(&self, path: &str) -> PyResult<()> {
        self.build_figure().save_svg(path).map_err(viz_err)
    }

    fn __repr__(&self) -> String {
        format!("Figure(plots={})", self.plots.len())
    }
}

impl PyFigure {
    /// Reconstruct a `scivex_viz::Figure` from the stored plot data.
    fn build_figure(&self) -> Figure {
        let mut axes = Axes::new();

        if let Some(ref t) = self.fig_title {
            axes = axes.title(t);
        }
        if let Some(ref l) = self.fig_x_label {
            axes = axes.x_label(l);
        }
        if let Some(ref l) = self.fig_y_label {
            axes = axes.y_label(l);
        }

        for plot in &self.plots {
            match plot {
                StoredPlot::Line { x, y, label } => {
                    let mut lp = LinePlot::new(x.clone(), y.clone());
                    if let Some(ref l) = *label {
                        lp = lp.label(l);
                    }
                    axes = axes.add_plot(lp);
                }
                StoredPlot::Scatter { x, y, label } => {
                    let mut sp = ScatterPlot::new(x.clone(), y.clone());
                    if let Some(ref l) = *label {
                        sp = sp.label(l);
                    }
                    axes = axes.add_plot(sp);
                }
            }
        }

        Figure::new().plot(axes)
    }
}
