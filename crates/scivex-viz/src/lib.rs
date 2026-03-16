//! `scivex-viz` — Visualization, plotting, and chart rendering.
//!
//! A from-scratch visualization library built on [`scivex_core`]. Produces SVG
//! output or terminal braille art with zero external dependencies for rendering.
//!
//! # Architecture
//!
//! Three-layer design:
//! - **User API**: [`Figure`], [`Axes`], plot builders ([`LinePlot`], [`ScatterPlot`], etc.)
//! - **Element Layer**: Backend-agnostic drawing primitives ([`Element`])
//! - **Renderer Layer**: [`SvgBackend`], [`TerminalBackend`], [`HtmlBackend`], [`BitmapBackend`]
//!
//! # Quick Start
//!
//! ```rust
//! use scivex_viz::prelude::*;
//!
//! let fig = Figure::new().plot(
//!     Axes::new()
//!         .title("Hello")
//!         .x_label("x")
//!         .y_label("y")
//!         .add_plot(LinePlot::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.5]))
//! );
//! let svg = fig.to_svg().unwrap();
//! ```

/// Annotations such as reference lines, text labels, and legends.
pub mod annotation;
/// Area / stacked area chart builder.
pub mod area;
/// Axes: the container for plots, labels, ticks, and grid lines.
pub mod axes;
/// Rendering backends (SVG, terminal braille, HTML, bitmap).
pub mod backend;
/// Color types, colormaps, and palettes.
pub mod color;
/// Contour plot builder (marching squares).
pub mod contour;
/// Backend-agnostic drawing primitives.
pub mod element;
/// Error types for the visualization crate.
pub mod error;
/// Error bar and confidence band builders.
pub mod error_bar;
/// Top-level figure container and output helpers.
pub mod figure;
/// Heatmap visualization builder.
pub mod heatmap;
/// Layout primitives: padding, bounding rectangles, and grid cells.
pub mod layout;
/// Pie / donut chart builder.
pub mod pie;
/// Plot builders: line, scatter, bar, and histogram.
pub mod plot;
/// Polar / radar chart builder.
pub mod polar;
/// Scale types for mapping data values to normalized coordinates.
pub mod scale;
/// Statistical visualization: regression, residual, Q-Q, correlation heatmap.
pub mod stat_plots;
/// Statistical plot builders (box plots).
pub mod statistical;
/// Styling primitives: strokes, fills, markers, fonts, and themes.
pub mod style;
/// Violin plot builder.
pub mod violin;

pub use annotation::{Annotation, LegendPosition};
pub use area::AreaPlot;
pub use axes::Axes;
pub use backend::{BitmapBackend, HtmlBackend, Renderer, SvgBackend, TerminalBackend};
pub use color::{Color, ColorMap};
pub use contour::ContourPlot;
pub use element::{Element, TextAnchor};
pub use error::{Result, VizError};
pub use error_bar::{ConfidenceBand, ErrorBarPlot};
pub use figure::Figure;
pub use heatmap::HeatmapBuilder;
pub use layout::{Layout, Padding, Rect};
pub use pie::PieChart;
pub use plot::{AxisRange, BarPlot, Histogram, LinePlot, PlotBuilder, ScatterPlot};
pub use polar::PolarPlot;
pub use scale::{LinearScale, LogScale, Scale};
pub use stat_plots::{CorrelationHeatmap, QQPlot, RegressionPlot, ResidualPlot};
pub use statistical::BoxPlotBuilder;
pub use style::{Fill, Font, Marker, MarkerShape, Stroke, Theme};
pub use violin::ViolinPlot;

/// Items intended for glob-import: `use scivex_viz::prelude::*;`
pub mod prelude {
    pub use crate::annotation::{Annotation, LegendPosition};
    pub use crate::area::AreaPlot;
    pub use crate::axes::Axes;
    pub use crate::backend::{BitmapBackend, HtmlBackend, Renderer, SvgBackend, TerminalBackend};
    pub use crate::color::{Color, ColorMap};
    pub use crate::contour::ContourPlot;
    pub use crate::element::{Element, TextAnchor};
    pub use crate::error::{Result, VizError};
    pub use crate::error_bar::{ConfidenceBand, ErrorBarPlot};
    pub use crate::figure::Figure;
    pub use crate::heatmap::HeatmapBuilder;
    pub use crate::layout::{Layout, Rect};
    pub use crate::pie::PieChart;
    pub use crate::plot::{AxisRange, BarPlot, Histogram, LinePlot, PlotBuilder, ScatterPlot};
    pub use crate::polar::PolarPlot;
    pub use crate::scale::{LinearScale, LogScale, Scale};
    pub use crate::stat_plots::{CorrelationHeatmap, QQPlot, RegressionPlot, ResidualPlot};
    pub use crate::statistical::BoxPlotBuilder;
    pub use crate::style::{Fill, Font, Marker, MarkerShape, Stroke, Theme};
    pub use crate::violin::ViolinPlot;
}
