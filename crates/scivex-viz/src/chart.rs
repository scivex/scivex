//! Declarative grammar-of-graphics API inspired by Vega-Lite.
//!
//! Provides a [`Chart`] builder that lets you describe *what* to visualize rather
//! than *how* to render it.  The API maps data fields to visual channels (x, y,
//! color, size) and renders to SVG or HTML with a single method call.
//!
//! # Examples
//!
//! ```
//! # use scivex_viz::chart::{Chart, Mark, ScaleType};
//! let chart = Chart::from_columns(vec![
//!         ("age", vec![25.0, 30.0, 35.0, 40.0]),
//!         ("salary", vec![50_000.0, 60_000.0, 55_000.0, 70_000.0]),
//!     ])
//!     .mark(Mark::Point)
//!     .encode_x("age", ScaleType::Linear)
//!     .encode_y("salary", ScaleType::Linear)
//!     .title("Salary vs Age")
//!     .width(600)
//!     .height(400);
//! let svg = chart.to_svg().unwrap();
//! assert!(svg.contains("<svg"));
//! ```

use std::fmt::Write;

use crate::color::Color;
use crate::error::{Result, VizError};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The visual mark type drawn for each data point.
///
/// # Examples
///
/// ```
/// # use scivex_viz::chart::Mark;
/// let m = Mark::Point;
/// assert!(matches!(m, Mark::Point));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    /// Scatter-style filled circles.
    Point,
    /// Connected line segments.
    Line,
    /// Vertical bars.
    Bar,
    /// Filled area under a line.
    Area,
    /// Circles (alias for `Point` with larger default radius).
    Circle,
    /// Horizontal or vertical reference rules.
    Rule,
    /// Text labels placed at data positions.
    Text,
}

/// The type of scale used to map a data field to a visual channel.
///
/// # Examples
///
/// ```
/// # use scivex_viz::chart::ScaleType;
/// let s = ScaleType::Linear;
/// assert!(matches!(s, ScaleType::Linear));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleType {
    /// Linear interpolation between min and max.
    Linear,
    /// Logarithmic (base-10) interpolation.
    Log,
    /// Discrete / categorical scale.
    Ordinal,
    /// Date/time scale (treated as linear on epoch seconds).
    Temporal,
}

/// A single encoding channel binding a data field to a visual property.
///
/// # Examples
///
/// ```
/// # use scivex_viz::chart::{Encoding, ScaleType};
/// let enc = Encoding { field: "x".to_string(), scale: ScaleType::Linear };
/// assert_eq!(enc.field, "x");
/// ```
#[derive(Debug, Clone)]
pub struct Encoding {
    /// Column / field name in the data.
    pub field: String,
    /// Scale type for this channel.
    pub scale: ScaleType,
}

/// A declarative chart specification.
///
/// Build a `Chart` by chaining `.mark()`, `.encode_x()`, `.encode_y()`, etc.,
/// then call `.to_svg()` or `.to_html()` to render.
///
/// # Examples
///
/// ```
/// # use scivex_viz::chart::{Chart, Mark, ScaleType};
/// let chart = Chart::from_columns(vec![("x", vec![1.0, 2.0]), ("y", vec![3.0, 4.0])])
///     .mark(Mark::Line)
///     .encode_x("x", ScaleType::Linear)
///     .encode_y("y", ScaleType::Linear);
/// let svg = chart.to_svg().unwrap();
/// assert!(svg.contains("<svg"));
/// ```
#[derive(Debug, Clone)]
pub struct Chart {
    /// Row-oriented data: each inner `Vec` is a row of `(column_name, value)`.
    data: Vec<Vec<(String, f64)>>,
    mark: Mark,
    x: Option<Encoding>,
    y: Option<Encoding>,
    color: Option<Encoding>,
    size: Option<Encoding>,
    title: Option<String>,
    width: usize,
    height: usize,
}

// Margins for axes / title.
const MARGIN_LEFT: f64 = 60.0;
const MARGIN_RIGHT: f64 = 20.0;
const MARGIN_TOP: f64 = 40.0;
const MARGIN_BOTTOM: f64 = 50.0;

/// Default categorical color palette (Tableau 10).
const PALETTE: [Color; 10] = [
    Color::rgb(31, 119, 180),
    Color::rgb(255, 127, 14),
    Color::rgb(44, 160, 44),
    Color::rgb(214, 39, 40),
    Color::rgb(148, 103, 189),
    Color::rgb(140, 86, 75),
    Color::rgb(227, 119, 194),
    Color::rgb(127, 127, 127),
    Color::rgb(188, 189, 34),
    Color::rgb(23, 190, 207),
];

impl Chart {
    // -------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------

    /// Create a chart from row-oriented data.
    ///
    /// Each element of `data` is a row represented as a `Vec` of
    /// `(column_name, value)` pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_viz::chart::Chart;
    /// let rows = vec![
    ///     vec![("x".to_string(), 1.0), ("y".to_string(), 2.0)],
    ///     vec![("x".to_string(), 3.0), ("y".to_string(), 4.0)],
    /// ];
    /// let chart = Chart::new(rows);
    /// ```
    #[must_use]
    pub fn new(data: Vec<Vec<(String, f64)>>) -> Self {
        Self {
            data,
            mark: Mark::Point,
            x: None,
            y: None,
            color: None,
            size: None,
            title: None,
            width: 600,
            height: 400,
        }
    }

    /// Create a chart from column-oriented data.
    ///
    /// Each element of `columns` is `(column_name, values)`.  All value
    /// vectors must have the same length.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_viz::chart::Chart;
    /// let chart = Chart::from_columns(vec![
    ///     ("x", vec![1.0, 2.0, 3.0]),
    ///     ("y", vec![4.0, 5.0, 6.0]),
    /// ]);
    /// ```
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_columns(columns: Vec<(&str, Vec<f64>)>) -> Self {
        let n_rows = columns.first().map_or(0, |(_, v)| v.len());
        let mut data = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let mut row = Vec::with_capacity(columns.len());
            for (name, vals) in &columns {
                if i < vals.len() {
                    row.push(((*name).to_string(), vals[i]));
                }
            }
            data.push(row);
        }
        Self::new(data)
    }

    // -------------------------------------------------------------------
    // Builder methods
    // -------------------------------------------------------------------

    /// Set the mark type.
    #[must_use]
    pub fn mark(mut self, mark: Mark) -> Self {
        self.mark = mark;
        self
    }

    /// Encode the x channel.
    #[must_use]
    pub fn encode_x(mut self, field: &str, scale: ScaleType) -> Self {
        self.x = Some(Encoding {
            field: field.to_string(),
            scale,
        });
        self
    }

    /// Encode the y channel.
    #[must_use]
    pub fn encode_y(mut self, field: &str, scale: ScaleType) -> Self {
        self.y = Some(Encoding {
            field: field.to_string(),
            scale,
        });
        self
    }

    /// Encode the color channel (maps a field to categorical colors).
    #[must_use]
    pub fn encode_color(mut self, field: &str, scale: ScaleType) -> Self {
        self.color = Some(Encoding {
            field: field.to_string(),
            scale,
        });
        self
    }

    /// Encode the size channel (maps a field to marker size).
    #[must_use]
    pub fn encode_size(mut self, field: &str, scale: ScaleType) -> Self {
        self.size = Some(Encoding {
            field: field.to_string(),
            scale,
        });
        self
    }

    /// Set the chart title.
    #[must_use]
    pub fn title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }

    /// Set the chart width in pixels.
    #[must_use]
    pub fn width(mut self, w: usize) -> Self {
        self.width = w;
        self
    }

    /// Set the chart height in pixels.
    #[must_use]
    pub fn height(mut self, h: usize) -> Self {
        self.height = h;
        self
    }

    // -------------------------------------------------------------------
    // Rendering
    // -------------------------------------------------------------------

    /// Render the chart to an SVG string.
    ///
    /// # Errors
    ///
    /// Returns [`VizError::EmptyData`] if the data or required encodings are
    /// missing, or [`VizError::InvalidParameter`] for invalid field names.
    pub fn to_svg(&self) -> Result<String> {
        self.validate()?;

        let w = self.width as f64;
        let h = self.height as f64;
        let plot_x = MARGIN_LEFT;
        let plot_y = MARGIN_TOP;
        let plot_w = w - MARGIN_LEFT - MARGIN_RIGHT;
        let plot_h = h - MARGIN_TOP - MARGIN_BOTTOM;

        let x_enc = self.x.as_ref().unwrap();
        let y_enc = self.y.as_ref().unwrap();

        let x_vals = self.extract_field(&x_enc.field)?;
        let y_vals = self.extract_field(&y_enc.field)?;

        let (x_min, x_max) = min_max_padded(&x_vals, x_enc.scale, self.mark);
        let (y_min, y_max) = min_max_padded(&y_vals, y_enc.scale, self.mark);

        let color_vals = self
            .color
            .as_ref()
            .map(|enc| self.extract_field(&enc.field))
            .transpose()?;
        let size_vals = self
            .size
            .as_ref()
            .map(|enc| self.extract_field(&enc.field))
            .transpose()?;

        let mut svg = String::with_capacity(4096);

        // Header
        writeln!(
            svg,
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">"#,
        )
        .expect("write to String is infallible");

        // Background
        writeln!(
            svg,
            r#"  <rect x="0" y="0" width="{w}" height="{h}" fill="white"/>"#,
        )
        .expect("write to String is infallible");

        // Plot area background
        writeln!(
            svg,
            r##"  <rect x="{plot_x:.2}" y="{plot_y:.2}" width="{plot_w:.2}" height="{plot_h:.2}" fill="#f9f9f9" stroke="#cccccc" stroke-width="0.5"/>"##,
        )
        .expect("write to String is infallible");

        // Grid lines
        Self::render_grid(
            &mut svg, plot_x, plot_y, plot_w, plot_h, y_min, y_max, x_min, x_max,
        );

        // Axes
        Self::render_axes(
            &mut svg, plot_x, plot_y, plot_w, plot_h, x_min, x_max, y_min, y_max, x_enc, y_enc,
        );

        // Data marks
        self.render_marks(
            &mut svg,
            &x_vals,
            &y_vals,
            color_vals.as_deref(),
            size_vals.as_deref(),
            plot_x,
            plot_y,
            plot_w,
            plot_h,
            x_min,
            x_max,
            y_min,
            y_max,
        );

        // Title
        if let Some(ref title) = self.title {
            let tx = w / 2.0;
            let ty = MARGIN_TOP / 2.0 + 4.0;
            writeln!(
                svg,
                r#"  <text x="{tx:.2}" y="{ty:.2}" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold" fill="black">{}</text>"#,
                escape_xml(title),
            )
            .expect("write to String is infallible");
        }

        // Legend (when color encoding is present)
        if let Some(ref color_enc) = self.color {
            Self::render_legend(&mut svg, color_enc, color_vals.as_deref(), plot_x, plot_y);
        }

        svg.push_str("</svg>\n");
        Ok(svg)
    }

    /// Render the chart to an HTML string wrapping the SVG.
    ///
    /// # Errors
    ///
    /// Same as [`Chart::to_svg`].
    pub fn to_html(&self) -> Result<String> {
        let svg = self.to_svg()?;
        let title = self.title.as_deref().unwrap_or("Chart");
        Ok(format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <style>body {{ display:flex; justify-content:center; align-items:center; min-height:100vh; margin:0; background:#fafafa; }}</style>
</head>
<body>
{svg}
</body>
</html>
"#,
            title = escape_xml(title),
            svg = svg,
        ))
    }

    // -------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------

    /// Validate that the chart has enough information to render.
    fn validate(&self) -> Result<()> {
        if self.data.is_empty() {
            return Err(VizError::EmptyData);
        }
        if self.x.is_none() {
            return Err(VizError::InvalidParameter {
                name: "x",
                reason: "x encoding is required",
            });
        }
        if self.y.is_none() {
            return Err(VizError::InvalidParameter {
                name: "y",
                reason: "y encoding is required",
            });
        }
        Ok(())
    }

    /// Extract values for a named field across all rows.
    fn extract_field(&self, field: &str) -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(self.data.len());
        for row in &self.data {
            let val = row.iter().find(|(name, _)| name == field).map(|(_, v)| *v);
            match val {
                Some(v) => values.push(v),
                None => {
                    return Err(VizError::RenderError(format!(
                        "field '{field}' not found in data row"
                    )));
                }
            }
        }
        Ok(values)
    }

    /// Render horizontal grid lines.
    #[allow(clippy::too_many_arguments)]
    fn render_grid(
        svg: &mut String,
        plot_x: f64,
        plot_y: f64,
        plot_w: f64,
        plot_h: f64,
        y_min: f64,
        y_max: f64,
        x_min: f64,
        x_max: f64,
    ) {
        let y_ticks = nice_ticks(y_min, y_max, 5);
        let y_range = y_max - y_min;
        for &tick in &y_ticks {
            if y_range.abs() < f64::EPSILON {
                continue;
            }
            let t = (tick - y_min) / y_range;
            let py = plot_y + plot_h - t * plot_h;
            writeln!(
                svg,
                r##"  <line x1="{:.2}" y1="{py:.2}" x2="{:.2}" y2="{py:.2}" stroke="#e0e0e0" stroke-width="0.5"/>"##,
                plot_x,
                plot_x + plot_w,
            )
            .expect("write to String is infallible");
        }
        let x_ticks = nice_ticks(x_min, x_max, 5);
        let x_range = x_max - x_min;
        for &tick in &x_ticks {
            if x_range.abs() < f64::EPSILON {
                continue;
            }
            let t = (tick - x_min) / x_range;
            let px = plot_x + t * plot_w;
            writeln!(
                svg,
                r##"  <line x1="{px:.2}" y1="{:.2}" x2="{px:.2}" y2="{:.2}" stroke="#e0e0e0" stroke-width="0.5"/>"##,
                plot_y,
                plot_y + plot_h,
            )
            .expect("write to String is infallible");
        }
    }

    /// Render x and y axes with ticks and labels.
    #[allow(clippy::too_many_arguments)]
    fn render_axes(
        svg: &mut String,
        plot_x: f64,
        plot_y: f64,
        plot_w: f64,
        plot_h: f64,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        x_enc: &Encoding,
        y_enc: &Encoding,
    ) {
        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        // X-axis line
        writeln!(
            svg,
            r#"  <line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="black" stroke-width="1"/>"#,
            plot_x,
            plot_y + plot_h,
            plot_x + plot_w,
            plot_y + plot_h,
        )
        .expect("write to String is infallible");

        // Y-axis line
        writeln!(
            svg,
            r#"  <line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="black" stroke-width="1"/>"#,
            plot_x, plot_y, plot_x, plot_y + plot_h,
        )
        .expect("write to String is infallible");

        // X tick marks and labels
        let x_ticks = nice_ticks(x_min, x_max, 5);
        for &tick in &x_ticks {
            let t = if x_range.abs() < f64::EPSILON {
                0.5
            } else {
                (tick - x_min) / x_range
            };
            let px = plot_x + t * plot_w;
            let tick_bottom = plot_y + plot_h;
            writeln!(
                svg,
                r#"  <line x1="{px:.2}" y1="{:.2}" x2="{px:.2}" y2="{:.2}" stroke="black" stroke-width="1"/>"#,
                tick_bottom,
                tick_bottom + 5.0,
            )
            .expect("write to String is infallible");
            writeln!(
                svg,
                r#"  <text x="{px:.2}" y="{:.2}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="black">{}</text>"#,
                tick_bottom + 18.0,
                format_tick(tick),
            )
            .expect("write to String is infallible");
        }

        // Y tick marks and labels
        let y_ticks = nice_ticks(y_min, y_max, 5);
        for &tick in &y_ticks {
            let t = if y_range.abs() < f64::EPSILON {
                0.5
            } else {
                (tick - y_min) / y_range
            };
            let py = plot_y + plot_h - t * plot_h;
            writeln!(
                svg,
                r#"  <line x1="{:.2}" y1="{py:.2}" x2="{:.2}" y2="{py:.2}" stroke="black" stroke-width="1"/>"#,
                plot_x - 5.0,
                plot_x,
            )
            .expect("write to String is infallible");
            writeln!(
                svg,
                r#"  <text x="{:.2}" y="{:.2}" text-anchor="end" font-family="sans-serif" font-size="11" fill="black">{}</text>"#,
                plot_x - 8.0,
                py + 4.0,
                format_tick(tick),
            )
            .expect("write to String is infallible");
        }

        // Axis field labels
        let x_label_x = plot_x + plot_w / 2.0;
        let x_label_y = plot_y + plot_h + 40.0;
        writeln!(
            svg,
            r#"  <text x="{x_label_x:.2}" y="{x_label_y:.2}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="black">{}</text>"#,
            escape_xml(&x_enc.field),
        )
        .expect("write to String is infallible");

        let y_label_x = 16.0;
        let y_label_y = plot_y + plot_h / 2.0;
        writeln!(
            svg,
            r#"  <text x="{y_label_x:.2}" y="{y_label_y:.2}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="black" transform="rotate(-90,{y_label_x:.2},{y_label_y:.2})">{}</text>"#,
            escape_xml(&y_enc.field),
        )
        .expect("write to String is infallible");
    }

    /// Render data marks into the plot area.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn render_marks(
        &self,
        svg: &mut String,
        x_vals: &[f64],
        y_vals: &[f64],
        color_vals: Option<&[f64]>,
        size_vals: Option<&[f64]>,
        plot_x: f64,
        plot_y: f64,
        plot_w: f64,
        plot_h: f64,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
    ) {
        let n = x_vals.len();
        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        // Resolve color per point.
        let colors = resolve_colors(color_vals, n);

        // Resolve size per point.
        let sizes = resolve_sizes(size_vals, n, self.mark);

        // Compute pixel positions.
        let px_points: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let tx = if x_range.abs() < f64::EPSILON {
                    0.5
                } else {
                    (x_vals[i] - x_min) / x_range
                };
                let ty = if y_range.abs() < f64::EPSILON {
                    0.5
                } else {
                    (y_vals[i] - y_min) / y_range
                };
                let px = plot_x + tx * plot_w;
                let py = plot_y + plot_h - ty * plot_h;
                (px, py)
            })
            .collect();

        match self.mark {
            Mark::Point | Mark::Circle => {
                for i in 0..n {
                    let (px, py) = px_points[i];
                    let c = colors[i];
                    let r = sizes[i];
                    writeln!(
                        svg,
                        r#"  <circle cx="{px:.2}" cy="{py:.2}" r="{r:.2}" fill="{}" opacity="0.8"/>"#,
                        c.to_svg_color(),
                    )
                    .expect("write to String is infallible");
                }
            }
            Mark::Line => {
                if n > 1 {
                    let mut path = String::with_capacity(n * 20);
                    for (i, &(px, py)) in px_points.iter().enumerate() {
                        if i == 0 {
                            write!(path, "M{px:.2},{py:.2}").expect("infallible");
                        } else {
                            write!(path, " L{px:.2},{py:.2}").expect("infallible");
                        }
                    }
                    let c = colors[0];
                    writeln!(
                        svg,
                        r#"  <path d="{path}" fill="none" stroke="{}" stroke-width="2"/>"#,
                        c.to_svg_color(),
                    )
                    .expect("write to String is infallible");
                }
            }
            Mark::Bar => {
                let bar_w = if n > 1 {
                    (plot_w / n as f64) * 0.8
                } else {
                    plot_w * 0.4
                };
                let baseline = plot_y + plot_h;
                for i in 0..n {
                    let (px, py) = px_points[i];
                    let c = colors[i];
                    let bx = px - bar_w / 2.0;
                    let bh = baseline - py;
                    writeln!(
                        svg,
                        r#"  <rect x="{bx:.2}" y="{py:.2}" width="{bar_w:.2}" height="{bh:.2}" fill="{}" opacity="0.85" stroke="white" stroke-width="0.5"/>"#,
                        c.to_svg_color(),
                    )
                    .expect("write to String is infallible");
                }
            }
            Mark::Area => {
                if n > 1 {
                    let baseline = plot_y + plot_h;
                    let mut path = String::with_capacity(n * 20 + 40);
                    let (first_px, _) = px_points[0];
                    write!(path, "M{first_px:.2},{baseline:.2}").expect("infallible");
                    for &(px, py) in &px_points {
                        write!(path, " L{px:.2},{py:.2}").expect("infallible");
                    }
                    let (last_px, _) = px_points[n - 1];
                    write!(path, " L{last_px:.2},{baseline:.2} Z").expect("infallible");
                    let c = colors[0];
                    writeln!(
                        svg,
                        r#"  <path d="{path}" fill="{}" opacity="0.4" stroke="{}" stroke-width="1.5"/>"#,
                        c.to_svg_color(),
                        c.to_svg_color(),
                    )
                    .expect("write to String is infallible");
                }
            }
            Mark::Rule => {
                for i in 0..n {
                    let (px, py) = px_points[i];
                    let c = colors[i];
                    writeln!(
                        svg,
                        r#"  <line x1="{px:.2}" y1="{:.2}" x2="{px:.2}" y2="{py:.2}" stroke="{}" stroke-width="1.5"/>"#,
                        plot_y + plot_h,
                        c.to_svg_color(),
                    )
                    .expect("write to String is infallible");
                }
            }
            Mark::Text => {
                for i in 0..n {
                    let (px, py) = px_points[i];
                    let c = colors[i];
                    let label = format_tick(y_vals[i]);
                    writeln!(
                        svg,
                        r#"  <text x="{px:.2}" y="{:.2}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="{}">{label}</text>"#,
                        py - 4.0,
                        c.to_svg_color(),
                    )
                    .expect("write to String is infallible");
                }
            }
        }
    }

    /// Render a color legend.
    fn render_legend(
        svg: &mut String,
        color_enc: &Encoding,
        color_vals: Option<&[f64]>,
        plot_x: f64,
        plot_y: f64,
    ) {
        let Some(vals) = color_vals else { return };

        // Collect unique category values (preserve order of first appearance).
        let mut categories: Vec<f64> = Vec::new();
        for &v in vals {
            if !categories.iter().any(|&c| (c - v).abs() < f64::EPSILON) {
                categories.push(v);
            }
        }

        let legend_x = plot_x + 10.0;
        let mut legend_y = plot_y + 10.0;

        // Legend title
        writeln!(
            svg,
            r#"  <text x="{:.2}" y="{:.2}" font-family="sans-serif" font-size="11" font-weight="bold" fill="black">{}</text>"#,
            legend_x,
            legend_y,
            escape_xml(&color_enc.field),
        )
        .expect("write to String is infallible");
        legend_y += 16.0;

        for (i, &cat) in categories.iter().enumerate() {
            let c = PALETTE[i % PALETTE.len()];
            writeln!(
                svg,
                r#"  <rect x="{:.2}" y="{:.2}" width="12" height="12" fill="{}"/>"#,
                legend_x,
                legend_y - 10.0,
                c.to_svg_color(),
            )
            .expect("write to String is infallible");
            writeln!(
                svg,
                r#"  <text x="{:.2}" y="{:.2}" font-family="sans-serif" font-size="11" fill="black">{}</text>"#,
                legend_x + 16.0,
                legend_y,
                format_tick(cat),
            )
            .expect("write to String is infallible");
            legend_y += 18.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Free-standing helpers
// ---------------------------------------------------------------------------

/// Compute min/max with optional padding for the given mark/scale.
fn min_max_padded(vals: &[f64], scale: ScaleType, mark: Mark) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in vals {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, 1.0);
    }

    // For bar charts, always include zero on the y-axis equivalent.
    if matches!(mark, Mark::Bar) {
        if lo > 0.0 {
            lo = 0.0;
        }
        if hi < 0.0 {
            hi = 0.0;
        }
    }

    // Add a small margin so points are not clipped at edges.
    let range = hi - lo;
    if range.abs() < f64::EPSILON {
        let pad = if lo.abs() < f64::EPSILON {
            1.0
        } else {
            lo.abs() * 0.1
        };
        lo -= pad;
        hi += pad;
    } else if matches!(scale, ScaleType::Linear | ScaleType::Temporal) {
        let pad = range * 0.05;
        lo -= pad;
        hi += pad;
    }

    (lo, hi)
}

/// Generate ~`n` nice tick values for a linear range.
fn nice_ticks(min: f64, max: f64, n: usize) -> Vec<f64> {
    if n == 0 || min >= max {
        return vec![];
    }
    let range = max - min;
    let rough_step = range / n as f64;
    let mag = 10.0_f64.powf(rough_step.log10().floor());
    let norm = rough_step / mag;
    let step = if norm < 1.5 {
        mag
    } else if norm < 3.0 {
        2.0 * mag
    } else if norm < 7.0 {
        5.0 * mag
    } else {
        10.0 * mag
    };

    let start = (min / step).ceil() * step;
    let mut ticks = Vec::new();
    let mut v = start;
    let limit = n * 4 + 2;
    while v <= max && ticks.len() < limit {
        ticks.push((v / step).round() * step);
        v += step;
    }
    ticks
}

/// Resolve per-point colors from the color encoding values.
fn resolve_colors(color_vals: Option<&[f64]>, n: usize) -> Vec<Color> {
    match color_vals {
        None => vec![PALETTE[0]; n],
        Some(vals) => {
            // Map unique values to palette indices.
            let mut categories: Vec<f64> = Vec::new();
            for &v in vals {
                if !categories.iter().any(|&c| (c - v).abs() < f64::EPSILON) {
                    categories.push(v);
                }
            }
            vals.iter()
                .map(|&v| {
                    let idx = categories
                        .iter()
                        .position(|&c| (c - v).abs() < f64::EPSILON)
                        .unwrap_or(0);
                    PALETTE[idx % PALETTE.len()]
                })
                .collect()
        }
    }
}

/// Resolve per-point sizes from the size encoding values.
fn resolve_sizes(size_vals: Option<&[f64]>, n: usize, mark: Mark) -> Vec<f64> {
    let default_r = match mark {
        Mark::Circle => 6.0,
        _ => 4.0,
    };
    match size_vals {
        None => vec![default_r; n],
        Some(vals) => {
            let lo = vals.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let range = hi - lo;
            let min_r = 2.0;
            let max_r = 12.0;
            vals.iter()
                .map(|&v| {
                    if range.abs() < f64::EPSILON {
                        default_r
                    } else {
                        min_r + (v - lo) / range * (max_r - min_r)
                    }
                })
                .collect()
        }
    }
}

/// Format a tick value, trimming unnecessary trailing zeros.
fn format_tick(v: f64) -> String {
    if (v - v.round()).abs() < f64::EPSILON && v.abs() < 1e9 {
        format!("{}", v as i64)
    } else {
        format!("{v:.2}")
    }
}

/// Minimal XML escaping for text content.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_columns() -> Vec<(&'static str, Vec<f64>)> {
        vec![
            ("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("y", vec![2.0, 4.0, 3.0, 5.0, 1.0]),
        ]
    }

    fn sample_chart() -> Chart {
        Chart::from_columns(sample_columns())
            .mark(Mark::Point)
            .encode_x("x", ScaleType::Linear)
            .encode_y("y", ScaleType::Linear)
    }

    #[test]
    fn from_columns_builds_rows() {
        let chart = Chart::from_columns(vec![("a", vec![1.0, 2.0]), ("b", vec![3.0, 4.0])]);
        assert_eq!(chart.data.len(), 2);
        assert_eq!(chart.data[0].len(), 2);
        assert_eq!(chart.data[0][0], ("a".to_string(), 1.0));
        assert_eq!(chart.data[0][1], ("b".to_string(), 3.0));
    }

    #[test]
    fn new_from_rows() {
        let rows = vec![
            vec![("x".to_string(), 1.0), ("y".to_string(), 2.0)],
            vec![("x".to_string(), 3.0), ("y".to_string(), 4.0)],
        ];
        let chart = Chart::new(rows)
            .mark(Mark::Line)
            .encode_x("x", ScaleType::Linear)
            .encode_y("y", ScaleType::Linear);
        let svg = chart.to_svg().unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<path"));
    }

    #[test]
    fn to_svg_contains_svg_tag() {
        let svg = sample_chart().to_svg().unwrap();
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn to_svg_contains_circles_for_point() {
        let svg = sample_chart().to_svg().unwrap();
        assert!(svg.contains("<circle"));
        // Should have 5 data circles.
        let count = svg.matches("<circle").count();
        assert_eq!(count, 5);
    }

    #[test]
    fn to_svg_line_mark() {
        let chart = Chart::from_columns(sample_columns())
            .mark(Mark::Line)
            .encode_x("x", ScaleType::Linear)
            .encode_y("y", ScaleType::Linear);
        let svg = chart.to_svg().unwrap();
        assert!(svg.contains("<path"));
        assert!(svg.contains("stroke"));
    }

    #[test]
    fn to_svg_bar_mark() {
        let chart = Chart::from_columns(sample_columns())
            .mark(Mark::Bar)
            .encode_x("x", ScaleType::Linear)
            .encode_y("y", ScaleType::Linear);
        let svg = chart.to_svg().unwrap();
        // Bars produce <rect> elements (beyond the background rects).
        let rect_count = svg.matches("<rect").count();
        // 1 background + 1 plot area + 5 bars = 7
        assert!(rect_count >= 7, "expected >=7 rects, got {rect_count}");
    }

    #[test]
    fn to_svg_area_mark() {
        let chart = Chart::from_columns(sample_columns())
            .mark(Mark::Area)
            .encode_x("x", ScaleType::Linear)
            .encode_y("y", ScaleType::Linear);
        let svg = chart.to_svg().unwrap();
        assert!(svg.contains("<path"));
        assert!(svg.contains("opacity"));
    }

    #[test]
    fn to_svg_rule_mark() {
        let chart = Chart::from_columns(sample_columns())
            .mark(Mark::Rule)
            .encode_x("x", ScaleType::Linear)
            .encode_y("y", ScaleType::Linear);
        let svg = chart.to_svg().unwrap();
        // Rules produce <line> elements.
        assert!(svg.contains("<line"));
    }

    #[test]
    fn to_svg_text_mark() {
        let chart = Chart::from_columns(sample_columns())
            .mark(Mark::Text)
            .encode_x("x", ScaleType::Linear)
            .encode_y("y", ScaleType::Linear);
        let svg = chart.to_svg().unwrap();
        assert!(svg.contains("<text"));
    }

    #[test]
    fn title_appears_in_svg() {
        let svg = sample_chart().title("My Title").to_svg().unwrap();
        assert!(svg.contains("My Title"));
    }

    #[test]
    fn width_and_height() {
        let svg = sample_chart().width(800).height(500).to_svg().unwrap();
        assert!(svg.contains("width=\"800\""));
        assert!(svg.contains("height=\"500\""));
    }

    #[test]
    fn to_html_wraps_svg() {
        let html = sample_chart().title("Test").to_html().unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<svg"));
        assert!(html.contains("Test"));
    }

    #[test]
    fn empty_data_returns_error() {
        let chart = Chart::new(vec![])
            .encode_x("x", ScaleType::Linear)
            .encode_y("y", ScaleType::Linear);
        assert!(chart.to_svg().is_err());
    }

    #[test]
    fn missing_x_encoding_returns_error() {
        let chart = Chart::from_columns(sample_columns()).encode_y("y", ScaleType::Linear);
        assert!(chart.to_svg().is_err());
    }

    #[test]
    fn missing_y_encoding_returns_error() {
        let chart = Chart::from_columns(sample_columns()).encode_x("x", ScaleType::Linear);
        assert!(chart.to_svg().is_err());
    }

    #[test]
    fn missing_field_returns_error() {
        let chart = Chart::from_columns(sample_columns())
            .mark(Mark::Point)
            .encode_x("nonexistent", ScaleType::Linear)
            .encode_y("y", ScaleType::Linear);
        assert!(chart.to_svg().is_err());
    }

    #[test]
    fn color_encoding_produces_legend() {
        let chart = Chart::from_columns(vec![
            ("x", vec![1.0, 2.0, 3.0]),
            ("y", vec![4.0, 5.0, 6.0]),
            ("group", vec![0.0, 1.0, 0.0]),
        ])
        .mark(Mark::Point)
        .encode_x("x", ScaleType::Linear)
        .encode_y("y", ScaleType::Linear)
        .encode_color("group", ScaleType::Ordinal);
        let svg = chart.to_svg().unwrap();
        // Legend should contain the field name.
        assert!(svg.contains("group"));
    }

    #[test]
    fn size_encoding_varies_radius() {
        let chart = Chart::from_columns(vec![
            ("x", vec![1.0, 2.0, 3.0]),
            ("y", vec![4.0, 5.0, 6.0]),
            ("s", vec![10.0, 50.0, 100.0]),
        ])
        .mark(Mark::Point)
        .encode_x("x", ScaleType::Linear)
        .encode_y("y", ScaleType::Linear)
        .encode_size("s", ScaleType::Linear);
        let svg = chart.to_svg().unwrap();
        // All three circles should exist.
        assert_eq!(svg.matches("<circle").count(), 3);
    }

    #[test]
    fn format_tick_integer() {
        assert_eq!(format_tick(42.0), "42");
        assert_eq!(format_tick(-7.0), "-7");
    }

    #[test]
    fn format_tick_decimal() {
        assert_eq!(format_tick(3.75), "3.75");
    }

    #[test]
    fn escape_xml_special_chars() {
        assert_eq!(
            escape_xml("<a>&\"b\"</a>"),
            "&lt;a&gt;&amp;&quot;b&quot;&lt;/a&gt;"
        );
    }

    #[test]
    fn nice_ticks_empty() {
        assert!(nice_ticks(5.0, 5.0, 5).is_empty());
        assert!(nice_ticks(10.0, 5.0, 5).is_empty());
        assert!(nice_ticks(0.0, 10.0, 0).is_empty());
    }

    #[test]
    fn nice_ticks_reasonable() {
        let ticks = nice_ticks(0.0, 100.0, 5);
        assert!(!ticks.is_empty());
        assert!(ticks.len() <= 22);
    }

    #[test]
    fn resolve_colors_no_encoding() {
        let colors = resolve_colors(None, 3);
        assert_eq!(colors.len(), 3);
        assert!(colors.iter().all(|&c| c == PALETTE[0]));
    }

    #[test]
    fn resolve_colors_with_categories() {
        let vals = [0.0, 1.0, 0.0, 2.0];
        let colors = resolve_colors(Some(&vals), 4);
        assert_eq!(colors.len(), 4);
        assert_eq!(colors[0], colors[2]); // same category
        assert_ne!(colors[0], colors[1]); // different categories
    }

    #[test]
    fn resolve_sizes_no_encoding() {
        let sizes = resolve_sizes(None, 3, Mark::Point);
        assert_eq!(sizes, vec![4.0; 3]);
    }

    #[test]
    fn resolve_sizes_maps_range() {
        let vals = [0.0, 50.0, 100.0];
        let sizes = resolve_sizes(Some(&vals), 3, Mark::Point);
        assert!((sizes[0] - 2.0).abs() < f64::EPSILON); // min_r
        assert!((sizes[2] - 12.0).abs() < f64::EPSILON); // max_r
        assert!(sizes[1] > sizes[0] && sizes[1] < sizes[2]); // mid
    }

    #[test]
    fn single_data_point_does_not_panic() {
        let chart = Chart::from_columns(vec![("x", vec![5.0]), ("y", vec![10.0])])
            .mark(Mark::Point)
            .encode_x("x", ScaleType::Linear)
            .encode_y("y", ScaleType::Linear);
        let svg = chart.to_svg().unwrap();
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn constant_values_do_not_panic() {
        let chart =
            Chart::from_columns(vec![("x", vec![3.0, 3.0, 3.0]), ("y", vec![7.0, 7.0, 7.0])])
                .mark(Mark::Point)
                .encode_x("x", ScaleType::Linear)
                .encode_y("y", ScaleType::Linear);
        let svg = chart.to_svg().unwrap();
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn circle_mark_same_as_point() {
        let chart = Chart::from_columns(sample_columns())
            .mark(Mark::Circle)
            .encode_x("x", ScaleType::Linear)
            .encode_y("y", ScaleType::Linear);
        let svg = chart.to_svg().unwrap();
        assert_eq!(svg.matches("<circle").count(), 5);
    }

    #[test]
    fn full_vegalite_style_usage() {
        let chart = Chart::from_columns(vec![
            ("age", vec![25.0, 30.0, 35.0, 40.0]),
            ("salary", vec![50_000.0, 60_000.0, 55_000.0, 70_000.0]),
            ("department", vec![1.0, 2.0, 1.0, 2.0]),
            ("experience", vec![2.0, 5.0, 8.0, 12.0]),
        ])
        .mark(Mark::Point)
        .encode_x("age", ScaleType::Linear)
        .encode_y("salary", ScaleType::Linear)
        .encode_color("department", ScaleType::Ordinal)
        .encode_size("experience", ScaleType::Linear)
        .title("Employee Salary vs Age")
        .width(600)
        .height(400);
        let svg = chart.to_svg().unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Employee Salary vs Age"));
        assert!(svg.contains("department"));
        assert_eq!(svg.matches("<circle").count(), 4);
    }
}
