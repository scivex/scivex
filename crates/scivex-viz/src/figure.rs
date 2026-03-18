use std::collections::HashMap;

use crate::axes::{Axes, AxesOverrides};
use crate::backend::{Renderer, SvgBackend, TerminalBackend};
use crate::error::Result;
use crate::layout::Layout;
use crate::style::Theme;

/// Top-level container for a visualization. A figure holds a layout and one or
/// more `Axes`, each placed in a grid cell.
pub struct Figure {
    width: f64,
    height: f64,
    layout: Layout,
    axes: Vec<(usize, usize, Axes)>,
    theme: Theme,
    share_x: bool,
    share_y: bool,
}

impl Figure {
    /// Create a new figure with default size (800×600).
    #[must_use]
    pub fn new() -> Self {
        Self {
            width: 800.0,
            height: 600.0,
            layout: Layout::single(),
            axes: Vec::new(),
            theme: Theme::default(),
            share_x: false,
            share_y: false,
        }
    }

    /// Set the figure size in pixels.
    #[must_use]
    pub fn size(mut self, w: f64, h: f64) -> Self {
        self.width = w;
        self.height = h;
        self
    }

    /// Set the visual theme for the figure.
    #[must_use]
    pub fn theme(mut self, t: Theme) -> Self {
        self.theme = t;
        self
    }

    /// Set the grid layout for multi-panel figures.
    #[must_use]
    pub fn layout(mut self, l: Layout) -> Self {
        self.layout = l;
        self
    }

    /// Share x-axis ranges across all subplots in the same column.
    ///
    /// When enabled, all axes in a column use the union of their x-data ranges,
    /// and only the bottom-most axes in each column shows x-tick labels.
    #[must_use]
    pub fn share_x(mut self, share: bool) -> Self {
        self.share_x = share;
        self
    }

    /// Share y-axis ranges across all subplots in the same row.
    ///
    /// When enabled, all axes in a row use the union of their y-data ranges,
    /// and only the left-most axes in each row shows y-tick labels.
    #[must_use]
    pub fn share_y(mut self, share: bool) -> Self {
        self.share_y = share;
        self
    }

    /// Add axes at a specific grid cell `(row, col)`.
    #[must_use]
    pub fn add_axes(mut self, row: usize, col: usize, axes: Axes) -> Self {
        self.axes.push((row, col, axes));
        self
    }

    /// Shorthand for adding a single axes at position `(0, 0)`.
    #[must_use]
    pub fn plot(self, axes: Axes) -> Self {
        self.add_axes(0, 0, axes)
    }

    /// Render the figure to an SVG string.
    pub fn to_svg(&self) -> Result<String> {
        let elements = self.render_all();
        SvgBackend.render(&elements, self.width, self.height)
    }

    /// Render the figure to a terminal braille string.
    pub fn to_terminal(&self) -> Result<String> {
        let elements = self.render_all();
        let backend = TerminalBackend::default();
        backend.render(&elements, self.width, self.height)
    }

    /// Write the figure as an SVG file.
    pub fn save_svg(&self, path: &str) -> Result<()> {
        let svg = self.to_svg()?;
        std::fs::write(path, svg)?;
        Ok(())
    }

    /// Print the figure to stdout as terminal braille art.
    pub fn show_terminal(&self) -> Result<()> {
        let output = self.to_terminal()?;
        print!("{output}");
        Ok(())
    }

    /// Display this figure inline in an evcxr Jupyter notebook.
    ///
    /// This method is auto-detected by the evcxr kernel and renders the plot
    /// as inline SVG in the notebook output cell.
    pub fn evcxr_display(&self) {
        if let Ok(svg) = self.to_svg() {
            println!("EVCXR_BEGIN_CONTENT image/svg+xml\n{svg}\nEVCXR_END_CONTENT");
        }
    }

    fn render_all(&self) -> Vec<crate::element::Element> {
        let mut elements = Vec::new();

        // Figure background.
        elements.push(crate::element::Element::Rect {
            x: 0.0,
            y: 0.0,
            w: self.width,
            h: self.height,
            fill: Some(crate::style::Fill::new(self.theme.background)),
            stroke: None,
        });

        // Compute per-axes overrides for shared axis logic.
        let overrides = self.compute_overrides();

        for (i, (row, col, axes)) in self.axes.iter().enumerate() {
            let bounds = self.layout.cell_bounds(*row, *col, self.width, self.height);
            let ax_elements = axes.render_elements_with_overrides(bounds, &overrides[i]);
            elements.extend(ax_elements);
        }

        elements
    }

    /// Compute per-axes overrides for shared x/y axis logic.
    fn compute_overrides(&self) -> Vec<AxesOverrides> {
        let mut overrides: Vec<AxesOverrides> =
            self.axes.iter().map(|_| AxesOverrides::default()).collect();

        if !self.share_x && !self.share_y {
            return overrides;
        }

        if self.share_x {
            // Find max row per column and unified x-range per column.
            let mut col_max_row: HashMap<usize, usize> = HashMap::new();
            let mut col_x_ranges: HashMap<usize, (f64, f64)> = HashMap::new();

            for (row, col, ax) in &self.axes {
                let entry = col_max_row.entry(*col).or_insert(0);
                if *row > *entry {
                    *entry = *row;
                }
                let (xr, _) = ax.data_ranges();
                let range = col_x_ranges.entry(*col).or_insert(xr);
                if xr.0 < range.0 {
                    range.0 = xr.0;
                }
                if xr.1 > range.1 {
                    range.1 = xr.1;
                }
            }

            for (i, (row, col, _)) in self.axes.iter().enumerate() {
                if let Some(&unified) = col_x_ranges.get(col) {
                    overrides[i].x_range = Some(unified);
                }
                if col_max_row.get(col).is_some_and(|&mr| *row < mr) {
                    overrides[i].hide_x_ticks = true;
                }
            }
        }

        if self.share_y {
            let mut row_min_col: HashMap<usize, usize> = HashMap::new();
            let mut row_y_ranges: HashMap<usize, (f64, f64)> = HashMap::new();

            for (row, col, ax) in &self.axes {
                let entry = row_min_col.entry(*row).or_insert(usize::MAX);
                if *col < *entry {
                    *entry = *col;
                }
                let (_, yr) = ax.data_ranges();
                let range = row_y_ranges.entry(*row).or_insert(yr);
                if yr.0 < range.0 {
                    range.0 = yr.0;
                }
                if yr.1 > range.1 {
                    range.1 = yr.1;
                }
            }

            for (i, (row, col, _)) in self.axes.iter().enumerate() {
                if let Some(&unified) = row_y_ranges.get(row) {
                    overrides[i].y_range = Some(unified);
                }
                if row_min_col.get(col).is_some_and(|&mc| *col > mc) {
                    overrides[i].hide_y_ticks = true;
                }
            }
        }

        overrides
    }
}

impl Default for Figure {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axes::Axes;
    use crate::color::Color;
    use crate::plot::LinePlot;

    #[test]
    fn figure_to_svg_end_to_end() {
        let fig = Figure::new().plot(
            Axes::new()
                .title("Test")
                .add_plot(LinePlot::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.5])),
        );
        let svg = fig.to_svg().unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("Test"));
    }

    #[test]
    fn figure_default() {
        let fig = Figure::default();
        assert!((fig.width - 800.0).abs() < f64::EPSILON);
        assert!((fig.height - 600.0).abs() < f64::EPSILON);
    }

    #[test]
    fn figure_size() {
        let fig = Figure::new().size(1200.0, 900.0);
        assert!((fig.width - 1200.0).abs() < f64::EPSILON);
        assert!((fig.height - 900.0).abs() < f64::EPSILON);
    }

    #[test]
    fn figure_theme() {
        let fig = Figure::new().theme(Theme::default_dark());
        assert_ne!(fig.theme.background, Color::WHITE);
    }

    #[test]
    fn figure_layout() {
        let fig = Figure::new().layout(Layout::grid(2, 3));
        assert_eq!(fig.layout.rows, 2);
        assert_eq!(fig.layout.cols, 3);
    }

    #[test]
    fn figure_add_axes() {
        let fig = Figure::new()
            .add_axes(0, 0, Axes::new())
            .add_axes(0, 1, Axes::new());
        assert_eq!(fig.axes.len(), 2);
    }

    #[test]
    fn figure_plot_shorthand() {
        let fig = Figure::new().plot(Axes::new());
        assert_eq!(fig.axes.len(), 1);
        assert_eq!(fig.axes[0].0, 0);
        assert_eq!(fig.axes[0].1, 0);
    }

    #[test]
    fn figure_svg_contains_viewbox() {
        let fig = Figure::new().size(400.0, 300.0).plot(Axes::new());
        let svg = fig.to_svg().unwrap();
        assert!(svg.contains("400"));
        assert!(svg.contains("300"));
    }

    #[test]
    fn figure_to_terminal() {
        let fig = Figure::new()
            .plot(Axes::new().add_plot(LinePlot::new(vec![0.0, 10.0], vec![0.0, 10.0])));
        let term = fig.to_terminal().unwrap();
        assert!(!term.is_empty());
    }

    #[test]
    fn figure_share_x() {
        let fig = Figure::new()
            .layout(Layout::grid(2, 1))
            .share_x(true)
            .add_axes(
                0,
                0,
                Axes::new().add_plot(LinePlot::new(vec![0.0, 5.0], vec![0.0, 1.0])),
            )
            .add_axes(
                1,
                0,
                Axes::new().add_plot(LinePlot::new(vec![0.0, 10.0], vec![0.0, 2.0])),
            );
        // Should render without panicking; shared x unifies ranges.
        let svg = fig.to_svg().unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn figure_share_y() {
        let fig = Figure::new()
            .layout(Layout::grid(1, 2))
            .share_y(true)
            .add_axes(
                0,
                0,
                Axes::new().add_plot(LinePlot::new(vec![0.0, 5.0], vec![0.0, 1.0])),
            )
            .add_axes(
                0,
                1,
                Axes::new().add_plot(LinePlot::new(vec![0.0, 5.0], vec![0.0, 5.0])),
            );
        let svg = fig.to_svg().unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn figure_share_both() {
        let fig = Figure::new()
            .layout(Layout::grid(2, 2))
            .share_x(true)
            .share_y(true)
            .add_axes(
                0,
                0,
                Axes::new().add_plot(LinePlot::new(vec![0.0, 5.0], vec![0.0, 1.0])),
            )
            .add_axes(
                0,
                1,
                Axes::new().add_plot(LinePlot::new(vec![0.0, 10.0], vec![0.0, 2.0])),
            )
            .add_axes(
                1,
                0,
                Axes::new().add_plot(LinePlot::new(vec![0.0, 5.0], vec![0.0, 3.0])),
            )
            .add_axes(
                1,
                1,
                Axes::new().add_plot(LinePlot::new(vec![0.0, 10.0], vec![0.0, 4.0])),
            );
        let svg = fig.to_svg().unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn figure_share_x_hides_top_ticks() {
        let fig = Figure::new()
            .layout(Layout::grid(2, 1))
            .share_x(true)
            .add_axes(0, 0, Axes::new().x_range(0.0, 10.0).y_range(0.0, 10.0))
            .add_axes(1, 0, Axes::new().x_range(0.0, 10.0).y_range(0.0, 10.0));
        let overrides = fig.compute_overrides();
        // Top row (row 0) should have x ticks hidden.
        assert!(overrides[0].hide_x_ticks);
        // Bottom row (row 1) should show x ticks.
        assert!(!overrides[1].hide_x_ticks);
    }

    #[test]
    fn figure_weighted_layout() {
        let fig = Figure::new()
            .layout(Layout::weighted_grid(vec![1.0, 3.0], vec![2.0, 1.0]))
            .add_axes(0, 0, Axes::new())
            .add_axes(1, 1, Axes::new());
        let svg = fig.to_svg().unwrap();
        assert!(svg.contains("<svg"));
    }
}
