use crate::axes::Axes;
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
}

impl Figure {
    #[must_use]
    pub fn new() -> Self {
        Self {
            width: 800.0,
            height: 600.0,
            layout: Layout::single(),
            axes: Vec::new(),
            theme: Theme::default(),
        }
    }

    #[must_use]
    pub fn size(mut self, w: f64, h: f64) -> Self {
        self.width = w;
        self.height = h;
        self
    }

    #[must_use]
    pub fn theme(mut self, t: Theme) -> Self {
        self.theme = t;
        self
    }

    #[must_use]
    pub fn layout(mut self, l: Layout) -> Self {
        self.layout = l;
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

        for (row, col, axes) in &self.axes {
            let bounds = self.layout.cell_bounds(*row, *col, self.width, self.height);
            let ax_elements = axes.render_elements(bounds);
            elements.extend(ax_elements);
        }

        elements
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
}
