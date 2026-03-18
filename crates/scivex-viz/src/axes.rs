use crate::annotation::Annotation;
use crate::element::{Element, TextAnchor};
use crate::layout::Rect;
use crate::plot::PlotBuilder;
use crate::scale::{LinearScale, Scale};
use crate::style::{Font, Stroke, Theme};

/// Overrides applied by the figure when sharing axes across subplots.
#[derive(Debug, Clone, Default)]
pub struct AxesOverrides {
    /// Override x-axis range.
    pub x_range: Option<(f64, f64)>,
    /// Override y-axis range.
    pub y_range: Option<(f64, f64)>,
    /// Force-hide x-axis ticks.
    pub hide_x_ticks: bool,
    /// Force-hide y-axis ticks.
    pub hide_y_ticks: bool,
}

/// A single set of axes containing plots, labels, and annotations.
pub struct Axes {
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    x_range: Option<(f64, f64)>,
    y_range: Option<(f64, f64)>,
    plots: Vec<Box<dyn PlotBuilder>>,
    annotations: Vec<Annotation>,
    show_grid: bool,
    show_x_ticks: bool,
    show_y_ticks: bool,
    theme: Theme,
}

impl Axes {
    /// Create a new axes with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            title: None,
            x_label: None,
            y_label: None,
            x_range: None,
            y_range: None,
            plots: Vec::new(),
            annotations: Vec::new(),
            show_grid: true,
            show_x_ticks: true,
            show_y_ticks: true,
            theme: Theme::default(),
        }
    }

    /// Set the axes title.
    #[must_use]
    pub fn title(mut self, t: &str) -> Self {
        self.title = Some(t.to_string());
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

    /// Set the x-axis data range manually.
    #[must_use]
    pub fn x_range(mut self, min: f64, max: f64) -> Self {
        self.x_range = Some((min, max));
        self
    }

    /// Set the y-axis data range manually.
    #[must_use]
    pub fn y_range(mut self, min: f64, max: f64) -> Self {
        self.y_range = Some((min, max));
        self
    }

    /// Show or hide x-axis tick marks and labels.
    #[must_use]
    pub fn hide_x_ticks(mut self, hide: bool) -> Self {
        self.show_x_ticks = !hide;
        self
    }

    /// Show or hide y-axis tick marks and labels.
    #[must_use]
    pub fn hide_y_ticks(mut self, hide: bool) -> Self {
        self.show_y_ticks = !hide;
        self
    }

    /// Override the x-axis range programmatically (used by shared axis logic).
    pub fn set_x_range(&mut self, min: f64, max: f64) {
        self.x_range = Some((min, max));
    }

    /// Override the y-axis range programmatically (used by shared axis logic).
    pub fn set_y_range(&mut self, min: f64, max: f64) {
        self.y_range = Some((min, max));
    }

    /// Get the auto-computed data ranges for this axes (public for shared axes).
    #[must_use]
    pub fn data_ranges(&self) -> ((f64, f64), (f64, f64)) {
        self.auto_ranges()
    }

    /// Enable or disable grid lines.
    #[must_use]
    pub fn grid(mut self, show: bool) -> Self {
        self.show_grid = show;
        self
    }

    /// Set the visual theme.
    #[must_use]
    pub fn theme(mut self, t: Theme) -> Self {
        self.theme = t;
        self
    }

    /// Add a plot to this axes.
    #[must_use]
    pub fn add_plot<P: PlotBuilder + 'static>(mut self, plot: P) -> Self {
        self.plots.push(Box::new(plot));
        self
    }

    /// Add an annotation (reference line, text, or legend).
    #[must_use]
    pub fn annotate(mut self, ann: Annotation) -> Self {
        self.annotations.push(ann);
        self
    }

    /// Render all axes elements (frame, ticks, grid, plots, labels) into the
    /// given pixel `bounds`.
    #[allow(clippy::too_many_lines)]
    pub fn render_elements(&self, bounds: Rect) -> Vec<Element> {
        self.render_elements_with_overrides(bounds, &AxesOverrides::default())
    }

    /// Render with optional overrides from shared-axis logic.
    #[allow(clippy::too_many_lines)]
    pub fn render_elements_with_overrides(
        &self,
        bounds: Rect,
        overrides: &AxesOverrides,
    ) -> Vec<Element> {
        let mut elements = Vec::new();

        // Compute data ranges from plots if not manually set.
        let (auto_xr, auto_yr) = self.auto_ranges();
        let x_range = overrides.x_range.or(self.x_range).unwrap_or(auto_xr);
        let y_range = overrides.y_range.or(self.y_range).unwrap_or(auto_yr);

        let show_x = self.show_x_ticks && !overrides.hide_x_ticks;
        let show_y = self.show_y_ticks && !overrides.hide_y_ticks;

        let x_scale = LinearScale::new(x_range.0, x_range.1);
        let y_scale = LinearScale::new(y_range.0, y_range.1);

        let plot_area = bounds;

        // Background.
        elements.push(Element::Rect {
            x: plot_area.x,
            y: plot_area.y,
            w: plot_area.w,
            h: plot_area.h,
            fill: Some(crate::style::Fill::new(self.theme.background)),
            stroke: None,
        });

        // Grid lines.
        if self.show_grid {
            self.render_grid(&mut elements, &x_scale, &y_scale, plot_area);
        }

        // Axes frame.
        elements.push(Element::Rect {
            x: plot_area.x,
            y: plot_area.y,
            w: plot_area.w,
            h: plot_area.h,
            fill: None,
            stroke: Some(Stroke::new(self.theme.foreground, 1.0)),
        });

        // Tick marks and labels (respects hide_x_ticks / hide_y_ticks).
        self.render_ticks_filtered(&mut elements, &x_scale, &y_scale, plot_area, show_x, show_y);

        // Plot data.
        for plot in &self.plots {
            let plot_elems = plot.build_elements(&x_scale, &y_scale, plot_area);
            elements.extend(plot_elems);
        }

        // Annotations.
        for ann in &self.annotations {
            let ann_elems = ann.render_elements(&x_scale, &y_scale, plot_area, &self.plots);
            elements.extend(ann_elems);
        }

        // Title.
        if let Some(ref title) = self.title {
            elements.push(Element::Text {
                x: plot_area.x + plot_area.w / 2.0,
                y: plot_area.y - 10.0,
                text: title.clone(),
                font: Font {
                    size: 14.0,
                    bold: true,
                    color: self.theme.foreground,
                    ..Font::default()
                },
                anchor: TextAnchor::Middle,
            });
        }

        // X axis label.
        if let Some(ref xlabel) = self.x_label {
            elements.push(Element::Text {
                x: plot_area.x + plot_area.w / 2.0,
                y: plot_area.y + plot_area.h + 40.0,
                text: xlabel.clone(),
                font: Font {
                    size: 12.0,
                    color: self.theme.foreground,
                    ..Font::default()
                },
                anchor: TextAnchor::Middle,
            });
        }

        // Y axis label.
        if let Some(ref ylabel) = self.y_label {
            elements.push(Element::Text {
                x: plot_area.x - 50.0,
                y: plot_area.y + plot_area.h / 2.0,
                text: ylabel.clone(),
                font: Font {
                    size: 12.0,
                    color: self.theme.foreground,
                    ..Font::default()
                },
                anchor: TextAnchor::Middle,
            });
        }

        elements
    }

    fn render_grid(
        &self,
        elements: &mut Vec<Element>,
        x_scale: &dyn Scale,
        y_scale: &dyn Scale,
        area: Rect,
    ) {
        let x_ticks = x_scale.nice_ticks(8);
        let y_ticks = y_scale.nice_ticks(6);
        let grid_stroke = Stroke::new(self.theme.grid_color, self.theme.grid_width);

        for &xt in &x_ticks {
            let px = area.x + x_scale.transform(xt) * area.w;
            elements.push(Element::Line {
                x1: px,
                y1: area.y,
                x2: px,
                y2: area.y + area.h,
                stroke: grid_stroke.clone(),
            });
        }
        for &yt in &y_ticks {
            let py = area.y + area.h - y_scale.transform(yt) * area.h;
            elements.push(Element::Line {
                x1: area.x,
                y1: py,
                x2: area.x + area.w,
                y2: py,
                stroke: grid_stroke.clone(),
            });
        }
    }

    fn render_ticks_filtered(
        &self,
        elements: &mut Vec<Element>,
        x_scale: &dyn Scale,
        y_scale: &dyn Scale,
        area: Rect,
        show_x: bool,
        show_y: bool,
    ) {
        let tick_len = 5.0;
        let tick_font = Font {
            size: 10.0,
            color: self.theme.foreground,
            ..Font::default()
        };

        if show_x {
            let x_ticks = x_scale.nice_ticks(8);
            for &xt in &x_ticks {
                let px = area.x + x_scale.transform(xt) * area.w;
                elements.push(Element::Line {
                    x1: px,
                    y1: area.y + area.h,
                    x2: px,
                    y2: area.y + area.h + tick_len,
                    stroke: Stroke::new(self.theme.foreground, 1.0),
                });
                elements.push(Element::Text {
                    x: px,
                    y: area.y + area.h + tick_len + 12.0,
                    text: format_tick(xt),
                    font: tick_font.clone(),
                    anchor: TextAnchor::Middle,
                });
            }
        }

        if show_y {
            let y_ticks = y_scale.nice_ticks(6);
            for &yt in &y_ticks {
                let py = area.y + area.h - y_scale.transform(yt) * area.h;
                elements.push(Element::Line {
                    x1: area.x - tick_len,
                    y1: py,
                    x2: area.x,
                    y2: py,
                    stroke: Stroke::new(self.theme.foreground, 1.0),
                });
                elements.push(Element::Text {
                    x: area.x - tick_len - 4.0,
                    y: py + 4.0,
                    text: format_tick(yt),
                    font: tick_font.clone(),
                    anchor: TextAnchor::End,
                });
            }
        }
    }

    /// Compute the union of data ranges from all plots.
    fn auto_ranges(&self) -> ((f64, f64), (f64, f64)) {
        let mut x_lo = f64::INFINITY;
        let mut x_hi = f64::NEG_INFINITY;
        let mut y_lo = f64::INFINITY;
        let mut y_hi = f64::NEG_INFINITY;

        for plot in &self.plots {
            let (xr, yr) = plot.data_range();
            if let Some((lo, hi)) = xr {
                if lo < x_lo {
                    x_lo = lo;
                }
                if hi > x_hi {
                    x_hi = hi;
                }
            }
            if let Some((lo, hi)) = yr {
                if lo < y_lo {
                    y_lo = lo;
                }
                if hi > y_hi {
                    y_hi = hi;
                }
            }
        }

        if !x_lo.is_finite() || !x_hi.is_finite() {
            x_lo = 0.0;
            x_hi = 1.0;
        }
        if !y_lo.is_finite() || !y_hi.is_finite() {
            y_lo = 0.0;
            y_hi = 1.0;
        }

        if (x_hi - x_lo).abs() < f64::EPSILON {
            x_lo -= 0.5;
            x_hi += 0.5;
        }
        if (y_hi - y_lo).abs() < f64::EPSILON {
            y_lo -= 0.5;
            y_hi += 0.5;
        }

        ((x_lo, x_hi), (y_lo, y_hi))
    }
}

impl Default for Axes {
    fn default() -> Self {
        Self::new()
    }
}

/// Format a tick value, removing trailing zeros.
fn format_tick(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let abs = v.abs();
    if !(1e-3..1e6).contains(&abs) {
        format!("{v:.2e}")
    } else if (v - v.round()).abs() < 1e-9 {
        format!("{v:.0}")
    } else {
        let s = format!("{v:.2}");
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::element::Element;
    use crate::plot::LinePlot;

    #[test]
    fn axes_default() {
        let a = Axes::default();
        assert!(a.title.is_none());
        assert!(a.x_label.is_none());
        assert!(a.y_label.is_none());
        assert!(a.x_range.is_none());
        assert!(a.y_range.is_none());
        assert!(a.show_grid);
        assert!(a.show_x_ticks);
        assert!(a.show_y_ticks);
        assert!(a.plots.is_empty());
        assert!(a.annotations.is_empty());
    }

    #[test]
    fn axes_title() {
        let a = Axes::new().title("My Title");
        assert_eq!(a.title.as_deref(), Some("My Title"));
    }

    #[test]
    fn axes_labels() {
        let a = Axes::new().x_label("X").y_label("Y");
        assert_eq!(a.x_label.as_deref(), Some("X"));
        assert_eq!(a.y_label.as_deref(), Some("Y"));
    }

    #[test]
    fn axes_ranges() {
        let a = Axes::new().x_range(0.0, 10.0).y_range(-5.0, 5.0);
        assert_eq!(a.x_range, Some((0.0, 10.0)));
        assert_eq!(a.y_range, Some((-5.0, 5.0)));
    }

    #[test]
    fn axes_grid_toggle() {
        let a = Axes::new().grid(false);
        assert!(!a.show_grid);
    }

    #[test]
    fn axes_theme() {
        let a = Axes::new().theme(Theme::default_dark());
        assert_ne!(a.theme.background, Color::WHITE);
    }

    #[test]
    fn axes_add_plot() {
        let a = Axes::new()
            .add_plot(LinePlot::new(vec![0.0], vec![0.0]))
            .add_plot(LinePlot::new(vec![1.0], vec![1.0]));
        assert_eq!(a.plots.len(), 2);
    }

    #[test]
    fn axes_annotate() {
        use crate::annotation::Annotation;
        let a = Axes::new()
            .annotate(Annotation::hline(5.0))
            .annotate(Annotation::vline(3.0));
        assert_eq!(a.annotations.len(), 2);
    }

    #[test]
    fn auto_range_no_plots() {
        let a = Axes::new();
        let (xr, yr) = a.auto_ranges();
        // Should default to (0,1) when no data
        assert!((xr.0 - 0.0).abs() < 1e-10);
        assert!((xr.1 - 1.0).abs() < 1e-10);
        assert!((yr.0 - 0.0).abs() < 1e-10);
        assert!((yr.1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn format_tick_scientific() {
        let s = format_tick(0.0001);
        assert!(s.contains('e'));
    }

    #[test]
    fn format_tick_decimal() {
        assert_eq!(format_tick(2.5), "2.5");
        assert_eq!(format_tick(3.10), "3.1");
    }

    #[test]
    fn render_with_title_and_labels() {
        let axes = Axes::new()
            .title("Title")
            .x_label("X Axis")
            .y_label("Y Axis")
            .add_plot(LinePlot::new(vec![0.0, 1.0], vec![0.0, 1.0]));
        let bounds = Rect {
            x: 70.0,
            y: 50.0,
            w: 700.0,
            h: 490.0,
        };
        let elems = axes.render_elements(bounds);
        // Should have title, x/y labels as text elements
        let text_count = elems
            .iter()
            .filter(|e| matches!(e, Element::Text { .. }))
            .count();
        assert!(
            text_count >= 3,
            "expected title + labels, got {text_count} text elements"
        );
    }

    #[test]
    fn render_no_grid() {
        let axes = Axes::new()
            .grid(false)
            .x_range(0.0, 10.0)
            .y_range(0.0, 10.0);
        let bounds = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 200.0,
        };
        let elems_no_grid = axes.render_elements(bounds);
        let axes_grid = Axes::new().grid(true).x_range(0.0, 10.0).y_range(0.0, 10.0);
        let elems_grid = axes_grid.render_elements(bounds);
        // Grid-enabled should produce more elements
        assert!(elems_grid.len() > elems_no_grid.len());
    }

    #[test]
    fn auto_range_from_data() {
        let axes = Axes::new().add_plot(LinePlot::new(vec![1.0, 5.0], vec![2.0, 8.0]));
        let (xr, yr) = axes.auto_ranges();
        assert!((xr.0 - 1.0).abs() < 1e-10);
        assert!((xr.1 - 5.0).abs() < 1e-10);
        assert!((yr.0 - 2.0).abs() < 1e-10);
        assert!((yr.1 - 8.0).abs() < 1e-10);
    }

    #[test]
    fn render_produces_elements() {
        let axes = Axes::new().add_plot(LinePlot::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.5]));
        let bounds = Rect {
            x: 70.0,
            y: 50.0,
            w: 700.0,
            h: 490.0,
        };
        let elems = axes.render_elements(bounds);
        assert!(elems.len() > 5);
    }

    #[test]
    fn format_tick_values() {
        assert_eq!(format_tick(0.0), "0");
        assert_eq!(format_tick(10.0), "10");
        assert_eq!(format_tick(2.5), "2.5");
    }

    #[test]
    fn hide_x_ticks_reduces_elements() {
        let bounds = Rect {
            x: 70.0,
            y: 50.0,
            w: 700.0,
            h: 490.0,
        };
        let axes_full = Axes::new().x_range(0.0, 10.0).y_range(0.0, 10.0);
        let axes_no_x = Axes::new()
            .x_range(0.0, 10.0)
            .y_range(0.0, 10.0)
            .hide_x_ticks(true);
        let full_elems = axes_full.render_elements(bounds);
        let no_x_elems = axes_no_x.render_elements(bounds);
        assert!(full_elems.len() > no_x_elems.len());
    }

    #[test]
    fn hide_y_ticks_reduces_elements() {
        let bounds = Rect {
            x: 70.0,
            y: 50.0,
            w: 700.0,
            h: 490.0,
        };
        let axes_full = Axes::new().x_range(0.0, 10.0).y_range(0.0, 10.0);
        let axes_no_y = Axes::new()
            .x_range(0.0, 10.0)
            .y_range(0.0, 10.0)
            .hide_y_ticks(true);
        let full_elems = axes_full.render_elements(bounds);
        let no_y_elems = axes_no_y.render_elements(bounds);
        assert!(full_elems.len() > no_y_elems.len());
    }

    #[test]
    fn set_range_programmatic() {
        let mut a = Axes::new();
        a.set_x_range(5.0, 15.0);
        a.set_y_range(-2.0, 8.0);
        assert_eq!(a.x_range, Some((5.0, 15.0)));
        assert_eq!(a.y_range, Some((-2.0, 8.0)));
    }

    #[test]
    fn data_ranges_public() {
        let a = Axes::new().add_plot(LinePlot::new(vec![2.0, 7.0], vec![3.0, 9.0]));
        let (xr, yr) = a.data_ranges();
        assert!((xr.0 - 2.0).abs() < 1e-10);
        assert!((xr.1 - 7.0).abs() < 1e-10);
        assert!((yr.0 - 3.0).abs() < 1e-10);
        assert!((yr.1 - 9.0).abs() < 1e-10);
    }
}
