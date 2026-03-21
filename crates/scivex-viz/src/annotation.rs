use crate::color::Color;
use crate::element::{Element, TextAnchor};
use crate::layout::Rect;
use crate::plot::PlotBuilder;
use crate::scale::Scale;
use crate::style::{Font, Stroke};

/// An annotation to draw on an axes.
///
/// # Examples
///
/// ```
/// # use scivex_viz::annotation::Annotation;
/// let line = Annotation::hline(5.0_f64);
/// assert!(matches!(line, Annotation::HLine { .. }));
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub enum Annotation {
    /// A horizontal reference line at a given y data value.
    HLine { y: f64, stroke: Stroke },
    /// A vertical reference line at a given x data value.
    VLine { x: f64, stroke: Stroke },
    /// A text label at a given data position.
    Text {
        x: f64,
        y: f64,
        text: String,
        font: Font,
    },
    /// An auto-generated legend collecting labels from plot builders.
    Legend { position: LegendPosition },
}

/// Position for the legend box.
///
/// # Examples
///
/// ```
/// # use scivex_viz::annotation::LegendPosition;
/// let pos = LegendPosition::TopRight;
/// assert_eq!(pos, LegendPosition::TopRight);
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegendPosition {
    TopRight,
    TopLeft,
    BottomRight,
    BottomLeft,
}

impl Annotation {
    /// Create a horizontal reference line.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_viz::annotation::Annotation;
    /// let ann = Annotation::hline(3.5_f64);
    /// assert!(matches!(ann, Annotation::HLine { .. }));
    /// ```
    #[must_use]
    pub fn hline(y: f64) -> Self {
        Self::HLine {
            y,
            stroke: Stroke::new(Color::GRAY, 1.0).dashed(vec![5.0, 3.0]),
        }
    }

    /// Create a vertical reference line.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_viz::annotation::Annotation;
    /// let ann = Annotation::vline(2.0_f64);
    /// assert!(matches!(ann, Annotation::VLine { .. }));
    /// ```
    #[must_use]
    pub fn vline(x: f64) -> Self {
        Self::VLine {
            x,
            stroke: Stroke::new(Color::GRAY, 1.0).dashed(vec![5.0, 3.0]),
        }
    }

    /// Create a text annotation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_viz::annotation::Annotation;
    /// let ann = Annotation::text(1.0_f64, 2.0_f64, "hello");
    /// assert!(matches!(ann, Annotation::Text { .. }));
    /// ```
    #[must_use]
    pub fn text(x: f64, y: f64, text: &str) -> Self {
        Self::Text {
            x,
            y,
            text: text.to_string(),
            font: Font::default(),
        }
    }

    /// Create a legend with the default position (top-right).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_viz::annotation::{Annotation, LegendPosition};
    /// let ann = Annotation::legend();
    /// assert!(matches!(ann, Annotation::Legend { position: LegendPosition::TopRight }));
    /// ```
    #[must_use]
    pub fn legend() -> Self {
        Self::Legend {
            position: LegendPosition::TopRight,
        }
    }

    /// Render the annotation to elements.
    pub(crate) fn render_elements(
        &self,
        x_scale: &dyn Scale,
        y_scale: &dyn Scale,
        area: Rect,
        plots: &[Box<dyn PlotBuilder>],
    ) -> Vec<Element> {
        match self {
            Self::HLine { y, stroke } => {
                let py = area.y + area.h - y_scale.transform(*y) * area.h;
                vec![Element::Line {
                    x1: area.x,
                    y1: py,
                    x2: area.x + area.w,
                    y2: py,
                    stroke: stroke.clone(),
                }]
            }
            Self::VLine { x, stroke } => {
                let px = area.x + x_scale.transform(*x) * area.w;
                vec![Element::Line {
                    x1: px,
                    y1: area.y,
                    x2: px,
                    y2: area.y + area.h,
                    stroke: stroke.clone(),
                }]
            }
            Self::Text { x, y, text, font } => {
                let px = area.x + x_scale.transform(*x) * area.w;
                let py = area.y + area.h - y_scale.transform(*y) * area.h;
                vec![Element::Text {
                    x: px,
                    y: py,
                    text: text.clone(),
                    font: font.clone(),
                    anchor: TextAnchor::Start,
                }]
            }
            Self::Legend { position } => render_legend(*position, area, plots),
        }
    }
}

fn render_legend(
    position: LegendPosition,
    area: Rect,
    plots: &[Box<dyn PlotBuilder>],
) -> Vec<Element> {
    let labels: Vec<&str> = plots.iter().filter_map(|p| p.label()).collect();
    if labels.is_empty() {
        return vec![];
    }

    let line_h = 16.0;
    let pad = 8.0;
    let swatch = 12.0;
    let max_label_w = labels
        .iter()
        .map(|l| l.len() as f64 * 7.0)
        .fold(0.0_f64, f64::max);
    let box_w = pad + swatch + 4.0 + max_label_w + pad;
    let box_h = pad + labels.len() as f64 * line_h + pad;

    let (bx, by) = match position {
        LegendPosition::TopRight => (area.x + area.w - box_w - 5.0, area.y + 5.0),
        LegendPosition::TopLeft => (area.x + 5.0, area.y + 5.0),
        LegendPosition::BottomRight => {
            (area.x + area.w - box_w - 5.0, area.y + area.h - box_h - 5.0)
        }
        LegendPosition::BottomLeft => (area.x + 5.0, area.y + area.h - box_h - 5.0),
    };

    let palette = crate::color::default_palette();
    let mut elements = vec![Element::Rect {
        x: bx,
        y: by,
        w: box_w,
        h: box_h,
        fill: Some(crate::style::Fill::new(Color::rgba(255, 255, 255, 220))),
        stroke: Some(Stroke::new(Color::GRAY, 0.5)),
    }];

    for (i, &lbl) in labels.iter().enumerate() {
        let cy = by + pad + i as f64 * line_h + line_h / 2.0;
        let color = palette[i % palette.len()];
        elements.push(Element::Rect {
            x: bx + pad,
            y: cy - swatch / 2.0,
            w: swatch,
            h: swatch,
            fill: Some(crate::style::Fill::new(color)),
            stroke: None,
        });
        elements.push(Element::Text {
            x: bx + pad + swatch + 4.0,
            y: cy + 4.0,
            text: lbl.to_string(),
            font: Font {
                size: 11.0,
                ..Font::default()
            },
            anchor: TextAnchor::Start,
        });
    }

    elements
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hline_constructor() {
        let ann = Annotation::hline(5.0);
        match ann {
            Annotation::HLine { y, stroke } => {
                assert!((y - 5.0).abs() < f64::EPSILON);
                assert!(stroke.dash.is_some());
            }
            _ => panic!("expected HLine"),
        }
    }

    #[test]
    fn vline_constructor() {
        let ann = Annotation::vline(3.0);
        match ann {
            Annotation::VLine { x, stroke } => {
                assert!((x - 3.0).abs() < f64::EPSILON);
                assert!(stroke.dash.is_some());
            }
            _ => panic!("expected VLine"),
        }
    }

    #[test]
    fn text_constructor() {
        let ann = Annotation::text(1.0, 2.0, "hello");
        match ann {
            Annotation::Text { x, y, text, .. } => {
                assert!((x - 1.0).abs() < f64::EPSILON);
                assert!((y - 2.0).abs() < f64::EPSILON);
                assert_eq!(text, "hello");
            }
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn legend_constructor() {
        let ann = Annotation::legend();
        match ann {
            Annotation::Legend { position } => {
                assert_eq!(position, LegendPosition::TopRight);
            }
            _ => panic!("expected Legend"),
        }
    }

    #[test]
    fn legend_positions_distinct() {
        assert_ne!(LegendPosition::TopRight, LegendPosition::TopLeft);
        assert_ne!(LegendPosition::BottomRight, LegendPosition::BottomLeft);
        assert_ne!(LegendPosition::TopRight, LegendPosition::BottomRight);
    }
}
