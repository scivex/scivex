use crate::style::{Fill, Font, Stroke};

/// Text horizontal alignment.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAnchor {
    /// Align text to the left (start of the text run).
    Start,
    /// Center-align text.
    Middle,
    /// Align text to the right (end of the text run).
    End,
}

/// Backend-agnostic drawing primitive.
///
/// All coordinates are in **pixel space** (post-transform from data space).
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub enum Element {
    /// A straight line segment from `(x1, y1)` to `(x2, y2)`.
    Line {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        stroke: Stroke,
    },
    /// A rectangle with optional fill and stroke.
    Rect {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        fill: Option<Fill>,
        stroke: Option<Stroke>,
    },
    /// A circle with optional fill and stroke.
    Circle {
        cx: f64,
        cy: f64,
        r: f64,
        fill: Option<Fill>,
        stroke: Option<Stroke>,
    },
    /// A text label at a given position.
    Text {
        x: f64,
        y: f64,
        text: String,
        font: Font,
        anchor: TextAnchor,
    },
    /// A series of connected line segments with optional fill.
    Polyline {
        points: Vec<(f64, f64)>,
        stroke: Stroke,
        fill: Option<Fill>,
    },
    /// A group of child elements.
    Group { elements: Vec<Element> },
}
