use crate::color::{self, Color};

/// Line/border stroke properties.
#[derive(Debug, Clone)]
pub struct Stroke {
    /// Stroke color.
    pub color: Color,
    /// Stroke width in pixels.
    pub width: f64,
    /// Optional dash pattern (alternating on/off lengths).
    pub dash: Option<Vec<f64>>,
}

impl Stroke {
    /// Create a new stroke with the given color and width.
    #[must_use]
    pub fn new(color: Color, width: f64) -> Self {
        Self {
            color,
            width,
            dash: None,
        }
    }

    /// Apply a dash pattern to this stroke.
    #[must_use]
    pub fn dashed(mut self, pattern: Vec<f64>) -> Self {
        self.dash = Some(pattern);
        self
    }
}

impl Default for Stroke {
    fn default() -> Self {
        Self::new(Color::BLACK, 1.0)
    }
}

/// Fill properties.
#[derive(Debug, Clone, Copy)]
pub struct Fill {
    /// Fill color.
    pub color: Color,
}

impl Fill {
    /// Create a new fill with the given color.
    #[must_use]
    pub const fn new(color: Color) -> Self {
        Self { color }
    }
}

/// Marker shape for scatter-style plots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerShape {
    /// A filled circle.
    Circle,
    /// A filled square.
    Square,
    /// A filled triangle.
    Triangle,
    /// An "X" cross shape.
    Cross,
    /// A "+" plus shape.
    Plus,
    /// A rotated square (diamond).
    Diamond,
}

/// A point marker (shape + size + color).
#[derive(Debug, Clone, Copy)]
pub struct Marker {
    /// The marker shape.
    pub shape: MarkerShape,
    /// Marker radius in pixels.
    pub size: f64,
    /// Marker color.
    pub color: Color,
}

impl Default for Marker {
    fn default() -> Self {
        Self {
            shape: MarkerShape::Circle,
            size: 4.0,
            color: Color::rgb(31, 119, 180),
        }
    }
}

/// Font specification for text elements.
#[derive(Debug, Clone)]
pub struct Font {
    /// Font family name (e.g. `"sans-serif"`).
    pub family: String,
    /// Font size in points.
    pub size: f64,
    /// Text color.
    pub color: Color,
    /// Whether the font is bold.
    pub bold: bool,
    /// Whether the font is italic.
    pub italic: bool,
}

impl Default for Font {
    fn default() -> Self {
        Self {
            family: "sans-serif".to_string(),
            size: 12.0,
            color: Color::BLACK,
            bold: false,
            italic: false,
        }
    }
}

/// A visual theme controlling default colors and fonts.
#[derive(Debug, Clone)]
pub struct Theme {
    /// Background fill color.
    pub background: Color,
    /// Foreground color for axes, ticks, and labels.
    pub foreground: Color,
    /// Categorical color palette for plot series.
    pub palette: Vec<Color>,
    /// Default font for text elements.
    pub font: Font,
    /// Grid line color.
    pub grid_color: Color,
    /// Grid line width in pixels.
    pub grid_width: f64,
}

impl Theme {
    /// Create the default light theme.
    #[must_use]
    pub fn default_light() -> Self {
        Self {
            background: Color::WHITE,
            foreground: Color::BLACK,
            palette: color::default_palette(),
            font: Font::default(),
            grid_color: Color::LIGHT_GRAY,
            grid_width: 0.5,
        }
    }

    /// Create the default dark theme.
    #[must_use]
    pub fn default_dark() -> Self {
        Self {
            background: Color::rgb(32, 32, 32),
            foreground: Color::rgb(220, 220, 220),
            palette: color::default_palette(),
            font: Font {
                color: Color::rgb(220, 220, 220),
                ..Font::default()
            },
            grid_color: Color::rgb(64, 64, 64),
            grid_width: 0.5,
        }
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::default_light()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stroke_new_defaults() {
        let s = Stroke::new(Color::RED, 2.0);
        assert_eq!(s.color, Color::RED);
        assert!((s.width - 2.0).abs() < f64::EPSILON);
        assert!(s.dash.is_none());
    }

    #[test]
    fn stroke_dashed() {
        let s = Stroke::new(Color::BLACK, 1.0).dashed(vec![5.0, 3.0]);
        assert_eq!(s.dash, Some(vec![5.0, 3.0]));
    }

    #[test]
    fn stroke_default() {
        let s = Stroke::default();
        assert_eq!(s.color, Color::BLACK);
        assert!((s.width - 1.0).abs() < f64::EPSILON);
        assert!(s.dash.is_none());
    }

    #[test]
    fn fill_new() {
        let f = Fill::new(Color::BLUE);
        assert_eq!(f.color, Color::BLUE);
    }

    #[test]
    fn marker_shape_values() {
        assert_ne!(MarkerShape::Circle, MarkerShape::Square);
        assert_ne!(MarkerShape::Triangle, MarkerShape::Cross);
        assert_ne!(MarkerShape::Plus, MarkerShape::Diamond);
    }

    #[test]
    fn marker_default() {
        let m = Marker::default();
        assert_eq!(m.shape, MarkerShape::Circle);
        assert!((m.size - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn font_default() {
        let f = Font::default();
        assert_eq!(f.family, "sans-serif");
        assert!((f.size - 12.0).abs() < f64::EPSILON);
        assert_eq!(f.color, Color::BLACK);
        assert!(!f.bold);
        assert!(!f.italic);
    }

    #[test]
    fn theme_light() {
        let t = Theme::default_light();
        assert_eq!(t.background, Color::WHITE);
        assert_eq!(t.foreground, Color::BLACK);
        assert!(!t.palette.is_empty());
        assert!((t.grid_width - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn theme_dark() {
        let t = Theme::default_dark();
        assert_ne!(t.background, Color::WHITE);
        assert_ne!(t.foreground, Color::BLACK);
        assert!(!t.palette.is_empty());
    }

    #[test]
    fn theme_default_is_light() {
        let d = Theme::default();
        let l = Theme::default_light();
        assert_eq!(d.background, l.background);
        assert_eq!(d.foreground, l.foreground);
    }
}
