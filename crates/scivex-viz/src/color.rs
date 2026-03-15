use crate::error::{Result, VizError};

/// An RGBA color with 8-bit channels.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Color {
    /// Red channel (0--255).
    pub r: u8,
    /// Green channel (0--255).
    pub g: u8,
    /// Blue channel (0--255).
    pub b: u8,
    /// Alpha channel (0 = transparent, 255 = opaque).
    pub a: u8,
}

impl Color {
    /// Pure red.
    pub const RED: Self = Self::rgb(255, 0, 0);
    /// Dark green.
    pub const GREEN: Self = Self::rgb(0, 128, 0);
    /// Pure blue.
    pub const BLUE: Self = Self::rgb(0, 0, 255);
    /// Pure black.
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    /// Pure white.
    pub const WHITE: Self = Self::rgb(255, 255, 255);
    /// Medium gray.
    pub const GRAY: Self = Self::rgb(128, 128, 128);
    /// Light gray.
    pub const LIGHT_GRAY: Self = Self::rgb(211, 211, 211);
    /// Orange.
    pub const ORANGE: Self = Self::rgb(255, 127, 14);
    /// Purple.
    pub const PURPLE: Self = Self::rgb(148, 103, 189);
    /// Cyan / deep sky blue.
    pub const CYAN: Self = Self::rgb(0, 191, 255);
    /// Pure yellow.
    pub const YELLOW: Self = Self::rgb(255, 255, 0);
    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };

    /// Create an opaque color from RGB components.
    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Create a color from RGBA components.
    #[must_use]
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Parse a hex color string like `"#FF0000"` or `"#FF0000FF"`.
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.strip_prefix('#').unwrap_or(hex);
        let parse_byte = |s: &str| {
            u8::from_str_radix(s, 16).map_err(|_| VizError::InvalidParameter {
                name: "hex",
                reason: "invalid hex color string",
            })
        };
        match hex.len() {
            6 => Ok(Self::rgb(
                parse_byte(&hex[0..2])?,
                parse_byte(&hex[2..4])?,
                parse_byte(&hex[4..6])?,
            )),
            8 => Ok(Self::rgba(
                parse_byte(&hex[0..2])?,
                parse_byte(&hex[2..4])?,
                parse_byte(&hex[4..6])?,
                parse_byte(&hex[6..8])?,
            )),
            _ => Err(VizError::InvalidParameter {
                name: "hex",
                reason: "expected 6 or 8 hex digits",
            }),
        }
    }

    /// Convert to hex string like `"#ff0000"` (or `"#ff000080"` if alpha < 255).
    #[must_use]
    pub fn to_hex(self) -> String {
        if self.a == 255 {
            format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
        } else {
            format!("#{:02x}{:02x}{:02x}{:02x}", self.r, self.g, self.b, self.a)
        }
    }

    /// Linearly interpolate between two colors. `t` is clamped to `[0, 1]`.
    #[must_use]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        let mix = |a: u8, b: u8| -> u8 {
            let v = f64::from(a) * (1.0 - t) + f64::from(b) * t;
            v.round() as u8
        };
        Self {
            r: mix(self.r, other.r),
            g: mix(self.g, other.g),
            b: mix(self.b, other.b),
            a: mix(self.a, other.a),
        }
    }

    /// SVG-compatible color string (e.g. `"rgb(255,0,0)"`).
    #[must_use]
    pub fn to_svg_color(self) -> String {
        if self.a == 255 {
            format!("rgb({},{},{})", self.r, self.g, self.b)
        } else {
            format!(
                "rgba({},{},{},{:.3})",
                self.r,
                self.g,
                self.b,
                f64::from(self.a) / 255.0
            )
        }
    }
}

/// A continuous colormap that maps values in `[0, 1]` to colors.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct ColorMap {
    colors: Vec<Color>,
}

impl ColorMap {
    /// Create a colormap from a list of evenly-spaced color stops.
    ///
    /// # Errors
    /// Returns `EmptyData` if `colors` is empty.
    pub fn new(colors: Vec<Color>) -> Result<Self> {
        if colors.is_empty() {
            return Err(VizError::EmptyData);
        }
        Ok(Self { colors })
    }

    /// Sample the colormap at `t` in `[0, 1]`.
    #[must_use]
    pub fn sample(&self, t: f64) -> Color {
        let t = t.clamp(0.0, 1.0);
        let n = self.colors.len();
        if n == 1 {
            return self.colors[0];
        }
        let scaled = t * (n - 1) as f64;
        let lo = (scaled.floor() as usize).min(n - 2);
        let frac = scaled - lo as f64;
        self.colors[lo].lerp(self.colors[lo + 1], frac)
    }

    /// Viridis-like colormap (8-stop approximation).
    #[must_use]
    pub fn viridis() -> Self {
        Self {
            colors: vec![
                Color::rgb(68, 1, 84),
                Color::rgb(72, 36, 117),
                Color::rgb(56, 88, 140),
                Color::rgb(39, 130, 142),
                Color::rgb(31, 158, 137),
                Color::rgb(78, 195, 107),
                Color::rgb(163, 219, 58),
                Color::rgb(253, 231, 37),
            ],
        }
    }

    /// Plasma-like colormap (8-stop approximation).
    #[must_use]
    pub fn plasma() -> Self {
        Self {
            colors: vec![
                Color::rgb(13, 8, 135),
                Color::rgb(84, 2, 163),
                Color::rgb(139, 10, 165),
                Color::rgb(185, 50, 137),
                Color::rgb(219, 92, 104),
                Color::rgb(244, 136, 73),
                Color::rgb(254, 188, 43),
                Color::rgb(240, 249, 33),
            ],
        }
    }

    /// Inferno-like colormap (8-stop approximation).
    #[must_use]
    pub fn inferno() -> Self {
        Self {
            colors: vec![
                Color::rgb(0, 0, 4),
                Color::rgb(40, 11, 84),
                Color::rgb(101, 21, 110),
                Color::rgb(159, 42, 99),
                Color::rgb(212, 72, 66),
                Color::rgb(245, 125, 21),
                Color::rgb(250, 193, 39),
                Color::rgb(252, 255, 164),
            ],
        }
    }

    /// Cool-warm diverging colormap.
    #[must_use]
    pub fn coolwarm() -> Self {
        Self {
            colors: vec![
                Color::rgb(59, 76, 192),
                Color::rgb(124, 159, 230),
                Color::rgb(196, 210, 240),
                Color::rgb(235, 235, 235),
                Color::rgb(240, 196, 179),
                Color::rgb(221, 132, 109),
                Color::rgb(180, 4, 38),
            ],
        }
    }
}

/// Default 10-color categorical palette (similar to matplotlib tab10).
#[must_use]
pub fn default_palette() -> Vec<Color> {
    vec![
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
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_roundtrip() {
        let c = Color::rgb(255, 128, 0);
        let hex = c.to_hex();
        let c2 = Color::from_hex(&hex).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    fn hex_with_alpha() {
        let c = Color::from_hex("#FF000080").unwrap();
        assert_eq!(c, Color::rgba(255, 0, 0, 128));
        assert_eq!(c.to_hex(), "#ff000080");
    }

    #[test]
    fn lerp_endpoints() {
        let a = Color::BLACK;
        let b = Color::WHITE;
        assert_eq!(a.lerp(b, 0.0), a);
        assert_eq!(a.lerp(b, 1.0), b);
    }

    #[test]
    fn lerp_midpoint() {
        let c = Color::rgb(0, 0, 0).lerp(Color::rgb(200, 100, 50), 0.5);
        assert_eq!(c, Color::rgb(100, 50, 25));
    }

    #[test]
    fn colormap_endpoints() {
        let cm = ColorMap::viridis();
        assert_eq!(cm.sample(0.0), Color::rgb(68, 1, 84));
        assert_eq!(cm.sample(1.0), Color::rgb(253, 231, 37));
    }

    #[test]
    fn colormap_sample_mid() {
        let cm = ColorMap::new(vec![Color::BLACK, Color::WHITE]).unwrap();
        let mid = cm.sample(0.5);
        assert_eq!(mid, Color::rgb(128, 128, 128));
    }

    #[test]
    fn named_constants() {
        assert_eq!(Color::RED, Color::rgb(255, 0, 0));
        assert_eq!(Color::BLACK, Color::rgb(0, 0, 0));
    }

    #[test]
    fn hex_without_hash() {
        let c = Color::from_hex("00FF00").unwrap();
        assert_eq!(c, Color::rgb(0, 255, 0));
    }

    #[test]
    fn hex_invalid_length() {
        let r = Color::from_hex("#FFF");
        assert!(r.is_err());
    }

    #[test]
    fn hex_invalid_chars() {
        let r = Color::from_hex("#ZZZZZZ");
        assert!(r.is_err());
    }

    #[test]
    fn svg_color_opaque() {
        let c = Color::rgb(10, 20, 30);
        assert_eq!(c.to_svg_color(), "rgb(10,20,30)");
    }

    #[test]
    fn svg_color_transparent() {
        let c = Color::rgba(10, 20, 30, 128);
        let s = c.to_svg_color();
        assert!(s.starts_with("rgba(10,20,30,"));
    }

    #[test]
    fn lerp_clamps_below_zero() {
        let a = Color::BLACK;
        let b = Color::WHITE;
        assert_eq!(a.lerp(b, -1.0), a);
    }

    #[test]
    fn lerp_clamps_above_one() {
        let a = Color::BLACK;
        let b = Color::WHITE;
        assert_eq!(a.lerp(b, 2.0), b);
    }

    #[test]
    fn transparent_constant() {
        assert_eq!(Color::TRANSPARENT.a, 0);
    }

    #[test]
    fn colormap_empty_errors() {
        let r = ColorMap::new(vec![]);
        assert!(r.is_err());
    }

    #[test]
    fn colormap_single_color() {
        let cm = ColorMap::new(vec![Color::RED]).unwrap();
        assert_eq!(cm.sample(0.0), Color::RED);
        assert_eq!(cm.sample(0.5), Color::RED);
        assert_eq!(cm.sample(1.0), Color::RED);
    }

    #[test]
    fn colormap_clamps_t() {
        let cm = ColorMap::viridis();
        let lo = cm.sample(-1.0);
        let hi = cm.sample(2.0);
        assert_eq!(lo, cm.sample(0.0));
        assert_eq!(hi, cm.sample(1.0));
    }

    #[test]
    fn plasma_colormap_endpoints() {
        let cm = ColorMap::plasma();
        assert_eq!(cm.sample(0.0), Color::rgb(13, 8, 135));
    }

    #[test]
    fn inferno_colormap_endpoints() {
        let cm = ColorMap::inferno();
        assert_eq!(cm.sample(0.0), Color::rgb(0, 0, 4));
    }

    #[test]
    fn coolwarm_colormap_endpoints() {
        let cm = ColorMap::coolwarm();
        assert_eq!(cm.sample(0.0), Color::rgb(59, 76, 192));
    }

    #[test]
    fn default_palette_has_ten_colors() {
        let p = default_palette();
        assert_eq!(p.len(), 10);
    }

    #[test]
    fn color_equality() {
        let a = Color::rgb(1, 2, 3);
        let b = Color::rgb(1, 2, 3);
        let c = Color::rgb(1, 2, 4);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
