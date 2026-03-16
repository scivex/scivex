/// Bitmap (PNG) rendering backend.
pub mod bitmap;
/// Interactive HTML+JS rendering backend.
pub mod html;
/// SVG rendering backend.
pub mod svg;
/// Terminal braille rendering backend.
pub mod terminal;

use crate::element::Element;
use crate::error::Result;

pub use bitmap::BitmapBackend;
pub use html::HtmlBackend;
pub use svg::SvgBackend;
pub use terminal::TerminalBackend;

/// Trait for rendering a list of drawing elements to a string representation.
pub trait Renderer {
    fn render(&self, elements: &[Element], width: f64, height: f64) -> Result<String>;
}
