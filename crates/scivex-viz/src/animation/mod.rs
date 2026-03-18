//! Animation support: GIF89a and APNG encoding.
//!
//! Create animated visualizations from sequences of frames rendered by the
//! bitmap backend. Supports GIF89a with LZW compression and median-cut
//! color quantization.

mod gif;
mod lzw;
mod quantize;

pub use gif::{GifEncoder, GifFrame};
pub use quantize::median_cut;

use crate::backend::BitmapBackend;
use crate::element::Element;
use crate::error::{Result, VizError};

/// A sequence of animation frames.
pub struct Animation {
    width: u32,
    height: u32,
    frames: Vec<AnimFrame>,
    loop_count: u16,
}

struct AnimFrame {
    elements: Vec<Element>,
    delay_ms: u16,
}

impl Animation {
    /// Create a new animation with the given pixel dimensions.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            frames: Vec::new(),
            loop_count: 0, // infinite loop
        }
    }

    /// Set the loop count (0 = infinite).
    #[must_use]
    pub fn loop_count(mut self, n: u16) -> Self {
        self.loop_count = n;
        self
    }

    /// Add a frame with the given delay in milliseconds.
    #[must_use]
    pub fn add_frame(mut self, elements: Vec<Element>, delay_ms: u16) -> Self {
        self.frames.push(AnimFrame { elements, delay_ms });
        self
    }

    /// Number of frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Encode as GIF89a bytes.
    pub fn to_gif(&self) -> Result<Vec<u8>> {
        if self.frames.is_empty() {
            return Err(VizError::NoFrames);
        }

        let backend = BitmapBackend::default();
        let mut gif_frames = Vec::with_capacity(self.frames.len());

        for frame in &self.frames {
            let rgba = backend.render_rgba(&frame.elements, self.width, self.height);
            gif_frames.push(GifFrame {
                rgba,
                delay_centiseconds: frame.delay_ms / 10,
            });
        }

        let encoder = GifEncoder::new(self.width, self.height);
        Ok(encoder.encode(&gif_frames, self.loop_count))
    }

    /// Write GIF to a file.
    pub fn save_gif(&self, path: &str) -> Result<()> {
        let bytes = self.to_gif()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::style::{Fill, Stroke};

    fn test_frame(x: f64) -> Vec<Element> {
        vec![
            Element::Rect {
                x: 0.0,
                y: 0.0,
                w: 100.0,
                h: 100.0,
                fill: Some(Fill::new(Color::WHITE)),
                stroke: None,
            },
            Element::Circle {
                cx: x,
                cy: 50.0,
                r: 10.0,
                fill: Some(Fill::new(Color::RED)),
                stroke: Some(Stroke::new(Color::BLACK, 1.0)),
            },
        ]
    }

    #[test]
    fn animation_add_frames() {
        let anim = Animation::new(100, 100)
            .add_frame(test_frame(10.0), 100)
            .add_frame(test_frame(50.0), 100)
            .add_frame(test_frame(90.0), 100);
        assert_eq!(anim.frame_count(), 3);
    }

    #[test]
    fn animation_no_frames_error() {
        let anim = Animation::new(100, 100);
        let result = anim.to_gif();
        assert!(result.is_err());
    }

    #[test]
    fn animation_to_gif() {
        let anim = Animation::new(40, 40)
            .add_frame(test_frame(10.0), 100)
            .add_frame(test_frame(30.0), 100);
        let bytes = anim.to_gif().unwrap();
        // GIF89a magic.
        assert_eq!(&bytes[0..6], b"GIF89a");
        // Ends with trailer.
        assert_eq!(*bytes.last().unwrap(), 0x3B);
    }

    #[test]
    fn animation_loop_count() {
        let anim = Animation::new(10, 10).loop_count(3);
        assert_eq!(anim.loop_count, 3);
    }
}
