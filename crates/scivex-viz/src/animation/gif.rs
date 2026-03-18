//! GIF89a encoder.
//!
//! Produces animated GIF files from sequences of RGBA frames using median-cut
//! quantization and LZW compression.

use super::lzw::lzw_encode;
use super::quantize::median_cut;

/// A single frame for GIF encoding.
pub struct GifFrame {
    /// Raw RGBA pixel data.
    pub rgba: Vec<u8>,
    /// Frame delay in centiseconds (1/100 s).
    pub delay_centiseconds: u16,
}

/// GIF89a encoder.
pub struct GifEncoder {
    width: u16,
    height: u16,
}

impl GifEncoder {
    /// Create a new encoder for the given dimensions.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self {
            width: width as u16,
            height: height as u16,
        }
    }

    /// Encode frames into a GIF89a byte vector.
    #[must_use]
    pub fn encode(&self, frames: &[GifFrame], loop_count: u16) -> Vec<u8> {
        let mut out = Vec::with_capacity(1024);

        // --- Header ---
        out.extend_from_slice(b"GIF89a");

        // Use a global palette from the first frame.
        let first_q = if frames.is_empty() {
            median_cut(&[], 2)
        } else {
            median_cut(&frames[0].rgba, 256)
        };

        let palette_size = first_q.palette.len().next_power_of_two().max(2);
        let color_table_bits = log2_ceil(palette_size) as u8;

        // --- Logical Screen Descriptor ---
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        // Packed: global color table flag (1), color resolution (3), sort (0), size of table (3).
        let packed = 0x80 | ((color_table_bits - 1) << 4) | (color_table_bits - 1);
        out.push(packed);
        out.push(0); // background color index
        out.push(0); // pixel aspect ratio

        // --- Global Color Table ---
        for i in 0..palette_size {
            if i < first_q.palette.len() {
                let c = first_q.palette[i];
                out.extend_from_slice(&c);
            } else {
                out.extend_from_slice(&[0, 0, 0]); // pad
            }
        }

        // --- Netscape Application Extension (for looping) ---
        out.push(0x21); // extension introducer
        out.push(0xFF); // application extension
        out.push(11); // block size
        out.extend_from_slice(b"NETSCAPE2.0");
        out.push(3); // sub-block size
        out.push(1); // sub-block ID
        out.extend_from_slice(&loop_count.to_le_bytes());
        out.push(0); // block terminator

        // --- Frames ---
        for (i, frame) in frames.iter().enumerate() {
            let q = if i == 0 {
                // Reuse already-computed quantization for first frame.
                median_cut(&frame.rgba, 256)
            } else {
                median_cut(&frame.rgba, 256)
            };

            self.write_frame(
                &mut out,
                &q.indices,
                &q.palette,
                frame.delay_centiseconds,
                q.transparent_index,
            );
        }

        // --- Trailer ---
        out.push(0x3B);

        out
    }

    fn write_frame(
        &self,
        out: &mut Vec<u8>,
        indices: &[u8],
        palette: &[[u8; 3]],
        delay: u16,
        transparent: Option<u8>,
    ) {
        // --- Graphic Control Extension ---
        out.push(0x21); // extension introducer
        out.push(0xF9); // graphic control
        out.push(4); // block size
        let disposal = 2u8; // restore to background
        let has_trans = u8::from(transparent.is_some());
        let packed = (disposal << 2) | has_trans;
        out.push(packed);
        out.extend_from_slice(&delay.to_le_bytes());
        out.push(transparent.unwrap_or(0));
        out.push(0); // block terminator

        // --- Image Descriptor ---
        out.push(0x2C); // image separator
        out.extend_from_slice(&0u16.to_le_bytes()); // left
        out.extend_from_slice(&0u16.to_le_bytes()); // top
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());

        // Local color table.
        let palette_size = palette.len().next_power_of_two().max(2);
        let lct_bits = log2_ceil(palette_size) as u8;
        let packed = 0x80 | (lct_bits - 1); // local color table flag + size
        out.push(packed);

        // Write local color table.
        for i in 0..palette_size {
            if i < palette.len() {
                out.extend_from_slice(&palette[i]);
            } else {
                out.extend_from_slice(&[0, 0, 0]);
            }
        }

        // --- LZW Image Data ---
        let min_code_size = lct_bits.max(2);
        out.push(min_code_size);

        let compressed = lzw_encode(min_code_size, indices);

        // Write as sub-blocks (max 255 bytes each).
        for chunk in compressed.chunks(255) {
            #[allow(clippy::cast_possible_truncation)]
            out.push(chunk.len() as u8);
            out.extend_from_slice(chunk);
        }
        out.push(0); // block terminator
    }
}

fn log2_ceil(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut bits = 0;
    let mut v = n - 1;
    while v > 0 {
        bits += 1;
        v >>= 1;
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gif_encode_single_frame() {
        // 2x2 red image.
        let rgba = vec![
            255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
        ];
        let frame = GifFrame {
            rgba,
            delay_centiseconds: 10,
        };
        let encoder = GifEncoder::new(2, 2);
        let bytes = encoder.encode(&[frame], 0);
        assert_eq!(&bytes[0..6], b"GIF89a");
        assert_eq!(*bytes.last().unwrap(), 0x3B);
    }

    #[test]
    fn gif_encode_multi_frame() {
        let red = vec![
            255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
        ];
        let blue = vec![
            0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255,
        ];
        let frames = vec![
            GifFrame {
                rgba: red,
                delay_centiseconds: 50,
            },
            GifFrame {
                rgba: blue,
                delay_centiseconds: 50,
            },
        ];
        let encoder = GifEncoder::new(2, 2);
        let bytes = encoder.encode(&frames, 0);
        assert_eq!(&bytes[0..6], b"GIF89a");
        // Should contain NETSCAPE extension.
        let netscape = b"NETSCAPE2.0";
        let found = bytes.windows(netscape.len()).any(|w| w == netscape);
        assert!(found);
    }

    #[test]
    fn log2_ceil_values() {
        assert_eq!(log2_ceil(1), 1);
        assert_eq!(log2_ceil(2), 1);
        assert_eq!(log2_ceil(3), 2);
        assert_eq!(log2_ceil(4), 2);
        assert_eq!(log2_ceil(5), 3);
        assert_eq!(log2_ceil(256), 8);
    }
}
