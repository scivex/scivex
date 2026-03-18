//! Median-cut color quantization.
//!
//! Reduces RGBA images to a 256-color palette for GIF encoding.

/// Result of color quantization.
pub struct QuantizedImage {
    /// The color palette (up to 256 entries, each [R, G, B]).
    pub palette: Vec<[u8; 3]>,
    /// Palette-indexed pixel data.
    pub indices: Vec<u8>,
    /// Index of the transparent color, if any.
    pub transparent_index: Option<u8>,
}

/// Quantize an RGBA image to at most 256 colors using median-cut.
///
/// - `rgba`: raw RGBA pixel data (4 bytes per pixel).
/// - `max_colors`: maximum palette size (capped at 256).
///
/// Pixels with alpha < 128 are mapped to a dedicated transparent palette entry.
pub fn median_cut(rgba: &[u8], max_colors: usize) -> QuantizedImage {
    let max_colors = max_colors.min(256);
    let n_pixels = rgba.len() / 4;

    // Collect unique opaque colors.
    let mut colors: Vec<[u8; 3]> = Vec::with_capacity(n_pixels);
    let mut has_transparent = false;

    for pixel in rgba.chunks_exact(4) {
        if pixel[3] < 128 {
            has_transparent = true;
        } else {
            colors.push([pixel[0], pixel[1], pixel[2]]);
        }
    }

    // Reserve one slot for transparency if needed.
    let palette_slots = if has_transparent {
        max_colors.saturating_sub(1)
    } else {
        max_colors
    };

    // Median-cut quantization.
    let palette = if colors.is_empty() {
        vec![[0, 0, 0]]
    } else {
        cut_boxes(&colors, palette_slots)
    };

    // Build transparency.
    let transparent_index = if has_transparent {
        #[allow(clippy::cast_possible_truncation)]
        Some(palette.len() as u8)
    } else {
        None
    };

    // Build full palette including transparent entry.
    let mut full_palette = palette;
    if has_transparent {
        full_palette.push([0, 0, 0]); // transparent color (doesn't matter visually).
    }

    // Map each pixel to nearest palette entry.
    let t_idx = transparent_index.unwrap_or(0);
    let indices: Vec<u8> = rgba
        .chunks_exact(4)
        .map(|pixel| {
            if pixel[3] < 128 {
                t_idx
            } else {
                nearest_color(
                    &full_palette,
                    pixel[0],
                    pixel[1],
                    pixel[2],
                    transparent_index,
                )
            }
        })
        .collect();

    QuantizedImage {
        palette: full_palette,
        indices,
        transparent_index,
    }
}

fn nearest_color(palette: &[[u8; 3]], r: u8, g: u8, b: u8, skip: Option<u8>) -> u8 {
    let mut best_idx = 0u8;
    let mut best_dist = u32::MAX;

    for (i, color) in palette.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let idx = i as u8;
        if skip.is_some_and(|s| s == idx) {
            continue;
        }
        let dr = i32::from(r) - i32::from(color[0]);
        let dg = i32::from(g) - i32::from(color[1]);
        let db = i32::from(b) - i32::from(color[2]);
        #[allow(clippy::cast_sign_loss)]
        let dist = (dr * dr + dg * dg + db * db) as u32;
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }

    best_idx
}

/// Median-cut: recursively split color boxes along their widest channel.
fn cut_boxes(colors: &[[u8; 3]], max_palette: usize) -> Vec<[u8; 3]> {
    if max_palette == 0 {
        return vec![[0, 0, 0]];
    }

    let mut boxes: Vec<Vec<[u8; 3]>> = vec![colors.to_vec()];

    while boxes.len() < max_palette {
        // Find the box with the widest range to split.
        let (split_idx, _) = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .max_by_key(|(_, b)| box_range(b))
            .unwrap_or((0, &boxes[0]));

        if boxes[split_idx].len() <= 1 {
            break;
        }

        let b = boxes.remove(split_idx);
        let (a, c) = split_box(b);
        boxes.push(a);
        boxes.push(c);
    }

    // Average each box to get palette colors.
    boxes.iter().map(|b| average_color(b)).collect()
}

fn box_range(colors: &[[u8; 3]]) -> u32 {
    let (mut r_min, mut g_min, mut b_min) = (255u8, 255u8, 255u8);
    let (mut r_max, mut g_max, mut b_max) = (0u8, 0u8, 0u8);
    for c in colors {
        r_min = r_min.min(c[0]);
        g_min = g_min.min(c[1]);
        b_min = b_min.min(c[2]);
        r_max = r_max.max(c[0]);
        g_max = g_max.max(c[1]);
        b_max = b_max.max(c[2]);
    }
    let dr = u32::from(r_max) - u32::from(r_min);
    let dg = u32::from(g_max) - u32::from(g_min);
    let db = u32::from(b_max) - u32::from(b_min);
    dr.max(dg).max(db)
}

fn split_box(mut colors: Vec<[u8; 3]>) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    // Find widest channel.
    let (mut r_min, mut g_min, mut b_min) = (255u8, 255u8, 255u8);
    let (mut r_max, mut g_max, mut b_max) = (0u8, 0u8, 0u8);
    for c in &colors {
        r_min = r_min.min(c[0]);
        g_min = g_min.min(c[1]);
        b_min = b_min.min(c[2]);
        r_max = r_max.max(c[0]);
        g_max = g_max.max(c[1]);
        b_max = b_max.max(c[2]);
    }
    let dr = r_max - r_min;
    let dg = g_max - g_min;
    let db = b_max - b_min;

    let channel = if dr >= dg && dr >= db {
        0
    } else if dg >= db {
        1
    } else {
        2
    };

    colors.sort_unstable_by_key(|c| c[channel]);
    let mid = colors.len() / 2;
    let right = colors.split_off(mid);
    (colors, right)
}

fn average_color(colors: &[[u8; 3]]) -> [u8; 3] {
    if colors.is_empty() {
        return [0, 0, 0];
    }
    let (mut r, mut g, mut b) = (0u64, 0u64, 0u64);
    for c in colors {
        r += u64::from(c[0]);
        g += u64::from(c[1]);
        b += u64::from(c[2]);
    }
    let n = colors.len() as u64;
    #[allow(clippy::cast_possible_truncation)]
    [(r / n) as u8, (g / n) as u8, (b / n) as u8]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_simple() {
        // 4 red pixels.
        let rgba = vec![
            255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
        ];
        let q = median_cut(&rgba, 256);
        assert!(!q.palette.is_empty());
        assert_eq!(q.indices.len(), 4);
        // All should map to the same index.
        assert!(q.indices.iter().all(|&i| i == q.indices[0]));
    }

    #[test]
    fn quantize_with_transparency() {
        let mut rgba = vec![255, 0, 0, 255]; // opaque red
        rgba.extend_from_slice(&[0, 0, 0, 0]); // transparent
        let q = median_cut(&rgba, 256);
        assert!(q.transparent_index.is_some());
        // Transparent pixel should use the transparent index.
        assert_eq!(q.indices[1], q.transparent_index.unwrap());
    }

    #[test]
    fn quantize_multi_color() {
        let mut rgba = Vec::new();
        for _ in 0..100 {
            rgba.extend_from_slice(&[255, 0, 0, 255]); // red
        }
        for _ in 0..100 {
            rgba.extend_from_slice(&[0, 0, 255, 255]); // blue
        }
        let q = median_cut(&rgba, 4);
        assert!(q.palette.len() <= 4);
        assert_eq!(q.indices.len(), 200);
    }

    #[test]
    fn quantize_empty() {
        let q = median_cut(&[], 256);
        assert!(!q.palette.is_empty());
        assert!(q.indices.is_empty());
    }

    #[test]
    fn average_color_basic() {
        let colors = vec![[0, 0, 0], [100, 100, 100]];
        let avg = average_color(&colors);
        assert_eq!(avg, [50, 50, 50]);
    }
}
