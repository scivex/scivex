//! Feature detection: Harris corners and FAST keypoints.

use scivex_core::Float;

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};

/// A detected corner / keypoint.
#[derive(Debug, Clone)]
pub struct Corner {
    /// Row (y) coordinate.
    pub row: usize,
    /// Column (x) coordinate.
    pub col: usize,
    /// Detector response value.
    pub response: f64,
}

/// Detect corners with the Harris corner detector.
///
/// 1. Compute image gradients `Ix`, `Iy` via 3x3 Sobel operators.
/// 2. For each pixel build the structure tensor `M` over a `block_size x block_size`
///    window: `M = [[sum Ix^2, sum IxIy], [sum IxIy, sum Iy^2]]`.
/// 3. Compute response `R = det(M) - k * trace(M)^2`.
/// 4. Suppress responses below `threshold` and apply 3x3 non-maximum suppression.
///
/// Only grayscale images are accepted (convert to gray first).
#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_possible_wrap)]
pub fn harris_corners<T: Float>(
    img: &Image<T>,
    k: T,
    threshold: T,
    block_size: usize,
) -> Result<Vec<Corner>> {
    if img.format() != PixelFormat::Gray {
        return Err(ImageError::UnsupportedChannels {
            channels: img.channels(),
        });
    }
    if block_size == 0 {
        return Err(ImageError::InvalidParameter {
            name: "block_size",
            reason: "must be positive",
        });
    }

    let (w, h) = img.dimensions();
    let src = img.as_slice();

    // Helper to read a grayscale pixel (clamped at borders).
    let px = |r: isize, c: isize| -> T {
        let rr = r.clamp(0, (h as isize) - 1) as usize;
        let cc = c.clamp(0, (w as isize) - 1) as usize;
        src[rr * w + cc]
    };

    // Sobel gradients
    let two = T::one() + T::one();
    let mut ix = vec![T::zero(); h * w];
    let mut iy = vec![T::zero(); h * w];

    for row in 0..h {
        for col in 0..w {
            let r = row as isize;
            let c = col as isize;
            // Sobel X: [[-1,0,1],[-2,0,2],[-1,0,1]]
            let gx = -px(r - 1, c - 1) + px(r - 1, c + 1) - two * px(r, c - 1) + two * px(r, c + 1)
                - px(r + 1, c - 1)
                + px(r + 1, c + 1);
            // Sobel Y: [[-1,-2,-1],[0,0,0],[1,2,1]]
            let gy = -px(r - 1, c - 1) - two * px(r - 1, c) - px(r - 1, c + 1)
                + px(r + 1, c - 1)
                + two * px(r + 1, c)
                + px(r + 1, c + 1);
            ix[row * w + col] = gx;
            iy[row * w + col] = gy;
        }
    }

    // Structure tensor + Harris response
    let half = (block_size / 2) as isize;
    let mut response = vec![T::zero(); h * w];

    for row in 0..h {
        for col in 0..w {
            let mut sxx = T::zero();
            let mut syy = T::zero();
            let mut sxy = T::zero();

            for dr in -half..=half {
                for dc in -half..=half {
                    let rr = (row as isize + dr).clamp(0, (h as isize) - 1) as usize;
                    let cc = (col as isize + dc).clamp(0, (w as isize) - 1) as usize;
                    let gx = ix[rr * w + cc];
                    let gy = iy[rr * w + cc];
                    sxx += gx * gx;
                    syy += gy * gy;
                    sxy += gx * gy;
                }
            }

            let det = sxx * syy - sxy * sxy;
            let trace = sxx + syy;
            response[row * w + col] = det - k * trace * trace;
        }
    }

    // Non-maximum suppression (3x3) + threshold
    let mut corners = Vec::new();
    for row in 0..h {
        for col in 0..w {
            let r = response[row * w + col];
            if r <= threshold {
                continue;
            }
            let mut is_max = true;
            'nms: for dr in -1isize..=1 {
                for dc in -1isize..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = row as isize + dr;
                    let nc = col as isize + dc;
                    if nr >= 0
                        && nr < h as isize
                        && nc >= 0
                        && nc < w as isize
                        && response[nr as usize * w + nc as usize] > r
                    {
                        is_max = false;
                        break 'nms;
                    }
                }
            }
            if is_max {
                corners.push(Corner {
                    row,
                    col,
                    response: r.to_f64(),
                });
            }
        }
    }

    Ok(corners)
}

/// FAST-9 feature detector on a grayscale `u8` image.
///
/// Uses a Bresenham circle of 16 pixels around each candidate. A corner is
/// detected when at least 9 contiguous pixels on the circle are all brighter
/// than `center + threshold` or all darker than `center - threshold`.
///
/// If `nonmax` is true, 3x3 non-maximum suppression is applied using the
/// number of contiguous pixels as the score.
#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_wrap)]
pub fn fast_features(img: &Image<u8>, threshold: u8, nonmax: bool) -> Result<Vec<Corner>> {
    // Bresenham circle of radius 3 (16 pixels), clockwise from top.
    const CIRCLE: [(isize, isize); 16] = [
        (-3, 0),
        (-3, 1),
        (-2, 2),
        (-1, 3),
        (0, 3),
        (1, 3),
        (2, 2),
        (3, 1),
        (3, 0),
        (3, -1),
        (2, -2),
        (1, -3),
        (0, -3),
        (-1, -3),
        (-2, -2),
        (-3, -1),
    ];

    if img.format() != PixelFormat::Gray {
        return Err(ImageError::UnsupportedChannels {
            channels: img.channels(),
        });
    }

    let (w, h) = img.dimensions();
    let src = img.as_slice();

    // Score buffer for NMS.
    let mut scores = vec![0u16; h * w];

    for row in 3..h.saturating_sub(3) {
        for col in 3..w.saturating_sub(3) {
            let center = i16::from(src[row * w + col]);
            let hi = center + i16::from(threshold);
            let lo = center - i16::from(threshold);

            // Classify each circle pixel.
            let mut brighter = [false; 16];
            let mut darker = [false; 16];
            for (i, &(dr, dc)) in CIRCLE.iter().enumerate() {
                let v =
                    i16::from(src[(row as isize + dr) as usize * w + (col as isize + dc) as usize]);
                brighter[i] = v > hi;
                darker[i] = v < lo;
            }

            // Check for 9 contiguous brighter or darker.
            let score_b = max_contiguous(&brighter);
            let score_d = max_contiguous(&darker);
            let score = score_b.max(score_d);

            if score >= 9 {
                scores[row * w + col] = score;
            }
        }
    }

    let mut corners = Vec::new();

    if nonmax {
        for row in 3..h.saturating_sub(3) {
            for col in 3..w.saturating_sub(3) {
                let s = scores[row * w + col];
                if s == 0 {
                    continue;
                }
                let mut is_max = true;
                'nms: for dr in -1isize..=1 {
                    for dc in -1isize..=1 {
                        if dr == 0 && dc == 0 {
                            continue;
                        }
                        let nr = (row as isize + dr) as usize;
                        let nc = (col as isize + dc) as usize;
                        if scores[nr * w + nc] > s {
                            is_max = false;
                            break 'nms;
                        }
                    }
                }
                if is_max {
                    corners.push(Corner {
                        row,
                        col,
                        response: f64::from(s),
                    });
                }
            }
        }
    } else {
        for row in 3..h.saturating_sub(3) {
            for col in 3..w.saturating_sub(3) {
                let s = scores[row * w + col];
                if s > 0 {
                    corners.push(Corner {
                        row,
                        col,
                        response: f64::from(s),
                    });
                }
            }
        }
    }

    Ok(corners)
}

/// Return the maximum number of contiguous `true` values in a circular array
/// of 16 elements.
fn max_contiguous(flags: &[bool; 16]) -> u16 {
    if flags.iter().all(|&f| !f) {
        return 0;
    }
    let mut best: u16 = 0;
    let mut run: u16 = 0;
    // Check the doubled ring to handle wrap-around.
    for i in 0..32 {
        if flags[i % 16] {
            run += 1;
            if run > best {
                best = run;
            }
        } else {
            run = 0;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple 16x16 checkerboard pattern useful for Harris detection.
    fn make_checkerboard() -> Image<f32> {
        let mut data = vec![0.0f32; 16 * 16];
        for r in 0..16 {
            for c in 0..16 {
                if (r / 4 + c / 4) % 2 == 0 {
                    data[r * 16 + c] = 1.0;
                }
            }
        }
        Image::from_raw(data, 16, 16, PixelFormat::Gray).unwrap()
    }

    #[test]
    fn test_harris_on_checkerboard() {
        let img = make_checkerboard();
        let corners = harris_corners(&img, 0.04, 0.0, 3).unwrap();
        // A checkerboard has real corners at the block junctions; we should
        // detect some.
        assert!(
            !corners.is_empty(),
            "Harris should detect corners on a checkerboard"
        );
    }

    #[test]
    fn test_fast_on_simple_corner() {
        // Create a 20x20 image with an L-shaped bright region.
        let mut data = vec![0u8; 20 * 20];
        // Horizontal bar
        for c in 5..15 {
            for r in 5..8 {
                data[r * 20 + c] = 200;
            }
        }
        // Vertical bar
        for r in 5..15 {
            for c in 5..8 {
                data[r * 20 + c] = 200;
            }
        }
        let img = Image::from_raw(data, 20, 20, PixelFormat::Gray).unwrap();
        let corners = fast_features(&img, 50, true).unwrap();
        // There should be at least one detected feature around the L corner.
        assert!(
            !corners.is_empty(),
            "FAST should detect features on an L-shape"
        );
    }

    #[test]
    fn test_harris_empty_image() {
        // Uniform image should yield no corners above a reasonable threshold.
        let data = vec![0.5f32; 10 * 10];
        let img = Image::from_raw(data, 10, 10, PixelFormat::Gray).unwrap();
        let corners = harris_corners(&img, 0.04, 0.001, 3).unwrap();
        assert!(
            corners.is_empty(),
            "uniform image should have no Harris corners"
        );
    }

    #[test]
    fn test_fast_threshold_filtering() {
        // With a very high threshold no features should pass.
        let mut data = vec![128u8; 20 * 20];
        // Small bright spot.
        data[10 * 20 + 10] = 200;
        let img = Image::from_raw(data, 20, 20, PixelFormat::Gray).unwrap();
        let corners = fast_features(&img, 250, false).unwrap();
        assert!(
            corners.is_empty(),
            "very high threshold should suppress all features"
        );
    }
}
