//! Optical flow estimation algorithms.
//!
//! Provides dense and sparse optical flow computation between two frames,
//! including Lucas-Kanade and Farneback methods, plus flow visualization.

use crate::color;
use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};

/// Result of optical flow estimation between two frames.
///
/// Contains per-pixel horizontal and vertical displacement vectors.
///
/// # Examples
///
/// ```
/// # use scivex_image::optical_flow::OpticalFlowResult;
/// let flow = OpticalFlowResult {
///     flow_x: vec![vec![0.0; 4]; 3],
///     flow_y: vec![vec![0.0; 4]; 3],
///     width: 4,
///     height: 3,
/// };
/// assert_eq!(flow.width, 4);
/// assert_eq!(flow.height, 3);
/// ```
#[derive(Debug, Clone)]
pub struct OpticalFlowResult {
    /// Horizontal flow displacement `[height][width]`.
    pub flow_x: Vec<Vec<f64>>,
    /// Vertical flow displacement `[height][width]`.
    pub flow_y: Vec<Vec<f64>>,
    /// Width of the flow field.
    pub width: usize,
    /// Height of the flow field.
    pub height: usize,
}

/// Convert an image to a grayscale f64 2D buffer `[height][width]`.
fn to_gray_buffer(img: &Image<f32>) -> Result<Vec<Vec<f64>>> {
    let gray = if img.format() == PixelFormat::Gray {
        img.clone()
    } else {
        color::to_grayscale(img)?
    };
    let w = gray.width();
    let h = gray.height();
    let src = gray.as_slice();
    let mut buf = vec![vec![0.0f64; w]; h];
    for row in 0..h {
        for col in 0..w {
            buf[row][col] = f64::from(src[row * w + col]);
        }
    }
    Ok(buf)
}

/// Type alias for the triple of gradient buffers (Ix, Iy, It).
type GradientBuffers = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Compute spatial gradients Ix, Iy using Sobel-like 3x3 kernels, and temporal
/// gradient It = next - prev, all returned as `[height][width]` buffers.
fn compute_gradients(prev: &[Vec<f64>], next: &[Vec<f64>], h: usize, w: usize) -> GradientBuffers {
    let mut ix = vec![vec![0.0f64; w]; h];
    let mut iy = vec![vec![0.0f64; w]; h];
    let mut it = vec![vec![0.0f64; w]; h];

    for row in 1..h.saturating_sub(1) {
        for col in 1..w.saturating_sub(1) {
            // Sobel Ix: [-1 0 1; -2 0 2; -1 0 1] / 8
            let gx = -prev[row - 1][col - 1] + prev[row - 1][col + 1] - 2.0 * prev[row][col - 1]
                + 2.0 * prev[row][col + 1]
                - prev[row + 1][col - 1]
                + prev[row + 1][col + 1];
            ix[row][col] = gx / 8.0;

            // Sobel Iy: [-1 -2 -1; 0 0 0; 1 2 1] / 8
            let gy = -prev[row - 1][col - 1] - 2.0 * prev[row - 1][col] - prev[row - 1][col + 1]
                + prev[row + 1][col - 1]
                + 2.0 * prev[row + 1][col]
                + prev[row + 1][col + 1];
            iy[row][col] = gy / 8.0;

            it[row][col] = next[row][col] - prev[row][col];
        }
    }

    (ix, iy, it)
}

/// Compute sparse optical flow using the Lucas-Kanade method.
///
/// For each pixel, solves a 2x2 linear system over a local window of size
/// `window_size x window_size` to determine the displacement `(u, v)`.
///
/// Both images are converted to grayscale internally if needed.
///
/// # Arguments
///
/// * `prev` - The previous (first) frame.
/// * `next` - The next (second) frame.
/// * `window_size` - Side length of the local window (must be odd and >= 3).
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::optical_flow;
/// let prev = Image::from_raw(vec![0.5f32; 25], 5, 5, PixelFormat::Gray).unwrap();
/// let next = prev.clone();
/// let flow = optical_flow::lucas_kanade(&prev, &next, 5).unwrap();
/// assert_eq!(flow.width, 5);
/// assert_eq!(flow.height, 5);
/// ```
pub fn lucas_kanade(
    prev: &Image<f32>,
    next: &Image<f32>,
    window_size: usize,
) -> Result<OpticalFlowResult> {
    if prev.dimensions() != next.dimensions() {
        return Err(ImageError::InvalidParameter {
            name: "next",
            reason: "dimensions must match previous frame",
        });
    }
    if window_size < 3 || window_size.is_multiple_of(2) {
        return Err(ImageError::InvalidParameter {
            name: "window_size",
            reason: "must be odd and >= 3",
        });
    }

    let prev_buf = to_gray_buffer(prev)?;
    let next_buf = to_gray_buffer(next)?;
    let w = prev.width();
    let h = prev.height();
    let half = window_size / 2;

    let (ix, iy, it) = compute_gradients(&prev_buf, &next_buf, h, w);

    let mut flow_x = vec![vec![0.0f64; w]; h];
    let mut flow_y = vec![vec![0.0f64; w]; h];

    for row in 0..h {
        for col in 0..w {
            let mut sum_ix2 = 0.0f64;
            let mut sum_iy2 = 0.0f64;
            let mut sum_ixiy = 0.0f64;
            let mut sum_ixit = 0.0f64;
            let mut sum_iyit = 0.0f64;

            let r_start = row.saturating_sub(half);
            let r_end = (row + half + 1).min(h);
            let c_start = col.saturating_sub(half);
            let c_end = (col + half + 1).min(w);

            for wr in r_start..r_end {
                for wc in c_start..c_end {
                    let gx = ix[wr][wc];
                    let gy = iy[wr][wc];
                    let gt = it[wr][wc];
                    sum_ix2 += gx * gx;
                    sum_iy2 += gy * gy;
                    sum_ixiy += gx * gy;
                    sum_ixit += gx * gt;
                    sum_iyit += gy * gt;
                }
            }

            // Solve [sum_ix2 sum_ixiy; sum_ixiy sum_iy2] [u; v] = -[sum_ixit; sum_iyit]
            let det = sum_ix2 * sum_iy2 - sum_ixiy * sum_ixiy;
            if det.abs() > 1e-12 {
                let inv_det = 1.0 / det;
                flow_x[row][col] = -(sum_iy2 * sum_ixit - sum_ixiy * sum_iyit) * inv_det;
                flow_y[row][col] = -(sum_ix2 * sum_iyit - sum_ixiy * sum_ixit) * inv_det;
            }
        }
    }

    Ok(OpticalFlowResult {
        flow_x,
        flow_y,
        width: w,
        height: h,
    })
}

/// Downsample a 2D buffer by the given scale factor using area averaging.
fn downsample_buffer(buf: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
    let h = buf.len();
    let w = if h > 0 { buf[0].len() } else { 0 };
    let new_h = ((h as f64) * scale).max(1.0) as usize;
    let new_w = ((w as f64) * scale).max(1.0) as usize;
    let mut out = vec![vec![0.0f64; new_w]; new_h];
    for (r, out_row) in out.iter_mut().enumerate() {
        for (c, out_val) in out_row.iter_mut().enumerate() {
            let src_r = (r as f64 / scale).min((h - 1) as f64);
            let src_c = (c as f64 / scale).min((w - 1) as f64);
            let r0 = src_r as usize;
            let c0 = src_c as usize;
            let r1 = (r0 + 1).min(h - 1);
            let c1 = (c0 + 1).min(w - 1);
            let fr = src_r - r0 as f64;
            let fc = src_c - c0 as f64;
            *out_val = buf[r0][c0] * (1.0 - fr) * (1.0 - fc)
                + buf[r0][c1] * (1.0 - fr) * fc
                + buf[r1][c0] * fr * (1.0 - fc)
                + buf[r1][c1] * fr * fc;
        }
    }
    out
}

/// Upsample a 2D buffer to the target dimensions using bilinear interpolation.
fn upsample_buffer(buf: &[Vec<f64>], target_h: usize, target_w: usize) -> Vec<Vec<f64>> {
    let h = buf.len();
    let w = if h > 0 { buf[0].len() } else { 0 };
    if h == 0 || w == 0 {
        return vec![vec![0.0; target_w]; target_h];
    }
    let mut out = vec![vec![0.0f64; target_w]; target_h];
    let scale_r = if target_h > 1 {
        (h - 1) as f64 / (target_h - 1) as f64
    } else {
        0.0
    };
    let scale_c = if target_w > 1 {
        (w - 1) as f64 / (target_w - 1) as f64
    } else {
        0.0
    };
    for (r, out_row) in out.iter_mut().enumerate() {
        for (c, out_val) in out_row.iter_mut().enumerate() {
            let src_r = r as f64 * scale_r;
            let src_c = c as f64 * scale_c;
            let r0 = src_r as usize;
            let c0 = src_c as usize;
            let r1 = (r0 + 1).min(h - 1);
            let c1 = (c0 + 1).min(w - 1);
            let fr = src_r - r0 as f64;
            let fc = src_c - c0 as f64;
            *out_val = buf[r0][c0] * (1.0 - fr) * (1.0 - fc)
                + buf[r0][c1] * (1.0 - fr) * fc
                + buf[r1][c0] * fr * (1.0 - fc)
                + buf[r1][c1] * fr * fc;
        }
    }
    out
}

/// Warp a buffer according to the given flow field using bilinear interpolation.
fn warp_buffer(
    buf: &[Vec<f64>],
    flow_x: &[Vec<f64>],
    flow_y: &[Vec<f64>],
    h: usize,
    w: usize,
) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0f64; w]; h];
    for r in 0..h {
        for c in 0..w {
            let src_r = r as f64 + flow_y[r][c];
            let src_c = c as f64 + flow_x[r][c];
            if src_r >= 0.0 && src_r < (h - 1) as f64 && src_c >= 0.0 && src_c < (w - 1) as f64 {
                let r0 = src_r as usize;
                let c0 = src_c as usize;
                let r1 = r0 + 1;
                let c1 = c0 + 1;
                let fr = src_r - r0 as f64;
                let fc = src_c - c0 as f64;
                out[r][c] = buf[r0][c0] * (1.0 - fr) * (1.0 - fc)
                    + buf[r0][c1] * (1.0 - fr) * fc
                    + buf[r1][c0] * fr * (1.0 - fc)
                    + buf[r1][c1] * fr * fc;
            }
        }
    }
    out
}

/// Compute dense optical flow using a simplified Farneback-style pyramid approach.
///
/// Builds an image pyramid by repeatedly downsampling by `pyr_scale`, then
/// estimates flow coarse-to-fine using iterative Lucas-Kanade refinement at
/// each pyramid level.
///
/// # Arguments
///
/// * `prev` - The previous frame.
/// * `next` - The next frame.
/// * `pyr_scale` - Downsampling factor per level (typically 0.5).
/// * `levels` - Number of pyramid levels.
/// * `win_size` - Window size for local least-squares (must be odd and >= 3).
/// * `iterations` - Number of refinement iterations per level.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::optical_flow;
/// let prev = Image::from_raw(vec![0.5f32; 100], 10, 10, PixelFormat::Gray).unwrap();
/// let next = prev.clone();
/// let flow = optical_flow::farneback(&prev, &next, 0.5, 3, 5, 3).unwrap();
/// assert_eq!(flow.width, 10);
/// ```
#[allow(clippy::too_many_lines)]
pub fn farneback(
    prev: &Image<f32>,
    next: &Image<f32>,
    pyr_scale: f64,
    levels: usize,
    win_size: usize,
    iterations: usize,
) -> Result<OpticalFlowResult> {
    if prev.dimensions() != next.dimensions() {
        return Err(ImageError::InvalidParameter {
            name: "next",
            reason: "dimensions must match previous frame",
        });
    }
    if pyr_scale <= 0.0 || pyr_scale >= 1.0 {
        return Err(ImageError::InvalidParameter {
            name: "pyr_scale",
            reason: "must be between 0 and 1 exclusive",
        });
    }
    if levels == 0 {
        return Err(ImageError::InvalidParameter {
            name: "levels",
            reason: "must be at least 1",
        });
    }
    if win_size < 3 || win_size.is_multiple_of(2) {
        return Err(ImageError::InvalidParameter {
            name: "win_size",
            reason: "must be odd and >= 3",
        });
    }

    let prev_buf = to_gray_buffer(prev)?;
    let next_buf = to_gray_buffer(next)?;
    let orig_w = prev.width();
    let orig_h = prev.height();

    // Build pyramids (level 0 = finest = original).
    let mut prev_pyr = vec![prev_buf];
    let mut next_pyr = vec![next_buf];
    for _ in 1..levels {
        let p = downsample_buffer(prev_pyr.last().unwrap(), pyr_scale);
        let n = downsample_buffer(next_pyr.last().unwrap(), pyr_scale);
        prev_pyr.push(p);
        next_pyr.push(n);
    }

    // Start from the coarsest level.
    let coarsest = levels - 1;
    let ch = prev_pyr[coarsest].len();
    let cw = if ch > 0 {
        prev_pyr[coarsest][0].len()
    } else {
        0
    };

    let mut flow_x = vec![vec![0.0f64; cw]; ch];
    let mut flow_y = vec![vec![0.0f64; cw]; ch];

    // Coarse to fine.
    for level in (0..levels).rev() {
        let prev_level = &prev_pyr[level];
        let next_level = &next_pyr[level];
        let lh = prev_level.len();
        let lw = if lh > 0 { prev_level[0].len() } else { 0 };

        // Upsample flow from coarser level if not at coarsest.
        if level < coarsest {
            flow_x = upsample_buffer(&flow_x, lh, lw);
            flow_y = upsample_buffer(&flow_y, lh, lw);
            // Scale flow vectors to account for resolution change.
            // The flow magnitude stays the same in pixel units at this level
            // since upsample_buffer already interpolated the values.
        }

        let half = win_size / 2;

        for _iter in 0..iterations {
            // Warp next image toward prev using current flow estimate.
            let warped = warp_buffer(next_level, &flow_x, &flow_y, lh, lw);

            // Compute gradients between prev and warped.
            let (ix, iy, it) = compute_gradients(prev_level, &warped, lh, lw);

            // Iterative Lucas-Kanade refinement.
            for row in 0..lh {
                for col in 0..lw {
                    let mut sum_ix2 = 0.0f64;
                    let mut sum_iy2 = 0.0f64;
                    let mut sum_ixiy = 0.0f64;
                    let mut sum_ixit = 0.0f64;
                    let mut sum_iyit = 0.0f64;

                    let r_start = row.saturating_sub(half);
                    let r_end = (row + half + 1).min(lh);
                    let c_start = col.saturating_sub(half);
                    let c_end = (col + half + 1).min(lw);

                    for wr in r_start..r_end {
                        for wc in c_start..c_end {
                            let gx = ix[wr][wc];
                            let gy = iy[wr][wc];
                            let gt = it[wr][wc];
                            sum_ix2 += gx * gx;
                            sum_iy2 += gy * gy;
                            sum_ixiy += gx * gy;
                            sum_ixit += gx * gt;
                            sum_iyit += gy * gt;
                        }
                    }

                    let det = sum_ix2 * sum_iy2 - sum_ixiy * sum_ixiy;
                    if det.abs() > 1e-12 {
                        let inv_det = 1.0 / det;
                        let du = -(sum_iy2 * sum_ixit - sum_ixiy * sum_iyit) * inv_det;
                        let dv = -(sum_ix2 * sum_iyit - sum_ixiy * sum_ixit) * inv_det;
                        flow_x[row][col] += du;
                        flow_y[row][col] += dv;
                    }
                }
            }
        }
    }

    // The flow at level 0 should already be at original resolution.
    // If pyramid produced a slightly different size, upsample to exact original.
    if flow_x.len() != orig_h || (!flow_x.is_empty() && flow_x[0].len() != orig_w) {
        flow_x = upsample_buffer(&flow_x, orig_h, orig_w);
        flow_y = upsample_buffer(&flow_y, orig_h, orig_w);
    }

    Ok(OpticalFlowResult {
        flow_x,
        flow_y,
        width: orig_w,
        height: orig_h,
    })
}

/// Convert an optical flow field to an RGB color visualization.
///
/// Motion direction is mapped to hue and motion magnitude is mapped to
/// brightness, producing an intuitive color-coded image.
///
/// # Examples
///
/// ```
/// # use scivex_image::optical_flow::{OpticalFlowResult, flow_to_color};
/// let flow = OpticalFlowResult {
///     flow_x: vec![vec![1.0; 4]; 3],
///     flow_y: vec![vec![0.0; 4]; 3],
///     width: 4,
///     height: 3,
/// };
/// let img = flow_to_color(&flow).unwrap();
/// assert_eq!(img.dimensions(), (4, 3));
/// assert_eq!(img.channels(), 3);
/// ```
pub fn flow_to_color(flow: &OpticalFlowResult) -> Result<Image<f32>> {
    let w = flow.width;
    let h = flow.height;

    if w == 0 || h == 0 {
        return Err(ImageError::InvalidDimensions {
            width: w,
            height: h,
        });
    }

    // Find maximum magnitude for normalization.
    let mut max_mag = 0.0f64;
    for row in 0..h {
        for col in 0..w {
            let mag = (flow.flow_x[row][col].powi(2) + flow.flow_y[row][col].powi(2)).sqrt();
            if mag > max_mag {
                max_mag = mag;
            }
        }
    }
    if max_mag < 1e-12 {
        max_mag = 1.0;
    }

    let mut data = Vec::with_capacity(h * w * 3);

    for row in 0..h {
        for col in 0..w {
            let fx = flow.flow_x[row][col];
            let fy = flow.flow_y[row][col];
            let mag = (fx * fx + fy * fy).sqrt();
            let angle = fy.atan2(fx); // radians in [-pi, pi]

            // Map angle to hue [0, 360).
            let hue = (angle.to_degrees() + 360.0) % 360.0;
            let saturation = 1.0;
            let value = (mag / max_mag).min(1.0);

            // HSV to RGB conversion.
            let (r, g, b) = hsv_to_rgb(hue, saturation, value);
            data.push(r as f32);
            data.push(g as f32);
            data.push(b as f32);
        }
    }

    Image::from_raw(data, w, h, PixelFormat::Rgb)
}

/// Convert HSV (h in [0,360), s in [0,1], v in [0,1]) to RGB in [0,1].
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let (r1, g1, b1) = if h_prime < 1.0 {
        (c, x, 0.0)
    } else if h_prime < 2.0 {
        (x, c, 0.0)
    } else if h_prime < 3.0 {
        (0.0, c, x)
    } else if h_prime < 4.0 {
        (0.0, x, c)
    } else if h_prime < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = v - c;
    (r1 + m, g1 + m, b1 + m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optical_flow_result_creation() {
        let flow = OpticalFlowResult {
            flow_x: vec![vec![1.0, 2.0]; 3],
            flow_y: vec![vec![-1.0, 0.5]; 3],
            width: 2,
            height: 3,
        };
        assert_eq!(flow.width, 2);
        assert_eq!(flow.height, 3);
        assert_eq!(flow.flow_x.len(), 3);
        assert_eq!(flow.flow_x[0].len(), 2);
        assert!((flow.flow_x[0][0] - 1.0).abs() < 1e-10);
        assert!((flow.flow_y[2][1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_lucas_kanade_zero_flow() {
        let data = vec![0.5f32; 10 * 10];
        let prev = Image::from_raw(data.clone(), 10, 10, PixelFormat::Gray).unwrap();
        let next = Image::from_raw(data, 10, 10, PixelFormat::Gray).unwrap();
        let flow = lucas_kanade(&prev, &next, 5).unwrap();
        assert_eq!(flow.width, 10);
        assert_eq!(flow.height, 10);
        for row in 0..flow.height {
            for col in 0..flow.width {
                assert!(
                    flow.flow_x[row][col].abs() < 1e-6,
                    "flow_x[{row}][{col}] = {} (expected ~0)",
                    flow.flow_x[row][col]
                );
                assert!(
                    flow.flow_y[row][col].abs() < 1e-6,
                    "flow_y[{row}][{col}] = {} (expected ~0)",
                    flow.flow_y[row][col]
                );
            }
        }
    }

    #[test]
    fn test_lucas_kanade_horizontal_shift() {
        // Use a sinusoidal pattern with varying gradients in both x and y,
        // making the 2x2 structure tensor well-conditioned at most pixels.
        let w = 40;
        let h = 40;
        let mut prev_data = vec![0.0f32; w * h];
        let mut next_data = vec![0.0f32; w * h];
        let pi = std::f32::consts::PI;

        for row in 0..h {
            for col in 0..w {
                let x = col as f32 / w as f32;
                let y = row as f32 / h as f32;
                let val = 0.5 + 0.3 * (2.0 * pi * x * 3.0).sin() * (2.0 * pi * y * 2.0).cos();
                prev_data[row * w + col] = val;
                let src_col = if col > 0 { col - 1 } else { 0 };
                let sx = src_col as f32 / w as f32;
                let shifted = 0.5 + 0.3 * (2.0 * pi * sx * 3.0).sin() * (2.0 * pi * y * 2.0).cos();
                next_data[row * w + col] = shifted;
            }
        }

        let prev = Image::from_raw(prev_data, w, h, PixelFormat::Gray).unwrap();
        let next = Image::from_raw(next_data, w, h, PixelFormat::Gray).unwrap();
        let flow = lucas_kanade(&prev, &next, 7).unwrap();

        // Check that some interior pixels detect nonzero flow magnitude.
        let border = 5;
        let mut nonzero_count = 0;
        let mut total = 0;
        for row in border..h - border {
            for col in border..w - border {
                total += 1;
                let mag = (flow.flow_x[row][col].powi(2) + flow.flow_y[row][col].powi(2)).sqrt();
                if mag > 0.01 {
                    nonzero_count += 1;
                }
            }
        }
        assert!(
            nonzero_count as f64 / total as f64 > 0.2,
            "Only {nonzero_count}/{total} pixels detected nonzero flow"
        );
    }

    #[test]
    fn test_farneback_zero_flow() {
        let data = vec![0.5f32; 10 * 10];
        let prev = Image::from_raw(data.clone(), 10, 10, PixelFormat::Gray).unwrap();
        let next = Image::from_raw(data, 10, 10, PixelFormat::Gray).unwrap();
        let flow = farneback(&prev, &next, 0.5, 2, 5, 2).unwrap();
        assert_eq!(flow.width, 10);
        assert_eq!(flow.height, 10);
        for row in 0..flow.height {
            for col in 0..flow.width {
                assert!(
                    flow.flow_x[row][col].abs() < 1e-6,
                    "flow_x[{row}][{col}] = {} (expected ~0)",
                    flow.flow_x[row][col]
                );
                assert!(
                    flow.flow_y[row][col].abs() < 1e-6,
                    "flow_y[{row}][{col}] = {} (expected ~0)",
                    flow.flow_y[row][col]
                );
            }
        }
    }

    #[test]
    fn test_flow_to_color_shape() {
        let flow = OpticalFlowResult {
            flow_x: vec![vec![1.0, -1.0, 0.0, 2.0]; 3],
            flow_y: vec![vec![0.0, 1.0, -1.0, 0.5]; 3],
            width: 4,
            height: 3,
        };
        let img = flow_to_color(&flow).unwrap();
        assert_eq!(img.dimensions(), (4, 3));
        assert_eq!(img.channels(), 3);
        assert_eq!(img.format(), PixelFormat::Rgb);

        // Verify pixel values are in [0, 1].
        let slice = img.as_slice();
        for &v in slice {
            assert!(
                (0.0..=1.0).contains(&v),
                "pixel value {v} out of [0, 1] range"
            );
        }
    }
}
