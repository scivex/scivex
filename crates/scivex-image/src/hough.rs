//! Hough transform for line and circle detection.
//!
//! Operates on binary edge images (grayscale `u8`). Non-zero pixels are
//! treated as edge points.

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};

/// A line detected by the Hough line transform, represented in polar form.
#[derive(Debug, Clone)]
pub struct HoughLine {
    /// Perpendicular distance from the origin to the line.
    pub rho: f64,
    /// Angle of the perpendicular in radians (0 to PI).
    pub theta: f64,
    /// Number of accumulator votes.
    pub votes: usize,
}

/// A circle detected by the Hough circle transform.
#[derive(Debug, Clone)]
pub struct HoughCircle {
    /// Row coordinate of the center.
    pub center_row: usize,
    /// Column coordinate of the center.
    pub center_col: usize,
    /// Radius in pixels.
    pub radius: usize,
    /// Number of accumulator votes.
    pub votes: usize,
}

/// Detect lines in a binary edge image using the Hough transform.
///
/// # Algorithm
///
/// 1. Create an accumulator array parameterised by `(rho, theta)`.
///    `rho` ranges from `-diagonal` to `+diagonal` with step `rho_resolution`;
///    `theta` ranges from `0` to `PI` with step `theta_resolution`.
/// 2. For every non-zero (edge) pixel `(col, row)`, iterate over all `theta`
///    values and compute `rho = col * cos(theta) + row * sin(theta)`.
///    Increment the corresponding accumulator cell.
/// 3. Extract all cells whose vote count meets or exceeds `threshold`.
///
/// The input image must be single-channel grayscale.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_wrap)]
pub fn hough_lines(
    edge_img: &Image<u8>,
    rho_resolution: f64,
    theta_resolution: f64,
    threshold: usize,
) -> Result<Vec<HoughLine>> {
    if edge_img.format() != PixelFormat::Gray {
        return Err(ImageError::UnsupportedChannels {
            channels: edge_img.channels(),
        });
    }
    if rho_resolution <= 0.0 {
        return Err(ImageError::InvalidParameter {
            name: "rho_resolution",
            reason: "must be positive",
        });
    }
    if theta_resolution <= 0.0 {
        return Err(ImageError::InvalidParameter {
            name: "theta_resolution",
            reason: "must be positive",
        });
    }

    let (w, h) = edge_img.dimensions();
    let diagonal = ((w * w + h * h) as f64).sqrt();

    // Build theta lookup table.
    let num_theta = (core::f64::consts::PI / theta_resolution).ceil() as usize;
    let cos_table: Vec<f64> = (0..num_theta)
        .map(|i| (i as f64 * theta_resolution).cos())
        .collect();
    let sin_table: Vec<f64> = (0..num_theta)
        .map(|i| (i as f64 * theta_resolution).sin())
        .collect();

    // Rho ranges from -diagonal to +diagonal.
    let num_rho = ((2.0 * diagonal) / rho_resolution).ceil() as usize + 1;
    let rho_offset = diagonal; // shift so index 0 corresponds to -diagonal

    // Accumulator: rho × theta
    let mut accum = vec![0usize; num_rho * num_theta];

    let src = edge_img.as_slice();

    for row in 0..h {
        for col in 0..w {
            if src[row * w + col] == 0 {
                continue;
            }
            let x = col as f64;
            let y = row as f64;
            for t in 0..num_theta {
                let rho = x * cos_table[t] + y * sin_table[t];
                let rho_idx = ((rho + rho_offset) / rho_resolution).round() as usize;
                if rho_idx < num_rho {
                    accum[rho_idx * num_theta + t] += 1;
                }
            }
        }
    }

    // Extract peaks.
    let mut lines = Vec::new();
    for ri in 0..num_rho {
        for ti in 0..num_theta {
            let votes = accum[ri * num_theta + ti];
            if votes >= threshold {
                let rho = ri as f64 * rho_resolution - rho_offset;
                let theta = ti as f64 * theta_resolution;
                lines.push(HoughLine { rho, theta, votes });
            }
        }
    }

    lines.sort_by(|a, b| b.votes.cmp(&a.votes));
    Ok(lines)
}

/// Detect circles in a binary edge image using the Hough circle transform.
///
/// # Algorithm
///
/// For each candidate radius `r` in `[min_radius, max_radius]`:
/// 1. Create a 2-D accumulator of size `height × width`.
/// 2. For every non-zero (edge) pixel `(col, row)`, sweep `theta` from 0 to
///    360 degrees in steps of roughly 5° and compute the candidate centre
///    `(a, b) = (row - r*sin(theta), col - r*cos(theta))`.
///    Increment `accumulator[a][b]`.
/// 3. Extract all cells whose vote count meets or exceeds `threshold`.
///
/// The input image must be single-channel grayscale.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::too_many_lines)]
pub fn hough_circles(
    edge_img: &Image<u8>,
    min_radius: usize,
    max_radius: usize,
    threshold: usize,
) -> Result<Vec<HoughCircle>> {
    if edge_img.format() != PixelFormat::Gray {
        return Err(ImageError::UnsupportedChannels {
            channels: edge_img.channels(),
        });
    }
    if min_radius > max_radius {
        return Err(ImageError::InvalidParameter {
            name: "min_radius",
            reason: "must not exceed max_radius",
        });
    }

    let (w, h) = edge_img.dimensions();
    let src = edge_img.as_slice();

    // Precompute angle table (~5° steps → 72 angles).
    let num_angles: usize = 72;
    let angle_step = 2.0 * core::f64::consts::PI / num_angles as f64;
    let cos_table: Vec<f64> = (0..num_angles)
        .map(|i| (i as f64 * angle_step).cos())
        .collect();
    let sin_table: Vec<f64> = (0..num_angles)
        .map(|i| (i as f64 * angle_step).sin())
        .collect();

    let mut circles = Vec::new();

    for r in min_radius..=max_radius {
        let rf = r as f64;
        let mut accum = vec![0usize; h * w];

        for row in 0..h {
            for col in 0..w {
                if src[row * w + col] == 0 {
                    continue;
                }
                for t in 0..num_angles {
                    let a = row as f64 - rf * sin_table[t];
                    let b = col as f64 - rf * cos_table[t];
                    let ai = a.round() as isize;
                    let bi = b.round() as isize;
                    if ai >= 0 && ai < h as isize && bi >= 0 && bi < w as isize {
                        accum[ai as usize * w + bi as usize] += 1;
                    }
                }
            }
        }

        for row in 0..h {
            for col in 0..w {
                let votes = accum[row * w + col];
                if votes >= threshold {
                    circles.push(HoughCircle {
                        center_row: row,
                        center_col: col,
                        radius: r,
                        votes,
                    });
                }
            }
        }
    }

    circles.sort_by(|a, b| b.votes.cmp(&a.votes));
    Ok(circles)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a grayscale image with a horizontal line at the given row.
    fn make_horizontal_line(width: usize, height: usize, line_row: usize) -> Image<u8> {
        let mut data = vec![0u8; width * height];
        for col in 0..width {
            data[line_row * width + col] = 255;
        }
        Image::from_raw(data, width, height, PixelFormat::Gray).unwrap()
    }

    /// Create a grayscale image with a vertical line at the given column.
    fn make_vertical_line(width: usize, height: usize, line_col: usize) -> Image<u8> {
        let mut data = vec![0u8; width * height];
        for row in 0..height {
            data[row * width + line_col] = 255;
        }
        Image::from_raw(data, width, height, PixelFormat::Gray).unwrap()
    }

    #[test]
    fn test_hough_lines_horizontal() {
        let img = make_horizontal_line(50, 50, 25);
        let lines = hough_lines(&img, 1.0, core::f64::consts::PI / 180.0, 40).unwrap();
        assert!(!lines.is_empty(), "should detect the horizontal line");

        // The strongest line should have theta near PI/2 (horizontal line
        // has rho = y * sin(theta), maximised when theta = PI/2).
        let best = &lines[0];
        let half_pi = core::f64::consts::FRAC_PI_2;
        assert!(
            (best.theta - half_pi).abs() < 0.1,
            "theta should be near PI/2 for a horizontal line, got {}",
            best.theta
        );
        // rho should be approximately the row index (25).
        assert!(
            (best.rho - 25.0).abs() < 2.0,
            "rho should be near 25 for the horizontal line, got {}",
            best.rho
        );
    }

    #[test]
    fn test_hough_lines_vertical() {
        let img = make_vertical_line(50, 50, 20);
        let lines = hough_lines(&img, 1.0, core::f64::consts::PI / 180.0, 40).unwrap();
        assert!(!lines.is_empty(), "should detect the vertical line");

        // A vertical line at col=20 has theta near 0 and rho near 20.
        let best = &lines[0];
        assert!(
            best.theta < 0.1 || best.theta > core::f64::consts::PI - 0.1,
            "theta should be near 0 or PI for a vertical line, got {}",
            best.theta
        );
        assert!(
            best.rho.abs() - 20.0 < 2.0 || (best.rho + 20.0).abs() < 2.0,
            "rho magnitude should be near 20 for the vertical line, got {}",
            best.rho
        );
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn test_hough_circles_drawn_circle() {
        // Draw a circle of radius 10 centred at (25, 25) on a 50×50 image.
        let mut img = Image::<u8>::new(50, 50, PixelFormat::Gray).unwrap();
        crate::draw::draw_circle(&mut img, 25, 25, 10, &[255]);

        let circles = hough_circles(&img, 8, 12, 20).unwrap();
        assert!(!circles.is_empty(), "should detect the drawn circle");

        // The best detection should be close to the true centre and radius.
        let best = &circles[0];
        assert!(
            (best.center_row as isize - 25_isize).abs() <= 2,
            "center_row should be near 25, got {}",
            best.center_row
        );
        assert!(
            (best.center_col as isize - 25_isize).abs() <= 2,
            "center_col should be near 25, got {}",
            best.center_col
        );
        assert!(
            (best.radius as isize - 10_isize).abs() <= 2,
            "radius should be near 10, got {}",
            best.radius
        );
    }

    #[test]
    fn test_empty_edge_image() {
        let img = Image::<u8>::new(20, 20, PixelFormat::Gray).unwrap();

        let lines = hough_lines(&img, 1.0, core::f64::consts::PI / 180.0, 1).unwrap();
        assert!(lines.is_empty(), "empty image should yield no lines");

        let circles = hough_circles(&img, 3, 10, 1).unwrap();
        assert!(circles.is_empty(), "empty image should yield no circles");
    }
}
