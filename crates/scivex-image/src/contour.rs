//! Contour detection and geometric analysis.
//!
//! Provides binary-image contour tracing via Moore neighborhood traversal,
//! plus area and perimeter computation.

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};

/// A contour is an ordered sequence of boundary pixel coordinates `(row, col)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contour {
    /// Ordered boundary points as `(row, col)`.
    pub points: Vec<(usize, usize)>,
}

/// Find contours in a grayscale `u8` image by thresholding and Moore
/// neighborhood boundary tracing.
///
/// Pixels with value `>= threshold` are considered foreground. Each connected
/// foreground boundary is returned as a separate [`Contour`].
#[allow(clippy::too_many_lines)]
pub fn find_contours(img: &Image<u8>, threshold: u8) -> Result<Vec<Contour>> {
    if img.format() != PixelFormat::Gray {
        return Err(ImageError::UnsupportedChannels {
            channels: img.channels(),
        });
    }

    let (w, h) = img.dimensions();
    let src = img.as_slice();

    // Build binary mask: true = foreground.
    let binary: Vec<bool> = src.iter().map(|&v| v >= threshold).collect();

    // Track which foreground pixels have already been assigned to a contour
    // boundary so we don't trace the same boundary twice.
    let mut visited = vec![false; h * w];

    let mut contours = Vec::new();

    // Scan top-to-bottom, left-to-right for unvisited foreground pixels that
    // sit on a boundary (have at least one background 4-neighbor or are on the
    // image border).
    for row in 0..h {
        for col in 0..w {
            let idx = row * w + col;
            if !binary[idx] || visited[idx] {
                continue;
            }
            if !is_boundary(row, col, w, h, &binary) {
                continue;
            }

            // Trace this boundary using Moore neighborhood tracing.
            let contour = trace_boundary(row, col, w, h, &binary, &mut visited);
            if !contour.points.is_empty() {
                contours.push(contour);
            }
        }
    }

    Ok(contours)
}

/// Check if a foreground pixel is on the boundary (has a background 4-neighbor
/// or is on the image edge).
fn is_boundary(row: usize, col: usize, w: usize, h: usize, binary: &[bool]) -> bool {
    if row == 0 || row == h - 1 || col == 0 || col == w - 1 {
        return true;
    }
    // 4-connected neighbors
    let neighbors = [
        (row.wrapping_sub(1), col),
        (row + 1, col),
        (row, col.wrapping_sub(1)),
        (row, col + 1),
    ];
    for &(nr, nc) in &neighbors {
        if nr >= h || nc >= w || !binary[nr * w + nc] {
            return true;
        }
    }
    false
}

/// Moore neighborhood boundary tracing starting from `(start_row, start_col)`.
///
/// The 8-connected Moore neighborhood is traversed clockwise. We enter from
/// the left (the scan direction), so the initial "backtrack" direction is west.
fn trace_boundary(
    start_row: usize,
    start_col: usize,
    w: usize,
    h: usize,
    binary: &[bool],
    visited: &mut [bool],
) -> Contour {
    // Moore neighborhood offsets (clockwise starting from up-left):
    // 0=NW, 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W
    const DIRS: [(isize, isize); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ];

    let mut points = Vec::new();
    let mut cur_row = start_row;
    let mut cur_col = start_col;
    // We entered from the left, so backtrack direction is west (index 7).
    let mut backtrack_dir: usize = 7;

    let max_steps = w * h * 2;

    for _ in 0..max_steps {
        visited[cur_row * w + cur_col] = true;
        points.push((cur_row, cur_col));

        // Search clockwise from the cell after the backtrack direction.
        let start_dir = (backtrack_dir + 1) % 8;
        let mut found = false;

        for i in 0..8 {
            let d = (start_dir + i) % 8;
            let nr = cur_row.cast_signed() + DIRS[d].0;
            let nc = cur_col.cast_signed() + DIRS[d].1;

            if nr >= 0 && nr < h.cast_signed() && nc >= 0 && nc < w.cast_signed() {
                let nr = nr.cast_unsigned();
                let nc = nc.cast_unsigned();
                if binary[nr * w + nc] {
                    // The backtrack direction is opposite of where we came from.
                    backtrack_dir = (d + 4) % 8;
                    cur_row = nr;
                    cur_col = nc;
                    found = true;
                    break;
                }
            }
        }

        if !found {
            // Isolated pixel.
            break;
        }

        // If we returned to the start, stop.
        if cur_row == start_row && cur_col == start_col {
            break;
        }
    }

    Contour { points }
}

/// Compute the area enclosed by a contour using the shoelace formula.
///
/// Treats points as `(x=col, y=row)` coordinates. Returns the absolute area.
pub fn contour_area(contour: &Contour) -> f64 {
    let pts = &contour.points;
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }

    let mut sum: f64 = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let (r0, c0) = pts[i];
        let (r1, c1) = pts[j];
        // Shoelace: cross product of (col, row) vectors.
        sum += c0 as f64 * r1 as f64;
        sum -= c1 as f64 * r0 as f64;
    }

    sum.abs() / 2.0
}

/// Compute the perimeter of a contour as the sum of Euclidean distances
/// between consecutive points.
pub fn contour_perimeter(contour: &Contour) -> f64 {
    let pts = &contour.points;
    let n = pts.len();
    if n < 2 {
        return 0.0;
    }

    let mut total: f64 = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let dr = pts[j].0 as f64 - pts[i].0 as f64;
        let dc = pts[j].1 as f64 - pts[i].1 as f64;
        total += (dr * dr + dc * dc).sqrt();
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a 10x10 image with a filled 4x4 rectangle at rows 3..7, cols 3..7.
    fn make_rect_image() -> Image<u8> {
        let mut data = vec![0u8; 100];
        for r in 3..7 {
            for c in 3..7 {
                data[r * 10 + c] = 255;
            }
        }
        Image::from_raw(data, 10, 10, PixelFormat::Gray).unwrap()
    }

    #[test]
    fn test_find_rectangle_contour() {
        let img = make_rect_image();
        let contours = find_contours(&img, 128).unwrap();
        assert!(!contours.is_empty(), "should find at least one contour");
        // All contour points should be within the rectangle bounds.
        for pt in &contours[0].points {
            assert!(pt.0 >= 3 && pt.0 < 7, "row {} out of bounds", pt.0);
            assert!(pt.1 >= 3 && pt.1 < 7, "col {} out of bounds", pt.1);
        }
    }

    #[test]
    fn test_contour_area_known_shape() {
        // A simple square contour: 4 corners of a 4x4 pixel square.
        let contour = Contour {
            points: vec![(0, 0), (0, 4), (4, 4), (4, 0)],
        };
        let area = contour_area(&contour);
        assert!((area - 16.0).abs() < 1e-6, "area should be 16, got {area}");
    }

    #[test]
    fn test_contour_perimeter() {
        // A square with side 3: (0,0) -> (0,3) -> (3,3) -> (3,0)
        let contour = Contour {
            points: vec![(0, 0), (0, 3), (3, 3), (3, 0)],
        };
        let perim = contour_perimeter(&contour);
        // Each side = 3, four sides = 12.
        assert!(
            (perim - 12.0).abs() < 1e-6,
            "perimeter should be 12, got {perim}"
        );
    }

    #[test]
    fn test_empty_image_no_contours() {
        let img = Image::<u8>::new(10, 10, PixelFormat::Gray).unwrap();
        let contours = find_contours(&img, 128).unwrap();
        assert!(contours.is_empty(), "empty image should yield no contours");
    }
}
