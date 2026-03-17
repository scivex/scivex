//! Image segmentation algorithms: connected components, region growing, watershed.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};

// ---------------------------------------------------------------------------
// Union-Find
// ---------------------------------------------------------------------------

/// Disjoint-set / union-find used by connected-component labeling.
struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            self.parent[x as usize] = self.parent[self.parent[x as usize] as usize];
            x = self.parent[x as usize];
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra as usize].cmp(&self.rank[rb as usize]) {
            std::cmp::Ordering::Less => self.parent[ra as usize] = rb,
            std::cmp::Ordering::Greater => self.parent[rb as usize] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb as usize] = ra;
                self.rank[ra as usize] += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Connected Components Labeling
// ---------------------------------------------------------------------------

/// Label connected components in a grayscale image using a two-pass algorithm.
///
/// Pixels with intensity strictly greater than `threshold` are treated as
/// foreground. The returned label image assigns each foreground pixel a
/// component ID starting at 1 (background is 0). The second return value is
/// the total number of distinct components found.
///
/// # Errors
///
/// Returns [`ImageError::UnsupportedChannels`] if the image is not grayscale.
pub fn connected_components(img: &Image<u8>, threshold: u8) -> Result<(Image<u32>, usize)> {
    if img.format() != PixelFormat::Gray {
        return Err(ImageError::UnsupportedChannels {
            channels: img.channels(),
        });
    }

    let (w, h) = img.dimensions();
    let src = img.as_slice();
    let mut labels = vec![0u32; h * w];
    let mut next_label: u32 = 1;
    // Over-allocate the union-find; label 0 is unused as a real label.
    let mut uf = UnionFind::new(h * w + 1);

    // ---- Pass 1: initial labeling ----
    for row in 0..h {
        for col in 0..w {
            let idx = row * w + col;
            if src[idx] <= threshold {
                continue; // background
            }

            let top_label = if row > 0 {
                labels[(row - 1) * w + col]
            } else {
                0
            };
            let left_label = if col > 0 {
                labels[row * w + col - 1]
            } else {
                0
            };

            match (top_label > 0, left_label > 0) {
                (false, false) => {
                    labels[idx] = next_label;
                    next_label += 1;
                }
                (true, false) => {
                    labels[idx] = top_label;
                }
                (false, true) => {
                    labels[idx] = left_label;
                }
                (true, true) => {
                    let min_label = top_label.min(left_label);
                    labels[idx] = min_label;
                    if top_label != left_label {
                        uf.union(top_label, left_label);
                    }
                }
            }
        }
    }

    // ---- Pass 2: resolve labels ----
    // Build a mapping from root representatives to consecutive IDs.
    let mut root_to_id: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut num_components: u32 = 0;

    for label in &mut labels {
        if *label == 0 {
            continue;
        }
        let root = uf.find(*label);
        let id = root_to_id.entry(root).or_insert_with(|| {
            num_components += 1;
            num_components
        });
        *label = *id;
    }

    let label_img = Image::from_raw(labels, w, h, PixelFormat::Gray)?;
    Ok((label_img, num_components as usize))
}

// ---------------------------------------------------------------------------
// Region Growing
// ---------------------------------------------------------------------------

/// Grow regions from seed pixels using a BFS flood-fill.
///
/// Starting from each `(row, col)` seed, neighboring pixels whose intensity
/// differs from the seed intensity by at most `tolerance` are added to the
/// region. The result is a binary mask: 255 for pixels inside a grown region,
/// 0 for background.
///
/// # Errors
///
/// Returns [`ImageError::UnsupportedChannels`] if the image is not grayscale.
/// Returns [`ImageError::InvalidParameter`] if a seed is out of bounds.
pub fn region_growing(
    img: &Image<u8>,
    seeds: &[(usize, usize)],
    tolerance: u8,
) -> Result<Image<u8>> {
    if img.format() != PixelFormat::Gray {
        return Err(ImageError::UnsupportedChannels {
            channels: img.channels(),
        });
    }

    let (w, h) = img.dimensions();
    let src = img.as_slice();
    let mut mask = vec![0u8; h * w];
    let mut visited = vec![false; h * w];

    for &(seed_row, seed_col) in seeds {
        if seed_row >= h || seed_col >= w {
            return Err(ImageError::InvalidParameter {
                name: "seed",
                reason: "seed coordinates out of bounds",
            });
        }

        let seed_idx = seed_row * w + seed_col;
        if visited[seed_idx] {
            continue;
        }

        let seed_val = src[seed_idx];
        let mut queue = VecDeque::new();
        queue.push_back((seed_row, seed_col));
        visited[seed_idx] = true;

        while let Some((r, c)) = queue.pop_front() {
            let idx = r * w + c;
            mask[idx] = 255;

            // 4-connected neighbors
            let neighbors: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
            for (dr, dc) in neighbors {
                let nr = r.cast_signed() + dr;
                let nc = c.cast_signed() + dc;
                if nr < 0 || nc < 0 {
                    continue;
                }
                let nr = nr as usize;
                let nc = nc as usize;
                if nr >= h || nc >= w {
                    continue;
                }
                let nidx = nr * w + nc;
                if visited[nidx] {
                    continue;
                }
                let diff = src[nidx].abs_diff(seed_val);
                if diff <= tolerance {
                    visited[nidx] = true;
                    queue.push_back((nr, nc));
                }
            }
        }
    }

    Image::from_raw(mask, w, h, PixelFormat::Gray)
}

// ---------------------------------------------------------------------------
// Watershed (simplified marker-based)
// ---------------------------------------------------------------------------

/// Entry for the watershed priority queue (min-heap via `Reverse`).
#[derive(Debug, Clone, Eq, PartialEq)]
struct WatershedEntry {
    gradient: u8,
    row: usize,
    col: usize,
}

impl Ord for WatershedEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Primary: lower gradient first (achieved via Reverse in the heap).
        // Tie-break on position for determinism.
        self.gradient
            .cmp(&other.gradient)
            .then_with(|| self.row.cmp(&other.row))
            .then_with(|| self.col.cmp(&other.col))
    }
}

impl PartialOrd for WatershedEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Boundary label used by the watershed algorithm.
pub const WATERSHED_BOUNDARY: u32 = u32::MAX;

/// Simplified marker-based watershed segmentation.
///
/// Given a grayscale gradient image and a marker image (0 = unlabeled,
/// \>0 = marker label), the algorithm floods from markers in order of
/// increasing gradient value. When flooding fronts from different markers
/// meet, pixels are marked as boundary ([`WATERSHED_BOUNDARY`]).
///
/// # Errors
///
/// Returns [`ImageError::UnsupportedChannels`] if the gradient image is not
/// grayscale.
/// Returns [`ImageError::InvalidDimensions`] if the marker image dimensions
/// do not match the gradient image.
#[allow(clippy::too_many_lines)]
pub fn watershed(img: &Image<u8>, markers: &Image<u32>) -> Result<Image<u32>> {
    if img.format() != PixelFormat::Gray {
        return Err(ImageError::UnsupportedChannels {
            channels: img.channels(),
        });
    }
    if markers.format() != PixelFormat::Gray {
        return Err(ImageError::UnsupportedChannels {
            channels: markers.channels(),
        });
    }

    let (w, h) = img.dimensions();
    if markers.width() != w || markers.height() != h {
        return Err(ImageError::InvalidDimensions {
            width: markers.width(),
            height: markers.height(),
        });
    }

    let gradient = img.as_slice();
    let marker_data = markers.as_slice();
    let mut labels = marker_data.to_vec();
    let mut in_queue = vec![false; h * w];

    // Min-heap: process lowest gradient first.
    let mut heap: BinaryHeap<Reverse<WatershedEntry>> = BinaryHeap::new();

    // Seed the queue with all unlabeled pixels adjacent to a marker.
    let neighbors: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    for row in 0..h {
        for col in 0..w {
            let idx = row * w + col;
            if labels[idx] == 0 {
                continue;
            }
            // Check if any neighbor is unlabeled — if so, push those neighbors.
            for (dr, dc) in &neighbors {
                let nr = row.cast_signed() + dr;
                let nc = col.cast_signed() + dc;
                if nr < 0 || nc < 0 {
                    continue;
                }
                let nr = nr as usize;
                let nc = nc as usize;
                if nr >= h || nc >= w {
                    continue;
                }
                let nidx = nr * w + nc;
                if labels[nidx] == 0 && !in_queue[nidx] {
                    in_queue[nidx] = true;
                    heap.push(Reverse(WatershedEntry {
                        gradient: gradient[nidx],
                        row: nr,
                        col: nc,
                    }));
                }
            }
        }
    }

    // Process the queue.
    while let Some(Reverse(entry)) = heap.pop() {
        let idx = entry.row * w + entry.col;

        // Determine label from already-labeled neighbors.
        let mut assigned_label: u32 = 0;
        let mut is_boundary = false;

        for (dr, dc) in &neighbors {
            let nr = entry.row.cast_signed() + dr;
            let nc = entry.col.cast_signed() + dc;
            if nr < 0 || nc < 0 {
                continue;
            }
            let nr = nr as usize;
            let nc = nc as usize;
            if nr >= h || nc >= w {
                continue;
            }
            let nidx = nr * w + nc;
            let nlabel = labels[nidx];
            if nlabel == 0 || nlabel == WATERSHED_BOUNDARY {
                continue;
            }
            if assigned_label == 0 {
                assigned_label = nlabel;
            } else if assigned_label != nlabel {
                is_boundary = true;
                break;
            }
        }

        if is_boundary {
            labels[idx] = WATERSHED_BOUNDARY;
        } else if assigned_label > 0 {
            labels[idx] = assigned_label;
        }

        // If this pixel was labeled (not left at 0), push its unlabeled neighbors.
        if labels[idx] != 0 {
            for (dr, dc) in &neighbors {
                let nr = entry.row.cast_signed() + dr;
                let nc = entry.col.cast_signed() + dc;
                if nr < 0 || nc < 0 {
                    continue;
                }
                let nr = nr as usize;
                let nc = nc as usize;
                if nr >= h || nc >= w {
                    continue;
                }
                let nidx = nr * w + nc;
                if labels[nidx] == 0 && !in_queue[nidx] {
                    in_queue[nidx] = true;
                    heap.push(Reverse(WatershedEntry {
                        gradient: gradient[nidx],
                        row: nr,
                        col: nc,
                    }));
                }
            }
        }
    }

    Image::from_raw(labels, w, h, PixelFormat::Gray)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a grayscale `Image<u8>` from a flat row-major slice.
    fn gray_u8(data: &[u8], w: usize, h: usize) -> Image<u8> {
        Image::from_raw(data.to_vec(), w, h, PixelFormat::Gray).unwrap()
    }

    /// Helper: create a grayscale `Image<u32>` from a flat row-major slice.
    fn gray_u32(data: &[u32], w: usize, h: usize) -> Image<u32> {
        Image::from_raw(data.to_vec(), w, h, PixelFormat::Gray).unwrap()
    }

    // -- Connected Components --

    #[test]
    fn test_connected_components_two_regions() {
        // 4x4 image with two separate foreground blobs (threshold = 0).
        // Row-major layout:
        //   1 1 0 0
        //   1 1 0 0
        //   0 0 1 1
        //   0 0 1 1
        #[rustfmt::skip]
        let data: Vec<u8> = vec![
            1, 1, 0, 0,
            1, 1, 0, 0,
            0, 0, 1, 1,
            0, 0, 1, 1,
        ];
        let img = gray_u8(&data, 4, 4);
        let (label_img, count) = connected_components(&img, 0).unwrap();
        assert_eq!(count, 2);

        let labels = label_img.as_slice();
        // The two blobs should have different non-zero labels.
        let top_left = labels[0];
        let bottom_right = labels[2 * 4 + 2];
        assert_ne!(top_left, 0);
        assert_ne!(bottom_right, 0);
        assert_ne!(top_left, bottom_right);
    }

    #[test]
    fn test_connected_components_correct_count() {
        // 5x1 row: three separate foreground pixels.
        let data: Vec<u8> = vec![1, 0, 1, 0, 1];
        let img = gray_u8(&data, 5, 1);
        let (_label_img, count) = connected_components(&img, 0).unwrap();
        assert_eq!(count, 3);
    }

    // -- Region Growing --

    #[test]
    fn test_region_growing_single_seed() {
        // 3x3 uniform image; growing from center should fill all pixels.
        let data = vec![100u8; 9];
        let img = gray_u8(&data, 3, 3);
        let mask = region_growing(&img, &[(1, 1)], 0).unwrap();
        assert!(mask.as_slice().iter().all(|&v| v == 255));
    }

    #[test]
    fn test_region_growing_respects_tolerance() {
        // 3x3 image: center = 100, neighbors = 110, corners = 200.
        #[rustfmt::skip]
        let data: Vec<u8> = vec![
            200, 110, 200,
            110, 100, 110,
            200, 110, 200,
        ];
        let img = gray_u8(&data, 3, 3);

        // Tolerance 10: center + direct neighbors only.
        let mask = region_growing(&img, &[(1, 1)], 10).unwrap();
        let m = mask.as_slice();
        // Center and 4-connected neighbors should be filled.
        assert_eq!(m[4], 255); // center
        assert_eq!(m[1], 255); // top
        assert_eq!(m[3], 255); // left
        assert_eq!(m[5], 255); // right
        assert_eq!(m[7], 255); // bottom
        // Corners (value 200, diff = 100) should not be filled.
        assert_eq!(m[0], 0);
        assert_eq!(m[2], 0);
        assert_eq!(m[6], 0);
        assert_eq!(m[8], 0);
    }

    // -- Watershed --

    #[test]
    fn test_watershed_separates_markers() {
        // 5x1 gradient image with two markers at opposite ends.
        // Gradient:   0  1  2  1  0
        // Markers:    1  0  0  0  2
        let gradient = gray_u8(&[0, 1, 2, 1, 0], 5, 1);
        let markers = gray_u32(&[1, 0, 0, 0, 2], 5, 1);

        let result = watershed(&gradient, &markers).unwrap();
        let labels = result.as_slice();

        // The two ends should keep their marker labels.
        assert_eq!(labels[0], 1);
        assert_eq!(labels[4], 2);

        // The center pixel (highest gradient) should be a boundary or one of
        // the labels; the key property is that labels[1] == 1 and labels[3] == 2
        // (they are closer to their respective markers and have lower gradient).
        assert_eq!(labels[1], 1);
        assert_eq!(labels[3], 2);
    }
}
