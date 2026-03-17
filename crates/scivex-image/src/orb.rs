//! ORB (Oriented FAST and Rotated BRIEF) feature descriptor.
//!
//! Provides a combined keypoint detector and binary descriptor suitable for
//! real-time feature matching. The implementation follows the ORB paper:
//! detect FAST keypoints, compute orientation via intensity centroid, then
//! build a rotation-aware BRIEF descriptor.

use crate::error::{ImageError, Result};
use crate::features::Corner;
use crate::image::{Image, PixelFormat};

/// Bresenham circle of radius 3 (16 pixels), clockwise from top.
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

/// A keypoint with orientation and scale information.
#[derive(Debug, Clone)]
pub struct Keypoint {
    /// Row (y) coordinate.
    pub row: usize,
    /// Column (x) coordinate.
    pub col: usize,
    /// Detector response value.
    pub response: f64,
    /// Orientation in radians, computed via intensity centroid.
    pub angle: f64,
    /// Scale pyramid level (0 = original resolution).
    pub octave: usize,
}

/// An ORB descriptor: a keypoint paired with a 256-bit binary descriptor.
#[derive(Debug, Clone)]
pub struct OrbDescriptor {
    /// The oriented keypoint.
    pub keypoint: Keypoint,
    /// 256-bit binary descriptor packed into 32 bytes.
    pub descriptor: [u8; 32],
}

/// ORB feature detector and descriptor extractor.
///
/// Detects FAST keypoints, computes orientation via intensity centroid,
/// and builds rotation-aware BRIEF descriptors.
#[derive(Debug, Clone)]
pub struct OrbDetector {
    /// Maximum number of keypoints to retain (sorted by response).
    pub n_features: usize,
    /// FAST intensity threshold.
    pub fast_threshold: u8,
    /// Number of scale pyramid levels (1 = no pyramid).
    pub n_levels: usize,
}

impl Default for OrbDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl OrbDetector {
    /// Create a new `OrbDetector` with default parameters.
    ///
    /// Defaults: 500 features, FAST threshold 20, 1 pyramid level.
    pub fn new() -> Self {
        Self {
            n_features: 500,
            fast_threshold: 20,
            n_levels: 1,
        }
    }

    /// Set the maximum number of keypoints to retain.
    pub fn with_n_features(mut self, n: usize) -> Self {
        self.n_features = n;
        self
    }

    /// Set the FAST detector threshold.
    pub fn with_fast_threshold(mut self, t: u8) -> Self {
        self.fast_threshold = t;
        self
    }

    /// Detect keypoints and compute ORB descriptors.
    ///
    /// The input image must be grayscale (`PixelFormat::Gray`).
    ///
    /// # Algorithm
    ///
    /// 1. Detect FAST keypoints using a 16-pixel Bresenham circle test.
    /// 2. Sort by response and keep the top `n_features`.
    /// 3. Compute orientation for each keypoint via intensity centroid.
    /// 4. Compute a rotation-aware 256-bit BRIEF descriptor.
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_possible_wrap)]
    pub fn detect_and_compute(&self, img: &Image<u8>) -> Result<Vec<OrbDescriptor>> {
        if img.format() != PixelFormat::Gray {
            return Err(ImageError::UnsupportedChannels {
                channels: img.channels(),
            });
        }

        let (w, h) = img.dimensions();
        let src = img.as_slice();

        // --- Step 1: FAST keypoint detection ---
        let mut corners = self.detect_fast(src, w, h);

        // --- Step 2: Sort by response (descending) and keep top n_features ---
        corners.sort_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        corners.truncate(self.n_features);

        // --- Step 3 & 4: Compute orientation and descriptors ---
        let patch_radius: usize = 15;
        let mut results = Vec::with_capacity(corners.len());

        for corner in &corners {
            // Skip keypoints too close to the border for patch computation.
            if corner.row < patch_radius
                || corner.col < patch_radius
                || corner.row + patch_radius >= h
                || corner.col + patch_radius >= w
            {
                continue;
            }

            // Compute orientation via intensity centroid.
            let angle = self.compute_orientation(src, w, corner.row, corner.col, patch_radius);

            let keypoint = Keypoint {
                row: corner.row,
                col: corner.col,
                response: corner.response,
                angle,
                octave: 0,
            };

            // Compute rotated BRIEF descriptor.
            let descriptor = self.compute_brief(src, w, h, corner.row, corner.col, angle);

            results.push(OrbDescriptor {
                keypoint,
                descriptor,
            });
        }

        Ok(results)
    }

    /// Detect FAST-9 keypoints with non-maximum suppression.
    #[allow(clippy::cast_possible_wrap)]
    fn detect_fast(&self, src: &[u8], w: usize, h: usize) -> Vec<Corner> {
        let threshold = self.fast_threshold;
        let mut scores = vec![0u16; h * w];

        for row in 3..h.saturating_sub(3) {
            for col in 3..w.saturating_sub(3) {
                let center = i16::from(src[row * w + col]);
                let hi = center + i16::from(threshold);
                let lo = center - i16::from(threshold);

                let mut brighter = [false; 16];
                let mut darker = [false; 16];
                for (i, &(dr, dc)) in CIRCLE.iter().enumerate() {
                    let v = i16::from(
                        src[(row as isize + dr) as usize * w + (col as isize + dc) as usize],
                    );
                    brighter[i] = v > hi;
                    darker[i] = v < lo;
                }

                let score_b = max_contiguous(&brighter);
                let score_d = max_contiguous(&darker);
                let score = score_b.max(score_d);

                if score >= 9 {
                    scores[row * w + col] = score;
                }
            }
        }

        // Non-maximum suppression (3x3).
        let mut corners = Vec::new();
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

        corners
    }

    /// Compute keypoint orientation via intensity centroid.
    ///
    /// Computes first-order moments `m10` and `m01` over a square patch
    /// of the given radius and returns `atan2(m01, m10)`.
    #[allow(clippy::cast_possible_wrap, clippy::unused_self)]
    fn compute_orientation(
        &self,
        src: &[u8],
        w: usize,
        row: usize,
        col: usize,
        radius: usize,
    ) -> f64 {
        let mut m10: f64 = 0.0;
        let mut m01: f64 = 0.0;

        let r = radius as isize;
        for dy in -r..=r {
            for dx in -r..=r {
                let pr = (row as isize + dy) as usize;
                let pc = (col as isize + dx) as usize;
                let intensity = f64::from(src[pr * w + pc]);
                m10 += dx as f64 * intensity;
                m01 += dy as f64 * intensity;
            }
        }

        m01.atan2(m10)
    }

    /// Compute a rotation-aware 256-bit BRIEF descriptor.
    ///
    /// Uses 256 deterministic point pairs, rotates them by the keypoint
    /// angle, and compares pixel intensities to build a binary string
    /// packed into 32 bytes.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::unused_self
    )]
    fn compute_brief(
        &self,
        src: &[u8],
        w: usize,
        h: usize,
        row: usize,
        col: usize,
        angle: f64,
    ) -> [u8; 32] {
        let mut descriptor = [0u8; 32];
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        for i in 0..256 {
            // Deterministic point-pair pattern.
            let dx1 = ((i * 7 + 3) % 31) as f64 - 15.0;
            let dy1 = ((i * 13 + 5) % 31) as f64 - 15.0;
            let dx2 = ((i * 11 + 7) % 31) as f64 - 15.0;
            let dy2 = ((i * 17 + 11) % 31) as f64 - 15.0;

            // Rotate by keypoint angle.
            let rx1 = (cos_a * dx1 - sin_a * dy1).round() as isize;
            let ry1 = (sin_a * dx1 + cos_a * dy1).round() as isize;
            let rx2 = (cos_a * dx2 - sin_a * dy2).round() as isize;
            let ry2 = (sin_a * dx2 + cos_a * dy2).round() as isize;

            // Sample pixel values (clamped to image bounds).
            let r1 = (row as isize + ry1).clamp(0, (h as isize) - 1) as usize;
            let c1 = (col as isize + rx1).clamp(0, (w as isize) - 1) as usize;
            let r2 = (row as isize + ry2).clamp(0, (h as isize) - 1) as usize;
            let c2 = (col as isize + rx2).clamp(0, (w as isize) - 1) as usize;

            let p1 = src[r1 * w + c1];
            let p2 = src[r2 * w + c2];

            // Set bit if p1 < p2.
            if p1 < p2 {
                descriptor[i / 8] |= 1 << (i % 8);
            }
        }

        descriptor
    }
}

/// Return the maximum number of contiguous `true` values in a circular
/// array of 16 elements.
fn max_contiguous(flags: &[bool; 16]) -> u16 {
    if flags.iter().all(|&f| !f) {
        return 0;
    }
    let mut best: u16 = 0;
    let mut run: u16 = 0;
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

    /// Build a 64x64 checkerboard pattern with 8x8 blocks.
    fn make_checkerboard_u8() -> Image<u8> {
        let size = 64;
        let mut data = vec![0u8; size * size];
        for r in 0..size {
            for c in 0..size {
                if (r / 8 + c / 8) % 2 == 0 {
                    data[r * size + c] = 255;
                }
            }
        }
        Image::from_raw(data, size, size, PixelFormat::Gray).unwrap()
    }

    #[test]
    fn test_orb_detect_on_pattern() {
        // Create a 64x64 image with bright rectangles on dark background.
        // This creates edges and corners that FAST can detect (9+ contiguous).
        let size = 64;
        let mut data = vec![0u8; size * size];
        // Draw several bright rectangles.
        for r in 16..32 {
            for c in 16..48 {
                data[r * size + c] = 255;
            }
        }
        for r in 36..52 {
            for c in 20..44 {
                data[r * size + c] = 255;
            }
        }
        let img = Image::from_raw(data, size, size, PixelFormat::Gray).unwrap();
        let detector = OrbDetector::new().with_fast_threshold(20);
        let descriptors = detector.detect_and_compute(&img).unwrap();
        assert!(
            !descriptors.is_empty(),
            "ORB should detect features on rectangles"
        );
    }

    #[test]
    fn test_orb_descriptor_length() {
        let img = make_checkerboard_u8();
        let detector = OrbDetector::new().with_fast_threshold(30);
        let descriptors = detector.detect_and_compute(&img).unwrap();
        for desc in &descriptors {
            assert_eq!(
                desc.descriptor.len(),
                32,
                "ORB descriptor should be 32 bytes (256 bits)"
            );
        }
    }

    #[test]
    fn test_orb_uniform_image_no_features() {
        // A completely uniform image should produce no keypoints.
        let data = vec![128u8; 32 * 32];
        let img = Image::from_raw(data, 32, 32, PixelFormat::Gray).unwrap();
        let detector = OrbDetector::new();
        let descriptors = detector.detect_and_compute(&img).unwrap();
        assert!(
            descriptors.is_empty(),
            "uniform image should produce no ORB features"
        );
    }

    #[test]
    fn test_orb_n_features_limits_output() {
        let img = make_checkerboard_u8();
        let detector = OrbDetector::new()
            .with_n_features(3)
            .with_fast_threshold(30);
        let descriptors = detector.detect_and_compute(&img).unwrap();
        assert!(
            descriptors.len() <= 3,
            "with_n_features(3) should return at most 3 descriptors, got {}",
            descriptors.len()
        );
    }
}
