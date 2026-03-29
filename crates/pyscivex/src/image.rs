//! Python bindings for scivex-image — image processing & computer vision.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scivex_image::prelude::*;
use scivex_image::{
    augment::{AugmentPipeline, AugmentStep},
    color, contour, draw, features, filter, histogram, hough, io as img_io, lanczos, matching,
    morphology, optical_flow, orb, segment, transform,
};

use crate::tensor::PyTensor;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn py_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Build an Image<u8> from flat pixel list + dimensions + channels.
fn image_u8_from_list(
    data: Vec<u8>,
    width: usize,
    height: usize,
    channels: usize,
) -> PyResult<Image<u8>> {
    let fmt = match channels {
        1 => PixelFormat::Gray,
        2 => PixelFormat::GrayAlpha,
        3 => PixelFormat::Rgb,
        4 => PixelFormat::Rgba,
        _ => return Err(py_err(format!("unsupported channel count: {channels}"))),
    };
    Image::from_raw(data, width, height, fmt).map_err(py_err)
}

fn format_str(fmt: PixelFormat) -> &'static str {
    match fmt {
        PixelFormat::Gray => "gray",
        PixelFormat::GrayAlpha => "gray_alpha",
        PixelFormat::Rgb => "rgb",
        PixelFormat::Rgba => "rgba",
    }
}

fn parse_resize_method(s: &str) -> PyResult<transform::ResizeMethod> {
    match s.to_lowercase().as_str() {
        "nearest" => Ok(transform::ResizeMethod::Nearest),
        "bilinear" => Ok(transform::ResizeMethod::Bilinear),
        _ => Err(py_err(format!("unknown resize method: {s}"))),
    }
}

fn parse_structuring_element(kind: &str, size: usize) -> PyResult<StructuringElement> {
    match kind.to_lowercase().as_str() {
        "rect" => Ok(StructuringElement::Rect(size, size)),
        "cross" => Ok(StructuringElement::Cross(size)),
        "disk" => Ok(StructuringElement::Disk(size)),
        _ => Err(py_err(format!(
            "unknown structuring element: {kind} (use rect/cross/disk)"
        ))),
    }
}

// ---------------------------------------------------------------------------
// PyImage — core image class (u8 backed)
// ---------------------------------------------------------------------------

#[pyclass(name = "Image")]
pub struct PyImage {
    inner: Image<u8>,
}

#[pymethods]
impl PyImage {
    /// Create a new image from flat pixel data.
    #[new]
    #[pyo3(signature = (data, width, height, channels = 3))]
    fn new(data: Vec<u8>, width: usize, height: usize, channels: usize) -> PyResult<Self> {
        let img = image_u8_from_list(data, width, height, channels)?;
        Ok(Self { inner: img })
    }

    /// Create a zero-filled image.
    #[staticmethod]
    #[pyo3(signature = (width, height, channels = 3))]
    fn zeros(width: usize, height: usize, channels: usize) -> PyResult<Self> {
        let fmt = match channels {
            1 => PixelFormat::Gray,
            3 => PixelFormat::Rgb,
            4 => PixelFormat::Rgba,
            _ => return Err(py_err(format!("unsupported channel count: {channels}"))),
        };
        let img = Image::new(width, height, fmt).map_err(py_err)?;
        Ok(Self { inner: img })
    }

    /// Load image from file (auto-detects format: ppm/pgm/bmp/png/jpeg).
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let img = img_io::load(path).map_err(py_err)?;
        Ok(Self { inner: img })
    }

    /// Save image to file (auto-detects format from extension).
    fn save(&self, path: &str) -> PyResult<()> {
        img_io::save(&self.inner, path).map_err(py_err)
    }

    /// Width in pixels.
    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    /// Height in pixels.
    #[getter]
    fn height(&self) -> usize {
        self.inner.height()
    }

    /// Number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.inner.channels()
    }

    /// Pixel format string.
    #[getter]
    fn format(&self) -> &'static str {
        format_str(self.inner.format())
    }

    /// (width, height) tuple.
    fn dimensions(&self) -> (usize, usize) {
        self.inner.dimensions()
    }

    /// Shape as [height, width, channels].
    fn shape(&self) -> Vec<usize> {
        vec![
            self.inner.height(),
            self.inner.width(),
            self.inner.channels(),
        ]
    }

    /// Get flat pixel data as list.
    fn to_list(&self) -> Vec<u8> {
        self.inner.as_slice().to_vec()
    }

    /// Convert to Tensor [height, width, channels].
    fn to_tensor(&self) -> PyTensor {
        let t = self.inner.as_tensor();
        // Convert u8 tensor to f64 tensor for PyTensor compatibility
        let data: Vec<f64> = t.as_slice().iter().map(|&v| v as f64 / 255.0).collect();
        let shape = t.shape().to_vec();
        let tensor = scivex_core::Tensor::from_vec(data, shape).expect("tensor from image data");
        PyTensor::from_f64(tensor)
    }

    /// Get pixel at (row, col) as list of channel values.
    fn get_pixel(&self, row: usize, col: usize) -> PyResult<Vec<u8>> {
        self.inner.get_pixel(row, col).map_err(py_err)
    }

    /// Set pixel at (row, col).
    fn set_pixel(&mut self, row: usize, col: usize, values: Vec<u8>) -> PyResult<()> {
        self.inner.set_pixel(row, col, &values).map_err(py_err)
    }

    // -- Color conversions --

    /// Convert the image to grayscale (single channel).
    fn to_grayscale(&self) -> PyResult<Self> {
        let img = color::to_grayscale(&self.inner).map_err(py_err)?;
        Ok(Self { inner: img })
    }

    /// Convert the image to RGB (three channels).
    fn to_rgb(&self) -> PyResult<Self> {
        let img = color::to_rgb(&self.inner).map_err(py_err)?;
        Ok(Self { inner: img })
    }

    /// Invert all pixel values (255 - value for each channel).
    fn invert(&self) -> Self {
        Self {
            inner: color::invert(&self.inner),
        }
    }

    // -- Transforms --

    /// Resize the image to (new_width, new_height) using the given method ("nearest" or "bilinear").
    #[pyo3(signature = (new_width, new_height, method = "nearest"))]
    fn resize(&self, new_width: usize, new_height: usize, method: &str) -> PyResult<Self> {
        let m = parse_resize_method(method)?;
        let img = transform::resize(&self.inner, new_width, new_height, m).map_err(py_err)?;
        Ok(Self { inner: img })
    }

    /// Crop a rectangular region starting at (x, y) with the given width and height.
    fn crop(&self, x: usize, y: usize, width: usize, height: usize) -> PyResult<Self> {
        let img = transform::crop(&self.inner, x, y, width, height).map_err(py_err)?;
        Ok(Self { inner: img })
    }

    /// Flip the image horizontally (mirror left-right).
    fn flip_horizontal(&self) -> Self {
        Self {
            inner: transform::flip_horizontal(&self.inner),
        }
    }

    /// Flip the image vertically (mirror top-bottom).
    fn flip_vertical(&self) -> Self {
        Self {
            inner: transform::flip_vertical(&self.inner),
        }
    }

    /// Rotate the image 90 degrees clockwise.
    fn rotate90(&self) -> Self {
        Self {
            inner: transform::rotate90(&self.inner),
        }
    }

    /// Rotate the image 180 degrees.
    fn rotate180(&self) -> Self {
        Self {
            inner: transform::rotate180(&self.inner),
        }
    }

    /// Rotate the image 270 degrees clockwise (90 degrees counter-clockwise).
    fn rotate270(&self) -> Self {
        Self {
            inner: transform::rotate270(&self.inner),
        }
    }

    /// Pad the image with the given number of pixels on each side, filled with `value`.
    #[pyo3(signature = (top, bottom, left, right, value = 0))]
    fn pad(&self, top: usize, bottom: usize, left: usize, right: usize, value: u8) -> Self {
        Self {
            inner: transform::pad(&self.inner, top, bottom, left, right, value),
        }
    }

    // -- Morphology --

    /// Apply morphological erosion with the given structuring element ("rect", "cross", or "disk").
    #[pyo3(signature = (kind = "rect", size = 3))]
    fn erode(&self, kind: &str, size: usize) -> PyResult<Self> {
        let se = parse_structuring_element(kind, size)?;
        let img = morphology::erode(&self.inner, &se).map_err(py_err)?;
        Ok(Self { inner: img })
    }

    /// Apply morphological dilation with the given structuring element ("rect", "cross", or "disk").
    #[pyo3(signature = (kind = "rect", size = 3))]
    fn dilate(&self, kind: &str, size: usize) -> PyResult<Self> {
        let se = parse_structuring_element(kind, size)?;
        let img = morphology::dilate(&self.inner, &se).map_err(py_err)?;
        Ok(Self { inner: img })
    }

    /// Apply morphological opening (erosion followed by dilation).
    #[pyo3(signature = (kind = "rect", size = 3))]
    fn opening(&self, kind: &str, size: usize) -> PyResult<Self> {
        let se = parse_structuring_element(kind, size)?;
        let img = morphology::opening(&self.inner, &se).map_err(py_err)?;
        Ok(Self { inner: img })
    }

    /// Apply morphological closing (dilation followed by erosion).
    #[pyo3(signature = (kind = "rect", size = 3))]
    fn closing(&self, kind: &str, size: usize) -> PyResult<Self> {
        let se = parse_structuring_element(kind, size)?;
        let img = morphology::closing(&self.inner, &se).map_err(py_err)?;
        Ok(Self { inner: img })
    }

    // -- Drawing --

    /// Draw a line from (x0, y0) to (x1, y1) with the given color (list of channel values).
    fn draw_line(&mut self, x0: isize, y0: isize, x1: isize, y1: isize, color: Vec<u8>) {
        draw::draw_line(&mut self.inner, x0, y0, x1, y1, &color);
    }

    /// Draw an unfilled rectangle at (x, y) with width w and height h.
    fn draw_rect(&mut self, x: isize, y: isize, w: usize, h: usize, color: Vec<u8>) {
        draw::draw_rect(&mut self.inner, x, y, w, h, &color);
    }

    /// Draw a filled rectangle at (x, y) with width w and height h.
    fn fill_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: Vec<u8>) {
        draw::fill_rect(&mut self.inner, x, y, w, h, &color);
    }

    /// Draw a circle centered at (cx, cy) with the given radius and color.
    fn draw_circle(&mut self, cx: isize, cy: isize, radius: usize, color: Vec<u8>) {
        draw::draw_circle(&mut self.inner, cx, cy, radius, &color);
    }

    // -- Histogram --

    /// Compute the pixel intensity histogram as a Tensor.
    fn histogram(&self) -> PyTensor {
        let h = histogram::histogram(&self.inner);
        let data: Vec<f64> = h.as_slice().iter().map(|&v| v as f64).collect();
        let shape = h.shape().to_vec();
        let t = scivex_core::Tensor::from_vec(data, shape).expect("histogram tensor");
        PyTensor::from_f64(t)
    }

    /// Apply histogram equalization to improve contrast.
    fn equalize(&self) -> Self {
        Self {
            inner: histogram::equalize(&self.inner),
        }
    }

    // -- Segmentation --

    /// Find connected components using the given binarization threshold. Returns (labels, count).
    fn connected_components(&self, threshold: u8) -> PyResult<(Vec<u32>, usize)> {
        let (label_img, count) =
            segment::connected_components(&self.inner, threshold).map_err(py_err)?;
        Ok((label_img.as_slice().to_vec(), count))
    }

    /// Segment the image by region growing from seed points with the given intensity tolerance.
    fn region_growing(&self, seeds: Vec<(usize, usize)>, tolerance: u8) -> PyResult<Vec<u8>> {
        let result = segment::region_growing(&self.inner, &seeds, tolerance).map_err(py_err)?;
        Ok(result.as_slice().to_vec())
    }

    /// Return a string representation of the image (width, height, channels, format).
    fn __repr__(&self) -> String {
        format!(
            "Image(width={}, height={}, channels={}, format='{}')",
            self.inner.width(),
            self.inner.height(),
            self.inner.channels(),
            format_str(self.inner.format()),
        )
    }
}

// ---------------------------------------------------------------------------
// Module-level functions (sv.image.*)
// ---------------------------------------------------------------------------

/// Convert RGB image to HSV (works on f32 images internally).
#[pyfunction]
fn rgb_to_hsv(img: &PyImage) -> PyResult<PyImage> {
    let f = img.inner.to_f32();
    let hsv = color::rgb_to_hsv(&f).map_err(py_err)?;
    Ok(PyImage { inner: hsv.to_u8() })
}

/// Convert HSV image to RGB.
#[pyfunction]
fn hsv_to_rgb(img: &PyImage) -> PyResult<PyImage> {
    let f = img.inner.to_f32();
    let rgb = color::hsv_to_rgb(&f).map_err(py_err)?;
    Ok(PyImage { inner: rgb.to_u8() })
}

// -- Filters (operate on f32 internally) --

/// Apply Gaussian blur with the given sigma (standard deviation).
#[pyfunction]
fn gaussian_blur(img: &PyImage, sigma: f32) -> PyResult<PyImage> {
    let f = img.inner.to_f32();
    let out = filter::gaussian_blur(&f, sigma).map_err(py_err)?;
    Ok(PyImage { inner: out.to_u8() })
}

/// Apply box blur (uniform averaging) with the given radius.
#[pyfunction]
fn box_blur(img: &PyImage, radius: usize) -> PyResult<PyImage> {
    let f = img.inner.to_f32();
    let out = filter::box_blur(&f, radius).map_err(py_err)?;
    Ok(PyImage { inner: out.to_u8() })
}

/// Apply a sharpening filter to enhance edges and details.
#[pyfunction]
fn sharpen(img: &PyImage) -> PyResult<PyImage> {
    let f = img.inner.to_f32();
    let out = filter::sharpen(&f).map_err(py_err)?;
    Ok(PyImage { inner: out.to_u8() })
}

/// Apply the Sobel edge detection filter (combined X and Y gradients).
#[pyfunction]
fn sobel(img: &PyImage) -> PyResult<PyImage> {
    let f = img.inner.to_f32();
    let out = filter::sobel(&f).map_err(py_err)?;
    Ok(PyImage { inner: out.to_u8() })
}

/// Apply the Sobel filter in the horizontal (X) direction.
#[pyfunction]
fn sobel_x(img: &PyImage) -> PyResult<PyImage> {
    let f = img.inner.to_f32();
    let out = filter::sobel_x(&f).map_err(py_err)?;
    Ok(PyImage { inner: out.to_u8() })
}

/// Apply the Sobel filter in the vertical (Y) direction.
#[pyfunction]
fn sobel_y(img: &PyImage) -> PyResult<PyImage> {
    let f = img.inner.to_f32();
    let out = filter::sobel_y(&f).map_err(py_err)?;
    Ok(PyImage { inner: out.to_u8() })
}

/// Apply a median filter with the given radius for noise reduction.
#[pyfunction]
fn median_filter(img: &PyImage, radius: usize) -> PyResult<PyImage> {
    let out = filter::median_filter(&img.inner, radius).map_err(py_err)?;
    Ok(PyImage { inner: out })
}

// -- Lanczos resize --

/// Resize the image using Lanczos interpolation. Parameter `a` controls the kernel size (default 3).
#[pyfunction]
#[pyo3(signature = (img, new_width, new_height, a = 3))]
fn resize_lanczos(
    img: &PyImage,
    new_width: usize,
    new_height: usize,
    a: usize,
) -> PyResult<PyImage> {
    let f = img.inner.to_f32();
    let out = lanczos::resize_lanczos(&f, new_width, new_height, a).map_err(py_err)?;
    Ok(PyImage { inner: out.to_u8() })
}

// -- Feature detection --

/// Detect corners using the Harris corner detector. Returns list of (row, col, response) tuples.
#[pyfunction]
#[pyo3(signature = (img, k = 0.04, threshold = 1000000.0, block_size = 3))]
fn harris_corners(
    img: &PyImage,
    k: f32,
    threshold: f32,
    block_size: usize,
) -> PyResult<Vec<(usize, usize, f64)>> {
    // Need grayscale f32
    let gray = color::to_grayscale(&img.inner).map_err(py_err)?;
    let f = gray.to_f32();
    let corners = features::harris_corners(&f, k, threshold, block_size).map_err(py_err)?;
    Ok(corners.iter().map(|c| (c.row, c.col, c.response)).collect())
}

/// Detect corners using the FAST feature detector. Returns list of (row, col, response) tuples.
#[pyfunction]
#[pyo3(signature = (img, threshold = 20, nonmax = true))]
fn fast_corners(img: &PyImage, threshold: u8, nonmax: bool) -> PyResult<Vec<(usize, usize, f64)>> {
    let gray = color::to_grayscale(&img.inner).map_err(py_err)?;
    let corners = features::fast_features(&gray, threshold, nonmax).map_err(py_err)?;
    Ok(corners.iter().map(|c| (c.row, c.col, c.response)).collect())
}

// -- ORB --

/// Detect ORB keypoints and compute binary descriptors. Returns list of dicts with row, col, response, angle, descriptor.
#[pyfunction]
#[pyo3(signature = (img, n_features = 500, fast_threshold = 20))]
fn orb_features(img: &PyImage, n_features: usize, fast_threshold: u8) -> PyResult<Vec<PyObject>> {
    let gray = color::to_grayscale(&img.inner).map_err(py_err)?;
    let detector = orb::OrbDetector::new()
        .with_n_features(n_features)
        .with_fast_threshold(fast_threshold);
    let descriptors = detector.detect_and_compute(&gray).map_err(py_err)?;

    Python::with_gil(|py| {
        let mut result = Vec::with_capacity(descriptors.len());
        for d in &descriptors {
            let dict = PyDict::new(py);
            dict.set_item("row", d.keypoint.row)?;
            dict.set_item("col", d.keypoint.col)?;
            dict.set_item("response", d.keypoint.response)?;
            dict.set_item("angle", d.keypoint.angle)?;
            dict.set_item("descriptor", d.descriptor.to_vec())?;
            result.push(dict.into_any().unbind());
        }
        Ok(result)
    })
}

// -- Feature matching --

/// Match binary feature descriptors using brute-force Hamming distance. Returns list of (query_idx, train_idx, distance).
#[pyfunction]
#[pyo3(signature = (query, train, ratio = None))]
fn match_features(
    query: Vec<Vec<u8>>,
    train: Vec<Vec<u8>>,
    ratio: Option<f64>,
) -> PyResult<Vec<(usize, usize, u32)>> {
    // Convert to [u8; 32] arrays
    let q: Vec<[u8; 32]> = query
        .iter()
        .map(|v| {
            let mut arr = [0u8; 32];
            let len = v.len().min(32);
            arr[..len].copy_from_slice(&v[..len]);
            arr
        })
        .collect();
    let t: Vec<[u8; 32]> = train
        .iter()
        .map(|v| {
            let mut arr = [0u8; 32];
            let len = v.len().min(32);
            arr[..len].copy_from_slice(&v[..len]);
            arr
        })
        .collect();

    let matcher = matching::BruteForceMatcher::new();
    let matches = if let Some(r) = ratio {
        matcher.match_with_ratio_test(&q, &t, r)
    } else {
        matcher.match_descriptors(&q, &t)
    };

    Ok(matches
        .iter()
        .map(|m| (m.query_idx, m.train_idx, m.distance))
        .collect())
}

// -- Hough transforms --

/// Detect lines using the Hough transform. Returns list of (rho, theta, votes) tuples.
#[pyfunction]
#[pyo3(signature = (img, rho_resolution = 1.0, theta_resolution = 0.01745329, threshold = 50))]
fn hough_lines(
    img: &PyImage,
    rho_resolution: f64,
    theta_resolution: f64,
    threshold: usize,
) -> PyResult<Vec<(f64, f64, usize)>> {
    let gray = color::to_grayscale(&img.inner).map_err(py_err)?;
    let lines =
        hough::hough_lines(&gray, rho_resolution, theta_resolution, threshold).map_err(py_err)?;
    Ok(lines.iter().map(|l| (l.rho, l.theta, l.votes)).collect())
}

/// Detect circles using the Hough transform. Returns list of (center_row, center_col, radius, votes).
#[pyfunction]
#[pyo3(signature = (img, min_radius, max_radius, threshold = 50))]
fn hough_circles(
    img: &PyImage,
    min_radius: usize,
    max_radius: usize,
    threshold: usize,
) -> PyResult<Vec<(usize, usize, usize, usize)>> {
    let gray = color::to_grayscale(&img.inner).map_err(py_err)?;
    let circles = hough::hough_circles(&gray, min_radius, max_radius, threshold).map_err(py_err)?;
    Ok(circles
        .iter()
        .map(|c| (c.center_row, c.center_col, c.radius, c.votes))
        .collect())
}

// -- Contours --

/// Find contours in the image after binarization at the given threshold. Returns list of point lists.
#[pyfunction]
#[pyo3(signature = (img, threshold = 128))]
fn find_contours(img: &PyImage, threshold: u8) -> PyResult<Vec<Vec<(usize, usize)>>> {
    let gray = color::to_grayscale(&img.inner).map_err(py_err)?;
    let contours = contour::find_contours(&gray, threshold).map_err(py_err)?;
    Ok(contours.iter().map(|c| c.points.clone()).collect())
}

/// Compute the area enclosed by a contour (list of (row, col) points).
#[pyfunction]
fn contour_area(points: Vec<(usize, usize)>) -> f64 {
    let c = contour::Contour { points };
    contour::contour_area(&c)
}

/// Compute the perimeter of a contour (list of (row, col) points).
#[pyfunction]
fn contour_perimeter(points: Vec<(usize, usize)>) -> f64 {
    let c = contour::Contour { points };
    contour::contour_perimeter(&c)
}

// -- Optical flow --

/// Compute sparse optical flow between two frames using the Lucas-Kanade method. Returns dict with flow_x, flow_y, width, height.
#[pyfunction]
#[pyo3(signature = (prev, next, window_size = 5))]
fn lucas_kanade(prev: &PyImage, next: &PyImage, window_size: usize) -> PyResult<PyObject> {
    let p = prev.inner.to_f32();
    let n = next.inner.to_f32();
    let result = optical_flow::lucas_kanade(&p, &n, window_size).map_err(py_err)?;

    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("flow_x", &result.flow_x)?;
        dict.set_item("flow_y", &result.flow_y)?;
        dict.set_item("width", result.width)?;
        dict.set_item("height", result.height)?;
        Ok(dict.into_any().unbind())
    })
}

// -- Augmentation pipeline --

#[pyclass(name = "AugmentPipeline")]
pub struct PyAugmentPipeline {
    steps: Vec<AugmentStep>,
}

#[pymethods]
impl PyAugmentPipeline {
    /// Create an empty augmentation pipeline.
    #[new]
    fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a random horizontal flip step with the given probability.
    fn random_flip_h(&mut self, prob: f64) {
        self.steps.push(AugmentStep::RandomFlipH { prob });
    }

    /// Add a random vertical flip step with the given probability.
    fn random_flip_v(&mut self, prob: f64) {
        self.steps.push(AugmentStep::RandomFlipV { prob });
    }

    /// Add a random brightness adjustment step with the given delta range.
    fn random_brightness(&mut self, delta: f64) {
        self.steps.push(AugmentStep::RandomBrightness { delta });
    }

    /// Add Gaussian noise with the given standard deviation (sigma).
    fn gaussian_noise(&mut self, sigma: f64) {
        self.steps.push(AugmentStep::GaussianNoise { sigma });
    }

    /// Add a random crop step that extracts a region of the given width and height.
    fn random_crop(&mut self, width: usize, height: usize) {
        self.steps.push(AugmentStep::RandomCrop { width, height });
    }

    /// Add color jitter augmentation with adjustable brightness, contrast, saturation, and hue.
    #[pyo3(signature = (brightness = 0.0, contrast = 0.0, saturation = 0.0, hue = 0.0))]
    fn color_jitter(&mut self, brightness: f64, contrast: f64, saturation: f64, hue: f64) {
        self.steps.push(AugmentStep::ColorJitter {
            brightness,
            contrast,
            saturation,
            hue,
        });
    }

    /// Add a random rotation step up to `max_angle` degrees.
    fn random_rotation(&mut self, max_angle: f64) {
        self.steps.push(AugmentStep::RandomRotation { max_angle });
    }

    /// Add a cutout (random erasing) step that zeros out a region of the given size.
    fn cutout(&mut self, width: usize, height: usize) {
        self.steps.push(AugmentStep::CutOut { width, height });
    }

    /// Add a normalization step that subtracts `mean` and divides by `std` per channel.
    fn normalize(&mut self, mean: Vec<f64>, std: Vec<f64>) {
        self.steps.push(AugmentStep::Normalize { mean, std });
    }

    /// Apply augmentation pipeline to an image.
    #[pyo3(signature = (img, seed = 42))]
    fn apply(&self, img: &PyImage, seed: u64) -> PyResult<PyImage> {
        let f = img.inner.to_f32();
        let mut pipeline = AugmentPipeline::new();
        for step in &self.steps {
            pipeline = pipeline.add(step.clone());
        }
        let mut rng = scivex_core::random::Rng::new(seed);
        let out = pipeline.apply(&f, &mut rng).map_err(py_err)?;
        Ok(PyImage { inner: out.to_u8() })
    }

    /// Return a string representation showing the number of augmentation steps.
    fn __repr__(&self) -> String {
        format!("AugmentPipeline(steps={})", self.steps.len())
    }
}

// -- Watershed (needs marker image) --

/// Apply watershed segmentation using marker-based regions. Returns flat label array.
#[pyfunction]
fn watershed(img: &PyImage, markers: Vec<u32>, width: usize, height: usize) -> PyResult<Vec<u32>> {
    let gray = color::to_grayscale(&img.inner).map_err(py_err)?;
    let marker_tensor =
        scivex_core::Tensor::from_vec(markers, vec![height, width, 1]).map_err(py_err)?;
    let marker_img = Image::from_tensor(marker_tensor, PixelFormat::Gray).map_err(py_err)?;
    let result = segment::watershed(&gray, &marker_img).map_err(py_err)?;
    Ok(result.as_slice().to_vec())
}

// ---------------------------------------------------------------------------
// Register submodule
// ---------------------------------------------------------------------------

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "image")?;

    // Core class
    m.add_class::<PyImage>()?;
    m.add_class::<PyAugmentPipeline>()?;

    // Color
    m.add_function(wrap_pyfunction!(rgb_to_hsv, &m)?)?;
    m.add_function(wrap_pyfunction!(hsv_to_rgb, &m)?)?;

    // Filters
    m.add_function(wrap_pyfunction!(gaussian_blur, &m)?)?;
    m.add_function(wrap_pyfunction!(box_blur, &m)?)?;
    m.add_function(wrap_pyfunction!(sharpen, &m)?)?;
    m.add_function(wrap_pyfunction!(sobel, &m)?)?;
    m.add_function(wrap_pyfunction!(sobel_x, &m)?)?;
    m.add_function(wrap_pyfunction!(sobel_y, &m)?)?;
    m.add_function(wrap_pyfunction!(median_filter, &m)?)?;

    // Lanczos
    m.add_function(wrap_pyfunction!(resize_lanczos, &m)?)?;

    // Features
    m.add_function(wrap_pyfunction!(harris_corners, &m)?)?;
    m.add_function(wrap_pyfunction!(fast_corners, &m)?)?;
    m.add_function(wrap_pyfunction!(orb_features, &m)?)?;
    m.add_function(wrap_pyfunction!(match_features, &m)?)?;

    // Hough
    m.add_function(wrap_pyfunction!(hough_lines, &m)?)?;
    m.add_function(wrap_pyfunction!(hough_circles, &m)?)?;

    // Contours
    m.add_function(wrap_pyfunction!(find_contours, &m)?)?;
    m.add_function(wrap_pyfunction!(contour_area, &m)?)?;
    m.add_function(wrap_pyfunction!(contour_perimeter, &m)?)?;

    // Optical flow
    m.add_function(wrap_pyfunction!(lucas_kanade, &m)?)?;

    // Segmentation
    m.add_function(wrap_pyfunction!(watershed, &m)?)?;

    parent.add_submodule(&m)?;
    Ok(())
}
