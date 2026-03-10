# scivex-image

Image processing for Scivex. Tensor-backed images with geometric transforms,
spatial filters, and drawing primitives.

## Highlights

- **Image<T>** — 2D image backed by `Tensor<T>` with `[height, width, channels]` layout
- **Pixel formats** — Gray, GrayAlpha, RGB, RGBA
- **Transforms** — Resize (nearest/bilinear), crop, flip, rotate (90/180/270), pad
- **Filters** — Gaussian blur, box blur, Sobel edge detection, 2D convolution
- **Color** — RGB/HSV conversion, grayscale conversion
- **Histogram** — Compute histograms, histogram equalization
- **Drawing** — Lines (Bresenham), rectangles, circles
- **I/O** — PPM, PGM, BMP file formats
- **Type conversion** — `Image<u8>` to/from `Image<f32>` with [0,255] / [0,1] scaling

## Usage

```rust
use scivex_image::prelude::*;

// Load and process
let img = Image::<u8>::load("input.bmp").unwrap();
let gray = img.to_grayscale();
let edges = gray.to_f32().sobel_magnitude();
let blurred = img.to_f32().gaussian_blur(1.5);

// Geometric transforms
let resized = img.resize(640, 480, ResizeMethod::Bilinear);
let cropped = img.crop(10, 10, 200, 200).unwrap();
let flipped = img.flip_horizontal();
```

## License

MIT
