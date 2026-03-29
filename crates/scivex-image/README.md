# scivex-image

Image processing for Scivex. Load, transform, and analyze images with
efficient Rust implementations.

## Highlights

- **Image types** — Grayscale, RGB, RGBA with generic pixel types
- **I/O** — PNG, JPEG, BMP, PPM reading and writing
- **Transforms** — Resize (bilinear, nearest), crop, rotate, flip, pad
- **Filters** — Gaussian blur, sharpen, edge detection (Sobel, Canny)
- **Morphology** — Erosion, dilation, opening, closing
- **Color** — RGB/HSV/HSL/grayscale conversions, histogram equalization
- **Drawing** — Lines, rectangles, circles, text rendering
- **Convolution** — Custom kernel convolution and separable filters

## Usage

```rust
use scivex_image::prelude::*;

let img = Image::read_png("photo.png").unwrap();
let resized = img.resize(224, 224, Interpolation::Bilinear);
let gray = resized.to_grayscale();
let edges = gray.sobel();
edges.write_png("edges.png").unwrap();
```

## License

MIT
