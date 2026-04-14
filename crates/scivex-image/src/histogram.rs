use scivex_core::Tensor;

use crate::image::Image;

/// Compute a histogram for each channel of a `u8` image.
///
/// Returns a tensor of shape `[channels, 256]` with `u32` counts.
///
/// # Examples
///
/// ```
/// # use scivex_image::{Image, PixelFormat};
/// # use scivex_image::histogram::histogram;
/// let img = Image::from_raw(vec![0u8; 9], 3, 3, PixelFormat::Gray).unwrap();
/// let hist = histogram(&img);
/// assert_eq!(hist.shape(), &[1, 256]);
/// assert_eq!(*hist.get(&[0, 0]).unwrap(), 9); // all pixels are 0
/// ```
pub fn histogram(img: &Image<u8>) -> Tensor<u32> {
    let c = img.channels();
    let src = img.as_slice();
    let mut bins = vec![0u32; c * 256];

    // Process whole pixels to avoid per-element modulo.
    for pixel in src.chunks_exact(c) {
        for (ch, &val) in pixel.iter().enumerate() {
            bins[ch * 256 + val as usize] += 1;
        }
    }

    Tensor::from_vec(bins, vec![c, 256]).expect("shape correct")
}

/// Compute the cumulative histogram from a histogram tensor.
///
/// Input shape: `[channels, 256]`. Output has the same shape.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::histogram::{histogram, cumulative_histogram};
/// let img = Image::from_raw(vec![0u8, 1, 2, 3], 4, 1, PixelFormat::Gray).unwrap();
/// let hist = histogram(&img);
/// let cum = cumulative_histogram(&hist);
/// assert_eq!(*cum.get(&[0, 3]).unwrap(), 4);
/// ```
pub fn cumulative_histogram(hist: &Tensor<u32>) -> Tensor<u32> {
    let shape = hist.shape();
    assert!(
        shape.len() == 2 && shape[1] == 256,
        "expected [C, 256] histogram"
    );
    let c = shape[0];
    let src = hist.as_slice();
    let mut out = vec![0u32; c * 256];

    for ch in 0..c {
        let base = ch * 256;
        out[base] = src[base];
        for i in 1..256 {
            out[base + i] = out[base + i - 1] + src[base + i];
        }
    }

    Tensor::from_vec(out, vec![c, 256]).expect("shape correct")
}

/// Histogram equalization for `u8` images.
///
/// Independently equalizes each channel.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::histogram;
/// let data: Vec<u8> = (0..25).map(|i| 100 + (i % 5) as u8).collect();
/// let img = Image::from_raw(data, 5, 5, PixelFormat::Gray).unwrap();
/// let eq = histogram::equalize(&img);
/// let s = eq.as_slice();
/// assert!(s.iter().max().unwrap() - s.iter().min().unwrap() > 4);
/// ```
pub fn equalize(img: &Image<u8>) -> Image<u8> {
    let (w, h) = img.dimensions();
    let c = img.channels();
    let total_pixels = (w * h) as f64;
    let hist = histogram(img);
    let cum = cumulative_histogram(&hist);
    let cum_data = cum.as_slice();

    let src = img.as_slice();
    let mut out = vec![0u8; src.len()];

    for (i, &val) in src.iter().enumerate() {
        let ch = i % c;
        // Find CDF minimum for this channel
        let base = ch * 256;
        let cdf_min = cum_data[base..base + 256]
            .iter()
            .copied()
            .find(|&v| v > 0)
            .unwrap_or(0);
        let cdf_val = cum_data[base + val as usize];

        if total_pixels as u32 == cdf_min {
            out[i] = val;
        } else {
            let equalized = (f64::from(cdf_val - cdf_min) / (total_pixels - f64::from(cdf_min))
                * 255.0
                + 0.5) as u8;
            out[i] = equalized;
        }
    }

    Image::from_raw(out, w, h, img.format()).expect("dimensions unchanged")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PixelFormat;

    #[test]
    fn test_histogram_pure_black() {
        let img = Image::from_raw(vec![0u8; 9], 3, 3, PixelFormat::Gray).unwrap();
        let hist = histogram(&img);
        assert_eq!(hist.shape(), &[1, 256]);
        assert_eq!(*hist.get(&[0, 0]).unwrap(), 9);
        assert_eq!(*hist.get(&[0, 1]).unwrap(), 0);
    }

    #[test]
    fn test_histogram_rgb() {
        let data = vec![0, 128, 255, 0, 128, 255, 0, 128, 255];
        let img = Image::from_raw(data, 3, 1, PixelFormat::Rgb).unwrap();
        let hist = histogram(&img);
        assert_eq!(hist.shape(), &[3, 256]);
        // R channel: all 0s
        assert_eq!(*hist.get(&[0, 0]).unwrap(), 3);
        // G channel: all 128s
        assert_eq!(*hist.get(&[1, 128]).unwrap(), 3);
        // B channel: all 255s
        assert_eq!(*hist.get(&[2, 255]).unwrap(), 3);
    }

    #[test]
    fn test_cumulative_histogram() {
        let img = Image::from_raw(vec![0u8, 1, 2, 3], 4, 1, PixelFormat::Gray).unwrap();
        let hist = histogram(&img);
        let cum = cumulative_histogram(&hist);
        assert_eq!(*cum.get(&[0, 0]).unwrap(), 1);
        assert_eq!(*cum.get(&[0, 1]).unwrap(), 2);
        assert_eq!(*cum.get(&[0, 2]).unwrap(), 3);
        assert_eq!(*cum.get(&[0, 3]).unwrap(), 4);
    }

    #[test]
    fn test_equalize_broadens_range() {
        // Image with narrow range: all values between 100-104
        let data: Vec<u8> = (0..25).map(|i| 100 + (i % 5)).collect();
        let img = Image::from_raw(data, 5, 5, PixelFormat::Gray).unwrap();
        let equalized = equalize(&img);

        // After equalization, range should be broader
        let eq_data = equalized.as_slice();
        let min = *eq_data.iter().min().unwrap();
        let max = *eq_data.iter().max().unwrap();
        assert!(
            max - min > 4,
            "equalized range should be broader: {min}..{max}"
        );
    }
}
