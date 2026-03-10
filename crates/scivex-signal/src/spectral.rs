//! Spectral analysis: STFT, inverse STFT, spectrogram, periodogram, Welch.

use scivex_core::{Float, Tensor, fft};

use crate::error::{Result, SignalError};
use crate::window;

/// Short-Time Fourier Transform.
///
/// `x` — input signal, shape `[N]`.
/// `window_size` — length of each frame.
/// `hop_size` — step between successive frames.
/// `win` — optional window tensor of length `window_size`. Defaults to Hann.
///
/// Returns a complex tensor of shape `[num_frames, freq_bins, 2]` where
/// `freq_bins = padded_window_size / 2 + 1` and padded is the next power of two >= `window_size`.
pub fn stft<T: Float>(
    x: &Tensor<T>,
    window_size: usize,
    hop_size: usize,
    win: Option<&Tensor<T>>,
) -> Result<Tensor<T>> {
    if x.ndim() != 1 {
        return Err(SignalError::InvalidParameter {
            name: "x",
            reason: "stft expects a 1-D signal",
        });
    }
    if window_size == 0 || hop_size == 0 {
        return Err(SignalError::InvalidParameter {
            name: "window_size/hop_size",
            reason: "must be > 0",
        });
    }

    let xs = x.as_slice();
    let n = xs.len();
    if n == 0 {
        return Err(SignalError::EmptyInput);
    }

    let default_win;
    let w = if let Some(w) = win {
        if w.ndim() != 1 || w.numel() != window_size {
            return Err(SignalError::DimensionMismatch {
                expected: window_size,
                got: w.numel(),
            });
        }
        w
    } else {
        default_win = window::hann::<T>(window_size)?;
        &default_win
    };
    let ws = w.as_slice();

    // Number of frames.
    let num_frames = if n >= window_size {
        (n - window_size) / hop_size + 1
    } else {
        1
    };

    let padded = window_size.next_power_of_two();
    let freq_bins = padded / 2 + 1;

    let mut output = vec![T::zero(); num_frames * freq_bins * 2];

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;

        // Extract and window the frame, zero-pad to power of two.
        let mut frame_data = vec![T::zero(); padded];
        let copy_len = window_size.min(n.saturating_sub(start));
        for i in 0..copy_len {
            frame_data[i] = xs[start + i] * ws[i];
        }

        let frame_tensor = Tensor::from_vec(frame_data, vec![padded])?;
        let spectrum = fft::rfft(&frame_tensor)?;
        let ss = spectrum.as_slice();

        let out_offset = frame_idx * freq_bins * 2;
        let copy_bins = freq_bins.min(spectrum.shape()[0]);
        for i in 0..copy_bins {
            output[out_offset + i * 2] = ss[i * 2];
            output[out_offset + i * 2 + 1] = ss[i * 2 + 1];
        }
    }

    Ok(Tensor::from_vec(output, vec![num_frames, freq_bins, 2])?)
}

/// Inverse Short-Time Fourier Transform via overlap-add.
///
/// `stft_result` — complex tensor of shape `[num_frames, freq_bins, 2]`.
/// `window_size` — original window size used in the forward STFT.
/// `hop_size` — step between successive frames.
/// `win` — optional window tensor. Defaults to Hann.
///
/// Returns the reconstructed real signal.
pub fn istft<T: Float>(
    stft_result: &Tensor<T>,
    window_size: usize,
    hop_size: usize,
    win: Option<&Tensor<T>>,
) -> Result<Tensor<T>> {
    if stft_result.ndim() != 3 || stft_result.shape()[2] != 2 {
        return Err(SignalError::InvalidParameter {
            name: "stft_result",
            reason: "expected shape [num_frames, freq_bins, 2]",
        });
    }

    let num_frames = stft_result.shape()[0];
    let freq_bins = stft_result.shape()[1];
    let padded = (freq_bins - 1) * 2;

    let default_win;
    let w = if let Some(w) = win {
        w
    } else {
        default_win = window::hann::<T>(window_size)?;
        &default_win
    };
    let ws = w.as_slice();

    let out_len = (num_frames - 1) * hop_size + window_size;
    let mut output = vec![T::zero(); out_len];
    let mut window_sum = vec![T::zero(); out_len];

    let stft_data = stft_result.as_slice();

    for frame_idx in 0..num_frames {
        // Extract this frame's spectrum.
        let offset = frame_idx * freq_bins * 2;
        let spec_data: Vec<T> = stft_data[offset..offset + freq_bins * 2].to_vec();
        let spec = Tensor::from_vec(spec_data, vec![freq_bins, 2])?;

        let frame = fft::irfft(&spec, padded)?;
        let fs = frame.as_slice();

        let start = frame_idx * hop_size;
        for i in 0..window_size.min(padded) {
            if start + i < out_len {
                output[start + i] += fs[i] * ws[i];
                window_sum[start + i] += ws[i] * ws[i];
            }
        }
    }

    // Normalize by window sum (avoid division by zero).
    let eps = T::from_f64(1e-10);
    for i in 0..out_len {
        if window_sum[i] > eps {
            output[i] /= window_sum[i];
        }
    }

    Ok(Tensor::from_vec(output, vec![out_len])?)
}

/// Compute a power spectrogram: `|STFT|^2`.
///
/// Returns a real tensor of shape `[num_frames, freq_bins]`.
pub fn spectrogram<T: Float>(
    x: &Tensor<T>,
    window_size: usize,
    hop_size: usize,
) -> Result<Tensor<T>> {
    let s = stft(x, window_size, hop_size, None)?;
    let num_frames = s.shape()[0];
    let freq_bins = s.shape()[1];
    let sd = s.as_slice();

    let mut power = vec![T::zero(); num_frames * freq_bins];
    for f in 0..num_frames {
        for b in 0..freq_bins {
            let idx = (f * freq_bins + b) * 2;
            let re = sd[idx];
            let im = sd[idx + 1];
            power[f * freq_bins + b] = re * re + im * im;
        }
    }

    Ok(Tensor::from_vec(power, vec![num_frames, freq_bins])?)
}

/// Periodogram: power spectral density estimate via FFT.
///
/// Returns `(frequencies, psd)` where both are 1-D tensors.
/// Frequencies are normalized to [0, 0.5] (fraction of sample rate).
pub fn periodogram<T: Float>(x: &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>)> {
    if x.ndim() != 1 {
        return Err(SignalError::InvalidParameter {
            name: "x",
            reason: "periodogram expects a 1-D signal",
        });
    }
    let n = x.numel();
    if n == 0 {
        return Err(SignalError::EmptyInput);
    }

    let spectrum = fft::rfft(x)?;
    let freq_bins = spectrum.shape()[0];
    let sd = spectrum.as_slice();
    let n_f = T::from_usize(n);

    let mut psd = vec![T::zero(); freq_bins];
    let mut freqs = vec![T::zero(); freq_bins];

    for i in 0..freq_bins {
        let re = sd[i * 2];
        let im = sd[i * 2 + 1];
        psd[i] = (re * re + im * im) / n_f;
        freqs[i] = T::from_usize(i) / T::from_usize(n.next_power_of_two());
    }

    // Scale non-DC and non-Nyquist bins by 2 for one-sided spectrum.
    let last = freq_bins - 1;
    for p in psd.iter_mut().take(last).skip(1) {
        *p *= T::from_f64(2.0);
    }

    Ok((
        Tensor::from_vec(freqs, vec![freq_bins])?,
        Tensor::from_vec(psd, vec![freq_bins])?,
    ))
}

/// Welch's method for power spectral density estimation.
///
/// Averages periodograms of overlapping segments.
///
/// `x` — input signal, shape `[N]`.
/// `segment_size` — length of each segment.
/// `overlap` — number of overlapping samples between segments.
///
/// Returns `(frequencies, psd)`.
pub fn welch<T: Float>(
    x: &Tensor<T>,
    segment_size: usize,
    overlap: usize,
) -> Result<(Tensor<T>, Tensor<T>)> {
    if x.ndim() != 1 {
        return Err(SignalError::InvalidParameter {
            name: "x",
            reason: "welch expects a 1-D signal",
        });
    }
    if segment_size == 0 {
        return Err(SignalError::InvalidParameter {
            name: "segment_size",
            reason: "must be > 0",
        });
    }
    if overlap >= segment_size {
        return Err(SignalError::InvalidParameter {
            name: "overlap",
            reason: "overlap must be less than segment_size",
        });
    }

    let xs = x.as_slice();
    let n = xs.len();
    if n == 0 {
        return Err(SignalError::EmptyInput);
    }

    let step = segment_size - overlap;
    let win = window::hann::<T>(segment_size)?;
    let ws = win.as_slice();

    let padded = segment_size.next_power_of_two();
    let freq_bins = padded / 2 + 1;

    let mut avg_psd = vec![T::zero(); freq_bins];
    let mut num_segments = 0usize;

    let mut start = 0;
    while start + segment_size <= n {
        // Window the segment and zero-pad.
        let mut seg = vec![T::zero(); padded];
        for i in 0..segment_size {
            seg[i] = xs[start + i] * ws[i];
        }
        let seg_tensor = Tensor::from_vec(seg, vec![padded])?;
        let spectrum = fft::rfft(&seg_tensor)?;
        let sd = spectrum.as_slice();

        let seg_f = T::from_usize(segment_size);
        for i in 0..freq_bins {
            let re = sd[i * 2];
            let im = sd[i * 2 + 1];
            avg_psd[i] += (re * re + im * im) / seg_f;
        }

        num_segments += 1;
        start += step;
    }

    if num_segments == 0 {
        return Err(SignalError::InvalidParameter {
            name: "segment_size",
            reason: "signal is shorter than segment_size",
        });
    }

    let num_seg_f = T::from_usize(num_segments);
    let mut freqs = vec![T::zero(); freq_bins];
    for i in 0..freq_bins {
        avg_psd[i] /= num_seg_f;
        freqs[i] = T::from_usize(i) / T::from_usize(padded);
    }

    // One-sided scaling.
    let last = freq_bins - 1;
    for p in avg_psd.iter_mut().take(last).skip(1) {
        *p *= T::from_f64(2.0);
    }

    Ok((
        Tensor::from_vec(freqs, vec![freq_bins])?,
        Tensor::from_vec(avg_psd, vec![freq_bins])?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    #[test]
    fn test_stft_istft_roundtrip() {
        // Constant signal: STFT->iSTFT should recover it.
        let n = 64;
        let data = vec![1.0_f64; n];
        let x = Tensor::from_vec(data.clone(), vec![n]).unwrap();
        let s = stft(&x, 16, 8, None).unwrap();
        let y = istft(&s, 16, 8, None).unwrap();

        // Check the middle portion (edges may have boundary effects).
        let ys = y.as_slice();
        for (i, &val) in ys.iter().enumerate().take(n - 8).skip(8) {
            assert!(
                (val - 1.0).abs() < 0.1,
                "sample {i}: got {val}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_spectrogram_sine() {
        // Generate a sine wave at bin frequency.
        let n = 128;
        let window_size = 32;
        let padded: usize = 32; // already power of 2
        let freq_idx: usize = 4; // target frequency bin
        let two_pi = 2.0 * std::f64::consts::PI;
        let data: Vec<f64> = (0..n)
            .map(|i| (two_pi * freq_idx as f64 * i as f64 / padded as f64).cos())
            .collect();
        let x = Tensor::from_vec(data, vec![n]).unwrap();
        let spec = spectrogram(&x, window_size, 16).unwrap();
        let num_frames = spec.shape()[0];
        let freq_bins = spec.shape()[1];
        let sd = spec.as_slice();

        // For interior frames, the peak should be at freq_idx.
        for f in 1..num_frames - 1 {
            let mut max_bin = 0;
            let mut max_val = 0.0_f64;
            for b in 0..freq_bins {
                let v = sd[f * freq_bins + b];
                if v > max_val {
                    max_val = v;
                    max_bin = b;
                }
            }
            assert_eq!(max_bin, freq_idx, "frame {f}: peak at {max_bin}");
        }
    }

    #[test]
    fn test_periodogram_parseval() {
        // Parseval's theorem: sum(|x|^2) ≈ sum(psd) * N / sample_rate
        let data = vec![1.0_f64, 0.5, -0.3, 0.7, 0.2, -0.5, 0.1, 0.8];
        let n = data.len();
        let time_energy: f64 = data.iter().map(|&x| x * x).sum();
        let x = Tensor::from_vec(data, vec![n]).unwrap();
        let (_, psd) = periodogram(&x).unwrap();
        let psd_sum: f64 = psd.as_slice().iter().sum();
        // These should be approximately equal.
        assert!(
            (time_energy - psd_sum).abs() < 0.5,
            "time={time_energy}, psd_sum={psd_sum}"
        );
    }

    #[test]
    fn test_welch_runs() {
        let data: Vec<f64> = (0..256).map(|i| (f64::from(i) * 0.1).sin()).collect();
        let x = Tensor::from_vec(data, vec![256]).unwrap();
        let (freqs, psd) = welch(&x, 64, 32).unwrap();
        assert!(!freqs.is_empty());
        assert_eq!(freqs.numel(), psd.numel());
        // PSD should be non-negative.
        assert!(psd.as_slice().iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_stft_empty() {
        let x = Tensor::<f64>::from_vec(vec![], vec![0]).unwrap();
        assert!(stft(&x, 4, 2, None).is_err());
    }

    #[test]
    fn test_periodogram_dc() {
        // A constant signal should have all power in DC.
        let data = vec![3.0_f64; 8];
        let x = Tensor::from_vec(data, vec![8]).unwrap();
        let (_, psd) = periodogram(&x).unwrap();
        let ps = psd.as_slice();
        // DC bin should dominate.
        let dc = ps[0];
        for (i, &p) in ps.iter().enumerate().skip(1) {
            assert!(p < dc || p < TOL, "bin {i} power {p} exceeds DC {dc}");
        }
    }
}
