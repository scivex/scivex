//! Audio feature extraction: mel spectrogram, MFCC, chroma, pitch detection.

use scivex_core::{Float, Tensor};

use crate::error::{Result, SignalError};
use crate::spectral;

// ---------------------------------------------------------------------------
// Mel-scale conversions
// ---------------------------------------------------------------------------

/// Convert a frequency in Hertz to the mel scale (HTK formula).
///
/// `mel = 2595 * log10(1 + hz / 700)`
///
/// # Examples
///
/// ```
/// # use scivex_signal::features::hz_to_mel;
/// let mel = hz_to_mel(1000.0_f64);
/// assert!((mel - 1000.0).abs() < 100.0); // ~1000 mel at 1000 Hz
/// ```
pub fn hz_to_mel<T: Float>(hz: T) -> T {
    let one = T::one();
    let c2595 = T::from_f64(2595.0);
    let c700 = T::from_f64(700.0);
    c2595 * (one + hz / c700).log10()
}

/// Convert a mel-scale value back to Hertz.
///
/// `hz = 700 * (10^(mel / 2595) - 1)`
///
/// # Examples
///
/// ```
/// # use scivex_signal::features::{hz_to_mel, mel_to_hz};
/// let mel = hz_to_mel(440.0_f64);
/// let hz = mel_to_hz(mel);
/// assert!((hz - 440.0).abs() < 1e-4);
/// ```
pub fn mel_to_hz<T: Float>(mel: T) -> T {
    let one = T::one();
    let c700 = T::from_f64(700.0);
    let c2595 = T::from_f64(2595.0);
    let ten = T::from_f64(10.0);
    c700 * (ten.powf(mel / c2595) - one)
}

// ---------------------------------------------------------------------------
// Mel filterbank
// ---------------------------------------------------------------------------

/// Create a bank of `n_mels` triangular filters spaced linearly in the mel
/// scale between `fmin` and `fmax`.
///
/// Returns a `Vec<Vec<T>>` of shape `[n_mels][n_fft/2+1]`.
///
/// # Examples
///
/// ```
/// # use scivex_signal::features::mel_filterbank;
/// let bank = mel_filterbank(26, 512, 22050.0_f64, 0.0, 11025.0).unwrap();
/// assert_eq!(bank.len(), 26);
/// assert_eq!(bank[0].len(), 257); // n_fft/2 + 1
/// ```
#[allow(clippy::too_many_lines)]
pub fn mel_filterbank<T: Float>(
    n_mels: usize,
    n_fft: usize,
    sample_rate: T,
    fmin: T,
    fmax: T,
) -> Result<Vec<Vec<T>>> {
    if n_mels == 0 {
        return Err(SignalError::InvalidParameter {
            name: "n_mels",
            reason: "must be > 0",
        });
    }
    if n_fft == 0 {
        return Err(SignalError::InvalidParameter {
            name: "n_fft",
            reason: "must be > 0",
        });
    }

    let freq_bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // n_mels + 2 points: left edge, centre of each filter, right edge.
    let n_points = n_mels + 2;
    let mut mel_points = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let frac = T::from_usize(i) / T::from_usize(n_points - 1);
        mel_points.push(mel_min + frac * (mel_max - mel_min));
    }

    // Convert mel points back to Hz, then to FFT bin indices (fractional).
    let two = T::from_f64(2.0);
    let n_fft_t = T::from_usize(n_fft);
    let hz_points: Vec<T> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<T> = hz_points
        .iter()
        .map(|&h| h * n_fft_t / sample_rate)
        .collect();

    let zero = T::zero();
    let one = T::one();
    let mut filters: Vec<Vec<T>> = Vec::with_capacity(n_mels);

    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        let mut filt = vec![zero; freq_bins];
        for (k, filt_val) in filt.iter_mut().enumerate() {
            let k_t = T::from_usize(k);
            let denom_up = center - left;
            let denom_down = right - center;

            if denom_up > zero && k_t >= left && k_t <= center {
                *filt_val = (k_t - left) / denom_up;
            } else if denom_down > zero && k_t > center && k_t <= right {
                *filt_val = (right - k_t) / denom_down;
            }
        }

        // Normalise so that the filter area is proportional to bandwidth.
        let area: T = filt.iter().copied().fold(zero, |a, b| a + b);
        if area > zero {
            let inv = two / (right - left + one);
            for v in &mut filt {
                if *v > zero {
                    *v *= inv;
                }
            }
        }

        filters.push(filt);
    }

    Ok(filters)
}

// ---------------------------------------------------------------------------
// Mel spectrogram
// ---------------------------------------------------------------------------

/// Compute the mel-scaled power spectrogram of a 1-D signal.
///
/// 1. Compute the power spectrogram via [`spectral::spectrogram`].
/// 2. Multiply each frame by the mel filterbank.
///
/// Returns a tensor of shape `[n_frames, n_mels]`.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_signal::features::mel_spectrogram;
/// let signal: Vec<f64> = (0..1024)
///     .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 16000.0).sin())
///     .collect();
/// let x = Tensor::from_vec(signal, vec![1024]).unwrap();
/// let ms = mel_spectrogram(&x, 16000.0, 256, 128, 40).unwrap();
/// assert_eq!(ms.shape()[1], 40);
/// ```
pub fn mel_spectrogram<T: Float>(
    x: &Tensor<T>,
    sample_rate: T,
    n_fft: usize,
    hop_size: usize,
    n_mels: usize,
) -> Result<Tensor<T>> {
    let power = spectral::spectrogram(x, n_fft, hop_size)?;
    let n_frames = power.shape()[0];
    let freq_bins = power.shape()[1];
    let pd = power.as_slice();

    let fmax = sample_rate / T::from_f64(2.0);
    let bank = mel_filterbank(n_mels, n_fft, sample_rate, T::zero(), fmax)?;

    let mut out = vec![T::zero(); n_frames * n_mels];
    for f in 0..n_frames {
        for m in 0..n_mels {
            let mut sum = T::zero();
            let bins = bank[m].len().min(freq_bins);
            for b in 0..bins {
                sum += pd[f * freq_bins + b] * bank[m][b];
            }
            out[f * n_mels + m] = sum;
        }
    }

    Ok(Tensor::from_vec(out, vec![n_frames, n_mels])?)
}

// ---------------------------------------------------------------------------
// DCT-II (private helper for MFCC)
// ---------------------------------------------------------------------------

/// Type-II Discrete Cosine Transform, returning the first `n_out` coefficients.
fn dct_type2<T: Float>(input: &[T], n_out: usize) -> Vec<T> {
    let n = input.len();
    if n == 0 {
        return vec![T::zero(); n_out];
    }
    let n_t = T::from_usize(n);
    let pi = T::from_f64(std::f64::consts::PI);
    let half = T::from_f64(0.5);

    let mut out = Vec::with_capacity(n_out);
    for k in 0..n_out {
        let k_t = T::from_usize(k);
        let mut sum = T::zero();
        for (i, &val) in input.iter().enumerate() {
            let i_t = T::from_usize(i);
            sum += val * ((pi * k_t * (i_t + half)) / n_t).cos();
        }
        out.push(sum);
    }
    out
}

// ---------------------------------------------------------------------------
// MFCC
// ---------------------------------------------------------------------------

/// Compute Mel-Frequency Cepstral Coefficients.
///
/// 1. Compute mel spectrogram.
/// 2. Take the log.
/// 3. Apply a Type-II DCT to each frame, keeping the first `n_mfcc` coefficients.
///
/// Returns a tensor of shape `[n_frames, n_mfcc]`.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_signal::features::mfcc;
/// let signal: Vec<f64> = (0..1024)
///     .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 16000.0).sin())
///     .collect();
/// let x = Tensor::from_vec(signal, vec![1024]).unwrap();
/// let coeffs = mfcc(&x, 16000.0, 13, 40, 256, 128).unwrap();
/// assert_eq!(coeffs.shape()[1], 13);
/// ```
pub fn mfcc<T: Float>(
    x: &Tensor<T>,
    sample_rate: T,
    n_mfcc: usize,
    n_mels: usize,
    n_fft: usize,
    hop_size: usize,
) -> Result<Tensor<T>> {
    if n_mfcc == 0 {
        return Err(SignalError::InvalidParameter {
            name: "n_mfcc",
            reason: "must be > 0",
        });
    }
    if n_mfcc > n_mels {
        return Err(SignalError::InvalidParameter {
            name: "n_mfcc",
            reason: "must be <= n_mels",
        });
    }

    let mel = mel_spectrogram(x, sample_rate, n_fft, hop_size, n_mels)?;
    let n_frames = mel.shape()[0];
    let md = mel.as_slice();

    let eps = T::from_f64(1e-10);
    let mut out = vec![T::zero(); n_frames * n_mfcc];

    for f in 0..n_frames {
        // Build log-mel vector for this frame.
        let mut log_mel = vec![T::zero(); n_mels];
        for m in 0..n_mels {
            let val = md[f * n_mels + m];
            // Clamp to eps to avoid log(0).
            let clamped = if val > eps { val } else { eps };
            log_mel[m] = clamped.ln();
        }

        let coeffs = dct_type2(&log_mel, n_mfcc);
        for k in 0..n_mfcc {
            out[f * n_mfcc + k] = coeffs[k];
        }
    }

    Ok(Tensor::from_vec(out, vec![n_frames, n_mfcc])?)
}

// ---------------------------------------------------------------------------
// Chroma STFT
// ---------------------------------------------------------------------------

/// Compute a chromagram by mapping FFT frequency bins to pitch classes.
///
/// Returns a tensor of shape `[n_frames, n_chroma]`, typically `n_chroma = 12`.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_signal::features::chroma_stft;
/// let signal: Vec<f64> = (0..2048)
///     .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 22050.0).sin())
///     .collect();
/// let x = Tensor::from_vec(signal, vec![2048]).unwrap();
/// let ch = chroma_stft(&x, 22050.0, 512, 256, 12).unwrap();
/// assert_eq!(ch.shape()[1], 12);
/// ```
#[allow(clippy::cast_possible_truncation)]
pub fn chroma_stft<T: Float>(
    x: &Tensor<T>,
    sample_rate: T,
    n_fft: usize,
    hop_size: usize,
    n_chroma: usize,
) -> Result<Tensor<T>> {
    if n_chroma == 0 {
        return Err(SignalError::InvalidParameter {
            name: "n_chroma",
            reason: "must be > 0",
        });
    }

    let power = spectral::spectrogram(x, n_fft, hop_size)?;
    let n_frames = power.shape()[0];
    let freq_bins = power.shape()[1];
    let pd = power.as_slice();

    let n_fft_t = T::from_usize(n_fft);
    let n_chroma_t = T::from_usize(n_chroma);
    let twelve = T::from_f64(12.0);
    let a440 = T::from_f64(440.0);
    let c_ref = a440 * T::from_f64(2.0).powf(T::from_f64(-9.0) / twelve); // C4 ≈ 261.6 Hz
    let zero = T::zero();

    let mut out = vec![zero; n_frames * n_chroma];

    for f in 0..n_frames {
        for b in 1..freq_bins {
            // Frequency of this bin.
            let freq = T::from_usize(b) * sample_rate / n_fft_t;
            if freq <= zero {
                continue;
            }

            // Pitch class: 12 * log2(freq / c_ref) mod n_chroma.
            let ratio = freq / c_ref;
            let log2_ratio = ratio.ln() / T::from_f64(2.0_f64.ln());
            let pitch_continuous = twelve * log2_ratio;
            // Map to [0, n_chroma) via modulo.
            let pc_f64 = pitch_continuous.to_f64();
            let nc_f64 = n_chroma_t.to_f64();
            let rem = ((pc_f64 % nc_f64) + nc_f64) % nc_f64;
            let chroma_idx = rem.floor() as usize % n_chroma;

            out[f * n_chroma + chroma_idx] += pd[f * freq_bins + b];
        }
    }

    Ok(Tensor::from_vec(out, vec![n_frames, n_chroma])?)
}

// ---------------------------------------------------------------------------
// YIN pitch detection
// ---------------------------------------------------------------------------

/// YIN pitch detection algorithm.
///
/// Estimates the fundamental frequency for each frame of the signal.
/// Returns `0.0` for frames where no clear pitch is detected.
///
/// `frame_size` — analysis window length in samples.
/// `hop_size` — step between successive frames.
/// `threshold` — cumulative mean normalised difference threshold (typical: 0.1--0.2).
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_signal::features::pitch_yin;
/// let sr = 16000.0_f64;
/// let signal: Vec<f64> = (0..4096)
///     .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sr).sin())
///     .collect();
/// let x = Tensor::from_vec(signal, vec![4096]).unwrap();
/// let pitches = pitch_yin(&x, sr, 1024, 512, 0.2).unwrap();
/// assert!(!pitches.is_empty());
/// ```
#[allow(clippy::too_many_lines)]
pub fn pitch_yin<T: Float>(
    x: &Tensor<T>,
    sample_rate: T,
    frame_size: usize,
    hop_size: usize,
    threshold: T,
) -> Result<Vec<T>> {
    if x.ndim() != 1 {
        return Err(SignalError::InvalidParameter {
            name: "x",
            reason: "pitch_yin expects a 1-D signal",
        });
    }
    if frame_size == 0 || hop_size == 0 {
        return Err(SignalError::InvalidParameter {
            name: "frame_size/hop_size",
            reason: "must be > 0",
        });
    }

    let xs = x.as_slice();
    let n = xs.len();
    if n == 0 {
        return Err(SignalError::EmptyInput);
    }

    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0);

    let half_w = frame_size / 2;
    let mut pitches = Vec::new();

    let mut start = 0;
    while start + frame_size <= n {
        let frame = &xs[start..start + frame_size];

        // Step 1: Difference function d(tau).
        let mut diff = vec![zero; half_w];
        diff[0] = zero;
        for tau in 1..half_w {
            let mut sum = zero;
            for j in 0..half_w {
                let delta = frame[j] - frame[j + tau];
                sum += delta * delta;
            }
            diff[tau] = sum;
        }

        // Step 2: Cumulative mean normalised difference.
        let mut cmnd = vec![zero; half_w];
        cmnd[0] = one;
        let mut running_sum = zero;
        for tau in 1..half_w {
            running_sum += diff[tau];
            if running_sum > zero {
                cmnd[tau] = diff[tau] * T::from_usize(tau) / running_sum;
            } else {
                cmnd[tau] = one;
            }
        }

        // Step 3: Absolute threshold — find the first tau where cmnd < threshold.
        let mut tau_est: Option<usize> = None;
        for tau in 2..half_w {
            if cmnd[tau] < threshold {
                // Look for the local minimum.
                let mut best_tau = tau;
                let mut best_val = cmnd[tau];
                let search_end = (tau + (half_w - tau) / 4).min(half_w);
                for (t, &cval) in cmnd.iter().enumerate().take(search_end).skip(tau + 1) {
                    if cval < best_val {
                        best_val = cval;
                        best_tau = t;
                    } else {
                        break;
                    }
                }
                tau_est = Some(best_tau);
                break;
            }
        }

        // Step 4: Parabolic interpolation around the chosen tau.
        let pitch = match tau_est {
            Some(tau) if tau > 0 && tau + 1 < half_w => {
                let s0 = cmnd[tau - 1];
                let s1 = cmnd[tau];
                let s2 = cmnd[tau + 1];
                let denom = s0 - two * s1 + s2;
                let adjustment = if denom.abs() > T::from_f64(1e-12) {
                    (s0 - s2) / (two * denom)
                } else {
                    zero
                };
                let tau_refined = T::from_usize(tau) + adjustment;
                if tau_refined > zero {
                    sample_rate / tau_refined
                } else {
                    zero
                }
            }
            Some(tau) if tau > 0 => sample_rate / T::from_usize(tau),
            _ => zero,
        };

        pitches.push(pitch);
        start += hop_size;
    }

    Ok(pitches)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-4;

    #[test]
    fn test_hz_mel_roundtrip() {
        let freqs = [0.0, 100.0, 440.0, 1000.0, 8000.0, 16000.0];
        for &hz in &freqs {
            let mel = hz_to_mel(hz);
            let back = mel_to_hz(mel);
            assert!(
                (hz - back).abs() < TOL,
                "roundtrip failed: {hz} -> {mel} -> {back}"
            );
        }
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let n_mels = 26;
        let n_fft = 512;
        let sr = 22050.0_f64;
        let bank = mel_filterbank(n_mels, n_fft, sr, 0.0, sr / 2.0).unwrap();
        assert_eq!(bank.len(), n_mels);
        let expected_bins = n_fft / 2 + 1;
        for (i, filt) in bank.iter().enumerate() {
            assert_eq!(filt.len(), expected_bins, "filter {i} has wrong length");
        }
    }

    #[test]
    fn test_mel_filterbank_nonneg() {
        let bank = mel_filterbank(10, 256, 16000.0_f64, 0.0, 8000.0).unwrap();
        for filt in &bank {
            for &v in filt {
                assert!(v >= 0.0, "negative filter value: {v}");
            }
        }
    }

    #[test]
    fn test_mel_spectrogram_shape() {
        let n = 1024;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 16000.0).sin())
            .collect();
        let x = Tensor::from_vec(data, vec![n]).unwrap();
        let ms = mel_spectrogram(&x, 16000.0, 256, 128, 40).unwrap();
        assert_eq!(ms.ndim(), 2);
        assert_eq!(ms.shape()[1], 40);
        // All values should be non-negative.
        assert!(ms.as_slice().iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_mfcc_shape() {
        let n = 1024;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 16000.0).sin())
            .collect();
        let x = Tensor::from_vec(data, vec![n]).unwrap();
        let c = mfcc(&x, 16000.0, 13, 40, 256, 128).unwrap();
        assert_eq!(c.ndim(), 2);
        assert_eq!(c.shape()[1], 13);
    }

    #[test]
    fn test_dct_type2_known() {
        // DCT-II of [1,1,1,1] should be [4,0,0,0].
        let input = [1.0_f64, 1.0, 1.0, 1.0];
        let result = dct_type2(&input, 4);
        assert!((result[0] - 4.0).abs() < TOL, "DC = {}", result[0]);
        for (k, &v) in result.iter().enumerate().skip(1) {
            assert!(v.abs() < TOL, "coeff[{k}] = {v}, expected ~0");
        }
    }

    #[test]
    fn test_chroma_shape() {
        let n = 2048;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 22050.0).sin())
            .collect();
        let x = Tensor::from_vec(data, vec![n]).unwrap();
        let ch = chroma_stft(&x, 22050.0, 512, 256, 12).unwrap();
        assert_eq!(ch.ndim(), 2);
        assert_eq!(ch.shape()[1], 12);
        // All values should be non-negative.
        assert!(ch.as_slice().iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_pitch_yin_sine() {
        // Generate a 440 Hz sine wave at 16 kHz sample rate.
        let sr = 16000.0_f64;
        let freq = 440.0_f64;
        let n = 4096;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin())
            .collect();
        let x = Tensor::from_vec(data, vec![n]).unwrap();
        let pitches = pitch_yin(&x, sr, 1024, 512, 0.2).unwrap();
        assert!(!pitches.is_empty());

        // At least some frames should detect ~ 440 Hz.
        let detected: Vec<_> = pitches
            .iter()
            .filter(|&&p| p > 400.0 && p < 480.0)
            .collect();
        assert!(
            !detected.is_empty(),
            "no frame detected ~440 Hz, got: {pitches:?}"
        );
    }

    #[test]
    fn test_pitch_yin_silence() {
        let n = 2048;
        let data = vec![0.0_f64; n];
        let x = Tensor::from_vec(data, vec![n]).unwrap();
        let pitches = pitch_yin(&x, 16000.0, 1024, 512, 0.1).unwrap();
        // Silence should yield 0.0 for all frames.
        for (i, &p) in pitches.iter().enumerate() {
            assert!(p.abs() < 1.0, "frame {i}: expected ~0 for silence, got {p}");
        }
    }
}
