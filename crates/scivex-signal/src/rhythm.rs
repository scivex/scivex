//! Beat tracking and onset detection for audio signals.
//!
//! Provides onset strength computation via spectral flux, onset detection
//! with adaptive thresholding, tempo estimation via autocorrelation, and
//! beat tracking that aligns beats to detected onsets.

use scivex_core::{Float, Tensor};

use crate::error::{Result, SignalError};
use crate::peak::find_peaks;
use crate::spectral;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of onset detection.
#[derive(Debug, Clone)]
pub struct OnsetResult<T: Float> {
    /// Onset strength envelope (one value per frame).
    pub onset_envelope: Vec<T>,
    /// Indices of detected onsets in the envelope.
    pub onset_frames: Vec<usize>,
    /// Onset times in seconds.
    pub onset_times: Vec<T>,
}

/// Result of beat tracking.
#[derive(Debug, Clone)]
pub struct BeatResult<T: Float> {
    /// Estimated tempo in BPM.
    pub tempo: T,
    /// Beat frame indices.
    pub beat_frames: Vec<usize>,
    /// Beat times in seconds.
    pub beat_times: Vec<T>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a frame index to time in seconds.
fn frame_to_time<T: Float>(frame: usize, hop_size: usize, sample_rate: T) -> T {
    T::from_usize(frame) * T::from_usize(hop_size) / sample_rate
}

/// Compute the magnitude spectrum from the STFT complex tensor.
///
/// Input shape: `[num_frames, freq_bins, 2]` (re, im pairs).
/// Output: `Vec<Vec<T>>` of shape `[num_frames][freq_bins]`.
fn stft_magnitudes<T: Float>(stft_result: &Tensor<T>) -> Vec<Vec<T>> {
    let shape = stft_result.shape();
    let num_frames = shape[0];
    let freq_bins = shape[1];
    let data = stft_result.as_slice();

    let mut mags = Vec::with_capacity(num_frames);
    for t in 0..num_frames {
        let mut frame_mags = Vec::with_capacity(freq_bins);
        for f in 0..freq_bins {
            let idx = (t * freq_bins + f) * 2;
            let re = data[idx];
            let im = data[idx + 1];
            frame_mags.push((re * re + im * im).sqrt());
        }
        mags.push(frame_mags);
    }
    mags
}

/// Compute spectral flux from magnitude spectra.
///
/// `SF[t] = sum_f max(0, |S[t,f]| - |S[t-1,f]|)` for t >= 1.
/// The first frame gets flux = 0.
fn spectral_flux<T: Float>(magnitudes: &[Vec<T>]) -> Vec<T> {
    let num_frames = magnitudes.len();
    let mut flux = Vec::with_capacity(num_frames);
    flux.push(T::zero());

    for t in 1..num_frames {
        let mut sum = T::zero();
        for (f, mag) in magnitudes[t].iter().enumerate() {
            let diff = *mag - magnitudes[t - 1][f];
            if diff > T::zero() {
                sum += diff;
            }
        }
        flux.push(sum);
    }
    flux
}

/// Compute the median of a slice. Returns zero for empty slices.
fn median<T: Float>(values: &[T]) -> T {
    if values.is_empty() {
        return T::zero();
    }
    let mut sorted: Vec<T> = values.to_vec();
    sorted.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / T::from_f64(2.0)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute onset strength envelope using spectral flux.
///
/// Spectral flux measures the positive change in magnitude spectrum
/// between consecutive STFT frames.
///
/// # Arguments
///
/// * `signal` — audio samples (1-D slice).
/// * `sample_rate` — sampling rate in Hz.
/// * `frame_size` — STFT frame/window size.
/// * `hop_size` — STFT hop size.
///
/// # Errors
///
/// Returns [`SignalError::EmptyInput`] when `signal` is empty and
/// [`SignalError::InvalidParameter`] for zero-valued sizes.
pub fn onset_strength<T: Float>(
    signal: &[T],
    sample_rate: T,
    frame_size: usize,
    hop_size: usize,
) -> Result<OnsetResult<T>> {
    if signal.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if frame_size == 0 || hop_size == 0 {
        return Err(SignalError::InvalidParameter {
            name: "frame_size/hop_size",
            reason: "must be > 0",
        });
    }
    if sample_rate <= T::zero() {
        return Err(SignalError::InvalidParameter {
            name: "sample_rate",
            reason: "must be positive",
        });
    }

    let x = Tensor::from_vec(signal.to_vec(), vec![signal.len()])?;
    let stft_result = spectral::stft(&x, frame_size, hop_size, None)?;
    let mags = stft_magnitudes(&stft_result);
    let envelope = spectral_flux(&mags);

    // Peak-pick the envelope for onset frames.
    let env_tensor = Tensor::from_vec(envelope.clone(), vec![envelope.len()])?;
    let onset_frames = find_peaks(&env_tensor, None, None)?;

    let onset_times: Vec<T> = onset_frames
        .iter()
        .map(|&f| frame_to_time(f, hop_size, sample_rate))
        .collect();

    Ok(OnsetResult {
        onset_envelope: envelope,
        onset_frames,
        onset_times,
    })
}

/// Detect onsets using spectral flux with adaptive thresholding.
///
/// Computes the onset strength envelope via [`onset_strength`], normalises it,
/// applies a median-based adaptive threshold, and picks peaks that exceed
/// the threshold.
///
/// # Examples
///
/// ```
/// # use scivex_signal::rhythm::detect_onsets;
/// // Generate a simple click at sample 4000.
/// let mut signal = vec![0.0_f64; 8000];
/// signal[4000] = 1.0;
/// signal[4001] = 0.8;
/// let result = detect_onsets(&signal, 8000.0, 512, 256).unwrap();
/// assert!(!result.onset_frames.is_empty());
/// ```
#[allow(clippy::too_many_lines)]
pub fn detect_onsets<T: Float>(
    signal: &[T],
    sample_rate: T,
    frame_size: usize,
    hop_size: usize,
) -> Result<OnsetResult<T>> {
    if signal.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if frame_size == 0 || hop_size == 0 {
        return Err(SignalError::InvalidParameter {
            name: "frame_size/hop_size",
            reason: "must be > 0",
        });
    }
    if sample_rate <= T::zero() {
        return Err(SignalError::InvalidParameter {
            name: "sample_rate",
            reason: "must be positive",
        });
    }

    // Step 1: compute onset strength envelope.
    let raw = onset_strength(signal, sample_rate, frame_size, hop_size)?;
    let envelope = raw.onset_envelope;

    if envelope.is_empty() {
        return Ok(OnsetResult {
            onset_envelope: envelope,
            onset_frames: vec![],
            onset_times: vec![],
        });
    }

    // Step 2: normalise envelope.
    let max_val = envelope
        .iter()
        .copied()
        .fold(T::zero(), |a, b| if b > a { b } else { a });

    let norm_env: Vec<T> = if max_val > T::zero() {
        envelope.iter().map(|&v| v / max_val).collect()
    } else {
        envelope.clone()
    };

    // Step 3: adaptive threshold = median(window) + delta.
    let delta = T::from_f64(0.1);
    let half_w: usize = 5; // look +/- 5 frames
    let n = norm_env.len();
    let mut threshold = Vec::with_capacity(n);
    for t in 0..n {
        let lo = t.saturating_sub(half_w);
        let hi = if t + half_w + 1 < n {
            t + half_w + 1
        } else {
            n
        };
        let med = median(&norm_env[lo..hi]);
        threshold.push(med + delta);
    }

    // Step 4: pick peaks above threshold.
    // Build a thresholded envelope (zero out below threshold).
    let thresholded: Vec<T> = norm_env
        .iter()
        .zip(threshold.iter())
        .map(|(&v, &th)| if v > th { v } else { T::zero() })
        .collect();

    let env_tensor = Tensor::from_vec(thresholded, vec![n])?;
    let onset_frames = find_peaks(&env_tensor, Some(T::from_f64(1e-6)), None)?;

    // Step 5: convert to times.
    let onset_times: Vec<T> = onset_frames
        .iter()
        .map(|&f| frame_to_time(f, hop_size, sample_rate))
        .collect();

    Ok(OnsetResult {
        onset_envelope: envelope,
        onset_frames,
        onset_times,
    })
}

/// Estimate tempo in BPM from an onset strength envelope.
///
/// Uses autocorrelation of the envelope, searching for the dominant
/// periodicity within a reasonable BPM range (30--300 BPM).
///
/// # Arguments
///
/// * `onset_envelope` — onset strength values, one per frame.
/// * `sample_rate` — audio sample rate in Hz.
/// * `hop_size` — STFT hop size used to produce the envelope.
///
/// # Errors
///
/// Returns an error if the envelope is too short to estimate tempo.
#[allow(clippy::too_many_lines)]
pub fn estimate_tempo<T: Float>(
    onset_envelope: &[T],
    sample_rate: T,
    hop_size: usize,
) -> Result<T> {
    let n = onset_envelope.len();
    if n < 4 {
        return Err(SignalError::InvalidParameter {
            name: "onset_envelope",
            reason: "too short to estimate tempo (need >= 4 frames)",
        });
    }
    if sample_rate <= T::zero() {
        return Err(SignalError::InvalidParameter {
            name: "sample_rate",
            reason: "must be positive",
        });
    }
    if hop_size == 0 {
        return Err(SignalError::InvalidParameter {
            name: "hop_size",
            reason: "must be > 0",
        });
    }

    // Frame rate in frames per second.
    let frame_rate = sample_rate / T::from_usize(hop_size);

    // BPM range: 30--300.
    let bpm_min = T::from_f64(30.0);
    let bpm_max = T::from_f64(300.0);

    // Corresponding lag range (in frames).
    // lag = 60 * frame_rate / bpm  =>  higher bpm -> smaller lag.
    let sixty = T::from_f64(60.0);
    let lag_min_f = (sixty * frame_rate / bpm_max).to_f64();
    let lag_max_f = (sixty * frame_rate / bpm_min).to_f64();

    let lag_min = (lag_min_f.ceil() as usize).max(1);
    let lag_max = (lag_max_f.floor() as usize).min(n - 1);

    if lag_min > lag_max || lag_max >= n {
        return Err(SignalError::InvalidParameter {
            name: "onset_envelope",
            reason: "signal too short for tempo estimation in 30-300 BPM range",
        });
    }

    // Subtract mean for better autocorrelation.
    let mean = onset_envelope.iter().copied().fold(T::zero(), |a, b| a + b) / T::from_usize(n);
    let centered: Vec<T> = onset_envelope.iter().map(|&v| v - mean).collect();

    // Compute normalised autocorrelation for lags in range.
    // Normalise each lag by the number of overlapping samples to avoid
    // bias toward longer lags.
    let mut acf = Vec::with_capacity(lag_max - lag_min + 1);
    for lag in lag_min..=lag_max {
        let mut corr = T::zero();
        let overlap = n - lag;
        for i in 0..overlap {
            corr += centered[i] * centered[i + lag];
        }
        // Normalise by overlap count.
        corr /= T::from_usize(overlap);
        acf.push((lag, corr));
    }

    // Find the first prominent peak in the ACF rather than the global
    // maximum. This prefers the fundamental period over its multiples.
    let acf_vals: Vec<T> = acf.iter().map(|&(_, c)| c).collect();
    let acf_len = acf_vals.len();

    let mut best_lag = acf[0].0;
    let mut best_corr = acf[0].1;

    // Find the global max for reference.
    let global_max =
        acf_vals.iter().copied().fold(
            T::from_f64(f64::NEG_INFINITY),
            |a, b| if b > a { b } else { a },
        );

    // Threshold: accept first peak that reaches >= 50% of the global max.
    // This favours the fundamental (shortest) period over sub-harmonics.
    let threshold_ratio = T::from_f64(0.50);
    let peak_threshold = global_max * threshold_ratio;

    // Scan for the first local maximum above the threshold.
    for i in 1..acf_len.saturating_sub(1) {
        if acf_vals[i] >= acf_vals[i - 1]
            && acf_vals[i] >= acf_vals[i + 1]
            && acf_vals[i] >= peak_threshold
        {
            best_lag = acf[i].0;
            best_corr = acf_vals[i];
            break;
        }
    }

    // Fallback: if no local peak found above threshold, use global max.
    if best_corr < peak_threshold {
        for &(lag, corr) in &acf {
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }
    }

    // Suppress unused-variable warning for the fallback assignment.
    let _ = best_corr;

    // Convert lag to BPM.
    let tempo = sixty * frame_rate / T::from_usize(best_lag);
    Ok(tempo)
}

/// Estimate tempo and track beats in an audio signal.
///
/// Uses onset strength via spectral flux, autocorrelation-based tempo
/// estimation, and a greedy alignment strategy that snaps beats to nearby
/// onset peaks.
///
/// # Arguments
///
/// * `signal` — audio samples (1-D).
/// * `sample_rate` — sampling rate in Hz.
/// * `frame_size` — STFT frame/window size.
/// * `hop_size` — STFT hop size.
///
/// # Errors
///
/// Returns an error for empty signals or invalid parameters.
#[allow(clippy::too_many_lines)]
pub fn beat_track<T: Float>(
    signal: &[T],
    sample_rate: T,
    frame_size: usize,
    hop_size: usize,
) -> Result<BeatResult<T>> {
    if signal.is_empty() {
        return Err(SignalError::EmptyInput);
    }

    // Step 1: compute onset strength envelope.
    let onset = onset_strength(signal, sample_rate, frame_size, hop_size)?;
    let envelope = &onset.onset_envelope;

    if envelope.len() < 4 {
        return Ok(BeatResult {
            tempo: T::zero(),
            beat_frames: vec![],
            beat_times: vec![],
        });
    }

    // Step 2: estimate tempo.
    let tempo = estimate_tempo(envelope, sample_rate, hop_size)?;

    // Step 3: compute beat period in frames.
    let sixty = T::from_f64(60.0);
    let frame_rate = sample_rate / T::from_usize(hop_size);
    let period_f = (sixty * frame_rate / tempo).to_f64();
    let period = period_f.round() as usize;

    if period == 0 {
        return Ok(BeatResult {
            tempo,
            beat_frames: vec![],
            beat_times: vec![],
        });
    }

    let n = envelope.len();

    // Step 4: find strongest onset in the first beat period to anchor.
    let search_end = period.min(n);
    let mut start_frame = 0usize;
    let mut best_val = T::from_f64(f64::NEG_INFINITY);
    for (i, &val) in envelope.iter().enumerate().take(search_end) {
        if val > best_val {
            best_val = val;
            start_frame = i;
        }
    }

    // Step 5: place beats at roughly `period` intervals, snapping to nearby
    // onset peaks within a tolerance window.
    let tolerance = (period / 4).max(1);
    let mut beat_frames: Vec<usize> = Vec::new();

    let mut expected = start_frame;
    while expected < n {
        // Search around expected position.
        let lo = expected.saturating_sub(tolerance);
        let hi = (expected + tolerance + 1).min(n);

        // Find the sample with the highest onset strength in the window.
        let mut best_idx = expected.min(n - 1);
        let mut best_strength = T::from_f64(f64::NEG_INFINITY);
        for (i, &val) in envelope.iter().enumerate().take(hi).skip(lo) {
            if val > best_strength {
                best_strength = val;
                best_idx = i;
            }
        }

        beat_frames.push(best_idx);
        expected = best_idx + period;
    }

    // Step 6: convert to times.
    let beat_times: Vec<T> = beat_frames
        .iter()
        .map(|&f| frame_to_time(f, hop_size, sample_rate))
        .collect();

    Ok(BeatResult {
        tempo,
        beat_frames,
        beat_times,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a click train: zero signal with impulses every `interval`
    /// samples.
    fn click_train(num_samples: usize, interval: usize, amplitude: f64) -> Vec<f64> {
        let mut signal = vec![0.0_f64; num_samples];
        let mut pos = 0;
        while pos < num_samples {
            signal[pos] = amplitude;
            // Add a short burst to make the click spectrally visible.
            if pos + 1 < num_samples {
                signal[pos + 1] = amplitude * 0.5;
            }
            pos += interval;
        }
        signal
    }

    #[test]
    fn test_onset_strength_impulse() {
        // Place impulses at known positions.
        let sr = 8000.0_f64;
        let hop = 256;
        let frame = 512;
        let interval = 2000; // ~4 Hz

        let signal = click_train(16000, interval, 1.0);
        let result = onset_strength(&signal, sr, frame, hop).unwrap();

        // The onset envelope should have clear peaks.
        assert!(!result.onset_envelope.is_empty());
        // There should be at least some detected onsets near the impulses.
        assert!(
            !result.onset_frames.is_empty(),
            "expected onset frames near impulse locations"
        );
    }

    #[test]
    fn test_detect_onsets_clicks() {
        let sr = 8000.0_f64;
        let hop = 256;
        let frame = 512;
        let interval = 2000; // expect ~8000/2000 = 4 clicks per second
        let num_samples = 16000; // 2 seconds

        let signal = click_train(num_samples, interval, 1.0);
        let result = detect_onsets(&signal, sr, frame, hop).unwrap();

        // We have clicks at 0, 2000, 4000, 6000, 8000, 10000, 12000, 14000 = 8 clicks.
        // Not all may be detected (edge effects), but we should get several.
        assert!(
            result.onset_frames.len() >= 2,
            "expected multiple onsets, got {}",
            result.onset_frames.len()
        );

        // Onset times should be non-negative and bounded.
        for &t in &result.onset_times {
            assert!(t >= 0.0);
            assert!(t <= 2.0 + 0.1); // 2 seconds + tolerance
        }
    }

    #[test]
    fn test_estimate_tempo_120bpm() {
        // 120 BPM = 2 beats per second.
        // At sr=8000, hop=256: frame_rate = 8000/256 = 31.25 fps.
        // Beat period = 60/120 * 31.25 = 15.625 frames.
        let sr = 8000.0_f64;
        let hop = 256;
        let frame = 512;
        let bpm = 120.0_f64;
        let interval = (sr * 60.0 / bpm) as usize; // 4000 samples

        // 6 seconds of audio.
        let signal = click_train(48000, interval, 1.0);
        let onset = onset_strength(&signal, sr, frame, hop).unwrap();

        let estimated = estimate_tempo(&onset.onset_envelope, sr, hop).unwrap();

        // Allow +/- 15 BPM tolerance (autocorrelation has finite resolution).
        assert!(
            (estimated - bpm).abs() < 15.0,
            "expected tempo ~{bpm} BPM, got {estimated}"
        );
    }

    #[test]
    fn test_beat_track_regular() {
        let sr = 8000.0_f64;
        let hop = 256;
        let frame = 512;
        let bpm = 120.0_f64;
        let interval = (sr * 60.0 / bpm) as usize;

        let signal = click_train(48000, interval, 1.0);
        let result = beat_track(&signal, sr, frame, hop).unwrap();

        // Tempo should be reasonable.
        assert!(
            result.tempo > 60.0 && result.tempo < 240.0,
            "tempo {} out of expected range",
            result.tempo
        );

        // Should have multiple beats.
        assert!(
            result.beat_frames.len() >= 3,
            "expected at least 3 beats, got {}",
            result.beat_frames.len()
        );

        // Beat frames should be roughly evenly spaced.
        if result.beat_frames.len() >= 3 {
            let intervals: Vec<usize> =
                result.beat_frames.windows(2).map(|w| w[1] - w[0]).collect();

            // All intervals should be within 50% of each other.
            let avg = intervals.iter().sum::<usize>() as f64 / intervals.len() as f64;
            for &gap in &intervals {
                let ratio = gap as f64 / avg;
                assert!(
                    ratio > 0.5 && ratio < 2.0,
                    "beat intervals not roughly even: gap={gap}, avg={avg}"
                );
            }
        }

        // Beat times should be ascending.
        for w in result.beat_times.windows(2) {
            assert!(w[1] > w[0], "beat times should be strictly increasing");
        }
    }

    #[test]
    fn test_onset_empty_signal() {
        let result = onset_strength::<f64>(&[], 44100.0, 2048, 512);
        assert!(result.is_err());
        match result {
            Err(SignalError::EmptyInput) => {}
            other => panic!("expected EmptyInput, got {other:?}"),
        }

        let result2 = detect_onsets::<f64>(&[], 44100.0, 2048, 512);
        assert!(result2.is_err());

        let result3 = beat_track::<f64>(&[], 44100.0, 2048, 512);
        assert!(result3.is_err());
    }
}
