//! Python bindings for [`scivex_signal`] — signal processing & audio.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use scivex_core::Tensor;

use crate::tensor::PyTensor;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sig_err(e: impl std::fmt::Debug) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}"))
}

fn tensor_from_list(data: Vec<f64>) -> PyResult<Tensor<f64>> {
    let n = data.len();
    Tensor::from_vec(data, vec![n])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

// ===========================================================================
// WINDOW FUNCTIONS
// ===========================================================================

/// Generate a Hann window of length `n`. Returns a 1-D tensor.
#[pyfunction]
fn hann(n: usize) -> PyResult<PyTensor> {
    let t = scivex_signal::window::hann::<f64>(n).map_err(sig_err)?;
    Ok(PyTensor::from_f64(t))
}

/// Generate a Hamming window of length `n`. Returns a 1-D tensor.
#[pyfunction]
fn hamming(n: usize) -> PyResult<PyTensor> {
    let t = scivex_signal::window::hamming::<f64>(n).map_err(sig_err)?;
    Ok(PyTensor::from_f64(t))
}

/// Generate a Blackman window of length `n`. Returns a 1-D tensor.
#[pyfunction]
fn blackman(n: usize) -> PyResult<PyTensor> {
    let t = scivex_signal::window::blackman::<f64>(n).map_err(sig_err)?;
    Ok(PyTensor::from_f64(t))
}

/// Generate a Bartlett (triangular) window of length `n`. Returns a 1-D tensor.
#[pyfunction]
fn bartlett(n: usize) -> PyResult<PyTensor> {
    let t = scivex_signal::window::bartlett::<f64>(n).map_err(sig_err)?;
    Ok(PyTensor::from_f64(t))
}

// ===========================================================================
// FILTERS
// ===========================================================================

/// Apply a causal IIR/FIR filter defined by numerator `b` and denominator `a` to signal `x`.
/// Returns the filtered signal as a 1-D tensor.
#[pyfunction]
fn lfilter(b: Vec<f64>, a: Vec<f64>, x: Vec<f64>) -> PyResult<PyTensor> {
    let bt = tensor_from_list(b)?;
    let at = tensor_from_list(a)?;
    let xt = tensor_from_list(x)?;
    let out = scivex_signal::filter::lfilter(&bt, &at, &xt).map_err(sig_err)?;
    Ok(PyTensor::from_f64(out))
}

/// Apply a zero-phase (forward-backward) filter with numerator `b` and denominator `a` to `x`.
/// Eliminates phase distortion by filtering in both directions.
#[pyfunction]
fn filtfilt(b: Vec<f64>, a: Vec<f64>, x: Vec<f64>) -> PyResult<PyTensor> {
    let bt = tensor_from_list(b)?;
    let at = tensor_from_list(a)?;
    let xt = tensor_from_list(x)?;
    let out = scivex_signal::filter::filtfilt(&bt, &at, &xt).map_err(sig_err)?;
    Ok(PyTensor::from_f64(out))
}

/// Design a low-pass FIR filter with the given normalized `cutoff` frequency and `num_taps`.
/// Returns the filter coefficients as a 1-D tensor.
#[pyfunction]
fn firwin_lowpass(cutoff: f64, num_taps: usize) -> PyResult<PyTensor> {
    let t = scivex_signal::filter::FirFilter::low_pass::<f64>(cutoff, num_taps).map_err(sig_err)?;
    Ok(PyTensor::from_f64(t))
}

/// Design a high-pass FIR filter with the given normalized `cutoff` frequency and `num_taps`.
/// Returns the filter coefficients as a 1-D tensor.
#[pyfunction]
fn firwin_highpass(cutoff: f64, num_taps: usize) -> PyResult<PyTensor> {
    let t =
        scivex_signal::filter::FirFilter::high_pass::<f64>(cutoff, num_taps).map_err(sig_err)?;
    Ok(PyTensor::from_f64(t))
}

/// Design a band-pass FIR filter between normalized frequencies `low` and `high` with `num_taps`.
/// Returns the filter coefficients as a 1-D tensor.
#[pyfunction]
fn firwin_bandpass(low: f64, high: f64, num_taps: usize) -> PyResult<PyTensor> {
    let t =
        scivex_signal::filter::FirFilter::band_pass::<f64>(low, high, num_taps).map_err(sig_err)?;
    Ok(PyTensor::from_f64(t))
}

// ===========================================================================
// SPECTRAL ANALYSIS
// ===========================================================================

/// Compute the Short-Time Fourier Transform of signal `x`.
/// Uses the given `window_size`, `hop_size`, and optional custom `window`.
/// Returns a 2-D complex-valued tensor (frequency bins x time frames).
#[pyfunction]
#[pyo3(signature = (x, window_size, hop_size, window=None))]
fn stft(
    x: Vec<f64>,
    window_size: usize,
    hop_size: usize,
    window: Option<Vec<f64>>,
) -> PyResult<PyTensor> {
    let xt = tensor_from_list(x)?;
    let win_t = window.map(tensor_from_list).transpose()?;
    let out = scivex_signal::spectral::stft(&xt, window_size, hop_size, win_t.as_ref())
        .map_err(sig_err)?;
    Ok(PyTensor::from_f64(out))
}

/// Compute the Inverse Short-Time Fourier Transform to reconstruct a time-domain signal.
/// Takes the STFT result, `window_size`, `hop_size`, and optional custom `window`.
#[pyfunction]
#[pyo3(signature = (stft_result, window_size, hop_size, window=None))]
fn istft(
    stft_result: &PyTensor,
    window_size: usize,
    hop_size: usize,
    window: Option<Vec<f64>>,
) -> PyResult<PyTensor> {
    let win_t = window.map(tensor_from_list).transpose()?;
    let out = scivex_signal::spectral::istft(
        stft_result.as_f64()?,
        window_size,
        hop_size,
        win_t.as_ref(),
    )
    .map_err(sig_err)?;
    Ok(PyTensor::from_f64(out))
}

/// Compute the power spectrogram of signal `x` with the given `window_size` and `hop_size`.
/// Returns a 2-D tensor of magnitude-squared STFT coefficients.
#[pyfunction]
fn spectrogram(x: Vec<f64>, window_size: usize, hop_size: usize) -> PyResult<PyTensor> {
    let xt = tensor_from_list(x)?;
    let out = scivex_signal::spectral::spectrogram(&xt, window_size, hop_size).map_err(sig_err)?;
    Ok(PyTensor::from_f64(out))
}

/// Estimate the power spectral density of signal `x` using a single FFT.
/// Returns a tuple of (frequencies, PSD) tensors.
#[pyfunction]
fn periodogram(x: Vec<f64>) -> PyResult<(PyTensor, PyTensor)> {
    let xt = tensor_from_list(x)?;
    let (freqs, psd) = scivex_signal::spectral::periodogram(&xt).map_err(sig_err)?;
    Ok((PyTensor::from_f64(freqs), PyTensor::from_f64(psd)))
}

/// Estimate the power spectral density using Welch's method with overlapping segments.
/// Returns a tuple of (frequencies, PSD) tensors.
#[pyfunction]
fn welch(x: Vec<f64>, segment_size: usize, overlap: usize) -> PyResult<(PyTensor, PyTensor)> {
    let xt = tensor_from_list(x)?;
    let (freqs, psd) =
        scivex_signal::spectral::welch(&xt, segment_size, overlap).map_err(sig_err)?;
    Ok((PyTensor::from_f64(freqs), PyTensor::from_f64(psd)))
}

// ===========================================================================
// WAVELETS
// ===========================================================================

fn parse_wavelet(name: &str) -> PyResult<scivex_signal::wavelet::Wavelet> {
    match name.to_lowercase().as_str() {
        "haar" | "db1" => Ok(scivex_signal::wavelet::Wavelet::Haar),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported wavelet: {name}. Use 'haar'"
        ))),
    }
}

/// Perform a single-level Discrete Wavelet Transform on signal `x`.
/// Returns a tuple of (approximation, detail) coefficient tensors.
#[pyfunction]
#[pyo3(signature = (x, wavelet="haar"))]
fn dwt(x: Vec<f64>, wavelet: &str) -> PyResult<(PyTensor, PyTensor)> {
    let xt = tensor_from_list(x)?;
    let wav = parse_wavelet(wavelet)?;
    let (approx, detail) = scivex_signal::wavelet::dwt(&xt, wav).map_err(sig_err)?;
    Ok((PyTensor::from_f64(approx), PyTensor::from_f64(detail)))
}

/// Perform the Inverse Discrete Wavelet Transform from `approx` and `detail` coefficients.
/// Reconstructs the original signal from a single-level DWT decomposition.
#[pyfunction]
#[pyo3(signature = (approx, detail, wavelet="haar"))]
fn idwt(approx: Vec<f64>, detail: Vec<f64>, wavelet: &str) -> PyResult<PyTensor> {
    let at = tensor_from_list(approx)?;
    let dt = tensor_from_list(detail)?;
    let wav = parse_wavelet(wavelet)?;
    let out = scivex_signal::wavelet::idwt(&at, &dt, wav).map_err(sig_err)?;
    Ok(PyTensor::from_f64(out))
}

// ===========================================================================
// AUDIO FEATURES
// ===========================================================================

/// Convert a frequency in Hertz to the Mel scale.
#[pyfunction]
fn hz_to_mel(hz: f64) -> f64 {
    scivex_signal::features::hz_to_mel(hz)
}

/// Convert a Mel-scale value back to Hertz.
#[pyfunction]
fn mel_to_hz(mel: f64) -> f64 {
    scivex_signal::features::mel_to_hz(mel)
}

/// Compute a Mel-scaled spectrogram of signal `x`.
/// Parameters: `sample_rate`, FFT size `n_fft`, `hop_size`, and number of Mel bands `n_mels`.
#[pyfunction]
fn mel_spectrogram(
    x: Vec<f64>,
    sample_rate: f64,
    n_fft: usize,
    hop_size: usize,
    n_mels: usize,
) -> PyResult<PyTensor> {
    let xt = tensor_from_list(x)?;
    let out = scivex_signal::features::mel_spectrogram(&xt, sample_rate, n_fft, hop_size, n_mels)
        .map_err(sig_err)?;
    Ok(PyTensor::from_f64(out))
}

/// Compute Mel-Frequency Cepstral Coefficients (MFCCs) for signal `x`.
/// Parameters: `sample_rate`, number of coefficients `n_mfcc`, `n_mels`, `n_fft`, and `hop_size`.
#[pyfunction]
fn mfcc(
    x: Vec<f64>,
    sample_rate: f64,
    n_mfcc: usize,
    n_mels: usize,
    n_fft: usize,
    hop_size: usize,
) -> PyResult<PyTensor> {
    let xt = tensor_from_list(x)?;
    let out = scivex_signal::features::mfcc(&xt, sample_rate, n_mfcc, n_mels, n_fft, hop_size)
        .map_err(sig_err)?;
    Ok(PyTensor::from_f64(out))
}

/// Compute a chromagram (chroma feature) from an STFT of signal `x`.
/// Maps spectral energy onto `n_chroma` pitch classes (default 12).
#[pyfunction]
#[pyo3(signature = (x, sample_rate, n_fft, hop_size, n_chroma=12))]
fn chroma_stft(
    x: Vec<f64>,
    sample_rate: f64,
    n_fft: usize,
    hop_size: usize,
    n_chroma: usize,
) -> PyResult<PyTensor> {
    let xt = tensor_from_list(x)?;
    let out = scivex_signal::features::chroma_stft(&xt, sample_rate, n_fft, hop_size, n_chroma)
        .map_err(sig_err)?;
    Ok(PyTensor::from_f64(out))
}

/// Estimate fundamental frequency (pitch) per frame using the YIN algorithm.
/// Returns a list of pitch estimates in Hz for each frame. `threshold` controls voicing sensitivity.
#[pyfunction]
#[pyo3(signature = (x, sample_rate, frame_size, hop_size, threshold=0.15))]
fn pitch_yin(
    x: Vec<f64>,
    sample_rate: f64,
    frame_size: usize,
    hop_size: usize,
    threshold: f64,
) -> PyResult<Vec<f64>> {
    let xt = tensor_from_list(x)?;
    scivex_signal::features::pitch_yin(&xt, sample_rate, frame_size, hop_size, threshold)
        .map_err(sig_err)
}

// ===========================================================================
// PEAK DETECTION
// ===========================================================================

/// Find local maxima (peaks) in signal `x`.
/// Optionally filter by `min_height` and minimum `min_distance` between peaks.
/// Returns a list of peak indices.
#[pyfunction]
#[pyo3(signature = (x, min_height=None, min_distance=None))]
fn find_peaks(
    x: Vec<f64>,
    min_height: Option<f64>,
    min_distance: Option<usize>,
) -> PyResult<Vec<usize>> {
    let xt = tensor_from_list(x)?;
    scivex_signal::peak::find_peaks(&xt, min_height, min_distance).map_err(sig_err)
}

/// Compute the prominence of each peak in `peaks` relative to signal `x`.
/// Returns a list of prominence values corresponding to each peak index.
#[pyfunction]
fn peak_prominences(x: Vec<f64>, peaks: Vec<usize>) -> PyResult<Vec<f64>> {
    let xt = tensor_from_list(x)?;
    scivex_signal::peak::peak_prominences(&xt, &peaks).map_err(sig_err)
}

// ===========================================================================
// RESAMPLING
// ===========================================================================

/// Resample signal `x` to `num_samples` using FFT-based interpolation.
/// Returns the resampled signal as a 1-D tensor.
#[pyfunction]
fn resample(x: Vec<f64>, num_samples: usize) -> PyResult<PyTensor> {
    let xt = tensor_from_list(x)?;
    let out = scivex_signal::resample::resample(&xt, num_samples).map_err(sig_err)?;
    Ok(PyTensor::from_f64(out))
}

/// Downsample signal `x` by an integer `factor` after applying an anti-aliasing filter.
/// Returns the decimated signal as a 1-D tensor.
#[pyfunction]
fn decimate(x: Vec<f64>, factor: usize) -> PyResult<PyTensor> {
    let xt = tensor_from_list(x)?;
    let out = scivex_signal::resample::decimate(&xt, factor).map_err(sig_err)?;
    Ok(PyTensor::from_f64(out))
}

// ===========================================================================
// CONVOLUTION
// ===========================================================================

fn parse_convolve_mode(mode: &str) -> PyResult<scivex_signal::convolution::ConvolveMode> {
    match mode.to_lowercase().as_str() {
        "full" => Ok(scivex_signal::convolution::ConvolveMode::Full),
        "same" => Ok(scivex_signal::convolution::ConvolveMode::Same),
        "valid" => Ok(scivex_signal::convolution::ConvolveMode::Valid),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown mode: {mode}. Use 'full', 'same', or 'valid'"
        ))),
    }
}

/// Convolve two 1-D signals `a` and `b`.
/// `mode` controls output size: "full" (default), "same", or "valid".
#[pyfunction]
#[pyo3(signature = (a, b, mode="full"))]
fn convolve(a: Vec<f64>, b: Vec<f64>, mode: &str) -> PyResult<PyTensor> {
    let at = tensor_from_list(a)?;
    let bt = tensor_from_list(b)?;
    let m = parse_convolve_mode(mode)?;
    let out = scivex_signal::convolution::convolve(&at, &bt, m).map_err(sig_err)?;
    Ok(PyTensor::from_f64(out))
}

/// Compute the cross-correlation of two 1-D signals `a` and `b`.
/// `mode` controls output size: "full" (default), "same", or "valid".
#[pyfunction]
#[pyo3(signature = (a, b, mode="full"))]
fn correlate(a: Vec<f64>, b: Vec<f64>, mode: &str) -> PyResult<PyTensor> {
    let at = tensor_from_list(a)?;
    let bt = tensor_from_list(b)?;
    let m = parse_convolve_mode(mode)?;
    let out = scivex_signal::convolution::correlate(&at, &bt, m).map_err(sig_err)?;
    Ok(PyTensor::from_f64(out))
}

// ===========================================================================
// AUDIO I/O
// ===========================================================================

#[pyclass(name = "AudioData")]
#[derive(Clone)]
pub struct PyAudioData {
    inner: scivex_signal::audio::AudioData,
}

#[pymethods]
impl PyAudioData {
    /// Raw samples as flat list (interleaved if multi-channel).
    #[getter]
    fn samples(&self) -> Vec<f64> {
        self.inner.samples.clone()
    }

    /// Number of audio channels (e.g. 1 for mono, 2 for stereo).
    #[getter]
    fn channels(&self) -> u16 {
        self.inner.channels
    }

    /// Sample rate in Hz (e.g. 44100).
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    /// Bit depth per sample (e.g. 16, 24).
    #[getter]
    fn bits_per_sample(&self) -> u16 {
        self.inner.bits_per_sample
    }

    /// Total number of sample frames (samples per channel).
    fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    /// Extract samples for a single channel `ch` (0-indexed). Returns a list of f64 values.
    fn channel(&self, ch: usize) -> PyResult<Vec<f64>> {
        self.inner.channel(ch).map_err(sig_err)
    }

    /// Mix all channels down to a single mono signal by averaging.
    fn to_mono(&self) -> Vec<f64> {
        self.inner.to_mono()
    }

    /// Return a human-readable string representation of the audio data.
    fn __repr__(&self) -> String {
        format!(
            "AudioData(channels={}, sample_rate={}, frames={}, bits={})",
            self.inner.channels,
            self.inner.sample_rate,
            self.inner.num_frames(),
            self.inner.bits_per_sample,
        )
    }
}

/// Read a WAV file from disk.
#[pyfunction]
fn read_wav(path: &str) -> PyResult<PyAudioData> {
    let data = std::fs::read(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("cannot read {path}: {e}")))?;
    let audio = scivex_signal::audio::read_wav(&data).map_err(sig_err)?;
    Ok(PyAudioData { inner: audio })
}

/// Write a WAV file to disk (16-bit PCM).
#[pyfunction]
#[pyo3(signature = (path, samples, channels=1, sample_rate=44100))]
fn write_wav(path: &str, samples: Vec<f64>, channels: u16, sample_rate: u32) -> PyResult<()> {
    let audio = scivex_signal::audio::AudioData {
        samples,
        channels,
        sample_rate,
        bits_per_sample: 16,
    };
    let bytes = scivex_signal::audio::write_wav(&audio).map_err(sig_err)?;
    std::fs::write(path, &bytes)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("cannot write {path}: {e}")))?;
    Ok(())
}

// ===========================================================================
// RHYTHM / BEAT TRACKING
// ===========================================================================

/// Compute the onset strength envelope of an audio signal.
/// Returns a dict with keys "onset_envelope", "onset_frames", and "onset_times".
#[pyfunction]
fn onset_strength(
    py: Python<'_>,
    signal: Vec<f64>,
    sample_rate: f64,
    frame_size: usize,
    hop_size: usize,
) -> PyResult<PyObject> {
    let result = scivex_signal::rhythm::onset_strength(&signal, sample_rate, frame_size, hop_size)
        .map_err(sig_err)?;
    let dict = PyDict::new(py);
    dict.set_item("onset_envelope", result.onset_envelope)?;
    dict.set_item("onset_frames", result.onset_frames)?;
    dict.set_item("onset_times", result.onset_times)?;
    Ok(dict.into_any().unbind())
}

/// Detect onset events (note/transient starts) in an audio signal.
/// Returns a dict with keys "onset_envelope", "onset_frames", and "onset_times".
#[pyfunction]
fn detect_onsets(
    py: Python<'_>,
    signal: Vec<f64>,
    sample_rate: f64,
    frame_size: usize,
    hop_size: usize,
) -> PyResult<PyObject> {
    let result = scivex_signal::rhythm::detect_onsets(&signal, sample_rate, frame_size, hop_size)
        .map_err(sig_err)?;
    let dict = PyDict::new(py);
    dict.set_item("onset_envelope", result.onset_envelope)?;
    dict.set_item("onset_frames", result.onset_frames)?;
    dict.set_item("onset_times", result.onset_times)?;
    Ok(dict.into_any().unbind())
}

/// Estimate tempo and locate beat positions in an audio signal.
/// Returns a dict with keys "tempo" (BPM), "beat_frames", and "beat_times".
#[pyfunction]
fn beat_track(
    py: Python<'_>,
    signal: Vec<f64>,
    sample_rate: f64,
    frame_size: usize,
    hop_size: usize,
) -> PyResult<PyObject> {
    let result = scivex_signal::rhythm::beat_track(&signal, sample_rate, frame_size, hop_size)
        .map_err(sig_err)?;
    let dict = PyDict::new(py);
    dict.set_item("tempo", result.tempo)?;
    dict.set_item("beat_frames", result.beat_frames)?;
    dict.set_item("beat_times", result.beat_times)?;
    Ok(dict.into_any().unbind())
}

// ===========================================================================
// REGISTER
// ===========================================================================

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "signal")?;

    // Windows
    m.add_function(wrap_pyfunction!(hann, &m)?)?;
    m.add_function(wrap_pyfunction!(hamming, &m)?)?;
    m.add_function(wrap_pyfunction!(blackman, &m)?)?;
    m.add_function(wrap_pyfunction!(bartlett, &m)?)?;

    // Filters
    m.add_function(wrap_pyfunction!(lfilter, &m)?)?;
    m.add_function(wrap_pyfunction!(filtfilt, &m)?)?;
    m.add_function(wrap_pyfunction!(firwin_lowpass, &m)?)?;
    m.add_function(wrap_pyfunction!(firwin_highpass, &m)?)?;
    m.add_function(wrap_pyfunction!(firwin_bandpass, &m)?)?;

    // Spectral
    m.add_function(wrap_pyfunction!(stft, &m)?)?;
    m.add_function(wrap_pyfunction!(istft, &m)?)?;
    m.add_function(wrap_pyfunction!(spectrogram, &m)?)?;
    m.add_function(wrap_pyfunction!(periodogram, &m)?)?;
    m.add_function(wrap_pyfunction!(welch, &m)?)?;

    // Wavelets
    m.add_function(wrap_pyfunction!(dwt, &m)?)?;
    m.add_function(wrap_pyfunction!(idwt, &m)?)?;

    // Audio features
    m.add_function(wrap_pyfunction!(hz_to_mel, &m)?)?;
    m.add_function(wrap_pyfunction!(mel_to_hz, &m)?)?;
    m.add_function(wrap_pyfunction!(mel_spectrogram, &m)?)?;
    m.add_function(wrap_pyfunction!(mfcc, &m)?)?;
    m.add_function(wrap_pyfunction!(chroma_stft, &m)?)?;
    m.add_function(wrap_pyfunction!(pitch_yin, &m)?)?;

    // Peak detection
    m.add_function(wrap_pyfunction!(find_peaks, &m)?)?;
    m.add_function(wrap_pyfunction!(peak_prominences, &m)?)?;

    // Resampling
    m.add_function(wrap_pyfunction!(resample, &m)?)?;
    m.add_function(wrap_pyfunction!(decimate, &m)?)?;

    // Convolution
    m.add_function(wrap_pyfunction!(convolve, &m)?)?;
    m.add_function(wrap_pyfunction!(correlate, &m)?)?;

    // Audio I/O
    m.add_class::<PyAudioData>()?;
    m.add_function(wrap_pyfunction!(read_wav, &m)?)?;
    m.add_function(wrap_pyfunction!(write_wav, &m)?)?;

    // Rhythm / beat tracking
    m.add_function(wrap_pyfunction!(onset_strength, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_onsets, &m)?)?;
    m.add_function(wrap_pyfunction!(beat_track, &m)?)?;

    parent.add_submodule(&m)?;
    Ok(())
}
