# scivex-signal

Signal processing for Scivex. Digital filtering, spectral analysis, wavelets,
peak detection, and resampling.

## Highlights

- **Window functions** — Hann, Hamming, Blackman, Bartlett
- **Digital filters** — FIR design (low/high/band-pass), IIR via `lfilter`/`filtfilt`
- **Convolution** — 1D convolution and correlation (Full/Same/Valid modes)
- **Spectral analysis** — STFT, inverse STFT, spectrogram, Welch, periodogram
- **Peak detection** — Local maxima with prominence and width estimation
- **Resampling** — FFT-based resample, decimate, interpolate
- **Wavelets** — Discrete wavelet transform (Haar) with inverse

## Usage

```rust
use scivex_signal::prelude::*;

// Apply a window and compute spectrogram
let window = hann::<f64>(1024);
let spec = spectrogram(&signal, 1024, 512).unwrap();

// Design and apply a low-pass FIR filter
let fir = FirFilter::low_pass(64, 0.1);
let filtered = lfilter(fir.coeffs(), &[1.0], &signal).unwrap();

// Zero-phase filtering
let smooth = filtfilt(&b_coeffs, &a_coeffs, &signal).unwrap();

// Peak detection
let peaks = find_peaks(&signal, Some(0.5), Some(0.1));
```

## License

MIT
