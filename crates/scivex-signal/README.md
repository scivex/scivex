# scivex-signal

Signal processing for Scivex. Digital filters, spectral analysis, wavelets,
and audio processing tools.

## Highlights

- **FIR filters** — Window-based design (Hamming, Hanning, Blackman, Kaiser)
- **IIR filters** — Butterworth, Chebyshev Type I/II filter design
- **Spectral analysis** — STFT, spectrogram, power spectral density, periodogram
- **Wavelets** — CWT, DWT, Haar, Daubechies, Morlet wavelets
- **Peak detection** — Find peaks with prominence, width, and distance constraints
- **Resampling** — Upsample, downsample, polyphase resampling
- **Windows** — Hamming, Hanning, Blackman, Kaiser, Gaussian
- **Convolution** — Linear and circular convolution, correlation
- **Audio** — WAV read/write, MFCC feature extraction

## Usage

```rust
use scivex_signal::prelude::*;

// Design a low-pass filter
let coeffs = fir_lowpass(0.2, 64, Window::Hamming);
let filtered = convolve(&signal, &coeffs);

// Spectrogram
let spec = stft(&signal, 1024, 256, Window::Hanning);

// Wavelet transform
let (approx, detail) = dwt(&signal, Wavelet::Haar);
```

## License

MIT
