#![allow(clippy::manual_is_multiple_of)]
//! `scivex-signal` — Signal processing, FFT, wavelets, and audio.
//!
//! Provides a from-scratch signal processing library with support for:
//! - Window functions (Hann, Hamming, Blackman, Bartlett)
//! - Digital filters (FIR design, `lfilter`, `filtfilt`)
//! - Spectral analysis (STFT, spectrogram, periodogram, Welch)
//! - Resampling (FFT-based resample, decimate, interpolate)
//! - Peak detection (`find_peaks`, prominences)
//! - 1-D convolution and correlation
//! - Wavelet transforms (DWT/IDWT with Haar)

/// WAV audio file reading and writing.
pub mod audio;
/// 1-D convolution and correlation.
pub mod convolution;
/// Signal processing error types.
pub mod error;
/// Digital filters (FIR design, `lfilter`, `filtfilt`).
pub mod filter;
/// Peak detection and prominence calculation.
pub mod peak;
/// FFT-based resampling, decimation, and interpolation.
pub mod resample;
/// Spectral analysis (STFT, spectrogram, periodogram, Welch).
pub mod spectral;
/// Wavelet transforms (DWT / IDWT).
pub mod wavelet;
/// Window functions (Hann, Hamming, Blackman, Bartlett).
pub mod window;

pub use error::{Result, SignalError};

/// Items intended for glob-import: `use scivex_signal::prelude::*;`
pub mod prelude {
    pub use crate::convolution::ConvolveMode;
    pub use crate::error::{Result, SignalError};
    pub use crate::wavelet::Wavelet;
}
