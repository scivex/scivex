//! Signal processing bindings for JavaScript.

use scivex_core::Tensor;
use wasm_bindgen::prelude::*;

/// Apply a Hann window of length n.
#[wasm_bindgen(js_name = "windowHann")]
pub fn window_hann(n: usize) -> Result<Vec<f64>, JsError> {
    let t: Tensor<f64> =
        scivex_signal::window::hann(n).map_err(|e| JsError::new(&e.to_string()))?;
    Ok(t.as_slice().to_vec())
}

/// Apply a Hamming window of length n.
#[wasm_bindgen(js_name = "windowHamming")]
pub fn window_hamming(n: usize) -> Result<Vec<f64>, JsError> {
    let t: Tensor<f64> =
        scivex_signal::window::hamming(n).map_err(|e| JsError::new(&e.to_string()))?;
    Ok(t.as_slice().to_vec())
}

/// Apply a Blackman window of length n.
#[wasm_bindgen(js_name = "windowBlackman")]
pub fn window_blackman(n: usize) -> Result<Vec<f64>, JsError> {
    let t: Tensor<f64> =
        scivex_signal::window::blackman(n).map_err(|e| JsError::new(&e.to_string()))?;
    Ok(t.as_slice().to_vec())
}

/// 1-D convolution (full mode).
#[wasm_bindgen(js_name = "convolve")]
pub fn convolve(signal: &[f64], kernel: &[f64]) -> Result<Vec<f64>, JsError> {
    let sig = Tensor::from_vec(signal.to_vec(), vec![signal.len()])
        .map_err(|e| JsError::new(&e.to_string()))?;
    let ker = Tensor::from_vec(kernel.to_vec(), vec![kernel.len()])
        .map_err(|e| JsError::new(&e.to_string()))?;
    let result = scivex_signal::convolution::convolve(
        &sig,
        &ker,
        scivex_signal::convolution::ConvolveMode::Full,
    )
    .map_err(|e| JsError::new(&e.to_string()))?;
    Ok(result.as_slice().to_vec())
}

/// Find peaks in a 1-D signal.
#[wasm_bindgen(js_name = "findPeaks")]
pub fn find_peaks(data: &[f64]) -> Result<Vec<usize>, JsError> {
    let t = Tensor::from_vec(data.to_vec(), vec![data.len()])
        .map_err(|e| JsError::new(&e.to_string()))?;
    scivex_signal::peak::find_peaks(&t, None, None).map_err(|e| JsError::new(&e.to_string()))
}

/// Read a WAV file from bytes and return samples + metadata.
#[wasm_bindgen]
pub struct WasmAudioData {
    samples: Vec<f64>,
    channels: u16,
    sample_rate: u32,
}

#[wasm_bindgen]
impl WasmAudioData {
    /// Read WAV data from a byte array.
    #[wasm_bindgen(js_name = "fromWav")]
    pub fn from_wav(data: &[u8]) -> Result<WasmAudioData, JsError> {
        let audio =
            scivex_signal::audio::read_wav(data).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmAudioData {
            samples: audio.samples,
            channels: audio.channels,
            sample_rate: audio.sample_rate,
        })
    }

    /// Get the samples as Float64Array.
    pub fn samples(&self) -> Vec<f64> {
        self.samples.clone()
    }

    /// Number of channels.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Sample rate in Hz.
    #[wasm_bindgen(js_name = "sampleRate")]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Number of frames (samples per channel).
    #[wasm_bindgen(js_name = "numFrames")]
    pub fn num_frames(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }
}
