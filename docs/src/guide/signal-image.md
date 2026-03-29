# Signal Processing & Image Analysis

This guide covers audio/signal processing pipelines and image analysis workflows.

## Audio Processing Pipeline

### WAV to Features

A typical audio ML pipeline: load WAV, compute spectrogram, extract features.

```rust
use scivex_signal::prelude::*;
use scivex_core::prelude::*;

// Load a WAV file
let (samples, sample_rate) = scivex_signal::io::read_wav("audio.wav")?;
println!("Duration: {:.2}s, Sample rate: {} Hz",
    samples.len() as f64 / sample_rate as f64, sample_rate);

// Short-Time Fourier Transform
let stft_result = stft(
    &samples,
    1024,       // window_size (FFT size)
    256,        // hop_size
    WindowType::Hann,
)?;
println!("STFT frames: {}, frequency bins: {}",
    stft_result.n_frames(), stft_result.n_freq_bins());
```

### Mel Spectrogram

```rust
// Compute mel spectrogram (standard for speech/music ML)
let mel = mel_spectrogram(
    &samples,
    sample_rate,
    1024,       // n_fft
    256,        // hop_length
    128,        // n_mels
)?;
println!("Mel spectrogram shape: {:?}", mel.shape());
```

### MFCC (Mel-Frequency Cepstral Coefficients)

```rust
// MFCCs — standard features for speech recognition
let mfcc = mfcc(
    &samples,
    sample_rate,
    13,         // n_mfcc
    1024,       // n_fft
    256,        // hop_length
    128,        // n_mels
)?;
println!("MFCC shape: {:?}", mfcc.shape()); // [n_mfcc, n_frames]
```

### Chroma Features

```rust
// Chroma — pitch class distribution (music analysis)
let chroma = chroma_stft(
    &samples,
    sample_rate,
    1024,       // n_fft
    256,        // hop_length
    12,         // n_chroma
)?;
println!("Chroma shape: {:?}", chroma.shape()); // [12, n_frames]
```

### Pitch Detection

```rust
// YIN pitch detection
let pitches = yin_pitch(
    &samples,
    sample_rate,
    50.0,       // f_min (Hz)
    500.0,      // f_max (Hz)
    1024,       // frame_length
    256,        // hop_length
)?;
println!("Detected pitches: {} frames", pitches.len());
```

### Beat Tracking

```rust
// Detect beats in audio
let beats = beat_track(&samples, sample_rate)?;
println!("Detected {} beats", beats.beat_times.len());
println!("Estimated tempo: {:.1} BPM", beats.tempo);

// Onset detection (note/drum hit starts)
let onsets = onset_detect(&samples, sample_rate)?;
println!("Detected {} onsets", onsets.len());
```

## Digital Filters

```rust
use scivex_signal::filters::*;

// Low-pass Butterworth filter
let filtered = butterworth_lowpass(&signal, 1000.0, sample_rate as f64, 4)?;

// Band-pass filter
let bandpass = butterworth_bandpass(&signal, 300.0, 3000.0, sample_rate as f64, 4)?;

// FIR filter design
let coeffs = fir_lowpass(64, 1000.0, sample_rate as f64, WindowType::Hamming)?;
let filtered = convolve(&signal, &coeffs)?;
```

## Wavelet Transform

```rust
use scivex_signal::wavelets::*;

// Continuous Wavelet Transform
let cwt = cwt_morlet(&signal, &scales, 6.0)?;

// Discrete Wavelet Transform (Haar)
let (approx, detail) = dwt(&signal, Wavelet::Haar)?;

// Multi-level decomposition
let levels = wavedec(&signal, Wavelet::Db4, 5)?;
```

## Complete Audio ML Pipeline

```rust
use scivex_signal::prelude::*;
use scivex_ml::prelude::*;

fn extract_audio_features(path: &str) -> Result<Vec<f64>> {
    let (samples, sr) = scivex_signal::io::read_wav(path)?;

    // Compute MFCCs
    let mfcc = mfcc(&samples, sr, 13, 2048, 512, 128)?;

    // Aggregate across time: mean + std of each MFCC coefficient
    let mut features = Vec::new();
    for i in 0..13 {
        let coeff: Vec<f64> = (0..mfcc.shape()[1])
            .map(|j| mfcc.get(&[i, j]))
            .collect();
        features.push(descriptive::mean(&coeff)?);
        features.push(descriptive::std_dev(&coeff)?);
    }
    Ok(features) // 26 features per audio file
}

// Use extracted features for classification
let mut dataset = Vec::new();
for (path, label) in audio_files {
    let features = extract_audio_features(path)?;
    dataset.push((features, label));
}
// ... train classifier on features
```

---

## Image Processing

### Loading & Basic Operations

```rust
use scivex_image::prelude::*;

// Load image
let img = scivex_image::io::read_bmp("photo.bmp")?;
println!("{}x{}, channels: {}", img.width(), img.height(), img.channels());

// Convert to grayscale
let gray = scivex_image::color::grayscale(&img)?;

// Resize
let resized = scivex_image::transform::resize(&img, 224, 224, Interpolation::Bilinear)?;

// Crop
let cropped = scivex_image::transform::crop(&img, 50, 50, 200, 200)?;
```

### Filtering & Edge Detection

```rust
// Gaussian blur
let blurred = scivex_image::filter::gaussian_blur(&gray, 5, 1.5)?;

// Sobel edge detection
let edges = scivex_image::filter::sobel_edges(&gray)?;

// Canny edge detection
let canny = scivex_image::filter::canny(&gray, 50.0, 150.0)?;

// Median filter (salt-and-pepper noise removal)
let denoised = scivex_image::filter::median_filter(&gray, 3)?;

// Morphological operations
let dilated = scivex_image::morphology::dilate(&binary, &kernel, 1)?;
let eroded = scivex_image::morphology::erode(&binary, &kernel, 1)?;
let opened = scivex_image::morphology::open(&binary, &kernel)?;
```

### Feature Extraction

```rust
// Harris corner detection
let corners = scivex_image::features::harris_corners(&gray, 3, 0.04, 0.01)?;
println!("Found {} corners", corners.len());

// HOG (Histogram of Oriented Gradients) features
let hog = scivex_image::features::hog(&gray, 8, 8, 9)?;
println!("HOG descriptor length: {}", hog.len());

// BRIEF descriptors
let keypoints = scivex_image::features::fast_keypoints(&gray, 20)?;
let descriptors = scivex_image::features::brief(&gray, &keypoints, 256)?;
```

### Optical Flow

```rust
use scivex_image::optical_flow::*;

// Lucas-Kanade optical flow between two frames
let flow = lucas_kanade(&frame1, &frame2, 15)?;

// Dense optical flow (Farneback)
let dense_flow = farneback(&frame1, &frame2, 0.5, 3, 15, 3, 5, 1.2)?;
```

### Data Augmentation Pipeline

Build an augmentation pipeline for ML training data preparation:

```rust
use scivex_image::prelude::*;
use scivex_image::augment::*;

fn augment_batch(images: &[Image], rng: &mut Rng) -> Vec<Image> {
    images.iter().map(|img| {
        let mut aug = img.clone();

        // Random horizontal flip (50% chance)
        if rng.uniform_f64() > 0.5 {
            aug = scivex_image::transform::flip_horizontal(&aug).unwrap();
        }

        // Random rotation (-15 to +15 degrees)
        let angle = rng.uniform_f64() * 30.0 - 15.0;
        aug = scivex_image::transform::rotate(&aug, angle).unwrap();

        // Random brightness adjustment
        let factor = 0.8 + rng.uniform_f64() * 0.4;
        aug = scivex_image::color::adjust_brightness(&aug, factor).unwrap();

        // Resize to model input size
        aug = scivex_image::transform::resize(&aug, 224, 224, Interpolation::Bilinear).unwrap();

        aug
    }).collect()
}
```

### Complete Image Classification Pipeline

```rust
use scivex_image::prelude::*;
use scivex_ml::prelude::*;

fn image_to_features(path: &str) -> Result<Vec<f64>> {
    let img = scivex_image::io::read_bmp(path)?;
    let gray = scivex_image::color::grayscale(&img)?;
    let resized = scivex_image::transform::resize(&gray, 64, 64, Interpolation::Bilinear)?;

    // Extract HOG features
    let hog = scivex_image::features::hog(&resized, 8, 8, 9)?;
    Ok(hog)
}

// Extract features from dataset
let features: Vec<Vec<f64>> = image_paths.iter()
    .map(|p| image_to_features(p))
    .collect::<Result<_>>()?;

// Train SVM classifier on HOG features
let x = Tensor::from_vec(features.concat(), vec![n_images, feature_dim])?;
let y = Tensor::from_vec(labels, vec![n_images])?;

let mut svm = Svm::new().kernel(Kernel::RBF(1.0)).c(10.0);
svm.fit(&x, &y)?;
let accuracy = svm.score(&x_test, &y_test)?;
println!("Classification accuracy: {:.2}%", accuracy * 100.0);
```
