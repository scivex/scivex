//! Reference-value tests comparing scivex-signal outputs against known SciPy.signal results.
//!
//! Every expected value was pre-computed with SciPy (Python) and hard-coded here so
//! that the Rust implementation can be validated against the de-facto reference.

use scivex_core::Tensor;
use scivex_signal::convolution::{ConvolveMode, convolve, correlate};
use scivex_signal::filter::lfilter;
use scivex_signal::peak::find_peaks;
use scivex_signal::spectral::welch;
use scivex_signal::window;

// ── helpers ──────────────────────────────────────────────────────────────

const TOL: f64 = 1e-10;
const TOL_WINDOW: f64 = 1e-6; // Blackman coefficients have slight float noise

fn vec1d(data: &[f64]) -> Tensor<f64> {
    Tensor::from_vec(data.to_vec(), vec![data.len()]).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch (got {}, expected {})",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{label}[{i}]: got {a}, expected {e} (diff = {})",
            (a - e).abs()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 1. Window functions
// ═══════════════════════════════════════════════════════════════════════

// scipy: signal.windows.hann(5) => [0.0, 0.5, 1.0, 0.5, 0.0]
#[test]
fn scipy_hann_5() {
    let w = window::hann::<f64>(5).unwrap();
    assert_close(w.as_slice(), &[0.0, 0.5, 1.0, 0.5, 0.0], TOL, "hann(5)");
}

// scipy: signal.windows.hamming(5) => [0.08, 0.54, 1.0, 0.54, 0.08]
#[test]
fn scipy_hamming_5() {
    let w = window::hamming::<f64>(5).unwrap();
    assert_close(
        w.as_slice(),
        &[0.08, 0.54, 1.0, 0.54, 0.08],
        TOL,
        "hamming(5)",
    );
}

// scipy: signal.windows.blackman(5)
// Exact: [-1.388e-17, 0.34, 1.0, 0.34, -1.388e-17]
// The endpoints are analytically zero; floating-point gives ~1e-17.
#[test]
fn scipy_blackman_5() {
    let w = window::blackman::<f64>(5).unwrap();
    let s = w.as_slice();
    assert!(s[0].abs() < TOL_WINDOW, "blackman[0] = {}", s[0]);
    assert!((s[1] - 0.34).abs() < TOL_WINDOW, "blackman[1] = {}", s[1]);
    assert!((s[2] - 1.0).abs() < TOL_WINDOW, "blackman[2] = {}", s[2]);
    assert!((s[3] - 0.34).abs() < TOL_WINDOW, "blackman[3] = {}", s[3]);
    assert!(s[4].abs() < TOL_WINDOW, "blackman[4] = {}", s[4]);
}

// Verify symmetry on a larger window (matches SciPy property).
#[test]
fn scipy_hann_symmetry_8() {
    let w = window::hann::<f64>(8).unwrap();
    let s = w.as_slice();
    for i in 0..4 {
        assert!(
            (s[i] - s[7 - i]).abs() < TOL,
            "hann(8): asymmetry at index {i}"
        );
    }
}

#[test]
fn scipy_hamming_symmetry_8() {
    let w = window::hamming::<f64>(8).unwrap();
    let s = w.as_slice();
    for i in 0..4 {
        assert!(
            (s[i] - s[7 - i]).abs() < TOL,
            "hamming(8): asymmetry at index {i}"
        );
    }
}

#[test]
fn scipy_blackman_symmetry_8() {
    let w = window::blackman::<f64>(8).unwrap();
    let s = w.as_slice();
    for i in 0..4 {
        assert!(
            (s[i] - s[7 - i]).abs() < TOL_WINDOW,
            "blackman(8): asymmetry at index {i}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 2. 1-D Convolution (np.convolve)
// ═══════════════════════════════════════════════════════════════════════

// scipy/numpy: np.convolve([1,2,3,4,5], [1,0,-1], mode='full')
//   = [1.0, 2.0, 2.0, 2.0, 2.0, -4.0, -5.0]
#[test]
fn scipy_convolve_full() {
    let x = vec1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let h = vec1d(&[1.0, 0.0, -1.0]);
    let c = convolve(&x, &h, ConvolveMode::Full).unwrap();
    assert_close(
        c.as_slice(),
        &[1.0, 2.0, 2.0, 2.0, 2.0, -4.0, -5.0],
        TOL,
        "convolve full [1,2,3,4,5]*[1,0,-1]",
    );
}

// same mode: center slice of full result, length = len(x) = 5
// full = [1, 2, 2, 2, 2, -4, -5], nb=3 => start = (3-1)/2 = 1
// same = full[1..6] = [2.0, 2.0, 2.0, 2.0, -4.0]
#[test]
fn scipy_convolve_same() {
    let x = vec1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let h = vec1d(&[1.0, 0.0, -1.0]);
    let c = convolve(&x, &h, ConvolveMode::Same).unwrap();
    assert_eq!(c.shape(), &[5]);
    assert_close(
        c.as_slice(),
        &[2.0, 2.0, 2.0, 2.0, -4.0],
        TOL,
        "convolve same [1,2,3,4,5]*[1,0,-1]",
    );
}

// valid mode: output length = max(5,3) - min(5,3) + 1 = 3
// start = min(5,3) - 1 = 2 => full[2..5] = [2.0, 2.0, 2.0]
#[test]
fn scipy_convolve_valid() {
    let x = vec1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let h = vec1d(&[1.0, 0.0, -1.0]);
    let c = convolve(&x, &h, ConvolveMode::Valid).unwrap();
    assert_eq!(c.shape(), &[3]);
    assert_close(
        c.as_slice(),
        &[2.0, 2.0, 2.0],
        TOL,
        "convolve valid [1,2,3,4,5]*[1,0,-1]",
    );
}

// Convolution identity: convolving with [1] yields the original signal.
#[test]
fn scipy_convolve_identity() {
    let x = vec1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let delta = vec1d(&[1.0]);
    let c = convolve(&x, &delta, ConvolveMode::Full).unwrap();
    assert_close(
        c.as_slice(),
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        TOL,
        "convolve identity",
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Cross-correlation (np.correlate)
// ═══════════════════════════════════════════════════════════════════════

// np.correlate([1,2,3], [0,1,0.5], 'full')
// correlate(x,y) = convolve(x, y[::-1])
// y_rev = [0.5, 1.0, 0.0]
// convolve([1,2,3], [0.5,1,0]) = [0.5, 2.0, 3.5, 3.0, 0.0]
#[test]
fn scipy_correlate_full() {
    let x = vec1d(&[1.0, 2.0, 3.0]);
    let y = vec1d(&[0.0, 1.0, 0.5]);
    let c = correlate(&x, &y, ConvolveMode::Full).unwrap();
    assert_close(
        c.as_slice(),
        &[0.5, 2.0, 3.5, 3.0, 0.0],
        TOL,
        "correlate full [1,2,3] x [0,1,0.5]",
    );
}

// Auto-correlation of an impulse: peak should be at center.
#[test]
fn scipy_correlate_autocorrelation_impulse() {
    let x = vec1d(&[0.0, 0.0, 1.0, 0.0, 0.0]);
    let c = correlate(&x, &x, ConvolveMode::Full).unwrap();
    let s = c.as_slice();
    // Full auto-correlation length = 2*5 - 1 = 9, peak at index 4.
    assert_eq!(s.len(), 9);
    let max_idx = s
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(max_idx, 4, "auto-correlation peak index");
    assert!((s[4] - 1.0).abs() < TOL, "auto-correlation peak value");
}

// ═══════════════════════════════════════════════════════════════════════
// 4. FIR filter via lfilter (signal.lfilter)
// ═══════════════════════════════════════════════════════════════════════

// scipy: signal.lfilter([1/3, 1/3, 1/3], [1], [1,2,3,4,5,6])
//   y[0] = (1/3)*1 + 0 + 0                = 0.333...
//   y[1] = (1/3)*2 + (1/3)*1 + 0          = 1.0
//   y[2] = (1/3)*3 + (1/3)*2 + (1/3)*1    = 2.0
//   y[3] = (1/3)*4 + (1/3)*3 + (1/3)*2    = 3.0
//   y[4] = (1/3)*5 + (1/3)*4 + (1/3)*3    = 4.0
//   y[5] = (1/3)*6 + (1/3)*5 + (1/3)*4    = 5.0
#[test]
fn scipy_lfilter_moving_average_3pt() {
    let third = 1.0 / 3.0;
    let b = vec1d(&[third, third, third]);
    let a = vec1d(&[1.0]);
    let x = vec1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let y = lfilter(&b, &a, &x).unwrap();
    let ys = y.as_slice();

    let expected = [1.0 / 3.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    assert_close(ys, &expected, 1e-10, "lfilter 3-point moving average");
}

// Identity filter: b=[1], a=[1] => output == input.
#[test]
fn scipy_lfilter_identity() {
    let b = vec1d(&[1.0]);
    let a = vec1d(&[1.0]);
    let x = vec1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = lfilter(&b, &a, &x).unwrap();
    assert_close(
        y.as_slice(),
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        TOL,
        "lfilter identity",
    );
}

// Gain filter: b=[2], a=[1] => output = 2*input.
#[test]
fn scipy_lfilter_gain() {
    let b = vec1d(&[2.0]);
    let a = vec1d(&[1.0]);
    let x = vec1d(&[1.0, 2.0, 3.0]);
    let y = lfilter(&b, &a, &x).unwrap();
    assert_close(y.as_slice(), &[2.0, 4.0, 6.0], TOL, "lfilter gain=2");
}

// First-difference filter: b=[1, -1], a=[1].
// y[n] = x[n] - x[n-1]  (with x[-1] = 0)
#[test]
fn scipy_lfilter_first_difference() {
    let b = vec1d(&[1.0, -1.0]);
    let a = vec1d(&[1.0]);
    let x = vec1d(&[1.0, 3.0, 6.0, 10.0]);
    let y = lfilter(&b, &a, &x).unwrap();
    // y = [1-0, 3-1, 6-3, 10-6] = [1, 2, 3, 4]
    assert_close(
        y.as_slice(),
        &[1.0, 2.0, 3.0, 4.0],
        TOL,
        "lfilter first difference",
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Peak detection (signal.find_peaks)
// ═══════════════════════════════════════════════════════════════════════

// scipy: signal.find_peaks([0, 1, 0, 2, 0, 3, 0, 2, 0]) => [1, 3, 5, 7]
#[test]
fn scipy_find_peaks_basic() {
    let x = vec1d(&[0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0]);
    let peaks = find_peaks(&x, None, None).unwrap();
    assert_eq!(peaks, vec![1, 3, 5, 7], "find_peaks basic");
}

// Single peak in a valley-peak-valley pattern.
#[test]
fn scipy_find_peaks_single() {
    let x = vec1d(&[0.0, 0.0, 5.0, 0.0, 0.0]);
    let peaks = find_peaks(&x, None, None).unwrap();
    assert_eq!(peaks, vec![2], "find_peaks single");
}

// No peaks when signal is monotonically increasing.
#[test]
fn scipy_find_peaks_monotonic() {
    let x = vec1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let peaks = find_peaks(&x, None, None).unwrap();
    assert!(peaks.is_empty(), "find_peaks monotonic should be empty");
}

// With min_height filter.
#[test]
fn scipy_find_peaks_min_height() {
    let x = vec1d(&[0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0]);
    let peaks = find_peaks(&x, Some(2.5), None).unwrap();
    // Only the peak at index 5 (value 3.0) exceeds height 2.5.
    assert_eq!(peaks, vec![5], "find_peaks with min_height=2.5");
}

// With min_distance filter.
#[test]
fn scipy_find_peaks_min_distance() {
    let x = vec1d(&[0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0]);
    let peaks = find_peaks(&x, None, Some(3)).unwrap();
    // Left-to-right greedy: keep index 1, skip 3 (dist=2 < 3), keep 5, skip 7 (dist=2 < 3).
    assert_eq!(peaks, vec![1, 5], "find_peaks with min_distance=3");
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Welch PSD (shape and basic properties)
// ═══════════════════════════════════════════════════════════════════════

// Verify that Welch produces non-negative PSD values and correct shape.
#[test]
fn scipy_welch_shape_and_nonneg() {
    let n = 256;
    let two_pi = 2.0 * std::f64::consts::PI;
    // Pure sine at a known normalized frequency.
    let data: Vec<f64> = (0..n)
        .map(|i| (two_pi * 0.1 * f64::from(i)).sin())
        .collect();
    let x = vec1d(&data);

    let segment_size = 64;
    let overlap = 32;
    let (freqs, psd) = welch(&x, segment_size, overlap).unwrap();

    // Expected freq_bins = next_power_of_two(64) / 2 + 1 = 64/2 + 1 = 33
    assert_eq!(freqs.numel(), 33, "welch freq bins");
    assert_eq!(psd.numel(), 33, "welch psd bins");

    // PSD values must be non-negative.
    for (i, &p) in psd.as_slice().iter().enumerate() {
        assert!(p >= 0.0, "welch psd[{i}] = {p} is negative");
    }

    // The peak PSD bin should correspond roughly to the sine frequency.
    let ps = psd.as_slice();
    let max_bin = ps
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    // Frequency resolution: each bin = 1/padded = 1/64 ≈ 0.015625
    // Sine at normalized freq 0.1 => expected bin ≈ 0.1 / 0.015625 ≈ 6.4
    // So the peak should be near bins 6 or 7.
    assert!(
        (4..=9).contains(&max_bin),
        "welch peak bin {max_bin} not near expected ~6"
    );
}

// DC signal should have all power in the DC bin.
#[test]
fn scipy_welch_dc_signal() {
    let data = vec![3.0_f64; 256];
    let x = vec1d(&data);
    let (_, psd) = welch(&x, 64, 32).unwrap();
    let ps = psd.as_slice();
    let dc = ps[0];
    // DC bin should be the largest bin (spectral leakage from windowing
    // means other bins won't be exactly zero, but DC should dominate).
    for (i, &p) in ps.iter().enumerate().skip(1) {
        assert!(
            p <= dc,
            "welch DC test: bin {i} power {p} should be <= DC power {dc}"
        );
    }
}
