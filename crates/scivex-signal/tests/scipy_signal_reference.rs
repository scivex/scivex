//! Reference tests comparing scivex-signal against known SciPy.signal values.

use scivex_core::Tensor;
use scivex_signal::convolution::{convolve, ConvolveMode};
use scivex_signal::window;

const TOL: f64 = 1e-10;

fn vec1d(data: &[f64]) -> Tensor<f64> {
    Tensor::from_vec(data.to_vec(), vec![data.len()]).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, msg: &str) {
    assert_eq!(actual.len(), expected.len(), "{msg}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{msg}: [{i}] got {a}, expected {e}"
        );
    }
}

// ─── Convolution ─────────────────────────────────────────────────────

#[test]
fn convolve_full_simple() {
    // np.convolve([1, 2, 3], [1, 1]) = [1, 3, 5, 3]
    let a = vec1d(&[1.0, 2.0, 3.0]);
    let b = vec1d(&[1.0, 1.0]);
    let c = convolve(&a, &b, ConvolveMode::Full).unwrap();
    assert_close(c.as_slice(), &[1.0, 3.0, 5.0, 3.0], TOL, "convolve full");
}

#[test]
fn convolve_full_identity() {
    // Convolving with [1] is identity
    let a = vec1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = vec1d(&[1.0]);
    let c = convolve(&a, &b, ConvolveMode::Full).unwrap();
    assert_close(
        c.as_slice(),
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        TOL,
        "convolve identity",
    );
}

#[test]
fn convolve_full_symmetric() {
    // np.convolve([1, 2, 1], [1, 2, 1]) = [1, 4, 8, 8, 4, 1] (wrong)
    // Actually: [1, 4, 6, 4, 1] — wait, let me compute:
    // [1,2,1] * [1,2,1]:
    // c[0] = 1*1 = 1
    // c[1] = 1*2 + 2*1 = 4
    // c[2] = 1*1 + 2*2 + 1*1 = 6
    // c[3] = 2*1 + 1*2 = 4
    // c[4] = 1*1 = 1
    let a = vec1d(&[1.0, 2.0, 1.0]);
    let b = vec1d(&[1.0, 2.0, 1.0]);
    let c = convolve(&a, &b, ConvolveMode::Full).unwrap();
    assert_close(
        c.as_slice(),
        &[1.0, 4.0, 6.0, 4.0, 1.0],
        TOL,
        "convolve symmetric",
    );
}

#[test]
fn convolve_same_mode() {
    // np.convolve([1, 2, 3, 4, 5], [1, 0, -1], mode='same')
    // full = [-1, -2, -2, -2, -2, 4, 5] (length 7)
    // Actually full result of [1,2,3,4,5] conv [1,0,-1]:
    // c[0] = 1*1 = 1
    // c[1] = 2*1 + 1*0 = 2
    // c[2] = 3*1 + 2*0 + 1*(-1) = 2
    // c[3] = 4*1 + 3*0 + 2*(-1) = 2
    // c[4] = 5*1 + 4*0 + 3*(-1) = 2
    // c[5] = 5*0 + 4*(-1) = -4
    // c[6] = 5*(-1) = -5
    // 'same' takes center 5 values: [2, 2, 2, 2, -4]
    let a = vec1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = vec1d(&[1.0, 0.0, -1.0]);
    let c = convolve(&a, &b, ConvolveMode::Same).unwrap();
    assert_eq!(c.as_slice().len(), 5);
    assert_close(
        c.as_slice(),
        &[2.0, 2.0, 2.0, 2.0, -4.0],
        TOL,
        "convolve same",
    );
}

// ─── Window functions ────────────────────────────────────────────────

#[test]
fn hann_window_endpoints_zero() {
    // Hann window: first and last values are 0, center is 1
    let w = window::hann::<f64>(5).unwrap();
    let d = w.as_slice();
    assert!(d[0].abs() < TOL, "hann[0]={}", d[0]);
    assert!(d[4].abs() < TOL, "hann[4]={}", d[4]);
    assert!((d[2] - 1.0).abs() < TOL, "hann[2]={}", d[2]);
}

#[test]
fn hamming_window_endpoints() {
    // Hamming: endpoints are 0.08 (= 0.54 - 0.46)
    let w = window::hamming::<f64>(5).unwrap();
    let d = w.as_slice();
    assert!((d[0] - 0.08).abs() < TOL, "hamming[0]={}", d[0]);
    assert!((d[4] - 0.08).abs() < TOL, "hamming[4]={}", d[4]);
}

#[test]
fn hann_window_symmetry() {
    let w = window::hann::<f64>(8).unwrap();
    let d = w.as_slice();
    for i in 0..4 {
        assert!(
            (d[i] - d[7 - i]).abs() < TOL,
            "hann[{i}]={} != hann[{}]={}",
            d[i],
            7 - i,
            d[7 - i]
        );
    }
}

#[test]
fn blackman_endpoints_zero() {
    let w = window::blackman::<f64>(5).unwrap();
    let d = w.as_slice();
    // Blackman endpoints: 0.42 - 0.5 + 0.08 = 0.0 (at cos(0)=1 and cos(0)=1)
    assert!(d[0].abs() < TOL, "blackman[0]={}", d[0]);
    assert!(d[4].abs() < TOL, "blackman[4]={}", d[4]);
}
