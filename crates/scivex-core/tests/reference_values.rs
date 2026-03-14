//! Reference value tests comparing scivex-core outputs against known NumPy/SciPy results.
//!
//! All reference values were pre-computed with NumPy 1.x / SciPy and are verified
//! to the tolerances specified below.

use scivex_core::linalg::{
    self, CholeskyDecomposition, EigDecomposition, LuDecomposition, QrDecomposition,
    SvdDecomposition,
};
use scivex_core::tensor::Tensor;
use scivex_core::{fft, linalg::dot};

// ---------------------------------------------------------------------------
// Tolerance helpers
// ---------------------------------------------------------------------------

/// Tolerance for decompositions (QR, LU, Cholesky, SVD, Eigen, FFT).
const DECOMP_TOL: f64 = 1e-6;

/// Tolerance for exact (integer-arithmetic) operations (matmul, dot, transpose).
const EXACT_TOL: f64 = 1e-10;

/// Assert that two f64 values are approximately equal within `tol`.
fn assert_approx(actual: f64, expected: f64, tol: f64, msg: &str) {
    assert!(
        (actual - expected).abs() < tol,
        "{msg}: expected {expected}, got {actual} (diff = {})",
        (actual - expected).abs()
    );
}

/// Assert that two slices are element-wise approximately equal within `tol`.
fn assert_slice_approx(actual: &[f64], expected: &[f64], tol: f64, msg: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{msg}: length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{msg}[{i}]: expected {e}, got {a} (diff = {})",
            (a - e).abs()
        );
    }
}

// ===========================================================================
// 1. Matrix multiplication: A @ B
//    A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
//    C = [[19, 22], [43, 50]]
// ===========================================================================

#[test]
fn test_matmul_numpy_reference() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let c = a.matmul(&b).unwrap();

    assert_slice_approx(c.as_slice(), &[19.0, 22.0, 43.0, 50.0], EXACT_TOL, "matmul");
}

// ===========================================================================
// 2. LU decomposition of [[2, 1], [6, 4]]
//    P = [[0, 1], [1, 0]], L = [[1, 0], [1/3, 1]], U = [[6, 4], [0, -1/3]]
//    Verify PA = LU
// ===========================================================================

#[test]
fn test_lu_decomposition_numpy_reference() {
    let a = Tensor::from_vec(vec![2.0, 1.0, 6.0, 4.0], vec![2, 2]).unwrap();
    let lu = LuDecomposition::decompose(&a).unwrap();

    let p = lu.p();
    let l = lu.l();
    let u = lu.u();

    // Verify P is a permutation matrix: P = [[0,1],[1,0]]
    let p_expected = &[0.0, 1.0, 1.0, 0.0];
    assert_slice_approx(p.as_slice(), p_expected, DECOMP_TOL, "LU P");

    // Verify L: [[1, 0], [1/3, 1]]
    let l_expected = &[1.0, 0.0, 1.0 / 3.0, 1.0];
    assert_slice_approx(l.as_slice(), l_expected, DECOMP_TOL, "LU L");

    // Verify U: [[6, 4], [0, -1/3]]
    let u_expected = &[6.0, 4.0, 0.0, -1.0 / 3.0];
    assert_slice_approx(u.as_slice(), u_expected, DECOMP_TOL, "LU U");

    // Verify PA = LU
    let pa = p.matmul(&a).unwrap();
    let lu_prod = l.matmul(&u).unwrap();
    assert_slice_approx(pa.as_slice(), lu_prod.as_slice(), DECOMP_TOL, "PA = LU");
}

// ===========================================================================
// 3. QR decomposition of [[1, -1], [1, 1]]
//    Q ~ [[-0.7071, 0.7071], [-0.7071, -0.7071]]
//    R ~ [[-1.4142, 0], [0, 1.4142]]
//    Verify A = QR and Q^T Q = I
// ===========================================================================

#[test]
fn test_qr_decomposition_numpy_reference() {
    let a = Tensor::from_vec(vec![1.0_f64, -1.0, 1.0, 1.0], vec![2, 2]).unwrap();
    let qr = QrDecomposition::decompose(&a).unwrap();

    let q = qr.q();
    let r = qr.r();

    // Verify Q values (signs may differ from NumPy due to Householder sign convention,
    // but Q^T Q = I and A = QR must hold)
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();

    // Check |Q[i,j]| matches expected magnitudes
    for &val in q.as_slice() {
        assert_approx(val.abs(), sqrt2_inv, DECOMP_TOL, "QR |Q| element");
    }

    // Check |R diagonal| values
    let sqrt2 = 2.0_f64.sqrt();
    assert_approx(r.as_slice()[0].abs(), sqrt2, DECOMP_TOL, "QR |R[0,0]|");
    assert_approx(r.as_slice()[3].abs(), sqrt2, DECOMP_TOL, "QR |R[1,1]|");

    // Off-diagonal of R should be zero (for this specific matrix)
    assert_approx(r.as_slice()[1].abs(), 0.0, DECOMP_TOL, "QR R[0,1]");

    // Verify A = QR
    let qr_prod = q.matmul(&r).unwrap();
    assert_slice_approx(qr_prod.as_slice(), a.as_slice(), DECOMP_TOL, "A = QR");

    // Verify Q^T Q = I
    let qt = q.transpose().unwrap();
    let qtq = qt.matmul(&q).unwrap();
    let eye = Tensor::<f64>::eye(2);
    assert_slice_approx(qtq.as_slice(), eye.as_slice(), DECOMP_TOL, "Q^T Q = I");
}

// ===========================================================================
// 4. Cholesky of [[4, 2], [2, 3]]
//    L = [[2, 0], [1, sqrt(2)]]
// ===========================================================================

#[test]
fn test_cholesky_numpy_reference() {
    let a = Tensor::from_vec(vec![4.0, 2.0, 2.0, 3.0], vec![2, 2]).unwrap();
    let chol = CholeskyDecomposition::decompose(&a).unwrap();
    let l = chol.l();

    let sqrt2 = 2.0_f64.sqrt();
    let l_expected = &[2.0, 0.0, 1.0, sqrt2];
    assert_slice_approx(l.as_slice(), l_expected, DECOMP_TOL, "Cholesky L");

    // Verify L L^T = A
    let lt = l.transpose().unwrap();
    let prod = l.matmul(&lt).unwrap();
    assert_slice_approx(prod.as_slice(), a.as_slice(), DECOMP_TOL, "L L^T = A");
}

// ===========================================================================
// 5. SVD of [[1, 2], [3, 4], [5, 6]]
//    S ~ [9.52552, 0.51430]
//    Verify reconstruction A = U diag(S) V^T
// ===========================================================================

#[test]
fn test_svd_numpy_reference() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
    let svd = SvdDecomposition::decompose(&a).unwrap();
    let s = svd.singular_values();

    // Check singular values
    assert_approx(s[0], 9.525_518_091_565_107, DECOMP_TOL, "SVD s[0]");
    assert_approx(s[1], 0.514_300_580_658_644_6, DECOMP_TOL, "SVD s[1]");

    // Verify reconstruction: A = U * diag(s) * V^T
    let u = svd.u();
    let vt = svd.vt();
    let (m, n) = (3, 2);
    let mut s_mat = vec![0.0_f64; m * n];
    for i in 0..n {
        s_mat[i * n + i] = s[i];
    }
    let s_tensor = Tensor::from_vec(s_mat, vec![m, n]).unwrap();
    let us = u.matmul(&s_tensor).unwrap();
    let reconstructed = us.matmul(&vt).unwrap();
    assert_slice_approx(
        reconstructed.as_slice(),
        a.as_slice(),
        DECOMP_TOL,
        "SVD reconstruction",
    );

    // Verify U^T U = I (m x m)
    let ut = u.transpose().unwrap();
    let utu = ut.matmul(&u).unwrap();
    let eye_m = Tensor::<f64>::eye(m);
    assert_slice_approx(utu.as_slice(), eye_m.as_slice(), DECOMP_TOL, "U^T U = I");

    // Verify V^T V = I (n x n)
    let v = vt.transpose().unwrap();
    let vtv = vt.matmul(&v).unwrap();
    let eye_n = Tensor::<f64>::eye(n);
    assert_slice_approx(vtv.as_slice(), eye_n.as_slice(), DECOMP_TOL, "V^T V = I");
}

// ===========================================================================
// 6. Eigenvalues of [[2, 1], [1, 2]]
//    eigenvalues = [1.0, 3.0]  (sorted ascending by np.linalg.eigh)
//    scivex sorts by descending absolute value, so expect [3.0, 1.0]
// ===========================================================================

#[test]
fn test_eig_numpy_reference() {
    let a = Tensor::from_vec(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
    let eig = EigDecomposition::decompose_symmetric(&a).unwrap();
    let vals = eig.eigenvalues();

    // scivex returns eigenvalues sorted by descending |value|
    assert_approx(vals[0], 3.0, DECOMP_TOL, "eig[0]");
    assert_approx(vals[1], 1.0, DECOMP_TOL, "eig[1]");

    // Verify reconstruction: A = V diag(d) V^T
    let v = eig.eigenvectors();
    let vt = v.transpose().unwrap();
    let mut d_data = vec![0.0_f64; 4];
    d_data[0] = vals[0];
    d_data[3] = vals[1];
    let d = Tensor::from_vec(d_data, vec![2, 2]).unwrap();
    let vd = v.matmul(&d).unwrap();
    let reconstructed = vd.matmul(&vt).unwrap();
    assert_slice_approx(
        reconstructed.as_slice(),
        a.as_slice(),
        DECOMP_TOL,
        "Eigen reconstruction",
    );

    // Verify V^T V = I (orthogonal eigenvectors)
    let vtv = vt.matmul(&v).unwrap();
    let eye = Tensor::<f64>::eye(2);
    assert_slice_approx(vtv.as_slice(), eye.as_slice(), DECOMP_TOL, "V^T V = I");
}

// ===========================================================================
// 7. Determinant of [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
//    det = -3.0
// ===========================================================================

#[test]
fn test_det_numpy_reference() {
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        vec![3, 3],
    )
    .unwrap();
    let det = linalg::det(&a).unwrap();
    assert_approx(det, -3.0, EXACT_TOL, "det");
}

/// Also verify via the convenience method on Tensor.
#[test]
fn test_det_tensor_method_numpy_reference() {
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        vec![3, 3],
    )
    .unwrap();
    let det = a.det().unwrap();
    assert_approx(det, -3.0, EXACT_TOL, "det (method)");
}

// ===========================================================================
// 8. Dot product
//    a = [1, 2, 3, 4, 5], b = [2, 3, 4, 5, 6]
//    dot = 70.0
// ===========================================================================

#[test]
fn test_dot_numpy_reference() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let b = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0], vec![5]).unwrap();
    let result = dot(&a, &b).unwrap();
    assert_approx(result, 70.0, EXACT_TOL, "dot product");
}

/// Also verify via the convenience method on Tensor.
#[test]
fn test_dot_tensor_method_numpy_reference() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let b = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0], vec![5]).unwrap();
    let result = a.dot(&b).unwrap();
    assert_approx(result, 70.0, EXACT_TOL, "dot product (method)");
}

// ===========================================================================
// 9. FFT of [1, 0, -1, 0]
//    Full complex FFT result: [0+0j, 2+0j, 0+0j, 2+0j]
//    Real parts: [0, 2, 0, 2], Imag parts: [0, 0, 0, 0]
// ===========================================================================

#[test]
fn test_fft_numpy_reference() {
    // scivex fft::fft expects complex input [N, 2] (interleaved re/im)
    // Signal [1, 0, -1, 0] as complex: [(1,0), (0,0), (-1,0), (0,0)]
    let input =
        Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], vec![4, 2]).unwrap();
    let spectrum = fft::fft(&input).unwrap();

    assert_eq!(spectrum.shape(), &[4, 2]);
    let s = spectrum.as_slice();

    // Expected: bins [0+0j, 2+0j, 0+0j, 2+0j]
    let expected_re = [0.0, 2.0, 0.0, 2.0];
    let expected_im = [0.0, 0.0, 0.0, 0.0];

    for i in 0..4 {
        assert_approx(
            s[i * 2],
            expected_re[i],
            DECOMP_TOL,
            &format!("FFT re[{i}]"),
        );
        assert_approx(
            s[i * 2 + 1],
            expected_im[i],
            DECOMP_TOL,
            &format!("FFT im[{i}]"),
        );
    }
}

/// Also test via rfft (real-to-complex), which returns only non-negative frequencies.
#[test]
fn test_rfft_numpy_reference() {
    let signal = Tensor::from_vec(vec![1.0, 0.0, -1.0, 0.0], vec![4]).unwrap();
    let spectrum = fft::rfft(&signal).unwrap();

    // rfft returns N/2+1 = 3 complex bins
    assert_eq!(spectrum.shape(), &[3, 2]);
    let s = spectrum.as_slice();

    // Bins 0..2 of the full FFT: [0+0j, 2+0j, 0+0j]
    let expected_re = [0.0, 2.0, 0.0];
    let expected_im = [0.0, 0.0, 0.0];

    for i in 0..3 {
        assert_approx(
            s[i * 2],
            expected_re[i],
            DECOMP_TOL,
            &format!("RFFT re[{i}]"),
        );
        assert_approx(
            s[i * 2 + 1],
            expected_im[i],
            DECOMP_TOL,
            &format!("RFFT im[{i}]"),
        );
    }
}

// ===========================================================================
// 10. Transpose of [[1, 2, 3], [4, 5, 6]]
//     A^T = [[1, 4], [2, 5], [3, 6]]
// ===========================================================================

#[test]
fn test_transpose_numpy_reference() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let at = a.transpose().unwrap();

    assert_eq!(at.shape(), &[3, 2]);
    assert_slice_approx(
        at.as_slice(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        EXACT_TOL,
        "transpose",
    );
}

// ===========================================================================
// Additional cross-checks combining multiple operations
// ===========================================================================

/// Verify that (A @ B)^T = B^T @ A^T for the reference matrices.
#[test]
fn test_matmul_transpose_identity() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

    let ab = a.matmul(&b).unwrap();
    let ab_t = ab.transpose().unwrap();

    let bt = b.transpose().unwrap();
    let at = a.transpose().unwrap();
    let bt_at = bt.matmul(&at).unwrap();

    assert_slice_approx(
        ab_t.as_slice(),
        bt_at.as_slice(),
        EXACT_TOL,
        "(AB)^T = B^T A^T",
    );
}

/// Verify det(A) via LU matches the known value, and that det(A^{-1}) = 1/det(A).
#[test]
fn test_det_inverse_relationship() {
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        vec![3, 3],
    )
    .unwrap();
    let det_a = linalg::det(&a).unwrap();
    assert_approx(det_a, -3.0, EXACT_TOL, "det(A)");

    let inv_a = linalg::inv(&a).unwrap();
    let det_inv_a = linalg::det(&inv_a).unwrap();
    assert_approx(det_inv_a, 1.0 / det_a, DECOMP_TOL, "det(A^{-1}) = 1/det(A)");
}

/// Verify FFT round-trip: ifft(fft(x)) = x for the reference signal.
#[test]
fn test_fft_ifft_roundtrip_numpy_reference() {
    let input =
        Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], vec![4, 2]).unwrap();
    let spectrum = fft::fft(&input).unwrap();
    let recovered = fft::ifft(&spectrum).unwrap();

    assert_slice_approx(
        recovered.as_slice(),
        input.as_slice(),
        DECOMP_TOL,
        "FFT round-trip",
    );
}

/// Verify that solving Ax = b via LU gives the correct result for a 3x3 system,
/// and that A * x = b.
#[test]
fn test_solve_numpy_reference() {
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        vec![3, 3],
    )
    .unwrap();
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let x = linalg::solve(&a, &b).unwrap();

    // np.linalg.solve result: [-1/3, 2/3, 0]
    assert_approx(x.as_slice()[0], -1.0 / 3.0, DECOMP_TOL, "solve x[0]");
    assert_approx(x.as_slice()[1], 2.0 / 3.0, DECOMP_TOL, "solve x[1]");
    assert_approx(x.as_slice()[2], 0.0, DECOMP_TOL, "solve x[2]");

    // Verify A * x = b
    let ax = a.matvec(&x).unwrap();
    assert_slice_approx(ax.as_slice(), b.as_slice(), DECOMP_TOL, "A * x = b");
}

/// Verify that Cholesky solve gives correct result and that A * x = b.
#[test]
fn test_cholesky_solve_numpy_reference() {
    let a = Tensor::from_vec(vec![4.0, 2.0, 2.0, 3.0], vec![2, 2]).unwrap();
    let b = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
    let chol = CholeskyDecomposition::decompose(&a).unwrap();
    let x = chol.solve(&b).unwrap();

    // Verify A * x = b
    let ax = a.matvec(&x).unwrap();
    assert_slice_approx(ax.as_slice(), b.as_slice(), DECOMP_TOL, "Cholesky A*x = b");
}
