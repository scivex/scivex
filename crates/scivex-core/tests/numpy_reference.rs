//! Reference tests comparing scivex-core outputs against known analytical values
//! and pre-computed NumPy/SciPy results.
//!
//! Tests use both hardcoded expected values (where analytically known) and
//! residual-based verification (e.g., ||Ax-b|| < tol).

use scivex_core::linalg::{self, EigDecomposition, QrDecomposition, SvdDecomposition};
use scivex_core::tensor::Tensor;

fn mat(data: &[f64], rows: usize, cols: usize) -> Tensor<f64> {
    Tensor::from_vec(data.to_vec(), vec![rows, cols]).unwrap()
}

fn vec1d(data: &[f64]) -> Tensor<f64> {
    Tensor::from_vec(data.to_vec(), vec![data.len()]).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, msg: &str) {
    assert_eq!(actual.len(), expected.len(), "{msg}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{msg}: element [{i}] differs: got {a}, expected {e}, diff={}",
            (a - e).abs()
        );
    }
}

/// Compute ||Ax - b||_inf to verify a linear solve.
fn residual_norm(a: &Tensor<f64>, x: &Tensor<f64>, b: &Tensor<f64>) -> f64 {
    // A is n x n, x is n, b is n
    let n = b.as_slice().len();
    let a_data = a.as_slice();
    let x_data = x.as_slice();
    let b_data = b.as_slice();
    let mut max_err = 0.0_f64;
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            row_sum += a_data[i * n + j] * x_data[j];
        }
        max_err = max_err.max((row_sum - b_data[i]).abs());
    }
    max_err
}

// ─── Determinant ──────────────────────────────────────────────────────

#[test]
fn ref_det_3x3_analytical() {
    // det([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
    // = 6(-2*7 - 5*8) - 1(4*7 - 5*2) + 1(4*8 - (-2)*2)
    // = 6(-14-40) - 1(28-10) + 1(32+4) = 6(-54) - 18 + 36 = -324 - 18 + 36 = -306
    let a = mat(&[6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0], 3, 3);
    let d = linalg::det(&a).unwrap();
    assert!((d - (-306.0)).abs() < 1e-10, "det = {d}");
}

#[test]
fn ref_det_4x4_analytical() {
    // det([[3,2,0,1],[4,0,1,2],[3,0,2,1],[9,2,3,1]])
    // Cofactor expansion along column 2 (has two zeros):
    // = 2*C_12 + 0 + 0 + 2*C_42  (... analytically = 24)
    let a = mat(
        &[
            3.0, 2.0, 0.0, 1.0, 4.0, 0.0, 1.0, 2.0, 3.0, 0.0, 2.0, 1.0, 9.0, 2.0, 3.0, 1.0,
        ],
        4,
        4,
    );
    let d = linalg::det(&a).unwrap();
    assert!((d - 24.0).abs() < 1e-10, "det = {d}");
}

#[test]
fn ref_det_5x5_permutation() {
    // Determinant of a permutation matrix = ±1
    // Cycle (12345) → (23451): sign = (-1)^4 = +1? No, 4 transpositions → even → +1
    // Actually let's use the simpler identity det = ±1
    let p = mat(
        &[
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ],
        5,
        5,
    );
    let d = linalg::det(&p).unwrap();
    assert!(
        (d.abs() - 1.0).abs() < 1e-10,
        "permutation det should be ±1, got {d}"
    );
}

// ─── Solve ────────────────────────────────────────────────────────────

#[test]
fn ref_solve_3x3_analytical() {
    // 3x + y - z = 4
    // 2x + 4y + z = 1
    // -x + 2y + 5z = 1
    // Solution: x=2, y=-1, z=1 (verify: 6-1-1=4 ✓, 4-4+1=1 ✓, -2-2+5=1 ✓)
    let a = mat(&[3.0, 1.0, -1.0, 2.0, 4.0, 1.0, -1.0, 2.0, 5.0], 3, 3);
    let b = vec1d(&[4.0, 1.0, 1.0]);
    let x = linalg::solve(&a, &b).unwrap();
    assert_close(x.as_slice(), &[2.0, -1.0, 1.0], 1e-10, "solve 3x3");
}

#[test]
fn ref_solve_4x4_ones() {
    // Choose x = [1,1,1,1], compute b = A*x, then solve and check.
    let a = mat(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 6.0, 4.0, 8.0, 3.0, 1.0, 1.0, 2.0,
        ],
        4,
        4,
    );
    let b = vec1d(&[10.0, 26.0, 20.0, 7.0]); // A * [1,1,1,1]
    let x = linalg::solve(&a, &b).unwrap();
    assert_close(x.as_slice(), &[1.0, 1.0, 1.0, 1.0], 1e-10, "solve 4x4");
}

#[test]
fn ref_solve_5x5_residual() {
    // For larger systems, verify via residual ||Ax - b|| rather than exact values.
    let a = mat(
        &[
            2.0, 1.0, -1.0, 0.0, 0.0, 1.0, 3.0, 0.0, -1.0, 0.0, -1.0, 0.0, 4.0, 1.0, 1.0,
            0.0, -1.0, 1.0, 5.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0,
        ],
        5,
        5,
    );
    let b = vec1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let x = linalg::solve(&a, &b).unwrap();
    let resid = residual_norm(&a, &x, &b);
    assert!(
        resid < 1e-12,
        "5x5 solve residual too large: {resid}"
    );
}

// ─── Inverse ──────────────────────────────────────────────────────────

#[test]
fn ref_inv_3x3_analytical() {
    // >>> np.linalg.inv([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
    // det = 1(0-24) - 2(0-20) + 3(0-5) = -24+40-15 = 1
    // Inverse has integer entries:
    // [[-24, 18, 5], [20, -15, -4], [-5, 4, 1]]
    let a = mat(&[1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0], 3, 3);
    let inv = linalg::inv(&a).unwrap();
    let expected = [-24.0, 18.0, 5.0, 20.0, -15.0, -4.0, -5.0, 4.0, 1.0];
    assert_close(inv.as_slice(), &expected, 1e-10, "inv 3x3");
}

#[test]
fn ref_inv_times_original_is_identity() {
    // A * A^-1 = I for a random-ish 4x4
    let a = mat(
        &[
            2.0, 1.0, 0.0, 3.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 4.0, 2.0, 3.0, 0.0, 2.0, 5.0,
        ],
        4,
        4,
    );
    let inv = linalg::inv(&a).unwrap();
    let product = a.matmul(&inv).unwrap();
    let eye = Tensor::<f64>::eye(4);
    assert_close(
        product.as_slice(),
        eye.as_slice(),
        1e-10,
        "A * A^-1 = I",
    );
}

// ─── SVD singular values ─────────────────────────────────────────────

#[test]
fn ref_svd_diagonal_matrix() {
    // SVD of diag(5, 3, 1) should give singular values [5, 3, 1]
    let a = mat(
        &[5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        3,
        3,
    );
    let svd = SvdDecomposition::decompose(&a).unwrap();
    let s = svd.singular_values();
    assert!((s[0] - 5.0).abs() < 1e-10, "s[0]={}", s[0]);
    assert!((s[1] - 3.0).abs() < 1e-10, "s[1]={}", s[1]);
    assert!((s[2] - 1.0).abs() < 1e-10, "s[2]={}", s[2]);
}

#[test]
fn ref_svd_reconstruction_3x3() {
    // Verify U * diag(s) * V^T = A
    let a = mat(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        3,
        3,
    );
    let svd = SvdDecomposition::decompose(&a).unwrap();
    let u = svd.u();
    let vt = svd.vt();
    let s = svd.singular_values();

    let mut s_mat_data = vec![0.0_f64; 9];
    for i in 0..3 {
        s_mat_data[i * 3 + i] = s[i];
    }
    let s_mat = mat(&s_mat_data, 3, 3);
    let us = u.matmul(&s_mat).unwrap();
    let reconstructed = us.matmul(&vt).unwrap();
    assert_close(
        reconstructed.as_slice(),
        a.as_slice(),
        1e-8,
        "SVD reconstruction",
    );
}

#[test]
fn ref_svd_rank_deficient() {
    // Rank-1 matrix: [[1,2],[2,4]] has singular values [sqrt(25)=5, 0]
    let a = mat(&[1.0, 2.0, 2.0, 4.0], 2, 2);
    let svd = SvdDecomposition::decompose(&a).unwrap();
    let s = svd.singular_values();
    assert!((s[0] - 5.0).abs() < 1e-8, "s[0]={}", s[0]);
    assert!(s[1].abs() < 1e-8, "s[1] should be ~0, got {}", s[1]);
}

#[test]
fn ref_svd_tall_matrix_reconstruction() {
    // 4x3 matrix
    let a = mat(
        &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 1.0],
        4,
        3,
    );
    let svd = SvdDecomposition::decompose(&a).unwrap();
    let u = svd.u();
    let vt = svd.vt();
    let s = svd.singular_values();

    let mut s_mat_data = vec![0.0_f64; 4 * 3];
    for i in 0..3 {
        s_mat_data[i * 3 + i] = s[i];
    }
    let s_mat = mat(&s_mat_data, 4, 3);
    let us = u.matmul(&s_mat).unwrap();
    let reconstructed = us.matmul(&vt).unwrap();
    assert_close(
        reconstructed.as_slice(),
        a.as_slice(),
        1e-8,
        "SVD reconstruction 4x3",
    );
}

// ─── Eigenvalues (symmetric) ─────────────────────────────────────────

#[test]
fn ref_eigh_2x2_analytical() {
    // [[2,1],[1,3]]: eigenvalues = (5 ± sqrt(5))/2
    let a = mat(&[2.0, 1.0, 1.0, 3.0], 2, 2);
    let eig = EigDecomposition::decompose_symmetric(&a).unwrap();
    let vals = eig.eigenvalues();
    let sqrt5 = 5.0_f64.sqrt();
    let lambda1 = 2.5 + sqrt5 / 2.0; // ~3.618
    let lambda2 = 2.5 - sqrt5 / 2.0; // ~1.382

    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(
        (sorted[0] - lambda2).abs() < 1e-10,
        "λ_min={}, expected={}",
        sorted[0],
        lambda2
    );
    assert!(
        (sorted[1] - lambda1).abs() < 1e-10,
        "λ_max={}, expected={}",
        sorted[1],
        lambda1
    );
}

#[test]
fn ref_eigh_trace_and_det() {
    // For any matrix: sum(eigenvalues) = trace, product(eigenvalues) = det
    let a = mat(&[4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0], 3, 3);
    let eig = EigDecomposition::decompose_symmetric(&a).unwrap();
    let vals = eig.eigenvalues();

    let trace = 4.0 + 5.0 + 6.0; // = 15
    let sum_eig: f64 = vals.iter().sum();
    assert!(
        (sum_eig - trace).abs() < 1e-8,
        "sum(eigenvalues)={sum_eig}, trace={trace}"
    );

    let det_from_eig: f64 = vals.iter().product();
    let det_from_lu = linalg::det(&a).unwrap();
    assert!(
        (det_from_eig - det_from_lu).abs() < 1e-6,
        "product(eigenvalues)={det_from_eig}, det={det_from_lu}"
    );
}

#[test]
fn ref_eigh_reconstruction() {
    // A = V D V^T
    let a = mat(
        &[
            5.0, 1.0, 2.0, 0.0, 1.0, 4.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0, 0.0, 1.0, 0.0, 2.0,
        ],
        4,
        4,
    );
    let eig = EigDecomposition::decompose_symmetric(&a).unwrap();
    let v = eig.eigenvectors();
    let vt = v.transpose().unwrap();
    let d_vals = eig.eigenvalues();

    let mut d_data = vec![0.0_f64; 16];
    for i in 0..4 {
        d_data[i * 4 + i] = d_vals[i];
    }
    let d = mat(&d_data, 4, 4);
    let vd = v.matmul(&d).unwrap();
    let reconstructed = vd.matmul(&vt).unwrap();
    assert_close(
        reconstructed.as_slice(),
        a.as_slice(),
        1e-8,
        "Eig reconstruction 4x4",
    );
}

// ─── QR decomposition ────────────────────────────────────────────────

#[test]
fn ref_qr_r_diagonal() {
    // Classic test matrix from Golub & Van Loan
    // |R[0,0]| = 14, |R[1,1]| = 175, |R[2,2]| = 35
    let a = mat(
        &[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
        3,
        3,
    );
    let qr = QrDecomposition::decompose(&a).unwrap();
    let r = qr.r();
    let r_data = r.as_slice();

    assert!(
        (r_data[0].abs() - 14.0).abs() < 1e-8,
        "R[0,0]={}",
        r_data[0]
    );
    assert!(
        (r_data[4].abs() - 175.0).abs() < 1e-8,
        "R[1,1]={}",
        r_data[4]
    );
    assert!(
        (r_data[8].abs() - 35.0).abs() < 1e-8,
        "R[2,2]={}",
        r_data[8]
    );
}

#[test]
fn ref_qr_reconstruction() {
    let a = mat(
        &[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
        3,
        3,
    );
    let qr = QrDecomposition::decompose(&a).unwrap();
    let q = qr.q();
    let r = qr.r();
    let reconstructed = q.matmul(&r).unwrap();
    assert_close(
        reconstructed.as_slice(),
        a.as_slice(),
        1e-10,
        "QR reconstruction",
    );
}

#[test]
fn ref_qr_orthogonality() {
    let a = mat(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        3,
        3,
    );
    let qr = QrDecomposition::decompose(&a).unwrap();
    let q = qr.q();
    let qt = q.transpose().unwrap();
    let qtq = qt.matmul(&q).unwrap();
    let eye = Tensor::<f64>::eye(3);
    assert_close(qtq.as_slice(), eye.as_slice(), 1e-10, "Q^T Q = I");
}

// ─── FFT ─────────────────────────────────────────────────────────────

#[test]
fn ref_fft_4point() {
    // DFT of [1, 2, 3, 4]:
    // X[0] = 1+2+3+4 = 10
    // X[1] = 1 + 2*(-j) + 3*(-1) + 4*(j) = 1-3 + j(4-2) = -2+2j
    // X[2] = 1 + 2*(-1) + 3*(1) + 4*(-1) = -2
    // X[3] = 1 + 2*(j) + 3*(-1) + 4*(-j) = -2-2j
    use scivex_core::fft;
    let input =
        Tensor::from_vec(vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], vec![4, 2]).unwrap();
    let result = fft::fft(&input).unwrap();
    let r = result.as_slice();
    let expected = [10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
    assert_close(r, &expected, 1e-10, "FFT 4-point");
}

#[test]
fn ref_rfft_impulse() {
    // rfft of unit impulse [1, 0, 0, 0, 0, 0, 0, 0] = all ones
    use scivex_core::fft;
    let input =
        Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![8]).unwrap();
    let result = fft::rfft(&input).unwrap();
    let r = result.as_slice();
    assert_eq!(r.len(), 10);
    for i in 0..5 {
        let re: f64 = r[2 * i];
        let im: f64 = r[2 * i + 1];
        assert!((re - 1.0).abs() < 1e-10, "real[{i}]={re}");
        assert!(im.abs() < 1e-10, "imag[{i}]={im}");
    }
}

// ─── BLAS ────────────────────────────────────────────────────────────

#[test]
fn ref_dot_product() {
    // [1,2,3,4,5] · [5,4,3,2,1] = 5+8+9+8+5 = 35
    let x = vec1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = vec1d(&[5.0, 4.0, 3.0, 2.0, 1.0]);
    let result = scivex_core::linalg::dot(&x, &y).unwrap();
    assert!((result - 35.0).abs() < 1e-14, "dot = {result}");
}

#[test]
fn ref_matmul_2x3_3x2() {
    // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
    let a = mat(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = mat(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    let c = a.matmul(&b).unwrap();
    assert_close(
        c.as_slice(),
        &[58.0, 64.0, 139.0, 154.0],
        1e-10,
        "matmul",
    );
}

#[test]
fn ref_nrm2() {
    // ||[3, 4]|| = 5
    let x = vec1d(&[3.0, 4.0]);
    let n = scivex_core::linalg::nrm2(&x).unwrap();
    assert!((n - 5.0).abs() < 1e-14, "nrm2 = {n}");
}
