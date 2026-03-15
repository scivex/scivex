//! Property-based tests for scivex-core using proptest.

use proptest::prelude::*;
use scivex_core::Tensor;
use scivex_core::linalg::{CholeskyDecomposition, LuDecomposition, QrDecomposition, dot};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn arb_vec(n: usize) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(-100.0_f64..100.0, n)
}

fn arb_matrix(n: usize, m: usize) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(-10.0_f64..10.0, n * m)
}

// ---------------------------------------------------------------------------
// Tensor creation properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn zeros_has_all_zeros(n in 1usize..500) {
        let t = Tensor::<f64>::zeros(vec![n]);
        prop_assert!(t.as_slice().iter().all(|&x| x == 0.0));
        prop_assert_eq!(t.numel(), n);
    }

    #[test]
    fn ones_has_all_ones(n in 1usize..500) {
        let t = Tensor::<f64>::ones(vec![n]);
        prop_assert!(t.as_slice().iter().all(|&x| (x - 1.0).abs() < f64::EPSILON));
        prop_assert_eq!(t.numel(), n);
    }

    #[test]
    fn full_fills_with_value(n in 1usize..500, v in -1000.0_f64..1000.0) {
        let t = Tensor::<f64>::full(vec![n], v);
        prop_assert!(t.as_slice().iter().all(|&x| (x - v).abs() < f64::EPSILON));
    }

    #[test]
    fn from_vec_preserves_data(data in arb_vec(100)) {
        let t = Tensor::from_vec(data.clone(), vec![100]).unwrap();
        prop_assert_eq!(t.as_slice(), &data[..]);
    }

    #[test]
    fn arange_produces_sequential(n in 1usize..500) {
        let t = Tensor::<f64>::arange(n);
        for i in 0..n {
            prop_assert!((t.as_slice()[i] - i as f64).abs() < f64::EPSILON);
        }
    }
}

// ---------------------------------------------------------------------------
// Reshape properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn reshape_preserves_data(data in arb_vec(120)) {
        let t = Tensor::from_vec(data.clone(), vec![120]).unwrap();
        let r = t.reshape(vec![10, 12]).unwrap();
        prop_assert_eq!(r.as_slice(), &data[..]);
        prop_assert_eq!(r.shape(), &[10, 12]);
    }

    #[test]
    fn reshape_roundtrip(rows in 1usize..20, cols in 1usize..20) {
        let n = rows * cols;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data.clone(), vec![rows, cols]).unwrap();
        let flat = t.reshape(vec![n]).unwrap();
        let back = flat.reshape(vec![rows, cols]).unwrap();
        prop_assert_eq!(back.as_slice(), &data[..]);
        prop_assert_eq!(back.shape(), &[rows, cols]);
    }
}

// ---------------------------------------------------------------------------
// Transpose properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn transpose_of_transpose_is_identity(
        rows in 2usize..20,
        cols in 2usize..20,
        data in arb_matrix(20, 20),
    ) {
        let n = rows * cols;
        let slice = &data[..n];
        let t = Tensor::from_vec(slice.to_vec(), vec![rows, cols]).unwrap();
        let tt = t.transpose().unwrap().transpose().unwrap();
        prop_assert_eq!(tt.shape(), t.shape());
        for (a, b) in tt.as_slice().iter().zip(t.as_slice().iter()) {
            prop_assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn transpose_shape_swap(rows in 2usize..20, cols in 2usize..20) {
        let t = Tensor::<f64>::ones(vec![rows, cols]);
        let tt = t.transpose().unwrap();
        prop_assert_eq!(tt.shape(), &[cols, rows]);
    }
}

// ---------------------------------------------------------------------------
// Element-wise arithmetic properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn add_commutative(data_a in arb_vec(50), data_b in arb_vec(50)) {
        let a = Tensor::from_vec(data_a, vec![50]).unwrap();
        let b = Tensor::from_vec(data_b, vec![50]).unwrap();
        let ab = &a + &b;
        let ba = &b + &a;
        for (x, y) in ab.as_slice().iter().zip(ba.as_slice().iter()) {
            prop_assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn mul_commutative(data_a in arb_vec(50), data_b in arb_vec(50)) {
        let a = Tensor::from_vec(data_a, vec![50]).unwrap();
        let b = Tensor::from_vec(data_b, vec![50]).unwrap();
        let ab = &a * &b;
        let ba = &b * &a;
        for (x, y) in ab.as_slice().iter().zip(ba.as_slice().iter()) {
            prop_assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn add_zero_identity(data in arb_vec(50)) {
        let a = Tensor::from_vec(data.clone(), vec![50]).unwrap();
        let z = Tensor::<f64>::zeros(vec![50]);
        let result = &a + &z;
        for (x, y) in result.as_slice().iter().zip(data.iter()) {
            prop_assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn mul_one_identity(data in arb_vec(50)) {
        let a = Tensor::from_vec(data.clone(), vec![50]).unwrap();
        let one = Tensor::<f64>::ones(vec![50]);
        let result = &a * &one;
        for (x, y) in result.as_slice().iter().zip(data.iter()) {
            prop_assert!((x - y).abs() < 1e-10);
        }
    }
}

// ---------------------------------------------------------------------------
// Dot product properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn dot_commutative(data_a in arb_vec(50), data_b in arb_vec(50)) {
        let a = Tensor::from_vec(data_a, vec![50]).unwrap();
        let b = Tensor::from_vec(data_b, vec![50]).unwrap();
        let ab = dot(&a, &b).unwrap();
        let ba = dot(&b, &a).unwrap();
        prop_assert!((ab - ba).abs() < 1e-6);
    }

    #[test]
    fn dot_with_zero_is_zero(data in arb_vec(50)) {
        let a = Tensor::from_vec(data, vec![50]).unwrap();
        let z = Tensor::<f64>::zeros(vec![50]);
        let result = dot(&a, &z).unwrap();
        prop_assert!(result.abs() < 1e-10);
    }

    #[test]
    fn dot_self_is_nonnegative(data in arb_vec(50)) {
        let a = Tensor::from_vec(data, vec![50]).unwrap();
        let result = dot(&a, &a).unwrap();
        prop_assert!(result >= -1e-10);
    }
}

// ---------------------------------------------------------------------------
// LU decomposition: P*A = L*U
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn lu_decomposition_reconstructs(n in 3usize..10) {
        // Diagonally dominant matrix (guaranteed non-singular)
        let mut data = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                data[i * n + j] = if i == j { (n * 10) as f64 } else { ((i + j) % 5) as f64 };
            }
        }
        let a = Tensor::from_vec(data, vec![n, n]).unwrap();
        let lu = LuDecomposition::decompose(&a).unwrap();
        let l = lu.l();
        let u = lu.u();
        let p = lu.p();
        // P * A should equal L * U
        let pa = p.matmul(&a).unwrap();
        let lu_product = l.matmul(&u).unwrap();
        for i in 0..n {
            for j in 0..n {
                let expected = *pa.get(&[i, j]).unwrap();
                let got = *lu_product.get(&[i, j]).unwrap();
                prop_assert!((expected - got).abs() < 1e-6,
                    "LU mismatch at ({}, {}): expected {}, got {}", i, j, expected, got);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// QR decomposition: A = Q*R, Q^T*Q = I
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn qr_orthogonal_q(n in 3usize..8) {
        let mut data = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                data[i * n + j] = if i == j { 10.0 } else { ((i * 3 + j * 7) % 10) as f64 };
            }
        }
        let a = Tensor::from_vec(data, vec![n, n]).unwrap();
        let qr = QrDecomposition::decompose(&a).unwrap();
        let q = qr.q();
        let qt = q.transpose().unwrap();
        let qtq = qt.matmul(&q).unwrap();
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                let got = *qtq.get(&[i, j]).unwrap();
                prop_assert!((expected - got).abs() < 1e-6,
                    "Q^T*Q not identity at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn qr_reconstructs(n in 3usize..8) {
        let mut data = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                data[i * n + j] = if i == j { 10.0 } else { ((i * 3 + j * 7) % 10) as f64 };
            }
        }
        let a = Tensor::from_vec(data, vec![n, n]).unwrap();
        let qr = QrDecomposition::decompose(&a).unwrap();
        let q = qr.q();
        let r = qr.r();
        let reconstructed = q.matmul(&r).unwrap();
        for i in 0..n {
            for j in 0..n {
                let expected = *a.get(&[i, j]).unwrap();
                let got = *reconstructed.get(&[i, j]).unwrap();
                prop_assert!((expected - got).abs() < 1e-6,
                    "QR reconstruction mismatch at ({}, {})", i, j);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cholesky decomposition: A = L*L^T (for SPD matrices)
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn cholesky_reconstructs(n in 3usize..8) {
        let raw: Vec<f64> = (0..n * n).map(|i| ((i * 7 + 3) % 20) as f64 - 10.0).collect();
        let m = Tensor::from_vec(raw, vec![n, n]).unwrap();
        let mt = m.transpose().unwrap();
        let mut mtm = mt.matmul(&m).unwrap().into_vec();
        for i in 0..n {
            mtm[i * n + i] += (n * 10) as f64;
        }
        let spd = Tensor::from_vec(mtm, vec![n, n]).unwrap();
        let chol = CholeskyDecomposition::decompose(&spd).unwrap();
        let l = chol.l();
        let lt = l.transpose().unwrap();
        let reconstructed = l.matmul(&lt).unwrap();
        for i in 0..n {
            for j in 0..n {
                let expected = *spd.get(&[i, j]).unwrap();
                let got = *reconstructed.get(&[i, j]).unwrap();
                prop_assert!((expected - got).abs() < 1e-4,
                    "Cholesky mismatch at ({}, {})", i, j);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Map preserves shape
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn map_preserves_shape(rows in 1usize..20, cols in 1usize..20) {
        let t = Tensor::<f64>::ones(vec![rows, cols]);
        let mapped = t.map(|x| x * 2.0);
        prop_assert_eq!(mapped.shape(), t.shape());
        prop_assert_eq!(mapped.numel(), t.numel());
    }

    #[test]
    fn map_identity(data in arb_vec(50)) {
        let t = Tensor::from_vec(data.clone(), vec![50]).unwrap();
        let mapped = t.map(|x| x);
        prop_assert_eq!(mapped.as_slice(), &data[..]);
    }
}

// ---------------------------------------------------------------------------
// Matmul associativity
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn matmul_associative(n in 3usize..6) {
        let data_a: Vec<f64> = (0..n * n).map(|i| ((i * 3 + 1) % 7) as f64).collect();
        let data_b: Vec<f64> = (0..n * n).map(|i| ((i * 5 + 2) % 7) as f64).collect();
        let data_c: Vec<f64> = (0..n * n).map(|i| ((i * 7 + 3) % 7) as f64).collect();
        let a = Tensor::from_vec(data_a, vec![n, n]).unwrap();
        let b = Tensor::from_vec(data_b, vec![n, n]).unwrap();
        let c = Tensor::from_vec(data_c, vec![n, n]).unwrap();
        let ab_c = a.matmul(&b).unwrap().matmul(&c).unwrap();
        let a_bc = a.matmul(&b.matmul(&c).unwrap()).unwrap();
        for (x, y) in ab_c.as_slice().iter().zip(a_bc.as_slice().iter()) {
            prop_assert!((x - y).abs() < 1e-6, "matmul not associative");
        }
    }
}
