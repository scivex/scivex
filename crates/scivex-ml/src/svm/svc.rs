use scivex_core::{Float, Tensor};

use crate::error::{MlError, Result};
use crate::traits::Predictor;

use super::kernel::Kernel;

/// Support Vector Classifier using the SMO (Sequential Minimal Optimization) algorithm.
///
/// Supports binary classification with arbitrary kernel functions.
/// Multi-class is handled via one-vs-one decomposition.
#[derive(Debug, Clone)]
pub struct SVC<T: Float> {
    kernel: Kernel<T>,
    c: T,
    tol: T,
    max_iter: usize,
    // Fitted state (binary)
    alphas: Option<Vec<T>>,
    bias: T,
    support_x: Option<Vec<Vec<T>>>,
    support_y: Option<Vec<T>>,
    // Multi-class state
    classes: Option<Vec<T>>,
    binary_models: Option<Vec<BinaryModel<T>>>,
}

/// Stores one binary SVM model (for one-vs-one).
#[derive(Debug, Clone)]
struct BinaryModel<T: Float> {
    class_pos: T,
    class_neg: T,
    alphas: Vec<T>,
    bias: T,
    support_x: Vec<Vec<T>>,
    support_y: Vec<T>,
}

impl<T: Float> SVC<T> {
    /// Create a new SVC.
    ///
    /// - `kernel`: kernel function
    /// - `c`: regularisation parameter (higher = less regularisation)
    pub fn new(kernel: Kernel<T>, c: f64) -> Result<Self> {
        if c <= 0.0 {
            return Err(MlError::InvalidParameter {
                name: "c",
                reason: "must be positive",
            });
        }
        Ok(Self {
            kernel,
            c: T::from_f64(c),
            tol: T::from_f64(1e-3),
            max_iter: 1000,
            alphas: None,
            bias: T::zero(),
            support_x: None,
            support_y: None,
            classes: None,
            binary_models: None,
        })
    }

    /// Set the convergence tolerance.
    pub fn set_tol(&mut self, tol: f64) -> &mut Self {
        self.tol = T::from_f64(tol.max(1e-12));
        self
    }

    /// Set the maximum number of SMO iterations.
    pub fn set_max_iter(&mut self, max_iter: usize) -> &mut Self {
        self.max_iter = max_iter.max(1);
        self
    }

    /// Return the support vectors as a flat vector of feature vectors.
    pub fn support_vectors(&self) -> Result<Vec<&[T]>> {
        if let Some(ref models) = self.binary_models {
            let mut svs = Vec::new();
            for m in models {
                for sv in &m.support_x {
                    svs.push(sv.as_slice());
                }
            }
            Ok(svs)
        } else if let Some(ref sx) = self.support_x {
            Ok(sx.iter().map(Vec::as_slice).collect())
        } else {
            Err(MlError::NotFitted)
        }
    }

    /// Decision function value for binary SVC (distance from hyperplane).
    pub fn decision_function(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let sx = self.support_x.as_ref().ok_or(MlError::NotFitted)?;
        let sy = self.support_y.as_ref().ok_or(MlError::NotFitted)?;
        let alphas = self.alphas.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        let data = x.as_slice();

        let mut out = vec![T::zero(); n];
        for i in 0..n {
            let xi = &data[i * p..(i + 1) * p];
            out[i] = decision_value(&self.kernel, alphas, sx, sy, self.bias, xi);
        }
        Tensor::from_vec(out, vec![n]).map_err(MlError::from)
    }

    fn fit_binary(&mut self, x_data: &[T], y_data: &[T], n: usize, p: usize, classes: &[T]) {
        let y_binary: Vec<T> = y_data
            .iter()
            .map(|&v| {
                if (v - classes[0]).abs() < T::epsilon() {
                    -T::one()
                } else {
                    T::one()
                }
            })
            .collect();

        let x_rows: Vec<Vec<T>> = (0..n)
            .map(|i| x_data[i * p..(i + 1) * p].to_vec())
            .collect();

        let (alphas, bias) = smo_train(
            &self.kernel,
            &x_rows,
            &y_binary,
            self.c,
            self.tol,
            self.max_iter,
        );

        let (sv_alphas, sv_x, sv_y) = extract_support_vectors(&alphas, &x_rows, &y_binary);

        self.alphas = Some(sv_alphas);
        self.bias = bias;
        self.support_x = Some(sv_x);
        self.support_y = Some(sv_y);
        self.binary_models = None;
    }

    fn fit_multiclass(&mut self, x_data: &[T], y_data: &[T], n: usize, p: usize, classes: &[T]) {
        let n_classes = classes.len();
        let mut models = Vec::new();

        for ci in 0..n_classes {
            for cj in (ci + 1)..n_classes {
                let class_pos = classes[ci];
                let class_neg = classes[cj];

                let mut sub_x = Vec::new();
                let mut sub_y = Vec::new();
                for s in 0..n {
                    let label = y_data[s];
                    if (label - class_pos).abs() < T::epsilon() {
                        sub_x.push(x_data[s * p..(s + 1) * p].to_vec());
                        sub_y.push(T::one());
                    } else if (label - class_neg).abs() < T::epsilon() {
                        sub_x.push(x_data[s * p..(s + 1) * p].to_vec());
                        sub_y.push(-T::one());
                    }
                }

                let (alphas, bias) = smo_train(
                    &self.kernel,
                    &sub_x,
                    &sub_y,
                    self.c,
                    self.tol,
                    self.max_iter,
                );
                let (sv_alphas, sv_x, sv_y) = extract_support_vectors(&alphas, &sub_x, &sub_y);

                models.push(BinaryModel {
                    class_pos,
                    class_neg,
                    alphas: sv_alphas,
                    bias,
                    support_x: sv_x,
                    support_y: sv_y,
                });
            }
        }

        self.binary_models = Some(models);
        self.alphas = None;
        self.support_x = None;
        self.support_y = None;
    }
}

impl<T: Float> Predictor<T> for SVC<T> {
    fn fit(&mut self, x: &Tensor<T>, y: &Tensor<T>) -> Result<()> {
        let (n, p) = matrix_shape(x)?;
        check_y(y, n)?;
        let y_data = y.as_slice();

        // Discover classes
        let mut classes: Vec<T> = Vec::new();
        for &v in y_data {
            if !classes.iter().any(|&c| (c - v).abs() < T::epsilon()) {
                classes.push(v);
            }
        }
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if classes.len() < 2 {
            return Err(MlError::InvalidParameter {
                name: "y",
                reason: "must contain at least 2 classes",
            });
        }

        let x_data = x.as_slice();

        if classes.len() == 2 {
            self.fit_binary(x_data, y_data, n, p, &classes);
        } else {
            self.fit_multiclass(x_data, y_data, n, p, &classes);
        }

        self.classes = Some(classes);
        Ok(())
    }

    fn predict(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        let classes = self.classes.as_ref().ok_or(MlError::NotFitted)?;
        let (n, p) = matrix_shape(x)?;
        let data = x.as_slice();

        let mut out = vec![T::zero(); n];

        if classes.len() == 2 {
            // Binary
            let sx = self.support_x.as_ref().ok_or(MlError::NotFitted)?;
            let sy = self.support_y.as_ref().ok_or(MlError::NotFitted)?;
            let alphas = self.alphas.as_ref().ok_or(MlError::NotFitted)?;

            for i in 0..n {
                let xi = &data[i * p..(i + 1) * p];
                let val = decision_value(&self.kernel, alphas, sx, sy, self.bias, xi);
                out[i] = if val >= T::zero() {
                    classes[1]
                } else {
                    classes[0]
                };
            }
        } else {
            // Multi-class: one-vs-one voting
            let models = self.binary_models.as_ref().ok_or(MlError::NotFitted)?;
            let n_classes = classes.len();

            for i in 0..n {
                let xi = &data[i * p..(i + 1) * p];
                let mut votes = vec![0usize; n_classes];

                for m in models {
                    let val = decision_value(
                        &self.kernel,
                        &m.alphas,
                        &m.support_x,
                        &m.support_y,
                        m.bias,
                        xi,
                    );
                    let winner = if val >= T::zero() {
                        m.class_pos
                    } else {
                        m.class_neg
                    };
                    // Find class index
                    if let Some(idx) = classes
                        .iter()
                        .position(|&c| (c - winner).abs() < T::epsilon())
                    {
                        votes[idx] += 1;
                    }
                }

                // Argmax
                let best = votes
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &v)| v)
                    .map_or(0, |(idx, _)| idx);
                out[i] = classes[best];
            }
        }

        Tensor::from_vec(out, vec![n]).map_err(MlError::from)
    }
}

// ── SMO algorithm ──

/// Extract support vectors (alpha > threshold) from training data.
fn extract_support_vectors<T: Float>(
    alphas: &[T],
    x_rows: &[Vec<T>],
    y: &[T],
) -> (Vec<T>, Vec<Vec<T>>, Vec<T>) {
    let mut sv_alphas = Vec::new();
    let mut sv_x = Vec::new();
    let mut sv_y = Vec::new();
    for (i, &alpha) in alphas.iter().enumerate() {
        if alpha > T::from_f64(1e-8) {
            sv_alphas.push(alpha);
            sv_x.push(x_rows[i].clone());
            sv_y.push(y[i]);
        }
    }
    (sv_alphas, sv_x, sv_y)
}

/// Simplified SMO algorithm for binary SVM training.
///
/// Returns (alphas, bias) for all training samples.
fn smo_train<T: Float>(
    kernel: &Kernel<T>,
    x: &[Vec<T>],
    y: &[T],
    c: T,
    tol: T,
    max_iter: usize,
) -> (Vec<T>, T) {
    let n = x.len();
    let mut alphas = vec![T::zero(); n];
    let mut b = T::zero();

    // Pre-compute kernel matrix for efficiency
    let mut k_matrix = vec![T::zero(); n * n];
    for i in 0..n {
        for j in i..n {
            let kij = kernel.compute(&x[i], &x[j]);
            k_matrix[i * n + j] = kij;
            k_matrix[j * n + i] = kij;
        }
    }

    let mut passes = 0;
    let two = T::from_f64(2.0);

    while passes < max_iter {
        let mut num_changed = 0;

        for i in 0..n {
            // Compute error for i
            let ei = svm_output(&alphas, y, &k_matrix, n, b, i) - y[i];

            // Check KKT violation
            if (y[i] * ei < -tol && alphas[i] < c) || (y[i] * ei > tol && alphas[i] > T::zero()) {
                // Select j != i (simple heuristic: max |Ei - Ej|)
                let j = select_j(i, n, &alphas, y, &k_matrix, b);

                let ej = svm_output(&alphas, y, &k_matrix, n, b, j) - y[j];

                let alpha_i_old = alphas[i];
                let alpha_j_old = alphas[j];

                // Compute bounds
                let (lo, hi) = if (y[i] - y[j]).abs() > T::epsilon() {
                    // y[i] != y[j]
                    let lo = (alphas[j] - alphas[i]).max(T::zero());
                    let hi = (c + alphas[j] - alphas[i]).min(c);
                    (lo, hi)
                } else {
                    let lo = (alphas[i] + alphas[j] - c).max(T::zero());
                    let hi = (alphas[i] + alphas[j]).min(c);
                    (lo, hi)
                };

                if (hi - lo).abs() < T::from_f64(1e-12) {
                    continue;
                }

                // Compute eta
                let eta = two * k_matrix[i * n + j] - k_matrix[i * n + i] - k_matrix[j * n + j];
                if eta >= T::zero() {
                    continue;
                }

                // Update alpha_j
                alphas[j] -= y[j] * (ei - ej) / eta;
                alphas[j] = alphas[j].max(lo).min(hi);

                if (alphas[j] - alpha_j_old).abs() < T::from_f64(1e-5) {
                    continue;
                }

                // Update alpha_i
                let alpha_j_new = alphas[j];
                alphas[i] += y[i] * y[j] * (alpha_j_old - alpha_j_new);

                // Update bias
                let b1 = b
                    - ei
                    - y[i] * (alphas[i] - alpha_i_old) * k_matrix[i * n + i]
                    - y[j] * (alphas[j] - alpha_j_old) * k_matrix[i * n + j];
                let b2 = b
                    - ej
                    - y[i] * (alphas[i] - alpha_i_old) * k_matrix[i * n + j]
                    - y[j] * (alphas[j] - alpha_j_old) * k_matrix[j * n + j];

                b = if alphas[i] > T::zero() && alphas[i] < c {
                    b1
                } else if alphas[j] > T::zero() && alphas[j] < c {
                    b2
                } else {
                    (b1 + b2) / two
                };

                num_changed += 1;
            }
        }

        if num_changed == 0 {
            passes += 1;
        } else {
            passes = 0;
        }
    }

    (alphas, b)
}

fn svm_output<T: Float>(alphas: &[T], y: &[T], k_matrix: &[T], n: usize, b: T, i: usize) -> T {
    let mut sum = T::zero();
    for j in 0..n {
        if alphas[j] > T::from_f64(1e-12) {
            sum += alphas[j] * y[j] * k_matrix[j * n + i];
        }
    }
    sum + b
}

fn select_j<T: Float>(i: usize, n: usize, alphas: &[T], y: &[T], k_matrix: &[T], b: T) -> usize {
    let ei = svm_output(alphas, y, k_matrix, n, b, i) - y[i];
    let mut best_j = usize::from(i == 0);
    let mut max_diff = T::zero();

    for j in 0..n {
        if j == i {
            continue;
        }
        let ej = svm_output(alphas, y, k_matrix, n, b, j) - y[j];
        let diff = (ei - ej).abs();
        if diff > max_diff {
            max_diff = diff;
            best_j = j;
        }
    }
    best_j
}

fn decision_value<T: Float>(
    kernel: &Kernel<T>,
    alphas: &[T],
    support_x: &[Vec<T>],
    support_y: &[T],
    bias: T,
    x: &[T],
) -> T {
    let mut sum = T::zero();
    for ((&alpha, sv), &sy) in alphas.iter().zip(support_x).zip(support_y) {
        sum += alpha * sy * kernel.compute(sv, x);
    }
    sum + bias
}

// ── helpers ──

fn matrix_shape<T: Float>(x: &Tensor<T>) -> Result<(usize, usize)> {
    let s = x.shape();
    if s.len() != 2 {
        return Err(MlError::DimensionMismatch {
            expected: 2,
            got: s.len(),
        });
    }
    if s[0] == 0 {
        return Err(MlError::EmptyInput);
    }
    Ok((s[0], s[1]))
}

fn check_y<T: Float>(y: &Tensor<T>, n: usize) -> Result<()> {
    if y.ndim() != 1 || y.shape()[0] != n {
        return Err(MlError::DimensionMismatch {
            expected: n,
            got: y.shape()[0],
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svc_linear_binary() {
        // Linearly separable data
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 1.0, 2.0, 2.0, 1.5, 0.5, 8.0, 8.0, 9.0, 9.0, 8.5, 7.5,
            ],
            vec![6, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();

        let mut svc = SVC::new(Kernel::Linear, 1.0).unwrap();
        svc.fit(&x, &y).unwrap();
        let preds = svc.predict(&x).unwrap();

        let correct: usize = preds
            .as_slice()
            .iter()
            .zip(y.as_slice())
            .filter(|(a, b)| (**a - **b).abs() < 0.5)
            .count();
        assert!(correct >= 5, "expected at least 5/6 correct, got {correct}");
    }

    #[test]
    fn test_svc_rbf_binary() {
        let x = Tensor::from_vec(
            vec![
                0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0,
            ],
            vec![6, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();

        let mut svc = SVC::new(Kernel::Rbf { gamma: 0.1 }, 10.0).unwrap();
        svc.fit(&x, &y).unwrap();
        let preds = svc.predict(&x).unwrap();

        let correct: usize = preds
            .as_slice()
            .iter()
            .zip(y.as_slice())
            .filter(|(a, b)| (**a - **b).abs() < 0.5)
            .count();
        assert!(correct >= 5, "expected at least 5/6 correct, got {correct}");
    }

    #[test]
    fn test_svc_multiclass() {
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 0.0, 1.5, 0.0, // class 0
                0.0, 5.0, 0.0, 5.5, // class 1
                5.0, 5.0, 5.5, 5.5, // class 2
            ],
            vec![6, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], vec![6]).unwrap();

        let mut svc = SVC::new(Kernel::Rbf { gamma: 0.5 }, 10.0).unwrap();
        svc.fit(&x, &y).unwrap();
        let preds = svc.predict(&x).unwrap();

        let correct: usize = preds
            .as_slice()
            .iter()
            .zip(y.as_slice())
            .filter(|(a, b)| (**a - **b).abs() < 0.5)
            .count();
        assert!(
            correct >= 4,
            "expected at least 4/6 correct in multiclass, got {correct}"
        );
    }

    #[test]
    fn test_svc_decision_function() {
        let x =
            Tensor::from_vec(vec![1.0_f64, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0], vec![4, 2]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();

        let mut svc = SVC::new(Kernel::Linear, 1.0).unwrap();
        svc.fit(&x, &y).unwrap();

        let df = svc.decision_function(&x).unwrap();
        assert_eq!(df.shape(), &[4]);
        // Class 0 points should have negative decision values, class 1 positive
        // Class 0 should have negative values, class 1 positive (or vice versa)
        let vals = df.as_slice();
        let signs_differ = (vals[0] < 0.0 && vals[2] > 0.0) || (vals[0] > 0.0 && vals[2] < 0.0);
        assert!(signs_differ, "decision values should separate classes");
    }

    #[test]
    fn test_svc_not_fitted() {
        let svc = SVC::<f64>::new(Kernel::Linear, 1.0).unwrap();
        let x = Tensor::from_vec(vec![1.0_f64, 2.0], vec![1, 2]).unwrap();
        assert!(svc.predict(&x).is_err());
    }

    #[test]
    fn test_svc_invalid_c() {
        assert!(SVC::<f64>::new(Kernel::Linear, 0.0).is_err());
        assert!(SVC::<f64>::new(Kernel::Linear, -1.0).is_err());
    }

    #[test]
    fn test_svc_support_vectors() {
        let x =
            Tensor::from_vec(vec![1.0_f64, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0], vec![4, 2]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();

        let mut svc = SVC::new(Kernel::Linear, 1.0).unwrap();
        svc.fit(&x, &y).unwrap();

        let svs = svc.support_vectors().unwrap();
        assert!(!svs.is_empty(), "should have at least one support vector");
    }

    #[test]
    fn test_svc_poly_kernel() {
        let x = Tensor::from_vec(
            vec![
                1.0_f64, 1.0, 2.0, 2.0, 1.5, 1.5, 8.0, 8.0, 9.0, 9.0, 8.5, 8.5,
            ],
            vec![6, 2],
        )
        .unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6]).unwrap();

        let mut svc = SVC::new(
            Kernel::Poly {
                degree: 2,
                gamma: 0.01,
                coef0: 1.0,
            },
            10.0,
        )
        .unwrap();
        svc.fit(&x, &y).unwrap();
        let preds = svc.predict(&x).unwrap();

        let correct: usize = preds
            .as_slice()
            .iter()
            .zip(y.as_slice())
            .filter(|(a, b)| (**a - **b).abs() < 0.5)
            .count();
        assert!(correct >= 4, "expected at least 4/6 correct, got {correct}");
    }
}
