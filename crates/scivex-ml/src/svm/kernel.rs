use scivex_core::Float;

/// Kernel function for SVM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel<T: Float> {
    /// Linear kernel: K(x, y) = x · y
    Linear,
    /// Radial Basis Function: K(x, y) = exp(-gamma * ||x - y||²)
    Rbf { gamma: T },
    /// Polynomial: K(x, y) = (gamma * x · y + coef0) ^ degree
    Poly { degree: u32, gamma: T, coef0: T },
    /// Sigmoid (tanh): K(x, y) = tanh(gamma * x · y + coef0)
    Sigmoid { gamma: T, coef0: T },
}

impl<T: Float> Kernel<T> {
    /// Compute the kernel value between two feature vectors.
    pub fn compute(&self, x: &[T], y: &[T]) -> T {
        match *self {
            Self::Linear => dot(x, y),
            Self::Rbf { gamma } => {
                let sq_dist = x
                    .iter()
                    .zip(y)
                    .map(|(&a, &b)| {
                        let d = a - b;
                        d * d
                    })
                    .fold(T::zero(), |a, b| a + b);
                (-gamma * sq_dist).exp()
            }
            Self::Poly {
                degree,
                gamma,
                coef0,
            } => {
                let d = gamma * dot(x, y) + coef0;
                let mut result = T::one();
                for _ in 0..degree {
                    result *= d;
                }
                result
            }
            Self::Sigmoid { gamma, coef0 } => {
                let z = gamma * dot(x, y) + coef0;
                let e2z = (z + z).exp();
                (e2z - T::one()) / (e2z + T::one())
            }
        }
    }

    /// Create an RBF kernel with gamma = 1 / n_features.
    pub fn rbf_auto(n_features: usize) -> Self {
        Self::Rbf {
            gamma: T::one() / T::from_usize(n_features),
        }
    }

    /// Create a polynomial kernel with common defaults.
    pub fn poly(degree: u32, n_features: usize) -> Self {
        Self::Poly {
            degree,
            gamma: T::one() / T::from_usize(n_features),
            coef0: T::zero(),
        }
    }
}

fn dot<T: Float>(x: &[T], y: &[T]) -> T {
    x.iter().zip(y).fold(T::zero(), |acc, (&a, &b)| acc + a * b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_kernel() {
        let x = [1.0_f64, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let k = Kernel::Linear;
        // 1*4 + 2*5 + 3*6 = 32
        assert!((k.compute(&x, &y) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_kernel() {
        let x = [1.0_f64, 0.0];
        let y = [0.0, 1.0];
        let k = Kernel::Rbf { gamma: 1.0 };
        // ||x-y||^2 = 2, exp(-2)
        let expected = (-2.0_f64).exp();
        assert!((k.compute(&x, &y) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_same_point() {
        let x = [1.0_f64, 2.0, 3.0];
        let k = Kernel::Rbf { gamma: 0.5 };
        assert!((k.compute(&x, &x) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_poly_kernel() {
        let x = [1.0_f64, 2.0];
        let y = [3.0, 4.0];
        let k = Kernel::Poly {
            degree: 2,
            gamma: 1.0,
            coef0: 1.0,
        };
        // (1*3 + 2*4 + 1)^2 = 12^2 = 144
        assert!((k.compute(&x, &y) - 144.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel() {
        let x = [1.0_f64, 0.0];
        let y = [0.0, 1.0];
        let k = Kernel::Sigmoid {
            gamma: 1.0,
            coef0: 0.0,
        };
        // tanh(0) = 0
        assert!(k.compute(&x, &y).abs() < 1e-10);
    }
}
