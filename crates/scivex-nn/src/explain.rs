//! Model interpretability: Integrated Gradients and SmoothGrad.
//!
//! These attribution methods help explain which input features are most
//! important for a model's prediction.

use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::variable::Variable;

/// Result of the Integrated Gradients attribution method.
#[derive(Debug, Clone)]
pub struct IntegratedGradientsResult<T: Float> {
    /// Attribution scores with the same shape as the input.
    pub attributions: Tensor<T>,
    /// Completeness check: `sum(attributions) - (f(input) - f(baseline))` at
    /// the target index. Should be close to zero for a good approximation.
    pub convergence_delta: T,
}

/// Compute Integrated Gradients attributions.
///
/// Given a model function `model`, an `input` tensor, a `baseline` tensor (same
/// shape as `input`), a `target_idx` (index into the model's output to
/// attribute), and `n_steps` interpolation steps, this computes the path
/// integral of gradients from baseline to input.
///
/// # Algorithm
///
/// 1. Generate `n_steps + 1` interpolated inputs:
///    `x_k = baseline + (k / n_steps) * (input - baseline)` for k = 0..=n_steps
/// 2. For each interpolated input, compute the gradient of `output[target_idx]`
///    w.r.t. the input.
/// 3. Average the gradients (trapezoidal rule).
/// 4. Multiply by `(input - baseline)` to get attributions.
///
/// # Errors
///
/// Returns [`NnError::ShapeMismatch`] if `input` and `baseline` have different
/// shapes, or [`NnError::InvalidParameter`] if `n_steps` is zero.
pub fn integrated_gradients<T, F>(
    model: &F,
    input: &Tensor<T>,
    baseline: &Tensor<T>,
    target_idx: usize,
    n_steps: usize,
) -> Result<IntegratedGradientsResult<T>>
where
    T: Float,
    F: Fn(&Variable<T>) -> Result<Variable<T>>,
{
    // Validate shapes match.
    if input.shape() != baseline.shape() {
        return Err(NnError::ShapeMismatch {
            expected: input.shape().to_vec(),
            got: baseline.shape().to_vec(),
        });
    }

    if n_steps == 0 {
        return Err(NnError::InvalidParameter {
            name: "n_steps",
            reason: "must be at least 1",
        });
    }

    let diff = input
        .zip_map(baseline, |a, b| a - b)
        .map_err(NnError::from)?;

    let n = input.numel();
    let mut accumulated = vec![T::zero(); n];

    for k in 0..=n_steps {
        let alpha = T::from_usize(k) / T::from_usize(n_steps);

        // x_k = baseline + alpha * diff
        let interp_data: Vec<T> = baseline
            .as_slice()
            .iter()
            .zip(diff.as_slice().iter())
            .map(|(&b, &d)| b + alpha * d)
            .collect();
        let interp =
            Tensor::from_vec(interp_data, input.shape().to_vec()).map_err(NnError::from)?;

        let grad = compute_gradient_at_target(model, &interp, target_idx)?;
        let grad_slice = grad.as_slice();

        for (acc, &g) in accumulated.iter_mut().zip(grad_slice.iter()) {
            *acc += g;
        }
    }

    // Average the accumulated gradients. We have (n_steps + 1) samples; using
    // the simple average (rectangle rule) which is the standard IG formulation.
    let divisor = T::from_usize(n_steps + 1);
    for v in &mut accumulated {
        *v /= divisor;
    }

    // attributions = avg_grads * (input - baseline)
    let attr_data: Vec<T> = accumulated
        .iter()
        .zip(diff.as_slice().iter())
        .map(|(&g, &d)| g * d)
        .collect();
    let attributions =
        Tensor::from_vec(attr_data, input.shape().to_vec()).map_err(NnError::from)?;

    // Convergence delta: sum(attributions) - (f(input)[target_idx] - f(baseline)[target_idx])
    let f_input = {
        let var = Variable::new(input.clone(), false);
        let out = model(&var)?;
        let out_data = out.data();
        let out_slice = out_data.as_slice();
        if target_idx >= out_slice.len() {
            return Err(NnError::IndexOutOfBounds {
                index: target_idx,
                len: out_slice.len(),
            });
        }
        out_slice[target_idx]
    };

    let f_baseline = {
        let var = Variable::new(baseline.clone(), false);
        let out = model(&var)?;
        out.data().as_slice()[target_idx]
    };

    let attr_sum = attributions.sum();
    let convergence_delta = attr_sum - (f_input - f_baseline);

    Ok(IntegratedGradientsResult {
        attributions,
        convergence_delta,
    })
}

/// Convenience wrapper that uses a zero baseline.
///
/// Equivalent to calling [`integrated_gradients`] with
/// `baseline = Tensor::zeros(input.shape())`.
pub fn integrated_gradients_zero_baseline<T, F>(
    model: &F,
    input: &Tensor<T>,
    target_idx: usize,
    n_steps: usize,
) -> Result<IntegratedGradientsResult<T>>
where
    T: Float,
    F: Fn(&Variable<T>) -> Result<Variable<T>>,
{
    let baseline = Tensor::zeros(input.shape().to_vec());
    integrated_gradients(model, input, &baseline, target_idx, n_steps)
}

/// SmoothGrad: average gradients over noisy copies of the input.
///
/// Adds Gaussian noise with standard deviation `noise_std` to `n_samples`
/// copies of the input, computes the gradient of `output[target_idx]` w.r.t.
/// each noisy input, and returns the averaged gradient.
///
/// Uses a simple xorshift-based PRNG seeded with `seed` to generate noise, so
/// results are reproducible.
///
/// # Errors
///
/// Returns [`NnError::InvalidParameter`] if `n_samples` is zero.
pub fn smooth_gradients<T, F>(
    model: &F,
    input: &Tensor<T>,
    target_idx: usize,
    n_samples: usize,
    noise_std: T,
    seed: u64,
) -> Result<Tensor<T>>
where
    T: Float,
    F: Fn(&Variable<T>) -> Result<Variable<T>>,
{
    if n_samples == 0 {
        return Err(NnError::InvalidParameter {
            name: "n_samples",
            reason: "must be at least 1",
        });
    }

    let n = input.numel();
    let mut accumulated = vec![T::zero(); n];
    let mut rng_state = seed;

    for _ in 0..n_samples {
        // Generate noisy input.
        let noisy_data: Vec<T> = input
            .as_slice()
            .iter()
            .map(|&v| {
                let noise = gaussian_noise(&mut rng_state) * noise_std.to_f64();
                v + T::from_f64(noise)
            })
            .collect();
        let noisy = Tensor::from_vec(noisy_data, input.shape().to_vec()).map_err(NnError::from)?;

        let grad = compute_gradient_at_target(model, &noisy, target_idx)?;
        let grad_slice = grad.as_slice();

        for (acc, &g) in accumulated.iter_mut().zip(grad_slice.iter()) {
            *acc += g;
        }
    }

    let divisor = T::from_usize(n_samples);
    for v in &mut accumulated {
        *v /= divisor;
    }

    Tensor::from_vec(accumulated, input.shape().to_vec()).map_err(NnError::from)
}

// ── Internal helpers ────────────────────────────────────────────────────

/// Compute the gradient of `model(input)[target_idx]` w.r.t. the input.
///
/// Creates a differentiable selection of the scalar at `target_idx` from the
/// model output and backpropagates through it.
fn compute_gradient_at_target<T, F>(
    model: &F,
    input: &Tensor<T>,
    target_idx: usize,
) -> Result<Tensor<T>>
where
    T: Float,
    F: Fn(&Variable<T>) -> Result<Variable<T>>,
{
    let input_var = Variable::new(input.clone(), true);
    let output = model(&input_var)?;

    let out_data = output.data();
    let out_len = out_data.numel();

    if target_idx >= out_len {
        return Err(NnError::IndexOutOfBounds {
            index: target_idx,
            len: out_len,
        });
    }

    // Extract the scalar at target_idx via a dot product with a one-hot vector.
    // scalar = sum(output * one_hot)  =>  d(scalar)/d(output) = one_hot
    let mut one_hot_data = vec![T::zero(); out_len];
    one_hot_data[target_idx] = T::one();
    let one_hot =
        Tensor::from_vec(one_hot_data, out_data.shape().to_vec()).map_err(NnError::from)?;
    let one_hot_var = Variable::new(one_hot, false);

    // Element-wise multiply then sum => scalar
    let masked = crate::ops::mul(&output, &one_hot_var);
    let scalar = crate::ops::sum(&masked);

    scalar.backward();

    input_var.grad().ok_or(NnError::NoGradient)
}

/// Simple xorshift64-based PRNG returning a u64.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a Gaussian-distributed f64 using the Box-Muller transform.
fn gaussian_noise(state: &mut u64) -> f64 {
    // Generate two uniform random numbers in (0, 1).
    let u1 = (xorshift64(state) as f64) / (u64::MAX as f64);
    let u2 = (xorshift64(state) as f64) / (u64::MAX as f64);
    // Clamp to avoid log(0).
    let u1 = if u1 < 1e-15 { 1e-15 } else { u1 };
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_core::Tensor;

    /// Linear model: f(x) = w * x (element-wise), output = [sum(w*x)].
    /// For IG, attributions should equal w * (input - baseline) exactly.
    #[test]
    fn test_integrated_gradients_linear() {
        let weights = vec![2.0_f64, 3.0, -1.0];
        let w_tensor = Tensor::from_vec(weights.clone(), vec![3]).unwrap();

        let model = |x: &Variable<f64>| -> Result<Variable<f64>> {
            let w = Variable::new(w_tensor.clone(), false);
            let prod = crate::ops::mul(x, &w);
            // Output is a 3-element vector; we'll attribute to index 0, 1, or 2.
            // Actually let's return the element-wise product as the "output".
            Ok(prod)
        };

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let baseline = Tensor::zeros(vec![3]);

        // Attribute to target_idx=0: output[0] = w[0]*x[0] = 2*x[0]
        // IG attribution[0] should be w[0]*(input[0]-baseline[0]) = 2*1 = 2
        // IG attribution[1] should be 0 (output[0] doesn't depend on x[1])
        // IG attribution[2] should be 0
        let result = integrated_gradients(&model, &input, &baseline, 0, 100).unwrap();
        let attr = result.attributions.as_slice();

        assert!(
            (attr[0] - 2.0).abs() < 1e-6,
            "attr[0] = {}, expected 2.0",
            attr[0]
        );
        assert!(attr[1].abs() < 1e-6, "attr[1] = {}, expected 0.0", attr[1]);
        assert!(attr[2].abs() < 1e-6, "attr[2] = {}, expected 0.0", attr[2]);
    }

    #[test]
    fn test_integrated_gradients_zero_baseline() {
        let w_tensor = Tensor::from_vec(vec![1.0_f64, -1.0], vec![2]).unwrap();

        let model = |x: &Variable<f64>| -> Result<Variable<f64>> {
            let w = Variable::new(w_tensor.clone(), false);
            let prod = crate::ops::mul(x, &w);
            Ok(prod)
        };

        let input = Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap();

        let result = integrated_gradients_zero_baseline(&model, &input, 0, 50).unwrap();

        // output[0] = 1.0 * x[0], so attribution[0] = 1.0 * 3.0 = 3.0
        let attr = result.attributions.as_slice();
        assert!(
            (attr[0] - 3.0).abs() < 1e-6,
            "attr[0] = {}, expected 3.0",
            attr[0]
        );
    }

    #[test]
    fn test_convergence_delta_small() {
        // For a linear model, the convergence delta should be exactly 0 (or very
        // close, limited only by floating-point precision).
        let w_tensor = Tensor::from_vec(vec![2.0_f64, -3.0, 0.5], vec![3]).unwrap();

        let model = |x: &Variable<f64>| -> Result<Variable<f64>> {
            let w = Variable::new(w_tensor.clone(), false);
            let prod = crate::ops::mul(x, &w);
            Ok(prod)
        };

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let baseline = Tensor::from_vec(vec![0.5, 0.5, 0.5], vec![3]).unwrap();

        let result = integrated_gradients(&model, &input, &baseline, 1, 200).unwrap();
        assert!(
            result.convergence_delta.to_f64().abs() < 1e-6,
            "convergence_delta = {}, expected ~0",
            result.convergence_delta.to_f64()
        );
    }

    #[test]
    fn test_smooth_gradients_shape() {
        let w_tensor = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap();

        let model = |x: &Variable<f64>| -> Result<Variable<f64>> {
            let w = Variable::new(w_tensor.clone(), false);
            let prod = crate::ops::mul(x, &w);
            Ok(prod)
        };

        let input = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();

        let result = smooth_gradients(&model, &input, 0, 10, 0.1, 42).unwrap();
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_integrated_gradients_input_validation() {
        let model = |x: &Variable<f64>| -> Result<Variable<f64>> { Ok(x.clone()) };

        let input = Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap();
        let bad_baseline = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();

        let result = integrated_gradients(&model, &input, &bad_baseline, 0, 10);
        assert!(result.is_err());

        match result.unwrap_err() {
            NnError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![2]);
                assert_eq!(got, vec![3]);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }
}
