//! Pooling layers (MaxPool, AvgPool for 1-D and 2-D).

use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::variable::Variable;

use super::Layer;

/// Given an output position, stride, kernel offset, and padding, compute the
/// input coordinate — returning `None` when it falls outside `[0, dim)`.
///
/// `raw = pos * stride + ki` must satisfy `raw >= pad && raw - pad < dim`.
fn pool_index(pos: usize, stride: usize, ki: usize, pad: usize, dim: usize) -> Option<usize> {
    let raw = pos * stride + ki;
    if raw >= pad && raw - pad < dim {
        Some(raw - pad)
    } else {
        None
    }
}

// ── MaxPool2d ─────────────────────────────────────────────────────────────

/// 2-D max pooling layer.
///
/// Input: `[batch, channels, height, width]`
/// Output: `[batch, channels, h_out, w_out]`
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct MaxPool2d {
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
}

impl MaxPool2d {
    /// Create a new MaxPool2d with given kernel size. Stride defaults to kernel size.
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_h: kernel_size.0,
            kernel_w: kernel_size.1,
            stride_h: kernel_size.0,
            stride_w: kernel_size.1,
            padding_h: 0,
            padding_w: 0,
        }
    }

    /// Set stride.
    pub fn set_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride_h = stride.0.max(1);
        self.stride_w = stride.1.max(1);
        self
    }

    /// Set padding.
    pub fn set_padding(mut self, padding: (usize, usize)) -> Self {
        self.padding_h = padding.0;
        self.padding_w = padding.1;
        self
    }

    fn output_size(&self, h: usize, w: usize) -> (usize, usize) {
        let ho = (h + 2 * self.padding_h - self.kernel_h) / self.stride_h + 1;
        let wo = (w + 2 * self.padding_w - self.kernel_w) / self.stride_w + 1;
        (ho, wo)
    }
}

impl<T: Float> Layer<T> for MaxPool2d {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: shape,
            });
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (ho, wo) = self.output_size(h, w);
        let data = x.data();
        let d = data.as_slice();
        let sh = self.stride_h;
        let sw = self.stride_w;
        let ph = self.padding_h;
        let pw = self.padding_w;
        let kh = self.kernel_h;
        let kw = self.kernel_w;

        let mut out = vec![T::zero(); n * c * ho * wo];
        let mut indices = vec![0usize; n * c * ho * wo]; // argmax for backward

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..ho {
                    for ow in 0..wo {
                        let out_idx = batch * c * ho * wo + ch * ho * wo + oh * wo + ow;
                        let mut max_val = T::neg_infinity();
                        let mut max_idx = 0;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                if let (Some(ih), Some(iw)) =
                                    (pool_index(oh, sh, ki, ph, h), pool_index(ow, sw, kj, pw, w))
                                {
                                    let in_idx = batch * c * h * w + ch * h * w + ih * w + iw;
                                    if d[in_idx] > max_val {
                                        max_val = d[in_idx];
                                        max_idx = in_idx;
                                    }
                                }
                            }
                        }
                        out[out_idx] = max_val;
                        indices[out_idx] = max_idx;
                    }
                }
            }
        }

        // SAFETY: out length == n * c * ho * wo by construction
        let out_tensor = Tensor::from_vec(out, vec![n, c, ho, wo]).expect("valid maxpool shape");
        let input_numel = n * c * h * w;
        let grad_fn = Box::new(move |g: &Tensor<T>| {
            let gd = g.as_slice();
            let mut gx = vec![T::zero(); input_numel];
            for (i, &idx) in indices.iter().enumerate() {
                gx[idx] += gd[i];
            }
            vec![Tensor::from_vec(gx, vec![n, c, h, w]).expect("valid shape")]
        });

        Ok(Variable::from_op(out_tensor, vec![x.clone()], grad_fn))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }
    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// ── AvgPool2d ─────────────────────────────────────────────────────────────

/// 2-D average pooling layer.
///
/// Input: `[batch, channels, height, width]`
/// Output: `[batch, channels, h_out, w_out]`
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct AvgPool2d {
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
}

impl AvgPool2d {
    /// Create a new AvgPool2d. Stride defaults to kernel size.
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_h: kernel_size.0,
            kernel_w: kernel_size.1,
            stride_h: kernel_size.0,
            stride_w: kernel_size.1,
            padding_h: 0,
            padding_w: 0,
        }
    }

    /// Set stride.
    pub fn set_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride_h = stride.0.max(1);
        self.stride_w = stride.1.max(1);
        self
    }

    /// Set padding.
    pub fn set_padding(mut self, padding: (usize, usize)) -> Self {
        self.padding_h = padding.0;
        self.padding_w = padding.1;
        self
    }

    fn output_size(&self, h: usize, w: usize) -> (usize, usize) {
        let ho = (h + 2 * self.padding_h - self.kernel_h) / self.stride_h + 1;
        let wo = (w + 2 * self.padding_w - self.kernel_w) / self.stride_w + 1;
        (ho, wo)
    }
}

impl<T: Float> Layer<T> for AvgPool2d {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: shape,
            });
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (ho, wo) = self.output_size(h, w);
        let data = x.data();
        let d = data.as_slice();
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let sh = self.stride_h;
        let sw = self.stride_w;
        let ph = self.padding_h;
        let pw = self.padding_w;

        let mut out = vec![T::zero(); n * c * ho * wo];

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..ho {
                    for ow in 0..wo {
                        let out_idx = batch * c * ho * wo + ch * ho * wo + oh * wo + ow;
                        let mut sum = T::zero();
                        let mut count = T::zero();
                        for ki in 0..kh {
                            for kj in 0..kw {
                                if let (Some(ih), Some(iw)) =
                                    (pool_index(oh, sh, ki, ph, h), pool_index(ow, sw, kj, pw, w))
                                {
                                    let in_idx = batch * c * h * w + ch * h * w + ih * w + iw;
                                    sum += d[in_idx];
                                    count += T::one();
                                }
                            }
                        }
                        if count > T::zero() {
                            out[out_idx] = sum / count;
                        }
                    }
                }
            }
        }

        // SAFETY: out length == n * c * ho * wo by construction
        let out_tensor = Tensor::from_vec(out, vec![n, c, ho, wo]).expect("valid avgpool shape");
        let input_numel = n * c * h * w;
        let grad_fn = Box::new(move |g: &Tensor<T>| {
            let gd = g.as_slice();
            let mut gx = vec![T::zero(); input_numel];
            for batch in 0..n {
                for ch in 0..c {
                    for oh in 0..ho {
                        for ow in 0..wo {
                            let g_idx = batch * c * ho * wo + ch * ho * wo + oh * wo + ow;
                            let mut count = T::zero();
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    if pool_index(oh, sh, ki, ph, h).is_some()
                                        && pool_index(ow, sw, kj, pw, w).is_some()
                                    {
                                        count += T::one();
                                    }
                                }
                            }
                            if count > T::zero() {
                                let scale = gd[g_idx] / count;
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        if let (Some(ih), Some(iw)) = (
                                            pool_index(oh, sh, ki, ph, h),
                                            pool_index(ow, sw, kj, pw, w),
                                        ) {
                                            let idx = batch * c * h * w + ch * h * w + ih * w + iw;
                                            gx[idx] += scale;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            vec![Tensor::from_vec(gx, vec![n, c, h, w]).expect("valid shape")]
        });

        Ok(Variable::from_op(out_tensor, vec![x.clone()], grad_fn))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }
    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// ── MaxPool1d ─────────────────────────────────────────────────────────────

/// 1-D max pooling. Input: `[batch, channels, length]`.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct MaxPool1d {
    inner: MaxPool2d,
}

impl MaxPool1d {
    /// Create with given kernel size. Stride defaults to kernel size.
    pub fn new(kernel_size: usize) -> Self {
        Self {
            inner: MaxPool2d::new((1, kernel_size)),
        }
    }

    /// Set stride.
    pub fn set_stride(mut self, stride: usize) -> Self {
        self.inner = self.inner.set_stride((1, stride));
        self
    }
}

impl<T: Float> Layer<T> for MaxPool1d {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, 0, 0],
                got: shape,
            });
        }
        let (n, c, l) = (shape[0], shape[1], shape[2]);
        let x4d = Variable::new(
            Tensor::from_vec(x.data().as_slice().to_vec(), vec![n, c, 1, l])
                .map_err(NnError::from)?,
            x.requires_grad(),
        );
        let out4d = self.inner.forward(&x4d)?;
        let os = out4d.shape();
        let result = Tensor::from_vec(out4d.data().as_slice().to_vec(), vec![os[0], os[1], os[3]])
            .map_err(NnError::from)?;
        Ok(Variable::from_op(
            result,
            vec![out4d],
            Box::new(move |g: &Tensor<T>| {
                let gs = g.shape();
                vec![
                    Tensor::from_vec(g.as_slice().to_vec(), vec![gs[0], gs[1], 1, gs[2]])
                        .expect("valid reshape"),
                ]
            }),
        ))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }
    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// ── AvgPool1d ─────────────────────────────────────────────────────────────

/// 1-D average pooling. Input: `[batch, channels, length]`.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct AvgPool1d {
    inner: AvgPool2d,
}

impl AvgPool1d {
    /// Create with given kernel size. Stride defaults to kernel size.
    pub fn new(kernel_size: usize) -> Self {
        Self {
            inner: AvgPool2d::new((1, kernel_size)),
        }
    }

    /// Set stride.
    pub fn set_stride(mut self, stride: usize) -> Self {
        self.inner = self.inner.set_stride((1, stride));
        self
    }
}

impl<T: Float> Layer<T> for AvgPool1d {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, 0, 0],
                got: shape,
            });
        }
        let (n, c, l) = (shape[0], shape[1], shape[2]);
        let x4d = Variable::new(
            Tensor::from_vec(x.data().as_slice().to_vec(), vec![n, c, 1, l])
                .map_err(NnError::from)?,
            x.requires_grad(),
        );
        let out4d = self.inner.forward(&x4d)?;
        let os = out4d.shape();
        let result = Tensor::from_vec(out4d.data().as_slice().to_vec(), vec![os[0], os[1], os[3]])
            .map_err(NnError::from)?;
        Ok(Variable::from_op(
            result,
            vec![out4d],
            Box::new(move |g: &Tensor<T>| {
                let gs = g.shape();
                vec![
                    Tensor::from_vec(g.as_slice().to_vec(), vec![gs[0], gs[1], 1, gs[2]])
                        .expect("valid reshape"),
                ]
            }),
        ))
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }
    fn train(&mut self) {}
    fn eval(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxpool2d_output_shape() {
        let pool = MaxPool2d::new((2, 2));
        let x = Variable::new(Tensor::ones(vec![1, 1, 4, 4]), true);
        let y: Variable<f64> = pool.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_maxpool2d_values() {
        let pool = MaxPool2d::new((2, 2));
        // [1, 1, 2, 2] = [[1, 2], [3, 4]]
        let x = Variable::new(
            Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]).unwrap(),
            true,
        );
        let y = pool.forward(&x).unwrap();
        assert_eq!(y.data().as_slice(), &[4.0]);
    }

    #[test]
    fn test_maxpool2d_backward() {
        let pool = MaxPool2d::new((2, 2));
        let x = Variable::new(
            Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]).unwrap(),
            true,
        );
        let y: Variable<f64> = pool.forward(&x).unwrap();
        let loss = crate::ops::sum(&y);
        loss.backward();
        let gx = x.grad().unwrap();
        // Gradient flows only to the max element (index 3 = value 4)
        assert_eq!(gx.as_slice(), &[0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_avgpool2d_output_shape() {
        let pool = AvgPool2d::new((2, 2));
        let x = Variable::new(Tensor::ones(vec![2, 3, 6, 6]), true);
        let y: Variable<f64> = pool.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 3, 3, 3]);
    }

    #[test]
    fn test_avgpool2d_values() {
        let pool = AvgPool2d::new((2, 2));
        let x = Variable::new(
            Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]).unwrap(),
            true,
        );
        let y: Variable<f64> = pool.forward(&x).unwrap();
        // avg(1, 2, 3, 4) = 2.5
        assert!((y.data().as_slice()[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_maxpool1d_output_shape() {
        let pool = MaxPool1d::new(2);
        let x = Variable::new(Tensor::ones(vec![1, 3, 8]), true);
        let y: Variable<f64> = pool.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 3, 4]);
    }

    #[test]
    fn test_avgpool1d_output_shape() {
        let pool = AvgPool1d::new(3).set_stride(1);
        let x = Variable::new(Tensor::ones(vec![2, 1, 10]), true);
        let y: Variable<f64> = pool.forward(&x).unwrap();
        // (10 - 3) / 1 + 1 = 8
        assert_eq!(y.shape(), vec![2, 1, 8]);
    }
}
