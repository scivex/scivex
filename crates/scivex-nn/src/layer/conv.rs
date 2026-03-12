//! Convolutional layers (Conv1d, Conv2d) using im2col.

use scivex_core::random::Rng;
use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::init;
use crate::variable::Variable;

use super::Layer;

// ── Shared im2col parameters ──────────────────────────────────────────────

/// Bundles convolution geometry so we don't pass 13 arguments everywhere.
#[derive(Clone, Copy)]
struct ConvShape {
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
}

impl ConvShape {
    fn h_out(&self) -> usize {
        (self.h + 2 * self.ph - self.dh * (self.kh - 1) - 1) / self.sh + 1
    }
    fn w_out(&self) -> usize {
        (self.w + 2 * self.pw - self.dw * (self.kw - 1) - 1) / self.sw + 1
    }
    fn col_rows(&self) -> usize {
        self.n * self.h_out() * self.w_out()
    }
    fn col_cols(&self) -> usize {
        self.c_in * self.kh * self.kw
    }
}

/// Check whether a signed index is in bounds.
fn in_bounds(
    pos: usize,
    stride: usize,
    ki: usize,
    dil: usize,
    pad: usize,
    dim: usize,
) -> Option<usize> {
    let raw = pos * stride + ki * dil;
    if raw >= pad && raw - pad < dim {
        Some(raw - pad)
    } else {
        None
    }
}

// ── Conv2d ────────────────────────────────────────────────────────────────

/// 2-D convolution layer.
///
/// Input shape: `[batch, in_channels, height, width]`
/// Output shape: `[batch, out_channels, h_out, w_out]`
///
/// Uses im2col internally: patches are unrolled into a matrix, then
/// a single matrix multiplication produces the convolution output.
pub struct Conv2d<T: Float> {
    weight: Variable<T>,       // [out_ch, in_ch, kh, kw]
    bias: Option<Variable<T>>, // [out_ch]
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
}

impl<T: Float> Conv2d<T> {
    /// Create a new Conv2d layer.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        use_bias: bool,
        rng: &mut Rng,
    ) -> Self {
        let (kh, kw) = kernel_size;
        let fan_in = in_channels * kh * kw;
        let w_data = init::kaiming_uniform::<T>(&[out_channels, fan_in], rng);
        let weight = Variable::new(
            Tensor::from_vec(
                w_data.as_slice().to_vec(),
                vec![out_channels, in_channels, kh, kw],
            )
            .expect("valid shape"),
            true,
        );
        let bias = if use_bias {
            Some(Variable::new(Tensor::zeros(vec![out_channels]), true))
        } else {
            None
        };
        Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_h: kh,
            kernel_w: kw,
            stride_h: 1,
            stride_w: 1,
            padding_h: 0,
            padding_w: 0,
            dilation_h: 1,
            dilation_w: 1,
        }
    }

    /// Set stride (height, width).
    pub fn set_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride_h = stride.0.max(1);
        self.stride_w = stride.1.max(1);
        self
    }

    /// Set padding (height, width).
    pub fn set_padding(mut self, padding: (usize, usize)) -> Self {
        self.padding_h = padding.0;
        self.padding_w = padding.1;
        self
    }

    /// Set dilation (height, width).
    pub fn set_dilation(mut self, dilation: (usize, usize)) -> Self {
        self.dilation_h = dilation.0.max(1);
        self.dilation_w = dilation.1.max(1);
        self
    }

    fn conv_shape(&self, n: usize, h: usize, w: usize) -> ConvShape {
        ConvShape {
            n,
            c_in: self.in_channels,
            h,
            w,
            kh: self.kernel_h,
            kw: self.kernel_w,
            sh: self.stride_h,
            sw: self.stride_w,
            ph: self.padding_h,
            pw: self.padding_w,
            dh: self.dilation_h,
            dw: self.dilation_w,
        }
    }

    /// Compute forward output and build gradient closure.
    fn forward_impl(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
        if shape.len() != 4 || shape[1] != self.in_channels {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, self.in_channels, 0, 0],
                got: shape,
            });
        }
        let (n, h, w) = (shape[0], shape[2], shape[3]);
        let cs = self.conv_shape(n, h, w);
        let (h_out, w_out) = (cs.h_out(), cs.w_out());
        let col_rows = cs.col_rows();
        let col_cols = cs.col_cols();
        let out_ch = self.out_channels;

        let col = im2col(x.data().as_slice(), &cs);
        let w_flat = self.weight.data();
        let w_slice = w_flat.as_slice();

        // col @ W^T → [col_rows, out_ch], then add bias, reorder to NCHW
        let out_tensor = compute_forward(
            &col,
            w_slice,
            self.bias.as_ref(),
            n,
            out_ch,
            h_out,
            w_out,
            col_rows,
            col_cols,
        );

        let col_captured = col;
        let w_captured = w_slice.to_vec();
        let in_ch = self.in_channels;
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let has_bias = self.bias.is_some();

        let mut parents = vec![x.clone(), self.weight.clone()];
        if let Some(b) = &self.bias {
            parents.push(b.clone());
        }

        let grad_fn = Box::new(move |g: &Tensor<T>| {
            conv2d_backward(
                g,
                &col_captured,
                &w_captured,
                &cs,
                out_ch,
                in_ch,
                kh,
                kw,
                col_rows,
                col_cols,
                h_out,
                w_out,
                has_bias,
            )
        });

        Ok(Variable::from_op(out_tensor, parents, grad_fn))
    }
}

impl<T: Float> Layer<T> for Conv2d<T> {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        self.forward_impl(x)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

/// Forward computation: col @ W^T + bias, reshaped to NCHW.
#[allow(clippy::too_many_arguments)]
fn compute_forward<T: Float>(
    col: &[T],
    w: &[T],
    bias: Option<&Variable<T>>,
    n: usize,
    out_ch: usize,
    h_out: usize,
    w_out: usize,
    col_rows: usize,
    col_cols: usize,
) -> Tensor<T> {
    let mut out = vec![T::zero(); col_rows * out_ch];
    for i in 0..col_rows {
        for o in 0..out_ch {
            let mut sum = T::zero();
            for k in 0..col_cols {
                sum += col[i * col_cols + k] * w[o * col_cols + k];
            }
            out[i * out_ch + o] = sum;
        }
    }

    if let Some(b) = bias {
        let b_data = b.data();
        let b_s = b_data.as_slice();
        for i in 0..col_rows {
            for o in 0..out_ch {
                out[i * out_ch + o] += b_s[o];
            }
        }
    }

    // Reorder [col_rows, out_ch] → [N, C_out, h_out, w_out]
    let mut nchw = vec![T::zero(); n * out_ch * h_out * w_out];
    for batch in 0..n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let row = batch * h_out * w_out + oh * w_out + ow;
                for o in 0..out_ch {
                    nchw[batch * out_ch * h_out * w_out + o * h_out * w_out + oh * w_out + ow] =
                        out[row * out_ch + o];
                }
            }
        }
    }

    Tensor::from_vec(nchw, vec![n, out_ch, h_out, w_out]).unwrap()
}

/// Backward pass for Conv2d.
#[allow(clippy::too_many_arguments)]
fn conv2d_backward<T: Float>(
    g: &Tensor<T>,
    col: &[T],
    w: &[T],
    cs: &ConvShape,
    out_ch: usize,
    in_ch: usize,
    kh: usize,
    kw: usize,
    col_rows: usize,
    col_cols: usize,
    h_out: usize,
    w_out: usize,
    has_bias: bool,
) -> Vec<Tensor<T>> {
    let n = cs.n;
    let g_data = g.as_slice();

    // Reshape NCHW grad → [col_rows, out_ch]
    let mut g_mat = vec![T::zero(); col_rows * out_ch];
    for batch in 0..n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let row = batch * h_out * w_out + oh * w_out + ow;
                for o in 0..out_ch {
                    g_mat[row * out_ch + o] = g_data
                        [batch * out_ch * h_out * w_out + o * h_out * w_out + oh * w_out + ow];
                }
            }
        }
    }

    // grad_weight = g_mat^T @ col → [out_ch, col_cols]
    let mut gw = vec![T::zero(); out_ch * col_cols];
    for o in 0..out_ch {
        for k in 0..col_cols {
            let mut sum = T::zero();
            for i in 0..col_rows {
                sum += g_mat[i * out_ch + o] * col[i * col_cols + k];
            }
            gw[o * col_cols + k] = sum;
        }
    }
    let gw_t = Tensor::from_vec(gw, vec![out_ch, in_ch, kh, kw]).expect("valid grad_w");

    // grad_col = g_mat @ W → [col_rows, col_cols]
    let mut g_col = vec![T::zero(); col_rows * col_cols];
    for i in 0..col_rows {
        for k in 0..col_cols {
            let mut sum = T::zero();
            for o in 0..out_ch {
                sum += g_mat[i * out_ch + o] * w[o * col_cols + k];
            }
            g_col[i * col_cols + k] = sum;
        }
    }

    let gx_data = col2im(&g_col, cs);
    let gx = Tensor::from_vec(gx_data, vec![n, in_ch, cs.h, cs.w]).expect("valid grad_x");

    if has_bias {
        let mut gb = vec![T::zero(); out_ch];
        for i in 0..col_rows {
            for o in 0..out_ch {
                gb[o] += g_mat[i * out_ch + o];
            }
        }
        vec![gx, gw_t, Tensor::from_vec(gb, vec![out_ch]).unwrap()]
    } else {
        vec![gx, gw_t]
    }
}

// ── Conv1d ────────────────────────────────────────────────────────────────

/// 1-D convolution layer.
///
/// Input shape: `[batch, in_channels, length]`
/// Output shape: `[batch, out_channels, length_out]`
///
/// Implemented by expanding to 4-D and delegating to [`Conv2d`].
pub struct Conv1d<T: Float> {
    inner: Conv2d<T>,
}

impl<T: Float> Conv1d<T> {
    /// Create a new Conv1d layer.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        use_bias: bool,
        rng: &mut Rng,
    ) -> Self {
        Self {
            inner: Conv2d::new(in_channels, out_channels, (1, kernel_size), use_bias, rng),
        }
    }

    /// Set stride.
    pub fn set_stride(mut self, stride: usize) -> Self {
        self.inner = self.inner.set_stride((1, stride));
        self
    }

    /// Set padding.
    pub fn set_padding(mut self, padding: usize) -> Self {
        self.inner = self.inner.set_padding((0, padding));
        self
    }

    /// Set dilation.
    pub fn set_dilation(mut self, dilation: usize) -> Self {
        self.inner = self.inner.set_dilation((1, dilation));
        self
    }
}

impl<T: Float> Layer<T> for Conv1d<T> {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, 0, 0],
                got: shape,
            });
        }
        let (n, c, l) = (shape[0], shape[1], shape[2]);
        let expanded = Variable::new(
            Tensor::from_vec(x.data().as_slice().to_vec(), vec![n, c, 1, l])
                .map_err(NnError::from)?,
            x.requires_grad(),
        );
        let out4d = self.inner.forward(&expanded)?;
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
        self.inner.parameters()
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

// ── im2col / col2im ──────────────────────────────────────────────────────

fn im2col<T: Float>(input: &[T], cs: &ConvShape) -> Vec<T> {
    let h_out = cs.h_out();
    let w_out = cs.w_out();
    let col_rows = cs.col_rows();
    let col_cols = cs.col_cols();
    let mut col = vec![T::zero(); col_rows * col_cols];

    for batch in 0..cs.n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let row = batch * h_out * w_out + oh * w_out + ow;
                let mut ci = 0;
                for c in 0..cs.c_in {
                    for ki in 0..cs.kh {
                        for kj in 0..cs.kw {
                            if let (Some(ih), Some(iw)) = (
                                in_bounds(oh, cs.sh, ki, cs.dh, cs.ph, cs.h),
                                in_bounds(ow, cs.sw, kj, cs.dw, cs.pw, cs.w),
                            ) {
                                let idx = batch * cs.c_in * cs.h * cs.w
                                    + c * cs.h * cs.w
                                    + ih * cs.w
                                    + iw;
                                col[row * col_cols + ci] = input[idx];
                            }
                            ci += 1;
                        }
                    }
                }
            }
        }
    }
    col
}

fn col2im<T: Float>(col: &[T], cs: &ConvShape) -> Vec<T> {
    let h_out = cs.h_out();
    let w_out = cs.w_out();
    let col_cols = cs.col_cols();
    let mut img = vec![T::zero(); cs.n * cs.c_in * cs.h * cs.w];

    for batch in 0..cs.n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let row = batch * h_out * w_out + oh * w_out + ow;
                let mut ci = 0;
                for c in 0..cs.c_in {
                    for ki in 0..cs.kh {
                        for kj in 0..cs.kw {
                            if let (Some(ih), Some(iw)) = (
                                in_bounds(oh, cs.sh, ki, cs.dh, cs.ph, cs.h),
                                in_bounds(ow, cs.sw, kj, cs.dw, cs.pw, cs.w),
                            ) {
                                let idx = batch * cs.c_in * cs.h * cs.w
                                    + c * cs.h * cs.w
                                    + ih * cs.w
                                    + iw;
                                img[idx] += col[row * col_cols + ci];
                            }
                            ci += 1;
                        }
                    }
                }
            }
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_output_shape() {
        let mut rng = Rng::new(42);
        let conv = Conv2d::<f64>::new(3, 16, (3, 3), true, &mut rng);
        let x = Variable::new(Tensor::ones(vec![2, 3, 8, 8]), true);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 16, 6, 6]);
    }

    #[test]
    fn test_conv2d_with_padding() {
        let mut rng = Rng::new(42);
        let conv = Conv2d::<f64>::new(1, 1, (3, 3), false, &mut rng).set_padding((1, 1));
        let x = Variable::new(Tensor::ones(vec![1, 1, 5, 5]), true);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 5, 5]);
    }

    #[test]
    fn test_conv2d_with_stride() {
        let mut rng = Rng::new(42);
        let conv = Conv2d::<f64>::new(1, 4, (3, 3), true, &mut rng).set_stride((2, 2));
        let x = Variable::new(Tensor::ones(vec![1, 1, 7, 7]), true);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 4, 3, 3]);
    }

    #[test]
    fn test_conv2d_backward() {
        let mut rng = Rng::new(42);
        let conv = Conv2d::<f64>::new(1, 1, (2, 2), true, &mut rng);
        let x = Variable::new(
            Tensor::from_vec(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                vec![1, 1, 3, 3],
            )
            .unwrap(),
            true,
        );
        let y = conv.forward(&x).unwrap();
        let loss = crate::ops::sum(&y);
        loss.backward();
        assert!(conv.weight.grad().is_some());
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_conv2d_parameters() {
        let mut rng = Rng::new(42);
        let conv = Conv2d::<f64>::new(3, 16, (3, 3), true, &mut rng);
        assert_eq!(conv.parameters().len(), 2);
        let conv_no_bias = Conv2d::<f64>::new(3, 16, (3, 3), false, &mut rng);
        assert_eq!(conv_no_bias.parameters().len(), 1);
    }

    #[test]
    fn test_conv1d_output_shape() {
        let mut rng = Rng::new(42);
        let conv = Conv1d::<f64>::new(4, 8, 3, true, &mut rng);
        let x = Variable::new(Tensor::ones(vec![2, 4, 16]), true);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 8, 14]);
    }

    #[test]
    fn test_conv1d_with_padding() {
        let mut rng = Rng::new(42);
        let conv = Conv1d::<f64>::new(1, 1, 3, false, &mut rng).set_padding(1);
        let x = Variable::new(Tensor::ones(vec![1, 1, 10]), true);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 10]);
    }
}
