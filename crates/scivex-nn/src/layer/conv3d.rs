//! 3-D volumetric convolution layer using im2col.

use scivex_core::random::Rng;
use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::init;
use crate::variable::Variable;

use super::Layer;

// ── ConvShape3d ──────────────────────────────────────────────────────────

/// Bundles 3-D convolution geometry so we don't pass 18+ arguments everywhere.
#[derive(Clone, Copy)]
struct ConvShape3d {
    n: usize,
    c_in: usize,
    d: usize,
    h: usize,
    w: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    sd: usize,
    sh: usize,
    sw: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    dd: usize,
    dh: usize,
    dw: usize,
}

impl ConvShape3d {
    fn d_out(&self) -> usize {
        (self.d + 2 * self.pd - self.dd * (self.kd - 1) - 1) / self.sd + 1
    }
    fn h_out(&self) -> usize {
        (self.h + 2 * self.ph - self.dh * (self.kh - 1) - 1) / self.sh + 1
    }
    fn w_out(&self) -> usize {
        (self.w + 2 * self.pw - self.dw * (self.kw - 1) - 1) / self.sw + 1
    }
    fn col_rows(&self) -> usize {
        self.n * self.d_out() * self.h_out() * self.w_out()
    }
    fn col_cols(&self) -> usize {
        self.c_in * self.kd * self.kh * self.kw
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

// ── Conv3d ───────────────────────────────────────────────────────────────

/// 3-D volumetric convolution layer.
///
/// Input shape: `[batch, in_channels, depth, height, width]` (NCDHW)
/// Output shape: `[batch, out_channels, d_out, h_out, w_out]`
///
/// Uses im2col internally: volumetric patches are unrolled into a matrix,
/// then a single matrix multiplication produces the convolution output.
pub struct Conv3d<T: Float> {
    weight: Variable<T>,       // [out_ch, in_ch, kd, kh, kw]
    bias: Option<Variable<T>>, // [out_ch]
    in_channels: usize,
    out_channels: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    padding_d: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_d: usize,
    dilation_h: usize,
    dilation_w: usize,
}

impl<T: Float> Conv3d<T> {
    /// Create a new Conv3d layer.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        use_bias: bool,
        rng: &mut Rng,
    ) -> Self {
        let (kd, kh, kw) = kernel_size;
        let fan_in = in_channels * kd * kh * kw;
        let w_data = init::kaiming_uniform::<T>(&[out_channels, fan_in], rng);
        let weight = Variable::new(
            Tensor::from_vec(
                w_data.as_slice().to_vec(),
                vec![out_channels, in_channels, kd, kh, kw],
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
            kernel_d: kd,
            kernel_h: kh,
            kernel_w: kw,
            stride_d: 1,
            stride_h: 1,
            stride_w: 1,
            padding_d: 0,
            padding_h: 0,
            padding_w: 0,
            dilation_d: 1,
            dilation_h: 1,
            dilation_w: 1,
        }
    }

    /// Set stride (depth, height, width).
    pub fn set_stride(mut self, stride: (usize, usize, usize)) -> Self {
        self.stride_d = stride.0.max(1);
        self.stride_h = stride.1.max(1);
        self.stride_w = stride.2.max(1);
        self
    }

    /// Set padding (depth, height, width).
    pub fn set_padding(mut self, padding: (usize, usize, usize)) -> Self {
        self.padding_d = padding.0;
        self.padding_h = padding.1;
        self.padding_w = padding.2;
        self
    }

    /// Set dilation (depth, height, width).
    pub fn set_dilation(mut self, dilation: (usize, usize, usize)) -> Self {
        self.dilation_d = dilation.0.max(1);
        self.dilation_h = dilation.1.max(1);
        self.dilation_w = dilation.2.max(1);
        self
    }

    fn conv_shape(&self, n: usize, d: usize, h: usize, w: usize) -> ConvShape3d {
        ConvShape3d {
            n,
            c_in: self.in_channels,
            d,
            h,
            w,
            kd: self.kernel_d,
            kh: self.kernel_h,
            kw: self.kernel_w,
            sd: self.stride_d,
            sh: self.stride_h,
            sw: self.stride_w,
            pd: self.padding_d,
            ph: self.padding_h,
            pw: self.padding_w,
            dd: self.dilation_d,
            dh: self.dilation_h,
            dw: self.dilation_w,
        }
    }

    /// Compute forward output and build gradient closure.
    fn forward_impl(&self, x: &Variable<T>) -> Result<Variable<T>> {
        let shape = x.shape();
        if shape.len() != 5 || shape[1] != self.in_channels {
            return Err(NnError::ShapeMismatch {
                expected: vec![0, self.in_channels, 0, 0, 0],
                got: shape,
            });
        }
        let (n, d, h, w) = (shape[0], shape[2], shape[3], shape[4]);
        let cs = self.conv_shape(n, d, h, w);
        let (d_out, h_out, w_out) = (cs.d_out(), cs.h_out(), cs.w_out());
        let col_rows = cs.col_rows();
        let col_cols = cs.col_cols();
        let out_ch = self.out_channels;

        let col = im2col_3d(x.data().as_slice(), &cs);
        let w_flat = self.weight.data();
        let w_slice = w_flat.as_slice();

        // col @ W^T → [col_rows, out_ch], then add bias, reorder to NCDHW
        let out_tensor = compute_forward_3d(
            &col,
            w_slice,
            self.bias.as_ref(),
            n,
            out_ch,
            d_out,
            h_out,
            w_out,
            col_rows,
            col_cols,
        );

        let col_captured = col;
        let w_captured = w_slice.to_vec();
        let in_ch = self.in_channels;
        let kd = self.kernel_d;
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let has_bias = self.bias.is_some();

        let mut parents = vec![x.clone(), self.weight.clone()];
        if let Some(b) = &self.bias {
            parents.push(b.clone());
        }

        let grad_fn = Box::new(move |g: &Tensor<T>| {
            conv3d_backward(
                g,
                &col_captured,
                &w_captured,
                &cs,
                out_ch,
                in_ch,
                kd,
                kh,
                kw,
                col_rows,
                col_cols,
                d_out,
                h_out,
                w_out,
                has_bias,
            )
        });

        Ok(Variable::from_op(out_tensor, parents, grad_fn))
    }
}

impl<T: Float> Layer<T> for Conv3d<T> {
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

/// Forward computation: col @ W^T + bias, reshaped to NCDHW.
#[allow(clippy::too_many_arguments)]
fn compute_forward_3d<T: Float>(
    col: &[T],
    w: &[T],
    bias: Option<&Variable<T>>,
    n: usize,
    out_ch: usize,
    d_out: usize,
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

    // Reorder [col_rows, out_ch] → [N, C_out, d_out, h_out, w_out]
    let spatial = d_out * h_out * w_out;
    let mut ncdhw = vec![T::zero(); n * out_ch * spatial];
    for batch in 0..n {
        for od in 0..d_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let row = batch * spatial + od * h_out * w_out + oh * w_out + ow;
                    for o in 0..out_ch {
                        ncdhw[batch * out_ch * spatial
                            + o * spatial
                            + od * h_out * w_out
                            + oh * w_out
                            + ow] = out[row * out_ch + o];
                    }
                }
            }
        }
    }

    // ncdhw length == n * out_ch * d_out * h_out * w_out by construction
    Tensor::from_vec(ncdhw, vec![n, out_ch, d_out, h_out, w_out])
        .expect("valid conv3d output shape")
}

/// Backward pass for Conv3d.
#[allow(clippy::too_many_arguments)]
fn conv3d_backward<T: Float>(
    g: &Tensor<T>,
    col: &[T],
    w: &[T],
    cs: &ConvShape3d,
    out_ch: usize,
    in_ch: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    col_rows: usize,
    col_cols: usize,
    d_out: usize,
    h_out: usize,
    w_out: usize,
    has_bias: bool,
) -> Vec<Tensor<T>> {
    let n = cs.n;
    let g_data = g.as_slice();
    let spatial = d_out * h_out * w_out;

    // Reshape NCDHW grad → [col_rows, out_ch]
    let mut g_mat = vec![T::zero(); col_rows * out_ch];
    for batch in 0..n {
        for od in 0..d_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let row = batch * spatial + od * h_out * w_out + oh * w_out + ow;
                    for o in 0..out_ch {
                        g_mat[row * out_ch + o] = g_data[batch * out_ch * spatial
                            + o * spatial
                            + od * h_out * w_out
                            + oh * w_out
                            + ow];
                    }
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
    let gw_t = Tensor::from_vec(gw, vec![out_ch, in_ch, kd, kh, kw]).expect("valid grad_w");

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

    let gx_data = col2im_3d(&g_col, cs);
    let gx = Tensor::from_vec(gx_data, vec![n, in_ch, cs.d, cs.h, cs.w]).expect("valid grad_x");

    if has_bias {
        let mut gb = vec![T::zero(); out_ch];
        for i in 0..col_rows {
            for o in 0..out_ch {
                gb[o] += g_mat[i * out_ch + o];
            }
        }
        vec![
            gx,
            gw_t,
            Tensor::from_vec(gb, vec![out_ch]).expect("valid bias grad shape"),
        ]
    } else {
        vec![gx, gw_t]
    }
}

// ── im2col_3d / col2im_3d ────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
fn im2col_3d<T: Float>(input: &[T], cs: &ConvShape3d) -> Vec<T> {
    let d_out = cs.d_out();
    let h_out = cs.h_out();
    let w_out = cs.w_out();
    let col_rows = cs.col_rows();
    let col_cols = cs.col_cols();
    let mut col = vec![T::zero(); col_rows * col_cols];

    for batch in 0..cs.n {
        for od in 0..d_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let row = batch * d_out * h_out * w_out + od * h_out * w_out + oh * w_out + ow;
                    let mut ci = 0;
                    for c in 0..cs.c_in {
                        for kd in 0..cs.kd {
                            for ki in 0..cs.kh {
                                for kj in 0..cs.kw {
                                    if let (Some(id), Some(ih), Some(iw)) = (
                                        in_bounds(od, cs.sd, kd, cs.dd, cs.pd, cs.d),
                                        in_bounds(oh, cs.sh, ki, cs.dh, cs.ph, cs.h),
                                        in_bounds(ow, cs.sw, kj, cs.dw, cs.pw, cs.w),
                                    ) {
                                        let idx = batch * cs.c_in * cs.d * cs.h * cs.w
                                            + c * cs.d * cs.h * cs.w
                                            + id * cs.h * cs.w
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
        }
    }
    col
}

fn col2im_3d<T: Float>(col: &[T], cs: &ConvShape3d) -> Vec<T> {
    let d_out = cs.d_out();
    let h_out = cs.h_out();
    let w_out = cs.w_out();
    let col_cols = cs.col_cols();
    let mut img = vec![T::zero(); cs.n * cs.c_in * cs.d * cs.h * cs.w];

    for batch in 0..cs.n {
        for od in 0..d_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let row = batch * d_out * h_out * w_out + od * h_out * w_out + oh * w_out + ow;
                    let mut ci = 0;
                    for c in 0..cs.c_in {
                        for kd in 0..cs.kd {
                            for ki in 0..cs.kh {
                                for kj in 0..cs.kw {
                                    if let (Some(id), Some(ih), Some(iw)) = (
                                        in_bounds(od, cs.sd, kd, cs.dd, cs.pd, cs.d),
                                        in_bounds(oh, cs.sh, ki, cs.dh, cs.ph, cs.h),
                                        in_bounds(ow, cs.sw, kj, cs.dw, cs.pw, cs.w),
                                    ) {
                                        let idx = batch * cs.c_in * cs.d * cs.h * cs.w
                                            + c * cs.d * cs.h * cs.w
                                            + id * cs.h * cs.w
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
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv3d_output_shape() {
        let mut rng = Rng::new(42);
        let conv = Conv3d::<f64>::new(3, 16, (3, 3, 3), true, &mut rng);
        let x = Variable::new(Tensor::ones(vec![2, 3, 8, 8, 8]), true);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 16, 6, 6, 6]);
    }

    #[test]
    fn test_conv3d_with_padding() {
        let mut rng = Rng::new(42);
        let conv = Conv3d::<f64>::new(1, 1, (3, 3, 3), false, &mut rng).set_padding((1, 1, 1));
        let x = Variable::new(Tensor::ones(vec![1, 1, 5, 5, 5]), true);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 5, 5, 5]);
    }

    #[test]
    fn test_conv3d_with_stride() {
        let mut rng = Rng::new(42);
        let conv = Conv3d::<f64>::new(1, 4, (3, 3, 3), true, &mut rng).set_stride((2, 2, 2));
        let x = Variable::new(Tensor::ones(vec![1, 1, 7, 7, 7]), true);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 4, 3, 3, 3]);
    }

    #[test]
    fn test_conv3d_backward() {
        let mut rng = Rng::new(42);
        let conv = Conv3d::<f64>::new(1, 1, (2, 2, 2), true, &mut rng);
        let data: Vec<f64> = (1..=27).map(f64::from).collect();
        let x = Variable::new(Tensor::from_vec(data, vec![1, 1, 3, 3, 3]).unwrap(), true);
        let y = conv.forward(&x).unwrap();
        let loss = crate::ops::sum(&y);
        loss.backward();
        assert!(conv.weight.grad().is_some());
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_conv3d_parameters() {
        let mut rng = Rng::new(42);
        let conv = Conv3d::<f64>::new(3, 16, (3, 3, 3), true, &mut rng);
        assert_eq!(conv.parameters().len(), 2);
        let conv_no_bias = Conv3d::<f64>::new(3, 16, (3, 3, 3), false, &mut rng);
        assert_eq!(conv_no_bias.parameters().len(), 1);
    }
}
