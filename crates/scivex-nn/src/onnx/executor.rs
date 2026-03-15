//! ONNX graph executor -- runs inference on a parsed ONNX model.
//!
//! Supports a practical subset of ONNX operators sufficient for running
//! common neural network architectures (MLPs, CNNs, basic transformers).

use std::collections::{HashMap, HashSet};

use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::onnx::ir::{OnnxGraph, OnnxModel, OnnxNode};

// -----------------------------------------------------------------------
// Inference session
// -----------------------------------------------------------------------

/// An executable ONNX inference session.
///
/// Holds the parsed graph and pre-loaded initializer tensors so that
/// [`run`](Self::run) can be called repeatedly without re-parsing.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct OnnxInferenceSession<T: Float> {
    /// The graph (owns nodes, initializer metadata, I/O info).
    graph: OnnxGraph,
    /// Pre-loaded initializer tensors keyed by name.
    initializers: HashMap<String, Tensor<T>>,
    /// Topologically sorted node indices.
    topo_order: Vec<usize>,
}

impl<T: Float> OnnxInferenceSession<T> {
    /// Create an inference session from a parsed ONNX model.
    pub fn from_model(model: OnnxModel) -> Result<Self> {
        let graph = model.graph;

        let mut initializers = HashMap::new();
        for init in &graph.initializers {
            let f32_data = init.to_f32_vec();
            let shape = init.dims_usize();
            let data: Vec<T> = f32_data.iter().map(|&v| T::from_f64(f64::from(v))).collect();
            let numel: usize = shape.iter().product();
            if data.len() != numel {
                return Err(NnError::OnnxError(format!(
                    "initializer '{}': data length {} != shape product {}",
                    init.name,
                    data.len(),
                    numel,
                )));
            }
            let tensor = Tensor::from_vec(data, shape).map_err(|e| {
                NnError::OnnxError(format!("initializer '{}': {e}", init.name))
            })?;
            initializers.insert(init.name.clone(), tensor);
        }

        let topo_order = topo_sort(&graph)?;

        Ok(Self {
            graph,
            initializers,
            topo_order,
        })
    }

    /// Run inference with the given named input tensors.
    ///
    /// Returns the graph output tensors in the same order as `graph.outputs`.
    pub fn run(&self, inputs: &[(&str, Tensor<T>)]) -> Result<Vec<Tensor<T>>> {
        let mut env: HashMap<String, Tensor<T>> = HashMap::new();

        for (name, tensor) in &self.initializers {
            env.insert(name.clone(), tensor.clone());
        }

        for &(name, ref tensor) in inputs {
            env.insert(name.to_owned(), tensor.clone());
        }

        for &idx in &self.topo_order {
            let node = &self.graph.nodes[idx];
            execute_node(node, &mut env)?;
        }

        let mut outputs = Vec::new();
        for out_info in &self.graph.outputs {
            let tensor = env.get(&out_info.name).ok_or_else(|| {
                NnError::OnnxError(format!(
                    "output tensor '{}' not found in execution environment",
                    out_info.name
                ))
            })?;
            outputs.push(tensor.clone());
        }

        Ok(outputs)
    }
}

// -----------------------------------------------------------------------
// Topological sort
// -----------------------------------------------------------------------

fn topo_sort(graph: &OnnxGraph) -> Result<Vec<usize>> {
    let n = graph.nodes.len();
    let mut producer: HashMap<&str, usize> = HashMap::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        for out in &node.outputs {
            producer.insert(out.as_str(), i);
        }
    }

    let mut in_degree = vec![0u32; n];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, node) in graph.nodes.iter().enumerate() {
        let mut seen_deps: HashSet<usize> = HashSet::new();
        for inp in &node.inputs {
            if inp.is_empty() {
                continue;
            }
            #[allow(clippy::collapsible_if)]
            if let Some(&prod_idx) = producer.get(inp.as_str()) {
                if prod_idx != i && seen_deps.insert(prod_idx) {
                    in_degree[i] += 1;
                    dependents[prod_idx].push(i);
                }
            }
        }
    }

    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);

    while let Some(idx) = queue.pop() {
        order.push(idx);
        for &dep in &dependents[idx] {
            in_degree[dep] -= 1;
            if in_degree[dep] == 0 {
                queue.push(dep);
            }
        }
    }

    if order.len() != n {
        return Err(NnError::OnnxError(
            "graph contains a cycle; cannot topologically sort".into(),
        ));
    }

    Ok(order)
}

// -----------------------------------------------------------------------
// Operator dispatch
// -----------------------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn execute_node<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
) -> Result<()> {
    match node.op_type.as_str() {
        "Add" => binary_elementwise(node, env, |a, b| a + b),
        "Sub" => binary_elementwise(node, env, |a, b| a - b),
        "Mul" => binary_elementwise(node, env, |a, b| a * b),
        "Div" => binary_elementwise(node, env, |a, b| a / b),
        "Relu" => unary_elementwise(node, env, |x| {
            if x > T::zero() { x } else { T::zero() }
        }),
        "Sigmoid" => unary_elementwise(node, env, |x| {
            T::one() / (T::one() + (-x).exp())
        }),
        "Tanh" => unary_elementwise(node, env, |x| {
            let e2x = (x + x).exp();
            (e2x - T::one()) / (e2x + T::one())
        }),
        "Softmax" => exec_softmax(node, env),
        "MatMul" => exec_matmul(node, env),
        "Gemm" => exec_gemm(node, env),
        "Reshape" => exec_reshape(node, env),
        "Transpose" => exec_transpose(node, env),
        "Flatten" => exec_flatten(node, env),
        "Concat" => exec_concat(node, env),
        "BatchNormalization" => exec_batchnorm(node, env),
        "Dropout" => exec_dropout(node, env),
        "Unsqueeze" => exec_unsqueeze(node, env),
        "Squeeze" => exec_squeeze(node, env),
        "Conv" => exec_conv(node, env),
        "MaxPool" => exec_maxpool(node, env),
        "AveragePool" => exec_avgpool(node, env),
        other => Err(NnError::OnnxError(format!(
            "unsupported ONNX operator: {other}"
        ))),
    }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

fn get_input<T: Float>(
    node: &OnnxNode,
    env: &HashMap<String, Tensor<T>>,
    index: usize,
) -> Result<Tensor<T>> {
    let name = node.inputs.get(index).ok_or_else(|| {
        NnError::OnnxError(format!(
            "{}: missing input at index {index}",
            node.op_type
        ))
    })?;
    env.get(name).cloned().ok_or_else(|| {
        NnError::OnnxError(format!(
            "{}: input tensor '{}' not found",
            node.op_type, name
        ))
    })
}

fn set_output<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
    index: usize,
    tensor: Tensor<T>,
) -> Result<()> {
    let name = node.outputs.get(index).ok_or_else(|| {
        NnError::OnnxError(format!(
            "{}: missing output at index {index}",
            node.op_type
        ))
    })?;
    env.insert(name.clone(), tensor);
    Ok(())
}

// -----------------------------------------------------------------------
// Broadcasting support
// -----------------------------------------------------------------------

fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut result = vec![0usize; ndim];

    for i in 0..ndim {
        let da = if i < ndim - a.len() { 1 } else { a[i - (ndim - a.len())] };
        let db = if i < ndim - b.len() { 1 } else { b[i - (ndim - b.len())] };
        if da == db {
            result[i] = da;
        } else if da == 1 {
            result[i] = db;
        } else if db == 1 {
            result[i] = da;
        } else {
            return Err(NnError::OnnxError(format!(
                "cannot broadcast shapes {a:?} and {b:?}"
            )));
        }
    }
    Ok(result)
}

fn broadcast_flat_index(shape: &[usize], strides: &[usize], nd_index: &[usize], ndim: usize) -> usize {
    let offset = ndim - shape.len();
    let mut flat = 0;
    for i in 0..shape.len() {
        let idx = nd_index[i + offset];
        let effective = if shape[i] == 1 { 0 } else { idx };
        flat += effective * strides[i];
    }
    flat
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn broadcast_binary<T: Float, F: Fn(T, T) -> T>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    f: F,
) -> Result<Tensor<T>> {
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    let numel: usize = out_shape.iter().product();
    let ndim = out_shape.len();

    let a_strides = compute_strides(a.shape());
    let b_strides = compute_strides(b.shape());

    let mut data = Vec::with_capacity(numel);
    let mut nd_index = vec![0usize; ndim];

    let a_slice = a.as_slice();
    let b_slice = b.as_slice();

    for _ in 0..numel {
        let ai = broadcast_flat_index(a.shape(), &a_strides, &nd_index, ndim);
        let bi = broadcast_flat_index(b.shape(), &b_strides, &nd_index, ndim);
        data.push(f(a_slice[ai], b_slice[bi]));

        for d in (0..ndim).rev() {
            nd_index[d] += 1;
            if nd_index[d] < out_shape[d] {
                break;
            }
            nd_index[d] = 0;
        }
    }

    Tensor::from_vec(data, out_shape).map_err(|e| NnError::OnnxError(format!("{e}")))
}

// -----------------------------------------------------------------------
// Operator implementations
// -----------------------------------------------------------------------

fn binary_elementwise<T: Float, F: Fn(T, T) -> T>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
    f: F,
) -> Result<()> {
    let a = get_input(node, env, 0)?;
    let b = get_input(node, env, 1)?;
    let result = broadcast_binary(&a, &b, f)?;
    set_output(node, env, 0, result)
}

fn unary_elementwise<T: Float, F: Fn(T) -> T>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
    f: F,
) -> Result<()> {
    let x = get_input(node, env, 0)?;
    let result = x.map(f);
    set_output(node, env, 0, result)
}

fn exec_softmax<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
) -> Result<()> {
    let x = get_input(node, env, 0)?;
    let axis = node.get_int_attr("axis", -1);
    let shape = x.shape().to_vec();
    let ndim = shape.len();

    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    let axis_usize = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };

    if axis_usize >= ndim {
        return Err(NnError::OnnxError(format!(
            "Softmax: axis {axis} out of range for ndim {ndim}"
        )));
    }

    let data = x.as_slice();
    let mut result = vec![T::zero(); data.len()];

    let outer: usize = shape[..axis_usize].iter().product();
    let axis_len = shape[axis_usize];
    let inner: usize = shape[axis_usize + 1..].iter().product();

    for o in 0..outer {
        for i in 0..inner {
            let mut max_val = T::neg_infinity();
            for a in 0..axis_len {
                let idx = o * axis_len * inner + a * inner + i;
                max_val = max_val.max(data[idx]);
            }

            let mut sum = T::zero();
            for a in 0..axis_len {
                let idx = o * axis_len * inner + a * inner + i;
                let e = (data[idx] - max_val).exp();
                result[idx] = e;
                sum += e;
            }

            for a in 0..axis_len {
                let idx = o * axis_len * inner + a * inner + i;
                result[idx] /= sum;
            }
        }
    }

    let out = Tensor::from_vec(result, shape).map_err(|e| NnError::OnnxError(format!("{e}")))?;
    set_output(node, env, 0, out)
}

fn exec_matmul<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
) -> Result<()> {
    let a = get_input(node, env, 0)?;
    let b = get_input(node, env, 1)?;

    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(NnError::OnnxError(format!(
            "MatMul: expected 2-D tensors, got {}D and {}D",
            a.ndim(),
            b.ndim()
        )));
    }

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    if k != b.shape()[0] {
        return Err(NnError::OnnxError(format!(
            "MatMul: inner dimensions mismatch: {} vs {}",
            k,
            b.shape()[0]
        )));
    }

    let a_data = a.as_slice();
    let b_data = b.as_slice();
    let mut out = vec![T::zero(); m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for p in 0..k {
                sum += a_data[i * k + p] * b_data[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }

    let result = Tensor::from_vec(out, vec![m, n]).map_err(|e| NnError::OnnxError(format!("{e}")))?;
    set_output(node, env, 0, result)
}

#[allow(clippy::too_many_lines)]
fn exec_gemm<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
) -> Result<()> {
    let a_in = get_input(node, env, 0)?;
    let b_in = get_input(node, env, 1)?;

    let alpha = T::from_f64(f64::from(node.get_float_attr("alpha", 1.0)));
    let beta = T::from_f64(f64::from(node.get_float_attr("beta", 1.0)));
    let trans_a = node.get_int_attr("transA", 0) != 0;
    let trans_b = node.get_int_attr("transB", 0) != 0;

    let a = if trans_a {
        a_in.transpose().map_err(|e| NnError::OnnxError(format!("{e}")))?
    } else {
        a_in
    };
    let b = if trans_b {
        b_in.transpose().map_err(|e| NnError::OnnxError(format!("{e}")))?
    } else {
        b_in
    };

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    if k != b.shape()[0] {
        return Err(NnError::OnnxError(format!(
            "Gemm: inner dimensions mismatch: {} vs {}",
            k,
            b.shape()[0]
        )));
    }

    let a_data = a.as_slice();
    let b_data = b.as_slice();
    let mut out = vec![T::zero(); m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for p in 0..k {
                sum += a_data[i * k + p] * b_data[p * n + j];
            }
            out[i * n + j] = alpha * sum;
        }
    }

    // Add bias C if provided.
    #[allow(clippy::collapsible_if)]
    if node.inputs.len() > 2 {
        if let Ok(c) = get_input(node, env, 2) {
            let c_data = c.as_slice();
            if c.ndim() == 1 && c.shape()[0] == n {
                for i in 0..m {
                    for j in 0..n {
                        out[i * n + j] += beta * c_data[j];
                    }
                }
            } else if c.ndim() == 2 && c.shape()[0] == m && c.shape()[1] == n {
                for idx in 0..m * n {
                    out[idx] += beta * c_data[idx];
                }
            } else if c.numel() == 1 {
                let scalar = c_data[0];
                for val in &mut out {
                    *val += beta * scalar;
                }
            }
        }
    }

    let result = Tensor::from_vec(out, vec![m, n]).map_err(|e| NnError::OnnxError(format!("{e}")))?;
    set_output(node, env, 0, result)
}

fn exec_reshape<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
) -> Result<()> {
    let x = get_input(node, env, 0)?;
    let shape_tensor = get_input(node, env, 1)?;

    let numel = x.numel();
    let shape_data = shape_tensor.as_slice();

    #[allow(clippy::cast_possible_truncation)]
    let mut new_shape: Vec<i64> = shape_data.iter().map(|&v| v.to_f64() as i64).collect();

    // Handle 0 = keep original dim.
    for (i, dim) in new_shape.iter_mut().enumerate() {
        if *dim == 0 {
            #[allow(clippy::collapsible_if)]
            if i < x.shape().len() {
                #[allow(clippy::cast_possible_wrap)]
                {
                    *dim = x.shape()[i] as i64;
                }
            }
        }
    }

    // Handle -1 = infer.
    let neg_count = new_shape.iter().filter(|&&d| d == -1).count();
    if neg_count == 1 {
        let known_product: i64 = new_shape.iter().filter(|&&d| d != -1).product();
        if known_product != 0 {
            #[allow(clippy::cast_possible_wrap)]
            let inferred = numel as i64 / known_product;
            for dim in &mut new_shape {
                if *dim == -1 {
                    *dim = inferred;
                    break;
                }
            }
        }
    }

    #[allow(clippy::cast_sign_loss)]
    let final_shape: Vec<usize> = new_shape.iter().map(|&d| d as usize).collect();

    let result = x.reshape(final_shape).map_err(|e| NnError::OnnxError(format!("{e}")))?;
    set_output(node, env, 0, result)
}

fn exec_transpose<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
) -> Result<()> {
    let x = get_input(node, env, 0)?;
    let perm = node.get_ints_attr("perm");

    let result = if perm.is_empty() {
        let axes: Vec<usize> = (0..x.ndim()).rev().collect();
        x.permute(&axes).map_err(|e| NnError::OnnxError(format!("{e}")))?
    } else {
        #[allow(clippy::cast_sign_loss)]
        let axes: Vec<usize> = perm.iter().map(|&p| p as usize).collect();
        x.permute(&axes).map_err(|e| NnError::OnnxError(format!("{e}")))?
    };

    set_output(node, env, 0, result)
}

fn exec_flatten<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
) -> Result<()> {
    let x = get_input(node, env, 0)?;
    let axis = node.get_int_attr("axis", 1);

    let shape = x.shape().to_vec();
    let ndim = shape.len();

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    let axis_usize = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };

    let outer: usize = if axis_usize == 0 {
        1
    } else {
        shape[..axis_usize].iter().product()
    };
    let inner: usize = shape[axis_usize..].iter().product();

    let result = x.reshape(vec![outer, inner]).map_err(|e| NnError::OnnxError(format!("{e}")))?;
    set_output(node, env, 0, result)
}

fn exec_concat<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
) -> Result<()> {
    let axis = node.get_int_attr("axis", 0);

    let mut tensors = Vec::new();
    for i in 0..node.inputs.len() {
        tensors.push(get_input(node, env, i)?);
    }

    if tensors.is_empty() {
        return Err(NnError::OnnxError("Concat: no inputs".into()));
    }

    let ndim = tensors[0].ndim();
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    let axis_usize = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };

    let refs: Vec<&Tensor<T>> = tensors.iter().collect();
    let result = Tensor::concat(&refs, axis_usize).map_err(|e| NnError::OnnxError(format!("{e}")))?;
    set_output(node, env, 0, result)
}

fn exec_batchnorm<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
) -> Result<()> {
    let x = get_input(node, env, 0)?;
    let scale = get_input(node, env, 1)?;
    let bias = get_input(node, env, 2)?;
    let mean = get_input(node, env, 3)?;
    let var = get_input(node, env, 4)?;

    let epsilon = T::from_f64(f64::from(node.get_float_attr("epsilon", 1e-5)));

    let shape = x.shape().to_vec();
    if shape.len() < 2 {
        return Err(NnError::OnnxError(
            "BatchNormalization: input must have at least 2 dimensions".into(),
        ));
    }

    let channels = shape[1];
    let scale_data = scale.as_slice();
    let bias_data = bias.as_slice();
    let mean_data = mean.as_slice();
    let var_data = var.as_slice();

    let x_data = x.as_slice();
    let mut out = vec![T::zero(); x_data.len()];

    let spatial: usize = shape[2..].iter().product();
    let batch = shape[0];

    for b in 0..batch {
        for c in 0..channels {
            let s = scale_data[c];
            let bi = bias_data[c];
            let m = mean_data[c];
            let v = var_data[c];
            let inv_std = T::one() / (v + epsilon).sqrt();

            for sp in 0..spatial {
                let idx = b * channels * spatial + c * spatial + sp;
                out[idx] = (x_data[idx] - m) * inv_std * s + bi;
            }
        }
    }

    let result = Tensor::from_vec(out, shape).map_err(|e| NnError::OnnxError(format!("{e}")))?;
    set_output(node, env, 0, result)
}

fn exec_dropout<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
) -> Result<()> {
    let x = get_input(node, env, 0)?;
    set_output(node, env, 0, x)
}

fn exec_unsqueeze<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
) -> Result<()> {
    let x = get_input(node, env, 0)?;

    let axes: Vec<i64> = if node.inputs.len() > 1 {
        if let Ok(axes_tensor) = get_input::<T>(node, env, 1) {
            #[allow(clippy::cast_possible_truncation)]
            let v: Vec<i64> = axes_tensor.as_slice().iter().map(|&v| v.to_f64() as i64).collect();
            v
        } else {
            node.get_ints_attr("axes")
        }
    } else {
        node.get_ints_attr("axes")
    };

    let mut shape = x.shape().to_vec();
    let mut sorted_axes = axes;
    sorted_axes.sort_unstable();

    for &axis in &sorted_axes {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
        let a = if axis < 0 {
            (shape.len() as i64 + 1 + axis) as usize
        } else {
            axis as usize
        };
        shape.insert(a, 1);
    }

    let result = x.reshape(shape).map_err(|e| NnError::OnnxError(format!("{e}")))?;
    set_output(node, env, 0, result)
}

fn exec_squeeze<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
) -> Result<()> {
    let x = get_input(node, env, 0)?;

    let axes: Vec<i64> = if node.inputs.len() > 1 {
        if let Ok(axes_tensor) = get_input::<T>(node, env, 1) {
            #[allow(clippy::cast_possible_truncation)]
            let v: Vec<i64> = axes_tensor.as_slice().iter().map(|&v| v.to_f64() as i64).collect();
            v
        } else {
            node.get_ints_attr("axes")
        }
    } else {
        node.get_ints_attr("axes")
    };

    let mut shape = x.shape().to_vec();
    if axes.is_empty() {
        shape.retain(|&d| d != 1);
    } else {
        let ndim = shape.len();
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
        let mut indices: Vec<usize> = axes.iter().map(|&a| {
            if a < 0 { (ndim as i64 + a) as usize } else { a as usize }
        }).collect();
        indices.sort_unstable();
        indices.reverse();
        for i in indices {
            if i < shape.len() && shape[i] == 1 {
                shape.remove(i);
            }
        }
    }

    if shape.is_empty() {
        shape.push(1);
    }

    let result = x.reshape(shape).map_err(|e| NnError::OnnxError(format!("{e}")))?;
    set_output(node, env, 0, result)
}

#[allow(clippy::too_many_lines)]
fn exec_conv<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
) -> Result<()> {
    let x = get_input(node, env, 0)?;
    let w = get_input(node, env, 1)?;

    let x_shape = x.shape().to_vec();
    let w_shape = w.shape().to_vec();

    if x_shape.len() != 4 || w_shape.len() != 4 {
        return Err(NnError::OnnxError(
            "Conv: only 2-D convolution (NCHW) is supported".into(),
        ));
    }

    let batch = x_shape[0];
    let c_in = x_shape[1];
    let h_in = x_shape[2];
    let w_in = x_shape[3];

    let c_out = w_shape[0];
    let c_per_group = w_shape[1];
    let kh = w_shape[2];
    let kw = w_shape[3];

    let strides_attr = node.get_ints_attr("strides");
    #[allow(clippy::cast_sign_loss)]
    let sh = if strides_attr.len() >= 2 { strides_attr[0] as usize } else { 1 };
    #[allow(clippy::cast_sign_loss)]
    let sw = if strides_attr.len() >= 2 { strides_attr[1] as usize } else { 1 };

    let pads_attr = node.get_ints_attr("pads");
    #[allow(clippy::cast_sign_loss)]
    let (ph_begin, pw_begin) = if pads_attr.len() >= 4 {
        (pads_attr[0] as usize, pads_attr[1] as usize)
    } else {
        (0, 0)
    };

    #[allow(clippy::cast_sign_loss)]
    let group = node.get_int_attr("group", 1) as usize;

    let h_out = (h_in + 2 * ph_begin - kh) / sh + 1;
    let w_out = (w_in + 2 * pw_begin - kw) / sw + 1;

    let x_data = x.as_slice();
    let w_data = w.as_slice();
    let mut out = vec![T::zero(); batch * c_out * h_out * w_out];

    let c_out_per_group = c_out / group;

    for b in 0..batch {
        for g in 0..group {
            for oc in 0..c_out_per_group {
                let abs_oc = g * c_out_per_group + oc;
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = T::zero();
                        for ic in 0..c_per_group {
                            let abs_ic = g * c_per_group + ic;
                            for fh in 0..kh {
                                for fw in 0..kw {
                                    let ih = oh * sh + fh;
                                    let iw = ow * sw + fw;
                                    if ih >= ph_begin && iw >= pw_begin {
                                        let ih_real = ih - ph_begin;
                                        let iw_real = iw - pw_begin;
                                        if ih_real < h_in && iw_real < w_in {
                                            let x_idx = b * c_in * h_in * w_in
                                                + abs_ic * h_in * w_in
                                                + ih_real * w_in
                                                + iw_real;
                                            let w_idx = abs_oc * c_per_group * kh * kw
                                                + ic * kh * kw
                                                + fh * kw
                                                + fw;
                                            sum += x_data[x_idx] * w_data[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                        let out_idx =
                            b * c_out * h_out * w_out + abs_oc * h_out * w_out + oh * w_out + ow;
                        out[out_idx] = sum;
                    }
                }
            }
        }
    }

    // Add bias if provided.
    #[allow(clippy::collapsible_if)]
    if node.inputs.len() > 2 {
        if let Ok(bias) = get_input::<T>(node, env, 2) {
            let bias_data = bias.as_slice();
            for b in 0..batch {
                for (c, &bval) in bias_data.iter().enumerate().take(c_out) {
                    for h in 0..h_out {
                        for w_i in 0..w_out {
                            let idx = b * c_out * h_out * w_out + c * h_out * w_out + h * w_out + w_i;
                            out[idx] += bval;
                        }
                    }
                }
            }
        }
    }

    let result = Tensor::from_vec(out, vec![batch, c_out, h_out, w_out])
        .map_err(|e| NnError::OnnxError(format!("{e}")))?;
    set_output(node, env, 0, result)
}

#[allow(clippy::too_many_lines)]
fn exec_maxpool<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
) -> Result<()> {
    let x = get_input(node, env, 0)?;
    let x_shape = x.shape().to_vec();

    if x_shape.len() != 4 {
        return Err(NnError::OnnxError(
            "MaxPool: only 2-D pooling (NCHW) is supported".into(),
        ));
    }

    let kernel_shape = node.get_ints_attr("kernel_shape");
    if kernel_shape.len() != 2 {
        return Err(NnError::OnnxError(
            "MaxPool: kernel_shape must have 2 elements".into(),
        ));
    }

    #[allow(clippy::cast_sign_loss)]
    let kh = kernel_shape[0] as usize;
    #[allow(clippy::cast_sign_loss)]
    let kw = kernel_shape[1] as usize;

    let strides_attr = node.get_ints_attr("strides");
    #[allow(clippy::cast_sign_loss)]
    let sh = if strides_attr.len() >= 2 { strides_attr[0] as usize } else { 1 };
    #[allow(clippy::cast_sign_loss)]
    let sw = if strides_attr.len() >= 2 { strides_attr[1] as usize } else { 1 };

    let pads_attr = node.get_ints_attr("pads");
    #[allow(clippy::cast_sign_loss)]
    let (ph, pw) = if pads_attr.len() >= 4 {
        (pads_attr[0] as usize, pads_attr[1] as usize)
    } else {
        (0, 0)
    };

    let batch = x_shape[0];
    let channels = x_shape[1];
    let h_in = x_shape[2];
    let w_in = x_shape[3];
    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let x_data = x.as_slice();
    let mut out = vec![T::neg_infinity(); batch * channels * h_out * w_out];

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = T::neg_infinity();
                    for fh in 0..kh {
                        for fw in 0..kw {
                            let ih = oh * sh + fh;
                            let iw = ow * sw + fw;
                            if ih >= ph && iw >= pw {
                                let ih_real = ih - ph;
                                let iw_real = iw - pw;
                                if ih_real < h_in && iw_real < w_in {
                                    let idx = b * channels * h_in * w_in
                                        + c * h_in * w_in
                                        + ih_real * w_in
                                        + iw_real;
                                    max_val = max_val.max(x_data[idx]);
                                }
                            }
                        }
                    }
                    let out_idx =
                        b * channels * h_out * w_out + c * h_out * w_out + oh * w_out + ow;
                    out[out_idx] = max_val;
                }
            }
        }
    }

    let result = Tensor::from_vec(out, vec![batch, channels, h_out, w_out])
        .map_err(|e| NnError::OnnxError(format!("{e}")))?;
    set_output(node, env, 0, result)
}

#[allow(clippy::too_many_lines)]
fn exec_avgpool<T: Float>(
    node: &OnnxNode,
    env: &mut HashMap<String, Tensor<T>>,
) -> Result<()> {
    let x = get_input(node, env, 0)?;
    let x_shape = x.shape().to_vec();

    if x_shape.len() != 4 {
        return Err(NnError::OnnxError(
            "AveragePool: only 2-D pooling (NCHW) is supported".into(),
        ));
    }

    let kernel_shape = node.get_ints_attr("kernel_shape");
    if kernel_shape.len() != 2 {
        return Err(NnError::OnnxError(
            "AveragePool: kernel_shape must have 2 elements".into(),
        ));
    }

    #[allow(clippy::cast_sign_loss)]
    let kh = kernel_shape[0] as usize;
    #[allow(clippy::cast_sign_loss)]
    let kw = kernel_shape[1] as usize;

    let strides_attr = node.get_ints_attr("strides");
    #[allow(clippy::cast_sign_loss)]
    let sh = if strides_attr.len() >= 2 { strides_attr[0] as usize } else { 1 };
    #[allow(clippy::cast_sign_loss)]
    let sw = if strides_attr.len() >= 2 { strides_attr[1] as usize } else { 1 };

    let pads_attr = node.get_ints_attr("pads");
    #[allow(clippy::cast_sign_loss)]
    let (ph, pw) = if pads_attr.len() >= 4 {
        (pads_attr[0] as usize, pads_attr[1] as usize)
    } else {
        (0, 0)
    };

    let batch = x_shape[0];
    let channels = x_shape[1];
    let h_in = x_shape[2];
    let w_in = x_shape[3];
    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let x_data = x.as_slice();
    let pool_size = T::from_usize(kh * kw);
    let mut out = vec![T::zero(); batch * channels * h_out * w_out];

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = T::zero();
                    for fh in 0..kh {
                        for fw in 0..kw {
                            let ih = oh * sh + fh;
                            let iw = ow * sw + fw;
                            if ih >= ph && iw >= pw {
                                let ih_real = ih - ph;
                                let iw_real = iw - pw;
                                if ih_real < h_in && iw_real < w_in {
                                    let idx = b * channels * h_in * w_in
                                        + c * h_in * w_in
                                        + ih_real * w_in
                                        + iw_real;
                                    sum += x_data[idx];
                                }
                            }
                        }
                    }
                    let out_idx =
                        b * channels * h_out * w_out + c * h_out * w_out + oh * w_out + ow;
                    out[out_idx] = sum / pool_size;
                }
            }
        }
    }

    let result = Tensor::from_vec(out, vec![batch, channels, h_out, w_out])
        .map_err(|e| NnError::OnnxError(format!("{e}")))?;
    set_output(node, env, 0, result)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::onnx::ir::*;

    /// Helper to build a simple graph and run it.
    fn run_graph(
        nodes: Vec<OnnxNode>,
        initializers: Vec<OnnxTensor>,
        inputs: &[(&str, Tensor<f32>)],
        input_infos: Vec<OnnxValueInfo>,
        output_names: &[&str],
    ) -> Result<Vec<Tensor<f32>>> {
        let graph = OnnxGraph {
            name: "test".into(),
            nodes,
            initializers,
            inputs: input_infos,
            outputs: output_names
                .iter()
                .map(|&n| OnnxValueInfo {
                    name: n.to_owned(),
                    data_type: OnnxDataType::Float,
                    shape: vec![],
                })
                .collect(),
        };
        let model = OnnxModel {
            ir_version: 7,
            opset_imports: vec![],
            graph,
            producer_name: "test".into(),
            model_version: 1,
        };
        let session = OnnxInferenceSession::<f32>::from_model(model)?;
        session.run(inputs)
    }

    fn make_initializer(name: &str, data: Vec<f32>, dims: Vec<i64>) -> OnnxTensor {
        OnnxTensor {
            name: name.to_owned(),
            data_type: OnnxDataType::Float,
            dims,
            float_data: data,
            double_data: vec![],
            int32_data: vec![],
            int64_data: vec![],
            raw_data: vec![],
        }
    }

    fn make_node(op: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> OnnxNode {
        OnnxNode {
            op_type: op.to_owned(),
            inputs: inputs.into_iter().map(String::from).collect(),
            outputs: outputs.into_iter().map(String::from).collect(),
            name: String::new(),
            attributes: vec![],
        }
    }

    fn input_info(name: &str) -> OnnxValueInfo {
        OnnxValueInfo {
            name: name.to_owned(),
            data_type: OnnxDataType::Float,
            shape: vec![],
        }
    }

    #[test]
    fn test_add_node() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![10.0_f32, 20.0, 30.0], vec![3]).unwrap();

        let node = make_node("Add", vec!["A", "B"], vec!["C"]);
        let results = run_graph(
            vec![node],
            vec![],
            &[("A", a), ("B", b)],
            vec![input_info("A"), input_info("B")],
            &["C"],
        )
        .unwrap();

        assert_eq!(results[0].as_slice(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_matmul_node() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_vec(vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();

        let node = make_node("MatMul", vec!["A", "B"], vec!["C"]);
        let results = run_graph(
            vec![node],
            vec![],
            &[("A", a), ("B", b)],
            vec![input_info("A"), input_info("B")],
            &["C"],
        )
        .unwrap();

        assert_eq!(results[0].shape(), &[2, 2]);
        assert_eq!(results[0].as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_relu_node() {
        let x = Tensor::from_vec(vec![-1.0_f32, 0.0, 1.0, -0.5, 2.0], vec![5]).unwrap();
        let node = make_node("Relu", vec!["X"], vec!["Y"]);
        let results = run_graph(
            vec![node],
            vec![],
            &[("X", x)],
            vec![input_info("X")],
            &["Y"],
        )
        .unwrap();
        assert_eq!(results[0].as_slice(), &[0.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_multi_node_chain() {
        let x = Tensor::from_vec(vec![1.0_f32, -1.0], vec![1, 2]).unwrap();
        let w = make_initializer("W", vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let b_init = make_initializer("bias", vec![-0.5, 0.5], vec![2]);

        let matmul = make_node("MatMul", vec!["X", "W"], vec!["mm_out"]);
        let add = make_node("Add", vec!["mm_out", "bias"], vec!["add_out"]);
        let relu = make_node("Relu", vec!["add_out"], vec!["Y"]);

        let results = run_graph(
            vec![matmul, add, relu],
            vec![w, b_init],
            &[("X", x)],
            vec![input_info("X")],
            &["Y"],
        )
        .unwrap();

        assert_eq!(results[0].shape(), &[1, 2]);
        assert_eq!(results[0].as_slice(), &[0.5, 0.0]);
    }

    #[test]
    fn test_gemm_node() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b_init = make_initializer("B", vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c_init = make_initializer("C", vec![1.0, 1.0], vec![2]);

        let mut node = make_node("Gemm", vec!["A", "B", "C"], vec!["Y"]);
        node.attributes.push(OnnxAttribute {
            name: "alpha".into(),
            value: OnnxAttributeValue::Float(1.0),
        });
        node.attributes.push(OnnxAttribute {
            name: "beta".into(),
            value: OnnxAttributeValue::Float(1.0),
        });

        let results = run_graph(
            vec![node],
            vec![b_init, c_init],
            &[("A", a)],
            vec![input_info("A")],
            &["Y"],
        )
        .unwrap();

        assert_eq!(results[0].shape(), &[2, 2]);
        assert_eq!(results[0].as_slice(), &[20.0, 23.0, 44.0, 51.0]);
    }

    #[test]
    fn test_reshape_transpose() {
        let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]).unwrap();
        let shape = make_initializer("shape", vec![2.0, 3.0], vec![2]);

        let reshape = make_node("Reshape", vec!["X", "shape"], vec!["reshaped"]);
        let mut transpose = make_node("Transpose", vec!["reshaped"], vec!["Y"]);
        transpose.attributes.push(OnnxAttribute {
            name: "perm".into(),
            value: OnnxAttributeValue::Ints(vec![1, 0]),
        });

        let results = run_graph(
            vec![reshape, transpose],
            vec![shape],
            &[("X", x)],
            vec![input_info("X")],
            &["Y"],
        )
        .unwrap();

        assert_eq!(results[0].shape(), &[3, 2]);
        assert_eq!(results[0].as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_softmax() {
        let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], vec![1, 3]).unwrap();

        let mut node = make_node("Softmax", vec!["X"], vec!["Y"]);
        node.attributes.push(OnnxAttribute {
            name: "axis".into(),
            value: OnnxAttributeValue::Int(1),
        });

        let results = run_graph(
            vec![node],
            vec![],
            &[("X", x)],
            vec![input_info("X")],
            &["Y"],
        )
        .unwrap();

        let out = results[0].as_slice();
        assert_eq!(out.len(), 3);

        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(out[0] < out[1]);
        assert!(out[1] < out[2]);
    }

    #[test]
    fn test_batchnorm() {
        let x = Tensor::from_vec(vec![2.0_f32, 4.0], vec![1, 2, 1, 1]).unwrap();
        let scale = make_initializer("scale", vec![1.0, 1.0], vec![2]);
        let bias = make_initializer("bias", vec![0.0, 0.0], vec![2]);
        let mean = make_initializer("mean", vec![0.0, 0.0], vec![2]);
        let var = make_initializer("var", vec![1.0, 1.0], vec![2]);

        let node = make_node(
            "BatchNormalization",
            vec!["X", "scale", "bias", "mean", "var"],
            vec!["Y"],
        );

        let results = run_graph(
            vec![node],
            vec![scale, bias, mean, var],
            &[("X", x)],
            vec![input_info("X")],
            &["Y"],
        )
        .unwrap();

        let out = results[0].as_slice();
        assert!((out[0] - 2.0).abs() < 0.01);
        assert!((out[1] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_full_mini_model() {
        let x = Tensor::from_vec(vec![1.0_f32, 2.0], vec![1, 2]).unwrap();

        let w1 = make_initializer("W1", vec![0.5, -0.5, 1.0, 0.0, -1.0, 1.0], vec![3, 2]);
        let b1 = make_initializer("b1", vec![0.1, 0.2, 0.3], vec![3]);
        let w2 = make_initializer("W2", vec![1.0, 2.0, 3.0], vec![1, 3]);
        let b2 = make_initializer("b2", vec![0.5], vec![1]);

        let mut gemm1 = make_node("Gemm", vec!["X", "W1", "b1"], vec!["gemm1_out"]);
        gemm1.attributes.push(OnnxAttribute {
            name: "transB".into(),
            value: OnnxAttributeValue::Int(1),
        });

        let relu = make_node("Relu", vec!["gemm1_out"], vec!["relu_out"]);

        let mut gemm2 = make_node("Gemm", vec!["relu_out", "W2", "b2"], vec!["Y"]);
        gemm2.attributes.push(OnnxAttribute {
            name: "transB".into(),
            value: OnnxAttributeValue::Int(1),
        });

        let results = run_graph(
            vec![gemm1, relu, gemm2],
            vec![w1, b1, w2, b2],
            &[("X", x)],
            vec![input_info("X")],
            &["Y"],
        )
        .unwrap();

        assert_eq!(results[0].shape(), &[1, 1]);

        let out_val = results[0].as_slice()[0];
        assert!((out_val - 6.8).abs() < 1e-4, "got {out_val}, expected 6.8");
    }
}
