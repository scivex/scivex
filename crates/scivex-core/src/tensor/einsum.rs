//! Einstein summation convention for tensor contractions.
//!
//! Implements the `einsum` function, which performs tensor contractions using
//! Einstein notation. The subscript string describes which indices to contract
//! over, enabling concise expression of a wide range of tensor operations.
//!
//! # Examples
//!
//! ```
//! # use scivex_core::Tensor;
//! # use scivex_core::tensor::einsum::einsum;
//! // Matrix multiply: C = A @ B
//! let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
//! let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
//! assert_eq!(c.shape(), &[2, 2]);
//! assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
//! ```

use crate::Scalar;
use crate::error::{CoreError, Result};
use crate::tensor::Tensor;
use std::collections::BTreeMap;

/// Parsed representation of an einsum subscript string.
struct EinsumParsed {
    /// Index labels for each input operand.
    input_indices: Vec<Vec<char>>,
    /// Index labels for the output.
    output_indices: Vec<char>,
    /// All unique index labels that appear in input but not output (contraction indices).
    contraction_indices: Vec<char>,
    /// Mapping from index label to its dimension size.
    index_sizes: BTreeMap<char, usize>,
}

/// Perform Einstein summation on the given operands.
///
/// The `subscripts` string follows NumPy-style Einstein notation:
/// - Input operand index lists are separated by commas.
/// - An optional `->` separates input indices from output indices.
/// - If `->` is omitted (implicit mode), repeated indices are summed over
///   and the output contains the remaining indices in sorted order.
///
/// # Supported operations
///
/// | Subscript | Operation |
/// |-----------|-----------|
/// | `"ij,jk->ik"` | Matrix multiply |
/// | `"ii->"` | Trace |
/// | `"ij->ji"` | Transpose |
/// | `"ij->"` | Sum all elements |
/// | `"i,i->"` | Dot product |
/// | `"i,j->ij"` | Outer product |
/// | `"ijk,ikl->ijl"` | Batched matmul |
/// | `"ij,j->i"` | Matrix-vector product |
///
/// # Errors
///
/// Returns [`CoreError::InvalidArgument`] if the subscript string is malformed
/// or the number of operands does not match the subscripts.
/// Returns [`CoreError::DimensionMismatch`] if an index label corresponds to
/// different sizes across operands.
///
/// # Examples
///
/// ```
/// # use scivex_core::Tensor;
/// # use scivex_core::tensor::einsum::einsum;
/// let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
/// let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
/// let dot = einsum("i,i->", &[&a, &b]).unwrap();
/// assert_eq!(dot.as_slice(), &[32.0]);
/// ```
pub fn einsum<T: Scalar>(subscripts: &str, operands: &[&Tensor<T>]) -> Result<Tensor<T>> {
    let parsed = parse_subscripts(subscripts, operands)?;
    execute_einsum(&parsed, operands)
}

/// Parse the subscript string and validate against the provided operands.
fn parse_subscripts<T: Scalar>(subscripts: &str, operands: &[&Tensor<T>]) -> Result<EinsumParsed> {
    let subscripts = subscripts.replace(' ', "");

    // Split into input and output parts.
    let (input_str, output_indices) = if let Some((inp, out)) = subscripts.split_once("->") {
        // Explicit mode: output indices are specified.
        let out_indices: Vec<char> = out.chars().collect();
        // Validate output indices are alphabetic.
        for &c in &out_indices {
            if !c.is_ascii_alphabetic() {
                return Err(CoreError::InvalidArgument {
                    reason: "output subscript indices must be ASCII letters",
                });
            }
        }
        (inp.to_string(), out_indices)
    } else {
        // Implicit mode: sum over repeated indices, output has sorted unique non-repeated indices.
        let inp = subscripts.clone();
        let mut counts: BTreeMap<char, usize> = BTreeMap::new();
        for c in inp.chars() {
            if c == ',' {
                continue;
            }
            if !c.is_ascii_alphabetic() {
                return Err(CoreError::InvalidArgument {
                    reason: "subscript indices must be ASCII letters",
                });
            }
            *counts.entry(c).or_insert(0) += 1;
        }
        // Output indices are those that appear exactly once, in sorted order.
        let out_indices: Vec<char> = counts
            .iter()
            .filter(|(_, count)| **count == 1)
            .map(|(&c, _)| c)
            .collect();
        (inp, out_indices)
    };

    // Parse input operand indices.
    let input_parts: Vec<&str> = input_str.split(',').collect();
    if input_parts.len() != operands.len() {
        return Err(CoreError::InvalidArgument {
            reason: "number of subscript operands does not match number of tensors",
        });
    }

    let mut input_indices: Vec<Vec<char>> = Vec::with_capacity(input_parts.len());
    for part in &input_parts {
        let indices: Vec<char> = part.chars().collect();
        for &c in &indices {
            if !c.is_ascii_alphabetic() {
                return Err(CoreError::InvalidArgument {
                    reason: "subscript indices must be ASCII letters",
                });
            }
        }
        input_indices.push(indices);
    }

    // Validate that each operand's ndim matches its index count.
    for (i, indices) in input_indices.iter().enumerate() {
        if indices.len() != operands[i].ndim() {
            return Err(CoreError::InvalidArgument {
                reason: "operand rank does not match number of subscript indices",
            });
        }
    }

    // Build index-to-size mapping and check consistency.
    let mut index_sizes: BTreeMap<char, usize> = BTreeMap::new();
    for (op_idx, indices) in input_indices.iter().enumerate() {
        let shape = operands[op_idx].shape();
        for (dim_idx, &c) in indices.iter().enumerate() {
            let size = shape[dim_idx];
            if let Some(&existing) = index_sizes.get(&c) {
                if existing != size {
                    return Err(CoreError::DimensionMismatch {
                        expected: vec![existing],
                        got: vec![size],
                    });
                }
            } else {
                index_sizes.insert(c, size);
            }
        }
    }

    // Validate that all output indices appear in the inputs.
    for &c in &output_indices {
        if !index_sizes.contains_key(&c) {
            return Err(CoreError::InvalidArgument {
                reason: "output index not found in any input operand",
            });
        }
    }

    // Contraction indices: appear in inputs but not in output.
    let contraction_indices: Vec<char> = index_sizes
        .keys()
        .filter(|c| !output_indices.contains(c))
        .copied()
        .collect();

    Ok(EinsumParsed {
        input_indices,
        output_indices,
        contraction_indices,
        index_sizes,
    })
}

/// Per-operand mapping from dimensions to output/contraction index positions.
struct OperandInfo {
    /// For each dimension of this operand, which index in our iteration
    /// order does it correspond to? We store (source, position) where
    /// source=0 means output_indices, source=1 means contraction_indices.
    dim_map: Vec<(usize, usize)>,
}

/// Execute the einsum operation described by `parsed` on the given `operands`.
fn execute_einsum<T: Scalar>(parsed: &EinsumParsed, operands: &[&Tensor<T>]) -> Result<Tensor<T>> {
    // Compute output shape.
    let output_shape: Vec<usize> = parsed
        .output_indices
        .iter()
        .map(|c| parsed.index_sizes[c])
        .collect();
    let output_numel: usize = if output_shape.is_empty() {
        1
    } else {
        output_shape.iter().product()
    };

    // Compute contraction range sizes.
    let contraction_sizes: Vec<usize> = parsed
        .contraction_indices
        .iter()
        .map(|c| parsed.index_sizes[c])
        .collect();
    let contraction_numel: usize = if contraction_sizes.is_empty() {
        1
    } else {
        contraction_sizes.iter().product()
    };

    // All unique indices: output indices first, then contraction indices.
    // We iterate over output indices in the outer loop and contraction indices in the inner loop.
    let mut result_data = vec![T::zero(); output_numel];

    // Precompute: for each operand, build a mapping from operand dimension to
    // (is_output, position_in_output_or_contraction).
    // This tells us where to look up each operand's index value.
    let operand_infos: Vec<OperandInfo> = parsed
        .input_indices
        .iter()
        .map(|indices| {
            let dim_map = indices
                .iter()
                .map(|c| {
                    if let Some(pos) = parsed.output_indices.iter().position(|oc| oc == c) {
                        (0, pos)
                    } else {
                        let pos = parsed
                            .contraction_indices
                            .iter()
                            .position(|cc| cc == c)
                            .expect("index must be in output or contraction");
                        (1, pos)
                    }
                })
                .collect();
            OperandInfo { dim_map }
        })
        .collect();

    // Iterate over all output index combinations.
    for (out_flat, result_elem) in result_data.iter_mut().enumerate() {
        // Decode output flat index into multi-index.
        let out_multi = flat_to_multi(out_flat, &output_shape);

        let mut sum = T::zero();

        // Iterate over all contraction index combinations.
        for contr_flat in 0..contraction_numel {
            let contr_multi = flat_to_multi(contr_flat, &contraction_sizes);

            // Compute the product of all operand elements at the current index combination.
            let mut product = T::one();
            for (op_idx, info) in operand_infos.iter().enumerate() {
                let operand = operands[op_idx];
                let op_shape = operand.shape();
                let mut op_index = vec![0usize; op_shape.len()];
                for (dim, &(source, pos)) in info.dim_map.iter().enumerate() {
                    op_index[dim] = if source == 0 {
                        out_multi[pos]
                    } else {
                        contr_multi[pos]
                    };
                }
                // We already validated shapes so this should not fail.
                let val = *operand.get(&op_index)?;
                product *= val;
            }
            sum += product;
        }

        *result_elem = sum;
    }

    Tensor::from_vec(result_data, output_shape)
}

/// Convert a flat index to a multi-dimensional index given a shape.
/// Returns an empty vec for scalar (empty shape).
fn flat_to_multi(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut multi = vec![0usize; shape.len()];
    for i in (0..shape.len()).rev() {
        multi[i] = flat % shape[i];
        flat /= shape[i];
    }
    multi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_einsum_matmul() {
        // "ij,jk->ik" — matrix multiply
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_einsum_trace() {
        // "ii->" — trace
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let t = einsum("ii->", &[&a]).unwrap();
        assert_eq!(t.shape(), &[] as &[usize]);
        assert_eq!(t.as_slice(), &[5.0]);
    }

    #[test]
    fn test_einsum_transpose() {
        // "ij->ji" — transpose
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let t = einsum("ij->ji", &[&a]).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_einsum_dot_product() {
        // "i,i->" — dot product
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        let d = einsum("i,i->", &[&a, &b]).unwrap();
        assert_eq!(d.as_slice(), &[32.0]);
    }

    #[test]
    fn test_einsum_outer_product() {
        // "i,j->ij" — outer product
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0], vec![2]).unwrap();
        let o = einsum("i,j->ij", &[&a, &b]).unwrap();
        assert_eq!(o.shape(), &[3, 2]);
        assert_eq!(o.as_slice(), &[4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    }

    #[test]
    fn test_einsum_batched_matmul() {
        // "ijk,ikl->ijl" — batched matmul
        // Batch size 2, 2x2 matrices each.
        #[rustfmt::skip]
        let a = Tensor::from_vec(
            vec![
                // batch 0: [[1,2],[3,4]]
                1.0_f64, 2.0, 3.0, 4.0,
                // batch 1: [[5,6],[7,8]]
                5.0, 6.0, 7.0, 8.0,
            ],
            vec![2, 2, 2],
        )
        .unwrap();
        #[rustfmt::skip]
        let b = Tensor::from_vec(
            vec![
                // batch 0: [[1,0],[0,1]]
                1.0_f64, 0.0, 0.0, 1.0,
                // batch 1: [[2,0],[0,2]]
                2.0, 0.0, 0.0, 2.0,
            ],
            vec![2, 2, 2],
        )
        .unwrap();
        let c = einsum("ijk,ikl->ijl", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);
        // batch 0: [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
        // batch 1: [[5,6],[7,8]] @ [[2,0],[0,2]] = [[10,12],[14,16]]
        assert_eq!(c.as_slice(), &[1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn test_einsum_matvec() {
        // "ij,j->i" — matrix-vector product
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let x = Tensor::from_vec(vec![5.0, 6.0], vec![2]).unwrap();
        let y = einsum("ij,j->i", &[&a, &x]).unwrap();
        assert_eq!(y.shape(), &[2]);
        assert_eq!(y.as_slice(), &[17.0, 39.0]);
    }

    #[test]
    fn test_einsum_sum_all() {
        // "ij->" — sum all elements
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let s = einsum("ij->", &[&a]).unwrap();
        assert_eq!(s.as_slice(), &[10.0]);
    }

    #[test]
    fn test_einsum_implicit_mode() {
        // No "->" — implicit: repeated index 'j' is summed, output is sorted unique = "ik"
        // Equivalent to "ij,jk->ik"
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = einsum("ij,jk", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_einsum_error_wrong_operand_count() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = einsum("ij,jk->ik", &[&a]);
        assert!(result.is_err());
    }

    #[test]
    fn test_einsum_error_inconsistent_dimensions() {
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        // j=3 in a but j=2 in b
        let result = einsum("ij,jk->ik", &[&a, &b]);
        assert!(result.is_err());
    }
}
