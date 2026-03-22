//! Einsum contraction path optimizer.
//!
//! Determines the optimal order in which to contract pairs of tensors in a
//! multi-operand einsum expression. This can dramatically reduce the number
//! of floating-point operations and intermediate memory usage compared to
//! contracting all tensors at once.
//!
//! Two strategies are provided:
//! - **Greedy**: O(n^3) heuristic that picks the cheapest pairwise contraction
//!   at each step. Works well in practice and is the default.
//! - **Optimal**: Exhaustive search over all contraction orderings. Guaranteed
//!   to find the minimum-FLOP path but is O(n!) and only practical for small
//!   numbers of operands (≤ ~6).
//!
//! # Examples
//!
//! ```
//! # use scivex_core::Tensor;
//! # use scivex_core::tensor::einsum_path::{einsum_path, PathStrategy, einsum_optimized};
//! let a = Tensor::from_vec(vec![1.0_f64; 6], vec![2, 3]).unwrap();
//! let b = Tensor::from_vec(vec![1.0_f64; 12], vec![3, 4]).unwrap();
//! let c = Tensor::from_vec(vec![1.0_f64; 20], vec![4, 5]).unwrap();
//!
//! // Get the contraction path
//! let info = einsum_path("ij,jk,kl->il", &[&a, &b, &c], PathStrategy::Greedy).unwrap();
//! assert_eq!(info.path.len(), 2); // Two pairwise contractions
//!
//! // Use the optimized einsum
//! let result = einsum_optimized("ij,jk,kl->il", &[&a, &b, &c], PathStrategy::Greedy).unwrap();
//! assert_eq!(result.shape(), &[2, 5]);
//! ```

use crate::Scalar;
use crate::error::{CoreError, Result};
use crate::tensor::Tensor;
use std::collections::{BTreeMap, BTreeSet};

use super::einsum::einsum;

/// Strategy for finding the contraction path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathStrategy {
    /// Greedy heuristic: at each step, contract the pair with the smallest
    /// intermediate tensor. O(n^3) in the number of operands.
    Greedy,
    /// Exhaustive search: tries all possible contraction orderings and picks
    /// the one with the lowest total FLOP count. O(n!) — only practical for
    /// ≤ ~6 operands.
    Optimal,
}

/// A single contraction step: contract operands at positions `(i, j)` in the
/// current operand list (where `i < j`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractionPair {
    /// First operand index (in the current list at this step).
    pub first: usize,
    /// Second operand index (in the current list at this step).
    pub second: usize,
}

/// Information about the chosen contraction path.
#[derive(Debug, Clone)]
pub struct PathInfo {
    /// Sequence of pairwise contractions. Each pair refers to indices in the
    /// operand list *at that step* (operands are removed and the intermediate
    /// result is appended after each step).
    pub path: Vec<ContractionPair>,
    /// Estimated total number of multiply-add operations (FLOPs).
    pub flops: usize,
    /// Size of the largest intermediate tensor (in number of elements).
    pub largest_intermediate: usize,
}

/// Internal representation of an operand's index structure.
#[derive(Debug, Clone)]
struct OperandDesc {
    /// Index labels for this operand.
    indices: Vec<char>,
}

/// Parsed subscript components.
type ParsedSubscripts = (Vec<Vec<char>>, Vec<char>, BTreeMap<char, usize>);

/// Compute the contraction path for an einsum expression.
///
/// Returns a [`PathInfo`] describing the order in which to contract pairs of
/// operands, along with estimated FLOPs and intermediate sizes.
pub fn einsum_path<T: Scalar>(
    subscripts: &str,
    operands: &[&Tensor<T>],
    strategy: PathStrategy,
) -> Result<PathInfo> {
    let (input_subs, output_sub, index_sizes) = parse_path_subscripts(subscripts, operands)?;

    let descs: Vec<OperandDesc> = input_subs
        .iter()
        .map(|indices| OperandDesc {
            indices: indices.clone(),
        })
        .collect();

    match strategy {
        PathStrategy::Greedy => greedy_path(&descs, &output_sub, &index_sizes),
        PathStrategy::Optimal => optimal_path(&descs, &output_sub, &index_sizes),
    }
}

/// Execute an einsum using an optimized contraction path.
///
/// For expressions with 2 or fewer operands, this falls through to the
/// standard `einsum`. For 3+ operands, it first computes a contraction path
/// and then executes pairwise contractions in that order.
pub fn einsum_optimized<T: Scalar>(
    subscripts: &str,
    operands: &[&Tensor<T>],
    strategy: PathStrategy,
) -> Result<Tensor<T>> {
    if operands.len() <= 2 {
        return einsum(subscripts, operands);
    }

    let (input_subs, output_sub, index_sizes) = parse_path_subscripts(subscripts, operands)?;

    let descs: Vec<OperandDesc> = input_subs
        .iter()
        .map(|indices| OperandDesc {
            indices: indices.clone(),
        })
        .collect();

    let path_info = match strategy {
        PathStrategy::Greedy => greedy_path(&descs, &output_sub, &index_sizes)?,
        PathStrategy::Optimal => optimal_path(&descs, &output_sub, &index_sizes)?,
    };

    execute_path(subscripts, operands, &path_info, &input_subs, &output_sub)
}

// ======================================================================
// Parsing
// ======================================================================

/// Parse subscripts for path computation. Returns (input index lists, output
/// indices, index-to-size mapping).
fn parse_path_subscripts<T: Scalar>(
    subscripts: &str,
    operands: &[&Tensor<T>],
) -> Result<ParsedSubscripts> {
    let subscripts = subscripts.replace(' ', "");

    let (inputs_str, output_sub) = if let Some((inp, out)) = subscripts.split_once("->") {
        let output_indices: Vec<char> = out.chars().collect();
        (inp.to_string(), output_indices)
    } else {
        // Implicit mode: output = sorted unique indices that appear exactly once
        let mut counts: BTreeMap<char, usize> = BTreeMap::new();
        for c in subscripts.chars() {
            if c == ',' {
                continue;
            }
            *counts.entry(c).or_insert(0) += 1;
        }
        let output_indices: Vec<char> = counts
            .iter()
            .filter(|(_, count)| **count == 1)
            .map(|(&c, _)| c)
            .collect();
        (subscripts.clone(), output_indices)
    };

    let input_parts: Vec<&str> = inputs_str.split(',').collect();
    if input_parts.len() != operands.len() {
        return Err(CoreError::InvalidArgument {
            reason: "number of subscript groups does not match number of operands",
        });
    }

    let mut input_subs = Vec::with_capacity(input_parts.len());
    let mut index_sizes: BTreeMap<char, usize> = BTreeMap::new();

    for (i, part) in input_parts.iter().enumerate() {
        let indices: Vec<char> = part.chars().collect();
        if indices.len() != operands[i].ndim() {
            return Err(CoreError::InvalidArgument {
                reason: "operand rank does not match number of subscript indices",
            });
        }
        let shape = operands[i].shape();
        for (d, &c) in indices.iter().enumerate() {
            if let Some(&existing) = index_sizes.get(&c) {
                if existing != shape[d] {
                    return Err(CoreError::DimensionMismatch {
                        expected: vec![existing],
                        got: vec![shape[d]],
                    });
                }
            } else {
                index_sizes.insert(c, shape[d]);
            }
        }
        input_subs.push(indices);
    }

    Ok((input_subs, output_sub, index_sizes))
}

// ======================================================================
// Cost estimation
// ======================================================================

/// Compute the cost (FLOPs) and output size of contracting two operands.
fn contraction_cost(
    a: &OperandDesc,
    b: &OperandDesc,
    output_indices: &[char],
    index_sizes: &BTreeMap<char, usize>,
) -> (usize, Vec<char>, Vec<usize>) {
    // Indices in the result of contracting a and b:
    // Keep indices that appear in a or b AND (appear in the final output OR
    // appear in other operands that haven't been contracted yet).
    // For simplicity in cost estimation, we keep all indices that are in a or b
    // but contract those that appear in BOTH a and b and NOT in output_indices.
    let a_set: BTreeSet<char> = a.indices.iter().copied().collect();
    let b_set: BTreeSet<char> = b.indices.iter().copied().collect();
    let output_set: BTreeSet<char> = output_indices.iter().copied().collect();

    // Contracted indices: in both a and b, not in the final output
    let contracted: BTreeSet<char> = a_set
        .intersection(&b_set)
        .filter(|c| !output_set.contains(c))
        .copied()
        .collect();

    // Result indices: union of a and b minus contracted
    let mut result_indices: Vec<char> = Vec::new();
    // First add a's indices (in order), skipping contracted
    for &c in &a.indices {
        if !contracted.contains(&c) && !result_indices.contains(&c) {
            result_indices.push(c);
        }
    }
    // Then add b's indices not already present
    for &c in &b.indices {
        if !contracted.contains(&c) && !result_indices.contains(&c) {
            result_indices.push(c);
        }
    }

    let result_shape: Vec<usize> = result_indices.iter().map(|c| index_sizes[c]).collect();

    let result_size: usize = result_shape.iter().product::<usize>().max(1);
    let contract_size: usize = contracted
        .iter()
        .map(|c| index_sizes[c])
        .product::<usize>()
        .max(1);

    // FLOPs ≈ result_size * contract_size (one multiply-add per element per contraction)
    let flops = result_size * contract_size;

    (flops, result_indices, result_shape)
}

/// Compute result indices when contracting two operands, keeping indices that
/// are needed by remaining operands or the final output.
fn pairwise_result_indices(
    a: &OperandDesc,
    b: &OperandDesc,
    remaining: &[OperandDesc],
    final_output: &[char],
    index_sizes: &BTreeMap<char, usize>,
) -> (Vec<char>, Vec<usize>) {
    let a_set: BTreeSet<char> = a.indices.iter().copied().collect();
    let b_set: BTreeSet<char> = b.indices.iter().copied().collect();

    // Indices needed by remaining operands or final output
    let mut needed: BTreeSet<char> = final_output.iter().copied().collect();
    for op in remaining {
        for &c in &op.indices {
            needed.insert(c);
        }
    }

    // Contract indices that appear in both a and b but are NOT needed elsewhere
    let contracted: BTreeSet<char> = a_set
        .intersection(&b_set)
        .filter(|c| !needed.contains(c))
        .copied()
        .collect();

    let mut result_indices: Vec<char> = Vec::new();
    for &c in &a.indices {
        if !contracted.contains(&c) && !result_indices.contains(&c) {
            result_indices.push(c);
        }
    }
    for &c in &b.indices {
        if !contracted.contains(&c) && !result_indices.contains(&c) {
            result_indices.push(c);
        }
    }

    let result_shape: Vec<usize> = result_indices.iter().map(|c| index_sizes[c]).collect();
    (result_indices, result_shape)
}

// ======================================================================
// Greedy path
// ======================================================================

#[allow(clippy::unnecessary_wraps)]
fn greedy_path(
    descs: &[OperandDesc],
    output_sub: &[char],
    index_sizes: &BTreeMap<char, usize>,
) -> Result<PathInfo> {
    let n = descs.len();
    if n <= 1 {
        return Ok(PathInfo {
            path: vec![],
            flops: 0,
            largest_intermediate: 0,
        });
    }

    let mut current: Vec<OperandDesc> = descs.to_vec();
    let mut path = Vec::with_capacity(n - 1);
    let mut total_flops = 0usize;
    let mut largest_intermediate = 0usize;

    while current.len() > 1 {
        let mut best_cost = usize::MAX;
        let mut best_i = 0;
        let mut best_j = 1;

        // Find the cheapest pair to contract
        for i in 0..current.len() {
            for j in (i + 1)..current.len() {
                // Build remaining list (excluding i and j)
                let (cost, _, _) =
                    contraction_cost(&current[i], &current[j], output_sub, index_sizes);

                // Tie-break: prefer contractions that reduce more indices
                if cost < best_cost {
                    best_cost = cost;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // Build remaining for the chosen pair
        let remaining: Vec<OperandDesc> = current
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != best_i && k != best_j)
            .map(|(_, d)| d.clone())
            .collect();

        let (result_indices, result_shape) = pairwise_result_indices(
            &current[best_i],
            &current[best_j],
            &remaining,
            output_sub,
            index_sizes,
        );

        let result_size: usize = result_shape.iter().product::<usize>().max(1);
        largest_intermediate = largest_intermediate.max(result_size);
        total_flops += best_cost;

        path.push(ContractionPair {
            first: best_i,
            second: best_j,
        });

        // Remove j first (larger index), then i
        current.remove(best_j);
        current.remove(best_i);

        // Append result
        current.push(OperandDesc {
            indices: result_indices,
        });
    }

    Ok(PathInfo {
        path,
        flops: total_flops,
        largest_intermediate,
    })
}

// ======================================================================
// Optimal path (exhaustive)
// ======================================================================

fn optimal_path(
    descs: &[OperandDesc],
    output_sub: &[char],
    index_sizes: &BTreeMap<char, usize>,
) -> Result<PathInfo> {
    let n = descs.len();
    if n <= 1 {
        return Ok(PathInfo {
            path: vec![],
            flops: 0,
            largest_intermediate: 0,
        });
    }
    if n > 8 {
        // Fall back to greedy for large inputs to avoid combinatorial explosion
        return greedy_path(descs, output_sub, index_sizes);
    }

    let mut best_path: Vec<ContractionPair> = Vec::new();
    let mut best_flops = usize::MAX;
    let mut best_largest = 0usize;

    find_optimal(
        descs,
        output_sub,
        index_sizes,
        &mut vec![],
        0,
        0,
        &mut best_path,
        &mut best_flops,
        &mut best_largest,
    );

    Ok(PathInfo {
        path: best_path,
        flops: best_flops,
        largest_intermediate: best_largest,
    })
}

#[allow(clippy::too_many_arguments)]
fn find_optimal(
    current: &[OperandDesc],
    output_sub: &[char],
    index_sizes: &BTreeMap<char, usize>,
    current_path: &mut Vec<ContractionPair>,
    current_flops: usize,
    current_largest: usize,
    best_path: &mut Vec<ContractionPair>,
    best_flops: &mut usize,
    best_largest: &mut usize,
) {
    if current.len() <= 1 {
        if current_flops < *best_flops {
            *best_flops = current_flops;
            *best_path = current_path.clone();
            *best_largest = current_largest;
        }
        return;
    }

    // Prune: if we've already exceeded the best, stop
    if current_flops >= *best_flops {
        return;
    }

    for i in 0..current.len() {
        for j in (i + 1)..current.len() {
            let remaining: Vec<OperandDesc> = current
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != i && k != j)
                .map(|(_, d)| d.clone())
                .collect();

            let (cost, _, _) = contraction_cost(&current[i], &current[j], output_sub, index_sizes);

            let (result_indices, result_shape) = pairwise_result_indices(
                &current[i],
                &current[j],
                &remaining,
                output_sub,
                index_sizes,
            );

            let result_size: usize = result_shape.iter().product::<usize>().max(1);

            let mut next = remaining;
            next.push(OperandDesc {
                indices: result_indices,
            });

            current_path.push(ContractionPair {
                first: i,
                second: j,
            });

            find_optimal(
                &next,
                output_sub,
                index_sizes,
                current_path,
                current_flops + cost,
                current_largest.max(result_size),
                best_path,
                best_flops,
                best_largest,
            );

            current_path.pop();
        }
    }
}

// ======================================================================
// Path execution
// ======================================================================

/// Execute an einsum following a precomputed contraction path.
fn execute_path<T: Scalar>(
    _subscripts: &str,
    operands: &[&Tensor<T>],
    path_info: &PathInfo,
    input_subs: &[Vec<char>],
    final_output: &[char],
) -> Result<Tensor<T>> {
    // We maintain a list of (indices, tensor) pairs. At each step, we contract
    // a pair and replace them with the result.
    let mut tensors: Vec<(Vec<char>, Tensor<T>)> = input_subs
        .iter()
        .zip(operands.iter())
        .map(|(indices, &t)| (indices.clone(), t.clone()))
        .collect();

    for step in &path_info.path {
        let j = step.second;
        let i = step.first;

        // Remove j first (larger), then i
        let (b_indices, b_tensor) = tensors.remove(j);
        let (a_indices, a_tensor) = tensors.remove(i);

        // Determine what indices should remain after this contraction
        let remaining_descs: Vec<OperandDesc> = tensors
            .iter()
            .map(|(indices, _)| OperandDesc {
                indices: indices.clone(),
            })
            .collect();

        let a_desc = OperandDesc {
            indices: a_indices.clone(),
        };
        let b_desc = OperandDesc {
            indices: b_indices.clone(),
        };

        // Use a dummy index_sizes from actual tensor shapes
        let mut local_sizes: BTreeMap<char, usize> = BTreeMap::new();
        for (c, &s) in a_indices.iter().zip(a_tensor.shape().iter()) {
            local_sizes.insert(*c, s);
        }
        for (c, &s) in b_indices.iter().zip(b_tensor.shape().iter()) {
            local_sizes.insert(*c, s);
        }

        let (result_indices, _result_shape) = pairwise_result_indices(
            &a_desc,
            &b_desc,
            &remaining_descs,
            final_output,
            &local_sizes,
        );

        // Build subscript string for this pairwise contraction
        let a_sub: String = a_indices.iter().collect();
        let b_sub: String = b_indices.iter().collect();
        let out_sub: String = result_indices.iter().collect();
        let pair_subscripts = format!("{a_sub},{b_sub}->{out_sub}");

        let result = einsum(&pair_subscripts, &[&a_tensor, &b_tensor])?;
        tensors.push((result_indices, result));
    }

    if tensors.len() == 1 {
        let (current_indices, tensor) = tensors.pop().unwrap();
        // If the indices don't match the final output order, we need a final
        // transpose/rearrangement via einsum
        if current_indices == final_output {
            Ok(tensor)
        } else {
            let cur_sub: String = current_indices.iter().collect();
            let out_sub: String = final_output.iter().collect();
            let reorder = format!("{cur_sub}->{out_sub}");
            einsum(&reorder, &[&tensor])
        }
    } else {
        Err(CoreError::InvalidArgument {
            reason: "einsum path execution did not reduce to a single tensor",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_einsum_path_chain_matmul() {
        // Chain of three matrix multiplies: A(2x3) @ B(3x4) @ C(4x5) -> (2x5)
        let a = Tensor::from_vec(vec![1.0_f64; 6], vec![2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f64; 12], vec![3, 4]).unwrap();
        let c = Tensor::from_vec(vec![1.0_f64; 20], vec![4, 5]).unwrap();

        let info = einsum_path("ij,jk,kl->il", &[&a, &b, &c], PathStrategy::Greedy).unwrap();
        assert_eq!(info.path.len(), 2);
        assert!(info.flops > 0);
    }

    #[test]
    fn test_einsum_path_optimal_vs_greedy() {
        let a = Tensor::from_vec(vec![1.0_f64; 6], vec![2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f64; 12], vec![3, 4]).unwrap();
        let c = Tensor::from_vec(vec![1.0_f64; 20], vec![4, 5]).unwrap();

        let greedy = einsum_path("ij,jk,kl->il", &[&a, &b, &c], PathStrategy::Greedy).unwrap();
        let optimal = einsum_path("ij,jk,kl->il", &[&a, &b, &c], PathStrategy::Optimal).unwrap();

        // Optimal should be <= greedy FLOPs
        assert!(optimal.flops <= greedy.flops);
        assert_eq!(optimal.path.len(), 2);
    }

    #[test]
    fn test_einsum_optimized_chain_matmul() {
        // Verify correctness: chain matmul matches direct einsum
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_vec(
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            vec![3, 4],
        )
        .unwrap();
        let c = Tensor::from_vec(
            vec![
                1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ],
            vec![4, 5],
        )
        .unwrap();

        let direct = einsum("ij,jk,kl->il", &[&a, &b, &c]).unwrap();
        let optimized =
            einsum_optimized("ij,jk,kl->il", &[&a, &b, &c], PathStrategy::Greedy).unwrap();

        assert_eq!(direct.shape(), optimized.shape());
        for (a, b) in direct.as_slice().iter().zip(optimized.as_slice().iter()) {
            assert!((a - b).abs() < 1e-10, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_einsum_optimized_four_operands() {
        // Four small matrices
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let d = Tensor::from_vec(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();

        let direct = einsum("ij,jk,kl,lm->im", &[&a, &b, &c, &d]).unwrap();
        let optimized =
            einsum_optimized("ij,jk,kl,lm->im", &[&a, &b, &c, &d], PathStrategy::Optimal).unwrap();

        assert_eq!(direct.shape(), optimized.shape());
        for (x, y) in direct.as_slice().iter().zip(optimized.as_slice().iter()) {
            assert!((x - y).abs() < 1e-10, "mismatch: {x} vs {y}");
        }
    }

    #[test]
    fn test_einsum_path_two_operands() {
        // With only two operands, path should be a single step
        let a = Tensor::from_vec(vec![1.0_f64; 6], vec![2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f64; 6], vec![3, 2]).unwrap();

        let result = einsum_optimized("ij,jk->ik", &[&a, &b], PathStrategy::Greedy).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_einsum_path_single_operand() {
        let a = Tensor::from_vec(vec![1.0_f64; 4], vec![2, 2]).unwrap();
        let info = einsum_path("ij->ji", &[&a], PathStrategy::Greedy).unwrap();
        assert!(info.path.is_empty());
    }

    #[test]
    fn test_einsum_path_asymmetric_shapes() {
        // Test where contraction order matters: A(100x2) @ B(2x3) @ C(3x100)
        // Greedy should prefer contracting the smaller intermediates first
        let a = Tensor::from_vec(vec![1.0_f64; 200], vec![100, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f64; 6], vec![2, 3]).unwrap();
        let c = Tensor::from_vec(vec![1.0_f64; 300], vec![3, 100]).unwrap();

        let info = einsum_path("ij,jk,kl->il", &[&a, &b, &c], PathStrategy::Greedy).unwrap();
        assert_eq!(info.path.len(), 2);
        // Should contract A@B first (result 100x3 = 300 elements, cost 100*3*2=600)
        // rather than B@C first (result 2x100 = 200 elements, cost 2*100*3=600)
        // Both have same cost in this case, but the important thing is it works
        assert!(info.flops > 0);
    }

    #[test]
    fn test_einsum_optimized_with_trace() {
        // Mix of contraction types: matmul + trace
        // "ij,jk,kk->i" — A @ B, then trace of second dimension
        let a = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let c = Tensor::from_vec(vec![3.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();

        let direct = einsum("ij,jk,kk->i", &[&a, &b, &c]).unwrap();
        let optimized =
            einsum_optimized("ij,jk,kk->i", &[&a, &b, &c], PathStrategy::Greedy).unwrap();

        assert_eq!(direct.shape(), optimized.shape());
        for (x, y) in direct.as_slice().iter().zip(optimized.as_slice().iter()) {
            assert!((x - y).abs() < 1e-10);
        }
    }
}
