//! Conversion between Scivex tensors and Apache Arrow arrays.
//!
//! Provides efficient conversion in both directions:
//! - [`tensor_f64_to_arrow`] / [`arrow_f64_to_tensor`] — `Tensor<f64>` ↔ `Float64Array`
//! - [`tensor_f32_to_arrow`] / [`arrow_f32_to_tensor`] — `Tensor<f32>` ↔ `Float32Array`
//! - [`tensor_to_record_batch`] / [`record_batch_to_tensor`] — 2-D matrix ↔ `RecordBatch`
//! - [`any_arrow_to_tensor_f64`] — any numeric Arrow array → `Tensor<f64>`

use std::sync::Arc;

use arrow_crate::array::{Array, ArrayRef, Float32Array, Float64Array};
use arrow_crate::datatypes::{Field, Schema};
use arrow_crate::record_batch::RecordBatch;

use scivex_core::Tensor;

use crate::error::{IoError, Result};

// ---------------------------------------------------------------------------
// Tensor → Arrow
// ---------------------------------------------------------------------------

/// Convert a `Tensor<f64>` to a `Float64Array`.
pub fn tensor_f64_to_arrow(tensor: &Tensor<f64>) -> Float64Array {
    Float64Array::from(tensor.as_slice().to_vec())
}

/// Convert a `Tensor<f32>` to a `Float32Array`.
pub fn tensor_f32_to_arrow(tensor: &Tensor<f32>) -> Float32Array {
    Float32Array::from(tensor.as_slice().to_vec())
}

/// Convert a 2-D `Tensor<f64>` to a `RecordBatch`.
///
/// For a tensor of shape `[n_rows, n_cols]`, produces a `RecordBatch` with
/// `n_cols` columns named `"col_0"`, `"col_1"`, etc.
pub fn tensor_to_record_batch(tensor: &Tensor<f64>) -> Result<RecordBatch> {
    tensor_to_record_batch_named(tensor, None)
}

/// Convert a 2-D `Tensor<f64>` to a `RecordBatch` with optional column names.
///
/// If `names` is `None`, columns are named `"col_0"`, `"col_1"`, etc.
pub fn tensor_to_record_batch_named(
    tensor: &Tensor<f64>,
    names: Option<&[&str]>,
) -> Result<RecordBatch> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(IoError::ArrowError(format!(
            "tensor_to_record_batch requires a 2-D tensor, got shape {shape:?}"
        )));
    }
    let (n_rows, n_cols) = (shape[0], shape[1]);
    let data = tensor.as_slice();

    if let Some(ns) = names
        && ns.len() != n_cols
    {
        return Err(IoError::ArrowError(format!(
            "expected {n_cols} column names, got {}",
            ns.len()
        )));
    }

    let mut fields = Vec::with_capacity(n_cols);
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(n_cols);

    for col in 0..n_cols {
        let name = names.map_or_else(|| format!("col_{col}"), |ns| ns[col].to_string());
        fields.push(Field::new(
            &name,
            arrow_crate::datatypes::DataType::Float64,
            false,
        ));

        let col_data: Vec<f64> = (0..n_rows).map(|row| data[row * n_cols + col]).collect();
        arrays.push(Arc::new(Float64Array::from(col_data)) as ArrayRef);
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays).map_err(|e| IoError::ArrowError(e.to_string()))
}

// ---------------------------------------------------------------------------
// Arrow → Tensor
// ---------------------------------------------------------------------------

/// Convert a `Float64Array` to a 1-D `Tensor<f64>`.
///
/// Null values are replaced with `0.0`.
pub fn arrow_f64_to_tensor(array: &Float64Array) -> Result<Tensor<f64>> {
    let data: Vec<f64> = (0..Array::len(array))
        .map(|i| {
            if Array::is_null(array, i) {
                0.0
            } else {
                array.value(i)
            }
        })
        .collect();
    let len = data.len();
    Tensor::from_vec(data, vec![len]).map_err(|e| IoError::ArrowError(e.to_string()))
}

/// Convert a `Float32Array` to a 1-D `Tensor<f32>`.
///
/// Null values are replaced with `0.0`.
pub fn arrow_f32_to_tensor(array: &Float32Array) -> Result<Tensor<f32>> {
    let data: Vec<f32> = (0..Array::len(array))
        .map(|i| {
            if Array::is_null(array, i) {
                0.0
            } else {
                array.value(i)
            }
        })
        .collect();
    let len = data.len();
    Tensor::from_vec(data, vec![len]).map_err(|e| IoError::ArrowError(e.to_string()))
}

/// Convert a `RecordBatch` of `Float64` columns to a 2-D `Tensor<f64>`.
///
/// All columns must be `Float64`. The result has shape `[n_rows, n_cols]`.
pub fn record_batch_to_tensor(batch: &RecordBatch) -> Result<Tensor<f64>> {
    let n_rows = batch.num_rows();
    let n_cols = batch.num_columns();

    if n_cols == 0 {
        return Tensor::from_vec(vec![], vec![0, 0])
            .map_err(|e| IoError::ArrowError(e.to_string()));
    }

    let mut data = vec![0.0_f64; n_rows * n_cols];

    for col in 0..n_cols {
        let array = batch.column(col);
        let f64_arr = array
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                IoError::ArrowError(format!(
                    "column {} ({}) is not Float64",
                    col,
                    batch.schema().field(col).name()
                ))
            })?;

        for row in 0..n_rows {
            data[row * n_cols + col] = if Array::is_null(f64_arr, row) {
                0.0
            } else {
                f64_arr.value(row)
            };
        }
    }

    Tensor::from_vec(data, vec![n_rows, n_cols]).map_err(|e| IoError::ArrowError(e.to_string()))
}

// ---------------------------------------------------------------------------
// Convenience: detect type from ArrayRef
// ---------------------------------------------------------------------------

/// Convert any supported Arrow array to a 1-D `Tensor<f64>`, casting as needed.
///
/// Supports integer and float types. Integer values are cast to f64.
/// Null values are replaced with `0.0`.
#[allow(clippy::cast_precision_loss)]
pub fn any_arrow_to_tensor_f64(array: &ArrayRef) -> Result<Tensor<f64>> {
    use arrow_crate::datatypes::DataType;

    macro_rules! extract_primitive {
        ($arr_type:ty, $conv:expr) => {{
            let arr = array
                .as_any()
                .downcast_ref::<$arr_type>()
                .ok_or_else(|| IoError::ArrowError("downcast failed".to_string()))?;
            let data: Vec<f64> = (0..Array::len(arr))
                .map(|i| {
                    if Array::is_null(arr, i) {
                        0.0
                    } else {
                        #[allow(clippy::redundant_closure_call)]
                        ($conv)(arr.value(i))
                    }
                })
                .collect();
            let len = data.len();
            Tensor::from_vec(data, vec![len]).map_err(|e| IoError::ArrowError(e.to_string()))
        }};
    }

    match array.data_type() {
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            arrow_f64_to_tensor(arr)
        }
        DataType::Float32 => extract_primitive!(Float32Array, |v: f32| f64::from(v)),
        DataType::Int64 => {
            extract_primitive!(arrow_crate::array::Int64Array, |v: i64| v as f64)
        }
        DataType::Int32 => {
            extract_primitive!(arrow_crate::array::Int32Array, |v: i32| f64::from(v))
        }
        DataType::Int16 => {
            extract_primitive!(arrow_crate::array::Int16Array, |v: i16| f64::from(v))
        }
        DataType::Int8 => {
            extract_primitive!(arrow_crate::array::Int8Array, |v: i8| f64::from(v))
        }
        DataType::UInt64 => {
            extract_primitive!(arrow_crate::array::UInt64Array, |v: u64| v as f64)
        }
        DataType::UInt32 => {
            extract_primitive!(arrow_crate::array::UInt32Array, |v: u32| f64::from(v))
        }
        DataType::UInt16 => {
            extract_primitive!(arrow_crate::array::UInt16Array, |v: u16| f64::from(v))
        }
        DataType::UInt8 => {
            extract_primitive!(arrow_crate::array::UInt8Array, |v: u8| f64::from(v))
        }
        dt => Err(IoError::ArrowError(format!(
            "unsupported Arrow type for tensor conversion: {dt:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_crate::array::Int32Array;

    #[test]
    fn test_tensor_f64_roundtrip() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let arrow = tensor_f64_to_arrow(&t);
        assert_eq!(arrow.len(), 4);
        assert!((arrow.value(0) - 1.0).abs() < 1e-10);
        assert!((arrow.value(3) - 4.0).abs() < 1e-10);

        let back = arrow_f64_to_tensor(&arrow).unwrap();
        assert_eq!(back.shape(), &[4]);
        assert!((back.as_slice()[0] - 1.0).abs() < 1e-10);
        assert!((back.as_slice()[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_f32_roundtrip() {
        let t = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], vec![3]).unwrap();
        let arrow = tensor_f32_to_arrow(&t);
        assert_eq!(arrow.len(), 3);

        let back = arrow_f32_to_tensor(&arrow).unwrap();
        assert_eq!(back.shape(), &[3]);
        assert!((back.as_slice()[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_to_record_batch() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let batch = tensor_to_record_batch(&t).unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(batch.schema().field(0).name(), "col_0");
        assert_eq!(batch.schema().field(2).name(), "col_2");

        let col0 = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((col0.value(0) - 1.0).abs() < 1e-10); // row 0, col 0
        assert!((col0.value(1) - 4.0).abs() < 1e-10); // row 1, col 0
    }

    #[test]
    fn test_tensor_to_record_batch_named() {
        let t = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
        let batch = tensor_to_record_batch_named(&t, Some(&["height", "weight"])).unwrap();

        assert_eq!(batch.schema().field(0).name(), "height");
        assert_eq!(batch.schema().field(1).name(), "weight");
    }

    #[test]
    fn test_record_batch_to_tensor() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let batch = tensor_to_record_batch(&t).unwrap();
        let back = record_batch_to_tensor(&batch).unwrap();

        assert_eq!(back.shape(), &[2, 3]);
        for (a, b) in t.as_slice().iter().zip(back.as_slice()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_any_arrow_to_tensor_f64_from_int() {
        let arr = Int32Array::from(vec![10, 20, 30]);
        let array_ref: ArrayRef = Arc::new(arr);
        let t = any_arrow_to_tensor_f64(&array_ref).unwrap();

        assert_eq!(t.shape(), &[3]);
        assert!((t.as_slice()[0] - 10.0).abs() < 1e-10);
        assert!((t.as_slice()[2] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_any_arrow_to_tensor_f64_from_f32() {
        let arr = Float32Array::from(vec![1.5_f32, 2.5, 3.5]);
        let array_ref: ArrayRef = Arc::new(arr);
        let t = any_arrow_to_tensor_f64(&array_ref).unwrap();

        assert_eq!(t.shape(), &[3]);
        assert!((t.as_slice()[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_arrow_with_nulls_to_tensor() {
        let arr = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        let t = arrow_f64_to_tensor(&arr).unwrap();

        assert_eq!(t.shape(), &[3]);
        assert!((t.as_slice()[0] - 1.0).abs() < 1e-10);
        assert!(t.as_slice()[1].abs() < 1e-10); // null → 0.0
        assert!((t.as_slice()[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_record_batch_non_f64_error() {
        let arr = Int32Array::from(vec![1, 2, 3]);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "x",
            arrow_crate::datatypes::DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr) as ArrayRef]).unwrap();
        let result = record_batch_to_tensor(&batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_1d_tensor_roundtrip() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let arrow = tensor_f64_to_arrow(&t);
        let back = arrow_f64_to_tensor(&arrow).unwrap();
        assert_eq!(t.as_slice(), back.as_slice());
    }

    #[test]
    fn test_tensor_to_record_batch_1d_error() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = tensor_to_record_batch(&t);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_to_record_batch_wrong_names_count() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = tensor_to_record_batch_named(&t, Some(&["a"]));
        assert!(result.is_err());
    }
}
