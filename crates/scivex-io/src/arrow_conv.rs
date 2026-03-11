//! Shared conversion utilities between Scivex DataFrames and Apache Arrow.
//!
//! Used by both the Parquet and Arrow IPC modules.

use std::sync::Arc;

use arrow_crate::array::{
    Array, ArrayRef, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_crate::datatypes::{DataType, Field, Schema};
use arrow_crate::record_batch::RecordBatch;

use scivex_frame::dtype::HasDType;
use scivex_frame::{AnySeries, DType, DataFrame, Series, StringSeries};

use crate::error::{IoError, Result};

/// Convert a [`DataFrame`] into a single Arrow [`RecordBatch`].
pub(crate) fn dataframe_to_record_batch(df: &DataFrame) -> Result<RecordBatch> {
    let mut fields = Vec::with_capacity(df.ncols());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(df.ncols());

    for col in df.columns() {
        let (field, array) = series_to_arrow(col.as_ref());
        fields.push(field);
        arrays.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays).map_err(|e| IoError::ArrowError(e.to_string()))
}

/// Build an Arrow array from a numeric Series, handling nulls.
fn numeric_to_arrow<T, A>(col: &dyn AnySeries, nullable: bool) -> ArrayRef
where
    T: scivex_core::Scalar + Copy + 'static,
    A: From<Vec<T>> + From<Vec<Option<T>>> + 'static,
    A: arrow_crate::array::Array,
{
    let s = col.as_any().downcast_ref::<Series<T>>().unwrap();
    if nullable {
        let values: Vec<Option<T>> = (0..s.len())
            .map(|i| {
                if col.is_null(i) {
                    None
                } else {
                    Some(s.as_slice()[i])
                }
            })
            .collect();
        Arc::new(A::from(values)) as ArrayRef
    } else {
        Arc::new(A::from(s.as_slice().to_vec())) as ArrayRef
    }
}

/// Convert a single [`AnySeries`] column to an Arrow field + array.
fn series_to_arrow(col: &dyn AnySeries) -> (Field, ArrayRef) {
    let name = col.name();
    let nullable = col.null_count() > 0;

    match col.dtype() {
        DType::I64 => {
            let array = numeric_to_arrow::<i64, Int64Array>(col, nullable);
            (Field::new(name, DataType::Int64, nullable), array)
        }
        DType::I32 => {
            let array = numeric_to_arrow::<i32, Int32Array>(col, nullable);
            (Field::new(name, DataType::Int32, nullable), array)
        }
        DType::I16 => {
            let array = numeric_to_arrow::<i16, Int16Array>(col, nullable);
            (Field::new(name, DataType::Int16, nullable), array)
        }
        DType::I8 => {
            let array = numeric_to_arrow::<i8, Int8Array>(col, nullable);
            (Field::new(name, DataType::Int8, nullable), array)
        }
        DType::U64 => {
            let array = numeric_to_arrow::<u64, UInt64Array>(col, nullable);
            (Field::new(name, DataType::UInt64, nullable), array)
        }
        DType::U32 => {
            let array = numeric_to_arrow::<u32, UInt32Array>(col, nullable);
            (Field::new(name, DataType::UInt32, nullable), array)
        }
        DType::U16 => {
            let array = numeric_to_arrow::<u16, UInt16Array>(col, nullable);
            (Field::new(name, DataType::UInt16, nullable), array)
        }
        DType::U8 | DType::Bool => {
            let array = numeric_to_arrow::<u8, UInt8Array>(col, nullable);
            (Field::new(name, DataType::UInt8, nullable), array)
        }
        DType::F64 => {
            let array = numeric_to_arrow::<f64, Float64Array>(col, nullable);
            (Field::new(name, DataType::Float64, nullable), array)
        }
        DType::F32 => {
            let array = numeric_to_arrow::<f32, Float32Array>(col, nullable);
            (Field::new(name, DataType::Float32, nullable), array)
        }
        DType::Str | DType::Categorical | DType::DateTime => {
            let values: Vec<Option<String>> = (0..col.len())
                .map(|i| {
                    if col.is_null(i) {
                        None
                    } else {
                        Some(col.display_value(i))
                    }
                })
                .collect();
            let array = Arc::new(StringArray::from(values)) as ArrayRef;
            (Field::new(name, DataType::Utf8, nullable), array)
        }
    }
}

/// Convert an Arrow [`RecordBatch`] into a [`DataFrame`].
pub(crate) fn record_batch_to_dataframe(batch: &RecordBatch) -> Result<DataFrame> {
    let mut columns: Vec<Box<dyn AnySeries>> = Vec::with_capacity(batch.num_columns());

    for (i, field) in batch.schema().fields().iter().enumerate() {
        let array = batch.column(i);
        let series = arrow_array_to_series(field.name(), array)?;
        columns.push(series);
    }

    DataFrame::new(columns).map_err(IoError::FrameError)
}

/// Convert multiple Arrow [`RecordBatch`]es into a single [`DataFrame`].
///
/// All batches must share the same schema.
pub(crate) fn record_batches_to_dataframe(batches: &[RecordBatch]) -> Result<DataFrame> {
    if batches.is_empty() {
        return Ok(DataFrame::empty());
    }

    if batches.len() == 1 {
        return record_batch_to_dataframe(&batches[0]);
    }

    // Concatenate batches by collecting all values per column.
    let schema = batches[0].schema();
    let num_cols = schema.fields().len();
    let mut columns: Vec<Box<dyn AnySeries>> = Vec::with_capacity(num_cols);

    for (col_idx, field) in schema.fields().iter().enumerate() {
        let concatenated = arrow_crate::compute::concat(
            &batches
                .iter()
                .map(|b| b.column(col_idx).as_ref())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| IoError::ArrowError(e.to_string()))?;

        let series = arrow_array_to_series(field.name(), &concatenated)?;
        columns.push(series);
    }

    DataFrame::new(columns).map_err(IoError::FrameError)
}

/// Downcast an Arrow array to a concrete type, returning an error on failure.
fn downcast_arrow<'a, T: 'static>(array: &'a ArrayRef, type_name: &str) -> Result<&'a T> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| IoError::ArrowError(format!("Failed to downcast {type_name}")))
}

/// Convert a single Arrow array into a boxed [`AnySeries`].
fn arrow_array_to_series(name: &str, array: &ArrayRef) -> Result<Box<dyn AnySeries>> {
    match array.data_type() {
        DataType::Int64 => {
            typed_array_to_series(name, downcast_arrow::<Int64Array>(array, "Int64Array")?)
        }
        DataType::Int32 => {
            typed_array_to_series(name, downcast_arrow::<Int32Array>(array, "Int32Array")?)
        }
        DataType::Int16 => {
            typed_array_to_series(name, downcast_arrow::<Int16Array>(array, "Int16Array")?)
        }
        DataType::Int8 => {
            typed_array_to_series(name, downcast_arrow::<Int8Array>(array, "Int8Array")?)
        }
        DataType::UInt64 => {
            typed_array_to_series(name, downcast_arrow::<UInt64Array>(array, "UInt64Array")?)
        }
        DataType::UInt32 => {
            typed_array_to_series(name, downcast_arrow::<UInt32Array>(array, "UInt32Array")?)
        }
        DataType::UInt16 => {
            typed_array_to_series(name, downcast_arrow::<UInt16Array>(array, "UInt16Array")?)
        }
        DataType::UInt8 => {
            typed_array_to_series(name, downcast_arrow::<UInt8Array>(array, "UInt8Array")?)
        }
        DataType::Float64 => {
            typed_array_to_series(name, downcast_arrow::<Float64Array>(array, "Float64Array")?)
        }
        DataType::Float32 => {
            typed_array_to_series(name, downcast_arrow::<Float32Array>(array, "Float32Array")?)
        }
        DataType::Utf8 => {
            let arr = downcast_arrow::<StringArray>(array, "StringArray")?;
            string_array_to_series(name, arr)
        }
        DataType::LargeUtf8 => {
            let arr =
                downcast_arrow::<arrow_crate::array::LargeStringArray>(array, "LargeStringArray")?;
            let values: Vec<Option<String>> = arr
                .iter()
                .map(|v: Option<&str>| v.map(String::from))
                .collect();
            build_string_series(name, &values)
        }
        // For unsupported types, convert to string representation.
        dt => {
            let values: Vec<Option<String>> = (0..array.len())
                .map(|i| {
                    if array.is_null(i) {
                        None
                    } else {
                        Some(format!(
                            "{:?}",
                            arrow_crate::util::display::array_value_to_string(array, i)
                        ))
                    }
                })
                .collect();
            let _ = dt;
            build_string_series(name, &values)
        }
    }
}

/// Convert a typed Arrow primitive array to a Scivex Series.
fn typed_array_to_series<T>(
    name: &str,
    arr: &arrow_crate::array::PrimitiveArray<T>,
) -> Result<Box<dyn AnySeries>>
where
    T: arrow_crate::datatypes::ArrowPrimitiveType,
    T::Native: scivex_core::Scalar + HasDType + 'static,
{
    let data: Vec<T::Native> = arr.values().iter().copied().collect();
    if Array::null_count(arr) > 0 {
        let nulls: Vec<bool> = (0..Array::len(arr))
            .map(|i| Array::is_null(arr, i))
            .collect();
        Ok(Box::new(Series::with_nulls(name, data, nulls)?))
    } else {
        Ok(Box::new(Series::new(name, data)))
    }
}

/// Convert an Arrow `StringArray` to a Scivex `StringSeries`.
fn string_array_to_series(name: &str, arr: &StringArray) -> Result<Box<dyn AnySeries>> {
    let values: Vec<Option<String>> = arr.iter().map(|v| v.map(String::from)).collect();
    build_string_series(name, &values)
}

/// Build a `StringSeries` from optional values.
fn build_string_series(name: &str, values: &[Option<String>]) -> Result<Box<dyn AnySeries>> {
    let has_nulls = values.iter().any(Option::is_none);
    if has_nulls {
        let data: Vec<String> = values
            .iter()
            .map(|v| v.clone().unwrap_or_default())
            .collect();
        let nulls: Vec<bool> = values.iter().map(Option::is_none).collect();
        Ok(Box::new(StringSeries::with_nulls(name, data, nulls)?))
    } else {
        let data: Vec<String> = values
            .iter()
            .map(|v| v.clone().unwrap_or_default())
            .collect();
        Ok(Box::new(StringSeries::new(name, data)))
    }
}
