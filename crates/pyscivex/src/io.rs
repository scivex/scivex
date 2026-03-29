//! Python bindings for scivex-io (CSV, JSON, Parquet, Excel, SQL, Arrow).

use pyo3::prelude::*;
use scivex_io::arrow::{read_arrow as rust_read_arrow, write_arrow as rust_write_arrow};
use scivex_io::csv::{CsvReaderBuilder, CsvWriterBuilder, read_csv_path};
use scivex_io::excel::{read_excel as rust_read_excel, write_excel as rust_write_excel};
use scivex_io::json::{JsonOrientation, JsonReaderBuilder, JsonWriterBuilder, read_json_path};
use scivex_io::parquet::{read_parquet as rust_read_parquet, write_parquet as rust_write_parquet};
use scivex_io::sql::sqlite::SqliteConnection;

use crate::frame::PyDataFrame;

/// Convert an `IoError` into a Python `IOError`.
#[allow(clippy::needless_pass_by_value)]
fn io_err(e: scivex_io::IoError) -> PyErr {
    pyo3::exceptions::PyIOError::new_err(e.to_string())
}

// =========================================================================
// CSV
// =========================================================================

/// Read a CSV file into a DataFrame.
///
///   df = sv.io.read_csv("data.csv")
///   df = sv.io.read_csv("data.csv", delimiter=";", has_header=True)
#[pyfunction]
#[pyo3(signature = (path, delimiter=None, has_header=None, skip_rows=None, max_rows=None))]
fn read_csv(
    path: &str,
    delimiter: Option<&str>,
    has_header: Option<bool>,
    skip_rows: Option<usize>,
    max_rows: Option<usize>,
) -> PyResult<PyDataFrame> {
    let needs_builder =
        delimiter.is_some() || has_header.is_some() || skip_rows.is_some() || max_rows.is_some();

    if needs_builder {
        let mut builder = CsvReaderBuilder::new();
        if let Some(d) = delimiter {
            let byte = d.as_bytes().first().copied().unwrap_or(b',');
            builder = builder.delimiter(byte);
        }
        if let Some(h) = has_header {
            builder = builder.has_header(h);
        }
        if let Some(s) = skip_rows {
            builder = builder.skip_rows(s);
        }
        if let Some(m) = max_rows {
            builder = builder.max_rows(Some(m));
        }
        let df = builder.read_path(path).map_err(io_err)?;
        Ok(PyDataFrame::from_inner(df))
    } else {
        let df = read_csv_path(path).map_err(io_err)?;
        Ok(PyDataFrame::from_inner(df))
    }
}

/// Write a DataFrame to a CSV file.
///
///   sv.io.write_csv(df, "output.csv")
#[pyfunction]
#[pyo3(name = "write_csv", signature = (df, path, delimiter=None, write_header=None))]
fn py_write_csv(
    df: &PyDataFrame,
    path: &str,
    delimiter: Option<&str>,
    write_header: Option<bool>,
) -> PyResult<()> {
    let needs_builder = delimiter.is_some() || write_header.is_some();

    if needs_builder {
        let mut builder = CsvWriterBuilder::new();
        if let Some(d) = delimiter {
            let byte = d.as_bytes().first().copied().unwrap_or(b',');
            builder = builder.delimiter(byte);
        }
        if let Some(h) = write_header {
            builder = builder.write_header(h);
        }
        builder.write_path(path, &df.inner).map_err(io_err)?;
    } else {
        let file = std::fs::File::create(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        scivex_io::csv::write_csv(file, &df.inner).map_err(io_err)?;
    }
    Ok(())
}

// =========================================================================
// JSON
// =========================================================================

/// Read a JSON file into a DataFrame.
///
///   df = sv.io.read_json("data.json")
///   df = sv.io.read_json("data.json", orient="columns")
#[pyfunction]
#[pyo3(signature = (path, orient=None))]
fn read_json(path: &str, orient: Option<&str>) -> PyResult<PyDataFrame> {
    if let Some(o) = orient {
        let orientation = match o {
            "records" => JsonOrientation::Records,
            "columns" => JsonOrientation::Columns,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "orient must be 'records' or 'columns'",
                ));
            }
        };
        let df = JsonReaderBuilder::new()
            .orientation(orientation)
            .read_path(path)
            .map_err(io_err)?;
        Ok(PyDataFrame::from_inner(df))
    } else {
        let df = read_json_path(path).map_err(io_err)?;
        Ok(PyDataFrame::from_inner(df))
    }
}

/// Write a DataFrame to a JSON file.
///
///   sv.io.write_json(df, "output.json")
///   sv.io.write_json(df, "output.json", orient="columns", pretty=True)
#[pyfunction]
#[pyo3(name = "write_json", signature = (df, path, orient=None, pretty=None))]
fn py_write_json(
    df: &PyDataFrame,
    path: &str,
    orient: Option<&str>,
    pretty: Option<bool>,
) -> PyResult<()> {
    let needs_builder = orient.is_some() || pretty.is_some();

    if needs_builder {
        let mut builder = JsonWriterBuilder::new();
        if let Some(o) = orient {
            let orientation = match o {
                "records" => JsonOrientation::Records,
                "columns" => JsonOrientation::Columns,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "orient must be 'records' or 'columns'",
                    ));
                }
            };
            builder = builder.orientation(orientation);
        }
        if let Some(p) = pretty {
            builder = builder.pretty(p);
        }
        builder.write_path(path, &df.inner).map_err(io_err)?;
    } else {
        let file = std::fs::File::create(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        scivex_io::json::write_json(file, &df.inner).map_err(io_err)?;
    }
    Ok(())
}

// =========================================================================
// Parquet
// =========================================================================

/// Read a Parquet file into a DataFrame.
///
///   df = sv.io.read_parquet("data.parquet")
#[pyfunction]
fn read_parquet(path: &str) -> PyResult<PyDataFrame> {
    let df = rust_read_parquet(path).map_err(io_err)?;
    Ok(PyDataFrame::from_inner(df))
}

/// Write a DataFrame to a Parquet file.
///
///   sv.io.write_parquet(df, "output.parquet")
#[pyfunction]
#[pyo3(name = "write_parquet")]
fn py_write_parquet(df: &PyDataFrame, path: &str) -> PyResult<()> {
    rust_write_parquet(&df.inner, path).map_err(io_err)?;
    Ok(())
}

// =========================================================================
// Excel
// =========================================================================

/// Read an Excel file into a DataFrame.
///
///   df = sv.io.read_excel("data.xlsx")
#[pyfunction]
fn read_excel(path: &str) -> PyResult<PyDataFrame> {
    let df = rust_read_excel(path).map_err(io_err)?;
    Ok(PyDataFrame::from_inner(df))
}

/// Write a DataFrame to an Excel (.xlsx) file.
///
///   sv.io.write_excel(df, "output.xlsx")
#[pyfunction]
#[pyo3(name = "write_excel")]
fn py_write_excel(df: &PyDataFrame, path: &str) -> PyResult<()> {
    rust_write_excel(path, &df.inner).map_err(io_err)?;
    Ok(())
}

// =========================================================================
// Arrow IPC
// =========================================================================

/// Read an Arrow IPC file into a DataFrame.
///
///   df = sv.io.read_arrow("data.arrow")
#[pyfunction]
fn read_arrow(path: &str) -> PyResult<PyDataFrame> {
    let df = rust_read_arrow(path).map_err(io_err)?;
    Ok(PyDataFrame::from_inner(df))
}

/// Write a DataFrame to an Arrow IPC file.
///
///   sv.io.write_arrow(df, "output.arrow")
#[pyfunction]
#[pyo3(name = "write_arrow")]
fn py_write_arrow(df: &PyDataFrame, path: &str) -> PyResult<()> {
    rust_write_arrow(&df.inner, path).map_err(io_err)?;
    Ok(())
}

// =========================================================================
// SQL (SQLite)
// =========================================================================

/// Execute a SQL query against a SQLite database and return the results
/// as a DataFrame.
///
///   df = sv.io.read_sql("SELECT * FROM users", "data.db")
#[pyfunction]
fn read_sql(query: &str, db_path: &str) -> PyResult<PyDataFrame> {
    let conn = SqliteConnection::open(db_path).map_err(io_err)?;
    let df = conn.read_sql(query).map_err(io_err)?;
    Ok(PyDataFrame::from_inner(df))
}

// =========================================================================
// NPY / NPZ (NumPy format)
// =========================================================================

/// Read a `.npy` file into a Tensor.
///
///   t = sv.io.read_npy("data.npy")
#[pyfunction]
fn read_npy(path: &str) -> PyResult<crate::tensor::PyTensor> {
    let tensor = scivex_io::npy::read_npy_path(path).map_err(io_err)?;
    Ok(crate::tensor::PyTensor::from_f64(tensor))
}

/// Write a Tensor to a `.npy` file.
///
///   sv.io.write_npy(tensor, "output.npy")
#[pyfunction]
#[pyo3(name = "write_npy")]
fn py_write_npy(tensor: &crate::tensor::PyTensor, path: &str) -> PyResult<()> {
    let t = tensor.as_f64()?;
    scivex_io::npy::write_npy_path(path, t).map_err(io_err)?;
    Ok(())
}

/// Read a `.npz` archive into a dict of Tensors.
///
///   tensors = sv.io.read_npz("data.npz")  # -> {"arr_0": Tensor, ...}
#[pyfunction]
fn read_npz(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let map = scivex_io::npy::read_npz_path(path).map_err(io_err)?;
    let dict = pyo3::types::PyDict::new(py);
    for (name, tensor) in map {
        let py_tensor = crate::tensor::PyTensor::from_f64(tensor);
        dict.set_item(name, py_tensor.into_pyobject(py)?)?;
    }
    Ok(dict.into_any().unbind())
}

/// Write a dict of Tensors to a `.npz` archive.
///
///   sv.io.write_npz({"a": tensor_a, "b": tensor_b}, "output.npz")
#[pyfunction]
#[pyo3(name = "write_npz")]
fn py_write_npz(tensors: &Bound<'_, pyo3::types::PyDict>, path: &str) -> PyResult<()> {
    let mut map = std::collections::HashMap::new();
    for (key, value) in tensors.iter() {
        let name: String = key.extract()?;
        let py_tensor: crate::tensor::PyTensor = value.extract()?;
        let tensor = py_tensor.to_f64_tensor();
        map.insert(name, tensor);
    }
    scivex_io::npy::write_npz_path(path, &map).map_err(io_err)?;
    Ok(())
}

// =========================================================================
// HDF5
// =========================================================================

/// Read a single dataset from an HDF5 file into a Tensor.
///
///   t = sv.io.read_hdf5_dataset("data.h5", "/group/dataset")
#[pyfunction]
fn read_hdf5_dataset(path: &str, dataset_path: &str) -> PyResult<crate::tensor::PyTensor> {
    let file = std::fs::File::open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut reader = std::io::BufReader::new(file);
    let tensor: scivex_core::Tensor<f64> =
        scivex_io::hdf5::read_hdf5_dataset(&mut reader, dataset_path).map_err(io_err)?;
    Ok(crate::tensor::PyTensor::from_f64(tensor))
}

/// Write a Tensor as a dataset in an HDF5 file.
///
///   sv.io.write_hdf5_dataset(tensor, "data.h5", "/group/dataset")
#[pyfunction]
#[pyo3(name = "write_hdf5_dataset")]
fn py_write_hdf5_dataset(
    tensor: &crate::tensor::PyTensor,
    path: &str,
    dataset_path: &str,
) -> PyResult<()> {
    let t = tensor.as_f64()?;
    let file = std::fs::File::create(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut writer = std::io::BufWriter::new(file);
    scivex_io::hdf5::write_hdf5_dataset(&mut writer, dataset_path, t).map_err(io_err)?;
    Ok(())
}

/// List all dataset paths in an HDF5 file.
///
///   datasets = sv.io.list_hdf5_datasets("data.h5")  # -> ["/group/ds1", ...]
#[pyfunction]
fn list_hdf5_datasets(path: &str) -> PyResult<Vec<String>> {
    let file = std::fs::File::open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut reader = std::io::BufReader::new(file);
    let datasets = scivex_io::hdf5::list_hdf5_datasets(&mut reader).map_err(io_err)?;
    Ok(datasets)
}

// =========================================================================
// ORC
// =========================================================================

/// Read an ORC file into a DataFrame.
///
///   df = sv.io.read_orc("data.orc")
#[pyfunction]
fn read_orc(path: &str) -> PyResult<PyDataFrame> {
    let df = scivex_io::orc::read_orc(path).map_err(io_err)?;
    Ok(PyDataFrame::from_inner(df))
}

/// Write a DataFrame to an ORC file.
///
///   sv.io.write_orc(df, "output.orc")
#[pyfunction]
#[pyo3(name = "write_orc")]
fn py_write_orc(df: &PyDataFrame, path: &str) -> PyResult<()> {
    scivex_io::orc::write_orc(&df.inner, path).map_err(io_err)?;
    Ok(())
}

// =========================================================================
// Avro
// =========================================================================

/// Read an Avro container file into a DataFrame.
///
///   df = sv.io.read_avro("data.avro")
#[pyfunction]
fn read_avro(path: &str) -> PyResult<PyDataFrame> {
    let file = std::fs::File::open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut reader = std::io::BufReader::new(file);
    let df = scivex_io::avro::read_avro(&mut reader).map_err(io_err)?;
    Ok(PyDataFrame::from_inner(df))
}

/// Write a DataFrame to an Avro container file.
///
///   sv.io.write_avro(df, "output.avro")
#[pyfunction]
#[pyo3(name = "write_avro")]
fn py_write_avro(df: &PyDataFrame, path: &str) -> PyResult<()> {
    let file = std::fs::File::create(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut writer = std::io::BufWriter::new(file);
    scivex_io::avro::write_avro(&mut writer, &df.inner).map_err(io_err)?;
    Ok(())
}

// =========================================================================
// Registration
// =========================================================================

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "io")?;
    // CSV
    m.add_function(wrap_pyfunction!(read_csv, &m)?)?;
    m.add_function(wrap_pyfunction!(py_write_csv, &m)?)?;
    // JSON
    m.add_function(wrap_pyfunction!(read_json, &m)?)?;
    m.add_function(wrap_pyfunction!(py_write_json, &m)?)?;
    // Parquet
    m.add_function(wrap_pyfunction!(read_parquet, &m)?)?;
    m.add_function(wrap_pyfunction!(py_write_parquet, &m)?)?;
    // Excel
    m.add_function(wrap_pyfunction!(read_excel, &m)?)?;
    m.add_function(wrap_pyfunction!(py_write_excel, &m)?)?;
    // Arrow IPC
    m.add_function(wrap_pyfunction!(read_arrow, &m)?)?;
    m.add_function(wrap_pyfunction!(py_write_arrow, &m)?)?;
    // SQL (SQLite)
    m.add_function(wrap_pyfunction!(read_sql, &m)?)?;
    // NPY / NPZ
    m.add_function(wrap_pyfunction!(read_npy, &m)?)?;
    m.add_function(wrap_pyfunction!(py_write_npy, &m)?)?;
    m.add_function(wrap_pyfunction!(read_npz, &m)?)?;
    m.add_function(wrap_pyfunction!(py_write_npz, &m)?)?;
    // HDF5
    m.add_function(wrap_pyfunction!(read_hdf5_dataset, &m)?)?;
    m.add_function(wrap_pyfunction!(py_write_hdf5_dataset, &m)?)?;
    m.add_function(wrap_pyfunction!(list_hdf5_datasets, &m)?)?;
    // ORC
    m.add_function(wrap_pyfunction!(read_orc, &m)?)?;
    m.add_function(wrap_pyfunction!(py_write_orc, &m)?)?;
    // Avro
    m.add_function(wrap_pyfunction!(read_avro, &m)?)?;
    m.add_function(wrap_pyfunction!(py_write_avro, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
