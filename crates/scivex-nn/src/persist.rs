//! Neural network weight persistence — save and load model parameters.
//!
//! Since neural network layers contain `Variable<T>` with closures and
//! reference-counted graph nodes, full model serialization is not possible.
//! Instead, this module provides weight-only persistence:
//!
//! - `save_weights` — extracts all parameter tensors and writes them to a binary file
//! - `load_weights` — reads tensors from a binary file and returns them
//!
//! ## Binary Format
//!
//! ```text
//! [4 bytes] Magic: "SVNN"
//! [4 bytes] Format version (little-endian u32)
//! [8 bytes] Number of tensors (little-endian u64)
//! For each tensor:
//!   [8 bytes] Number of dimensions (little-endian u64)
//!   [N × 8 bytes] Shape dimensions (each little-endian u64)
//!   [M × 8 bytes] Data values (each little-endian f64)
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! // Save
//! let params = model.parameters();
//! save_weights("model.bin", &params)?;
//!
//! // Load — reconstruct model first, then load weights
//! let mut model = build_model(&mut rng);
//! let tensors = load_weights::<f64>("model.bin")?;
//! let params = model.parameters();
//! for (param, tensor) in params.iter().zip(tensors.iter()) {
//!     param.set_data(tensor.clone());
//! }
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};
use crate::variable::Variable;

const MAGIC: &[u8; 4] = b"SVNN";
const FORMAT_VERSION: u32 = 1;

fn io_err() -> NnError {
    NnError::InvalidParameter {
        name: "io",
        reason: "I/O error during weight persistence",
    }
}

fn to_f64<T: Float>(v: T) -> f64 {
    let s = format!("{v:?}");
    s.parse::<f64>().unwrap_or(0.0)
}

/// Save neural network parameters to a binary file.
///
/// Extracts the underlying tensor data from each `Variable` and writes
/// shapes and values in a portable binary format.
///
/// # Examples
///
/// ```ignore
/// # use scivex_nn::layer::{Linear, Layer};
/// # use scivex_nn::persist::{save_weights, load_weights};
/// # use scivex_core::random::Rng;
/// let mut rng = Rng::new(42);
/// let layer = Linear::<f64>::new(4, 2, true, &mut rng);
/// let params = layer.parameters();
/// save_weights("/tmp/model.bin", &params).unwrap();
/// ```
pub fn save_weights<T: Float>(path: &str, params: &[Variable<T>]) -> Result<()> {
    let f = File::create(path).map_err(|_| io_err())?;
    let mut w = BufWriter::new(f);

    w.write_all(MAGIC).map_err(|_| io_err())?;
    w.write_all(&FORMAT_VERSION.to_le_bytes())
        .map_err(|_| io_err())?;
    w.write_all(&(params.len() as u64).to_le_bytes())
        .map_err(|_| io_err())?;

    for param in params {
        let data = param.data();
        let shape = data.shape();
        let values = data.as_slice();

        // Write number of dimensions
        w.write_all(&(shape.len() as u64).to_le_bytes())
            .map_err(|_| io_err())?;

        // Write shape
        for &dim in shape {
            w.write_all(&(dim as u64).to_le_bytes())
                .map_err(|_| io_err())?;
        }

        // Write data as f64
        for &v in values {
            w.write_all(&to_f64(v).to_le_bytes())
                .map_err(|_| io_err())?;
        }
    }

    w.flush().map_err(|_| io_err())?;
    Ok(())
}

/// Load neural network parameters from a binary file.
///
/// Returns a vector of `Tensor<T>` in the same order they were saved.
/// The caller is responsible for mapping these back to the model's
/// `Variable` parameters (typically via `Variable::set_data`).
///
/// # Examples
///
/// ```ignore
/// # use scivex_nn::persist::load_weights;
/// // Assuming "model.bin" was previously saved with save_weights.
/// let tensors = load_weights::<f64>("/tmp/model.bin").unwrap();
/// ```
pub fn load_weights<T: Float>(path: &str) -> Result<Vec<Tensor<T>>> {
    let f = File::open(path).map_err(|_| io_err())?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(|_| io_err())?;
    if &magic != MAGIC {
        return Err(NnError::InvalidParameter {
            name: "file",
            reason: "not a valid SVNN weight file",
        });
    }

    let mut ver = [0u8; 4];
    r.read_exact(&mut ver).map_err(|_| io_err())?;
    let version = u32::from_le_bytes(ver);
    if version > FORMAT_VERSION {
        return Err(NnError::InvalidParameter {
            name: "version",
            reason: "file version is newer than supported",
        });
    }

    let mut n_buf = [0u8; 8];
    r.read_exact(&mut n_buf).map_err(|_| io_err())?;
    let n_tensors = u64::from_le_bytes(n_buf) as usize;

    let mut tensors = Vec::with_capacity(n_tensors);

    for _ in 0..n_tensors {
        // Read number of dimensions
        let mut ndim_buf = [0u8; 8];
        r.read_exact(&mut ndim_buf).map_err(|_| io_err())?;
        let ndim = u64::from_le_bytes(ndim_buf) as usize;

        // Read shape
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let mut dim_buf = [0u8; 8];
            r.read_exact(&mut dim_buf).map_err(|_| io_err())?;
            shape.push(u64::from_le_bytes(dim_buf) as usize);
        }

        // Read data
        let numel: usize = shape.iter().product();
        let mut data = Vec::with_capacity(numel);
        for _ in 0..numel {
            let mut val_buf = [0u8; 8];
            r.read_exact(&mut val_buf).map_err(|_| io_err())?;
            data.push(T::from_f64(f64::from_le_bytes(val_buf)));
        }

        let tensor = Tensor::from_vec(data, shape).map_err(NnError::from)?;
        tensors.push(tensor);
    }

    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{Layer, Linear};
    use scivex_core::random::Rng;

    fn temp_path(name: &str) -> String {
        let dir = std::env::temp_dir();
        format!(
            "{}/scivex_nn_test_{name}_{}.bin",
            dir.display(),
            std::process::id()
        )
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut rng = Rng::new(42);
        let linear = Linear::<f64>::new(4, 3, true, &mut rng);
        let params = linear.parameters();

        let path = temp_path("linear");
        save_weights(&path, &params).unwrap();
        let loaded = load_weights::<f64>(&path).unwrap();

        assert_eq!(loaded.len(), params.len());
        for (orig, load) in params.iter().zip(loaded.iter()) {
            assert_eq!(orig.shape(), load.shape().to_vec());
            let orig_data = orig.data();
            let orig_s = orig_data.as_slice();
            let load_s = load.as_slice();
            for (a, b) in orig_s.iter().zip(load_s.iter()) {
                assert!((*a - *b).abs() < 1e-10);
            }
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_load_shapes() {
        let mut rng = Rng::new(42);
        let linear = Linear::<f64>::new(8, 16, true, &mut rng);
        let params = linear.parameters();

        let path = temp_path("shapes");
        save_weights(&path, &params).unwrap();
        let loaded = load_weights::<f64>(&path).unwrap();

        // weight: [16, 8], bias: [16]
        assert_eq!(loaded[0].shape(), &[16, 8]);
        assert_eq!(loaded[1].shape(), &[16]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_load_no_bias() {
        let mut rng = Rng::new(42);
        let linear = Linear::<f64>::new(4, 3, false, &mut rng);
        let params = linear.parameters();
        assert_eq!(params.len(), 1); // weight only

        let path = temp_path("no_bias");
        save_weights(&path, &params).unwrap();
        let loaded = load_weights::<f64>(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].shape(), &[3, 4]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_invalid_file() {
        let path = temp_path("nonexistent_svnn_file");
        assert!(load_weights::<f64>(&path).is_err());
    }

    #[test]
    fn test_wrong_magic() {
        let path = temp_path("bad_magic");
        std::fs::write(&path, b"BADX").unwrap();
        assert!(load_weights::<f64>(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_empty_params() {
        let path = temp_path("empty");
        save_weights::<f64>(&path, &[]).unwrap();
        let loaded = load_weights::<f64>(&path).unwrap();
        assert!(loaded.is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_apply_loaded_weights() {
        let mut rng = Rng::new(42);
        let linear1 = Linear::<f64>::new(4, 3, true, &mut rng);
        let params1 = linear1.parameters();

        let path = temp_path("apply");
        save_weights(&path, &params1).unwrap();
        let loaded = load_weights::<f64>(&path).unwrap();

        // Create a new model and load weights into it
        let mut rng2 = Rng::new(99);
        let linear2 = Linear::<f64>::new(4, 3, true, &mut rng2);
        let params2 = linear2.parameters();

        for (p, t) in params2.iter().zip(loaded.iter()) {
            p.set_data(t.clone());
        }

        // Verify weights match
        for (p1, p2) in params1.iter().zip(params2.iter()) {
            let d1 = p1.data();
            let d2 = p2.data();
            assert_eq!(d1.as_slice(), d2.as_slice());
        }

        // Verify forward pass matches
        let x = Variable::new(Tensor::ones(vec![2, 4]), false);
        let y1 = linear1.forward(&x).unwrap();
        let y2 = linear2.forward(&x).unwrap();
        let y1d = y1.data();
        let y2d = y2.data();
        assert_eq!(y1d.as_slice(), y2d.as_slice());

        std::fs::remove_file(&path).ok();
    }
}
