//! Model persistence — save and load trained ML models.
//!
//! Provides a [`Persistable`] trait and binary serialization for all major
//! scivex-ml model types. The binary format is compact and portable:
//!
//! ```text
//! [4 bytes] Magic: "SVEX"
//! [4 bytes] Format version (little-endian u32)
//! [4 bytes] Model tag length (little-endian u32)
//! [N bytes] Model tag (UTF-8 string)
//! [...]     Model-specific payload
//! ```
//!
//! All numeric values are stored as little-endian f64 for maximum precision.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};

use scivex_core::Float;

use crate::error::{MlError, Result};

// ── Magic & Version ─────────────────────────────────────────────────────

const MAGIC: &[u8; 4] = b"SVEX";
const FORMAT_VERSION: u32 = 1;

// ── Persistable Trait ───────────────────────────────────────────────────

/// A model that can be saved to and loaded from disk.
pub trait Persistable: Sized {
    /// Save the model to a file in binary format.
    fn save(&self, path: &str) -> Result<()>;

    /// Load a model from a binary file.
    fn load(path: &str) -> Result<Self>;
}

// ── Binary Writer ───────────────────────────────────────────────────────

pub(crate) struct BinWriter {
    w: BufWriter<File>,
}

impl BinWriter {
    fn create(path: &str, model_tag: &str) -> Result<Self> {
        let f = File::create(path).map_err(io_err)?;
        let mut w = BufWriter::new(f);
        w.write_all(MAGIC).map_err(io_err)?;
        w.write_all(&FORMAT_VERSION.to_le_bytes()).map_err(io_err)?;
        let tag_bytes = model_tag.as_bytes();
        w.write_all(&(tag_bytes.len() as u32).to_le_bytes())
            .map_err(io_err)?;
        w.write_all(tag_bytes).map_err(io_err)?;
        Ok(Self { w })
    }

    fn write_f64(&mut self, v: f64) -> Result<()> {
        self.w.write_all(&v.to_le_bytes()).map_err(io_err)
    }

    fn write_usize(&mut self, v: usize) -> Result<()> {
        self.w.write_all(&(v as u64).to_le_bytes()).map_err(io_err)
    }

    fn write_bool(&mut self, v: bool) -> Result<()> {
        self.w.write_all(&[u8::from(v)]).map_err(io_err)
    }

    fn write_float<T: Float>(&mut self, v: T) -> Result<()> {
        self.write_f64(to_f64(v))
    }

    fn write_option_float<T: Float>(&mut self, v: Option<&T>) -> Result<()> {
        match v {
            Some(val) => {
                self.write_bool(true)?;
                self.write_float(*val)
            }
            None => self.write_bool(false),
        }
    }

    fn write_vec_float<T: Float>(&mut self, v: &[T]) -> Result<()> {
        self.write_usize(v.len())?;
        for &val in v {
            self.write_float(val)?;
        }
        Ok(())
    }

    fn write_option_vec_float<T: Float>(&mut self, v: Option<&Vec<T>>) -> Result<()> {
        match v {
            Some(vec) => {
                self.write_bool(true)?;
                self.write_vec_float(vec)
            }
            None => self.write_bool(false),
        }
    }

    fn write_option_usize(&mut self, v: Option<&usize>) -> Result<()> {
        match v {
            Some(val) => {
                self.write_bool(true)?;
                self.write_usize(*val)
            }
            None => self.write_bool(false),
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.w.flush().map_err(io_err)
    }
}

// ── Binary Reader ───────────────────────────────────────────────────────

pub(crate) struct BinReader {
    r: BufReader<File>,
}

impl BinReader {
    fn open(path: &str, expected_tag: &str) -> Result<Self> {
        let f = File::open(path).map_err(io_err)?;
        let mut r = BufReader::new(f);

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic).map_err(io_err)?;
        if &magic != MAGIC {
            return Err(MlError::InvalidParameter {
                name: "file",
                reason: "not a valid SVEX model file",
            });
        }

        let mut ver = [0u8; 4];
        r.read_exact(&mut ver).map_err(io_err)?;
        let version = u32::from_le_bytes(ver);
        if version > FORMAT_VERSION {
            return Err(MlError::InvalidParameter {
                name: "version",
                reason: "file version is newer than supported",
            });
        }

        let mut tag_len = [0u8; 4];
        r.read_exact(&mut tag_len).map_err(io_err)?;
        let len = u32::from_le_bytes(tag_len) as usize;
        let mut tag_bytes = vec![0u8; len];
        r.read_exact(&mut tag_bytes).map_err(io_err)?;
        let tag = String::from_utf8(tag_bytes).map_err(|_| MlError::InvalidParameter {
            name: "tag",
            reason: "invalid UTF-8 model tag",
        })?;
        if tag != expected_tag {
            return Err(MlError::InvalidParameter {
                name: "model_type",
                reason: "model type mismatch",
            });
        }

        Ok(Self { r })
    }

    fn read_f64(&mut self) -> Result<f64> {
        let mut buf = [0u8; 8];
        self.r.read_exact(&mut buf).map_err(io_err)?;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_usize(&mut self) -> Result<usize> {
        let mut buf = [0u8; 8];
        self.r.read_exact(&mut buf).map_err(io_err)?;
        Ok(u64::from_le_bytes(buf) as usize)
    }

    fn read_bool(&mut self) -> Result<bool> {
        let mut buf = [0u8; 1];
        self.r.read_exact(&mut buf).map_err(io_err)?;
        Ok(buf[0] != 0)
    }

    fn read_float<T: Float>(&mut self) -> Result<T> {
        Ok(T::from_f64(self.read_f64()?))
    }

    fn read_option_float<T: Float>(&mut self) -> Result<Option<T>> {
        if self.read_bool()? {
            Ok(Some(self.read_float()?))
        } else {
            Ok(None)
        }
    }

    fn read_vec_float<T: Float>(&mut self) -> Result<Vec<T>> {
        let len = self.read_usize()?;
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(self.read_float()?);
        }
        Ok(v)
    }

    fn read_option_vec_float<T: Float>(&mut self) -> Result<Option<Vec<T>>> {
        if self.read_bool()? {
            Ok(Some(self.read_vec_float()?))
        } else {
            Ok(None)
        }
    }

    fn read_option_usize(&mut self) -> Result<Option<usize>> {
        if self.read_bool()? {
            Ok(Some(self.read_usize()?))
        } else {
            Ok(None)
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

#[allow(clippy::needless_pass_by_value)]
fn io_err(e: io::Error) -> MlError {
    MlError::InvalidParameter {
        name: "io",
        reason: if e.kind() == io::ErrorKind::NotFound {
            "file not found"
        } else {
            "I/O error"
        },
    }
}

fn to_f64<T: Float>(v: T) -> f64 {
    // Float only has f32 and f64 impls; use format/parse
    let s = format!("{v:?}");
    s.parse::<f64>().unwrap_or(0.0)
}

// ── LinearRegression ────────────────────────────────────────────────────

use crate::linear::{LinearRegression, Ridge};

impl<T: Float> Persistable for LinearRegression<T> {
    fn save(&self, path: &str) -> Result<()> {
        let mut w = BinWriter::create(path, "LinearRegression")?;
        w.write_option_vec_float(self.weights.as_ref())?;
        w.write_option_float(self.bias.as_ref())?;
        w.flush()
    }

    fn load(path: &str) -> Result<Self> {
        let mut r = BinReader::open(path, "LinearRegression")?;
        let weights = r.read_option_vec_float()?;
        let bias = r.read_option_float()?;
        Ok(Self { weights, bias })
    }
}

impl<T: Float> Persistable for Ridge<T> {
    fn save(&self, path: &str) -> Result<()> {
        let mut w = BinWriter::create(path, "Ridge")?;
        w.write_float(self.alpha)?;
        w.write_option_vec_float(self.weights.as_ref())?;
        w.write_option_float(self.bias.as_ref())?;
        w.flush()
    }

    fn load(path: &str) -> Result<Self> {
        let mut r = BinReader::open(path, "Ridge")?;
        let alpha = r.read_float()?;
        let weights = r.read_option_vec_float()?;
        let bias = r.read_option_float()?;
        Ok(Self {
            alpha,
            weights,
            bias,
        })
    }
}

// ── Decision Trees ──────────────────────────────────────────────────────

use crate::tree::{DecisionTreeClassifier, DecisionTreeRegressor};

// The tree Node enum needs manual serialization
use crate::tree::decision_tree::Node;

fn write_tree_node<T: Float>(w: &mut BinWriter, node: &Node<T>) -> Result<()> {
    match node {
        Node::Leaf { value } => {
            w.write_bool(true)?; // is_leaf
            w.write_float(*value)
        }
        Node::Split {
            feature,
            threshold,
            left,
            right,
        } => {
            w.write_bool(false)?; // not leaf
            w.write_usize(*feature)?;
            w.write_float(*threshold)?;
            write_tree_node(w, left)?;
            write_tree_node(w, right)
        }
    }
}

fn read_tree_node<T: Float>(r: &mut BinReader) -> Result<Node<T>> {
    let is_leaf = r.read_bool()?;
    if is_leaf {
        let value = r.read_float()?;
        Ok(Node::Leaf { value })
    } else {
        let feature = r.read_usize()?;
        let threshold = r.read_float()?;
        let left = Box::new(read_tree_node(r)?);
        let right = Box::new(read_tree_node(r)?);
        Ok(Node::Split {
            feature,
            threshold,
            left,
            right,
        })
    }
}

fn write_option_node<T: Float>(w: &mut BinWriter, node: Option<&Node<T>>) -> Result<()> {
    match node {
        Some(n) => {
            w.write_bool(true)?;
            write_tree_node(w, n)
        }
        None => w.write_bool(false),
    }
}

fn read_option_node<T: Float>(r: &mut BinReader) -> Result<Option<Node<T>>> {
    if r.read_bool()? {
        Ok(Some(read_tree_node(r)?))
    } else {
        Ok(None)
    }
}

impl<T: Float> Persistable for DecisionTreeClassifier<T> {
    fn save(&self, path: &str) -> Result<()> {
        let mut w = BinWriter::create(path, "DecisionTreeClassifier")?;
        w.write_option_usize(self.max_depth.as_ref())?;
        w.write_usize(self.min_samples_split)?;
        write_option_node(&mut w, self.root.as_ref())?;
        w.write_option_vec_float(self.classes.as_ref())?;
        w.flush()
    }

    fn load(path: &str) -> Result<Self> {
        let mut r = BinReader::open(path, "DecisionTreeClassifier")?;
        let max_depth = r.read_option_usize()?;
        let min_samples_split = r.read_usize()?;
        let root = read_option_node(&mut r)?;
        let classes = r.read_option_vec_float()?;
        Ok(Self {
            max_depth,
            min_samples_split,
            root,
            classes,
        })
    }
}

impl<T: Float> Persistable for DecisionTreeRegressor<T> {
    fn save(&self, path: &str) -> Result<()> {
        let mut w = BinWriter::create(path, "DecisionTreeRegressor")?;
        w.write_option_usize(self.max_depth.as_ref())?;
        w.write_usize(self.min_samples_split)?;
        write_option_node(&mut w, self.root.as_ref())?;
        w.flush()
    }

    fn load(path: &str) -> Result<Self> {
        let mut r = BinReader::open(path, "DecisionTreeRegressor")?;
        let max_depth = r.read_option_usize()?;
        let min_samples_split = r.read_usize()?;
        let root = read_option_node(&mut r)?;
        Ok(Self {
            max_depth,
            min_samples_split,
            root,
        })
    }
}

// ── KMeans ──────────────────────────────────────────────────────────────

use crate::cluster::KMeans;

impl<T: Float> Persistable for KMeans<T> {
    fn save(&self, path: &str) -> Result<()> {
        let mut w = BinWriter::create(path, "KMeans")?;
        w.write_usize(self.n_clusters)?;
        w.write_usize(self.max_iter)?;
        w.write_float(self.tol)?;
        w.write_usize(self.n_init)?;
        w.write_f64(self.seed as f64)?;
        w.write_option_vec_float(self.centroids.as_ref())?;
        w.write_usize(self.n_features)?;
        w.write_option_float(self.inertia.as_ref())?;
        w.flush()
    }

    fn load(path: &str) -> Result<Self> {
        let mut r = BinReader::open(path, "KMeans")?;
        let n_clusters = r.read_usize()?;
        let max_iter = r.read_usize()?;
        let tol = r.read_float()?;
        let n_init = r.read_usize()?;
        let seed = r.read_f64()? as u64;
        let centroids = r.read_option_vec_float()?;
        let n_features = r.read_usize()?;
        let inertia = r.read_option_float()?;
        Ok(Self {
            n_clusters,
            max_iter,
            tol,
            n_init,
            seed,
            centroids,
            n_features,
            inertia,
        })
    }
}

// ── Preprocessing ───────────────────────────────────────────────────────

use crate::preprocessing::{MinMaxScaler, StandardScaler};

impl<T: Float> Persistable for StandardScaler<T> {
    fn save(&self, path: &str) -> Result<()> {
        let mut w = BinWriter::create(path, "StandardScaler")?;
        w.write_option_vec_float(self.mean.as_ref())?;
        w.write_option_vec_float(self.std.as_ref())?;
        w.flush()
    }

    fn load(path: &str) -> Result<Self> {
        let mut r = BinReader::open(path, "StandardScaler")?;
        let mean = r.read_option_vec_float()?;
        let std = r.read_option_vec_float()?;
        Ok(Self { mean, std })
    }
}

impl<T: Float> Persistable for MinMaxScaler<T> {
    fn save(&self, path: &str) -> Result<()> {
        let mut w = BinWriter::create(path, "MinMaxScaler")?;
        w.write_option_vec_float(self.min.as_ref())?;
        w.write_option_vec_float(self.range.as_ref())?;
        w.flush()
    }

    fn load(path: &str) -> Result<Self> {
        let mut r = BinReader::open(path, "MinMaxScaler")?;
        let min = r.read_option_vec_float()?;
        let range = r.read_option_vec_float()?;
        Ok(Self { min, range })
    }
}

// ── PCA ─────────────────────────────────────────────────────────────────

use crate::decomposition::PCA;

impl<T: Float> Persistable for PCA<T> {
    fn save(&self, path: &str) -> Result<()> {
        let mut w = BinWriter::create(path, "PCA")?;
        w.write_usize(self.n_components)?;
        w.write_option_vec_float(self.mean.as_ref())?;
        w.write_option_vec_float(self.components.as_ref())?;
        w.write_option_vec_float(self.explained_variance.as_ref())?;
        w.write_option_float(self.total_variance.as_ref())?;
        w.write_usize(self.n_features)?;
        w.flush()
    }

    fn load(path: &str) -> Result<Self> {
        let mut r = BinReader::open(path, "PCA")?;
        let n_components = r.read_usize()?;
        let mean = r.read_option_vec_float()?;
        let components = r.read_option_vec_float()?;
        let explained_variance = r.read_option_vec_float()?;
        let total_variance = r.read_option_float()?;
        let n_features = r.read_usize()?;
        Ok(Self {
            n_components,
            mean,
            components,
            explained_variance,
            total_variance,
            n_features,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scivex_core::Tensor;

    fn temp_path(name: &str) -> String {
        format!("/tmp/scivex_test_{name}_{}.bin", std::process::id())
    }

    #[test]
    fn test_linear_regression_roundtrip() {
        use crate::traits::Predictor;
        let mut model = LinearRegression::<f64>::new();
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![4]).unwrap();
        model.fit(&x, &y).unwrap();

        let path = temp_path("linreg");
        model.save(&path).unwrap();
        let loaded = LinearRegression::<f64>::load(&path).unwrap();

        let x_test = Tensor::from_vec(vec![5.0], vec![1, 1]).unwrap();
        let pred_orig = model.predict(&x_test).unwrap();
        let pred_loaded = loaded.predict(&x_test).unwrap();
        assert!(
            (pred_orig.as_slice()[0] - pred_loaded.as_slice()[0]).abs() < 1e-10,
            "predictions should match"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ridge_roundtrip() {
        let path = temp_path("ridge");
        let model = Ridge::<f64> {
            alpha: 0.5,
            weights: Some(vec![1.0, 2.0, 3.0]),
            bias: Some(0.5),
        };
        model.save(&path).unwrap();
        let loaded = Ridge::<f64>::load(&path).unwrap();
        assert!((loaded.alpha - 0.5).abs() < 1e-10);
        assert_eq!(loaded.weights.as_ref().unwrap().len(), 3);
        assert!((loaded.bias.unwrap() - 0.5).abs() < 1e-10);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_decision_tree_roundtrip() {
        use crate::traits::Predictor;
        let mut tree = DecisionTreeClassifier::<f64>::new(None, 2);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let y = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], vec![4]).unwrap();
        tree.fit(&x, &y).unwrap();

        let path = temp_path("dtree");
        tree.save(&path).unwrap();
        let loaded = DecisionTreeClassifier::<f64>::load(&path).unwrap();

        let x_test = Tensor::from_vec(vec![1.5, 3.5], vec![2, 1]).unwrap();
        let pred_orig = tree.predict(&x_test).unwrap();
        let pred_loaded = loaded.predict(&x_test).unwrap();
        assert_eq!(pred_orig.as_slice(), pred_loaded.as_slice());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_kmeans_roundtrip() {
        let path = temp_path("kmeans");
        let model = KMeans::<f64> {
            n_clusters: 3,
            max_iter: 100,
            tol: 1e-4,
            n_init: 10,
            seed: 42,
            centroids: Some(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            n_features: 2,
            inertia: Some(1.5),
        };
        model.save(&path).unwrap();
        let loaded = KMeans::<f64>::load(&path).unwrap();
        assert_eq!(loaded.n_clusters, 3);
        assert_eq!(loaded.centroids.as_ref().unwrap().len(), 6);
        assert!((loaded.inertia.unwrap() - 1.5).abs() < 1e-10);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_scaler_roundtrip() {
        let path = temp_path("scaler");
        let model = StandardScaler::<f64> {
            mean: Some(vec![1.0, 2.0]),
            std: Some(vec![0.5, 1.5]),
        };
        model.save(&path).unwrap();
        let loaded = StandardScaler::<f64>::load(&path).unwrap();
        assert_eq!(loaded.mean.as_ref().unwrap(), &[1.0, 2.0]);
        assert_eq!(loaded.std.as_ref().unwrap(), &[0.5, 1.5]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_wrong_model_type() {
        let path = temp_path("wrong_type");
        let model = StandardScaler::<f64> {
            mean: Some(vec![1.0]),
            std: Some(vec![1.0]),
        };
        model.save(&path).unwrap();
        // Try loading as wrong type
        assert!(LinearRegression::<f64>::load(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_file_not_found() {
        assert!(LinearRegression::<f64>::load("/tmp/nonexistent_svex_file.bin").is_err());
    }

    #[test]
    fn test_pca_roundtrip() {
        let path = temp_path("pca");
        let model = PCA::<f64> {
            n_components: 2,
            mean: Some(vec![1.0, 2.0, 3.0]),
            components: Some(vec![0.5, 0.5, 0.5, -0.5, 0.5, -0.5]),
            explained_variance: Some(vec![2.0, 1.0]),
            total_variance: Some(3.5),
            n_features: 3,
        };
        model.save(&path).unwrap();
        let loaded = PCA::<f64>::load(&path).unwrap();
        assert_eq!(loaded.n_components, 2);
        assert_eq!(loaded.n_features, 3);
        assert_eq!(loaded.components.as_ref().unwrap().len(), 6);
        std::fs::remove_file(&path).ok();
    }
}
