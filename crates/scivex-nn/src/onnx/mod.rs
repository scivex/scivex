//! ONNX model loading and inference.
//!
//! This module provides a from-scratch ONNX runtime: it parses the protobuf
//! wire format without any external protobuf dependencies, builds an
//! intermediate representation of the computation graph, and executes it.
//!
//! # Example
//!
//! ```rust,ignore
//! use scivex_nn::onnx::{load_onnx, OnnxInferenceSession};
//!
//! let bytes = std::fs::read("model.onnx").unwrap();
//! let model = load_onnx(&bytes).unwrap();
//! let session = OnnxInferenceSession::<f32>::from_model(model).unwrap();
//!
//! let input = scivex_core::Tensor::from_vec(vec![1.0_f32; 784], vec![1, 784]).unwrap();
//! let outputs = session.run(&[("input", input)]).unwrap();
//! ```

mod executor;
pub mod ir;
mod parser;
pub(crate) mod proto;

pub use executor::OnnxInferenceSession;
pub use ir::{
    OnnxAttribute, OnnxAttributeValue, OnnxDataType, OnnxGraph, OnnxModel, OnnxNode,
    OnnxOpsetImport, OnnxTensor, OnnxValueInfo,
};
pub use parser::load_onnx;
