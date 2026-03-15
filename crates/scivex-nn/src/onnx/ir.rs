//! ONNX intermediate representation types.
//!
//! These Rust types mirror the ONNX protobuf schema at a structural level but
//! are decoded from raw bytes by our own minimal protobuf parser.

/// ONNX tensor element data types (matches `onnx::TensorProto::DataType` enum values).
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxDataType {
    /// 32-bit float.
    Float = 1,
    /// 8-bit unsigned integer.
    Uint8 = 2,
    /// 8-bit signed integer.
    Int8 = 3,
    /// 16-bit unsigned integer.
    Uint16 = 4,
    /// 16-bit signed integer.
    Int16 = 5,
    /// 32-bit signed integer.
    Int32 = 6,
    /// 64-bit signed integer.
    Int64 = 7,
    /// UTF-8 string.
    String = 8,
    /// Boolean.
    Bool = 9,
    /// 64-bit float.
    Double = 11,
    /// 32-bit unsigned integer.
    Uint32 = 12,
    /// 64-bit unsigned integer.
    Uint64 = 13,
}

impl OnnxDataType {
    /// Decode from the protobuf enum integer.
    pub(crate) fn from_i32(v: i32) -> Option<Self> {
        match v {
            1 => Some(Self::Float),
            2 => Some(Self::Uint8),
            3 => Some(Self::Int8),
            4 => Some(Self::Uint16),
            5 => Some(Self::Int16),
            6 => Some(Self::Int32),
            7 => Some(Self::Int64),
            8 => Some(Self::String),
            9 => Some(Self::Bool),
            11 => Some(Self::Double),
            12 => Some(Self::Uint32),
            13 => Some(Self::Uint64),
            _ => None,
        }
    }
}

/// An ONNX tensor (weight / initializer / constant).
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    /// Optional tensor name.
    pub name: String,
    /// Element data type.
    pub data_type: OnnxDataType,
    /// Dimensions.
    pub dims: Vec<i64>,
    /// Raw float data (populated for float tensors).
    pub float_data: Vec<f32>,
    /// Raw double data.
    pub double_data: Vec<f64>,
    /// Raw int32 data.
    pub int32_data: Vec<i32>,
    /// Raw int64 data.
    pub int64_data: Vec<i64>,
    /// Raw bytes (used when data is stored in the `raw_data` field).
    pub raw_data: Vec<u8>,
}

impl OnnxTensor {
    /// Create a new empty tensor.
    pub fn new() -> Self {
        Self {
            name: String::new(),
            data_type: OnnxDataType::Float,
            dims: Vec::new(),
            float_data: Vec::new(),
            double_data: Vec::new(),
            int32_data: Vec::new(),
            int64_data: Vec::new(),
            raw_data: Vec::new(),
        }
    }

    /// Return float data, decoding from `raw_data` if `float_data` is empty.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        if !self.float_data.is_empty() {
            return self.float_data.clone();
        }
        if !self.raw_data.is_empty() {
            match self.data_type {
                OnnxDataType::Float => {
                    return self
                        .raw_data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                }
                OnnxDataType::Double => {
                    return self
                        .raw_data
                        .chunks_exact(8)
                        .map(|c| {
                            #[allow(clippy::cast_possible_truncation)]
                            let v = f64::from_le_bytes([
                                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                            ]) as f32;
                            v
                        })
                        .collect();
                }
                _ => {}
            }
        }
        if !self.double_data.is_empty() {
            #[allow(clippy::cast_possible_truncation)]
            return self.double_data.iter().map(|&v| v as f32).collect();
        }
        if !self.int64_data.is_empty() {
            #[allow(clippy::cast_possible_truncation)]
            return self.int64_data.iter().map(|&v| v as f32).collect();
        }
        if !self.int32_data.is_empty() {
            #[allow(clippy::cast_precision_loss)]
            return self.int32_data.iter().map(|&v| v as f32).collect();
        }
        Vec::new()
    }

    /// Return the dimensions as `Vec<usize>`.
    pub fn dims_usize(&self) -> Vec<usize> {
        #[allow(clippy::cast_sign_loss)]
        self.dims.iter().map(|&d| d as usize).collect()
    }
}

impl Default for OnnxTensor {
    fn default() -> Self {
        Self::new()
    }
}

/// Attribute value kinds in an ONNX node.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub enum OnnxAttributeValue {
    /// A single float.
    Float(f32),
    /// A single integer.
    Int(i64),
    /// A UTF-8 string.
    String(String),
    /// A tensor value.
    Tensor(OnnxTensor),
    /// A sub-graph.
    Graph(OnnxGraph),
    /// A list of floats.
    Floats(Vec<f32>),
    /// A list of integers.
    Ints(Vec<i64>),
    /// A list of strings.
    Strings(Vec<String>),
}

/// An attribute on an ONNX node.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct OnnxAttribute {
    /// Attribute name.
    pub name: String,
    /// Attribute value.
    pub value: OnnxAttributeValue,
}

/// A single node (operator) in the ONNX computation graph.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct OnnxNode {
    /// Operator type (e.g. "Add", "MatMul", "Relu").
    pub op_type: String,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// Node name (optional, for debugging).
    pub name: String,
    /// Attributes for the operator.
    pub attributes: Vec<OnnxAttribute>,
}

impl OnnxNode {
    /// Create a new node.
    pub fn new(op_type: &str) -> Self {
        Self {
            op_type: op_type.to_owned(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            name: String::new(),
            attributes: Vec::new(),
        }
    }

    /// Look up an attribute by name.
    pub fn get_attr(&self, name: &str) -> Option<&OnnxAttributeValue> {
        self.attributes
            .iter()
            .find(|a| a.name == name)
            .map(|a| &a.value)
    }

    /// Get an integer attribute, returning a default if missing.
    pub fn get_int_attr(&self, name: &str, default: i64) -> i64 {
        match self.get_attr(name) {
            Some(OnnxAttributeValue::Int(v)) => *v,
            _ => default,
        }
    }

    /// Get a float attribute, returning a default if missing.
    pub fn get_float_attr(&self, name: &str, default: f32) -> f32 {
        match self.get_attr(name) {
            Some(OnnxAttributeValue::Float(v)) => *v,
            _ => default,
        }
    }

    /// Get a list-of-ints attribute, returning empty if missing.
    pub fn get_ints_attr(&self, name: &str) -> Vec<i64> {
        match self.get_attr(name) {
            Some(OnnxAttributeValue::Ints(v)) => v.clone(),
            _ => Vec::new(),
        }
    }
}

/// A typed input/output descriptor for the graph.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct OnnxValueInfo {
    /// Tensor name.
    pub name: String,
    /// Element data type.
    pub data_type: OnnxDataType,
    /// Shape (may contain -1 for dynamic dims).
    pub shape: Vec<i64>,
}

/// An ONNX computation graph.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct OnnxGraph {
    /// Graph name.
    pub name: String,
    /// Computation nodes.
    pub nodes: Vec<OnnxNode>,
    /// Weight / constant initializers.
    pub initializers: Vec<OnnxTensor>,
    /// Graph inputs.
    pub inputs: Vec<OnnxValueInfo>,
    /// Graph outputs.
    pub outputs: Vec<OnnxValueInfo>,
}

impl OnnxGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            name: String::new(),
            nodes: Vec::new(),
            initializers: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
}

impl Default for OnnxGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// An ONNX opset import.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct OnnxOpsetImport {
    /// Domain (empty string = default ONNX domain).
    pub domain: String,
    /// Opset version.
    pub version: i64,
}

/// Top-level ONNX model.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct OnnxModel {
    /// IR version.
    pub ir_version: i64,
    /// Opset imports.
    pub opset_imports: Vec<OnnxOpsetImport>,
    /// The computation graph.
    pub graph: OnnxGraph,
    /// Producer name.
    pub producer_name: String,
    /// Model version.
    pub model_version: i64,
}

impl OnnxModel {
    /// Create a new empty model.
    pub fn new() -> Self {
        Self {
            ir_version: 0,
            opset_imports: Vec::new(),
            graph: OnnxGraph::new(),
            producer_name: String::new(),
            model_version: 0,
        }
    }
}

impl Default for OnnxModel {
    fn default() -> Self {
        Self::new()
    }
}
