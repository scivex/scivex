//! Parse protobuf-encoded ONNX bytes into our IR types.
//!
//! The ONNX file format is a protobuf `ModelProto`. We parse just enough of the
//! protobuf schema to reconstruct the model graph, nodes, and tensors.

use crate::error::Result;
use crate::onnx::ir::{
    OnnxAttribute, OnnxAttributeValue, OnnxDataType, OnnxGraph, OnnxModel, OnnxNode,
    OnnxOpsetImport, OnnxTensor, OnnxValueInfo,
};
use crate::onnx::proto::{
    self, decode_packed_varints, get_all_bytes, get_all_strings, get_bytes, get_string, get_varint,
    FieldValue,
};

// -----------------------------------------------------------------------
// ONNX protobuf field numbers (from onnx.proto3)
// -----------------------------------------------------------------------

// ModelProto
const MODEL_IR_VERSION: u32 = 1;
const MODEL_OPSET_IMPORT: u32 = 8;
const MODEL_PRODUCER_NAME: u32 = 2;
const MODEL_MODEL_VERSION: u32 = 4;
const MODEL_GRAPH: u32 = 7;

// OpsetImport
const OPSET_DOMAIN: u32 = 1;
const OPSET_VERSION: u32 = 2;

// GraphProto
const GRAPH_NODE: u32 = 1;
const GRAPH_NAME: u32 = 2;
const GRAPH_INITIALIZER: u32 = 5;
const GRAPH_INPUT: u32 = 11;
const GRAPH_OUTPUT: u32 = 12;

// NodeProto
const NODE_INPUT: u32 = 1;
const NODE_OUTPUT: u32 = 2;
const NODE_NAME: u32 = 3;
const NODE_OP_TYPE: u32 = 4;
const NODE_ATTRIBUTE: u32 = 5;

// AttributeProto
const ATTR_NAME: u32 = 1;
const ATTR_F: u32 = 4;
const ATTR_I: u32 = 3;
const ATTR_S: u32 = 5;
const ATTR_T: u32 = 6;
const ATTR_G: u32 = 7;
const ATTR_FLOATS: u32 = 8;
const ATTR_INTS: u32 = 9;
const ATTR_STRINGS: u32 = 10;
const ATTR_TYPE: u32 = 2;

// TensorProto
const TENSOR_DIMS: u32 = 1;
const TENSOR_DATA_TYPE: u32 = 2;
const TENSOR_FLOAT_DATA: u32 = 4;
const TENSOR_INT32_DATA: u32 = 5;
const TENSOR_INT64_DATA: u32 = 7;
const TENSOR_RAW_DATA: u32 = 13;
const TENSOR_DOUBLE_DATA: u32 = 10;
const TENSOR_NAME: u32 = 8;

// ValueInfoProto
const VALUE_INFO_NAME: u32 = 1;
const VALUE_INFO_TYPE: u32 = 2;

// TypeProto
const TYPE_TENSOR_TYPE: u32 = 1;

// TypeProto.Tensor
const TENSOR_TYPE_ELEM_TYPE: u32 = 1;
const TENSOR_TYPE_SHAPE: u32 = 2;

// TensorShapeProto
const SHAPE_DIM: u32 = 1;

// TensorShapeProto.Dimension
const DIM_VALUE: u32 = 1;

// -----------------------------------------------------------------------
// Public entry point
// -----------------------------------------------------------------------

/// Parse an ONNX model from raw protobuf bytes.
pub fn load_onnx(bytes: &[u8]) -> Result<OnnxModel> {
    let fields = proto::parse_fields(bytes)?;
    parse_model(&fields)
}

// -----------------------------------------------------------------------
// Internal parsers
// -----------------------------------------------------------------------

fn parse_model(fields: &[proto::Field<'_>]) -> Result<OnnxModel> {
    let mut model = OnnxModel::new();

    if let Some(v) = get_varint(fields, MODEL_IR_VERSION) {
        #[allow(clippy::cast_possible_wrap)]
        {
            model.ir_version = v as i64;
        }
    }

    if let Some(v) = get_varint(fields, MODEL_MODEL_VERSION) {
        #[allow(clippy::cast_possible_wrap)]
        {
            model.model_version = v as i64;
        }
    }

    if let Some(s) = get_string(fields, MODEL_PRODUCER_NAME) {
        model.producer_name = s;
    }

    // Opset imports
    for opset_bytes in get_all_bytes(fields, MODEL_OPSET_IMPORT) {
        let opset_fields = proto::parse_fields(opset_bytes)?;
        let domain = get_string(&opset_fields, OPSET_DOMAIN).unwrap_or_default();
        #[allow(clippy::cast_possible_wrap)]
        let version = get_varint(&opset_fields, OPSET_VERSION).unwrap_or(0) as i64;
        model.opset_imports.push(OnnxOpsetImport { domain, version });
    }

    // Graph
    if let Some(graph_bytes) = get_bytes(fields, MODEL_GRAPH) {
        let graph_fields = proto::parse_fields(graph_bytes)?;
        model.graph = parse_graph(&graph_fields)?;
    }

    Ok(model)
}

fn parse_graph(fields: &[proto::Field<'_>]) -> Result<OnnxGraph> {
    let mut graph = OnnxGraph::new();

    if let Some(s) = get_string(fields, GRAPH_NAME) {
        graph.name = s;
    }

    for node_bytes in get_all_bytes(fields, GRAPH_NODE) {
        let node_fields = proto::parse_fields(node_bytes)?;
        graph.nodes.push(parse_node(&node_fields)?);
    }

    for init_bytes in get_all_bytes(fields, GRAPH_INITIALIZER) {
        let init_fields = proto::parse_fields(init_bytes)?;
        graph.initializers.push(parse_tensor(&init_fields)?);
    }

    for input_bytes in get_all_bytes(fields, GRAPH_INPUT) {
        let vi_fields = proto::parse_fields(input_bytes)?;
        graph.inputs.push(parse_value_info(&vi_fields)?);
    }

    for output_bytes in get_all_bytes(fields, GRAPH_OUTPUT) {
        let vi_fields = proto::parse_fields(output_bytes)?;
        graph.outputs.push(parse_value_info(&vi_fields)?);
    }

    Ok(graph)
}

fn parse_node(fields: &[proto::Field<'_>]) -> Result<OnnxNode> {
    let mut node = OnnxNode::new("");

    node.inputs = get_all_strings(fields, NODE_INPUT);
    node.outputs = get_all_strings(fields, NODE_OUTPUT);

    if let Some(s) = get_string(fields, NODE_NAME) {
        node.name = s;
    }
    if let Some(s) = get_string(fields, NODE_OP_TYPE) {
        node.op_type = s;
    }

    for attr_bytes in get_all_bytes(fields, NODE_ATTRIBUTE) {
        let attr_fields = proto::parse_fields(attr_bytes)?;
        node.attributes.push(parse_attribute(&attr_fields)?);
    }

    Ok(node)
}

#[allow(clippy::too_many_lines)]
fn parse_attribute(fields: &[proto::Field<'_>]) -> Result<OnnxAttribute> {
    let name = get_string(fields, ATTR_NAME).unwrap_or_default();

    #[allow(clippy::cast_possible_truncation)]
    let attr_type = get_varint(fields, ATTR_TYPE).unwrap_or(0) as u32;

    // ONNX AttributeProto.AttributeType enum:
    // 1=FLOAT, 2=INT, 3=STRING, 4=TENSOR, 5=GRAPH, 6=FLOATS, 7=INTS, 8=STRINGS
    let value = match attr_type {
        1 => {
            let mut val: f32 = 0.0;
            for f in fields {
                #[allow(clippy::collapsible_if)]
                if f.field_number == ATTR_F {
                    if let FieldValue::Fixed32(bits) = f.value {
                        val = f32::from_bits(bits);
                    }
                }
            }
            OnnxAttributeValue::Float(val)
        }
        2 => {
            #[allow(clippy::cast_possible_wrap)]
            let val = get_varint(fields, ATTR_I).unwrap_or(0) as i64;
            OnnxAttributeValue::Int(val)
        }
        3 => {
            let val = get_string(fields, ATTR_S).unwrap_or_default();
            OnnxAttributeValue::String(val)
        }
        4 => {
            if let Some(t_bytes) = get_bytes(fields, ATTR_T) {
                let t_fields = proto::parse_fields(t_bytes)?;
                OnnxAttributeValue::Tensor(parse_tensor(&t_fields)?)
            } else {
                OnnxAttributeValue::Tensor(OnnxTensor::new())
            }
        }
        5 => {
            if let Some(g_bytes) = get_bytes(fields, ATTR_G) {
                let g_fields = proto::parse_fields(g_bytes)?;
                OnnxAttributeValue::Graph(parse_graph(&g_fields)?)
            } else {
                OnnxAttributeValue::Graph(OnnxGraph::new())
            }
        }
        6 => {
            let mut floats = Vec::new();
            for f in fields {
                if f.field_number == ATTR_FLOATS {
                    match f.value {
                        FieldValue::Fixed32(bits) => {
                            floats.push(f32::from_bits(bits));
                        }
                        FieldValue::Bytes(b) => {
                            for chunk in b.chunks_exact(4) {
                                floats.push(f32::from_le_bytes([
                                    chunk[0], chunk[1], chunk[2], chunk[3],
                                ]));
                            }
                        }
                        _ => {}
                    }
                }
            }
            OnnxAttributeValue::Floats(floats)
        }
        7 => {
            let mut ints = Vec::new();
            for f in fields {
                if f.field_number == ATTR_INTS {
                    match &f.value {
                        FieldValue::Varint(v) => {
                            #[allow(clippy::cast_possible_wrap)]
                            ints.push(*v as i64);
                        }
                        FieldValue::Bytes(b) => {
                            let vals = decode_packed_varints(b)?;
                            for v in vals {
                                #[allow(clippy::cast_possible_wrap)]
                                ints.push(v as i64);
                            }
                        }
                        _ => {}
                    }
                }
            }
            OnnxAttributeValue::Ints(ints)
        }
        8 => {
            let vals = get_all_strings(fields, ATTR_STRINGS);
            OnnxAttributeValue::Strings(vals)
        }
        _ => {
            if let Some(t_bytes) = get_bytes(fields, ATTR_T) {
                let t_fields = proto::parse_fields(t_bytes)?;
                OnnxAttributeValue::Tensor(parse_tensor(&t_fields)?)
            } else if get_varint(fields, ATTR_I).is_some() {
                #[allow(clippy::cast_possible_wrap)]
                let val = get_varint(fields, ATTR_I).unwrap_or(0) as i64;
                OnnxAttributeValue::Int(val)
            } else {
                OnnxAttributeValue::Float(0.0)
            }
        }
    };

    Ok(OnnxAttribute { name, value })
}

#[allow(clippy::too_many_lines)]
fn parse_tensor(fields: &[proto::Field<'_>]) -> Result<OnnxTensor> {
    let mut tensor = OnnxTensor::new();

    if let Some(s) = get_string(fields, TENSOR_NAME) {
        tensor.name = s;
    }

    if let Some(dt) = get_varint(fields, TENSOR_DATA_TYPE) {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        if let Some(t) = OnnxDataType::from_i32(dt as i32) {
            tensor.data_type = t;
        }
    }

    parse_tensor_dims(fields, &mut tensor)?;
    parse_tensor_float_data(fields, &mut tensor);
    parse_tensor_int32_data(fields, &mut tensor)?;
    parse_tensor_int64_data(fields, &mut tensor)?;
    parse_tensor_double_data(fields, &mut tensor);

    if let Some(raw) = get_bytes(fields, TENSOR_RAW_DATA) {
        tensor.raw_data = raw.to_vec();
    }

    Ok(tensor)
}

fn parse_tensor_dims(fields: &[proto::Field<'_>], tensor: &mut OnnxTensor) -> Result<()> {
    for f in fields {
        if f.field_number == TENSOR_DIMS {
            match &f.value {
                FieldValue::Varint(v) => {
                    #[allow(clippy::cast_possible_wrap)]
                    tensor.dims.push(*v as i64);
                }
                FieldValue::Bytes(b) => {
                    let vals = decode_packed_varints(b)?;
                    for v in vals {
                        #[allow(clippy::cast_possible_wrap)]
                        tensor.dims.push(v as i64);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn parse_tensor_float_data(fields: &[proto::Field<'_>], tensor: &mut OnnxTensor) {
    for f in fields {
        if f.field_number == TENSOR_FLOAT_DATA {
            match &f.value {
                FieldValue::Fixed32(bits) => {
                    tensor.float_data.push(f32::from_bits(*bits));
                }
                FieldValue::Bytes(b) => {
                    for chunk in b.chunks_exact(4) {
                        tensor
                            .float_data
                            .push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                }
                _ => {}
            }
        }
    }
}

fn parse_tensor_int32_data(fields: &[proto::Field<'_>], tensor: &mut OnnxTensor) -> Result<()> {
    for f in fields {
        if f.field_number == TENSOR_INT32_DATA {
            match &f.value {
                FieldValue::Varint(v) => {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                    tensor.int32_data.push(*v as i32);
                }
                FieldValue::Bytes(b) => {
                    let vals = decode_packed_varints(b)?;
                    for v in vals {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                        tensor.int32_data.push(v as i32);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn parse_tensor_int64_data(fields: &[proto::Field<'_>], tensor: &mut OnnxTensor) -> Result<()> {
    for f in fields {
        if f.field_number == TENSOR_INT64_DATA {
            match &f.value {
                FieldValue::Varint(v) => {
                    #[allow(clippy::cast_possible_wrap)]
                    tensor.int64_data.push(*v as i64);
                }
                FieldValue::Bytes(b) => {
                    let vals = decode_packed_varints(b)?;
                    for v in vals {
                        #[allow(clippy::cast_possible_wrap)]
                        tensor.int64_data.push(v as i64);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn parse_tensor_double_data(fields: &[proto::Field<'_>], tensor: &mut OnnxTensor) {
    for f in fields {
        if f.field_number == TENSOR_DOUBLE_DATA {
            match &f.value {
                FieldValue::Fixed64(bits) => {
                    tensor.double_data.push(f64::from_bits(*bits));
                }
                FieldValue::Bytes(b) => {
                    for chunk in b.chunks_exact(8) {
                        tensor.double_data.push(f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]));
                    }
                }
                _ => {}
            }
        }
    }
}

fn parse_value_info(fields: &[proto::Field<'_>]) -> Result<OnnxValueInfo> {
    let name = get_string(fields, VALUE_INFO_NAME).unwrap_or_default();
    let mut data_type = OnnxDataType::Float;
    let mut shape = Vec::new();

    if let Some(type_bytes) = get_bytes(fields, VALUE_INFO_TYPE) {
        let type_fields = proto::parse_fields(type_bytes)?;

        if let Some(tt_bytes) = get_bytes(&type_fields, TYPE_TENSOR_TYPE) {
            let tt_fields = proto::parse_fields(tt_bytes)?;

            if let Some(et) = get_varint(&tt_fields, TENSOR_TYPE_ELEM_TYPE) {
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                if let Some(t) = OnnxDataType::from_i32(et as i32) {
                    data_type = t;
                }
            }

            if let Some(shape_bytes) = get_bytes(&tt_fields, TENSOR_TYPE_SHAPE) {
                let shape_fields = proto::parse_fields(shape_bytes)?;
                for dim_bytes in get_all_bytes(&shape_fields, SHAPE_DIM) {
                    let dim_fields = proto::parse_fields(dim_bytes)?;
                    #[allow(clippy::cast_possible_wrap)]
                    let dim_val = get_varint(&dim_fields, DIM_VALUE).unwrap_or(0) as i64;
                    shape.push(dim_val);
                }
            }
        }
    }

    Ok(OnnxValueInfo {
        name,
        data_type,
        shape,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::proto::encode_varint;

    /// Helper: encode a protobuf tag.
    fn encode_tag(field_number: u32, wire_type: u8) -> Vec<u8> {
        encode_varint(u64::from((field_number << 3) | u32::from(wire_type)))
    }

    /// Helper: encode a varint field.
    fn encode_varint_field(field_number: u32, value: u64) -> Vec<u8> {
        let mut out = encode_tag(field_number, 0);
        out.extend_from_slice(&encode_varint(value));
        out
    }

    /// Helper: encode a length-delimited field.
    fn encode_bytes_field(field_number: u32, data: &[u8]) -> Vec<u8> {
        let mut out = encode_tag(field_number, 2);
        out.extend_from_slice(&encode_varint(data.len() as u64));
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn test_parse_tensor_from_raw_bytes() {
        let mut tensor_bytes = Vec::new();

        tensor_bytes.extend_from_slice(&encode_varint_field(TENSOR_DIMS, 2));
        tensor_bytes.extend_from_slice(&encode_varint_field(TENSOR_DIMS, 2));
        tensor_bytes.extend_from_slice(&encode_varint_field(TENSOR_DATA_TYPE, 1));

        let mut raw = Vec::new();
        for &v in &[1.0_f32, 2.0, 3.0, 4.0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        tensor_bytes.extend_from_slice(&encode_bytes_field(TENSOR_RAW_DATA, &raw));

        let fields = proto::parse_fields(&tensor_bytes).unwrap();
        let tensor = parse_tensor(&fields).unwrap();

        assert_eq!(tensor.dims, vec![2, 2]);
        assert_eq!(tensor.data_type, OnnxDataType::Float);
        let f32_data = tensor.to_f32_vec();
        assert_eq!(f32_data, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
