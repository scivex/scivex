//! Lightweight in-process model inference server with batching support.
//!
//! This module provides an [`InferenceServer`] that wraps any model implementing
//! [`InferenceModel`] and adds batched inference, latency tracking, and request
//! statistics. No external HTTP or async dependencies are required — everything
//! runs in-process.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use scivex_core::{Float, Tensor};

use crate::error::{NnError, Result};

// ---------------------------------------------------------------------------
// InferenceModel trait
// ---------------------------------------------------------------------------

/// A model that can perform inference (forward pass without gradient tracking).
pub trait InferenceModel<T: Float>: Send + Sync {
    /// Run inference on a single input batch.
    fn predict(&self, input: &Tensor<T>) -> Result<Tensor<T>>;

    /// Return the expected input shape (excluding batch dimension).
    fn input_shape(&self) -> &[usize];

    /// Return the output shape (excluding batch dimension).
    fn output_shape(&self) -> &[usize];

    /// Model name/identifier.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// InferenceConfig
// ---------------------------------------------------------------------------

/// Configuration for the inference server.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum batch size for batched inference.
    pub max_batch_size: usize,
    /// Model name.
    pub model_name: String,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            model_name: String::from("model"),
        }
    }
}

// ---------------------------------------------------------------------------
// Request / Response
// ---------------------------------------------------------------------------

/// Request for inference.
pub struct InferenceRequest<T: Float> {
    /// Input tensor (single sample or batch).
    pub input: Tensor<T>,
    /// Optional request ID for tracking.
    pub request_id: Option<String>,
}

/// Response from inference.
pub struct InferenceResponse<T: Float> {
    /// Output tensor.
    pub output: Tensor<T>,
    /// Request ID (if provided in request).
    pub request_id: Option<String>,
    /// Inference time in microseconds.
    pub latency_us: u64,
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Model inference statistics.
#[derive(Debug, Clone)]
pub struct InferenceStats {
    /// Total number of requests processed.
    pub total_requests: usize,
    /// Total number of samples processed.
    pub total_samples: usize,
    /// Average latency in microseconds.
    pub avg_latency_us: f64,
    /// Maximum latency in microseconds.
    pub max_latency_us: u64,
    /// Minimum latency in microseconds.
    pub min_latency_us: u64,
}

/// Internal mutable stats accumulator.
struct InferenceStatsInner {
    total_requests: usize,
    total_samples: usize,
    total_latency_us: u64,
    max_latency_us: u64,
    min_latency_us: u64,
}

impl InferenceStatsInner {
    fn new() -> Self {
        Self {
            total_requests: 0,
            total_samples: 0,
            total_latency_us: 0,
            max_latency_us: 0,
            min_latency_us: u64::MAX,
        }
    }

    fn record(&mut self, samples: usize, latency_us: u64) {
        self.total_requests += 1;
        self.total_samples += samples;
        self.total_latency_us += latency_us;
        if latency_us > self.max_latency_us {
            self.max_latency_us = latency_us;
        }
        if latency_us < self.min_latency_us {
            self.min_latency_us = latency_us;
        }
    }

    fn snapshot(&self) -> InferenceStats {
        let avg = if self.total_requests > 0 {
            self.total_latency_us as f64 / self.total_requests as f64
        } else {
            0.0
        };
        InferenceStats {
            total_requests: self.total_requests,
            total_samples: self.total_samples,
            avg_latency_us: avg,
            max_latency_us: self.max_latency_us,
            min_latency_us: if self.total_requests == 0 {
                0
            } else {
                self.min_latency_us
            },
        }
    }
}

// ---------------------------------------------------------------------------
// InferenceServer
// ---------------------------------------------------------------------------

/// In-process model inference server with batching support.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use scivex_core::Tensor;
/// use scivex_nn::serve::{
///     FnModel, InferenceConfig, InferenceRequest, InferenceServer,
/// };
///
/// // Create a simple identity model.
/// let model = FnModel::<f64>::new(
///     |input| Ok(input.clone()),
///     vec![4],
///     vec![4],
///     "identity",
/// );
///
/// let server = InferenceServer::new(
///     Arc::new(model),
///     InferenceConfig::default(),
/// );
///
/// let input = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
/// let resp = server.infer(InferenceRequest {
///     input,
///     request_id: Some("req-1".into()),
/// }).unwrap();
///
/// assert_eq!(resp.output.shape(), &[1, 4]);
/// assert_eq!(resp.request_id.as_deref(), Some("req-1"));
/// ```
pub struct InferenceServer<T: Float> {
    model: Arc<dyn InferenceModel<T>>,
    config: InferenceConfig,
    stats: Mutex<InferenceStatsInner>,
}

impl<T: Float> InferenceServer<T> {
    /// Create a new inference server with the given model and configuration.
    pub fn new(model: Arc<dyn InferenceModel<T>>, config: InferenceConfig) -> Self {
        Self {
            model,
            config,
            stats: Mutex::new(InferenceStatsInner::new()),
        }
    }

    /// Run inference on a single input.
    pub fn infer(&self, request: InferenceRequest<T>) -> Result<InferenceResponse<T>> {
        let start = Instant::now();

        let samples = batch_size_of(&request.input);
        let output = self.model.predict(&request.input)?;

        let latency_us = start.elapsed().as_micros() as u64;

        if let Ok(mut stats) = self.stats.lock() {
            stats.record(samples, latency_us);
        }

        Ok(InferenceResponse {
            output,
            request_id: request.request_id,
            latency_us,
        })
    }

    /// Run batched inference on multiple inputs.
    ///
    /// Inputs are concatenated along the batch dimension (dim 0), a single
    /// forward pass is executed, and the output is split back into individual
    /// responses.
    pub fn infer_batch(
        &self,
        requests: Vec<InferenceRequest<T>>,
    ) -> Result<Vec<InferenceResponse<T>>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let start = Instant::now();

        // Normalise each input to have an explicit batch dimension.
        let mut batched_inputs: Vec<Tensor<T>> = Vec::with_capacity(requests.len());
        let mut batch_sizes: Vec<usize> = Vec::with_capacity(requests.len());
        let mut request_ids: Vec<Option<String>> = Vec::with_capacity(requests.len());

        // Determine reference shape (everything except batch dim).
        let first_shape = normalised_shape(&requests[0].input);
        let sample_shape: &[usize] = &first_shape[1..];

        for req in &requests {
            let norm = normalised_shape(&req.input);
            if norm[1..] != *sample_shape {
                return Err(NnError::ShapeMismatch {
                    expected: sample_shape.to_vec(),
                    got: norm[1..].to_vec(),
                });
            }
            batch_sizes.push(norm[0]);
        }

        // Collect request ids and inputs.
        for req in requests {
            request_ids.push(req.request_id);
            batched_inputs.push(normalise_input(req.input)?);
        }

        // Concatenate along batch dimension.
        let total_batch: usize = batch_sizes.iter().sum();
        let sample_numel: usize = sample_shape.iter().product();
        let mut concat_data: Vec<T> = Vec::with_capacity(total_batch * sample_numel);
        for inp in &batched_inputs {
            concat_data.extend_from_slice(inp.as_slice());
        }
        let mut concat_shape = vec![total_batch];
        concat_shape.extend_from_slice(sample_shape);
        let concat_tensor = Tensor::from_vec(concat_data, concat_shape)?;

        // Single forward pass.
        let output = self.model.predict(&concat_tensor)?;

        let latency_us = start.elapsed().as_micros() as u64;

        // Split output back.
        let output_data = output.as_slice();
        let output_shape = output.shape().to_vec();
        let out_sample_shape = &output_shape[1..];
        let out_sample_numel: usize = out_sample_shape.iter().product();

        let mut responses = Vec::with_capacity(batch_sizes.len());
        let mut offset = 0usize;
        let per_request_latency = latency_us / batch_sizes.len() as u64;

        for (i, &bs) in batch_sizes.iter().enumerate() {
            let n = bs * out_sample_numel;
            let chunk = output_data[offset..offset + n].to_vec();
            let mut chunk_shape = vec![bs];
            chunk_shape.extend_from_slice(out_sample_shape);
            let out_tensor = Tensor::from_vec(chunk, chunk_shape)?;
            responses.push(InferenceResponse {
                output: out_tensor,
                request_id: request_ids[i].clone(),
                latency_us: per_request_latency,
            });
            offset += n;
        }

        // Update stats.
        if let Ok(mut stats) = self.stats.lock() {
            stats.record(total_batch, latency_us);
        }

        Ok(responses)
    }

    /// Get inference statistics.
    pub fn stats(&self) -> InferenceStats {
        self.stats
            .lock()
            .map(|s| s.snapshot())
            .unwrap_or(InferenceStats {
                total_requests: 0,
                total_samples: 0,
                avg_latency_us: 0.0,
                max_latency_us: 0,
                min_latency_us: 0,
            })
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            *stats = InferenceStatsInner::new();
        }
    }

    /// Get the server configuration.
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// FnModel
// ---------------------------------------------------------------------------

/// Inference function type alias.
type InferenceFn<T> = Box<dyn Fn(&Tensor<T>) -> Result<Tensor<T>> + Send + Sync>;

/// A simple wrapper that turns any closure into an [`InferenceModel`].
pub struct FnModel<T: Float> {
    func: InferenceFn<T>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    name: String,
}

impl<T: Float> FnModel<T> {
    /// Create a new `FnModel` from a closure.
    pub fn new<F>(func: F, input_shape: Vec<usize>, output_shape: Vec<usize>, name: &str) -> Self
    where
        F: Fn(&Tensor<T>) -> Result<Tensor<T>> + Send + Sync + 'static,
    {
        Self {
            func: Box::new(func),
            input_shape,
            output_shape,
            name: name.to_string(),
        }
    }
}

impl<T: Float> InferenceModel<T> for FnModel<T> {
    fn predict(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        (self.func)(input)
    }

    fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the batch size (first dimension, or 1 for 1-D tensors).
fn batch_size_of<T: Float>(t: &Tensor<T>) -> usize {
    let s = t.shape();
    if s.len() <= 1 { 1 } else { s[0] }
}

/// Return the shape with an explicit batch dim (prepend 1 for 1-D inputs).
fn normalised_shape<T: Float>(t: &Tensor<T>) -> Vec<usize> {
    let s = t.shape();
    if s.len() <= 1 {
        let mut v = vec![1];
        v.extend_from_slice(s);
        v
    } else {
        s.to_vec()
    }
}

/// If the tensor is 1-D, reshape it to [1, d]. Otherwise return as-is.
fn normalise_input<T: Float>(t: Tensor<T>) -> Result<Tensor<T>> {
    if t.shape().len() <= 1 {
        let d = t.numel();
        let data = t.as_slice().to_vec();
        Tensor::from_vec(data, vec![1, d]).map_err(Into::into)
    } else {
        Ok(t)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_model() -> FnModel<f64> {
        FnModel::new(|input| Ok(input.clone()), vec![4], vec![4], "identity")
    }

    fn doubling_model() -> FnModel<f64> {
        FnModel::new(
            |input| {
                let data: Vec<f64> = input.as_slice().iter().map(|&x| x * 2.0).collect();
                Ok(Tensor::from_vec(data, input.shape().to_vec())?)
            },
            vec![4],
            vec![4],
            "doubler",
        )
    }

    #[test]
    fn test_inference_server_basic() {
        let model = identity_model();
        let server = InferenceServer::new(Arc::new(model), InferenceConfig::default());

        let input = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let resp = server
            .infer(InferenceRequest {
                input: input.clone(),
                request_id: Some("r1".into()),
            })
            .unwrap();

        assert_eq!(resp.output.shape(), &[1, 4]);
        assert_eq!(resp.output.as_slice(), input.as_slice());
        assert_eq!(resp.request_id.as_deref(), Some("r1"));
    }

    #[test]
    fn test_inference_batch() {
        let model = doubling_model();
        let server = InferenceServer::new(Arc::new(model), InferenceConfig::default());

        let requests = vec![
            InferenceRequest {
                input: Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap(),
                request_id: Some("a".into()),
            },
            InferenceRequest {
                input: Tensor::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![1, 4]).unwrap(),
                request_id: Some("b".into()),
            },
        ];

        let responses = server.infer_batch(requests).unwrap();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0].output.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(responses[1].output.as_slice(), &[10.0, 12.0, 14.0, 16.0]);
        assert_eq!(responses[0].request_id.as_deref(), Some("a"));
        assert_eq!(responses[1].request_id.as_deref(), Some("b"));
    }

    #[test]
    fn test_inference_stats() {
        let model = identity_model();
        let server = InferenceServer::new(Arc::new(model), InferenceConfig::default());

        // Initially empty.
        let s = server.stats();
        assert_eq!(s.total_requests, 0);
        assert_eq!(s.total_samples, 0);

        // Run two inferences.
        for _ in 0..3 {
            let input = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
            server
                .infer(InferenceRequest {
                    input,
                    request_id: None,
                })
                .unwrap();
        }

        let s = server.stats();
        assert_eq!(s.total_requests, 3);
        assert_eq!(s.total_samples, 3);
        assert!(s.avg_latency_us >= 0.0);

        // Reset.
        server.reset_stats();
        let s = server.stats();
        assert_eq!(s.total_requests, 0);
    }

    #[test]
    fn test_inference_empty_batch() {
        let model = identity_model();
        let server = InferenceServer::new(Arc::new(model), InferenceConfig::default());

        let responses = server.infer_batch(Vec::new()).unwrap();
        assert!(responses.is_empty());
    }

    #[test]
    fn test_fn_model() {
        let model = FnModel::<f64>::new(
            |input| {
                let s: f64 = input.as_slice().iter().sum();
                Ok(Tensor::from_vec(vec![s], vec![1])?)
            },
            vec![3],
            vec![1],
            "summer",
        );

        assert_eq!(model.name(), "summer");
        assert_eq!(model.input_shape(), &[3]);
        assert_eq!(model.output_shape(), &[1]);

        let input = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let output = model.predict(&input).unwrap();
        assert_eq!(output.as_slice(), &[6.0]);
    }
}
