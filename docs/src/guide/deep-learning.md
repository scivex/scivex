# Deep Learning with scivex-nn

The `scivex-nn` crate provides a complete deep learning framework built on
reverse-mode automatic differentiation. Everything -- tensors, autograd,
layers, optimizers, and serialization -- is implemented from scratch in Rust
with no external math library dependencies.

This guide covers the full workflow: building computation graphs, assembling
models, training them, and saving the results.

---

## Automatic Differentiation

### Variable: The Computation Graph Node

`Variable<T>` wraps a `Tensor<T>` and records operations to enable
reverse-mode autodiff. Create leaf variables, run operations, then call
`.backward()` to compute gradients.

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

// Create leaf variables with gradient tracking
let x = Variable::new(
    Tensor::<f64>::from_vec(vec![2.0, 3.0], vec![2]).unwrap(),
    true,  // requires_grad
);

let w = Variable::new(
    Tensor::<f64>::from_vec(vec![0.5, -1.0], vec![2]).unwrap(),
    true,
);
```

### Forward and Backward Passes

Operations on variables build a computation graph. Calling `.backward()` on
a scalar output propagates gradients to all ancestors.

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let x = Variable::new(
    Tensor::<f64>::from_vec(vec![2.0, 3.0], vec![2]).unwrap(),
    true,
);
let w = Variable::new(
    Tensor::<f64>::from_vec(vec![0.5, -1.0], vec![2]).unwrap(),
    true,
);

// Element-wise multiply, then sum to scalar
let product = mul(&x, &w);
let loss = sum(&product);

// Backward pass
loss.backward();

// Gradients: d(sum(x*w))/dx = w, d(sum(x*w))/dw = x
let x_grad = x.grad().unwrap();  // [0.5, -1.0]
let w_grad = w.grad().unwrap();  // [2.0, 3.0]
```

### Differentiable Operations

The `ops` module provides all the building blocks. Each records a `grad_fn`
closure for the backward pass.

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let a = Variable::new(Tensor::<f64>::ones(vec![2, 3]), true);
let b = Variable::new(Tensor::<f64>::ones(vec![2, 3]), true);

// Arithmetic (also available as operator overloads: +, -, *, unary -)
let c = add(&a, &b);
let d = sub(&a, &b);
let e = mul(&a, &b);
let f = neg(&a);
let g = scalar_mul(&a, 2.0);

// Matrix multiplication: [m, k] @ [k, n] -> [m, n]
let m1 = Variable::new(Tensor::<f64>::ones(vec![2, 3]), true);
let m2 = Variable::new(Tensor::<f64>::ones(vec![3, 4]), true);
let result = matmul(&m1, &m2);  // shape [2, 4]

// Reductions
let s = sum(&a);    // sum all elements to scalar
let m = mean(&a);   // mean of all elements

// Element-wise power
let squared = pow(&a, 2.0);
```

Operator overloads are supported, so you can write natural expressions:

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;

let a = Variable::new(Tensor::<f64>::ones(vec![3]), true);
let b = Variable::new(Tensor::<f64>::ones(vec![3]), true);

let c = &a + &b;
let d = &a - &b;
let e = &a * &b;
let f = -&a;
```

### Key Variable Methods

| Method | Description |
|--------|-------------|
| `Variable::new(tensor, requires_grad)` | Create a leaf variable |
| `.data()` | Clone the underlying tensor |
| `.shape()` | Get the tensor shape |
| `.grad()` | Get the accumulated gradient (if any) |
| `.backward()` | Run reverse-mode autodiff from this scalar |
| `.zero_grad()` | Reset gradient to `None` |
| `.detach()` | Create a new leaf with same data, no graph history |
| `.set_data(tensor)` | Replace the underlying tensor (used by optimizers) |
| `.requires_grad()` | Check if this variable tracks gradients |

---

## Activation Functions

Activation functions are available both as free functions (in the
`functional` module) and as stateless layers (for use in `Sequential`).

### Functional API

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let x = Variable::new(
    Tensor::<f64>::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4]).unwrap(),
    true,
);

let a = relu(&x);       // max(0, x) -> [0, 0, 1, 2]
let b = sigmoid(&x);    // 1 / (1 + exp(-x))
let c = tanh_fn(&x);    // (exp(x) - exp(-x)) / (exp(x) + exp(-x))

// Softmax requires 2-D input [batch, classes]
let logits = Variable::new(
    Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap(),
    true,
);
let probs = softmax(&logits).unwrap();         // rows sum to 1
let log_probs = log_softmax(&logits).unwrap();  // numerically stable
```

All activation functions are fully differentiable and participate in the
autograd graph.

### Layer API

Activation layers implement the `Layer` trait, so they can be used inside
`Sequential`:

```rust
use scivex_nn::prelude::*;

// These are zero-parameter layers
let relu_layer = ReLU;
let sigmoid_layer = Sigmoid;
let tanh_layer = Tanh;
```

---

## Layers

All layers implement the `Layer<T>` trait:

```rust
pub trait Layer<T: Float> {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>>;
    fn parameters(&self) -> Vec<Variable<T>>;
    fn train(&mut self);
    fn eval(&mut self);
}
```

### Linear (Fully Connected)

Computes `y = x @ W^T + b`. Weights are initialized with Kaiming uniform.

```rust
use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let mut rng = Rng::new(42);

// Linear(in_features, out_features, use_bias, rng)
let linear = Linear::<f64>::new(784, 128, true, &mut rng);

let x = Variable::new(Tensor::ones(vec![32, 784]), true);  // batch of 32
let y = linear.forward(&x).unwrap();  // shape: [32, 128]

// Access weight and bias
let weight = linear.weight();          // Variable with shape [128, 784]
let bias = linear.bias();             // Option<&Variable<T>>

// Parameters for optimizer
let params = linear.parameters();      // vec of 2 Variables (weight + bias)
```

### Conv2d (2-D Convolution)

Input shape: `[batch, in_channels, height, width]`. Uses im2col internally.

```rust
use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let mut rng = Rng::new(42);

// Conv2d(in_channels, out_channels, kernel_size, use_bias, rng)
let conv = Conv2d::<f64>::new(3, 16, (3, 3), true, &mut rng)
    .set_stride((1, 1))
    .set_padding((1, 1))
    .set_dilation((1, 1));

let x = Variable::new(Tensor::ones(vec![1, 3, 28, 28]), false);
let y = conv.forward(&x).unwrap();  // [1, 16, 28, 28] with padding=1
```

Conv1d and Conv3d are also available with analogous APIs.

### BatchNorm1d and BatchNorm2d

Normalizes features across the batch dimension. Uses batch statistics during
training and running statistics during evaluation.

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

// BatchNorm1d for input shape [batch, features]
let mut bn = BatchNorm1d::<f64>::new(128);

let x = Variable::new(Tensor::ones(vec![32, 128]), true);
let y = bn.forward(&x).unwrap();  // [32, 128], normalized

// Switch to eval mode (uses running statistics)
bn.eval();
let y_eval = bn.forward(&x).unwrap();

// Switch back to training
bn.train();
```

Parameters: `gamma` (scale) and `beta` (shift), both shape `[features]`.

### Dropout

Randomly zeroes elements during training with probability `p`. Uses inverted
dropout (scales by `1/(1-p)`) so no scaling is needed at evaluation time.

```rust
use scivex_core::random::Rng;
use scivex_nn::Variable;
use scivex_core::Tensor;
use scivex_nn::prelude::*;

let mut dropout = Dropout::<f64>::new(0.5, Rng::new(42));

let x = Variable::new(Tensor::ones(vec![32, 128]), true);

// Training: randomly zeroes ~50% of elements
let y_train = dropout.forward(&x).unwrap();

// Eval: identity function
dropout.eval();
let y_eval = dropout.forward(&x).unwrap();
```

Dropout has no learnable parameters.

### RNN, LSTM, and GRU

Recurrent layers expect input shape `[batch, seq_len * input_size]` and
return `[batch, seq_len * hidden_size]`.

```rust
use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let mut rng = Rng::new(42);

// SimpleRNN(input_size, hidden_size, seq_len, rng)
let rnn = SimpleRNN::<f64>::new(10, 20, 5, &mut rng);

// Input: [batch=4, seq_len * input_size = 5 * 10 = 50]
let x = Variable::new(Tensor::ones(vec![4, 50]), true);
let output = rnn.forward(&x).unwrap();  // [4, 5 * 20 = 100]

// LSTM(input_size, hidden_size, seq_len, rng)
let lstm = LSTM::<f64>::new(10, 20, 5, &mut rng);
let y = lstm.forward(&x).unwrap();

// GRU(input_size, hidden_size, seq_len, rng)
let gru = GRU::<f64>::new(10, 20, 5, &mut rng);
let z = gru.forward(&x).unwrap();
```

All recurrent layers implement backpropagation through time (BPTT).

### Embedding, LayerNorm, Flatten, and Pooling

Additional layers available in the `layer` module:

```rust
use scivex_core::random::Rng;
use scivex_nn::prelude::*;

let mut rng = Rng::new(42);

// Embedding(vocab_size, embed_dim, rng)
let embedding = Embedding::<f64>::new(10000, 256, &mut rng);

// LayerNorm(features)
let ln = LayerNorm::<f64>::new(256);

// Flatten — reshapes to [batch, -1]
let flatten = Flatten;

// Pooling layers
let maxpool = MaxPool2d::new(2, 2);    // (kernel_size, stride)
let avgpool = AvgPool2d::new(2, 2);
```

### Attention and Transformer Layers

Multi-head attention and full transformer encoder/decoder layers are
included:

```rust
use scivex_core::random::Rng;
use scivex_nn::prelude::*;

let mut rng = Rng::new(42);

// MultiHeadAttention(d_model, num_heads, rng)
let mha = MultiHeadAttention::<f64>::new(512, 8, &mut rng);

// TransformerEncoderLayer(d_model, num_heads, d_ff, dropout_p, rng)
let encoder = TransformerEncoderLayer::<f64>::new(512, 8, 2048, 0.1, &mut rng);

// TransformerDecoderLayer(d_model, num_heads, d_ff, dropout_p, rng)
let decoder = TransformerDecoderLayer::<f64>::new(512, 8, 2048, 0.1, &mut rng);

// Positional encodings
let sinusoidal = SinusoidalPositionalEncoding::<f64>::new(512, 1000);
let rope = RotaryPositionalEncoding::<f64>::new(512, 1000);

// Causal mask for autoregressive decoding
let mask = causal_mask::<f64>(10);  // 10x10 upper-triangular mask
```

---

## Building Models with Sequential

`Sequential` chains layers in order. It implements `Layer`, so it can be
nested and its parameters are collected automatically.

```rust
use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let mut rng = Rng::new(42);

let model: Sequential<f64> = Sequential::new(vec![
    Box::new(Linear::new(784, 256, true, &mut rng)),
    Box::new(ReLU),
    Box::new(Dropout::new(0.5, Rng::new(1))),
    Box::new(Linear::new(256, 128, true, &mut rng)),
    Box::new(ReLU),
    Box::new(Linear::new(128, 10, true, &mut rng)),
]);

// Forward pass
let x = Variable::new(Tensor::ones(vec![32, 784]), false);
let logits = model.forward(&x).unwrap();  // [32, 10]

// All parameters from all layers
let params = model.parameters();  // 6 Variables (3 weights + 3 biases)
```

You can also build incrementally:

```rust
use scivex_core::random::Rng;
use scivex_nn::prelude::*;

let mut rng = Rng::new(42);

let mut model: Sequential<f64> = Sequential::new(vec![]);
model.push(Box::new(Linear::new(784, 128, true, &mut rng)));
model.push(Box::new(ReLU));
model.push(Box::new(Linear::new(128, 10, true, &mut rng)));
```

### Train/Eval Mode

Layers like `Dropout` and `BatchNorm` behave differently during training and
evaluation. Toggle mode on the entire model:

```rust
use scivex_core::random::Rng;
use scivex_nn::prelude::*;

let mut rng = Rng::new(42);
let mut model: Sequential<f64> = Sequential::new(vec![
    Box::new(Linear::new(10, 5, true, &mut rng)),
    Box::new(Dropout::new(0.2, Rng::new(1))),
]);

model.train();  // training mode (dropout active)
model.eval();   // evaluation mode (dropout disabled)
```

---

## Loss Functions

All loss functions return a scalar `Variable` that can be backpropagated.

### Mean Squared Error

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let pred = Variable::new(
    Tensor::<f64>::from_vec(vec![1.0, 2.0], vec![2]).unwrap(),
    true,
);
let target = Variable::new(
    Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap(),
    false,
);

let loss = mse_loss(&pred, &target).unwrap();
// mean((1-3)^2 + (2-4)^2) = 4.0
loss.backward();
```

### Cross-Entropy (Classification)

Takes logits `[batch, classes]` and integer targets `[batch]` (as floats).

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let logits = Variable::new(
    Tensor::<f64>::from_vec(
        vec![2.0, 1.0, 0.1,   // sample 0: class 0 most likely
             0.1, 0.5, 3.0],  // sample 1: class 2 most likely
        vec![2, 3],
    ).unwrap(),
    true,
);
let targets = Variable::new(
    Tensor::from_vec(vec![0.0, 2.0], vec![2]).unwrap(),  // correct classes
    false,
);

let loss = cross_entropy_loss(&logits, &targets).unwrap();
loss.backward();
```

### Binary Cross-Entropy

For binary classification with sigmoid outputs in `[0, 1]`.

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let pred = Variable::new(
    Tensor::<f64>::from_vec(vec![0.9, 0.1, 0.8], vec![3]).unwrap(),
    true,
);
let target = Variable::new(
    Tensor::from_vec(vec![1.0, 0.0, 1.0], vec![3]).unwrap(),
    false,
);

let loss = bce_loss(&pred, &target).unwrap();
loss.backward();
```

### Other Loss Functions

| Function | Description |
|----------|-------------|
| `huber_loss(pred, target, delta)` | Huber loss (smooth L1 with threshold `delta`) |
| `smooth_l1_loss(pred, target, beta)` | Smooth L1 loss parameterized by `beta` |
| `focal_loss(logits, targets, gamma, alpha)` | Focal loss for class-imbalanced binary classification |
| `kl_divergence(log_p, log_q)` | KL divergence between two log-probability distributions |
| `hinge_loss(pred, target)` | Hinge loss for SVM-style binary classification (targets: -1 or +1) |

---

## Optimizers

All optimizers implement the `Optimizer<T>` trait:

```rust
pub trait Optimizer<T: Float> {
    fn step(&mut self);       // apply parameter update
    fn zero_grad(&mut self);  // reset all gradients
}
```

### SGD

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let params = vec![
    Variable::new(Tensor::<f64>::ones(vec![10, 5]), true),
];

let mut optimizer = SGD::new(params, 0.01)  // lr = 0.01
    .with_momentum(0.9)
    .with_weight_decay(1e-4);
```

### Adam

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let params = vec![
    Variable::new(Tensor::<f64>::ones(vec![10, 5]), true),
];

let mut optimizer = Adam::new(params, 0.001)
    .with_beta1(0.9)
    .with_beta2(0.999)
    .with_eps(1e-8);
```

### AdamW

Adam with decoupled weight decay (Loshchilov & Hutter, 2017).

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let params = vec![
    Variable::new(Tensor::<f64>::ones(vec![10, 5]), true),
];

let mut optimizer = AdamW::new(params, 0.001);
```

### RMSprop

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let params = vec![
    Variable::new(Tensor::<f64>::ones(vec![10, 5]), true),
];

let mut optimizer = RMSprop::new(params, 0.01)
    .with_alpha(0.99)
    .with_momentum(0.0)
    .with_weight_decay(0.0);
```

### Adagrad

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let params = vec![
    Variable::new(Tensor::<f64>::ones(vec![10, 5]), true),
];

let mut optimizer = Adagrad::new(params, 0.01);
```

### Learning Rate Schedulers

Schedulers implement the `LrScheduler<T>` trait, returning the learning rate
for a given step or epoch.

```rust
use scivex_nn::prelude::*;

// Step decay: lr *= gamma every step_size epochs
let step_lr = StepLR::<f64>::new(0.1, 30, 0.1);  // (base_lr, step_size, gamma)
assert_eq!(step_lr.get_lr(0), 0.1);

// Exponential decay: lr = base_lr * gamma^step
let exp_lr = ExponentialLR::<f64>::new(0.1, 0.95);

// Cosine annealing from base_lr to eta_min over t_max steps
let cosine = CosineAnnealingLR::<f64>::new(0.1, 100, 1e-6);

// Linear warmup then cosine decay
let warmup_cosine = WarmupCosineDecay::<f64>::new(0.1, 10, 100, 1e-6);
```

---

## Training Loop Pattern

Here is the standard training loop using the low-level API:

```rust
use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let mut rng = Rng::new(42);

// 1. Build the model
let mut model: Sequential<f64> = Sequential::new(vec![
    Box::new(Linear::new(784, 128, true, &mut rng)),
    Box::new(ReLU),
    Box::new(Linear::new(128, 10, true, &mut rng)),
]);

// 2. Create optimizer
let mut optimizer = Adam::new(model.parameters(), 0.001);

// 3. Training loop
let num_epochs = 10;
for epoch in 0..num_epochs {
    optimizer.zero_grad();

    // Forward pass (replace with your actual data)
    let x = Variable::new(Tensor::ones(vec![32, 784]), false);
    let targets = Variable::new(
        Tensor::from_vec(vec![0.0; 32], vec![32]).unwrap(),
        false,
    );

    let logits = model.forward(&x).unwrap();
    let loss = cross_entropy_loss(&logits, &targets).unwrap();

    // Backward pass
    loss.backward();

    // Optional: gradient clipping
    clip_grad_norm(&model.parameters(), 1.0);

    // Update parameters
    optimizer.step();
}
```

### Using the Trainer

The `Trainer` struct manages the epoch loop with callback support:

```rust
use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let mut rng = Rng::new(42);
let mut model: Sequential<f64> = Sequential::new(vec![
    Box::new(Linear::new(4, 8, true, &mut rng)),
    Box::new(ReLU),
    Box::new(Linear::new(8, 1, true, &mut rng)),
]);
let mut optimizer = Adam::new(model.parameters(), 0.001);

let mut trainer = Trainer::new(100);

// Add early stopping: stop if no improvement for 10 epochs
trainer.add_callback(Box::new(EarlyStopping::new(10, 1e-6)));

let history = trainer.fit(|epoch| {
    optimizer.zero_grad();

    let x = Variable::new(Tensor::ones(vec![16, 4]), false);
    let target = Variable::new(Tensor::ones(vec![16, 1]), false);

    let pred = model.forward(&x)?;
    let loss = mse_loss(&pred, &target)?;

    loss.backward();
    optimizer.step();

    Ok(loss.data().as_slice()[0])
}).unwrap();

// Inspect training history
let final_loss = history.losses.last().unwrap();
let best_epoch = history.best_epoch;
let stopped_early = history.stopped_early;
```

### DataLoader

Use `TensorDataset` and `DataLoader` for batched iteration:

```rust
use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::prelude::*;

// Create dataset
let x_data = Tensor::<f64>::ones(vec![1000, 784]);
let y_data = Tensor::<f64>::zeros(vec![1000]);
let dataset = TensorDataset::new(x_data, y_data).unwrap();

// Create data loader with shuffling
let mut rng = Rng::new(42);
let loader = DataLoader::new(&dataset, 32, true, Some(&mut rng));

// Iterate over batches
for batch_result in loader {
    let (x_batch, y_batch) = batch_result.unwrap();
    // x_batch: [32, 784], y_batch: [32]
}
```

### Gradient Clipping

Two clipping strategies are available:

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let params = vec![
    Variable::new(Tensor::<f64>::ones(vec![10]), true),
];

// Clip by global L2 norm (returns the original norm)
let original_norm = clip_grad_norm(&params, 1.0);

// Clip by value (element-wise clamping to [-clip, clip])
clip_grad_value(&params, 0.5);
```

---

## Weight Initialization

The `init` module provides standard initialization strategies:

```rust
use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::prelude::*;

let mut rng = Rng::new(42);

// Xavier (Glorot) — good for sigmoid/tanh
let w1: Tensor<f64> = xavier_uniform(&[128, 784], &mut rng);
let w2: Tensor<f64> = xavier_normal(&[128, 784], &mut rng);

// Kaiming (He) — good for ReLU
let w3: Tensor<f64> = kaiming_uniform(&[128, 784], &mut rng);
let w4: Tensor<f64> = kaiming_normal(&[128, 784], &mut rng);
```

Layer constructors use these internally (`Linear` uses Kaiming uniform,
`SimpleRNN`/`LSTM`/`GRU` use Xavier uniform).

---

## Model Saving and Loading

### Native Binary Format

The `persist` module provides weight-only persistence in a compact binary
format (magic: `SVNN`).

```rust
use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let mut rng = Rng::new(42);
let model: Sequential<f64> = Sequential::new(vec![
    Box::new(Linear::new(784, 128, true, &mut rng)),
    Box::new(ReLU),
    Box::new(Linear::new(128, 10, true, &mut rng)),
]);

// Save weights
let params = model.parameters();
save_weights("model.bin", &params).unwrap();

// Load weights into a new model (must have the same architecture)
let mut rng2 = Rng::new(0);
let model2: Sequential<f64> = Sequential::new(vec![
    Box::new(Linear::new(784, 128, true, &mut rng2)),
    Box::new(ReLU),
    Box::new(Linear::new(128, 10, true, &mut rng2)),
]);
let tensors = load_weights::<f64>("model.bin").unwrap();
let params2 = model2.parameters();
for (param, tensor) in params2.iter().zip(tensors.iter()) {
    param.set_data(tensor.clone());
}
```

### SafeTensors Format

HuggingFace's SafeTensors format is supported for interoperability:

```rust
use scivex_nn::prelude::*;

// Save as SafeTensors (named tensors as HashMap<String, Tensor>)
// save_safetensors("model.safetensors", &named_tensors).unwrap();

// Load from SafeTensors
// let tensors = load_safetensors::<f64>("model.safetensors").unwrap();
```

### GGUF Format

The GGUF format (used by llama.cpp) is supported for loading and saving
quantized models:

```rust
use scivex_nn::prelude::*;

// Load a GGUF file
// let gguf = load_gguf::<f64>("model.gguf").unwrap();

// Save in GGUF format
// save_gguf("model.gguf", &gguf).unwrap();
```

---

## ONNX Model Loading and Inference

scivex-nn includes a from-scratch ONNX runtime that parses the protobuf wire
format without external dependencies.

```rust
use scivex_core::Tensor;
use scivex_nn::prelude::*;

// Load an ONNX model from bytes
let bytes = std::fs::read("model.onnx").unwrap();
let model = load_onnx(&bytes).unwrap();

// Create an inference session
let session = OnnxInferenceSession::<f32>::from_model(model).unwrap();

// Run inference
let input = Tensor::from_vec(vec![1.0_f32; 784], vec![1, 784]).unwrap();
let outputs = session.run(&[("input", input)]).unwrap();
```

The ONNX runtime supports common operators (MatMul, Conv, Relu, Add,
Reshape, etc.) and can run models exported from PyTorch and TensorFlow.

---

## Custom Layers

Implement the `Layer<T>` trait to create custom layers:

```rust
use scivex_core::{Float, Tensor};
use scivex_core::random::Rng;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

/// A residual block: output = relu(linear(x)) + x
struct ResidualBlock<T: Float> {
    linear: Linear<T>,
}

impl<T: Float> ResidualBlock<T> {
    fn new(features: usize, rng: &mut Rng) -> Self {
        Self {
            linear: Linear::new(features, features, true, rng),
        }
    }
}

impl<T: Float> Layer<T> for ResidualBlock<T> {
    fn forward(&self, x: &Variable<T>) -> scivex_nn::Result<Variable<T>> {
        let h = self.linear.forward(x)?;
        let h = relu(&h);
        Ok(&h + x)  // residual connection
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        self.linear.parameters()
    }

    fn train(&mut self) {
        self.linear.train();
    }

    fn eval(&mut self) {
        self.linear.eval();
    }
}
```

Custom layers can be used with `Sequential` via `Box::new(...)`.

---

## GPU Support

GPU-accelerated training is available behind the `gpu` feature flag, using
wgpu as the compute backend. All GPU operations use `f32` precision.

### Enabling GPU Support

Add to your `Cargo.toml`:

```toml
[dependencies]
scivex = { version = "0.1", features = ["nn", "gpu"] }
```

### GPU API

The GPU module mirrors the CPU API with `Gpu`-prefixed types:

```rust
#[cfg(feature = "gpu")]
{
    use scivex_nn::gpu::prelude::*;

    // GPU variable (wraps a GPU-resident tensor)
    // let x = GpuVariable::new(tensor, true);

    // GPU layers
    // let linear = GpuLinear::new(784, 128, true, &mut rng);

    // GPU optimizers
    // let optimizer = GpuAdam::new(params, 0.001);
    // let sgd = GpuSGD::new(params, 0.01);

    // GPU loss functions
    // let loss = gpu_mse_loss(&pred, &target);
    // let loss = gpu_cross_entropy_loss(&logits, &targets);
}
```

Available GPU components:

| CPU Type | GPU Equivalent |
|----------|---------------|
| `Variable` | `GpuVariable` |
| `Linear` | `GpuLinear` |
| `SGD` | `GpuSGD` |
| `Adam` | `GpuAdam` |
| `mse_loss` | `gpu_mse_loss` |
| `cross_entropy_loss` | `gpu_cross_entropy_loss` |

The `GpuLayer` and `GpuOptimizer` traits mirror their CPU counterparts.

---

## Complete Example: MNIST Classifier

Putting it all together -- a multi-layer perceptron for digit classification:

```rust
use scivex_core::Tensor;
use scivex_core::random::Rng;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

fn main() -> scivex_nn::Result<()> {
    let mut rng = Rng::new(42);

    // Build model
    let mut model: Sequential<f64> = Sequential::new(vec![
        Box::new(Linear::new(784, 256, true, &mut rng)),
        Box::new(BatchNorm1d::new(256)),
        Box::new(ReLU),
        Box::new(Dropout::new(0.3, Rng::new(1))),
        Box::new(Linear::new(256, 128, true, &mut rng)),
        Box::new(ReLU),
        Box::new(Linear::new(128, 10, true, &mut rng)),
    ]);

    // Optimizer with learning rate scheduling
    let mut optimizer = Adam::new(model.parameters(), 0.001);
    let scheduler = StepLR::new(0.001, 10, 0.5);

    // Training with early stopping
    let mut trainer = Trainer::new(50);
    trainer.add_callback(Box::new(EarlyStopping::new(5, 1e-6)));

    model.train();

    let history = trainer.fit(|epoch| {
        optimizer.zero_grad();

        // Replace with real data loading
        let x = Variable::new(Tensor::ones(vec![64, 784]), false);
        let targets = Variable::new(
            Tensor::from_vec(vec![0.0; 64], vec![64]).unwrap(),
            false,
        );

        let logits = model.forward(&x)?;
        let loss = cross_entropy_loss(&logits, &targets)?;

        loss.backward();
        clip_grad_norm(&model.parameters(), 1.0);
        optimizer.step();

        Ok(loss.data().as_slice()[0])
    })?;

    // Switch to eval mode for inference
    model.eval();

    // Save trained weights
    save_weights("mnist_model.bin", &model.parameters())?;

    Ok(())
}
```

---

## API Reference Summary

### Core Types

| Type | Description |
|------|-------------|
| `Variable<T>` | Autograd-enabled tensor wrapper |
| `Sequential<T>` | Layer container that chains layers in order |
| `TensorDataset<T>` | Dataset from paired input/target tensors |
| `DataLoader<T, D>` | Batched iterator over a dataset |
| `Trainer<T>` | Training loop manager with callback support |
| `TrainingHistory<T>` | Record of completed training run |

### Layers

| Layer | Input Shape | Description |
|-------|-------------|-------------|
| `Linear` | `[batch, in]` | Fully connected: `y = xW^T + b` |
| `Conv1d` | `[batch, channels, length]` | 1-D convolution |
| `Conv2d` | `[batch, channels, h, w]` | 2-D convolution (im2col) |
| `Conv3d` | `[batch, channels, d, h, w]` | 3-D convolution |
| `BatchNorm1d` | `[batch, features]` | 1-D batch normalization |
| `BatchNorm2d` | `[batch, channels, h, w]` | 2-D batch normalization |
| `LayerNorm` | `[batch, features]` | Layer normalization |
| `Dropout` | any | Inverted dropout regularization |
| `Embedding` | `[batch, seq_len]` | Token embedding lookup |
| `SimpleRNN` | `[batch, seq*input]` | Elman RNN |
| `LSTM` | `[batch, seq*input]` | Long short-term memory |
| `GRU` | `[batch, seq*input]` | Gated recurrent unit |
| `MultiHeadAttention` | `[batch, seq, d_model]` | Scaled dot-product attention |
| `TransformerEncoderLayer` | `[batch, seq, d_model]` | Full transformer encoder block |
| `TransformerDecoderLayer` | `[batch, seq, d_model]` | Full transformer decoder block |
| `MaxPool1d` / `MaxPool2d` | varies | Max pooling |
| `AvgPool1d` / `AvgPool2d` | varies | Average pooling |
| `Flatten` | any | Reshape to `[batch, -1]` |
| `ReLU` / `Sigmoid` / `Tanh` | any | Activation layers |

### Optimizers

| Optimizer | Key Parameters |
|-----------|----------------|
| `SGD` | `lr`, `momentum`, `weight_decay` |
| `Adam` | `lr`, `beta1`, `beta2`, `eps` |
| `AdamW` | `lr`, `beta1`, `beta2`, `eps`, `weight_decay` |
| `RMSprop` | `lr`, `alpha`, `eps`, `momentum`, `weight_decay` |
| `Adagrad` | `lr` |

### LR Schedulers

| Scheduler | Strategy |
|-----------|----------|
| `StepLR` | Multiply by `gamma` every `step_size` epochs |
| `ExponentialLR` | Multiply by `gamma` every step |
| `CosineAnnealingLR` | Half-cosine decay to `eta_min` |
| `LinearLR` | Linear interpolation between two rates |
| `ReduceLROnPlateau` | Reduce when metric plateaus |
| `WarmupCosineDecay` | Linear warmup then cosine decay |
