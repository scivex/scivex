# Migrating from PyTorch to Scivex

This guide maps PyTorch concepts and APIs to their Scivex equivalents.
All Scivex examples assume you have brought the neural network prelude
into scope:

```rust
use scivex::prelude::*;            // umbrella re-exports
use scivex_core::Tensor;
use scivex_core::random::Rng;
```

---

## Quick Reference Table

| Concept | PyTorch (Python) | Scivex (Rust) |
|---|---|---|
| Tensor with grad | `torch.tensor([1.0], requires_grad=True)` | `Variable::new(Tensor::from_vec(vec![1.0], vec![1]).unwrap(), true)` |
| Backpropagation | `loss.backward()` | `loss.backward()` |
| Read gradient | `x.grad` | `x.grad()` (returns `Option<Tensor<T>>`) |
| Zero gradients | `optimizer.zero_grad()` | `optimizer.zero_grad()` |
| Detach | `x.detach()` | `x.detach()` |
| Linear layer | `nn.Linear(in, out)` | `Linear::new(in, out, true, &mut rng)` |
| Conv2d layer | `nn.Conv2d(c_in, c_out, (kh, kw))` | `Conv2d::new(c_in, c_out, (kh, kw), true, &mut rng)` |
| BatchNorm1d | `nn.BatchNorm1d(features)` | `BatchNorm1d::new(features)` |
| Dropout | `nn.Dropout(p)` | `Dropout::new(T::from_f64(p), Rng::new(seed))` |
| Sequential model | `nn.Sequential(...)` | `Sequential::new(vec![Box::new(...), ...])` |
| ReLU (functional) | `F.relu(x)` | `relu(&x)` |
| Sigmoid (functional) | `torch.sigmoid(x)` | `sigmoid(&x)` |
| Softmax | `F.softmax(x, dim=-1)` | `softmax(&x)?` |
| MSE loss | `F.mse_loss(pred, target)` | `mse_loss(&pred, &target)?` |
| Cross-entropy | `F.cross_entropy(logits, targets)` | `cross_entropy_loss(&logits, &targets)?` |
| BCE loss | `F.binary_cross_entropy(pred, target)` | `bce_loss(&pred, &target)?` |
| Huber loss | `F.huber_loss(pred, target, delta)` | `huber_loss(&pred, &target, delta)?` |
| SGD optimizer | `optim.SGD(params, lr=0.01)` | `SGD::new(params, 0.01)` |
| Adam optimizer | `optim.Adam(params, lr=0.001)` | `Adam::new(params, 0.001)` |
| AdamW optimizer | `optim.AdamW(params, lr=0.001)` | `AdamW::new(params, 0.001)` |
| StepLR scheduler | `StepLR(optimizer, step_size, gamma)` | `StepLR::new(base_lr, step_size, gamma)` |
| Grad clipping (norm) | `nn.utils.clip_grad_norm_(params, max_norm)` | `clip_grad_norm(&params, max_norm)` |
| Grad clipping (value) | `nn.utils.clip_grad_value_(params, value)` | `clip_grad_value(&params, value)` |
| Save model | `torch.save(model.state_dict(), path)` | `save_weights(path, &model.parameters())?` |
| Load model | `model.load_state_dict(torch.load(path))` | `let tensors = load_weights::<f64>(path)?;` |
| Dataset | `torch.utils.data.Dataset` | `Dataset<T>` trait |
| DataLoader | `DataLoader(dataset, batch_size=32)` | `DataLoader::new(&dataset, 32, true, Some(&mut rng))` |
| Training mode | `model.train()` | `model.train()` |
| Eval mode | `model.eval()` | `model.eval()` |

---

## Tensors and Autograd

### Creating a Variable with Gradient Tracking

**PyTorch:**

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
```

**Scivex:**

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;

let t = Tensor::<f64>::from_vec(
    vec![1.0, 2.0, 3.0, 4.0],
    vec![2, 2],
).unwrap();
let x = Variable::new(t, true);  // requires_grad = true
```

### Forward Computation and Backward Pass

**PyTorch:**

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad)  # tensor([4., 6.])
```

**Scivex:**

```rust
use scivex_core::Tensor;
use scivex_nn::Variable;
use scivex_nn::prelude::*;

let t = Tensor::<f64>::from_vec(vec![2.0, 3.0], vec![2]).unwrap();
let x = Variable::new(t, true);
let y = mul(&x, &x);        // element-wise x * x
let z = sum(&y);             // reduce to scalar
z.backward();
let grad = x.grad().unwrap(); // Tensor with [4.0, 6.0]
```

### Available Differentiable Operations

All operations in the `ops` module record their gradient functions
automatically. The following are available:

| Function | Description |
|---|---|
| `add(&a, &b)` | Element-wise addition |
| `sub(&a, &b)` | Element-wise subtraction |
| `mul(&a, &b)` | Element-wise multiplication (Hadamard) |
| `neg(&a)` | Negation |
| `matmul(&a, &b)` | Matrix multiplication (`[m,k] @ [k,n]`) |
| `sum(&a)` | Sum all elements to scalar |
| `mean(&a)` | Mean of all elements to scalar |
| `pow(&a, exponent)` | Element-wise power |
| `scalar_mul(&a, s)` | Multiply by scalar |
| `add_bias(&input, &bias)` | Add 1-D bias to 2-D input (broadcasting) |

Operator overloads (`+`, `-`, `*`) are also implemented for `&Variable<T>`,
so `&a + &b` works the same as `add(&a, &b)`.

---

## Defining Layers

### The Layer Trait

Every layer in Scivex implements the `Layer<T>` trait:

```rust
pub trait Layer<T: Float> {
    fn forward(&self, x: &Variable<T>) -> Result<Variable<T>>;
    fn parameters(&self) -> Vec<Variable<T>>;
    fn train(&mut self);
    fn eval(&mut self);
}
```

This replaces PyTorch's `nn.Module`. The key differences:

- `forward` takes `&Variable<T>` instead of `Tensor` and returns a `Result`.
- `parameters()` returns a `Vec<Variable<T>>` directly (no generator/iterator protocol).
- There is no `__call__` magic; you call `.forward(&input)` explicitly.

### Linear Layer

**PyTorch:**

```python
layer = nn.Linear(784, 128)
out = layer(x)
```

**Scivex:**

```rust
let mut rng = Rng::new(42);
let layer = Linear::<f64>::new(784, 128, true, &mut rng);
let out = layer.forward(&x)?;
```

The fourth argument (`true`) controls whether a bias term is included.
Weight initialization defaults to Kaiming uniform.

### Conv2d Layer

**PyTorch:**

```python
conv = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1)
```

**Scivex:**

```rust
let conv = Conv2d::<f64>::new(3, 64, (3, 3), true, &mut rng)
    .set_stride((1, 1))
    .set_padding((1, 1));
```

Stride, padding, and dilation default to `(1,1)`, `(0,0)`, and `(1,1)`
respectively. Use the builder methods `set_stride`, `set_padding`, and
`set_dilation` to configure them.

### Activation Layers

Scivex provides activations both as free functions and as zero-parameter
`Layer` implementations:

| PyTorch | Scivex (functional) | Scivex (layer) |
|---|---|---|
| `F.relu(x)` | `relu(&x)` | `ReLU` |
| `torch.sigmoid(x)` | `sigmoid(&x)` | `Sigmoid` |
| `torch.tanh(x)` | `tanh_fn(&x)` | `Tanh` |
| `F.softmax(x, dim=-1)` | `softmax(&x)?` | -- |
| `F.log_softmax(x, dim=-1)` | `log_softmax(&x)?` | -- |

The layer forms (`ReLU`, `Sigmoid`, `Tanh`) are unit structs that can be
placed inside a `Sequential`.

---

## Building Models with Sequential

**PyTorch:**

```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 10),
)
out = model(x)
```

**Scivex:**

```rust
let mut rng = Rng::new(42);
let mut model: Sequential<f64> = Sequential::new(vec![
    Box::new(Linear::new(784, 128, true, &mut rng)),
    Box::new(ReLU),
    Box::new(Dropout::new(0.5_f64, Rng::new(123))),
    Box::new(Linear::new(128, 10, true, &mut rng)),
]);
let out = model.forward(&x)?;
```

`Sequential` itself implements `Layer<T>`, so it composes and nests.
To get all learnable parameters, call `model.parameters()`.

---

## Loss Functions

All loss functions take `&Variable<T>` arguments and return
`Result<Variable<T>>`. The returned variable is a scalar connected to
the computation graph, so calling `.backward()` on it propagates
gradients to all upstream parameters.

**PyTorch:**

```python
loss = F.cross_entropy(logits, targets)
loss.backward()
```

**Scivex:**

```rust
// logits: Variable<f64> with shape [batch, classes]
// targets: Variable<f64> with shape [batch] (class indices as floats)
let loss = cross_entropy_loss(&logits, &targets)?;
loss.backward();
```

Available loss functions:

| Function | Description |
|---|---|
| `mse_loss(&pred, &target)` | Mean squared error |
| `cross_entropy_loss(&logits, &targets)` | Cross-entropy (log-softmax + NLL) |
| `bce_loss(&pred, &target)` | Binary cross-entropy |
| `huber_loss(&pred, &target, delta)` | Huber / smooth-L1 loss |
| `focal_loss(...)` | Focal loss for class imbalance |
| `hinge_loss(...)` | SVM-style hinge loss |
| `kl_divergence(...)` | KL divergence |
| `smooth_l1_loss(...)` | Smooth L1 loss |

---

## Optimizers

### Creating and Using an Optimizer

**PyTorch:**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()
    out = model(x)
    loss = F.mse_loss(out, y)
    loss.backward()
    optimizer.step()
```

**Scivex:**

```rust
let params = model.parameters();
let mut optimizer = Adam::new(params, 0.001_f64);

for epoch in 0..100 {
    optimizer.zero_grad();
    let out = model.forward(&x)?;
    let loss = mse_loss(&out, &y)?;
    loss.backward();
    optimizer.step();
}
```

### Available Optimizers

| Optimizer | Constructor | Builder methods |
|---|---|---|
| `SGD` | `SGD::new(params, lr)` | `.with_momentum(m)`, `.with_weight_decay(wd)` |
| `Adam` | `Adam::new(params, lr)` | `.with_beta1(b1)`, `.with_beta2(b2)`, `.with_eps(e)` |
| `AdamW` | `AdamW::new(params, lr)` | `.with_beta1(b1)`, `.with_beta2(b2)`, `.with_eps(e)` |
| `RMSprop` | `RMSprop::new(params, lr)` | -- |
| `Adagrad` | `Adagrad::new(params, lr)` | -- |

All optimizers implement the `Optimizer<T>` trait:

```rust
pub trait Optimizer<T: Float> {
    fn step(&mut self);
    fn zero_grad(&mut self);
}
```

### Learning Rate Schedulers

Schedulers implement the `LrScheduler<T>` trait with `get_lr(step) -> T`:

| Scheduler | Constructor |
|---|---|
| `StepLR` | `StepLR::new(base_lr, step_size, gamma)` |
| `ExponentialLR` | `ExponentialLR::new(base_lr, gamma)` |
| `CosineAnnealingLR` | `CosineAnnealingLR::new(base_lr, t_max)` |
| `LinearLR` | `LinearLR::new(start_lr, end_lr, total_steps)` |
| `ReduceLROnPlateau` | -- |
| `WarmupCosineDecay` | -- |

---

## Training Loop

### Manual Loop

The manual loop maps almost line-for-line from PyTorch:

**PyTorch:**

```python
model.train()
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

**Scivex:**

```rust
model.train();
for epoch in 0..num_epochs {
    for batch in &mut loader {
        let (batch_x, batch_y) = batch?;
        let x = Variable::new(batch_x, false);
        let y = Variable::new(batch_y, false);

        optimizer.zero_grad();
        let pred = model.forward(&x)?;
        let loss = mse_loss(&pred, &y)?;
        loss.backward();

        let params = model.parameters();
        clip_grad_norm(&params, 1.0);

        optimizer.step();
    }
}
```

### Using the Trainer

Scivex also provides a `Trainer` abstraction with callback support:

```rust
let mut trainer = Trainer::<f64>::new(100); // 100 epochs
trainer.add_callback(Box::new(EarlyStopping::new(10, 1e-6)));

let history = trainer.fit(|epoch| {
    optimizer.zero_grad();
    let out = model.forward(&x)?;
    let loss = mse_loss(&out, &y)?;
    loss.backward();
    optimizer.step();

    // Return the scalar loss value for this epoch.
    Ok(loss.data().as_slice()[0])
})?;

println!("Best epoch: {}", history.best_epoch);
println!("Stopped early: {}", history.stopped_early);
```

Available callbacks:

- `EarlyStopping::new(patience, min_delta)` -- stops when loss plateaus.
- `ModelCheckpoint` -- saves weights at best epochs.
- `LossLogger` -- records loss history.

---

## Datasets and DataLoaders

**PyTorch:**

```python
dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**Scivex:**

```rust
let dataset = TensorDataset::new(x_tensor, y_tensor)?;
let mut rng = Rng::new(42);
let loader = DataLoader::new(&dataset, 32, true, Some(&mut rng));

for batch in loader {
    let (batch_x, batch_y) = batch?;
    // ...
}
```

You can also implement the `Dataset<T>` trait for custom datasets:

```rust
pub trait Dataset<T: Float> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)>;
}
```

---

## Weight Initialization

| PyTorch | Scivex |
|---|---|
| `nn.init.xavier_uniform_(w)` | `xavier_uniform::<f64>(&[out, inp], &mut rng)` |
| `nn.init.xavier_normal_(w)` | `xavier_normal::<f64>(&[out, inp], &mut rng)` |
| `nn.init.kaiming_uniform_(w)` | `kaiming_uniform::<f64>(&[out, inp], &mut rng)` |
| `nn.init.kaiming_normal_(w)` | `kaiming_normal::<f64>(&[out, inp], &mut rng)` |

These functions return a `Tensor<T>`. To apply custom initialization to
a layer parameter, use `param.set_data(new_tensor)`.

---

## Saving and Loading Models

**PyTorch:**

```python
torch.save(model.state_dict(), "model.pt")

model.load_state_dict(torch.load("model.pt"))
```

**Scivex:**

```rust
// Save
let params = model.parameters();
save_weights("model.bin", &params)?;

// Load -- reconstruct the model architecture first, then load weights
let mut model = build_model(&mut rng);
let tensors = load_weights::<f64>("model.bin")?;
for (param, tensor) in model.parameters().iter().zip(tensors.iter()) {
    param.set_data(tensor.clone());
}
```

Scivex also supports SafeTensors and GGUF formats via `save_safetensors`,
`load_safetensors`, `save_gguf`, and `load_gguf`.

---

## Key Differences from PyTorch

### 1. Explicit RNG threading

PyTorch uses global random state. Scivex requires you to pass an `&mut Rng`
to any operation that needs randomness (layer construction, dropout,
data shuffling). This makes randomness deterministic and reproducible
without global state.

### 2. Ownership and borrowing in the backward pass

`Variable<T>` uses `Rc<RefCell<...>>` internally so cloning is cheap
(reference count bump). After calling `backward()`, gradients are
accumulated in each leaf variable. You read them with `x.grad()` which
returns `Option<Tensor<T>>`. You must call `zero_grad()` (either on the
optimizer or on individual variables) before each training step to clear
stale gradients.

### 3. No dynamic computation graph rebuild

PyTorch rebuilds the computation graph on every forward pass (define-by-run).
Scivex does the same -- each call to `forward` creates a fresh graph of
`Variable` nodes linked by closures. The difference is that Rust's
ownership model means you cannot hold onto intermediate graph nodes
across forward passes without explicitly cloning the `Variable`.

### 4. All fallible operations return Result

Where PyTorch raises Python exceptions, Scivex returns `Result<T, NnError>`.
Shape mismatches, out-of-bounds indices, and I/O errors are all represented
as `NnError` variants. Use `?` to propagate errors in your training code.

### 5. Generic over float type

All Scivex neural network types are generic over `T: Float`. You choose
`f32` or `f64` at the type level:

```rust
let model: Sequential<f32> = Sequential::new(vec![
    Box::new(Linear::<f32>::new(784, 128, true, &mut rng)),
    Box::new(ReLU),
    Box::new(Linear::<f32>::new(128, 10, true, &mut rng)),
]);
```

### 6. No Python-style module registration

PyTorch automatically registers submodules assigned as attributes in
`__init__`. In Scivex, you collect parameters explicitly by implementing
the `parameters()` method on your struct. `Sequential` handles this
automatically for its contained layers.

### 7. Train/eval mode is explicit

Call `model.train()` and `model.eval()` to switch behavior of layers
like `Dropout` and `BatchNorm`. This works identically to PyTorch, but
requires `&mut self` since it mutates internal state.

---

## Complete Example: MNIST-style Classifier

**PyTorch:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10),
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
model.train()

for epoch in range(20):
    optimizer.zero_grad()
    logits = model(x_train)
    loss = F.cross_entropy(logits, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

**Scivex:**

```rust
use scivex::prelude::*;
use scivex_core::{Tensor, random::Rng};
use scivex_nn::prelude::*;

fn main() -> scivex_nn::Result<()> {
    let mut rng = Rng::new(42);

    // Build model
    let mut model: Sequential<f64> = Sequential::new(vec![
        Box::new(Linear::new(784, 256, true, &mut rng)),
        Box::new(ReLU),
        Box::new(Dropout::new(0.2, Rng::new(99))),
        Box::new(Linear::new(256, 10, true, &mut rng)),
    ]);

    let params = model.parameters();
    let mut optimizer = Adam::new(params, 1e-3);

    model.train();

    for epoch in 0..20 {
        optimizer.zero_grad();

        let logits = model.forward(&x_train)?;
        let loss = cross_entropy_loss(&logits, &y_train)?;
        loss.backward();
        optimizer.step();

        let loss_val = loss.data().as_slice()[0];
        println!("Epoch {epoch}: loss = {loss_val:.4f}");
    }

    // Save trained weights
    save_weights("mnist_model.bin", &model.parameters())?;

    Ok(())
}
```
