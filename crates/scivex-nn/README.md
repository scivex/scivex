# scivex-nn

Neural networks and automatic differentiation for Scivex. Build and train
feedforward networks with a PyTorch-inspired API.

## Highlights

- **Reverse-mode autograd** — `Variable<T>` with computation graph and `backward()`
- **Differentiable ops** — add, sub, mul, matmul, pow, sum, mean with gradient tracking
- **Activations** — ReLU, sigmoid, tanh, softmax, log-softmax
- **Layers** — Linear, ReLU, Sigmoid, Tanh, Sequential, Dropout, BatchNorm1d
- **Loss functions** — MSE, Cross-Entropy, Binary Cross-Entropy
- **Optimizers** — SGD (with momentum), Adam
- **Initialization** — Xavier (Glorot) and Kaiming (He), uniform and normal
- **Data** — Dataset trait, TensorDataset, DataLoader with batching and shuffle

## Usage

```rust
use scivex_nn::prelude::*;

// Build a model
let mut model = Sequential::new(vec![
    Box::new(Linear::new(784, 256, &mut rng)),
    Box::new(ReLU),
    Box::new(Dropout::new(0.3)),
    Box::new(Linear::new(256, 10, &mut rng)),
]);

// Training loop
let mut optimizer = Adam::new(model.parameters(), 1e-3);
for batch in data_loader {
    let logits = model.forward(&batch.input);
    let loss = cross_entropy_loss(&logits, &batch.target);
    optimizer.zero_grad();
    loss.backward(None);
    optimizer.step();
}
```

## License

MIT
