# scivex-nn

Neural networks for Scivex. Autograd engine, layers, optimizers, and loss
functions for building and training deep learning models.

## Highlights

- **Autograd** — Automatic differentiation with dynamic computation graph
- **Layers** — Linear, Conv1d/2d/3d, RNN, LSTM, GRU, BatchNorm, LayerNorm, Dropout
- **Attention** — MultiHeadAttention, MultiQuery, GroupedQuery, Flash, Rotary/Sinusoidal positional encoding
- **Transformer** — TransformerEncoderLayer, TransformerDecoderLayer
- **Pooling** — MaxPool1d/2d, AvgPool1d/2d, AdaptiveAvgPool, GlobalAvgPool
- **Activations** — ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU
- **Optimizers** — SGD (with momentum), Adam, AdamW, RMSprop, Adagrad
- **Losses** — MSE, CrossEntropy, BinaryCrossEntropy, Huber, L1
- **Schedulers** — StepLR, CosineAnnealing, ReduceOnPlateau, WarmupCosine
- **Mixed precision** — FP16/BF16 support via scivex-core half types
- **GPU support** — Optional GPU acceleration via scivex-gpu

## Usage

```rust
use scivex_nn::prelude::*;

let mut model = Sequential::new(vec![
    Box::new(Linear::new(784, 256, &mut rng)),
    Box::new(ReLU),
    Box::new(Linear::new(256, 10, &mut rng)),
]);

let mut optimizer = Adam::new(model.parameters(), 0.001);
let loss = cross_entropy_loss(&logits, &targets);
loss.backward(None);
optimizer.step();
```

## License

MIT
