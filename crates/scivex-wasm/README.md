# scivex-wasm

WebAssembly bindings for Scivex. Run tensors, DataFrames, statistics,
machine learning, and more in the browser or Node.js.

## Highlights

- **Tensor operations** — Create, manipulate, and compute tensors in JS/TS
- **DataFrames** — Load and analyze tabular data in the browser
- **Statistics** — Distributions, hypothesis tests, correlation
- **Machine learning** — Train and predict with ML models client-side
- **Signal processing** — FFT, filters, spectrograms in the browser
- **Graph algorithms** — Shortest paths, PageRank, community detection
- **Symbolic math** — Expression parsing, differentiation, simplification

## Usage

```bash
npm install scivex
```

```javascript
import init, { Tensor, DataFrame } from 'scivex';

await init();

const a = Tensor.from_array([1, 2, 3, 4], [2, 2]);
const b = Tensor.ones([2, 2]);
const c = a.add(b);
console.log(c.to_array());
```

## Building

```bash
wasm-pack build --target web
```

## License

MIT
