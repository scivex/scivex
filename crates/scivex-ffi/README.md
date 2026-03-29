# scivex-ffi

C Foreign Function Interface bindings for Scivex. Produces a shared library
(`.so`/`.dylib`/`.dll`) callable from any language with C FFI support —
Julia, R, Swift, Python (ctypes), and more.

## Highlights

- **Tensor ops** — Create, reshape, slice, and compute tensors via C API
- **DataFrames** — Load CSV, filter, group, and aggregate from C
- **Statistics** — Distributions, hypothesis tests, regression
- **ML models** — Train and predict with linear models, trees, and ensembles
- **Visualization** — Generate SVG plots from C
- **Memory safe** — Opaque handles with explicit alloc/free functions

## Building

```bash
cargo build -p scivex-ffi --release

# Output: target/release/libscivex_ffi.{so,dylib,dll}
```

## Usage (C)

```c
#include "scivex.h"

ScivexTensor* a = scivex_tensor_from_array(data, shape, ndim);
ScivexTensor* b = scivex_tensor_ones(shape, ndim);
ScivexTensor* c = scivex_tensor_add(a, b);

scivex_tensor_free(a);
scivex_tensor_free(b);
scivex_tensor_free(c);
```

## License

MIT
