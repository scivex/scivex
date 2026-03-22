//! Optimized WGSL compute shaders and workgroup tuning utilities.
//!
//! This module provides pre-written, high-performance WGSL shader source strings
//! for common GPU compute patterns: tiled matrix multiplication, fused element-wise
//! operations, numerically stable softmax, and tree-based reductions.
//!
//! It also provides [`WorkgroupConfig`] for computing optimal dispatch dimensions
//! and [`DoubleBuffer`] for pipeline-overlapping label management.
//!
//! # Design
//!
//! All shader sources are `&'static str` constants — no GPU device is needed to
//! access them. The [`WorkgroupConfig`] methods are pure arithmetic and can be used
//! independently of any GPU runtime.

/// Default 1D workgroup size (threads per workgroup).
const DEFAULT_WG_SIZE_1D: u32 = 256;

/// Default 2D tile dimension (16x16 = 256 threads per workgroup).
const TILE_SIZE: u32 = 16;

// ---------------------------------------------------------------------------
// Optimized shader source strings
// ---------------------------------------------------------------------------

/// Optimized WGSL shader source strings for common GPU compute operations.
///
/// Each associated constant is a complete WGSL shader module ready to be compiled
/// via `wgpu::Device::create_shader_module`. The shaders use best-practice patterns
/// such as coalesced memory access, shared-memory tiling, and workgroup-level
/// parallel reduction.
pub struct OptimizedShaders;

impl OptimizedShaders {
    /// Tiled matrix multiplication shader using workgroup-shared memory.
    ///
    /// Computes `C = A * B` where A is `M x K` and B is `K x N`, producing `M x N`.
    ///
    /// Uses `TILE_SIZE` (16) square workgroups. Each tile of A and B is cooperatively
    /// loaded into `var<workgroup>` shared memory, then every thread computes one
    /// element of the output tile by accumulating partial dot products over the K
    /// dimension. Bounds checks ensure correctness for non-tile-aligned dimensions.
    ///
    /// Uniform buffer layout (`Params`): `M`, `N`, `K` as `u32`.
    ///
    /// Bindings:
    /// - `@group(0) @binding(0)` — `a: array<f32>` (read)
    /// - `@group(0) @binding(1)` — `b: array<f32>` (read)
    /// - `@group(0) @binding(2)` — `c: array<f32>` (read_write)
    /// - `@group(0) @binding(3)` — `params: Params` (uniform)
    pub const TILED_MATMUL: &'static str = r"
struct Params {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 16u;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn tiled_matmul(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;

    var acc: f32 = 0.0;

    let num_tiles = (params.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A into shared memory (coalesced along K).
        let a_col = t * TILE + local_col;
        if row < params.M && a_col < params.K {
            tile_a[local_row * TILE + local_col] = a[row * params.K + a_col];
        } else {
            tile_a[local_row * TILE + local_col] = 0.0;
        }

        // Load tile of B into shared memory (coalesced along N).
        let b_row = t * TILE + local_row;
        if b_row < params.K && col < params.N {
            tile_b[local_row * TILE + local_col] = b[b_row * params.N + col];
        } else {
            tile_b[local_row * TILE + local_col] = 0.0;
        }

        workgroupBarrier();

        // Accumulate partial dot product from the tile.
        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            acc = acc + tile_a[local_row * TILE + k] * tile_b[k * TILE + local_col];
        }

        workgroupBarrier();
    }

    if row < params.M && col < params.N {
        c[row * params.N + col] = acc;
    }
}
";

    /// Fused add + ReLU shader.
    ///
    /// Computes `result[i] = max(0.0, a[i] + b[i])` in a single kernel launch,
    /// avoiding an intermediate buffer for the addition.
    ///
    /// Bindings:
    /// - `@group(0) @binding(0)` — `a: array<f32>` (read)
    /// - `@group(0) @binding(1)` — `b: array<f32>` (read)
    /// - `@group(0) @binding(2)` — `result: array<f32>` (read_write)
    pub const FUSED_ADD_RELU: &'static str = r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn fused_add_relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&a) {
        result[idx] = max(0.0, a[idx] + b[idx]);
    }
}
";

    /// Fused multiply-add (FMA) shader.
    ///
    /// Computes `result[i] = a[i] * b[i] + c[i]` in a single kernel launch.
    ///
    /// Bindings:
    /// - `@group(0) @binding(0)` — `a: array<f32>` (read)
    /// - `@group(0) @binding(1)` — `b: array<f32>` (read)
    /// - `@group(0) @binding(2)` — `c: array<f32>` (read)
    /// - `@group(0) @binding(3)` — `result: array<f32>` (read_write)
    pub const FUSED_MUL_ADD: &'static str = r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn fused_mul_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&a) {
        result[idx] = fma(a[idx], b[idx], c[idx]);
    }
}
";

    /// Numerically stable softmax shader (three-pass).
    ///
    /// Pass 1 — find the row-wise maximum for numerical stability.
    /// Pass 2 — compute `exp(x - max)` and accumulate the sum.
    /// Pass 3 — normalize each element by the sum.
    ///
    /// All three passes use workgroup-shared memory for the reduction steps.
    ///
    /// Uniform buffer (`SoftmaxParams`): `rows` and `cols` as `u32`.
    ///
    /// Bindings:
    /// - `@group(0) @binding(0)` — `input: array<f32>` (read)
    /// - `@group(0) @binding(1)` — `output: array<f32>` (read_write)
    /// - `@group(0) @binding(2)` — `params: SoftmaxParams` (uniform)
    pub const SOFTMAX: &'static str = r"
struct SoftmaxParams {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: SoftmaxParams;

const WG_SIZE: u32 = 256u;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    if row >= params.rows {
        return;
    }

    let local_id = lid.x;
    let row_offset = row * params.cols;

    // --- Pass 1: find row max ---
    var local_max: f32 = -3.402823e+38;  // -FLT_MAX
    var i: u32 = local_id;
    while i < params.cols {
        let val = input[row_offset + i];
        if val > local_max {
            local_max = val;
        }
        i = i + WG_SIZE;
    }
    shared[local_id] = local_max;
    workgroupBarrier();

    // Tree reduction for max.
    var stride: u32 = WG_SIZE / 2u;
    while stride > 0u {
        if local_id < stride {
            if shared[local_id + stride] > shared[local_id] {
                shared[local_id] = shared[local_id + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = shared[0];
    workgroupBarrier();

    // --- Pass 2: exp(x - max) and sum ---
    var local_sum: f32 = 0.0;
    i = local_id;
    while i < params.cols {
        let val = exp(input[row_offset + i] - row_max);
        output[row_offset + i] = val;
        local_sum = local_sum + val;
        i = i + WG_SIZE;
    }
    shared[local_id] = local_sum;
    workgroupBarrier();

    // Tree reduction for sum.
    stride = WG_SIZE / 2u;
    while stride > 0u {
        if local_id < stride {
            shared[local_id] = shared[local_id] + shared[local_id + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_sum = shared[0];
    workgroupBarrier();

    // --- Pass 3: normalize ---
    i = local_id;
    while i < params.cols {
        output[row_offset + i] = output[row_offset + i] / row_sum;
        i = i + WG_SIZE;
    }
}
";

    /// Workgroup-level tree reduction (sum) shader.
    ///
    /// Each workgroup cooperatively sums its elements using a binary-tree pattern
    /// in shared memory. The per-workgroup partial sums are written to the output
    /// buffer; a second dispatch can reduce those further if needed.
    ///
    /// Bindings:
    /// - `@group(0) @binding(0)` — `input: array<f32>` (read)
    /// - `@group(0) @binding(1)` — `output: array<f32>` (read_write)
    pub const TREE_REDUCE_SUM: &'static str = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 256u;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn tree_reduce_sum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_id = lid.x;
    let global_id = gid.x;

    // Load one element per thread; out-of-bounds threads contribute 0.
    if global_id < arrayLength(&input) {
        shared[local_id] = input[global_id];
    } else {
        shared[local_id] = 0.0;
    }
    workgroupBarrier();

    // Binary-tree reduction in shared memory.
    var stride: u32 = WG_SIZE / 2u;
    while stride > 0u {
        if local_id < stride {
            shared[local_id] = shared[local_id] + shared[local_id + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Thread 0 writes the workgroup partial sum.
    if local_id == 0u {
        output[wid.x] = shared[0];
    }
}
";
}

// ---------------------------------------------------------------------------
// Workgroup configuration and dispatch helpers
// ---------------------------------------------------------------------------

/// Workgroup size configuration and dispatch dimension calculator.
///
/// Provides sensible defaults for 1D and 2D workgroups and computes the number
/// of workgroups to dispatch for a given problem size.
///
/// # Examples
///
/// ```
/// use scivex_gpu::shader_opt::WorkgroupConfig;
///
/// let (x, y, z) = WorkgroupConfig::dispatch_1d(1000);
/// assert_eq!((x, y, z), (4, 1, 1)); // ceil(1000 / 256) = 4
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkgroupConfig {
    /// Number of threads along the X axis.
    pub x: u32,
    /// Number of threads along the Y axis.
    pub y: u32,
    /// Number of threads along the Z axis.
    pub z: u32,
}

impl WorkgroupConfig {
    /// Default 1D workgroup: 256 threads along X.
    pub fn default_1d() -> Self {
        Self {
            x: DEFAULT_WG_SIZE_1D,
            y: 1,
            z: 1,
        }
    }

    /// Default 2D workgroup for matrix operations: 16 x 16 = 256 threads.
    pub fn default_2d() -> Self {
        Self {
            x: TILE_SIZE,
            y: TILE_SIZE,
            z: 1,
        }
    }

    /// Compute dispatch dimensions for a 1D problem of `total_elements`.
    ///
    /// Returns `(num_workgroups_x, 1, 1)` using the default 1D workgroup size
    /// of 256.
    ///
    /// # Examples
    ///
    /// ```
    /// use scivex_gpu::shader_opt::WorkgroupConfig;
    ///
    /// assert_eq!(WorkgroupConfig::dispatch_1d(256), (1, 1, 1));
    /// assert_eq!(WorkgroupConfig::dispatch_1d(257), (2, 1, 1));
    /// assert_eq!(WorkgroupConfig::dispatch_1d(0), (0, 1, 1));
    /// ```
    pub fn dispatch_1d(total_elements: usize) -> (u32, u32, u32) {
        let groups = Self::num_workgroups(total_elements, DEFAULT_WG_SIZE_1D);
        (groups, 1, 1)
    }

    /// Compute dispatch dimensions for a 2D matrix operation with `rows` x `cols`.
    ///
    /// Returns `(ceil(cols / 16), ceil(rows / 16), 1)` matching the 16 x 16 tile
    /// convention where X maps to columns and Y maps to rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scivex_gpu::shader_opt::WorkgroupConfig;
    ///
    /// assert_eq!(WorkgroupConfig::dispatch_2d(32, 32), (2, 2, 1));
    /// assert_eq!(WorkgroupConfig::dispatch_2d(17, 33), (3, 2, 1));
    /// ```
    pub fn dispatch_2d(rows: usize, cols: usize) -> (u32, u32, u32) {
        let groups_x = Self::num_workgroups(cols, TILE_SIZE);
        let groups_y = Self::num_workgroups(rows, TILE_SIZE);
        (groups_x, groups_y, 1)
    }

    /// Compute the number of workgroups needed to cover `total` elements with
    /// the given `workgroup_size`.
    ///
    /// This is the standard ceiling-division: `(total + workgroup_size - 1) / workgroup_size`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scivex_gpu::shader_opt::WorkgroupConfig;
    ///
    /// assert_eq!(WorkgroupConfig::num_workgroups(1024, 256), 4);
    /// assert_eq!(WorkgroupConfig::num_workgroups(1025, 256), 5);
    /// assert_eq!(WorkgroupConfig::num_workgroups(0, 256), 0);
    /// ```
    pub fn num_workgroups(total: usize, workgroup_size: u32) -> u32 {
        let ws = workgroup_size as usize;
        total.div_ceil(ws) as u32
    }
}

// ---------------------------------------------------------------------------
// Double-buffering label helper
// ---------------------------------------------------------------------------

/// Label generator for double-buffered GPU pipelines.
///
/// Double buffering allows overlapping data upload with compute by alternating
/// between two buffers. This struct simply generates consistent label strings
/// for the two buffers; actual buffer creation uses `wgpu::Device::create_buffer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoubleBuffer {
    /// Label for the first buffer.
    pub buffer_a_label: String,
    /// Label for the second buffer.
    pub buffer_b_label: String,
}

impl DoubleBuffer {
    /// Create a double-buffer label pair from a base `name`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scivex_gpu::shader_opt::DoubleBuffer;
    ///
    /// let db = DoubleBuffer::new("staging");
    /// assert_eq!(db.buffer_a_label, "staging_a");
    /// assert_eq!(db.buffer_b_label, "staging_b");
    /// ```
    pub fn new(name: &str) -> Self {
        Self {
            buffer_a_label: format!("{name}_a"),
            buffer_b_label: format!("{name}_b"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workgroup_dispatch_1d() {
        assert_eq!(WorkgroupConfig::dispatch_1d(0), (0, 1, 1));
        assert_eq!(WorkgroupConfig::dispatch_1d(1), (1, 1, 1));
        assert_eq!(WorkgroupConfig::dispatch_1d(256), (1, 1, 1));
        assert_eq!(WorkgroupConfig::dispatch_1d(257), (2, 1, 1));
        assert_eq!(WorkgroupConfig::dispatch_1d(1000), (4, 1, 1));
        assert_eq!(WorkgroupConfig::dispatch_1d(1_000_000), (3907, 1, 1));
    }

    #[test]
    fn test_workgroup_dispatch_2d() {
        assert_eq!(WorkgroupConfig::dispatch_2d(0, 0), (0, 0, 1));
        assert_eq!(WorkgroupConfig::dispatch_2d(16, 16), (1, 1, 1));
        assert_eq!(WorkgroupConfig::dispatch_2d(17, 16), (1, 2, 1));
        assert_eq!(WorkgroupConfig::dispatch_2d(32, 32), (2, 2, 1));
        assert_eq!(WorkgroupConfig::dispatch_2d(17, 33), (3, 2, 1));
        assert_eq!(WorkgroupConfig::dispatch_2d(128, 64), (4, 8, 1));
    }

    #[test]
    fn test_num_workgroups() {
        assert_eq!(WorkgroupConfig::num_workgroups(0, 256), 0);
        assert_eq!(WorkgroupConfig::num_workgroups(1, 256), 1);
        assert_eq!(WorkgroupConfig::num_workgroups(256, 256), 1);
        assert_eq!(WorkgroupConfig::num_workgroups(257, 256), 2);
        assert_eq!(WorkgroupConfig::num_workgroups(1024, 256), 4);
        assert_eq!(WorkgroupConfig::num_workgroups(1025, 256), 5);
        // Different workgroup sizes.
        assert_eq!(WorkgroupConfig::num_workgroups(100, 16), 7);
        assert_eq!(WorkgroupConfig::num_workgroups(64, 64), 1);
    }

    #[test]
    fn test_shader_strings_not_empty() {
        assert!(!OptimizedShaders::TILED_MATMUL.is_empty());
        assert!(!OptimizedShaders::FUSED_ADD_RELU.is_empty());
        assert!(!OptimizedShaders::FUSED_MUL_ADD.is_empty());
        assert!(!OptimizedShaders::SOFTMAX.is_empty());
        assert!(!OptimizedShaders::TREE_REDUCE_SUM.is_empty());

        // Verify key shader entry points are present.
        assert!(OptimizedShaders::TILED_MATMUL.contains("fn tiled_matmul"));
        assert!(OptimizedShaders::FUSED_ADD_RELU.contains("fn fused_add_relu"));
        assert!(OptimizedShaders::FUSED_MUL_ADD.contains("fn fused_mul_add"));
        assert!(OptimizedShaders::SOFTMAX.contains("fn softmax"));
        assert!(OptimizedShaders::TREE_REDUCE_SUM.contains("fn tree_reduce_sum"));

        // Verify shared memory usage in tiled matmul.
        assert!(OptimizedShaders::TILED_MATMUL.contains("var<workgroup>"));
        assert!(OptimizedShaders::TILED_MATMUL.contains("workgroupBarrier"));
    }

    #[test]
    fn test_double_buffer_labels() {
        let db = DoubleBuffer::new("staging");
        assert_eq!(db.buffer_a_label, "staging_a");
        assert_eq!(db.buffer_b_label, "staging_b");

        let db2 = DoubleBuffer::new("upload_weights");
        assert_eq!(db2.buffer_a_label, "upload_weights_a");
        assert_eq!(db2.buffer_b_label, "upload_weights_b");

        // Empty name edge case.
        let db3 = DoubleBuffer::new("");
        assert_eq!(db3.buffer_a_label, "_a");
        assert_eq!(db3.buffer_b_label, "_b");
    }
}
