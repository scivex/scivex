//! GPU-backed autograd variable.
//!
//! [`GpuVariable`] is the GPU equivalent of [`Variable<T>`](crate::Variable) — it wraps a
//! [`GpuTensor`] and records operations for reverse-mode automatic differentiation.
//! All data is `f32` because most GPUs lack native `f64` support.

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;

use scivex_core::Tensor;
use scivex_gpu::{GpuDevice, GpuTensor};

use crate::error::Result;

/// Global counter for node ids.
fn next_id() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Gradient function: given the upstream gradient, returns gradients for each parent.
type GpuGradFn = Box<dyn Fn(&GpuTensor) -> Vec<GpuTensor>>;

/// Internal node of the GPU computation graph.
struct GpuNode {
    data: GpuTensor,
    grad: Option<GpuTensor>,
    requires_grad: bool,
    grad_fn: Option<GpuGradFn>,
    parents: Vec<GpuVariable>,
    /// Unique id for topological-sort visited tracking.
    id: usize,
}

/// A GPU-backed autograd variable.
///
/// This is the GPU equivalent of [`Variable<T>`](crate::Variable). It wraps a
/// [`GpuTensor`] (f32-only) and builds a computation graph for automatic
/// differentiation via [`backward`](Self::backward).
///
/// Cloning a `GpuVariable` is cheap — it just increments an `Rc` reference count.
#[derive(Clone)]
pub struct GpuVariable {
    inner: Rc<RefCell<GpuNode>>,
}

impl GpuVariable {
    // ── Constructors ────────────────────────────────────────────────

    /// Create a new leaf variable from a [`GpuTensor`].
    pub fn new(data: GpuTensor, requires_grad: bool) -> Self {
        Self {
            inner: Rc::new(RefCell::new(GpuNode {
                data,
                grad: None,
                requires_grad,
                grad_fn: None,
                parents: Vec::new(),
                id: next_id(),
            })),
        }
    }

    /// Create an internal (non-leaf) variable produced by an operation.
    pub(crate) fn from_op(data: GpuTensor, parents: Vec<GpuVariable>, grad_fn: GpuGradFn) -> Self {
        Self {
            inner: Rc::new(RefCell::new(GpuNode {
                data,
                grad: None,
                requires_grad: true,
                grad_fn: Some(grad_fn),
                parents,
                id: next_id(),
            })),
        }
    }

    // ── Accessors ───────────────────────────────────────────────────

    /// Download the underlying data to a CPU [`Tensor<f32>`].
    pub fn data_cpu(&self) -> Result<Tensor<f32>> {
        Ok(self.inner.borrow().data.to_tensor()?)
    }

    /// Execute a function with a shared reference to the underlying [`GpuTensor`].
    ///
    /// This borrows the inner `RefCell` for the duration of the closure, so
    /// callers must not hold a mutable borrow at the same time.
    pub fn with_data<R>(&self, f: impl FnOnce(&GpuTensor) -> R) -> R {
        let node = self.inner.borrow();
        f(&node.data)
    }

    /// Return the shape of the underlying tensor.
    pub fn shape(&self) -> Vec<usize> {
        self.inner.borrow().data.shape().to_vec()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.inner.borrow().data.numel()
    }

    /// Get a clone of the [`GpuDevice`] this variable lives on.
    pub fn device(&self) -> GpuDevice {
        self.inner.borrow().data.device().clone()
    }

    /// Whether this variable tracks gradients.
    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
    }

    /// Download the accumulated gradient to CPU, if any.
    pub fn grad_cpu(&self) -> Option<Result<Tensor<f32>>> {
        let node = self.inner.borrow();
        node.grad.as_ref().map(|g| Ok(g.to_tensor()?))
    }

    /// Unique node id (used internally for graph traversal).
    fn id(&self) -> usize {
        self.inner.borrow().id
    }

    // ── Gradient helpers ────────────────────────────────────────────

    /// Reset the gradient to `None`.
    pub fn zero_grad(&self) {
        self.inner.borrow_mut().grad = None;
    }

    /// Replace the data tensor (used by optimizers and weight updates).
    pub fn set_data(&self, data: GpuTensor) {
        self.inner.borrow_mut().data = data;
    }

    /// Detach from the computation graph, returning a new leaf variable
    /// with the same data but no graph history.
    ///
    /// This downloads and re-uploads the data, since `GpuTensor` is not `Clone`.
    pub fn detach(&self) -> Result<Self> {
        let cpu = self.data_cpu()?;
        let device = self.device();
        let gpu = GpuTensor::from_tensor(&device, &cpu);
        Ok(Self::new(gpu, false))
    }

    /// Accumulate `g` into this node's gradient (summing if one already exists).
    pub(crate) fn acc_grad(&self, g: GpuTensor) {
        let mut node = self.inner.borrow_mut();
        node.grad = Some(match node.grad.take() {
            Some(existing) => {
                scivex_gpu::ops::add(&existing, &g).expect("gradient shapes must match")
            }
            None => g,
        });
    }

    // ── Backward pass ───────────────────────────────────────────────

    /// Run reverse-mode automatic differentiation starting from this variable.
    ///
    /// This variable is expected to be a scalar (single-element tensor). Its
    /// gradient is seeded with `1.0`. After `backward()`, each ancestor with
    /// `requires_grad == true` will have its `.grad_cpu()` populated.
    pub fn backward(&self) {
        // Topological sort (leaves first). Reverse for output-first order.
        let mut order = self.topo_sort();
        order.reverse();

        // Seed gradient with ones.
        {
            let node = self.inner.borrow();
            let shape = node.data.shape().to_vec();
            let device = node.data.device().clone();
            drop(node);
            let ones = scivex_gpu::ops::fill(&device, shape, 1.0).expect("fill for seed gradient");
            self.acc_grad(ones);
        }

        // Reverse walk.
        for var in &order {
            // Extract what we need and drop the borrow before touching parents.
            let (grad_fn, parents, grad_cpu) = {
                let node = var.inner.borrow();
                let gf = node.grad_fn.is_some();
                let parents = node.parents.clone();
                // Download gradient to CPU so we can release the borrow and re-upload.
                let grad_cpu = node.grad.as_ref().map(|g| {
                    let shape = g.shape().to_vec();
                    let device = g.device().clone();
                    let tensor = g.to_tensor().expect("gradient download");
                    (tensor, shape, device)
                });
                (gf, parents, grad_cpu)
            };

            if let (true, Some((grad_tensor, _grad_shape, grad_device))) = (grad_fn, grad_cpu) {
                // Re-upload gradient and call grad_fn.
                let grad_gpu = GpuTensor::from_tensor(&grad_device, &grad_tensor);

                // Now borrow again to get the grad_fn reference.
                let node = var.inner.borrow();
                let gf = node.grad_fn.as_ref().expect("checked is_some above");
                let parent_grads = gf(&grad_gpu);
                drop(node);

                for (parent, pg) in parents.iter().zip(parent_grads) {
                    if parent.requires_grad() {
                        parent.acc_grad(pg);
                    }
                }
            }
        }
    }

    /// Topological sort via iterative DFS.
    fn topo_sort(&self) -> Vec<GpuVariable> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        let mut stack: Vec<(GpuVariable, bool)> = vec![(self.clone(), false)];

        while let Some((var, processed)) = stack.pop() {
            let vid = var.id();
            if processed {
                if !visited.contains(&vid) {
                    visited.insert(vid);
                    order.push(var);
                }
                continue;
            }
            if visited.contains(&vid) {
                continue;
            }
            stack.push((var.clone(), true));
            let node = var.inner.borrow();
            for parent in &node.parents {
                if !visited.contains(&parent.id()) {
                    stack.push((parent.clone(), false));
                }
            }
        }

        order
    }
}

impl fmt::Debug for GpuVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node = self.inner.borrow();
        f.debug_struct("GpuVariable")
            .field("shape", &node.data.shape())
            .field("requires_grad", &node.requires_grad)
            .field("has_grad", &node.grad.is_some())
            .finish()
    }
}
