use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;

use scivex_core::{Float, Tensor};

/// Closure that computes parent gradients from the output gradient.
///
/// Given the gradient of the output, returns a `Vec` of gradients for each parent
/// in the same order as `parents`.
type GradFn<T> = Box<dyn Fn(&Tensor<T>) -> Vec<Tensor<T>>>;

/// Internal node of the autograd computation graph.
struct Node<T: Float> {
    data: Tensor<T>,
    grad: Option<Tensor<T>>,
    requires_grad: bool,
    grad_fn: Option<GradFn<T>>,
    parents: Vec<Variable<T>>,
    /// Unique id for topological-sort visited tracking.
    id: usize,
}

/// Global counter for node ids.
fn next_id() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// A variable in the computation graph that wraps a [`Tensor`] and supports
/// reverse-mode automatic differentiation.
///
/// `Variable<T>` uses shared ownership (`Rc<RefCell<...>>`) so that the same
/// node can appear as a parent in multiple downstream operations. Cloning a
/// `Variable` is cheap — it just increments the reference count.
pub struct Variable<T: Float> {
    inner: Rc<RefCell<Node<T>>>,
}

impl<T: Float> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl<T: Float> Variable<T> {
    // ── Constructors ────────────────────────────────────────────────

    /// Create a new leaf variable.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_core::Tensor;
    /// # use scivex_nn::Variable;
    /// let t = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let v = Variable::new(t, true);
    /// assert!(v.requires_grad());
    /// assert_eq!(v.shape(), vec![3]);
    /// ```
    pub fn new(data: Tensor<T>, requires_grad: bool) -> Self {
        Self {
            inner: Rc::new(RefCell::new(Node {
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
    pub(crate) fn from_op(data: Tensor<T>, parents: Vec<Variable<T>>, grad_fn: GradFn<T>) -> Self {
        Self {
            inner: Rc::new(RefCell::new(Node {
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

    /// Return a clone of the underlying tensor data.
    pub fn data(&self) -> Tensor<T> {
        self.inner.borrow().data.clone()
    }

    /// Return the shape of the underlying tensor.
    pub fn shape(&self) -> Vec<usize> {
        self.inner.borrow().data.shape().to_vec()
    }

    /// Return the accumulated gradient, if any.
    pub fn grad(&self) -> Option<Tensor<T>> {
        self.inner.borrow().grad.clone()
    }

    /// Whether this variable tracks gradients.
    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
    }

    /// Unique node id (used internally for graph traversal).
    pub(crate) fn id(&self) -> usize {
        self.inner.borrow().id
    }

    // ── Gradient helpers ────────────────────────────────────────────

    /// Reset the gradient to `None`.
    pub fn zero_grad(&self) {
        self.inner.borrow_mut().grad = None;
    }

    /// Detach from the computation graph, returning a new leaf variable
    /// with the same data but no graph history.
    pub fn detach(&self) -> Self {
        Self::new(self.data(), false)
    }

    /// Replace the data tensor (used by optimizers and weight loading).
    pub fn set_data(&self, data: Tensor<T>) {
        self.inner.borrow_mut().data = data;
    }

    /// Accumulate `g` into this node's gradient (summing if one already exists).
    pub(crate) fn acc_grad(&self, g: &Tensor<T>) {
        let mut node = self.inner.borrow_mut();
        node.grad = Some(match node.grad.take() {
            Some(existing) => &existing + g,
            None => g.clone(),
        });
    }

    // ── Backward pass ───────────────────────────────────────────────

    /// Run reverse-mode automatic differentiation starting from this variable.
    ///
    /// This variable is expected to be a scalar (single-element tensor). Its
    /// gradient is seeded with `1.0`. After `backward()`, each ancestor with
    /// `requires_grad == true` will have its `.grad()` populated.
    pub fn backward(&self) {
        // Topological sort produces post-order (leaves first).
        // Reverse to get output-first order for backward pass.
        let mut order = self.topo_sort();
        order.reverse();

        // Seed gradient.
        {
            let node = self.inner.borrow();
            let ones = Tensor::ones(node.data.shape().to_vec());
            drop(node);
            self.acc_grad(&ones);
        }

        // Reverse walk.
        for var in &order {
            let node = var.inner.borrow();
            let grad_fn = node.grad_fn.as_ref();
            let parents_clone: Vec<Variable<T>> = node.parents.clone();
            let grad_val = node.grad.clone();

            if let (Some(gf), Some(g)) = (grad_fn, grad_val) {
                let parent_grads = gf(&g);
                // Drop the borrow before touching parents.
                drop(node);
                for (parent, pg) in parents_clone.iter().zip(parent_grads) {
                    if parent.requires_grad() {
                        parent.acc_grad(&pg);
                    }
                }
            }
        }
    }

    /// Topological sort via iterative DFS (output-first order).
    fn topo_sort(&self) -> Vec<Variable<T>> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();

        // Iterative DFS with explicit stack.
        // Each entry is (variable, processed_flag).
        let mut stack: Vec<(Variable<T>, bool)> = vec![(self.clone(), false)];

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
            // Push this node again with processed=true so it gets added after children.
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

impl<T: Float> fmt::Debug for Variable<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node = self.inner.borrow();
        f.debug_struct("Variable")
            .field("shape", &node.data.shape())
            .field("requires_grad", &node.requires_grad)
            .field("has_grad", &node.grad.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaf_variable() {
        let t = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let v = Variable::new(t.clone(), true);
        assert_eq!(v.data().as_slice(), t.as_slice());
        assert!(v.requires_grad());
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_detach() {
        let t = Tensor::<f64>::ones(vec![2, 3]);
        let v = Variable::new(t, true);
        let d = v.detach();
        assert!(!d.requires_grad());
    }

    #[test]
    fn test_zero_grad() {
        let t = Tensor::<f64>::ones(vec![2]);
        let v = Variable::new(t, true);
        v.acc_grad(&Tensor::ones(vec![2]));
        assert!(v.grad().is_some());
        v.zero_grad();
        assert!(v.grad().is_none());
    }

    #[test]
    fn test_scalar_backward() {
        // f(x) = x, x is scalar => grad = 1
        let x = Variable::new(Tensor::from_vec(vec![3.0_f64], vec![1]).unwrap(), true);
        // Identity op
        let y = Variable::from_op(
            x.data(),
            vec![x.clone()],
            Box::new(|g: &Tensor<f64>| vec![g.clone()]),
        );
        y.backward();
        let g = x.grad().unwrap();
        assert_eq!(g.as_slice(), &[1.0]);
    }

    #[test]
    fn test_shape_accessor() {
        let t = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let v = Variable::new(t, false);
        assert_eq!(v.shape(), vec![2, 3]);
    }

    #[test]
    fn test_no_grad_variable_backward_does_not_accumulate() {
        // A variable with requires_grad=false should not accumulate gradients
        // even when it appears as a parent.
        let x = Variable::new(Tensor::from_vec(vec![2.0_f64], vec![1]).unwrap(), false);
        let y = Variable::new(Tensor::from_vec(vec![3.0_f64], vec![1]).unwrap(), true);
        let z = Variable::from_op(
            &x.data() + &y.data(),
            vec![x.clone(), y.clone()],
            Box::new(|g: &Tensor<f64>| vec![g.clone(), g.clone()]),
        );
        z.backward();
        // x does not require grad, so it should have no accumulated gradient.
        assert!(x.grad().is_none());
        // y requires grad, so it should have accumulated gradient.
        assert!(y.grad().is_some());
        assert_eq!(y.grad().unwrap().as_slice(), &[1.0]);
    }

    #[test]
    fn test_gradient_accumulation() {
        // When acc_grad is called twice, gradients should sum.
        let v = Variable::new(Tensor::from_vec(vec![1.0_f64, 2.0], vec![2]).unwrap(), true);
        v.acc_grad(&Tensor::from_vec(vec![1.0, 1.0], vec![2]).unwrap());
        v.acc_grad(&Tensor::from_vec(vec![2.0, 3.0], vec![2]).unwrap());
        let g = v.grad().unwrap();
        assert_eq!(g.as_slice(), &[3.0, 4.0]);
    }
}
