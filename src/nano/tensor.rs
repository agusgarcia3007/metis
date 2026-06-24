//! The autograd core: a 2D float32 tensor with a gradient, and a Tape that records each
//! op's backward closure so `backward` can replay them in reverse.
//!
//! The Go original used a pointer graph (`*Tensor` with a `bw func()` closure per node). This
//! Rust port models the same graph with `Rc<RefCell<Tensor>>` handles captured by boxed backward
//! closures stored on the Tape, recorded in creation (topological) order.
//!
//! Note on parallelism: the Go engine parallelized matmuls across goroutines (`parFor`). Each
//! output element's reduction is computed by a single worker, so results are independent of worker
//! count. This port computes those loops sequentially — numerically identical and fully
//! deterministic — trading some throughput for a safe, allocation-free translation.

use std::cell::RefCell;
use std::rc::Rc;

/// Tensor is a 2D [r,c] float32 array with a gradient buffer.
pub struct Tensor {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub r: usize,
    pub c: usize,
}

/// A shared, interior-mutable handle to a tensor (the node type of the autograd graph).
pub type T = Rc<RefCell<Tensor>>;

impl Tensor {
    /// ZeroGrad clears this tensor's gradient buffer between steps.
    pub fn zero_grad(&mut self) {
        for g in self.grad.iter_mut() {
            *g = 0.0;
        }
    }
}

/// Tape records each op's backward closure in creation order so [`Tape::backward`] can replay
/// them in reverse. Parameters are leaves and live off-tape.
pub struct Tape {
    pub(crate) bws: Vec<Box<dyn Fn()>>,
}

impl Default for Tape {
    fn default() -> Self {
        Tape::new()
    }
}

impl Tape {
    /// NewTape returns an empty tape for one forward/backward pass.
    pub fn new() -> Tape {
        Tape { bws: Vec::new() }
    }

    /// node allocates a fresh tracked tensor (zeroed data + grad).
    pub(crate) fn node(r: usize, c: usize) -> T {
        Rc::new(RefCell::new(Tensor {
            data: vec![0.0; r * c],
            grad: vec![0.0; r * c],
            r,
            c,
        }))
    }

    pub(crate) fn push_bw(&mut self, f: Box<dyn Fn()>) {
        self.bws.push(f);
    }

    /// Backward seeds d(loss)/d(loss)=1 and replays every recorded backward in reverse order.
    pub fn backward(&self, loss: &T) {
        loss.borrow_mut().grad[0] = 1.0;
        for bw in self.bws.iter().rev() {
            bw();
        }
    }
}

/// NewParam creates a persistent leaf parameter tensor (kept across steps).
pub fn new_param(r: usize, c: usize) -> T {
    Rc::new(RefCell::new(Tensor {
        data: vec![0.0; r * c],
        grad: vec![0.0; r * c],
        r,
        c,
    }))
}

/// Leaf wraps existing data as a non-tracked input (e.g. token indices, masks).
#[allow(dead_code)]
pub fn leaf(data: Vec<f32>, r: usize, c: usize) -> T {
    let n = r * c;
    Rc::new(RefCell::new(Tensor {
        data,
        grad: vec![0.0; n],
        r,
        c,
    }))
}

/// GELU activation (tanh approximation).
pub(crate) fn gelu_tanh(x: f32) -> f32 {
    // 0.5x(1+tanh(√(2/π)(x+0.044715x³)))
    let x3 = x * x * x;
    let inner = 0.7978845608_f32 * (x + 0.044715_f32 * x3);
    0.5 * x * (1.0 + (inner as f64).tanh() as f32)
}

pub(crate) fn gelu_grad(x: f32) -> f32 {
    let x3 = x * x * x;
    let inner = 0.7978845608_f32 * (x + 0.044715_f32 * x3);
    let t = (inner as f64).tanh() as f32;
    let sech2 = 1.0 - t * t;
    let dinner = 0.7978845608_f32 * (1.0 + 3.0 * 0.044715_f32 * x * x);
    0.5 * (1.0 + t) + 0.5 * x * sech2 * dinner
}
