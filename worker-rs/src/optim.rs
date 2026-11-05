//! Gradient clipping and AdamW, as PyTorch computes them (`clip_grad_norm_`, `_single_tensor_adamw`).
//!
//! Every update runs as tensor operations on the parameters' device, in PyTorch's operation order, so on the CPU
//! one step matches PyTorch to float32 rounding.

use burn::tensor::Tensor;

use crate::model::{read_f32, Params};

/// AdamW hyper-parameters. Weight decay applies to parameters of rank >= 2 only, as in the worker's two groups.
#[derive(Clone, Debug)]
pub struct AdamW {
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay_matrix: f64,
    pub weight_decay_vector: f64,
}

impl Default for AdamW {
    /// The worker's optimizer: `AdamW(lr, betas=(0.9, 0.95))`, eps 1e-8, decay 0.01 on matrices.
    fn default() -> Self {
        Self { beta1: 0.9, beta2: 0.95, eps: 1e-8, weight_decay_matrix: 0.01, weight_decay_vector: 0.0 }
    }
}

/// One parameter's state: PyTorch's `step`, `exp_avg` and `exp_avg_sq`.
#[derive(Clone)]
pub struct Slot {
    pub step: u64,
    pub m: Tensor<1>,
    pub v: Tensor<1>,
}

/// Optimizer state aligned with a [`Params`] list. A slot appears on the parameter's first gradient.
#[derive(Clone, Default)]
pub struct OptState {
    pub slots: Vec<Option<Slot>>,
}

impl OptState {
    /// True when every slot's size matches the parameter it belongs to.
    pub fn fits(&self, params: &Params) -> bool {
        self.slots.len() == params.len()
            && self.slots.iter().zip(&params.shapes).all(|(s, shape)| match s {
                Some(s) => s.m.dims()[0] == shape.iter().product::<usize>().max(1),
                None => true,
            })
    }

    /// Keeps each parameter's scale (`exp_avg_sq`) and step count but drops the momentum, as the Python worker does
    /// when it gets the same block again.
    pub fn drop_momentum(&mut self) {
        for s in self.slots.iter_mut().flatten() {
            s.m = s.m.zeros_like();
        }
    }
}

/// `torch.nn.utils.clip_grad_norm_(params, max_norm)`: scales the gradients in place when their total 2-norm
/// exceeds `max_norm` and returns the norm before clipping.
pub async fn clip_grad_norm(grads: &mut [Option<Tensor<1>>], max_norm: f32) -> f32 {
    let sums: Vec<Tensor<1>> = grads.iter().flatten().map(|g| (g.clone() * g.clone()).sum()).collect();
    if sums.is_empty() {
        return 0.0;
    }
    let sums = read_f32(Tensor::cat(sums, 0)).await;
    // per-tensor norms in float32 (as torch.linalg.vector_norm returns them), then the norm of the norms
    let total = sums.iter().map(|&s| (s.sqrt() as f64).powi(2)).sum::<f64>().sqrt() as f32;
    let coef = (max_norm / (total + 1e-6)).min(1.0);
    if coef < 1.0 {
        for g in grads.iter_mut().flatten() {
            *g = g.clone() * coef;
        }
    }
    total
}

impl AdamW {
    /// One `optimizer.step()` at learning rate `lr`. Parameters whose gradient is `None` are left alone, as PyTorch
    /// skips parameters without `.grad`.
    pub fn step(&self, params: &mut Params, grads: &[Option<Tensor<1>>], state: &mut OptState, lr: f64) {
        assert_eq!(grads.len(), params.len(), "one gradient slot per parameter");
        if state.slots.len() != params.len() {
            state.slots.resize(params.len(), None);
        }
        let (b1, b2) = (self.beta1, self.beta2);
        for (i, g) in grads.iter().enumerate() {
            let Some(g) = g else { continue };
            let wd = if params.shapes[i].len() >= 2 { self.weight_decay_matrix } else { self.weight_decay_vector };
            let slot = state.slots[i].get_or_insert_with(|| Slot { step: 0, m: g.zeros_like(), v: g.zeros_like() });
            slot.step += 1;
            let step = slot.step as f64;
            // param.mul_(1 - lr * weight_decay)
            let p = params.tensors[i].clone() * ((1.0 - lr * wd) as f32);
            // exp_avg.lerp_(grad, 1 - beta1); exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            let m = slot.m.clone() + (g.clone() - slot.m.clone()) * ((1.0 - b1) as f32);
            let v = slot.v.clone() * (b2 as f32) + (g.clone() * ((1.0 - b2) as f32)) * g.clone();
            // denom = exp_avg_sq.sqrt() / sqrt(1 - beta2**step) + eps; param.addcdiv_(exp_avg, denom, -lr / (1 - beta1**step))
            let step_size = lr / (1.0 - b1.powf(step));
            let bias2_sqrt = (1.0 - b2.powf(step)).sqrt();
            let denom = v.clone().sqrt() / (bias2_sqrt as f32) + (self.eps as f32);
            params.tensors[i] = p + (m.clone() * ((-step_size) as f32)) / denom;
            slot.m = m;
            slot.v = v;
        }
    }
}
