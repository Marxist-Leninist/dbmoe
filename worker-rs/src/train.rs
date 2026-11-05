//! Training: one step (`dblock_loss`, backward, clip, AdamW) and one lease (`train_lease` in dbmoe.py).

use burn::tensor::{Device, Tensor, TensorData};

use crate::model::{dblock_loss, read_f32, Batch, Cfg, HostTensor, Named, Params};
use crate::optim::{clip_grad_norm, AdamW, OptState};
use crate::sigma::{block_range, sample_sigma, Rng};

/// A job from the coordinator (the lease payload of dbmoe.py).
#[derive(Clone, Debug)]
pub struct Lease {
    pub version: String,
    pub lease: String,
    pub block: usize,
    pub block_version: i64,
    pub cfg: Cfg,
    pub sigmas: Vec<f64>,
    pub steps: usize,
    pub batch: usize,
    pub lr: f64,
    pub ttl: f64,
    /// `Shared.state_dict()`, keys without the `shared.` prefix.
    pub shared: Named,
    /// `Block.state_dict()` of the leased block, keys without the `block.` prefix.
    pub block_state: Named,
    /// Training windows, `[rows, ctx + 1]` bytes, row-major.
    pub data: Vec<u8>,
    pub data_rows: usize,
    pub data_cols: usize,
}

impl Lease {
    pub fn validate(&self) -> Result<(), String> {
        self.cfg.validate()?;
        if self.block >= self.cfg.blocks {
            return Err(format!("lease: block {} of {}", self.block, self.cfg.blocks));
        }
        let s = &self.sigmas;
        if self.cfg.blocks.checked_add(1) != Some(s.len())
            || s.iter().any(|v| !(v.is_finite() && *v > 0.0))
            || s.windows(2).any(|w| w[0] >= w[1])
        {
            return Err("lease: sigmas must be blocks+1 increasing positive numbers".into());
        }
        if self.steps == 0 || self.batch == 0 || self.steps > 100_000 || self.batch > 100_000 {
            return Err(format!("lease: steps={} batch={}", self.steps, self.batch));
        }
        if self.data_cols != self.cfg.ctx + 1 || self.data_rows.checked_mul(self.data_cols) != Some(self.data.len()) {
            return Err(format!("lease: data is [{}, {}], expected [*, {}]", self.data_rows, self.data_cols, self.cfg.ctx + 1));
        }
        if self.steps.checked_mul(self.batch).map_or(true, |need| self.data_rows < need) {
            return Err(format!("lease: {} windows for {} steps x {}", self.data_rows, self.steps, self.batch));
        }
        if !(self.lr.is_finite() && self.lr >= 0.0 && self.lr < 1.0) {
            return Err(format!("lease: lr {}", self.lr));
        }
        Ok(())
    }

    /// The parameters under the model's names (`shared.*`, `block.*`).
    pub fn named(&self) -> Named {
        let mut n = Named::new();
        for (k, v) in &self.shared {
            n.insert(format!("shared.{k}"), v.clone());
        }
        for (k, v) in &self.block_state {
            n.insert(format!("block.{k}"), v.clone());
        }
        n
    }
}

/// What a lease sends back (the update payload of dbmoe.py, without the worker's name and device).
#[derive(Clone, Debug)]
pub struct Update {
    pub lease: String,
    pub block: usize,
    pub losses: Vec<f64>,
    pub seconds: f64,
    pub tokens: u64,
    pub fresh_optimizer: bool,
    pub d_shared: Named,
    pub d_block: Named,
}

/// AdamW state of the two most recent blocks, so a worker that gets the same block again keeps its optimizer.
#[derive(Default)]
pub struct OptCache {
    entries: Vec<(usize, OptState, i64)>,
}

impl OptCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn take(&mut self, block: usize) -> Option<(OptState, i64)> {
        let i = self.entries.iter().position(|e| e.0 == block)?;
        let (_, s, v) = self.entries.remove(i);
        Some((s, v))
    }

    fn put(&mut self, block: usize, state: OptState, version: i64) {
        self.entries.retain(|e| e.0 != block);
        self.entries.push((block, state, version));
        while self.entries.len() > 2 {
            self.entries.remove(0);
        }
    }
}

/// Results of one training step.
#[derive(Clone, Copy, Debug)]
pub struct StepOut {
    pub loss: f32,
    pub ce: f32,
    pub grad_norm: f32,
}

/// One optimizer step: forward, backward, `clip_grad_norm_(clip)`, AdamW at `lr`.
#[allow(clippy::too_many_arguments)]
pub async fn train_step(
    params: &mut Params,
    state: &mut OptState,
    opt: &AdamW,
    cfg: &Cfg,
    batch: &Batch,
    eps: Tensor<3>,
    lr: f64,
    clip: f32,
) -> StepOut {
    let leaves = params.leaves();
    let fwd = dblock_loss(&leaves, cfg, batch, eps).await;
    let grads = fwd.loss.backward();
    let mut g: Vec<Option<Tensor<1>>> = leaves.tensors.iter().map(|t| t.grad(&grads)).collect();
    let grad_norm = clip_grad_norm(&mut g, clip).await;
    opt.step(params, &g, state, lr);
    let stats = read_f32(Tensor::cat(vec![fwd.loss.detach(), fwd.ce], 0)).await;
    StepOut { loss: stats[0], ce: stats[1], grad_norm }
}

/// Trains the leased block (and the shared parameters) for `lease.steps` steps and returns the deltas, as
/// `train_lease` does: a fresh optimizer first watches a few steps at lr 0 and warms up over 10; a cached one (same
/// block, at most 4 versions old) keeps its per-parameter scale but not its momentum.
pub async fn train_lease(lease: &Lease, device: &Device, cache: &mut OptCache, rng: &mut Rng, opt: &AdamW) -> Result<Update, String> {
    lease.validate()?;
    let cfg = &lease.cfg;
    let (lo, hi) = block_range(cfg, &lease.sigmas, lease.block);
    let named = lease.named();
    let mut params = Params::from_host(cfg, &named, device)?;
    let mut state = OptState::default();
    let mut warm = 10usize;
    if let Some((cached, version)) = cache.take(lease.block) {
        if lease.block_version.checked_sub(version).is_some_and(|age| (0..=4).contains(&age)) && cached.fits(&params) {
            state = cached;
            state.drop_momentum();
            warm = 0;
        }
    }
    let (steps, bs, t1) = (lease.steps, lease.batch, lease.data_cols);
    let settle = if warm > 0 { (steps / 4).min(8) } else { 0 };
    let mut losses = Vec::with_capacity(steps);
    let t0 = web_time::Instant::now();
    for s in 0..steps {
        let scale = if warm == 0 {
            1.0
        } else if s < settle {
            0.0
        } else {
            ((s + 1 - settle) as f64 / warm as f64).min(1.0)
        };
        let rows = &lease.data[s * bs * t1..(s + 1) * bs * t1];
        let batch = Batch { ids: rows.iter().map(|&x| x as usize).collect(), b: bs, t1, sigma: sample_sigma(cfg, lo, hi, bs, rng) };
        batch.validate(cfg)?;
        // Draw sigma and Gaussian noise from the worker's seeded stream, never Burn's separate backend RNG.
        let eps = Tensor::<3>::from_data(
            TensorData::new(rng.standard_normals(bs * (t1 - 1) * cfg.d), [bs, t1 - 1, cfg.d]),
            device,
        );
        let out = train_step(&mut params, &mut state, opt, cfg, &batch, eps, lease.lr * scale, 1.0).await;
        if !out.loss.is_finite() {
            return Err(format!("step {s}: loss is not finite"));
        }
        losses.push((out.ce as f64 * 1e4).round() / 1e4);
    }
    let seconds = ((t0.elapsed().as_secs_f64() * 100.0).round() / 100.0).max(0.01);
    let after = params.to_host().await;
    let (mut d_shared, mut d_block) = (Named::new(), Named::new());
    for (name, now) in after {
        let before = &named[&name];
        let d = HostTensor { shape: now.shape, data: now.data.iter().zip(&before.data).map(|(a, b)| a - b).collect() };
        if let Some(k) = name.strip_prefix("shared.") {
            d_shared.insert(k.to_string(), d);
        } else if let Some(k) = name.strip_prefix("block.") {
            d_block.insert(k.to_string(), d);
        }
    }
    cache.put(lease.block, state, lease.block_version.saturating_add(1));
    Ok(Update {
        lease: lease.lease.clone(),
        block: lease.block,
        losses,
        seconds,
        tokens: (steps * bs * cfg.ctx) as u64,
        fresh_optimizer: warm > 0,
        d_shared,
        d_block,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bench::random_params, block_on};

    fn tiny_lease() -> Lease {
        let cfg = Cfg { d: 8, heads: 2, layers_per_block: 1, blocks: 2, experts: 2, ffn: 12, ctx: 4, ..Cfg::live() };
        let named = random_params(&cfg, &mut Rng::new(5));
        let block_state = named
            .iter()
            .filter_map(|(name, value)| name.strip_prefix("block.").map(|name| (name.to_string(), value.clone())))
            .collect();
        let shared = named
            .into_iter()
            .filter_map(|(name, value)| name.strip_prefix("shared.").map(|name| (name.to_string(), value)))
            .collect();
        Lease {
            version: "test".into(),
            lease: "seeded-replay".into(),
            block: 0,
            block_version: 0,
            cfg,
            sigmas: vec![0.3397, 1.2, 80.0],
            steps: 2,
            batch: 1,
            lr: 1e-4,
            ttl: 60.0,
            shared,
            block_state,
            data: vec![7, 65, 110, 116, 33, 32, 7, 65, 110, 116],
            data_rows: 2,
            data_cols: 5,
        }
    }

    fn replay(lease: &Lease, seed: u64) -> Update {
        let device = Device::flex();
        let mut cache = OptCache::new();
        let mut rng = Rng::new(seed);
        let opt = AdamW::default();
        block_on(train_lease(lease, &device, &mut cache, &mut rng, &opt)).unwrap()
    }

    #[test]
    fn seeded_replay_controls_sigma_and_gaussian_training_noise() {
        let lease = tiny_lease();
        let first = replay(&lease, 123);
        let same = replay(&lease, 123);
        assert_eq!(first.losses, same.losses);
        assert_eq!(first.d_shared, same.d_shared);
        assert_eq!(first.d_block, same.d_block);

        let different = replay(&lease, 124);
        assert_ne!(first.d_block, different.d_block);
    }
}
