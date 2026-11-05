//! Training-step timing on random bytes: the `bench` command and the browser page.

use burn::tensor::{Device, Distribution, Tensor};
use serde::Serialize;

use crate::model::{Batch, Cfg, HostTensor, Named, Params};
use crate::optim::{AdamW, OptState};
use crate::sigma::Rng;
use crate::train::train_step;

/// Parameters at roughly the model's initial scales: unit LayerNorm weights and output scale, zero biases, small
/// uniform values elsewhere. Good enough for timing; not a trained model.
pub fn random_params(cfg: &Cfg, rng: &mut Rng) -> Named {
    cfg.param_specs()
        .into_iter()
        .map(|(k, shape)| {
            let n: usize = shape.iter().product::<usize>().max(1);
            let unit = k == "shared.scale" || k.ends_with("ln.weight") || k.ends_with("ln1.weight") || k.ends_with("ln2.weight");
            let zero = k.ends_with(".bias");
            let data = (0..n)
                .map(|_| if unit { 1.0 } else if zero { 0.0 } else { ((rng.uniform() * 2.0 - 1.0) * 0.08) as f32 })
                .collect();
            (k, HostTensor { shape, data })
        })
        .collect()
}

/// Timing of `steps` training steps after `warmup` untimed ones.
#[derive(Clone, Debug, Serialize)]
pub struct BenchResult {
    pub steps: usize,
    pub batch: usize,
    pub ctx: usize,
    pub step_s: Vec<f64>,
    pub median_step_s: f64,
    pub mean_step_s: f64,
    /// Training bytes per second, `batch * ctx / median step time` (the unit the coordinator reports).
    pub bytes_per_s: f64,
    pub last_loss: f32,
}

/// Times full training steps (forward, backward, clipping, AdamW) of `cfg` on random bytes. `on_step(step, seconds,
/// loss, grad_norm)` is called after every step, warm-up included.
pub async fn run(
    cfg: &Cfg,
    device: &Device,
    steps: usize,
    warmup: usize,
    batch: usize,
    seed: u64,
    mut on_step: impl FnMut(usize, f64, f32, f32),
) -> Result<BenchResult, String> {
    cfg.validate()?;
    if steps == 0 || batch == 0 {
        return Err("bench: steps and batch must be positive".into());
    }
    let mut rng = Rng::new(seed);
    let mut params = Params::from_host(cfg, &random_params(cfg, &mut rng), device)?;
    let (mut state, opt, t1) = (OptState::default(), AdamW::default(), cfg.ctx + 1);
    let (mut times, mut last) = (Vec::with_capacity(steps), f32::NAN);
    for s in 0..warmup + steps {
        let b = Batch {
            ids: (0..batch * t1).map(|_| (rng.next_u64() % cfg.vocab as u64) as usize).collect(),
            b: batch,
            t1,
            sigma: (0..batch).map(|_| (0.5 + rng.uniform() * 3.0) as f32).collect(),
        };
        let t0 = web_time::Instant::now();
        let eps = Tensor::<3>::random([batch, cfg.ctx, cfg.d], Distribution::Normal(0.0, 1.0), device);
        let out = train_step(&mut params, &mut state, &opt, cfg, &b, eps, 1e-4, 1.0).await;
        let dt = t0.elapsed().as_secs_f64();
        on_step(s, dt, out.loss, out.grad_norm);
        last = out.loss;
        if s >= warmup {
            times.push(dt);
        }
    }
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = sorted[sorted.len() / 2];
    Ok(BenchResult {
        steps,
        batch,
        ctx: cfg.ctx,
        mean_step_s: times.iter().sum::<f64>() / times.len() as f64,
        step_s: times,
        median_step_s: median,
        bytes_per_s: (batch * cfg.ctx) as f64 / median,
        last_loss: last,
    })
}
