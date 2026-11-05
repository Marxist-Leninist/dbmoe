//! The numerical contract with PyTorch: `apple/parity.json` (format `dbmoe.native.parity.v1`, written by
//! `apple/generate_parity_fixture.py`) holds one training step's inputs and PyTorch's outputs. [`run`] repeats the
//! step here and compares every number.

use burn::tensor::{Device, Tensor, TensorData};
use serde::Serialize;
use serde_json::Value;

use crate::model::{dblock_loss, read_f32, Batch, Cfg, HostTensor, Named, Params};
use crate::optim::{clip_grad_norm, AdamW, OptState};

/// A parsed parity fixture.
pub struct Fixture {
    pub case: String,
    pub source_sha256: String,
    pub cfg: Cfg,
    pub initial: Named,
    pub batch: Batch,
    pub epsilon: HostTensor,
    pub lr: f64,
    pub opt: AdamW,
    pub clip_norm: f32,
    pub expected: Expected,
}

/// PyTorch's results for the fixture's step.
pub struct Expected {
    pub loss: f64,
    pub ce: f64,
    pub auxiliary: f64,
    pub expert_fractions: Vec<HostTensor>,
    pub denoised: HostTensor,
    pub logits: HostTensor,
    pub gradient_norm_before_clip: f64,
    pub gradients: Named,
    pub clipped_gradients: Named,
    pub after_step: Named,
    pub deltas: Named,
}

fn host(v: &Value, what: &str) -> Result<HostTensor, String> {
    let shape = v["shape"]
        .as_array()
        .ok_or(format!("{what}: no shape"))?
        .iter()
        .map(|x| x.as_u64().map(|x| x as usize).ok_or(format!("{what}: bad shape")))
        .collect::<Result<Vec<_>, _>>()?;
    let data = v["data"]
        .as_array()
        .ok_or(format!("{what}: no data"))?
        .iter()
        .map(|x| x.as_f64().map(|x| x as f32).ok_or(format!("{what}: bad value")))
        .collect::<Result<Vec<_>, _>>()?;
    if data.len() != shape.iter().product::<usize>() {
        return Err(format!("{what}: {} values for shape {shape:?}", data.len()));
    }
    Ok(HostTensor { shape, data })
}

/// A name -> tensor map; `null` entries (PyTorch's `grad is None`) are left out.
fn named(v: &Value, what: &str) -> Result<Named, String> {
    let obj = v.as_object().ok_or(format!("{what}: not an object"))?;
    let mut out = Named::new();
    for (k, t) in obj {
        if !t.is_null() {
            out.insert(k.clone(), host(t, &format!("{what}.{k}"))?);
        }
    }
    Ok(out)
}

fn num(v: &Value, what: &str) -> Result<f64, String> {
    v.as_f64().ok_or(format!("{what}: not a number"))
}

/// Parses a fixture.
pub fn load(text: &str) -> Result<Fixture, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("fixture: {e}"))?;
    if v["format"] != "dbmoe.native.parity.v1" {
        return Err(format!("fixture: unknown format {}", v["format"]));
    }
    let cfg: Cfg = serde_json::from_value(v["cfg"].clone()).map_err(|e| format!("fixture cfg: {e}"))?;
    let b = host(&v["batch"], "batch")?;
    if b.shape.len() != 2 {
        return Err("fixture: batch must be [b, t+1]".into());
    }
    let sigma = host(&v["sigma"], "sigma")?;
    let batch = Batch { ids: b.data.iter().map(|&x| x as usize).collect(), b: b.shape[0], t1: b.shape[1], sigma: sigma.data };
    batch.validate(&cfg)?;
    let o = &v["optimizer"];
    if o["step"].as_u64() != Some(1) {
        return Err("fixture: only a first optimizer step is supported".into());
    }
    let betas = o["betas"].as_array().ok_or("fixture: optimizer.betas")?;
    let opt = AdamW {
        beta1: num(&betas[0], "beta1")?,
        beta2: num(&betas[1], "beta2")?,
        eps: num(&o["eps"], "eps")?,
        weight_decay_matrix: num(&o["weight_decay_matrix"], "weight_decay_matrix")?,
        weight_decay_vector: num(&o["weight_decay_vector"], "weight_decay_vector")?,
    };
    let e = &v["expected"];
    let fractions = e["expert_fractions"]
        .as_array()
        .ok_or("fixture: expert_fractions")?
        .iter()
        .enumerate()
        .map(|(i, f)| host(f, &format!("expert_fractions[{i}]")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Fixture {
        case: v["fixture_case"].as_str().unwrap_or("").to_string(),
        source_sha256: v["source_sha256"].as_str().unwrap_or("").to_string(),
        cfg,
        initial: named(&v["initial"], "initial")?,
        batch,
        epsilon: host(&v["epsilon"], "epsilon")?,
        lr: num(&o["lr"], "lr")?,
        opt,
        clip_norm: num(&o["clip_norm"], "clip_norm")? as f32,
        expected: Expected {
            loss: num(&e["loss"], "loss")?,
            ce: num(&e["ce"], "ce")?,
            auxiliary: num(&e["auxiliary"], "auxiliary")?,
            expert_fractions: fractions,
            denoised: host(&e["denoised"], "denoised")?,
            logits: host(&e["logits"], "logits")?,
            gradient_norm_before_clip: num(&e["gradient_norm_before_clip"], "gradient_norm_before_clip")?,
            gradients: named(&e["gradients"], "gradients")?,
            clipped_gradients: named(&e["clipped_gradients"], "clipped_gradients")?,
            after_step: named(&e["after_step"], "after_step")?,
            deltas: named(&e["deltas"], "deltas")?,
        },
    })
}

/// How close is close enough: `|got - want| <= atol + rtol * |want|` for every number.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Tolerance {
    pub atol: f64,
    pub rtol: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self { atol: 1e-5, rtol: 1e-4 }
    }
}

/// One compared quantity.
#[derive(Clone, Debug, Serialize)]
pub struct Check {
    pub name: String,
    pub n: usize,
    pub max_abs: f64,
    /// Largest error relative to the tensor's largest magnitude.
    pub max_rel_to_max: f64,
    pub worst: String,
    pub ok: bool,
}

/// The outcome of a parity run.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub case: String,
    pub device: String,
    pub tolerance: Tolerance,
    pub ok: bool,
    pub max_abs: f64,
    pub checks: Vec<Check>,
}

fn compare(name: &str, got: &[f32], want: &[f32], tol: Tolerance) -> Check {
    if got.len() != want.len() {
        return Check {
            name: name.into(),
            n: want.len(),
            max_abs: f64::INFINITY,
            max_rel_to_max: f64::INFINITY,
            worst: format!("{} values, expected {}", got.len(), want.len()),
            ok: false,
        };
    }
    let scale = want.iter().fold(0f64, |m, &w| m.max((w as f64).abs()));
    let (mut max_abs, mut worst, mut ok) = (0f64, String::new(), true);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g as f64 - w as f64).abs();
        if !err.is_finite() || err > tol.atol + tol.rtol * (w as f64).abs() {
            ok = false;
        }
        if !(err <= max_abs) {
            max_abs = err;
            worst = format!("[{i}] got {g:e} want {w:e}");
        }
    }
    Check { name: name.into(), n: want.len(), max_abs, max_rel_to_max: if scale > 0.0 { max_abs / scale } else { max_abs }, worst, ok }
}

/// Runs the fixture's training step on `device` and compares the forward values, the gradients before and after
/// clipping, the gradient norm, and the parameters and deltas after one AdamW step.
pub async fn run(fx: &Fixture, device: &Device, tol: Tolerance) -> Result<Report, String> {
    let cfg = &fx.cfg;
    let mut params = Params::from_host(cfg, &fx.initial, device)?;
    let eps_shape = &fx.epsilon.shape;
    if eps_shape != &[fx.batch.b, fx.batch.t1 - 1, cfg.d] {
        return Err(format!("fixture: epsilon shape {eps_shape:?}"));
    }
    let eps = Tensor::<3>::from_data(TensorData::new(fx.epsilon.data.clone(), [eps_shape[0], eps_shape[1], eps_shape[2]]), device);
    let leaves = params.leaves();
    let fwd = dblock_loss(&leaves, cfg, &fx.batch, eps).await;
    let x = &fx.expected;
    let mut checks = vec![];
    let scalar = |t: Tensor<1>| async move { read_f32(t).await };
    checks.push(compare("loss", &scalar(fwd.loss.clone().detach()).await, &[x.loss as f32], tol));
    checks.push(compare("ce", &scalar(fwd.ce.clone()).await, &[x.ce as f32], tol));
    checks.push(compare("auxiliary", &scalar(fwd.aux.clone()).await, &[x.auxiliary as f32], tol));
    for (i, (got, want)) in fwd.fractions.iter().zip(&x.expert_fractions).enumerate() {
        checks.push(compare(&format!("expert_fractions[{i}]"), got, &want.data, Tolerance { atol: 0.0, rtol: 0.0 }));
    }
    if fwd.fractions.len() != x.expert_fractions.len() {
        checks.push(compare("expert_fractions", &[], &[0.0], tol));
    }
    checks.push(compare("denoised", &read_f32(fwd.denoised.clone()).await, &x.denoised.data, tol));
    checks.push(compare("logits", &read_f32(fwd.logits.clone()).await, &x.logits.data, tol));

    let grads = fwd.loss.backward();
    let mut g: Vec<Option<Tensor<1>>> = leaves.tensors.iter().map(|t| t.grad(&grads)).collect();
    for (i, name) in params.names.iter().enumerate() {
        match (&g[i], x.gradients.get(name)) {
            (Some(t), Some(want)) => checks.push(compare(&format!("grad {name}"), &read_f32(t.clone()).await, &want.data, tol)),
            (None, None) => {}
            (got, _) => checks.push(compare(&format!("grad {name} (presence: got {})", got.is_some()), &[], &[0.0], tol)),
        }
    }
    let norm = clip_grad_norm(&mut g, fx.clip_norm).await;
    checks.push(compare("gradient_norm_before_clip", &[norm], &[x.gradient_norm_before_clip as f32], tol));
    for (i, name) in params.names.iter().enumerate() {
        if let (Some(t), Some(want)) = (&g[i], x.clipped_gradients.get(name)) {
            checks.push(compare(&format!("clipped {name}"), &read_f32(t.clone()).await, &want.data, tol));
        }
    }
    let before = params.to_host().await;
    let mut state = OptState::default();
    fx.opt.step(&mut params, &g, &mut state, fx.lr);
    let after = params.to_host().await;
    for name in &params.names {
        let (a, b) = (&after[name], &before[name]);
        if let Some(want) = x.after_step.get(name) {
            checks.push(compare(&format!("after_step {name}"), &a.data, &want.data, tol));
        }
        if let Some(want) = x.deltas.get(name) {
            let d: Vec<f32> = a.data.iter().zip(&b.data).map(|(p, q)| p - q).collect();
            checks.push(compare(&format!("delta {name}"), &d, &want.data, tol));
        }
    }
    let ok = checks.iter().all(|c| c.ok);
    let max_abs = checks.iter().fold(0f64, |m, c| m.max(c.max_abs));
    Ok(Report { case: fx.case.clone(), device: format!("{device:?}"), tolerance: tol, ok, max_abs, checks })
}
