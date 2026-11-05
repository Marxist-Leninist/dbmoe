//! The DBMoE model maths, ported line by line from `dbmoe.py`.
//!
//! Parameters are kept flat (rank 1) and reshaped where they are used; reshapes are free and keep the optimizer
//! rank-independent. Linear weights keep PyTorch's `[out, in]` layout, so state dicts move across unchanged.

use std::collections::BTreeMap;

use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::{Device, IndexingUpdateOp, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

/// `SD` in dbmoe.py: the standard deviation of the embedding rows the blocks denoise.
pub const SD: f32 = 0.5;
/// `torch.nn.LayerNorm`'s default epsilon.
pub const LN_EPS: f32 = 1e-5;

fn default_overlap() -> f64 {
    0.1
}

/// The model configuration (`make_cfg` in dbmoe.py). Unknown keys are ignored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cfg {
    pub vocab: usize,
    pub d: usize,
    pub heads: usize,
    pub layers_per_block: usize,
    pub blocks: usize,
    pub experts: usize,
    pub ffn: usize,
    pub ctx: usize,
    pub aux: f64,
    pub sig_pm: f64,
    pub sig_ps: f64,
    pub sig_min: f64,
    pub sig_max: f64,
    #[serde(default = "default_overlap")]
    pub overlap: f64,
}

impl Cfg {
    /// The configuration the live network trains (dbmoe.py `DEFAULT_CFG` with the 4-layer, 528-wide block).
    pub fn live() -> Self {
        Self {
            vocab: 256,
            d: 256,
            heads: 4,
            layers_per_block: 4,
            blocks: 6,
            experts: 4,
            ffn: 528,
            ctx: 256,
            aux: 0.01,
            sig_pm: 1.6284,
            sig_ps: 0.9,
            sig_min: 0.3397,
            sig_max: 80.0,
            overlap: 0.1,
        }
    }

    /// Checks the sizes the maths relies on.
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab == 0 || self.d == 0 || self.heads == 0 || self.experts == 0 || self.ffn == 0 || self.ctx == 0 {
            return Err("cfg: sizes must be positive".into());
        }
        if self.d % self.heads != 0 {
            return Err(format!("cfg: d={} is not divisible by heads={}", self.d, self.heads));
        }
        if self.blocks == 0 || self.layers_per_block == 0 {
            return Err("cfg: blocks and layers_per_block must be positive".into());
        }
        let big = 1usize << 20;
        if self.vocab > big || self.d > big || self.ffn > big || self.ctx > 1 << 14 || self.experts > 4096 || self.layers_per_block > 4096 || self.blocks > 4096 {
            return Err("cfg: implausibly large sizes".into());
        }
        if ![self.aux, self.sig_pm, self.sig_ps, self.sig_min, self.sig_max, self.overlap].iter().all(|v| v.is_finite()) || self.sig_ps <= 0.0 {
            return Err("cfg: noise settings must be finite (and sig_ps positive)".into());
        }
        Ok(())
    }

    /// The parameters one block job trains, as `(name, shape)` in PyTorch `named_parameters()` order: the shared
    /// embedding/positions/head (`shared.*`), then one block (`block.*`).
    pub fn param_specs(&self) -> Vec<(String, Vec<usize>)> {
        let (v, d, f, e) = (self.vocab, self.d, self.ffn, self.experts);
        let mut s: Vec<(String, Vec<usize>)> = vec![
            ("shared.pos".into(), vec![self.ctx, d]),
            ("shared.scale".into(), vec![]),
            ("shared.emb.weight".into(), vec![v, d]),
            ("shared.ln.weight".into(), vec![d]),
            ("shared.ln.bias".into(), vec![d]),
            ("block.inp.weight".into(), vec![d, d]),
            ("block.sig.weight".into(), vec![d, 1]),
            ("block.sig.bias".into(), vec![d]),
        ];
        for l in 0..self.layers_per_block {
            let p = format!("block.layers.{l}");
            s.push((format!("{p}.ln1.weight"), vec![d]));
            s.push((format!("{p}.ln1.bias"), vec![d]));
            s.push((format!("{p}.ln2.weight"), vec![d]));
            s.push((format!("{p}.ln2.bias"), vec![d]));
            s.push((format!("{p}.qkv.weight"), vec![3 * d, d]));
            s.push((format!("{p}.proj.weight"), vec![d, d]));
            s.push((format!("{p}.moe.w1"), vec![e, d, f]));
            s.push((format!("{p}.moe.w2"), vec![e, f, d]));
            s.push((format!("{p}.moe.router.weight"), vec![e, d]));
        }
        s.push(("block.ln.weight".into(), vec![d]));
        s.push(("block.ln.bias".into(), vec![d]));
        s.push(("block.out.weight".into(), vec![d, d]));
        s
    }
}

/// A tensor on the host: row-major `f32` data and its PyTorch shape (`[]` for a scalar).
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl HostTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Named host tensors, e.g. `"block.layers.0.qkv.weight"`.
pub type Named = BTreeMap<String, HostTensor>;

/// The trainable parameters of one block job, on a device, in [`Cfg::param_specs`] order.
#[derive(Clone)]
pub struct Params {
    pub names: Vec<String>,
    pub shapes: Vec<Vec<usize>>,
    pub tensors: Vec<Tensor<1>>,
    index: BTreeMap<String, usize>,
}

impl Params {
    /// Uploads named host tensors; the names and shapes must be exactly those of [`Cfg::param_specs`].
    pub fn from_host(cfg: &Cfg, named: &Named, device: &Device) -> Result<Self, String> {
        cfg.validate()?;
        let specs = cfg.param_specs();
        if named.len() != specs.len() {
            let want: Vec<&str> = specs.iter().map(|s| s.0.as_str()).collect();
            let extra: Vec<&String> = named.keys().filter(|k| !want.contains(&k.as_str())).collect();
            return Err(format!("expected {} parameters, got {} (unexpected: {extra:?})", specs.len(), named.len()));
        }
        let (mut names, mut shapes, mut tensors, mut index) = (vec![], vec![], vec![], BTreeMap::new());
        for (name, shape) in specs {
            let t = named.get(&name).ok_or_else(|| format!("missing parameter {name}"))?;
            if t.shape != shape {
                return Err(format!("{name}: shape {:?}, expected {shape:?}", t.shape));
            }
            let n: usize = shape.iter().product();
            if t.data.len() != n {
                return Err(format!("{name}: {} values for shape {shape:?}", t.data.len()));
            }
            if t.data.iter().any(|v| !v.is_finite()) {
                return Err(format!("{name}: not finite"));
            }
            index.insert(name.clone(), names.len());
            tensors.push(Tensor::<1>::from_data(TensorData::new(t.data.clone(), [n.max(1)]), device));
            names.push(name);
            shapes.push(shape);
        }
        Ok(Self { names, shapes, tensors, index })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn device(&self) -> Device {
        self.tensors[0].device()
    }

    /// Fresh gradient-tracking leaves holding the current values.
    pub fn leaves(&self) -> Self {
        Self {
            names: self.names.clone(),
            shapes: self.shapes.clone(),
            tensors: self.tensors.iter().map(|t| t.clone().autodiff().require_grad()).collect(),
            index: self.index.clone(),
        }
    }

    /// Downloads every parameter.
    pub async fn to_host(&self) -> Named {
        let mut out = Named::new();
        for i in 0..self.len() {
            out.insert(
                self.names[i].clone(),
                HostTensor { shape: self.shapes[i].clone(), data: read_f32(self.tensors[i].clone()).await },
            );
        }
        out
    }

    fn at(&self, name: &str) -> usize {
        *self.index.get(name).unwrap_or_else(|| panic!("no parameter {name}"))
    }

    /// Rank-1 view (vectors, and the scalar `shared.scale` as shape `[1]`).
    pub fn r1(&self, name: &str) -> Tensor<1> {
        self.tensors[self.at(name)].clone()
    }

    pub fn r2(&self, name: &str) -> Tensor<2> {
        let i = self.at(name);
        let s = &self.shapes[i];
        assert_eq!(s.len(), 2, "{name} is not a matrix");
        self.tensors[i].clone().reshape([s[0], s[1]])
    }

    pub fn r3(&self, name: &str) -> Tensor<3> {
        let i = self.at(name);
        let s = &self.shapes[i];
        assert_eq!(s.len(), 3, "{name} is not rank 3");
        self.tensors[i].clone().reshape([s[0], s[1], s[2]])
    }
}

/// Reads a float tensor back to the host as `f32` (asynchronously, as browsers require).
pub async fn read_f32<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    let data = t.into_data_async().await.expect("tensor readback failed");
    data.try_into_vec_as::<f32>().expect("float tensor data")
}

/// Reads an integer tensor back to the host.
pub async fn read_i64<const D: usize>(t: Tensor<D, Int>) -> Vec<i64> {
    let data = t.into_data_async().await.expect("tensor readback failed");
    data.try_into_vec_as::<i64>().expect("int tensor data")
}

fn ints<const D: usize>(values: &[usize], shape: [usize; D], device: &Device) -> Tensor<D, Int> {
    let v: Vec<i64> = values.iter().map(|&x| x as i64).collect();
    Tensor::<D, Int>::from_data(TensorData::new(v, shape), device)
}

fn floats<const D: usize>(values: Vec<f32>, shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::<D>::from_data(TensorData::new(values, shape), device)
}

/// `torch.nn.LayerNorm` over the last dimension (biased variance).
fn layer_norm<const D: usize>(x: Tensor<D>, w: Tensor<1>, b: Tensor<1>) -> Tensor<D> {
    let mut shape = [1usize; D];
    shape[D - 1] = x.dims()[D - 1];
    let mean = x.clone().mean_dim(D - 1);
    let xc = x - mean;
    let var = (xc.clone() * xc.clone()).mean_dim(D - 1);
    xc / (var + LN_EPS).sqrt() * w.reshape(shape) + b.reshape(shape)
}

/// `nn.Linear(bias=False)` with PyTorch's `[out, in]` weight on `[rows, in]`.
fn linear(x: Tensor<2>, w: Tensor<2>) -> Tensor<2> {
    x.matmul(w.transpose())
}

/// Exact (erf) GELU, written as PyTorch's CPU kernel writes it: `x * 0.5 * (1 + erf(x * 1/sqrt(2)))`.
fn gelu(x: Tensor<2>) -> Tensor<2> {
    (x.clone() * 0.5) * ((x * std::f32::consts::FRAC_1_SQRT_2).erf() + 1.0)
}

/// `edm_pre(sigma)` in float32: `(c_skip, c_out, c_in)`.
pub fn edm_pre(s: f32) -> (f32, f32, f32) {
    let v = s * s + SD * SD;
    let r = v.sqrt();
    (SD * SD / v, s * SD / r, 1.0 / r)
}

/// `edm_w(sigma)`: the EDM loss weight, clamped at 5.
pub fn edm_w(s: f32) -> f32 {
    let a = s * SD;
    ((s * s + SD * SD) / (a * a)).min(5.0)
}

/// One training batch: byte ids `[b, t1]` (row-major; ids `[:, :-1]` are the context, `[:, 1:]` the targets) and
/// one noise level per row.
#[derive(Clone, Debug)]
pub struct Batch {
    pub ids: Vec<usize>,
    pub b: usize,
    pub t1: usize,
    pub sigma: Vec<f32>,
}

impl Batch {
    pub fn validate(&self, cfg: &Cfg) -> Result<(), String> {
        if self.b == 0 || self.t1 < 2 {
            return Err(format!("batch: need b >= 1 and t+1 >= 2, got b={} t+1={}", self.b, self.t1));
        }
        if self.t1 - 1 > cfg.ctx {
            return Err(format!("batch: {} positions exceed ctx {}", self.t1 - 1, cfg.ctx));
        }
        if self.ids.len() != self.b * self.t1 {
            return Err(format!("batch: {} ids for [{}, {}]", self.ids.len(), self.b, self.t1));
        }
        if let Some(&bad) = self.ids.iter().find(|&&i| i >= cfg.vocab) {
            return Err(format!("batch: id {bad} outside vocab {}", cfg.vocab));
        }
        if self.sigma.len() != self.b || self.sigma.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
            return Err("batch: need one finite positive sigma per row".into());
        }
        Ok(())
    }
}

/// What `dblock_loss` returns, plus the diagnostics the parity fixture checks. `ce`, `aux`, `denoised` and
/// `logits` are detached.
pub struct Forward {
    pub loss: Tensor<1>,
    pub ce: Tensor<1>,
    pub aux: Tensor<1>,
    pub fractions: Vec<Vec<f32>>,
    pub denoised: Tensor<3>,
    pub logits: Tensor<3>,
}

/// `dblock_loss(shared, block, ..., sigma=batch.sigma)` with `torch.randn_like(e)` replaced by `eps` (`[b, t, d]`).
///
/// The device is the parameters' device; pass [`Params::leaves`] to differentiate.
pub async fn dblock_loss(p: &Params, cfg: &Cfg, batch: &Batch, eps: Tensor<3>) -> Forward {
    let device = p.device();
    let (b, t, d, v) = (batch.b, batch.t1 - 1, cfg.d, cfg.vocab);
    let mut ids = Vec::with_capacity(b * t);
    let mut tgt = Vec::with_capacity(b * t);
    for r in 0..b {
        let row = &batch.ids[r * batch.t1..(r + 1) * batch.t1];
        ids.extend_from_slice(&row[..t]);
        tgt.extend_from_slice(&row[1..]);
    }

    // Shared.table(): rows rescaled to RMS SD (also the tied output head).
    let w = p.r2("shared.emb.weight");
    let rms = (w.clone() * w.clone()).mean_dim(1).add_scalar(1e-6).sqrt();
    let table = w * (rms.recip() * SD);
    let embed = |ix: &[usize]| table.clone().select(0, ints(ix, [b * t], &device)).reshape([b, t, d]);

    // z = e(target) + sigma * eps
    let col = |vals: Vec<f32>| floats(vals, [b, 1, 1], &device);
    let pre: Vec<(f32, f32, f32)> = batch.sigma.iter().map(|&s| edm_pre(s)).collect();
    let z = embed(&tgt) + col(batch.sigma.clone()) * eps;

    // Block.forward(context(ids), z, sigma)
    let ctx = embed(&ids) + p.r2("shared.pos").slice([0..t, 0..d]).reshape([1, t, d]);
    let noise = floats(batch.sigma.iter().map(|&s| s.ln() / 4.0).collect(), [b, 1], &device);
    let sig = (linear(noise, p.r2("block.sig.weight")) + p.r1("block.sig.bias").reshape([1, d])).reshape([b, 1, d]);
    let c_in = col(pre.iter().map(|x| x.2).collect());
    let inp = linear((z.clone() * c_in).reshape([b * t, d]), p.r2("block.inp.weight")).reshape([b, t, d]);
    let mut h = ctx + inp + sig;
    let causal = causal_mask(t, &device);
    let mut aux = Tensor::<1>::zeros([1], &device);
    let mut fractions = Vec::with_capacity(cfg.layers_per_block);
    for l in 0..cfg.layers_per_block {
        let pre = format!("block.layers.{l}");
        h = attention(h, p, &pre, cfg.heads, causal.clone());
        let ln2 = layer_norm(h.clone(), p.r1(&format!("{pre}.ln2.weight")), p.r1(&format!("{pre}.ln2.bias")));
        let (m, a, fr) = moe(ln2, p, &format!("{pre}.moe"), cfg).await;
        h = h + m;
        aux = aux + a;
        fractions.push(fr);
    }
    let o = linear(
        layer_norm(h, p.r1("block.ln.weight"), p.r1("block.ln.bias")).reshape([b * t, d]),
        p.r2("block.out.weight"),
    )
    .reshape([b, t, d]);
    let denoised = z * col(pre.iter().map(|x| x.0).collect()) + o * col(pre.iter().map(|x| x.1).collect());

    // Shared.logits(D) and the weighted cross-entropy
    let k = (1.0 / (SD as f64 * (d as f64).sqrt())) as f32;
    let hd = layer_norm(denoised.clone(), p.r1("shared.ln.weight"), p.r1("shared.ln.bias")).reshape([b * t, d]);
    let logits = hd.matmul(table.transpose()) * (p.r1("shared.scale").reshape([1, 1]) * k);
    let ce = log_softmax(logits.clone(), 1).gather(1, ints(&tgt, [b * t, 1], &device)).reshape([b, t]).neg();
    let wv = floats(batch.sigma.iter().map(|&s| edm_w(s)).collect(), [b, 1], &device);
    let loss = (wv.clone() * ce.clone()).mean() / wv.mean() + aux.clone() * (cfg.aux as f32);
    Forward {
        loss,
        ce: ce.mean().detach(),
        aux: aux.detach(),
        fractions,
        denoised: denoised.detach(),
        logits: logits.reshape([b, t, v]).detach(),
    }
}

/// Additive causal mask `[1, 1, t, t]`: 0 on and below the diagonal, -inf above.
fn causal_mask(t: usize, device: &Device) -> Tensor<4> {
    let mut m = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            m[i * t + j] = f32::NEG_INFINITY;
        }
    }
    floats(m, [1, 1, t, t], device)
}

/// Causal multi-head self-attention, returning `x + proj(attention(ln1(x)))` (`Layer.forward`, first half).
fn attention(x: Tensor<3>, p: &Params, pre: &str, heads: usize, causal: Tensor<4>) -> Tensor<3> {
    let [b, t, d] = x.dims();
    let dh = d / heads;
    let h = layer_norm(x.clone(), p.r1(&format!("{pre}.ln1.weight")), p.r1(&format!("{pre}.ln1.bias")));
    let qkv = linear(h.reshape([b * t, d]), p.r2(&format!("{pre}.qkv.weight"))).reshape([b, t, 3 * d]);
    let head = |i: usize| qkv.clone().slice([0..b, 0..t, i * d..(i + 1) * d]).reshape([b, t, heads, dh]).swap_dims(1, 2);
    let (q, k, v) = (head(0), head(1), head(2));
    let scale = (1.0 / (dh as f64).sqrt()) as f32;
    let att = softmax(q.matmul(k.swap_dims(2, 3)) * scale + causal, 3);
    let a = att.matmul(v).swap_dims(1, 2).reshape([b * t, d]);
    x + linear(a, p.r2(&format!("{pre}.proj.weight"))).reshape([b, t, d])
}

/// Top-1 mixture of experts (`MoE.forward`): returns the output, the Switch balance loss and expert fractions.
///
/// Routing is decided on the host (the argmax is read back), so each expert runs only on its own tokens, as in
/// PyTorch's `index_select`/`index_add_`.
async fn moe(x: Tensor<3>, p: &Params, pre: &str, cfg: &Cfg) -> (Tensor<3>, Tensor<1>, Vec<f32>) {
    let [b, t, d] = x.dims();
    let (n, e_count, ffn) = (b * t, cfg.experts, cfg.ffn);
    let device = x.device();
    let flat = x.reshape([n, d]);
    let prob = softmax(linear(flat.clone(), p.r2(&format!("{pre}.router.weight"))), 1);
    let top_i: Vec<usize> = read_i64(prob.clone().argmax(1)).await.into_iter().map(|i| i as usize).collect();
    let top_p = prob.clone().gather(1, ints(&top_i, [n, 1], &device));
    let (w1, w2) = (p.r3(&format!("{pre}.w1")), p.r3(&format!("{pre}.w2")));
    let mut out = Tensor::<2>::zeros([n, d], &device);
    let mut counts = vec![0usize; e_count];
    for e in 0..e_count {
        let idx: Vec<usize> = (0..n).filter(|&r| top_i[r] == e).collect();
        counts[e] = idx.len();
        if idx.is_empty() {
            continue;
        }
        let it = ints(&idx, [idx.len()], &device);
        let w1e = w1.clone().slice([e..e + 1, 0..d, 0..ffn]).reshape([d, ffn]);
        let w2e = w2.clone().slice([e..e + 1, 0..ffn, 0..d]).reshape([ffn, d]);
        let hdn = gelu(flat.clone().select(0, it.clone()).matmul(w1e)).matmul(w2e);
        out = out.select_assign(0, it.clone(), hdn * top_p.clone().select(0, it), IndexingUpdateOp::Add);
    }
    let frac: Vec<f32> = counts.iter().map(|&c| c as f32 / n.max(1) as f32).collect();
    let aux = (floats(frac.clone(), [1, e_count], &device) * prob.mean_dim(0)).sum() * (e_count as f32);
    (out.reshape([b, t, d]), aux, frac)
}
