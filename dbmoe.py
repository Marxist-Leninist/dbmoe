#!/usr/bin/env python3
"""dbmoe.py -- a small DiffusionBlocks-MoE byte-level language model trained across many machines.

OpenTransformers / AGILLM-4 lineage (block sigma partition, EDM preconditioning and loss weight ported from
AGILLM-4 dblocks_train.py) and the recipe of Shing, Koyama and Akiba, "DiffusionBlocks" (ICLR 2026, arXiv:2506.14202).
One file; workers need only PyTorch (CPU, NVIDIA CUDA, AMD ROCm, Intel XPU or macOS MPS).

  coordinator  python dbmoe.py coord --dir RUN --host 127.0.0.1 --port 8790 --tokens tokens.json \
                   --hf "HuggingFaceFW/fineweb-edu:sample/10BT/*.parquet"
  worker       python dbmoe.py worker [--device cpu|cuda|rocm|xpu|mps|auto] [--threads 2] [--steps 40] [--batch 8] [--hours 1] [--no-serve]
                   (no token needed: the worker enrols itself; --url, --name and --token TOKEN|@FILE remain available)
  sample       python dbmoe.py sample --ckpt RUN/master.pt --prompt "The " [--n 200] [--mode chain|top]
  eval         python dbmoe.py eval --ckpt RUN/master.pt --heldout RUN/heldout.pt
  check        python dbmoe.py check-device --device rocm  # offline forward/backward and wire-format test

MODEL.  Bytes (vocab 256) -> one shared embedding table (rows renormalised to RMS SD=0.5; also the tied output head)
plus learned positions, then B independent blocks.  Each block = L pre-LN causal transformer layers whose FFN is a
top-1 mixture of experts (Switch-style balance loss), a noise-level embedding and an output projection.

DIFFUSIONBLOCKS OBJECTIVE.  What is diffused is the embedding of the NEXT byte.  Block i owns the noise range
[s_i, s_{i+1}] (equal-probability slices of a log-normal noise distribution, AGILLM-4 _block_sigmas).  Training step
for block i: sigma ~ the log-normal restricted to [s_i, s_{i+1}], widened by the paper's overlap (gamma 0.1), z = e(x[t+1]) +
sigma*eps; the block reads the CLEAN causal context
e(x[<=t]) + pos and c_in*z, outputs D = c_skip*z + c_out*F_i(.) (EDM preconditioning), and the loss is
w(sigma) * CE(head(D), x[t+1]) + 0.01 * balance.  A block never needs another block's activations or gradients,
so each machine trains a different block and ships back only that block plus a small delta of the shared
embedding / position / norm parameters.  (Unlike the EDM image defaults, the log-normal is re-centred on
sigma* = SD*sqrt(d/ln V), the noise level at which one byte's embedding stops being readable; with the image
defaults 5 of 6 blocks would sit where the target can simply be read off z.)

z contains the answer on purpose: that is what a denoiser is given (paper, Appendix B and E.4).  It means block CE at low
noise says little about language, so compute is shared out by how much each block still has to learn, and generation
quality is measured separately: evaluate_chain() and evaluate_generation().

GENERATION chains the blocks as a diffusion sampler for the next byte's embedding: z ~ N(0, s_B^2), then blocks
B-1 ... 0 each take one Euler step of the EDM probability-flow ODE across their own range (the paper's recipe: the total
compute of one byte is one pass through all layers), stopping early once a posterior has collapsed, with the x0 estimate
softmax(logits) @ table (CDCD score interpolation).  Already-known positions are kept at their true embedding plus
the current noise level, exactly as in training.  `--mode top` instead samples from the top block alone (one pass).

NETWORK.  HTTP + bearer token.  GET lease -> {lease, block, cfg, sigmas, shared, block_state, data, steps, batch, lr};
POST update <- {lease, block, d_block, d_shared, losses, tokens, seconds}.  Payloads are torch.save dicts of
tensors and plain values and are always loaded with weights_only=True (no pickle code execution).  Each update is
checked (keys, shapes, finite, |delta| <= 3.5*lr*steps, the Adam bound) before it is merged.  Workers without PyTorch
(worker-rs/, the Rust worker) opt into safetensors instead: GET lease?fmt=safetensors, and POST update with
Content-Type application/vnd.dbmoe.update+safetensors; the same fields and the same checks.  Merging is
asynchronous delta averaging: block += d_block / (leases that overlapped on that block).  Shared parameters define the
target every block denoises toward, so they move in rounds: deltas are banked per block and applied together every 12
updates, weighted by what each block still has to learn, so all blocks see one stable embedding between rounds.  Workers
keep Adam's per-parameter scale between leases of a block but not its momentum, which points where old weights wanted to go.
tokens.json is re-read when it changes.  GET status (JSON), GET code (this file), GET / (join
page), GET join.sh (installer), GET sample?prompt=..&n=..&mode=chain|top (text, generated on the coordinator).

OPEN JOIN.  Nobody has to ask anybody.  A worker started without --token does a few seconds of proof of work against
GET/POST join and receives its own credential (only its hash is stored; it does not expire).  What protects
the model is a test, not a person: before an open participant's update is merged, the coordinator measures its effect
on held-out text it never hands out (the leased block, plus one other block for the shared parameters), at the weight
it would merge with.  An update that makes the loss worse is refused, and repeat offenders are suspended automatically.

NETWORK INFERENCE.  GET sample?net=1 generates with the blocks running on the workers: one hop per block per byte.  Each
worker holds one block of a frozen snapshot (GET iblock) and long-polls POST ipoll while it trains.  The coordinator only
routes: z goes to block B-1's worker and comes back, goes to block B-2's, and so on, and block 0's worker returns the
byte's logits.  A slow or vanished worker's hop is rerouted; if no worker has a block the coordinator runs that hop itself
and the reply says so.  Results are spot-checked against the snapshot and workers that lie stop being used.  The reply is
a stream of JSON lines naming the (anonymous) worker behind every hop.
"""
import argparse, array, base64, collections, ctypes, hashlib, hmac, http.client as httpc, io, ipaddress, json, math, os, platform, random, re, secrets, select, socket, struct, sys, threading, time, urllib.error, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import torch
except ImportError:
    sys.exit('dbmoe needs PyTorch.  CPU build: python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu')
import torch.nn as nn
import torch.nn.functional as F

SD = 0.5
VERSION = 'dbmoe-v4-20261105'
DEFAULT_URL = 'https://codex.opentransformers.online/dbmoe'
STEPS_PER_BLOCK = 1                                     # the paper's inference: each block once, so one byte costs one full-depth pass
EXIT_CONF = 0.98                                        # stop the chain once a block's posterior has collapsed onto one byte
LN2 = math.log(2)


# ----------------------------------------------------------------------------------------------------------------- model
def _cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _ppf(p):
    return float(torch.erfinv(torch.tensor(2 * p - 1.0, dtype=torch.float64)) * math.sqrt(2))


def block_sigmas(B, smin=0.002, smax=80.0, pm=-1.2, ps=1.2):
    """AGILLM-4 dblocks_train._block_sigmas: equal-probability partition of a log-normal noise distribution."""
    a, b = _cdf((math.log(smin) - pm) / ps), _cdf((math.log(smax) - pm) / ps)
    s = [float(math.exp(pm + ps * _ppf(a + (b - a) * (i / B)))) for i in range(B + 1)]
    s[0], s[-1] = float(smin), float(smax)
    return s


def edm_pre(s):
    s = s[:, None, None]
    return SD ** 2 / (s ** 2 + SD ** 2), s * SD / (s ** 2 + SD ** 2) ** 0.5, 1 / (s ** 2 + SD ** 2) ** 0.5


def edm_w(s, wmax=5.0):
    return ((s ** 2 + SD ** 2) / (s * SD) ** 2).clamp(max=wmax)


DEFAULT_CFG = dict(vocab=256, d=256, heads=4, layers_per_block=2, blocks=6, experts=4, ffn=512, ctx=256, aux=0.01)
CFG_1P6B = dict(vocab=256, d=1024, heads=16, layers_per_block=2, blocks=24, experts=8, ffn=1728, ctx=256, aux=0.01)


def make_cfg(over=None):
    cfg = dict(DEFAULT_CFG)
    cfg.update(over or {})
    s_star = SD * math.sqrt(cfg['d'] / math.log(cfg['vocab']))
    cfg.setdefault('sig_pm', round(math.log(1.5 * s_star), 4))
    cfg.setdefault('sig_ps', 0.9)
    cfg.setdefault('sig_min', round(s_star / 10, 4))
    cfg.setdefault('sig_max', 80.0)
    return cfg


def cfg_sigmas(cfg):
    return block_sigmas(cfg['blocks'], cfg['sig_min'], cfg['sig_max'], cfg['sig_pm'], cfg['sig_ps'])


class MoE(nn.Module):
    def __init__(self, d, ffn, E):
        super().__init__()
        self.router = nn.Linear(d, E, bias=False)
        self.w1 = nn.Parameter(torch.randn(E, d, ffn) * d ** -0.5)
        self.w2 = nn.Parameter(torch.randn(E, ffn, d) * ffn ** -0.5 * 0.5)
        self.E = E

    def forward(self, x):
        Bsz, T, d = x.shape
        flat = x.reshape(-1, d)
        prob = self.router(flat).float().softmax(-1)
        top_p, top_i = prob.max(-1)
        out = torch.zeros_like(flat)
        for e in range(self.E):
            idx = (top_i == e).nonzero(as_tuple=True)[0]
            if idx.numel():
                h = F.gelu(flat.index_select(0, idx) @ self.w1[e]) @ self.w2[e]
                out.index_add_(0, idx, h * top_p.index_select(0, idx)[:, None].to(h.dtype))
        # bincount is unavailable on some MPS releases; the selected expert ids are unchanged.
        counts = portable_expert_counts(top_i, self.E) if top_i.device.type == 'mps' else torch.bincount(top_i, minlength=self.E)
        frac = counts.float() / max(1, top_i.numel())
        aux = self.E * (frac.detach() * prob.mean(0)).sum()
        return out.reshape(Bsz, T, d), aux, frac


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['d']
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.moe, self.heads = MoE(d, cfg['ffn'], cfg['experts']), cfg['heads']

    def forward(self, x):
        Bsz, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, -1)
        sh = lambda t: t.view(Bsz, T, self.heads, d // self.heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(Bsz, T, d))
        m, aux, frac = self.moe(self.ln2(x))
        return x + m, aux, frac


class Block(nn.Module):
    """One DiffusionBlocks unit: denoises the next-byte embedding over one sigma range, given the clean causal context."""
    def __init__(self, cfg):
        super().__init__()
        d = cfg['d']
        self.inp = nn.Linear(d, d, bias=False)          # mixes c_in*z into the context stream
        self.sig = nn.Linear(1, d)                      # noise-level embedding, c_noise = ln(sigma)/4
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg['layers_per_block']))
        self.ln, self.out = nn.LayerNorm(d), nn.Linear(d, d, bias=False)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, ctx, z, sigma):
        c_skip, c_out, c_in = edm_pre(sigma)
        h = ctx + self.inp(c_in * z) + self.sig((sigma.log() / 4)[:, None, None])
        aux, fracs = 0.0, []
        for L in self.layers:
            h, a, fr = L(h)
            aux = aux + a
            fracs.append(fr)
        return c_skip * z + c_out * self.out(self.ln(h)), aux, fracs


class Shared(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(cfg['vocab'], cfg['d'])
        nn.init.normal_(self.emb.weight, std=SD)
        self.pos = nn.Parameter(torch.randn(cfg['ctx'], cfg['d']) * 0.02)
        self.ln = nn.LayerNorm(cfg['d'])
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.k = 1.0 / (SD * math.sqrt(cfg['d']))

    def table(self):                                    # rows at RMS SD: the "data" the blocks denoise has std SD
        w = self.emb.weight
        return w * (SD / w.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt())

    def embed(self, ids):
        return F.embedding(ids, self.table())

    def context(self, ids):
        return self.embed(ids) + self.pos[: ids.size(1)]

    def logits(self, D):
        return self.ln(D) @ self.table().t() * (self.scale * self.k)


def dblock_loss(shared, block, lo, hi, x, aux_w, sigma=None, generator=None):
    """x: [batch, T+1] byte ids.  DiffusionBlocks loss for one block at sigma ~ logU[lo, hi] (or a given sigma)."""
    ids, tgt = x[:, :-1], x[:, 1:]
    if sigma is None:
        sigma = torch.exp(torch.empty(x.size(0), device=x.device).uniform_(math.log(lo), math.log(hi)))
    e = shared.embed(tgt)
    noise = (torch.randn_like(e) if generator is None else
             torch.randn(e.shape, dtype=e.dtype, device=e.device, generator=generator))
    z = e + sigma[:, None, None] * noise
    D, aux, fracs = block(shared.context(ids), z, sigma)
    ce = F.cross_entropy(shared.logits(D).flatten(0, 1).float(), tgt.flatten(), reduction='none').view_as(tgt)
    w = edm_w(sigma)[:, None]
    loss = (w * ce).mean() / w.mean() + aux_w * aux
    return loss, ce.mean().detach(), fracs


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.shared = Shared(cfg)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg['blocks']))
        self.sigmas = cfg_sigmas(cfg)


def block_grid(sigmas, i, steps_per_block=STEPS_PER_BLOCK):
    lo, hi = sigmas[i], sigmas[i + 1]
    return [hi * (lo / hi) ** (k / steps_per_block) for k in range(steps_per_block + 1)]


def block_range(cfg, sigmas, b):
    """Block b's training range: its slice of the noise schedule, widened by the paper's overlap so that generation
    does not ask a block to work at the very edge of what it was trained on."""
    lo, hi = sigmas[b], sigmas[b + 1]
    a = (hi / lo) ** cfg.get('overlap', 0.1)
    return max(lo / a, sigmas[0]), min(hi * a, sigmas[-1])


def sample_sigma(cfg, lo, hi, n, device='cpu'):
    """sigma ~ the log-normal noise distribution restricted to [lo, hi]: each block sees its part of one global distribution."""
    pm, ps = cfg['sig_pm'], cfg['sig_ps']
    a, b = _cdf((math.log(lo) - pm) / ps), _cdf((math.log(hi) - pm) / ps)
    q = torch.empty(n).uniform_(a, b).clamp(1e-6, 1 - 1e-6)     # on the CPU: erfinv is not on every GPU backend
    return torch.exp(pm + ps * math.sqrt(2) * torch.erfinv(2 * q - 1)).to(device)


@torch.no_grad()
def prepare_segment(shared, raw_ids, ek_seed):
    """Per-byte immutable context, reused by all local blocks of one frozen snapshot."""
    ids = torch.tensor([list(raw_ids)], dtype=torch.long)
    tab = shared.table()
    ctx = F.embedding(ids, tab) + shared.pos[:ids.size(1)]
    known = F.embedding(ids[:, 1:], tab)
    ek = torch.randn(known.shape, generator=torch.Generator().manual_seed(ek_seed))
    return ctx, tab, known, ek


@torch.no_grad()
def run_segment(shared, block, raw_ids, ek_seed, z, grid, final, exit_conf=0.0, prepared=None):
    """One block's part of one byte: Euler steps of the probability-flow ODE over its noise range.  z: [1, 1, d] at
    grid[0].  Returns z at grid[-1], and the byte's logits [vocab] if this is the last block or the posterior has collapsed."""
    ctx, tab, known, ek = prepare_segment(shared, raw_ids, ek_seed) if prepared is None else prepared
    lg = None
    for a, b in zip(grid[:-1], grid[1:]):
        D, _, _ = block(ctx, torch.cat([known + a * ek, z], 1), torch.full((1,), float(a)))
        lg = shared.ln(D[:, -1:]) @ tab.t() * (shared.scale * shared.k)
        p = lg.softmax(-1)
        if exit_conf and float(p.max()) >= exit_conf:
            return z, lg[0, -1]
        z = z + (b - a) * (z - p @ tab) / a
    return z, (lg[0, -1] if final else None)


@torch.no_grad()
def next_byte_logits(model, ids, mode='chain', steps_per_block=STEPS_PER_BLOCK, gen=None, exit_conf=EXIT_CONF):
    """ids: LongTensor [1, T] -> logits [vocab] for byte T+1."""
    cfg, sig = model.cfg, model.sigmas
    raw, seed = ids[0].tolist(), int(torch.randint(2 ** 31 - 1, (1,), generator=gen))
    z = torch.randn(1, 1, cfg['d'], generator=gen) * sig[-1]
    if mode == 'top':                                   # one pass at pure noise: the top block's own prediction
        return run_segment(model.shared, model.blocks[-1], raw, seed, z, [sig[-1], sig[-2]], True)[1]
    for i in reversed(range(cfg['blocks'])):
        z, lg = run_segment(model.shared, model.blocks[i], raw, seed, z, block_grid(sig, i, steps_per_block), i == 0, exit_conf)
        if lg is not None:
            return lg


def pick_byte(lg, temp, top_k, gen):
    """temp 0 is greedy, which is what the paper uses after the chain: the chain's starting noise already supplies the randomness."""
    if temp <= 0:
        return int(lg.argmax())
    lg = lg / temp
    if top_k:
        v, _ = lg.topk(min(top_k, lg.numel()))
        lg = lg.masked_fill(lg < v[-1], -float('inf'))
    return int(torch.multinomial(lg.softmax(-1), 1, generator=gen))


@torch.no_grad()
def generate(model, prompt, n=200, temp=0.8, top_k=40, seed=0, mode='chain', ctx_len=None):
    gen = torch.Generator().manual_seed(seed)
    ids = encode(prompt) or [10]
    L = min(ctx_len or model.cfg['ctx'], model.cfg['ctx'])
    for _ in range(n):
        ids.append(pick_byte(next_byte_logits(model, torch.tensor([ids[-L:]]), mode=mode, gen=gen), temp, top_k, gen))
    return decode(ids)


@torch.no_grad()
def evaluate(model, x):
    """x: [n, T+1] uint8/long held-out windows.  Per-block CE at each block's mid sigma, and the top block's CE at
    sigma_max (the next-byte prediction from context alone: a proper conditional distribution, reported as bits/byte)."""
    x = x.long()
    sig, B = model.sigmas, model.cfg['blocks']
    # Private RNG: concurrent gate/model construction must not perturb evaluation noise.
    gen = torch.Generator(device=x.device).manual_seed(0)
    per = []
    for i in range(B):
        s = torch.full((x.size(0),), math.sqrt(sig[i] * sig[i + 1]))
        per.append(round(float(dblock_loss(model.shared, model.blocks[i], 0, 0, x, 0, sigma=s, generator=gen)[1]), 4))
    _, ce, fr = dblock_loss(model.shared, model.blocks[-1], 0, 0, x, 0, sigma=torch.full((x.size(0),), sig[-1]), generator=gen)
    return dict(ar_ce=round(float(ce), 4), ar_bits_per_byte=round(float(ce) / LN2, 4), per_block_ce=per,
                top_block_expert_frac=[[round(float(v), 3) for v in f] for f in fr])


@torch.no_grad()
def evaluate_chain(model, x, T=64, samples=4, steps_per_block=STEPS_PER_BLOCK, seed=0):
    """How well does the chain predict the next byte, at each depth?  The average softmax over chain samples entering a
    block estimates the predictive distribution at that depth (in the ideal model every depth gives the same one).  Top
    entry is the top block alone.  Near the bottom the softmaxes are almost one-hot, so a few samples read pessimistically."""
    sh, cfg, sig, B = model.shared, model.cfg, model.sigmas, model.cfg['blocks']
    seg = x.long().unfold(1, T + 1, T + 1).reshape(-1, T + 1)
    ids, tgt = seg[:, :-1], seg[:, -1]
    n, idx = ids.size(0), torch.arange(ids.size(0))
    gen = torch.Generator().manual_seed(seed)
    ctx, tab, known = sh.context(ids), sh.table(), sh.embed(ids[:, 1:])
    mix = [0.0] * B
    for _ in range(samples):
        ek = torch.randn(known.shape, generator=gen)
        z = torch.randn(n, 1, cfg['d'], generator=gen) * sig[-1]
        for i in reversed(range(B)):
            grid = block_grid(sig, i, steps_per_block)
            for j, (a, b) in enumerate(zip(grid[:-1], grid[1:])):
                D, _, _ = model.blocks[i](ctx, torch.cat([known + a * ek, z], 1), torch.full((n,), float(a)))
                p = sh.logits(D[:, -1:]).softmax(-1)
                if j == 0:
                    mix[i] = mix[i] + p[:, 0]
                z = z + (b - a) * (z - p @ tab) / a
    return dict(chain_bits_by_block=[round(float(-(m[idx, tgt] / samples).clamp_min(1e-6).log().mean()) / LN2, 4) for m in mix])


class Teacher:
    """A character n-gram model of real text, used the way the paper uses a teacher model (generative perplexity): how
    likely does the statistics of real text find what the model wrote?  Lower is better; real text scores real_bits."""
    PRIMES = [(1000003 ** i) % (2 ** 61 - 1) for i in range(8)]

    def __init__(self, x, bits=22, order=5):
        x = x.long().reshape(-1)
        self.order, self.size, self.total = order, 1 << bits, float(x.numel())
        self.tables = [torch.bincount(self._hash(x, k), minlength=self.size).float() for k in range(1, order + 1)]

    def _hash(self, x, k):
        h = ((x.unfold(0, k, 1) + 1) * torch.tensor(self.PRIMES[:k])).sum(-1)
        return ((h * 2654435761) >> 11) & (self.size - 1)

    def bits(self, x):
        """Average bits per byte of the byte sequence x [n], from the order-th byte on."""
        x = x.long().reshape(-1)
        m = x.numel() - self.order + 1
        p = 0.01 / 256 + 0.99 * self.tables[0][self._hash(x, 1)[-m:]] / self.total
        for k in range(2, self.order + 1):
            num = self.tables[k - 1][self._hash(x, k)[-m:]]
            den = self.tables[k - 2][self._hash(x[:-1], k - 1)[-m:]]
            p = 0.3 * p + 0.7 * (num / den.clamp_min(1)).clamp_max(1.0)
        return float(-p.clamp_min(1e-9).log().mean()) / LN2


@torch.no_grad()
def chain_batch(model, ids, gen, steps_per_block=STEPS_PER_BLOCK):
    """One chain per row: ids [n, T] -> logits [n, vocab] for the next byte."""
    sh, cfg, sig, n = model.shared, model.cfg, model.sigmas, ids.size(0)
    ctx, tab, known = sh.context(ids), sh.table(), sh.embed(ids[:, 1:])
    ek = torch.randn(known.shape, generator=gen)
    z, lg = torch.randn(n, 1, cfg['d'], generator=gen) * sig[-1], None
    for i in reversed(range(cfg['blocks'])):
        grid = block_grid(sig, i, steps_per_block)
        for a, b in zip(grid[:-1], grid[1:]):
            D, _, _ = model.blocks[i](ctx, torch.cat([known + a * ek, z], 1), torch.full((n,), float(a)))
            lg = sh.logits(D[:, -1:])
            z = z + (b - a) * (z - lg.softmax(-1) @ tab) / a
    return lg[:, 0]


@torch.no_grad()
def evaluate_generation(model, x, teacher, n=48, prompt_len=32, ctx_len=128, seed=0, modes=None):
    """What actually comes out, scored by the teacher: the end-to-end test that block losses cannot give."""
    gen = torch.Generator().manual_seed(seed)
    x = x.long()
    out = dict(real_bits=round(sum(teacher.bits(r) for r in x[:, 1:]) / x.size(0), 4))
    for name, top, temp in modes or (('chain', False, 0.0), ('top', True, 0.75)):
        ids = x[:, :prompt_len].clone()
        for _ in range(n):
            w = ids[:, -ctx_len:]
            if top:
                sh, sig = model.shared, model.sigmas
                known = sh.embed(w[:, 1:])
                z = torch.randn(w.size(0), 1, model.cfg['d'], generator=gen) * sig[-1]
                D, _, _ = model.blocks[-1](sh.context(w), torch.cat([known + sig[-1] * torch.randn(known.shape, generator=gen), z], 1), torch.full((w.size(0),), float(sig[-1])))
                lg = sh.logits(D[:, -1:])[:, 0]
            else:
                lg = chain_batch(model, w, gen)
            if temp <= 0:
                nxt = lg.argmax(-1, keepdim=True)
            else:
                lg = lg / temp
                v, _ = lg.topk(40, -1)
                nxt = torch.multinomial(lg.masked_fill(lg < v[:, -1:], -float('inf')).softmax(-1), 1, generator=gen)
            ids = torch.cat([ids, nxt], 1)
        out[f'{name}_gen_bits'] = round(sum(teacher.bits(r[prompt_len - teacher.order + 1:]) for r in ids) / ids.size(0), 4)
        out[f'{name}_sample'] = decode(ids[0, prompt_len:].tolist())[:n]
    return out


# ------------------------------------------------------------------------------------------------------------- helpers
def to_bytes(obj):
    b = io.BytesIO()
    torch.save(obj, b)
    return b.getvalue()


def from_bytes(b):
    return torch.load(io.BytesIO(b), map_location='cpu', weights_only=True)


def load_update_pickle(body, max_bytes, max_entries=4096):
    """Load only bounded, uncompressed torch.save ZIPs from untrusted updates.

    Parse the central directory ourselves before torch.load: ZipFile eagerly allocates one
    ZipInfo per entry, and legacy pickle storage declarations can allocate beyond the input
    length. The directory, records, declared storage and entry count are all bounded first.
    Trusted coordinator leases still use from_bytes(), not this untrusted-update gate.
    """
    if len(body) > max_bytes:
        raise ValueError('update body exceeds its byte budget')
    if body[:4] != b'PK\x03\x04':
        raise ValueError('update requires a torch.save ZIP archive')
    eocd = body.rfind(b'PK\x05\x06', max(0, len(body) - (22 + 65535)))
    if eocd < 0 or eocd + 22 > len(body):
        raise ValueError('update archive has no valid end record')
    try:
        _, disk, cd_disk, n_disk, n_total, cd_size, cd_off, comment_len = \
            struct.unpack_from('<4s4H2LH', body, eocd)
    except struct.error as e:
        raise ValueError('update archive has a truncated end record') from e
    if eocd + 22 + comment_len != len(body):
        raise ValueError('update archive has trailing or inconsistent data')
    if (disk or cd_disk or n_disk != n_total or not n_total or n_total == 0xffff or
            cd_size == 0xffffffff or cd_off == 0xffffffff):
        raise ValueError('multi-disk and ZIP64 update archives are unsupported')
    if n_total > max_entries:
        raise ValueError(f'update archive has too many entries ({n_total} > {max_entries})')
    if cd_size > (4 << 20):
        raise ValueError('update archive directory exceeds its bounded budget')
    cd_end = cd_off + cd_size
    if cd_off > eocd or cd_end > eocd:
        raise ValueError('update archive has an inconsistent central directory')
    if cd_end != eocd:
        # PyTorch writes a 56-byte ZIP64 end record plus its 20-byte locator even
        # for small ordinary updates. Permit exactly that trailer, nothing else.
        if (eocd - cd_end != 76 or body[cd_end:cd_end + 4] != b'PK' or
                body[cd_end + 56:cd_end + 60] != b'PK' or
                struct.unpack_from('<Q', body, cd_end + 4)[0] != 44):
            raise ValueError('update archive has an inconsistent central directory')

    entries = []
    total = 0
    pos = cd_off
    end = cd_off + cd_size
    for _ in range(n_total):
        if pos + 46 > end or body[pos:pos + 4] != b'PK\x01\x02':
            raise ValueError('update archive has a truncated central directory')
        fields = struct.unpack_from('<4s6H3L5H2L', body, pos)
        flags, method = fields[3], fields[4]
        compressed, uncompressed = fields[8], fields[9]
        name_len, extra_len, entry_comment_len = fields[10:13]
        disk_start, local_off = fields[13], fields[16]
        next_pos = pos + 46 + name_len + extra_len + entry_comment_len
        if next_pos > end:
            raise ValueError('update archive has a truncated directory entry')
        if (flags & 1) or method != 0 or compressed != uncompressed:
            raise ValueError('compressed or encrypted update archives are unsupported')
        if (compressed == 0xffffffff or uncompressed == 0xffffffff or
                local_off == 0xffffffff or disk_start == 0xffff):
            raise ValueError('ZIP64 update records are unsupported')
        if local_off + 30 > cd_off or body[local_off:local_off + 4] != b'PK\x03\x04':
            raise ValueError('update archive has an invalid local record')
        try:
            name = body[pos + 46:pos + 46 + name_len].decode('ascii')
        except UnicodeDecodeError as e:
            raise ValueError('update archive has a non-ASCII record name') from e
        entries.append((name, uncompressed))
        total += uncompressed
        if total > min(max_bytes, len(body)):
            raise ValueError('update archive exceeds its bounded storage budget')
        pos = next_pos
    if pos != end or len(entries) != n_total:
        raise ValueError('update archive entry count does not match its directory')

    names = [name for name, _ in entries]
    if not names or len(set(names)) != len(names) or '/' not in names[0]:
        raise ValueError('invalid or duplicate update records')
    root = names[0].split('/', 1)[0] + '/'
    allowed = {'data.pkl', 'byteorder', 'version', '.format_version',
               '.storage_alignment', '.data/serialization_id'}
    metadata_bytes = 0
    for name, size in entries:
        if not name.startswith(root):
            raise ValueError('inconsistent update archive root')
        short = name[len(root):]
        if not re.fullmatch(r'data/[0-9]+', short):
            if short not in allowed:
                raise ValueError('unsupported update archive record')
            metadata_bytes += size
    if root + 'data.pkl' not in names or metadata_bytes > (1 << 20):
        raise ValueError('update metadata exceeds its bounded budget or is missing')
    return torch.load(io.BytesIO(body), map_location='cpu', weights_only=True)


# Opt-in safetensors wire format, for workers without PyTorch (worker-rs).  A lease or an update is one safetensors file:
# its tensors in the tensor section, every other field as JSON in one metadata entry.  No pickle and no numpy.
ST_DTYPES = {'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16, 'I64': torch.int64,
             'I32': torch.int32, 'I16': torch.int16, 'I8': torch.int8, 'U8': torch.uint8}
ST_NAMES = {v: k for k, v in ST_DTYPES.items()}
ST_LEASE_TYPE = 'application/vnd.dbmoe.lease+safetensors'
ST_UPDATE_TYPE = 'application/vnd.dbmoe.update+safetensors'


def st_dumps(tensors, metadata=None):
    """safetensors bytes: an 8-byte little-endian header length, the JSON header padded to 8 bytes, then each tensor's
    raw little-endian bytes in name order."""
    if sys.byteorder != 'little':
        raise RuntimeError('the safetensors wire format needs a little-endian machine')
    header, chunks, off = {}, [], 0
    if metadata:
        header['__metadata__'] = {str(k): str(v) for k, v in metadata.items()}
    for name in sorted(tensors):
        t = tensors[name].detach().to('cpu').contiguous()
        if name == '__metadata__' or t.dtype not in ST_NAMES:
            raise ValueError(f'safetensors: cannot store {name!r} ({t.dtype})')
        n = t.numel() * t.element_size()
        chunks.append(ctypes.string_at(t.data_ptr(), n) if n else b'')
        header[name] = dict(dtype=ST_NAMES[t.dtype], shape=list(t.shape), data_offsets=[off, off + n])
        off += n
    h = json.dumps(header, separators=(',', ':')).encode()
    h += b' ' * (-len(h) % 8)
    return struct.pack('<Q', len(h)) + h + b''.join(chunks)


def st_loads(buf, max_header=1 << 20, max_tensors=4096):
    """Parse safetensors bytes strictly: known dtypes, byte sizes that match the shapes, offsets that tile the data
    section exactly, no duplicate names, at most max_tensors tensors, string metadata.  The whole layout is checked
    before any tensor is built, so the tensors never take more memory than the data section itself (an update has
    a few dozen tensors and a header of a few KB).  Raises ValueError otherwise."""
    if sys.byteorder != 'little':
        raise RuntimeError('the safetensors wire format needs a little-endian machine')
    buf = memoryview(buf)
    if len(buf) < 8:
        raise ValueError('safetensors: too short')
    (n,) = struct.unpack('<Q', buf[:8])
    if n > min(max_header, len(buf) - 8):
        raise ValueError('safetensors: bad header length')

    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError(f'safetensors: duplicate key {k!r}')
            d[k] = v
        return d
    try:
        header = json.loads(bytes(buf[8:8 + n]).decode('utf-8'), object_pairs_hook=unique)
    except UnicodeDecodeError as e:
        raise ValueError(f'safetensors: header is not UTF-8: {e}') from None
    if not isinstance(header, dict):
        raise ValueError('safetensors: header is not an object')
    meta = header.pop('__metadata__', None) or {}
    if not isinstance(meta, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()):
        raise ValueError('safetensors: metadata must map strings to strings')
    if len(header) > max_tensors:
        raise ValueError(f'safetensors: more than {max_tensors} tensors')
    data, entries = buf[8 + n:], []
    for name, e in header.items():                      # first pass: validate only, copy nothing
        if not isinstance(e, dict) or set(e) != {'dtype', 'shape', 'data_offsets'} or e['dtype'] not in ST_DTYPES:
            raise ValueError(f'safetensors: bad entry for {name!r}')
        shape, offs = e['shape'], e['data_offsets']
        if not isinstance(shape, list) or len(shape) > 8 or not all(type(s) is int and s >= 0 for s in shape):
            raise ValueError(f'safetensors: bad shape for {name!r}')
        if (not isinstance(offs, list) or len(offs) != 2 or not all(type(o) is int for o in offs)
                or not 0 <= offs[0] <= offs[1] <= len(data)):
            raise ValueError(f'safetensors: bad data_offsets for {name!r}')
        dt = ST_DTYPES[e['dtype']]
        if math.prod(shape) * torch.empty((), dtype=dt).element_size() != offs[1] - offs[0]:
            raise ValueError(f'safetensors: the size of {name!r} does not match its shape')
        entries.append((offs[0], offs[1], name, dt, shape))
    pos = 0
    for a, b, *_ in sorted(entries, key=lambda x: (x[0], x[1])):
        if a != pos:
            raise ValueError('safetensors: tensor data has gaps or overlaps')
        pos = b
    if pos != len(data):
        raise ValueError('safetensors: trailing bytes after the last tensor')
    out = {}                                            # second pass: the spans tile the data, so this copies it once
    for a, b, name, dt, shape in entries:
        out[name] = torch.frombuffer(bytearray(data[a:b]), dtype=dt).reshape(shape) if b > a else torch.empty(shape, dtype=dt)
    return out, meta


def lease_to_safetensors(L):
    """A lease as safetensors: tensors shared.*, block.* and data; every other field as JSON in 'dbmoe.lease'."""
    tensors = {f'shared.{k}': v for k, v in L['shared'].items()}
    tensors.update({f'block.{k}': v for k, v in L['block_state'].items()})
    tensors['data'] = L['data']
    meta = {k: v for k, v in L.items() if k not in ('shared', 'block_state', 'data')}
    return st_dumps(tensors, {'dbmoe.lease': json.dumps(meta, allow_nan=False)})


def update_to_safetensors(up):
    """Encode worker deltas without pickle; other fields remain JSON metadata."""
    tensors = {f'd_shared.{k}': v for k, v in up['d_shared'].items()}
    tensors.update({f'd_block.{k}': v for k, v in up['d_block'].items()})
    meta = {k: v for k, v in up.items() if k not in ('d_shared', 'd_block')}
    return st_dumps(tensors, {'dbmoe.update': json.dumps(meta, allow_nan=False)})


def update_from_safetensors(body):
    """The update dict from a safetensors upload: d_shared.* and d_block.* tensors plus the fields in 'dbmoe.update'.
    It then goes through exactly the checks a torch.save upload does."""
    tensors, meta = st_loads(body)
    up = json.loads(meta.get('dbmoe.update', 'null'))
    if not isinstance(up, dict):
        raise ValueError("safetensors update: no 'dbmoe.update' metadata")
    up = {k: v for k, v in up.items() if k not in ('d_shared', 'd_block')}
    up['d_shared'], up['d_block'] = {}, {}
    for name, t in tensors.items():
        group, _, key = name.partition('.')
        if group not in ('d_shared', 'd_block') or not key:
            raise ValueError(f'safetensors update: unexpected tensor {name!r}')
        up[group][key] = t
    return up


def encode(text):
    return list(text.encode('utf-8', errors='ignore'))


def decode(ids):
    return bytes(int(t) for t in ids).decode('utf-8', errors='replace')


def load_ckpt(path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    m = Model(ck['cfg'])
    m.load_state_dict(ck['state'])
    return m.eval(), ck


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# --------------------------------------------------------------------------------------------------------------- data
def choose_training_block(weights, accepted, inflight, job_tokens, last=None):
    """Weighted byte deficit; callers reserve the returned block under their lease lock.

    Completion history is bounded; live reservations prevent concurrent dispatches
    oversubscribing the same deficit. Stickiness can only spend 5% of one job.
    This balances exposure to the existing noise curriculum, not raw update counts.
    """
    total_weight = sum(weights)
    total = sum(accepted) + sum(inflight) + job_tokens
    deficit = [total * w / total_weight - accepted[i] - inflight[i]
               for i, w in enumerate(weights)]
    best = max(range(len(weights)), key=lambda i: (deficit[i], -i))
    if last in range(len(weights)) and deficit[last] >= deficit[best] - job_tokens * 0.05:
        return last
    return best


class LeaseCapacityError(RuntimeError):
    pass


class TextSource:
    """Real text as a byte stream: a Hugging Face dataset (streamed, never cached to disk) or a local text file.
    A background thread keeps a buffer of up to buf_mb MB filled; windows() cuts [n, T+1] blocks off its front."""
    def __init__(self, hf=None, text_file=None, field='text', skip_docs=0, buf_mb=16):
        self.hf, self.text_file, self.field = hf, text_file, field
        self.docs, self.bytes, self.err, self.epoch = int(skip_docs), 0, None, 0
        self.buf, self.cap = bytearray(), buf_mb << 20
        self.cv = threading.Condition()
        threading.Thread(target=self._fill, daemon=True).start()

    def _iter(self):
        skip = self.docs
        if self.text_file:
            CH = 1 << 16
            size = os.path.getsize(self.text_file)
            with open(self.text_file, 'rb') as f:
                f.seek((skip * CH) % max(1, size - CH))
                while True:
                    b = f.read(CH)
                    if not b:
                        f.seek(0)
                        continue
                    yield b
        name, _, sub = self.hf.partition(':')
        if '*' in sub or sub.endswith('.parquet'):
            # REPO:GLOB -- read the dataset's parquet files directly, one 1000-row group at a time (HTTP range reads,
            # a few MB of RAM, nothing on disk); resuming skips whole row groups from the footer without downloading them.
            import pyarrow.parquet as pq
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem()
            files = sorted(fs.glob(f'datasets/{name}/{sub}'))
            if not files:
                raise RuntimeError(f'no files match datasets/{name}/{sub}')
            for f in files:
                with fs.open(f, 'rb', block_size=8 << 20) as fh:
                    pf = pq.ParquetFile(fh)
                    for rg in range(pf.num_row_groups):
                        n = pf.metadata.row_group(rg).num_rows
                        if skip >= n:
                            skip -= n
                            continue
                        rows = pf.read_row_group(rg, columns=[self.field]).column(self.field).to_pylist()
                        for t in rows[skip:]:
                            yield ((t or '').strip() + '\n\n').encode('utf-8', errors='ignore') if (t or '').strip() else b''
                        skip = 0
        else:                                           # REPO[:CONFIG] through the datasets library (streaming)
            from datasets import load_dataset
            ds = load_dataset(name, sub or None, split='train', streaming=True)
            if skip:
                ds = ds.skip(skip)
            for row in ds:
                t = (row.get(self.field) or '').strip()
                yield (t + '\n\n').encode('utf-8', errors='ignore') if t else b''
        self.epoch += 1

    def _fill(self):
        while True:
            try:
                for chunk in self._iter():
                    with self.cv:
                        self.cv.wait_for(lambda: len(self.buf) < self.cap)
                        self.buf += chunk
                        self.docs += 1
                        self.bytes += len(chunk)
                        self.err = None
                        self.cv.notify_all()
                self.docs = 0                                  # dataset exhausted: next epoch from the start
            except Exception as e:                             # HF network blips: retry, resuming at self.docs
                self.err = f'{type(e).__name__}: {e}'[:300]
                print(f'[data] {self.err}; retry in 30 s', flush=True)
                time.sleep(30)

    def windows(self, n, T, timeout=600):
        need = n * (T + 1)
        with self.cv:
            if not self.cv.wait_for(lambda: len(self.buf) >= need, timeout):
                raise RuntimeError(f'data source stalled ({self.err})')
            out = bytearray(self.buf[:need])
            del self.buf[:need]
            self.cv.notify_all()
        return torch.frombuffer(out, dtype=torch.uint8).view(n, T + 1).clone()


# -------------------------------------------------------------------------------------------------------- coordinator
JOIN_PAGE = """DiffusionBlocks-MoE distributed training  ({version})
====================================================================
A {params:,}-parameter byte-level language model ({blocks} DiffusionBlocks, each {lpb} transformer layers with a
{experts}-expert MoE FFN, d={d}, ctx={ctx}) trained on {data}.
Every block owns one slice of the noise range, so each computer trains a different block with no gradient traffic
between them; only the trained block (about {mb:.0f} MB) goes back to the coordinator.

  updates {updates:,}   training bytes {tokens:,}   active workers {active}   AR bits/byte {bpb}

Join: one command, no account, no token, nobody to ask (Linux or macOS, Python 3.9+, a GPU is optional):
  curl -fsSL {base}/join.sh | sh
By hand, also on Windows (use py instead of python3 there, and curl.exe in PowerShell):
  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  curl -fsSL {base}/code -o dbmoe.py
  python3 dbmoe.py worker [--device cuda|rocm|xpu|mps] [--threads 4] [--hours 1]
Want your own token first?  python3 dbmoe.py enrol   (optional; the worker does this by itself)
Many machines, one identity: python3 dbmoe.py enrol, copy worker.token to each one, run python3 dbmoe.py worker --token @worker.token --name MACHINE
AMD GPUs need a ROCm build of PyTorch: https://github.com/Marxist-Leninist/dbmoe#amd-gpus-rocm
Macs: https://github.com/Marxist-Leninist/dbmoe#apple-mac-macos
Intel GPUs need an XPU build of PyTorch: https://github.com/Marxist-Leninist/dbmoe#intel-gpus-xpu
The worker gets its own credential by itself, trains one block at a time and serves that block for network inference
(--no-serve turns serving off).  Every update is tested on held-out text before it is merged.  Ctrl+C stops it.
Leaving for good: curl -fsSL {base}/leave.sh | sh     (or python3 dbmoe.py leave; takes you off the public list and ends your credential)
Live status: {base}/status     Sample text: {base}/sample?prompt=The%20&n=120     Network sample: {base}/sample?net=1&route=auto&prompt=The%20&n=32
"""

LEAVE_SH = r"""#!/bin/sh
# Leave the dbmoe network for good: your workers come off the public list and your credential stops working.
# What you already contributed stays in the model, because it cannot be separated out.  You can rejoin at any time.
set -eu
BASE="@BASE@"
if command -v sha256sum >/dev/null 2>&1; then H=$(printf %s "$BASE" | sha256sum | cut -c1-12); else H=$(printf %s "$BASE" | shasum -a 256 | cut -c1-12); fi
F="$HOME/.dbmoe/$H.token"
[ -s "$F" ] || { echo "No saved credential at $F. If you minted your own, use: python3 dbmoe.py leave --token @worker.token" >&2; exit 1; }
printf 'Authorization: Bearer %s\n' "$(cat "$F")" | curl -fsS -X POST -H @- -H 'Content-Type: application/json' -d '{}' "$BASE/leave"
rm -f "$F" "$HOME/.dbmoe/worker-name"
echo
echo "You have left. Any worker still running stops by itself. Your files in ~/dbmoe-worker are yours to delete."
"""

JOIN_SH = r"""#!/bin/sh
# Joins the dbmoe training and inference network.  Worker source: @BASE@/code
# Installs into ~/dbmoe-worker (or $DBMOE_DIR).  Ctrl+C stops it; delete that folder to remove it.
# Options go after "sh -s --", for example:  curl -fsSL @BASE@/join.sh | sh -s -- --threads 4 --hours 1
set -eu
BASE="@BASE@"
DIR="${DBMOE_DIR:-$HOME/dbmoe-worker}"
mkdir -p "$DIR"
cd "$DIR"
PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "dbmoe needs Python 3.9 or newer" >&2; exit 1; }
if ! "$PY" -c 'import torch' 2>/dev/null; then
  [ -x venv/bin/python ] || "$PY" -m venv venv
  PY=venv/bin/python
  if ! "$PY" -c 'import torch' 2>/dev/null; then
    echo "Installing PyTorch (CPU build) into $DIR/venv ..."
    "$PY" -m pip install --quiet --upgrade pip
    if [ "$(uname -s)" = Darwin ]; then "$PY" -m pip install --quiet torch; else "$PY" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu; fi
  fi
fi
curl -fsSL "$BASE/code" -o dbmoe.py.new
mv dbmoe.py.new dbmoe.py
exec "$PY" -u dbmoe.py worker --url "$BASE" "$@"
"""


def client_key(ip):
    """What one address means for the per-address caps: an IPv4 address, or the /64 of an IPv6 one."""
    try:
        addr = ipaddress.ip_address(ip)
        if addr.version == 6 and addr.ipv4_mapped:
            addr = addr.ipv4_mapped
        if addr.version == 6:
            return str(ipaddress.ip_network(f'{addr}/64', strict=False).network_address)
        return str(addr)
    except ValueError:
        return str(ip)[:64]


class Enrolment:
    """Open sign-up: anyone's worker gets its own revocable credential with no human in the loop.  GET join returns a
    signed challenge, POST join with its proof of work returns a token, which does not expire.  Only tokens' SHA-256 and a
    keyed hash of the address are stored.  The daily caps are off unless set; gate_update() is what protects the model."""
    STRIKES = 6
    SUSPEND_S = 6 * 3600

    def __init__(self, path, bits=18, per_client_day=0, global_day=0, ttl=600):
        self.path, self.bits, self.per_client_day, self.global_day, self.ttl = path, bits, per_client_day, global_day, ttl
        self.idle_ttl = 30 * 86400  # activity renews this horizon; no lifetime limit
        self.lock, self.spent = threading.Lock(), {}
        if os.path.exists(path):
            with open(path) as f:
                self.db = json.load(f)
        else:
            self.db = dict(secret=secrets.token_hex(32), participants={})
            self._save()
        self.secret = self.db['secret'].encode()
        self.by_id = {r['id']: r for r in self.db['participants'].values()}

    def _save(self):
        tmp = self.path + '.tmp'
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(self.db, f)
        os.replace(tmp, self.path)

    def _sign(self, text):
        return hmac.new(self.secret, text.encode(), hashlib.sha256).hexdigest()

    def challenge(self):
        ch = f'{int(time.time())}.{secrets.token_hex(12)}'
        return dict(challenge=ch, sig=self._sign(ch), bits=self.bits)

    def redeem(self, msg, client):
        ch, sig, nonce = msg.get('challenge'), msg.get('sig'), msg.get('nonce')
        if not all(isinstance(v, str) and len(v) <= 128 for v in (ch, sig, nonce)):
            return 400, dict(ok=False, error='challenge, sig and nonce are required')
        if not hmac.compare_digest(self._sign(ch), sig):
            return 400, dict(ok=False, error='not one of our challenges')
        t = time.time()
        if not 0 <= t - int(ch.split('.', 1)[0]) <= self.ttl:
            return 400, dict(ok=False, error='challenge expired; ask for a new one')
        if int.from_bytes(hashlib.sha256(f'{ch}.{sig}.{nonce}'.encode()).digest(), 'big') >> (256 - self.bits):
            return 400, dict(ok=False, error='proof of work does not check out')
        who = self._sign('client:' + client)[:32]
        with self.lock:
            self.spent = {k: v for k, v in self.spent.items() if t - v <= self.ttl}
            if ch in self.spent:
                return 400, dict(ok=False, error='challenge already used')
            day = [r for r in self.db['participants'].values() if t - r['t'] < 86400]
            if self.global_day > 0 and len(day) >= self.global_day:
                return 429, dict(ok=False, error='the run has taken its maximum number of new workers today; try tomorrow')
            if self.per_client_day > 0 and sum(r['client'] == who for r in day) >= self.per_client_day:
                return 429, dict(ok=False, error='this address has joined the maximum number of times today; reuse your saved credential')
            self.spent[ch] = t
            tok = 'dbw_' + secrets.token_urlsafe(32)
            rec = dict(id='anon-' + secrets.token_hex(6), t=t, last_seen=t, renewed_at=t, expires=t+self.idle_ttl, client=who, until=0, fails=0, ok=0)
            self.db['participants'][hashlib.sha256(tok.encode()).hexdigest()] = rec
            self.by_id[rec['id']] = rec
            self._save()
        return 200, dict(ok=True, token=tok, participant=rec['id'])

    def lookup(self, tok):
        with self.lock:
            r = self.db['participants'].get(hashlib.sha256(tok.encode()).hexdigest())
            if not r or r.get('revoked'):
                return None
            t = time.time()
            # Legacy participants migrate on first authenticated use, never a surprise cutoff.
            expires = r.get('expires', t + self.idle_ttl)
            if expires <= t:
                return None  # worker automatically re-enrols after long inactivity
            r['last_seen'], r['expires'] = t, t + self.idle_ttl
            if t - r.get('renewed_at', 0) >= 3600:
                r['renewed_at'] = t
                self._save()
            return r['id']

    def known(self, pid):
        return pid in self.by_id

    def leave(self, tok):
        """A participant leaving for good: the credential stops working at once.  Only its hash is remembered, so a
        worker still running with it can be told to stop instead of quietly signing up again."""
        h = hashlib.sha256(tok.encode()).hexdigest()
        with self.lock:
            r = self.db['participants'].pop(h, None)
            if not r:
                return False
            self.by_id.pop(r['id'], None)
            self.db.setdefault('left', {})[h] = time.time()
            self._save()
            return True

    def has_left(self, tok):
        return hashlib.sha256(tok.encode()).hexdigest() in self.db.get('left', {})

    def blocked(self, pid):
        r = self.by_id.get(pid)
        if r and r['until'] > time.time():
            return f"suspended until {time.strftime('%H:%M UTC', time.gmtime(r['until']))}: recent work failed the automatic checks"
        return None

    def passed(self, pid):
        with self.lock:
            r = self.by_id.get(pid)
            if r:
                r['ok'] += 1
                r['fails'] = max(0, r['fails'] - 0.25)  # a leaky count: someone failing half the time still gets stopped

    def strike(self, pid, weight=1):
        with self.lock:
            r = self.by_id.get(pid)
            if r:
                r['fails'] += weight
                if r['fails'] >= self.STRIKES:
                    r['until'], r['fails'] = time.time() + self.SUSPEND_S, 0
                self._save()

    def stats(self):
        t = time.time()
        with self.lock:
            rs = list(self.by_id.values())
        return dict(participants=len(rs), joined_24h=sum(t - r['t'] < 86400 for r in rs), suspended=sum(r['until'] > t for r in rs))


@torch.no_grad()
def gate_loss(cfg, sigmas, x, b, shared, block):
    """Block b's loss on fixed held-out windows at fixed noise levels and fixed noise, so two measurements differ only by the weights."""
    x = x.long()
    s = torch.exp(torch.linspace(math.log(sigmas[b]), math.log(sigmas[b + 1]), x.size(0)))
    noise = torch.randn(x.size(0), x.size(1) - 1, cfg['d'], generator=torch.Generator().manual_seed(1000 + b))
    ids, tgt = x[:, :-1], x[:, 1:]
    D, _, _ = block(shared.context(ids), shared.embed(tgt) + s[:, None, None] * noise, s)
    ce = F.cross_entropy(shared.logits(D).flatten(0, 1).float(), tgt.flatten(), reduction='none').view_as(tgt)
    w = edm_w(s)[:, None]
    return float((w * ce).mean() / w.mean())


@torch.no_grad()
def round_measure(cfg, sigmas, x, sh_sd, blocks, delta):
    """Held-out loss on each of these blocks, before and after a round of shared-parameter deltas: {block: [before, after]}."""
    shared, block, out = Shared(cfg), Block(cfg), {}
    shared.load_state_dict(sh_sd)
    shared.eval(), block.eval()
    for b, sd in blocks.items():
        block.load_state_dict(sd)
        out[b] = [gate_loss(cfg, sigmas, x, b, shared, block)]
    for k, p in shared.state_dict().items():
        p.add_(delta[k].to(p.dtype))
    for b, sd in blocks.items():
        block.load_state_dict(sd)
        out[b].append(gate_loss(cfg, sigmas, x, b, shared, block))
    return out


@torch.no_grad()
def gate_measure(cfg, sigmas, x, b, o, sh_sd, b_sd, o_sd, d_shared, d_block, w_shared, w_block, trained, margin):
    """Cheapest and strongest first.  At full weight, did the update lower the loss on text from its own lease?  Noise and
    empty uploads stop there.  Only then, on held-out text: its block part at the weight it merges with, which happens at
    once, and on its own its shared part at the most that can count for in a round."""
    shared, block = Shared(cfg), Block(cfg)
    shared.load_state_dict(sh_sd)
    block.load_state_dict(b_sd)
    shared.eval(), block.eval()

    def nudge(w_s, w_b):
        for k, p in shared.state_dict().items():
            p.add_(d_shared[k].to(p.dtype), alpha=w_s)
        for k, p in block.state_dict().items():
            p.add_(d_block[k].to(p.dtype), alpha=w_b)

    m = dict(trained_before=gate_loss(cfg, sigmas, trained, b, shared, block), before=None, after=None, shared_after=None, other_before=None, other_after=None)
    nudge(1.0, 1.0)
    m['trained_after'] = gate_loss(cfg, sigmas, trained, b, shared, block)
    if not m['trained_after'] < m['trained_before'] * (1 - margin):     # also true for NaN
        return m
    shared.load_state_dict(sh_sd)
    block.load_state_dict(b_sd)
    m['before'] = gate_loss(cfg, sigmas, x, b, shared, block)
    if o is not None:
        other = Block(cfg)
        other.load_state_dict(o_sd)
        other.eval()
        m['other_before'] = gate_loss(cfg, sigmas, x, o, shared, other)
    nudge(0.0, w_block)
    m['after'] = gate_loss(cfg, sigmas, x, b, shared, block)
    block.load_state_dict(b_sd)
    nudge(w_shared, 0.0)
    m['shared_after'] = gate_loss(cfg, sigmas, x, b, shared, block)
    if o is not None:
        m['other_after'] = gate_loss(cfg, sigmas, x, o, shared, other)
    return m


def encode_floats(t):
    return base64.b64encode(array.array('f', t.detach().float().reshape(-1).tolist()).tobytes()).decode()


def decode_floats(s, n):
    raw = base64.b64decode(s, validate=True)
    if len(raw) != 4 * n:
        raise ValueError('wrong tensor length')
    return torch.tensor(array.array('f', raw).tolist(), dtype=torch.float32)


def sock_alive(sock):
    """True while the peer keeps the connection open: nothing readable, or real data waiting (EOF or an error means gone)."""
    try:
        if hasattr(select, 'poll'):
            p = select.poll()
            p.register(sock, select.POLLIN | select.POLLHUP | select.POLLERR)
            ready = p.poll(0)
        else:   # Windows has no poll(); select() on one socket has no descriptor-number limit there
            readable, _, broken = select.select([sock], [], [sock], 0)
            ready = readable or broken
        return not ready or bool(sock.recv(1, socket.MSG_PEEK))
    except Exception:
        return False


class _Slot:
    """One worker parked in a long poll, waiting for a hop of the block it holds."""
    __slots__ = ('block', 'conn', 'event', 'job')

    def __init__(self, block, conn):
        self.block, self.conn, self.event, self.job = block, conn, threading.Event(), None


class AllocatorMaintenance:
    """Return unused native heap pages, at most once a minute; never touch live tensors."""
    def __init__(self, interval=60.0, clock=time.monotonic):
        self.interval, self.clock = interval, clock
        self._lock = threading.Lock()
        self._last = float('-inf')
        self._trim = None
        if sys.platform.startswith('linux'):
            try:
                trim = ctypes.CDLL(None).malloc_trim
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
                self._trim = trim
            except (AttributeError, OSError):
                pass

    def maybe_trim(self):
        if self._trim is None or not self._lock.acquire(blocking=False):
            return False
        try:
            now = self.clock()
            if now - self._last < self.interval:
                return False
            self._last = now
            self._trim(0)
            return True
        except Exception:
            self._trim = None
            return False
        finally:
            self._lock.release()


class NetInfer:
    """Network-distributed inference: the chain of blocks runs on the workers and the coordinator only routes.
    Workers hold one block of a frozen snapshot and long-poll for hops.  Snapshots roll every `refresh` seconds; the
    last two stay live here, so hops from a worker that has not refreshed yet can still be checked."""
    def __init__(self, C, refresh=600.0, poll_s=20.0, hop_timeout=5.0, retries=2, idle_wait=3.0, spb=STEPS_PER_BLOCK, ctx_len=128,
                 audit_trusted=0.25, audit_open=0.25, max_peers=2048, online_s=45.0):
        self.C, self.refresh, self.poll_s, self.hop_timeout, self.retries = C, refresh, poll_s, hop_timeout, retries
        self.idle_wait, self.spb, self.ctx_len, self.audit_trusted, self.audit_open = idle_wait, spb, ctx_len, audit_trusted, audit_open
        self.max_peers, self.online_s = max_peers, online_s
        self.cv = threading.Condition(threading.RLock())
        self.snaps, self.current = collections.OrderedDict(), None
        self.active_snaps = collections.Counter()
        self.snapshot_lock = threading.Lock()
        self.snapshot_epoch = secrets.randbits(51) + (1 << 51)  # JSON-safe, fresh process identity
        self.peers, self.idle, self.pending, self.fetching, self.last_fetch = {}, {}, {}, {}, {}
        self.gen_lock = threading.Lock()
        self.stats = dict(generations=0, bytes=0, seconds=0.0, hops_network=0, hops_coordinator=0, retries=0, audits=0, audit_failures=0)
        self.recent = collections.deque(maxlen=64)
        self.auto_choices = {}                 # (snapshot, sampler) -> measured paired route choice
        self.memory_maintenance = AllocatorMaintenance()
        self.snapshot()
        threading.Thread(target=self._housekeeping, daemon=True).start()

    # --- snapshots
    def snapshot(self):
        with self.snapshot_lock:
            return self._snapshot()

    def _snapshot(self):
        with self.C.lock:
            m = self.C.copy_model()
            updates = self.C.meta['updates']
        digest = hashlib.sha256(json.dumps(dict(cfg=m.cfg, sigmas=m.sigmas), sort_keys=True).encode())
        for name, value in sorted(m.state_dict().items()):
            digest.update(name.encode())
            digest.update(str((tuple(value.shape), str(value.dtype))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        state_hash = digest.hexdigest()
        with self.cv:
            sid = (self.current if self.current is not None else self.snapshot_epoch) + 1
            self.snaps[sid] = dict(id=sid, model=m, updates=updates, state_hash=state_hash, t=time.time(), blobs={})
            self.current = sid
            keep = set(list(self.snaps)[-2:]) | set(self.active_snaps)
            for old in list(self.snaps):
                if old not in keep:
                    del self.snaps[old]
                    for key in [key for key in self.auto_choices if key[0] == old]:
                        del self.auto_choices[key]
            for slot in self.idle.values():             # wake parked workers so they can pick up the new weights
                slot.event.set()
        return sid

    def _housekeeping(self):
        while True:
            time.sleep(10)
            try:
                with self.C.lock:
                    updates = self.C.meta['updates']
                s = self.snaps[self.current]
                if time.time() - s['t'] >= self.refresh and updates != s['updates']:
                    self.snapshot()
                with self.cv:
                    t = time.time()
                    for k in [k for k, q in self.peers.items() if t - q['seen'] > 120 and k not in self.idle]:
                        del self.peers[k]
                # Outside model/network locks; glibc only releases unused heap pages.
                self.memory_maintenance.maybe_trim()
            except Exception as e:
                print(f'[net] housekeeping: {e!r}', flush=True)

    def block_blob(self, snap, block, owner):
        with self.cv:
            s = self.snaps.get(snap)
            if not s or not 0 <= block < s['model'].cfg['blocks']:
                return None, 'unknown snapshot or block'
            if self.C.is_open(owner):
                t = time.time()
                if t - self.last_fetch.get(owner, 0) < 15 / max(1, sum(q['owner'] == owner for q in self.peers.values())):
                    return None, 'slow down'
                self.last_fetch[owner] = t
            blob = s['blobs'].get(block)
        if blob is None:
            m = s['model']
            blob = to_bytes(dict(snap=snap, block=block, cfg=m.cfg, sigmas=m.sigmas,
                                 shared={k: v.clone() for k, v in m.shared.state_dict().items()},
                                 block_state={k: v.clone() for k, v in m.blocks[block].state_dict().items()}))
            s['blobs'][block] = blob
        return blob, None

    # --- worker side of the router
    def _target(self, key, p, now):
        """Keep every block covered: a worker stays on its block unless another block has nobody and this one has company."""
        cover = [0] * self.C.model.cfg['blocks']
        for k, q in self.peers.items():
            if k != key and now - q['seen'] < self.online_s and not q['banned'] and q['target'] is not None:
                cover[q['target']] += 1
        cur, low = p['target'], min(cover)
        if cur is None or (low == 0 and cover[cur] > 0):
            p['target'] = random.choice([i for i in range(len(cover)) if cover[i] == low])
        return p['target']

    def _may_fetch(self, key, block, now, urgent):
        """Workers of one block refresh one at a time, so a refresh never leaves the block dark."""
        cur = self.fetching.get(block)
        if cur and cur[0] != key and now < cur[1] and not urgent:
            return False
        self.fetching[block] = (key, now + 30)
        return True

    def poll(self, owner, worker, msg, conn):
        key = f"{worker}#{str(msg.get('session', ''))[:32]}"
        now, B = time.time(), self.C.model.cfg['blocks']
        snap, block, res = msg.get('snap'), msg.get('block'), msg.get('result')
        snap = snap if type(snap) is int else None
        block = block if type(block) is int and 0 <= block < B else None
        is_open = self.C.is_open(owner)
        with self.cv:
            p = self.peers.get(key)
            if p is None:
                if len(self.peers) >= self.max_peers or (is_open and sum(q['owner'] == owner for q in self.peers.values()) >= 256):
                    return dict(idle=True, retry_s=60)
                p = self.peers[key] = dict(owner=owner, worker=worker, open=is_open, target=None, ewma=200.0, hops=0, strikes=0, banned=False)
            p.update(seen=now, snap=snap, block=block)
            if isinstance(res, dict):
                P = self.pending.get(str(res.get('id')))
                if P and P['key'] == key:
                    P['result'] = res
                    P['event'].set()
            if p['banned']:
                return dict(idle=True, retry_s=600)
            fetching = self.fetching.get(block)
            if fetching and fetching[0] == key and snap == self.current:
                del self.fetching[block]
            live = block is not None and snap in self.snaps
            keep_pinned = live and self.active_snaps.get(snap, 0) > 0
            target = block if keep_pinned else self._target(key, p, now)
            if not keep_pinned and (not live or block != target or snap != self.current):
                if self._may_fetch(key, target, now, urgent=not live or block != target):
                    return dict(assign=dict(snap=self.current, block=target))
                if not live:
                    return dict(idle=True, retry_s=3)
            old = self.idle.get(key)
            if old:
                old.event.set()
            slot = self.idle[key] = _Slot(block, conn)
            self.cv.notify_all()
        slot.event.wait(self.poll_s)
        with self.cv:
            if self.idle.get(key) is slot:
                del self.idle[key]
            job = slot.job
        return dict(job=job) if job else dict(idle=True, retry_s=0)

    def forget(self, owner, worker=None):
        removed = set()
        with self.cv:
            for k in [k for k, q in self.peers.items() if q['owner'] == owner and (worker is None or q['worker'] == worker)]:
                removed.add(self.peers[k]['worker'])
                del self.peers[k]
                slot = self.idle.pop(k, None)
                if slot:
                    slot.event.set()
        return removed

    # --- generation side of the router
    def coverage(self, snapshot_id=None):
        now = time.time()
        with self.cv:
            model = self.C.model if snapshot_id is None else self.snaps[snapshot_id]['model']
            cov = [0] * model.cfg['blocks']
            for q in self.peers.values():
                if (now - q['seen'] < self.online_s and not q['banned'] and q.get('block') is not None
                        and 0 <= q['block'] < len(cov) and q.get('snap') in self.snaps
                        and (snapshot_id is None or q.get('snap') == snapshot_id)):
                    cov[q['block']] += 1
        return cov

    def _claim(self, job, tried):
        """Give the hop to the fastest idle worker holding its block; if they are all mid-hop, wait briefly."""
        deadline = time.time() + self.idle_wait
        with self.cv:
            while True:
                best = bestp = bestkey = None
                now = time.time()
                for key, slot in list(self.idle.items()):
                    p = self.peers.get(key)
                    if slot.block != job['block'] or key in tried or not p or p['banned'] or p.get('snap') not in self.snaps or p.get('snap') != job.get('snap'):
                        continue
                    if not sock_alive(slot.conn):
                        del self.idle[key]
                        slot.event.set()
                        p['seen'] = 0.0                 # gone: don't wait for it (its next poll, if any, brings it back)
                        continue
                    if best is None or p['ewma'] < bestp['ewma']:
                        best, bestp, bestkey = slot, p, key
                if best is not None:
                    del self.idle[bestkey]
                    j = dict(job, id=secrets.token_hex(8), snap=bestp['snap'])
                    P = dict(key=bestkey, event=threading.Event(), result=None, job=j, t0=time.time())
                    self.pending[j['id']] = P
                    best.job = j
                    best.event.set()
                    return P, bestp
                someone = any(k not in tried and not q['banned'] and q.get('block') == job['block'] and q.get('snap') == job.get('snap') and q.get('snap') in self.snaps
                              and now - q['seen'] < self.online_s for k, q in self.peers.items())
                left = deadline - now
                if not someone or left <= 0:
                    return None, None
                self.cv.wait(left)

    def _decode(self, res, final, cfg=None):
        cfg = self.C.model.cfg if cfg is None else cfg
        if res.get('error'):
            raise ValueError(str(res['error'])[:200])
        zl = decode_floats(res['z'], cfg['d']).view(1, 1, -1)
        lg = decode_floats(res['logits'], cfg['vocab']) if res.get('logits') else None
        if final and lg is None:
            raise ValueError('the last block must return logits')
        if not bool(torch.isfinite(zl).all()) or float(zl.abs().max()) > 1e4:
            raise ValueError('z out of range')
        if lg is not None and (not bool(torch.isfinite(lg).all()) or float(lg.abs().max()) > 1e5):
            raise ValueError('logits out of range')
        return zl, lg

    def _local(self, job, model, exit_conf=None):
        return run_segment(model.shared, model.blocks[job['block']], base64.b64decode(job['ids']), job['seed'],
                           decode_floats(job['z'], model.cfg['d']).view(1, 1, -1), job['grid'], job['final'],
                           job.get('exit_conf', 0.0) if exit_conf is None else exit_conf)

    def _matches(self, j, zl, lg, model=None):
        """Does this result agree with the snapshot it claims to come from?  Returns (agrees or None, z, logits) recomputed here."""
        if model is None:
            with self.cv:
                s = self.snaps.get(j['snap'])
            if not s:
                return None, None, None
            model = s['model']
        zc, lc = self._local(j, model)
        if (lg is None) != (lc is None) and 0.0 < j.get('exit_conf', 0.0) < 1.0:  # only configured confidence exits admit threshold tolerance
            zc, lc = self._local(j, model, j.get('exit_conf', 0.0) + (-0.01 if lg is not None else 0.01))
        ok = (lg is None) == (lc is None) and float((zl - zc).abs().max()) <= 0.02 * max(1.0, float(zc.abs().max()))
        if ok and lc is not None:
            ok = float((lg - lc).abs().max()) <= 0.02 * max(1.0, float(lc.abs().max()))
        return ok, zc, lc

    def _strike(self, key, p):
        with self.cv:
            self.stats['audit_failures'] += 1
            p['strikes'] += 1
            if p['strikes'] >= 2:
                p['banned'] = True
                slot = self.idle.pop(key, None)
                if slot:
                    slot.event.set()
        if p['open']:
            self.C.enrol.strike(p['owner'], 2)
        self.C.log(ev='hop_mismatch', worker=p['worker'], strikes=p['strikes'], banned=p['banned'])

    def _audit_later(self, key, p, j, zl, lg, model):
        try:
            ok, _, _ = self._matches(j, zl, lg, model=model)
            with self.cv:
                self.stats['audits'] += ok is not None
            if ok is False:
                self._strike(key, p)
        except Exception as e:
            print(f'[net] audit: {e!r}', flush=True)

    def hop(self, job):
        """One block's part of one byte, on a worker if at all possible.  Returns z, logits-or-None, and what ran it."""
        tried = set()
        verification_failed = False
        for _ in range(0 if job.get('route') == 'single' else self.retries + 1):
            P, p = self._claim(job, tried)
            if P is None:
                break
            key, j = P['key'], P['job']
            got = P['event'].wait(self.hop_timeout)
            with self.cv:
                self.pending.pop(j['id'], None)
            tried.add(key)
            ms = (time.time() - P['t0']) * 1000
            zl = lg = None
            if got:
                try:
                    zl, lg = self._decode(P['result'], job['final'], self.snaps[job['snap']]['model'].cfg)
                    if not job['final'] and not job.get('exit_conf', 0.0) and lg is not None:
                        zl = None
                        raise ValueError('unexpected early-exit logits in full chain')
                except Exception as e:
                    print(f'[net] bad hop result: {e!r}', flush=True)
            if zl is None:
                with self.cv:
                    p['ewma'] = min(p['ewma'] * 2 + 500, 60000.0)
                    p['seen'] = 0.0
                    self.stats['retries'] += 1
                continue
            with self.cv:
                p['ewma'] = 0.7 * p['ewma'] + 0.3 * ms
                p['hops'] += 1
                first_hops = p['open'] and p['hops'] <= 5
            if first_hops:                              # a new volunteer's first hops are checked before anyone sees them
                ok, zc, lc = self._matches(j, zl, lg)
                with self.cv:
                    self.stats['audits'] += ok is not None
                if ok is False:
                    self._strike(key, p)
                    if job.get('route') == 'distributed':
                        # Discard the bad result, but allow another worker of this exact
                        # snapshot/block to finish the hop within the existing budget.
                        verification_failed = True
                        with self.cv:
                            self.stats['retries'] += 1
                        continue
                    with self.cv:
                        self.stats['hops_coordinator'] += 1
                    return zc, lc, dict(block=job['block'], via='coordinator', ms=round(ms))
            elif random.random() < (self.audit_open if p['open'] else self.audit_trusted):
                threading.Thread(target=self._audit_later, args=(key, p, j, zl, lg, self.snaps[j['snap']]['model']), daemon=True).start()
            with self.cv:
                self.stats['hops_network'] += 1
            return zl, lg, dict(block=job['block'], via=self.C.alias(p['worker']), ms=round(ms))
        if job.get('route') == 'distributed':
            if verification_failed:
                raise RuntimeError('distributed verification failed; no single-host substitution')
            raise RuntimeError('distributed route unavailable; no single-host substitution')
        t0 = time.time()
        with self.cv:
            model = self.snaps[job['snap']]['model']
        zl, lg = self._local(job, model)
        with self.cv:
            self.stats['hops_coordinator'] += 1
        return zl, lg, dict(block=job['block'], via='coordinator', ms=round((time.time() - t0) * 1000))

    def _auto_choice(self, snapshot_id, sampler, temp, top_k, seed):
        """Measure one byte on each real route, then cache the faster route for this snapshot/sampler.

        The probes use fixed public text and their own pinned generations. They never consume the caller's
        prompt or alter its random stream. A failed distributed probe selects a working single route; there is
        no hidden fallback during the actual generation.
        """
        key = (snapshot_id, sampler)
        with self.cv:
            cached = self.auto_choices.get(key)
        if cached:
            return dict(cached, cached=True)
        measured = {}
        for route in ('single', 'distributed'):
            started = time.monotonic()
            try:
                events = list(self.generate('The ', 1, temp=temp, top_k=top_k, seed=seed, sampler=sampler,
                                            route=route, snapshot_id=snapshot_id, _calibration=True))
                if not events or events[-1].get('t') != 'done' or events[-1].get('bytes') != 1:
                    raise RuntimeError('calibration did not complete')
                measured[route] = dict(ok=True, seconds=round(time.monotonic() - started, 4))
            except Exception as e:
                measured[route] = dict(ok=False, seconds=round(time.monotonic() - started, 4), error=type(e).__name__)
        usable = [route for route in ('single', 'distributed') if measured[route]['ok']]
        if not usable:
            raise RuntimeError('automatic calibration failed on both routes')
        route = min(usable, key=lambda name: measured[name]['seconds'])
        result = dict(route=route, sampler=sampler, snapshot_id=snapshot_id, measured=measured,
                      cached=False, measured_at=time.time(), probe='one byte per route on the pinned snapshot')
        with self.cv:
            self.auto_choices[key] = dict(result)
        return result

    def generate(self, prompt, n, temp=0.0, top_k=40, seed=0, sampler='chain', route='network', snapshot_id=None, _calibration=False, budget_s=None):
        """Yield JSON-ready events using one pinned snapshot and the requested sampler."""
        if route not in ('auto', 'single', 'distributed', 'network'):
            raise ValueError('route must be auto, single, distributed or network')
        if sampler not in ('chain', 'top'):
            raise ValueError('sampler must be chain or top')
        deadline = time.monotonic() + budget_s if budget_s is not None else None
        requested_route = route
        gen = torch.Generator().manual_seed(seed)
        ids = encode(prompt) or [10]
        with self.cv:
            sid = self.current if snapshot_id is None else snapshot_id
            if sid not in self.snaps:
                raise ValueError('requested snapshot is unavailable')
        auto_info = None
        if requested_route == 'auto':
            auto_info = self._auto_choice(sid, sampler, temp, top_k, seed)
            route = auto_info['route']
        with self.cv:
            if sid not in self.snaps:
                raise ValueError('requested snapshot is unavailable')
            s0 = self.snaps[sid]
            cfg, sig = s0['model'].cfg, s0['model'].sigmas
            if route == 'distributed':
                required = {cfg['blocks'] - 1} if sampler == 'top' else set(range(cfg['blocks']))
                available = {p.get('block') for p in self.peers.values()
                             if p.get('snap') == sid and not p.get('banned')
                             and time.time() - p.get('seen', 0) < self.online_s}
                if not required <= available:
                    raise RuntimeError('requested distributed snapshot lacks block coverage')
            self.active_snaps[sid] += 1
        B, spb = cfg['blocks'], self.spb
        # The UI promises every block for chain; confidence shortcuts are not a full chain.
        exit_conf = 0.0
        sampler_config = dict(temp=temp, top_k=top_k, steps_per_block=spb, ctx_len=self.ctx_len, exit_conf=exit_conf)
        t_start, out, used, on_net, on_coord = time.time(), 0, set(), 0, 0
        try:
            yield dict(t='start', n=n, prompt_bytes=len(ids), coverage=self.coverage(sid), snapshot_id=sid,
                       state_hash=s0['state_hash'], route=route, requested_route=requested_route, auto_calibration=auto_info,
                       calibration=_calibration, sampler=sampler, sampler_config=sampler_config, seed=seed,
                       snapshot_updates=s0['updates'], snapshot_age_s=round(time.time() - s0['t']))
            for k in range(n):
                if deadline is not None and time.monotonic() >= deadline:
                    break
                t0 = time.time()
                raw_ids = bytes(ids[-self.ctx_len:])
                raw = base64.b64encode(raw_ids).decode()
                ek_seed = int(torch.randint(2 ** 31 - 1, (1,), generator=gen))
                zl = torch.randn(1, 1, cfg['d'], generator=gen) * sig[-1]
                lg, hops = None, []
                prepared = prepare_segment(s0['model'].shared, raw_ids, ek_seed) if route == 'single' else None
                for i in ([B - 1] if sampler == 'top' else reversed(range(B))):
                    grid = [sig[-1], sig[-2]] if sampler == 'top' else block_grid(sig, i, spb)
                    if route == 'single':
                        hop_start = time.time()
                        zl, lg = run_segment(s0['model'].shared, s0['model'].blocks[i], raw_ids, ek_seed,
                                             zl, grid, sampler == 'top' or i == 0, exit_conf, prepared=prepared)
                        info = dict(block=i, via='coordinator', ms=round((time.time() - hop_start) * 1000))
                        with self.cv:
                            self.stats['hops_coordinator'] += 1
                    else:
                        zl, lg, info = self.hop(dict(block=i, ids=raw, seed=ek_seed, z=encode_floats(zl), grid=grid,
                                                     final=sampler == 'top' or i == 0, exit_conf=exit_conf, snap=sid, route=route))
                    hops.append(info)
                    if info['via'] == 'coordinator':
                        on_coord += 1
                    else:
                        on_net += 1
                        used.add(info['via'])
                    if lg is not None:                  # the last block, or the posterior has already collapsed
                        break
                tok = pick_byte(lg, temp, top_k, gen)
                ids.append(tok)
                out += 1
                yield dict(t='byte', i=k, b=tok, ms=round((time.time() - t0) * 1000), hops=hops)
        finally:
            dt = time.time() - t_start
            summary = dict(bytes=out, seconds=round(dt, 2), bytes_per_s=round(out / max(dt, 1e-6), 3), hops_network=on_net,
                           hops_coordinator=on_coord, hops_per_byte=round((on_net + on_coord) / max(out, 1), 2), workers=len(used), snapshot_id=sid, state_hash=s0['state_hash'], route=route,
                           requested_route=requested_route, auto_calibration=auto_info, calibration=_calibration,
                           sampler=sampler, sampler_config=sampler_config, seed=seed, t=time.time(),
                           stop_reason='time_budget' if deadline is not None and time.monotonic() >= deadline and out < n else 'length')
            with self.cv:
                self.active_snaps[sid] -= 1
                if not self.active_snaps[sid]:
                    del self.active_snaps[sid]
                self.stats['generations'] += 1
                self.stats['bytes'] += out
                self.stats['seconds'] += dt
                self.recent.append(summary)
        yield dict(summary, t='done')

    def status(self):
        now = time.time()
        with self.cv:
            cov, st = self.coverage(), dict(self.stats)
            s = self.snaps[self.current]
            online = sum(now - q['seen'] < self.online_s and not q['banned'] for q in self.peers.values())
            peers = {self.C.alias(q['worker']): dict(block=q.get('block'), hops=q['hops'], ms=round(q['ewma']))
                     for q in self.peers.values() if now - q['seen'] < self.online_s and not q['banned']}
            banned = sum(q['banned'] for q in self.peers.values())
            recent = [r for r in self.recent if now - r['t'] < 3600]
            last = dict(self.recent[-1]) if self.recent else None
        hops = st['hops_network'] + st['hops_coordinator']
        secs = sum(r['seconds'] for r in recent)
        if last:
            last['age_s'] = round(now - last.pop('t'))
        st['seconds'] = round(st['seconds'], 1)
        return dict(online_workers=online, banned_workers=banned, peers=peers, coverage=cov, covered_blocks=sum(c > 0 for c in cov), blocks=len(cov),
                    ready=all(c > 0 for c in cov), snapshot=dict(id=s['id'], state_hash=s['state_hash'], updates=s['updates'], age_s=round(now - s['t'])),
                    bytes_per_s_last_hour=round(sum(r['bytes'] for r in recent) / secs, 3) if secs else None,
                    network_hop_share=round(st['hops_network'] / hops, 4) if hops else None, last_generation=last, **st)




class Coordinator:
    def __init__(self, a):
        self.dir = a.dir
        self.rate_started_at = time.time()  # Fresh process boundary for accepted-update throughput.
        os.makedirs(os.path.join(self.dir, 'snapshots'), exist_ok=True)
        self.path = os.path.join(self.dir, 'master.pt')
        self.log_path = os.path.join(self.dir, 'events.jsonl')
        self.lock = threading.RLock()
        self.save_lock = threading.Lock()
        self.pending_leases = {}
        self.pending_workers = set()
        self.max_identity_leases = 256
        self.max_active_leases = 512                    # what one coordinator can serve, not a limit on people
        self.reserved_leases = 8                        # only tokens from tokens.json can use these
        self.max_open_leases = 256                      # per credential, so one token can cover a whole network of machines
        self.open_lease_interval, self.last_open_lease = 2.0, {}
        self.gate_strict, self.shared_bank, self.shared_round = False, {}, 12
        self.round_busy, self.block_gates = False, collections.defaultdict(threading.Lock)
        self.gate_drift, self.check_windows, self.min_steps = {}, 6, a.min_steps
        torch.set_num_threads(a.eval_threads)
        self.upload_slots = threading.BoundedSemaphore(32)
        self.update_formats, self.pickle_last = collections.Counter(), None   # wire format of each parsed /update body
        self.format_lock = threading.Lock()
        self.pickle_policy = getattr(a, 'pickle_updates', 'guarded')
        if self.pickle_policy not in ('guarded', 'reject'):
            raise SystemExit(f'--pickle-updates / DBMOE_PICKLE_UPDATES must be guarded or reject, not {self.pickle_policy!r}')
        self.gate_slots = threading.BoundedSemaphore(2)     # tests run two at a time; uploads wait their turn
        self.tok_path, self.tok_mtime, self.tokens = a.tokens, None, {}
        self.steps, self.batch, self.lr, self.ttl = a.steps, a.batch, a.lr, a.lease_ttl
        self.max_steps, self.max_batch = a.max_steps, a.max_batch
        self.eval_every, self.snap_every, self.eval_threads = a.eval_every, a.snap_every, a.eval_threads
        if os.path.exists(self.path):
            self.model, ck = load_ckpt(self.path)
            self.meta = ck['meta']
            print(f'[coord] resumed {self.path}: {self.meta["updates"]} updates', flush=True)
        else:
            torch.manual_seed(a.seed)
            self.model = Model(make_cfg(json.loads(a.cfg) if a.cfg else {}))
            self.meta = dict(version=VERSION, created=now(), updates=0, tokens=0, data_docs=0, data_source=a.hf or a.text_file,
                             block_updates=[0] * self.model.cfg['blocks'], block_ce=[None] * self.model.cfg['blocks'],
                             workers={}, evals=[], rejected=0)
        self.meta.setdefault('training_balance', dict(start_updates=self.meta['updates'],
                            accepted_tokens=[0] * self.model.cfg['blocks'], recent=[]))
        self.meta.setdefault('worker_alias_salt', secrets.token_hex(32))
        self.meta.setdefault('gate', dict(passed=0, refused=0))
        self.leases = {}
        self.data = TextSource(hf=a.hf, text_file=a.text_file, skip_docs=self.meta.get('data_docs', 0))
        hp = os.path.join(self.dir, 'heldout.pt')
        if os.path.exists(hp):
            self.heldout = torch.load(hp, weights_only=True)
        else:
            self.heldout = self.data.windows(a.heldout, self.model.cfg['ctx'])      # first windows of the stream
            torch.save(self.heldout, hp)
        self.last_save, self.eval_busy, self.eval_started, self.sample_lock = time.time(), False, 0, threading.Lock()
        gp = os.path.join(self.dir, 'gate.pt')          # never sent to anyone: open participants' updates must not hurt it
        if os.path.exists(gp):
            self.gate_x = torch.load(gp, weights_only=True)
        else:
            self.gate_x = self.data.windows(a.gate_windows, self.model.cfg['ctx'])
            torch.save(self.gate_x, gp)
        self.gate_all, self.gate_tol = a.gate_all, a.gate_tol
        self.teacher, self.gate_recent, self.learn_margin = None, collections.deque(maxlen=200), a.learn_margin
        threading.Thread(target=self._load_teacher, daemon=True).start()
        self.enrol = Enrolment(os.path.join(self.dir, 'enrolled.json'), bits=a.join_bits, per_client_day=a.join_per_client_day,
                               global_day=a.join_per_day) if a.open_join else None
        self.save()
        self.net = NetInfer(self, refresh=a.net_refresh)

    # --- persistence / logging / auth
    def save(self, snapshot=False):
        # Serialize atomic checkpoint replacement across concurrent update handlers.
        with self.save_lock:
            self._save(snapshot)

    def _save(self, snapshot=False):
        with self.lock:
            self.meta['data_docs'] = self.data.docs
            blob = dict(cfg=self.model.cfg, state={k: v.clone() for k, v in self.model.state_dict().items()},
                        meta=json.loads(json.dumps(self.meta)))
        tmp = self.path + '.tmp'
        torch.save(blob, tmp)
        os.replace(tmp, self.path)
        if snapshot:
            sd = os.path.join(self.dir, 'snapshots')
            torch.save(blob, os.path.join(sd, f'u{blob["meta"]["updates"]:07d}.pt'))
            for f in sorted(f for f in os.listdir(sd) if f.endswith('.pt'))[:-3]:
                os.remove(os.path.join(sd, f))

    def log(self, **ev):
        ev['t'] = now()
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(ev) + '\n')

    def auth(self, headers):
        if not self.tok_path:
            return 'anon'
        try:
            mt = os.path.getmtime(self.tok_path)
            if mt != self.tok_mtime:
                self.tokens, self.tok_mtime = json.load(open(self.tok_path)), mt
        except Exception as e:
            print(f'[coord] tokens file: {e}', flush=True)
        tok = (headers.get('Authorization') or '').removeprefix('Bearer ').strip()
        if not tok:
            return None
        return self.tokens.get(tok) or (self.enrol.lookup(tok) if self.enrol else None)

    def is_open(self, owner):
        return bool(self.enrol) and (self.enrol.known(owner) or str(owner).startswith('self-'))

    def blocked(self, owner):
        return self.enrol.blocked(owner) if self.is_open(owner) else None

    def auth_failure(self, headers):
        tok = (headers.get('Authorization') or '').removeprefix('Bearer ').strip()
        if tok and self.enrol and self.enrol.has_left(tok):
            return 410, 'this credential has left the run; start the worker again to rejoin with a new one'
        return 401, 'missing or unknown token'

    def leave(self, who, headers, name=None):
        """Take a participant's workers off the public list: all of them, or one by label.  Their accepted updates stay
        in the model and the totals, because they cannot be separated out; only the record goes."""
        prefix, one = f'{who}/', f'{who}/{name}' if name else None
        # Collect identities before removing their last record. Never nest the
        # inference condition and model lock: other paths use both independently.
        gone = set(self.net.forget(who, one))
        with self.lock:
            trained = [k for k in self.meta['workers'] if k.startswith(prefix) and (one is None or k == one)]
            gone.update(trained)
            for k in trained:
                del self.meta['workers'][k]
            for k in [k for k, v in self.leases.items() if v.get('owner') == who and (one is None or v['worker'] == one)]:
                gone.add(self.leases[k]['worker'])
                del self.leases[k]
            left = self.meta.setdefault('left_workers', [])
            left.extend(self.alias(k) for k in sorted(gone) if self.alias(k) not in left)
            del left[:-5000]
        tok = (headers.get('Authorization') or '').removeprefix('Bearer ').strip()
        ended = bool(not name and self.enrol and self.enrol.leave(tok))
        self.save()
        self.log(ev='left', participant=who, name=name, records=len(gone), credential_ended=ended)
        return dict(ok=True, removed=len(gone), credential_ended=ended)

    def has_lease(self, owner):
        with self.lock:
            self.expire()
            return any(v.get('owner') == owner for v in self.leases.values())

    def alias(self, name):
        return 'worker-' + hmac.new(self.meta['worker_alias_salt'].encode(), name.encode(), hashlib.sha256).hexdigest()[:16]

    # --- leases
    def expire(self):
        t = time.time()
        for k in [k for k, v in self.leases.items() if t - v['t0'] > v['ttl']]:
            v = self.leases.pop(k)
            self.log(ev='lease_expired', lease=k, block=v['block'], worker=v['worker'])

    def lease(self, worker, steps, batch, session='', owner=None, client=None):
        owner = owner or worker.split('/', 1)[0]
        is_open = self.is_open(owner)
        # Reserve capacity before consuming data, including concurrent requests.
        with self.lock:
            self.expire()
            key = (owner, worker)
            if key in self.pending_workers:
                raise LeaseCapacityError('lease request already pending for this worker; retry shortly')
            pending = self.pending_leases.get(owner, 0)
            if is_open:
                t = time.time()
                if t - self.last_open_lease.get(worker, 0) < self.open_lease_interval:     # each machine, so one token can run a whole network of them
                    raise LeaseCapacityError('too many lease requests; slow down')
                if len(self.last_open_lease) > 20000:
                    self.last_open_lease = {k: v for k, v in self.last_open_lease.items() if t - v < 60}
                self.last_open_lease[worker] = t
            live = [v for v in self.leases.values() if not (v['worker'] == worker and v.get('owner', v['worker'].split('/', 1)[0]) == owner)]     # the same owner's stable worker replaces its prior job, including after restart
            replacing = len(live) < len(self.leases)
            own = sum(v.get('owner', v['worker'].split('/', 1)[0]) == owner for v in live)
            total = len(live) + sum(self.pending_leases.values())
            cap = self.max_open_leases if is_open else self.max_identity_leases
            if own + pending >= cap or total >= self.max_active_leases - (self.reserved_leases if is_open and not replacing else 0):
                raise LeaseCapacityError('active lease capacity reached; finish existing work first')
            self.pending_leases[owner] = pending + 1
            self.pending_workers.add(key)
        try:
            return self._lease(worker, steps, batch, session, owner, client)
        finally:
            with self.lock:
                self.pending_workers.discard(key)
                left = self.pending_leases[owner] - 1
                if left:
                    self.pending_leases[owner] = left
                else:
                    self.pending_leases.pop(owner, None)

    def _lease(self, worker, steps, batch, session='', owner=None, client=None):
        steps = max(self.min_steps, min(self.max_steps, steps or self.steps))
        batch = max(1, min(self.max_batch, batch or self.batch))
        data = self.data.windows(steps * batch, self.model.cfg['ctx'])      # before taking the lock (may wait on HF)
        with self.lock:
            self.expire()
            for k in [k for k, v in self.leases.items() if v['worker'] == worker and v.get('owner', v['worker'].split('/', 1)[0]) == owner]:
                self.log(ev='lease_abandoned', lease=k, block=self.leases.pop(k)['block'], worker=worker)
            B, bu = self.model.cfg['blocks'], self.meta['block_updates']
            active = [0] * B
            for v in self.leases.values():
                active[v['block']] += 1
            w = self.meta['workers'].get(worker) or {}
            balance = self.training_balance()
            b = choose_training_block(self.block_weights(), balance['recent_accepted_tokens'],
                                      balance['inflight_tokens'], steps * batch * self.model.cfg['ctx'],
                                      w.get('last_block'))
            for v in self.leases.values():
                v['all'] += 1
                v['same'] += v['block'] == b
            wall = w.get('wall_s')
            ttl = self.ttl if not wall else max(300.0, min(self.ttl, 8 * wall * steps / max(1, w.get('steps', steps)) + 120))
            lid = secrets.token_hex(8)
            self.leases[lid] = dict(block=b, worker=worker, owner=owner, client=client, session=session, t0=time.time(), ttl=ttl, steps=steps, batch=batch,
                                    same=active[b] + 1, all=len(self.leases) + 1)
            self.leases[lid]['check'] = data[torch.randperm(data.size(0))[:self.check_windows]].clone()
            payload = dict(version=VERSION, lease=lid, block=b, block_version=bu[b], cfg=self.model.cfg, sigmas=self.model.sigmas,
                           shared={k: v.clone() for k, v in self.model.shared.state_dict().items()},
                           block_state={k: v.clone() for k, v in self.model.blocks[b].state_dict().items()},
                           steps=steps, batch=batch, lr=self.lr, ttl=ttl, data=data)
        self.log(ev='lease', lease=lid, block=b, worker=worker, steps=steps, batch=batch, ttl=round(ttl))
        return payload

    def _load_teacher(self):
        """About 4 MB of real text; only its n-gram statistics are used.  In the background, so a restart does not wait for it."""
        tp = os.path.join(self.dir, 'teacher.pt')
        try:
            if os.path.exists(tp):
                tx = torch.load(tp, weights_only=True)
            else:
                tx = self.data.windows(16000, self.model.cfg['ctx'])
                torch.save(tx, tp)
            self.teacher = Teacher(tx)
        except Exception as e:
            print(f'[coord] teacher: {e!r}', flush=True)

    def training_balance(self):
        """Call under self.lock: expired/replaced/rejected leases are not reservations."""
        B = self.model.cfg['blocks']
        state = self.meta.get('training_balance') or dict(start_updates=self.meta['updates'],
                    accepted_tokens=[0] * B, recent=[])
        accepted, inflight = [0] * B, [0] * B
        for b, tokens in state['recent']:
            accepted[b] += tokens
        for lease in self.leases.values():
            inflight[lease['block']] += lease['steps'] * lease['batch'] * self.model.cfg['ctx']
        weights = self.block_weights()
        return dict(policy='weighted-byte-deficit-v1', history_limit_updates=1024,
                    history_updates=len(state['recent']), start_updates=state['start_updates'],
                    accepted_tokens=list(state['accepted_tokens']), recent_accepted_tokens=accepted,
                    inflight_tokens=inflight, target_shares=[w / sum(weights) for w in weights],
                    units='byte-target presentations; accepted counts begin at start_updates')

    def block_weights(self):
        """How much compute each block deserves.  Low-noise blocks can read the target off z, which is their job, so their
        loss is near zero and more updates teach them little about language; the floor stops them being starved."""
        B = self.model.cfg['blocks']
        ev = self.meta['evals'][-1].get('per_block_ce') if self.meta['evals'] else None
        if not ev or len(ev) != B or not sum(ev):
            return [1.0] * B
        return [max(0.25, c * B / sum(ev)) for c in ev]

    def _bank_shared(self, b, d):
        """Shared parameters define the target every block denoises toward, so they must not drift under one block's
        gradients between other blocks' leases.  Deltas wait here and go in together as one round."""
        bank = self.shared_bank.setdefault(b, dict(n=0, sum={k: torch.zeros_like(v) for k, v in d.items()}))
        bank['n'] += 1
        for k, v in d.items():
            bank['sum'][k].add_(v)
        if sum(v['n'] for v in self.shared_bank.values()) >= self.shared_round and not self.round_busy:
            self.round_busy = True
            threading.Thread(target=self._commit_round, args=(self.shared_bank,), daemon=True).start()
            self.shared_bank = {}

    def _commit_round(self, bank):
        """A round is what really changes the shared parameters, so it gets its own test, weighted by block_weights(), on
        the top block and one other.  Each delta in it passed alone; this catches what they do together.  Merges wait while
        it runs, so what is tested is exactly what goes in, and no update under test sees the shared parameters change."""
        held = []
        try:
            for i in range(self.model.cfg['blocks']):
                self.block_gates[i].acquire()
                held.append(self.block_gates[i])
            cfg, B = self.model.cfg, self.model.cfg['blocks']
            with self.lock:
                wt = self.block_weights()
                delta = {k: sum(v['sum'][k] * (wt[i] / sum(wt) / v['n']) for i, v in bank.items()) for k in self.model.shared.state_dict()}
                sh = {k: v.clone() for k, v in self.model.shared.state_dict().items()}
                blocks = {i: {k: v.clone() for k, v in self.model.blocks[i].state_dict().items()} for i in {B - 1, random.randrange(B)}}
                tol, slack = (0.0, 0.0) if self.gate_strict else (self.gate_tol, 0.003)
            m = round_measure(cfg, self.model.sigmas, self.gate_x, sh, blocks, delta)
            ok = all(math.isfinite(a) and a <= z * (1 + tol) + slack for z, a in m.values())
            with self.lock, torch.no_grad():
                if ok:
                    for k, p in self.model.shared.state_dict().items():
                        p.add_(delta[k].to(p.dtype))
                    self.meta['shared_rounds'] = self.meta.get('shared_rounds', 0) + 1
                else:
                    self.meta['gate']['rounds_refused'] = self.meta['gate'].get('rounds_refused', 0) + 1
            self.log(ev='shared_round', ok=ok, deltas=sum(v['n'] for v in bank.values()), held_out={str(b): [round(z, 5), round(a, 5)] for b, (z, a) in m.items()})
        except Exception as e:
            print(f'[coord] shared round failed: {e!r}', flush=True)
        finally:
            for g in held:
                g.release()
            self.round_busy = False

    def _check(self, up, L):
        # Validate every field used after merging before touching parameters or counters.
        losses = up.get('losses')
        sec = up.get('seconds')
        if not isinstance(losses, list) or len(losses) != L['steps']:
            return 'losses: expected one for each step of the job'
        if 'tokens' in up and (type(up['tokens']) is not int or up['tokens'] != L['steps'] * L['batch'] * self.model.cfg['ctx']):
            return 'tokens: expected steps x batch x context for this job'
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1e6 for v in losses):
            return 'losses: expected finite numbers between zero and 1e6'
        if type(sec) not in (int, float) or not math.isfinite(sec) or not 0 < sec <= 86400:
            return 'seconds: expected finite positive duration'
        if any(not isinstance(up.get(k, ''), str) for k in ('host', 'device', 'name')):
            return 'worker metadata: expected strings'
        lim = 3.5 * self.lr * L['steps'] + 1e-6
        b = L['block']
        for name, ref, d in (('block', self.model.blocks[b].state_dict(), up.get('d_block')),
                             ('shared', self.model.shared.state_dict(), up.get('d_shared'))):
            if not isinstance(d, dict) or set(d) != set(ref):
                return f'{name}: parameter names differ'
            for k, t in d.items():
                if not isinstance(t, torch.Tensor) or t.layout != torch.strided or t.device.type != 'cpu' or t.shape != ref[k].shape or t.dtype != ref[k].dtype or not t.is_floating_point():
                    return f'{name}.{k}: wrong shape/type'
                if not bool(torch.isfinite(t).all()):
                    return f'{name}.{k}: not finite'
                m = float(t.abs().max()) if t.numel() else 0.0
                if m > lim:
                    return f'{name}.{k}: max|delta| {m:.3g} > {lim:.3g}'
        return None

    def gate_update(self, worker, owner, up):
        with self.gate_slots:
            return self._gate_update(worker, owner, up)

    def _gate_update(self, worker, owner, up):
        """The test that stands in for asking anyone's permission, the same for every computer.  An update must lower the
        loss on text from its own lease, which noise or an empty upload cannot do, and must not hurt held-out text nobody is
        given, at the weight it would merge with.  Returns None to go on, a dict if it passed, or the refusal to send."""
        t0 = time.time()
        with self.lock:
            self.expire()
            lease = up.get('lease')
            L = self.leases.get(lease) if isinstance(lease, str) else None
            if L is None or L['block'] != up.get('block') or L['worker'] != worker or self._check(up, L):
                return None                             # update() reports these
            wt = self.block_weights()
            same = L['same']
            b, w_block, w_shared = L['block'], 1.0 / same, wt[L['block']] / sum(wt)     # the most its shared part can count for in a round
            others = [i for i in range(self.model.cfg['blocks']) if i != b]
            o = random.choice(others) if others and random.random() < 0.25 else None    # the effect on other blocks, one time in four
            sh = {k: v.clone() for k, v in self.model.shared.state_dict().items()}
            bs = {k: v.clone() for k, v in self.model.blocks[b].state_dict().items()}
            xs = {k: v.clone() for k, v in self.model.blocks[o].state_dict().items()} if o is not None else None
        m = gate_measure(self.model.cfg, self.model.sigmas, self.gate_x, b, o, sh, bs, xs, up['d_shared'], up['d_block'],
                         w_shared, w_block, L['check'], self.learn_margin)
        why = None
        before, after = m['before'], m['after']
        if before is None:
            why = f"the change on its training text did not meet the required learning margin ({m['trained_before']:.4f} -> {m['trained_after']:.4f})"
        elif not math.isfinite(after) or not math.isfinite(m['shared_after']) or (o is not None and not math.isfinite(m['other_after'])):
            why = 'the result is not a number'
        else:
            tol, slack = (0.0, 0.0) if self.gate_strict else (self.gate_tol, 0.003)     # strict: held-out quality slipped, so only improvements
            rel = (after - before) / max(before, 0.1)
            with self.lock:
                best_all = self.meta.setdefault('gate_best', {})
                best = best_all[str(b)] = min(best_all.get(str(b), before), before)
                drift = self.gate_drift[owner] = 0.8 * self.gate_drift.get(owner, 0.0) + rel
            self.gate_recent.append(rel)
            if after > before * (1 + tol) + slack:
                why = f'it made held-out text worse on block {b} ({before:.4f} -> {after:.4f})'
            elif m['shared_after'] > before * (1 + self.gate_tol) + 0.003:      # strict or not: the round gets its own test
                why = f"its shared part made held-out text worse on block {b} ({before:.4f} -> {m['shared_after']:.4f})"
            elif o is not None and m['other_after'] > m['other_before'] * (1 + self.gate_tol) + 0.003:
                why = f"it made held-out text worse on block {o} ({m['other_before']:.4f} -> {m['other_after']:.4f})"
            elif after > before and after > best * 1.05 + slack:    # no number of small steps can take a block far from its best
                why = f'block {b} is already well above its best ({best:.4f}), so only improvements go in ({before:.4f} -> {after:.4f})'
            elif drift > 0.08:
                why = f'recent updates from this participant keep making held-out text worse ({drift:+.3f})'
        ok = why is None
        self.log(ev='gate', lease=lease, worker=worker, block=b, other=o, why=why, ms=round((time.time() - t0) * 1000), w_block=round(w_block, 3),
                 w_shared=round(w_shared, 3), fresh=bool(up.get('fresh_optimizer')), ok=ok, **{k: None if v is None else round(v, 5) for k, v in m.items()})
        with self.lock:
            self.meta['gate']['passed' if ok else 'refused'] += 1
            if not ok and self.leases.pop(lease, None) is not None:
                self.meta['rejected'] += 1
                w = self.meta['workers'].setdefault(worker, dict(updates=0, tokens=0, seconds=0.0))
                w['refused'] = w.get('refused', 0) + 1
        if not ok and self.is_open(owner) and not self.gate_strict:     # in strict mode the refusal says more about the network than the worker
            near_miss = before is None and m['trained_after'] <= m['trained_before'] * 1.05     # common on a block with little left to learn
            if not (before is None and m['trained_after'] < m['trained_before']):               # it did learn, just not enough to count
                self.enrol.strike(owner, 0.5 if near_miss else 1)
        if ok:
            return {'tested': True, 'same': same}
        return 422, dict(ok=False, error=f'not merged: {why}')

    def update(self, worker, up):
        b = up.get('block')
        if type(b) is not int or not 0 <= b < self.model.cfg['blocks']:
            return self._update(worker, up)             # which reports it
        with self.block_gates[b]:                       # a block's next update is tested once the last is in, so each is tested on what it merges into
            return self._update(worker, up)

    def _update(self, worker, up):
        owner = worker.split('/', 1)[0]
        checked = None
        if self.gate_all or self.is_open(owner):
            checked = self.gate_update(worker, owner, up)
            if isinstance(checked, tuple):
                return checked
        with self.lock:
            self.expire()
            if not isinstance(up.get('lease'), str) or type(up.get('block')) is not int:
                return 400, dict(ok=False, error='invalid lease or block')
            L = self.leases.get(up.get('lease'))
            if L is None or L['block'] != up.get('block') or L['worker'] != worker:
                return 409, dict(ok=False, error='unknown, expired or foreign lease')
            err = self._check(up, L)
            del self.leases[up['lease']]
            if err:
                self.meta['rejected'] += 1
                self.log(ev='rejected', lease=up['lease'], block=L['block'], worker=worker, error=err)
                return 422, dict(ok=False, error=err)
            b = L['block']
            with torch.no_grad():
                for k, p in self.model.blocks[b].state_dict().items():
                    p.add_(up['d_block'][k].to(p.dtype), alpha=1.0 / (checked['same'] if checked else L['same']))    # the weight it was tested at
                self._bank_shared(b, up['d_shared'])
            m = self.meta
            toks = int(L['steps'] * L['batch'] * self.model.cfg['ctx'])
            m['updates'] += 1
            m['tokens'] += toks
            m['block_updates'][b] += 1
            balance = m.setdefault('training_balance', dict(start_updates=m['updates'] - 1,
                                  accepted_tokens=[0] * self.model.cfg['blocks'], recent=[]))
            balance['accepted_tokens'][b] += toks
            balance['recent'].append([b, toks])
            del balance['recent'][:-1024]

            losses = [float(v) for v in (up.get('losses') or [])][:10000]
            lo = round(sum(losses[-5:]) / len(losses[-5:]), 4) if losses else None
            prev = m['block_ce'][b]
            if not self.is_open(L.get('owner')):        # open participants' self-reported losses don't drive the public curve
                m['block_ce'][b] = lo if prev is None or lo is None else round(0.7 * prev + 0.3 * lo, 4)
            sec = float(up.get('seconds') or 0)
            wall = time.time() - L['t0']
            w = m['workers'].setdefault(worker, dict(updates=0, tokens=0, seconds=0.0))
            w.update(updates=w['updates'] + 1, tokens=w['tokens'] + toks, seconds=round(w['seconds'] + sec, 1), last=now(),
                     last_block=b, device=str(up.get('device', ''))[:32], measured_bytes_s=round(toks / max(wall, 1e-6), 1),
                     tok_s=round(toks / max(sec, 1e-6), 1), steps=L['steps'],
                     wall_s=round(wall if not w.get('wall_s') else 0.7 * w['wall_s'] + 0.3 * wall, 1))
            w.pop('host', None)                         # do not retain legacy machine hostnames
            w.pop('hardware', None)                     # omission withdraws the previous public report
            hardware = normalize_hardware(up.get('hardware'))
            if hardware:
                w['hardware'] = hardware
            self.log(ev='update', lease=up['lease'], block=b, worker=worker, same=L['same'], all=L['all'],
                     ce_first=losses[0] if losses else None, ce_last=lo, tokens=toks, seconds=round(sec, 2),
                     tok_s=round(toks / max(sec, 1e-6), 1))
            n = m['updates']
            snap = bool(self.snap_every) and n % self.snap_every == 0
            do_save = snap or time.time() - self.last_save > 60
            last_ev = m['evals'][-1]['updates'] if m['evals'] else 0
            do_eval = bool(self.eval_every) and n - max(last_ev, self.eval_started) >= self.eval_every and not self.eval_busy
            if do_eval:
                self.eval_busy, self.eval_started = True, n
            if do_save:
                self.last_save = time.time()
        if checked and self.is_open(owner):
            self.enrol.passed(owner)
        if do_save:
            self.save(snapshot=snap)
        if do_eval:
            threading.Thread(target=self.run_eval, daemon=True).start()
        return 200, dict(ok=True, updates=n)

    def generation_choice(self):
        """The route whose writing measured closest to real text; the top block until there is a measurement."""
        with self.lock:
            ev = self.meta['evals'][-1] if self.meta['evals'] else {}
            return dict(mode=ev.get('serve_mode', 'top'), temp=ev.get('serve_temp', 0.75))

    def copy_model(self):
        with self.lock:
            m = Model(self.model.cfg)
            m.load_state_dict(self.model.state_dict())
        return m.eval()

    def run_eval(self):
        try:
            torch.set_num_threads(self.eval_threads)
            n0 = self.meta['updates']
            m, t0 = self.copy_model(), time.time()
            ev = dict(updates=n0, tokens=self.meta['tokens'], t=now(), **evaluate(m, self.heldout))
            ev.update(evaluate_chain(m, self.heldout))
            if self.teacher:
                ev.update(evaluate_generation(m, self.heldout, self.teacher))
                close = lambda k: abs(ev[k] - ev['real_bits'])
                ev['serve_mode'] = 'chain' if close('chain_gen_bits') <= close('top_gen_bits') else 'top'
                ev['serve_temp'] = 0.0 if ev['serve_mode'] == 'chain' else 0.75
            ev['sample'] = generate(m, 'The ', n=100, temp=ev.get('serve_temp', 0.0), mode=ev.get('serve_mode', 'chain'), ctx_len=128)
            ev['eval_seconds'] = round(time.time() - t0, 1)
            with self.lock:
                self.meta['evals'] = (self.meta['evals'] + [ev])[-100:]
                self.gate_strict = ev['ar_bits_per_byte'] > 1.05 * min(e['ar_bits_per_byte'] for e in self.meta['evals'])
            self.log(ev='eval', **ev)
            print(f'[coord] eval @{n0}: AR {ev["ar_bits_per_byte"]} bits/byte  per-block CE {ev["per_block_ce"]}  {ev["sample"]!r}', flush=True)
        except Exception as e:
            print(f'[coord] eval failed: {e!r}', flush=True)
        finally:
            self.eval_busy = False

    def recent_training_rate(self):
        """Accepted byte-target presentations per wall second, from this process's event log.

        Bounded metadata read only; no worker IDs or event bodies leave this method.
        A restart starts a new measurement window, never splices a counter reset.
        """
        import datetime as _dt
        import stat as _stat
        end = time.time()
        cached = getattr(self, '_rate_cache', None)
        if cached and 0 <= end - cached[0] < 2:
            return dict(cached[1])
        start = max(self.rate_started_at, end - 60.0)
        span = end - start
        out = dict(state='warming_up', bytes_per_s=None, window_s=round(max(0, span), 2),
                   accepted_byte_targets=None, accepted_updates=None,
                   window_end_utc=_dt.datetime.fromtimestamp(end, _dt.timezone.utc).isoformat(),
                   source='coordinator_accepted_update_log', process_started_at=self.rate_started_at,
                   units='byte-target presentations per wall second')
        if span < 30:
            return out
        try:
            fd = os.open(self.log_path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not _stat.S_ISREG(info.st_mode):
                    raise ValueError('not_regular')
                offset = max(0, info.st_size - 262144)
                stream.seek(offset)
                body = stream.read(262144)
            lines = body.splitlines()
            if offset:
                lines = lines[1:]
            earliest, total, updates, malformed = None, 0, 0, False
            for line in lines:
                try:
                    event = json.loads(line)
                    epoch = _dt.datetime.fromisoformat(event['t'].replace('Z', '+00:00')).timestamp()
                    earliest = epoch if earliest is None else min(earliest, epoch)
                    if start < epoch <= end and event.get('ev') == 'update':
                        value = event.get('tokens')
                        if type(value) is not int or value < 0:
                            raise ValueError('bad_counter')
                        total += value
                        updates += 1
                except (ValueError, TypeError, KeyError, AttributeError):
                    malformed = True
            if malformed or (offset and (earliest is None or earliest > start)):
                out['state'] = 'history_incomplete'
            else:
                out.update(state='measured', bytes_per_s=round(total / span, 2),
                           accepted_byte_targets=total, accepted_updates=updates)
        except (OSError, ValueError):
            out['state'] = 'history_unavailable'
        self._rate_cache = (end, dict(out))
        return out

    def public_status(self):
        # Private identities remain in checkpoints; public metrics use stable pseudonyms.
        with self.lock:
            result = json.loads(json.dumps(self.status()))
            recent = list(self.gate_recent)
            gate = dict(self.meta['gate'], strict=self.gate_strict, shared_rounds=self.meta.get('shared_rounds', 0), recent_tested=len(recent),
                        recent_improved=round(sum(r < 0 for r in recent) / len(recent), 4) if recent else None,
                        recent_avg_change=round(sum(recent) / len(recent), 5) if recent else None)
        result['workers'] = {self.alias(name): {k: v for k, v in stats.items() if k != 'host'}
                             for name, stats in result['workers'].items()}
        for lease in result['active_leases']:
            lease['worker'] = self.alias(lease['worker'])
        result['gate'] = gate
        with self.lock:
            result['left_workers'] = [a for a in self.meta.get('left_workers', []) if a not in result['workers']]
        result['training_throughput'] = self.recent_training_rate()
        result['net_inference'] = self.net.status()
        result['generation'] = self.generation_choice()
        result['open_join'] = dict(open=True, **self.enrol.stats()) if self.enrol else dict(open=False)
        return result

    def note_update_format(self, safetensors):
        """Count the wire format of each /update body that parsed, so --pickle-updates reject can wait until no worker
        still sends torch.save (pickle) updates."""
        with self.format_lock:
            self.update_formats['safetensors' if safetensors else 'pickle'] += 1
            if not safetensors:
                self.pickle_last = time.time()

    def update_format_status(self):
        lock = getattr(self, 'format_lock', None)
        if lock is None:                              # never let format telemetry break /status
            return dict(safetensors=0, pickle=0, pickle_last_age_s=None,
                        pickle_policy=getattr(self, 'pickle_policy', 'guarded'))
        with lock:
            return dict(safetensors=self.update_formats['safetensors'], pickle=self.update_formats['pickle'],
                        pickle_last_age_s=None if self.pickle_last is None else round(time.time() - self.pickle_last),
                        pickle_policy=self.pickle_policy)

    def status(self):
        with self.lock:
            self.expire()
            m = self.meta
            return dict(version=VERSION, cfg=self.model.cfg, sigmas=[round(s, 4) for s in self.model.sigmas],
                        params=sum(p.numel() for p in self.model.parameters()), updates=m['updates'], tokens=m['tokens'],
                        rejected=m['rejected'], block_updates=m['block_updates'], block_ce=m['block_ce'],
                        training_balance=self.training_balance(),
                        update_formats=self.update_format_status(),
                        active_leases=[dict(block=v['block'], worker=v['worker'], age_s=round(time.time() - v['t0']))
                                       for v in self.leases.values()],
                        workers={k: {f: v for f, v in w.items() if f not in ('host', 'ip')} for k, w in m['workers'].items()}, last_eval=(m['evals'][-1] if m['evals'] else None),
                        eval_history=[dict(updates=e['updates'], t=e.get('t'), ar_bpb=e['ar_bits_per_byte'], chain_gen_bits=e.get('chain_gen_bits'),
                                           top_gen_bits=e.get('top_gen_bits'), real_bits=e.get('real_bits')) for e in m['evals']],
                        data=dict(source=m.get('data_source'), docs=self.data.docs, buffered_mb=round(len(self.data.buf) / 2**20, 1),
                                  error=self.data.err),
                        lease_defaults=dict(steps=self.steps, batch=self.batch, lr=self.lr, max_steps=self.max_steps,
                                            max_batch=self.max_batch, ttl_s=self.ttl))


def owner_inference_session(cookie):
    """Validate the existing Linger session only; never grant access by caller IP."""
    if not cookie or len(cookie) > 8192 or '\r' in cookie or '\n' in cookie:
        return False
    from http.cookies import SimpleCookie
    try:
        parsed = SimpleCookie()
        parsed.load(cookie)
        if not parsed.get('linger') or not parsed['linger'].value:
            return False
        owner_cookie = parsed['linger'].OutputString()
    except Exception:
        return False
    conn = httpc.HTTPConnection('127.0.0.1', 8795, timeout=2)
    try:
        conn.request('GET', '/check', headers={'Cookie': owner_cookie,
                     'X-Owner-Access': '0', 'X-Real-IP': '192.0.2.1', 'Host': '127.0.0.1'})
        response = conn.getresponse()
        return response.status == 200
    except (OSError, httpc.HTTPException):
        return False
    finally:
        conn.close()


def serve(a):
    C = Coordinator(a)
    src = open(os.path.abspath(__file__), 'rb').read()
    block_mb = 4 * sum(p.numel() for p in C.model.blocks[0].parameters()) / 1e6

    class H(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.settimeout(60)

        def _reject(self, code, message):
            # Do not reuse connections with unread or invalid request bodies.
            self.close_connection = True
            return self._send(code, dict(ok=False, error=message))

        def _send(self, code, body, ctype='application/json'):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, indent=1).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *x):
            pass

        def _base(self):
            proto = self.headers.get('X-Forwarded-Proto') or 'http'
            host = self.headers.get('X-Forwarded-Host') or self.headers.get('Host') or f'{a.host}:{a.port}'
            prefix = (self.headers.get('X-Forwarded-Prefix') or '').rstrip('/')
            return f'{proto}://{host}{prefix}'

        def _client(self):
            peer, xff = self.client_address[0], self.headers.get('X-Forwarded-For')
            if xff and peer in (a.host, '127.0.0.1', '::1'):     # our own edge appends the real address last
                return client_key(xff.split(',')[-1].strip())
            return client_key(peer)

        def _read_small_json(self, limit=1 << 16):
            lengths = self.headers.get_all('Content-Length', [])
            if self.headers.get('Transfer-Encoding') or len(lengths) != 1:
                raise ValueError('unsupported body framing')
            n = int(lengths[0])
            if not 0 < n <= limit:
                raise ValueError('bad size')
            self.connection.settimeout(30)
            msg = json.loads(self.rfile.read(n))
            if not isinstance(msg, dict):
                raise ValueError('bad payload')
            return msg

        def _net_sample(self, q, owner=False):
            prompt = q.get('prompt', 'The ')[:200]
            sampler = q.get('sampler') or q.get('mode') or 'chain'
            if sampler not in ('chain', 'top'):
                return self._send(400, dict(error='sampler must be chain or top'))
            try:
                n, temp, seed = max(1, min(2048 if owner else 160, int(q.get('n', 64)))), float(q.get('temp', 0.0 if sampler == 'chain' else 0.75)), int(q.get('seed', 0)) % (2 ** 31)
                snapshot_id = int(q['snapshot']) if 'snapshot' in q else None
            except (ValueError, TypeError, OverflowError):
                return self._send(400, dict(error='invalid generation parameters or snapshot'))
            if not math.isfinite(temp):
                return self._send(400, dict(error='temp must be finite'))
            route = q.get('route', 'network')
            if route not in ('auto', 'single', 'distributed', 'network'):
                return self._send(400, dict(error='route must be auto, single, distributed or network'))
            if not C.net.gen_lock.acquire(blocking=False):
                return self._send(503, dict(error='a generation is already running; try again shortly'))
            g = None
            headers_started = False
            try:
                g = C.net.generate(prompt, n, temp=temp, seed=seed, sampler=sampler, route=route, snapshot_id=snapshot_id, budget_s=120 if owner else None)
                first = next(g)  # Validate and pin before HTTP success.
                self.close_connection = True
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Accel-Buffering', 'no')
                self.send_header('Connection', 'close')
                self.end_headers()
                headers_started = True
                self.wfile.write((json.dumps(first, separators=(',', ':')) + '\n').encode())
                self.wfile.flush()
                for ev in g:
                    self.wfile.write((json.dumps(ev, separators=(',', ':')) + '\n').encode())
                    self.wfile.flush()
            except OSError:
                pass                                    # the reader went away: stop generating
            except Exception as e:
                print(f'[net] generation failed: {e!r}', flush=True)
                if not headers_started:
                    return self._send(409, dict(error='requested route or snapshot unavailable'))
                try:
                    self.wfile.write((json.dumps(dict(t='error', error='generation failed')) + '\n').encode())
                except OSError:
                    pass
            finally:
                if g is not None:
                    g.close()
                C.net.gen_lock.release()

        def do_GET(self):
            try:
                u = urllib.parse.urlparse(self.path)
                p = u.path.rstrip('/').split('/')[-1]
                q = dict(urllib.parse.parse_qsl(u.query))
                if p == 'status':
                    return self._send(200, C.public_status())
                if p == 'code':
                    return self._send(200, src, 'text/x-python; charset=utf-8')
                if p in ('owner-inference', 'owner-sample'):
                    if self._client() != os.environ.get('DBMOE_OWNER_IP', '') or not owner_inference_session(self.headers.get('Cookie', '')):
                        return self._send(403, dict(error='existing owner sign-in required'))
                    if p == 'owner-inference':
                        return self._send(200, dict(owner=True, max_bytes=2048, budget_s=120, context_bytes=C.net.ctx_len))
                    return self._net_sample(q, owner=True)
                if p == 'sample':
                    if q.get('net'):
                        return self._net_sample(q)
                    if not C.sample_lock.acquire(blocking=False):
                        return self._send(503, dict(error='a sample is already being generated; try again shortly'))
                    try:
                        torch.set_num_threads(a.eval_threads)
                        choice, asked = C.generation_choice(), q.get('mode') or q.get('sampler')
                        mode = asked if asked in ('chain', 'top') else choice['mode']
                        temp = choice['temp'] if asked not in ('chain', 'top') else 0.0 if mode == 'chain' else 0.75
                        txt = generate(C.copy_model(), q.get('prompt', 'The ')[:200], n=max(1, min(240, int(q.get('n', 120)))),
                                       temp=float(q.get('temp', temp)), seed=int(q.get('seed', 0)), mode=mode, ctx_len=128)
                    finally:
                        C.sample_lock.release()
                    return self._send(200, txt, 'text/plain; charset=utf-8')
                if p == 'join':
                    if not C.enrol:
                        return self._send(404, dict(error='open join is off on this run'))
                    return self._send(200, C.enrol.challenge())
                if p == 'join.sh':
                    return self._send(200, JOIN_SH.replace('@BASE@', self._base()), 'text/x-shellscript; charset=utf-8')
                if p == 'leave.sh':
                    return self._send(200, LEAVE_SH.replace('@BASE@', self._base()), 'text/x-shellscript; charset=utf-8')
                if p == 'iblock':
                    who = C.auth(self.headers)
                    if not who:
                        code, why = C.auth_failure(self.headers)
                        return self._send(code, dict(error=why))
                    why = C.blocked(who)
                    if why:
                        return self._send(403, dict(error=why))
                    blob, why = C.net.block_blob(int(q.get('snap', 0)), int(q.get('block', -1)), who)
                    if blob is None:
                        return self._send(429 if why == 'slow down' else 404, dict(error=why))
                    return self._send(200, blob, 'application/octet-stream')
                if p == 'lease':
                    who = C.auth(self.headers)
                    if not who:
                        code, why = C.auth_failure(self.headers)
                        return self._send(code, dict(error=why))
                    why = C.blocked(who)
                    if why:
                        return self._send(403, dict(error=why))
                    name = f"{who}/{q.get('name', 'worker')[:40]}"
                    lease = C.lease(name, int(q.get('steps') or 0), int(q.get('batch') or 0), q.get('session', '')[:32], owner=who, client=self._client())
                    if q.get('fmt') == 'safetensors':           # opt-in, for workers without PyTorch
                        return self._send(200, lease_to_safetensors(lease), ST_LEASE_TYPE)
                    return self._send(200, to_bytes(lease), 'application/octet-stream')
                if p in ('', 'dbmoe', 'index.html'):
                    s = C.status()
                    ev = s['last_eval'] or {}
                    page = JOIN_PAGE.format(version=VERSION, params=s['params'], blocks=s['cfg']['blocks'], lpb=s['cfg']['layers_per_block'],
                                            experts=s['cfg']['experts'], d=s['cfg']['d'], ctx=s['cfg']['ctx'], data=s['data']['source'],
                                            mb=block_mb, updates=s['updates'], tokens=s['tokens'], active=len(s['active_leases']),
                                            bpb=ev.get('ar_bits_per_byte', 'n/a'), base=self._base())
                    return self._send(200, page, 'text/plain; charset=utf-8')
                return self._send(404, dict(error='not found', endpoints=['', 'status', 'sample', 'code', 'join', 'join.sh', 'lease', 'update', 'iblock', 'ipoll']))
            except LeaseCapacityError as e:
                return self._send(429, dict(error=str(e)))
            except Exception as e:
                return self._send(500, dict(error=repr(e)[:300]))

        def do_POST(self):
            slot = False
            try:
                p = urllib.parse.urlparse(self.path).path.rstrip('/').split('/')[-1]
                if p == 'join':
                    if not C.enrol:
                        return self._reject(404, 'open join is off on this run')
                    code, r = C.enrol.redeem(self._read_small_json(), self._client())
                    return self._send(code, r)
                # Authenticate and validate framing before accepting an upload.
                who = C.auth(self.headers)
                if not who:
                    return self._reject(*C.auth_failure(self.headers))
                if p == 'leave':
                    name = self._read_small_json().get('name')
                    if name is not None and not isinstance(name, str):
                        return self._reject(400, 'invalid worker name')
                    return self._send(200, C.leave(who, self.headers, name[:40] if name else None))
                why = C.blocked(who)
                if why:
                    return self._reject(403, why)
                if p == 'ipoll':
                    msg = self._read_small_json()
                    name = msg.get('name', 'worker')
                    if not isinstance(name, str):
                        return self._reject(400, 'invalid worker name')
                    return self._send(200, C.net.poll(who, f'{who}/{name[:40]}', msg, self.connection))
                if p != 'update':
                    return self._reject(404, 'not found')
                if not C.has_lease(who):
                    return self._reject(409, 'no active lease')
                ctype = (self.headers.get('Content-Type') or '').split(';')[0].strip().lower()
                if ctype != ST_UPDATE_TYPE and C.pickle_policy == 'reject':     # decided before any body byte is read
                    return self._reject(415, 'updates require safetensors; upgrade the worker (re-run the join script, '
                                             'or fetch /code)')
                lengths = self.headers.get_all('Content-Length', [])
                if self.headers.get('Transfer-Encoding') or len(lengths) != 1:
                    return self._reject(400, 'unsupported body framing')
                try:
                    n = int(lengths[0])
                except (TypeError, ValueError):
                    return self._reject(400, 'invalid content length')
                if not 0 < n <= a.max_upload_mb << 20:
                    return self._reject(413, 'bad size')
                slot = C.upload_slots.acquire(blocking=False)
                if not slot:
                    return self._reject(429, 'upload capacity reached')
                deadline = time.monotonic() + 60
                body = bytearray()
                while len(body) < n:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return self._reject(408, 'upload deadline exceeded')
                    self.connection.settimeout(remaining)
                    chunk = self.rfile.read(min(65536, n - len(body)))
                    if not chunk:
                        return self._reject(400, 'incomplete body')
                    body.extend(chunk)
                up = update_from_safetensors(body) if ctype == ST_UPDATE_TYPE else load_update_pickle(body, a.max_upload_mb << 20)
                if not isinstance(up, dict):
                    return self._reject(400, 'bad payload')
                C.note_update_format(ctype == ST_UPDATE_TYPE)
                name = up.get('name', 'worker')
                if not isinstance(name, str):
                    return self._reject(400, 'invalid worker name')
                code, r = C.update(f"{who}/{name[:40]}", up)
                return self._send(code, r)
            except (TimeoutError, socket.timeout):
                return self._reject(408, 'upload deadline exceeded')
            except Exception as e:
                return self._reject(400, repr(e)[:300])
            finally:
                if slot:
                    C.upload_slots.release()
                self.connection.settimeout(60)

    srv = ThreadingHTTPServer((a.host, a.port), H)
    srv.daemon_threads = True

    def stop(signum, frame):                            # save the merged master before exiting (SIGTERM / Ctrl-C)
        print(f'[coord] signal {signum}: saving {C.path} and exiting', flush=True)
        try:
            C.save()
        finally:
            os._exit(0)
    import signal
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f'[coord] {VERSION} on {a.host}:{a.port}  dir {a.dir}  data {a.hf or a.text_file}  params '
          f'{sum(p.numel() for p in C.model.parameters()):,}  sigmas {[round(s, 3) for s in C.model.sigmas]}', flush=True)
    srv.serve_forever()


# ------------------------------------------------------------------------------------------------------------- worker
def http(method, url, token, body=None, timeout=900, content_type='application/octet-stream'):
    req = urllib.request.Request(url, data=body, method=method, headers={'Authorization': f'Bearer {token}',
                                                                          'Content-Type': content_type,
                                                                          'User-Agent': VERSION})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def portable_expert_counts(top_i, experts):
    """Fixed-size expert histogram without bincount, which some MPS builds lack."""
    return (top_i[:, None] == torch.arange(experts, device=top_i.device)[None, :]).sum(0)


def cpu_model_name():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.lower().startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    if platform.system() == 'Darwin':
        try:
            import subprocess
            return subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], timeout=5, text=True).strip()
        except Exception:
            pass
    return platform.processor()


def ram_gib():
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30
    except (AttributeError, OSError, ValueError):
        return None


def normalize_hardware(value):
    """Allowlist optional public, self-reported specs. Never forward arbitrary client metadata."""
    if not isinstance(value, dict) or type(value.get('schema')) is not int or value['schema'] != 1:
        return None
    if value.get('self_reported') is not True:
        return None
    if not isinstance(value.get('backend'), str) or value['backend'] not in ('cpu', 'cuda', 'rocm', 'xpu', 'mps'):
        return None
    out = dict(schema=1, self_reported=True, backend=value['backend'])
    for key in ('cpu_model', 'os', 'cpu_arch', 'torch_version', 'runtime_version', 'gpu_name'):
        text = value.get(key)
        if isinstance(text, str):
            out[key] = ''.join(c for c in text if c.isprintable()).strip()[:80]
    threads = value.get('cpu_threads')
    if type(threads) is int and 1 <= threads <= 4096:
        out['cpu_threads'] = threads
    cores = value.get('cpu_cores')
    if type(cores) is int and 1 <= cores <= 4096:
        out['cpu_cores'] = cores
    ram = value.get('ram_gib')
    if type(ram) in (int, float) and 0 < ram <= 1048576:
        out['ram_gib'] = round(float(ram), 1)
    memory = value.get('gpu_memory_gib')
    if type(memory) in (int, float) and 0 < memory <= 65536:
        out['gpu_memory_gib'] = round(float(memory), 1)
    kind = value.get('gpu_memory_kind')
    if isinstance(kind, str) and kind in ('shared', 'dedicated', 'unknown'):
        out['gpu_memory_kind'] = kind
    return out


def worker_hardware(device_info, threads):
    """Called only after --share-hardware opt-in. No hostnames, IPs, serials or filesystem paths."""
    backend = device_info['backend']
    report = dict(schema=1, self_reported=True, backend=backend, cpu_arch=platform.machine(), cpu_model=cpu_model_name(),
                  os=platform.system(), cpu_cores=os.cpu_count() or 1, ram_gib=ram_gib(),
                  cpu_threads=max(1, threads), torch_version=str(torch.__version__),
                  runtime_version=str(torch.version.hip if backend == 'rocm' else
                                      torch.version.cuda if backend == 'cuda' else
                                      (getattr(torch.version, 'xpu', None) or '') if backend == 'xpu' else
                                      platform.mac_ver()[0] if backend == 'mps' else ''))
    if backend in ('cuda', 'rocm'):
        report['gpu_name'] = device_info['gpu_name']
        report['gpu_memory_gib'] = torch.cuda.get_device_properties(device_info['torch_device']).total_memory / 2**30
    elif backend == 'xpu':
        report['gpu_name'] = device_info['gpu_name']
        props = torch.xpu.get_device_properties(device_info['torch_device'])
        report['gpu_memory_gib'] = props.total_memory / 2**30
        report['gpu_memory_kind'] = xpu_memory_kind(props)
    elif backend == 'mps':
        report['gpu_name'] = device_info.get('gpu_name', 'Metal (MPS)')
    return normalize_hardware(report)


def xpu_available():
    """Optional on older/CPU-only PyTorch builds; never require Intel packages for CPU workers."""
    backend = getattr(torch, 'xpu', None)
    try:
        return bool(backend is not None and backend.is_available())
    except Exception:
        return False


def mps_available():
    backend = getattr(torch.backends, 'mps', None)
    try:
        return bool(backend and backend.is_built() and backend.is_available())
    except Exception:
        return False


def resolve_worker_device(requested):
    """Resolve public device names without changing the CPU-default participation policy.

    ROCm PyTorch intentionally exposes AMD devices through torch.cuda / cuda:N.
    The public label identifies the real backend, including for legacy --device cuda.
    An explicit GPU request must never silently start a CPU worker.
    """
    spec = str(requested).strip().lower()
    if spec == 'auto':
        spec = 'cuda' if torch.cuda.is_available() else 'xpu' if xpu_available() else 'mps' if mps_available() else 'cpu'
    if spec == 'cpu':
        return dict(torch_device='cpu', device='cpu', backend='cpu')
    if spec in ('mps', 'mps:0'):
        backend = getattr(torch.backends, 'mps', None)
        if backend is None or not backend.is_built():
            raise RuntimeError('Apple MPS requested, but this PyTorch build has no MPS support. '
                               'Install compatible macOS PyTorch; see README.md. Use --device cpu explicitly for CPU training.')
        if not backend.is_available():
            raise RuntimeError('Apple MPS requested, but PyTorch cannot access an MPS GPU. '
                               'Check the supported Mac, macOS and PyTorch versions; see README.md. Use --device cpu explicitly for CPU training.')
        return dict(torch_device='mps', device='mps', backend='mps')
    kind, sep, index = spec.partition(':')
    if kind not in ('cuda', 'rocm', 'hip', 'xpu', 'intel') or (sep and (not index.isascii() or not index.isdigit())):
        raise ValueError('device must be cpu, auto, cuda[:N], rocm[:N], xpu[:N], or mps '
                         '(hip aliases rocm; intel aliases xpu)')
    if kind in ('xpu', 'intel'):
        if not xpu_available():
            raise RuntimeError('Intel XPU requested, but PyTorch cannot access an Intel GPU. '
                               'Install XPU-enabled PyTorch and a supported Intel GPU/OS/driver; see README.md. '
                               'Use --device cpu explicitly for CPU training.')
        count = torch.xpu.device_count()
        idx = int(index) if sep else torch.xpu.current_device()
        if not 0 <= idx < count:
            raise ValueError(f'Intel GPU index {idx} is out of range: PyTorch sees {count} XPU device(s)')
        return dict(torch_device=f'xpu:{idx}', device=f'xpu:{idx}', backend='xpu')
    hip = getattr(torch.version, 'hip', None)
    if kind in ('rocm', 'hip') and not hip:
        raise RuntimeError('AMD ROCm requested, but this PyTorch build has no HIP support. '
                           'Install ROCm-enabled PyTorch for your supported GPU/OS; see README.md. '
                           'Use --device cpu explicitly for CPU training.')
    if not torch.cuda.is_available():
        raise RuntimeError(f'{kind} requested, but PyTorch cannot access a GPU. Check the PyTorch build, '
                           'GPU/OS support, driver, and device permissions. Use --device cpu for CPU training.')
    count = torch.cuda.device_count()
    idx = int(index) if sep else torch.cuda.current_device()
    if idx >= count:
        raise ValueError(f'GPU index {idx} is out of range: PyTorch sees {count} device(s)')
    backend = 'rocm' if hip else 'cuda'
    return dict(torch_device=f'cuda:{idx}', device=f'{backend}:{idx}', backend=backend)


def device_smoke_test(device):
    """Train a tiny synthetic lease locally, then verify portable CPU deltas. No network or token required."""
    cfg = make_cfg(dict(d=16, heads=2, layers_per_block=1, blocks=2, experts=2, ffn=32, ctx=8))
    shared, block = Shared(cfg), Block(cfg)
    L = dict(cfg=cfg, block=0, block_version=0, sigmas=cfg_sigmas(cfg), lease='offline-device-check',
             shared=shared.state_dict(), block_state=block.state_dict(), steps=2, batch=1, lr=1e-3,
             data=torch.arange(18, dtype=torch.uint8).reshape(2, 9))
    up = from_bytes(to_bytes(train_lease(L, device)))
    if not all(math.isfinite(v) for v in up['losses']):
        raise RuntimeError('device check produced non-finite losses')
    for delta, ref in ((up['d_shared'], L['shared']), (up['d_block'], L['block_state'])):
        if set(delta) != set(ref):
            raise RuntimeError('device check returned incompatible parameter keys')
        for k, t in delta.items():
            if t.device.type != 'cpu' or t.dtype != torch.float32 or t.shape != ref[k].shape:
                raise RuntimeError(f'device check returned an incompatible delta: {k}')
            if not bool(torch.isfinite(t).all()) or float(t.abs().max()) > 3.5 * L['lr'] * L['steps'] + 1e-6:
                raise RuntimeError(f'device check returned an invalid delta: {k}')
    return up['losses']


def xpu_memory_kind(properties):
    """Use PyTorch's explicit flag; never guess integrated graphics from a GPU name."""
    integrated = getattr(properties, 'is_integrated_gpu', None)
    return ('shared' if integrated else 'dedicated') if type(integrated) is bool else 'unknown'


def configure_xpu_memory(info, fraction):
    """Optional process-local allocator cap, not a reservation or a cap on total host RAM."""
    if fraction is None:
        return
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('--xpu-memory-fraction must be finite, greater than 0 and at most 1')
    if info['backend'] != 'xpu':
        raise ValueError('--xpu-memory-fraction requires an Intel XPU device; use --device xpu[:N]')
    setter = getattr(torch.xpu, 'set_per_process_memory_fraction', None)
    if not callable(setter):
        raise RuntimeError('This PyTorch XPU build cannot enforce --xpu-memory-fraction. '
                           'Install a compatible current XPU build; the requested cap was not applied.')
    device = info['torch_device']
    try:
        total = torch.xpu.get_device_properties(device).total_memory
        if type(total) not in (int, float) or not math.isfinite(total) or total <= 0:
            raise RuntimeError('PyTorch did not report a valid total device memory size')
        setter(float(fraction), device=device)
    except Exception as e:
        raise RuntimeError(f'Cannot apply the requested XPU memory cap on {device}: {e}') from e
    info['xpu_memory_fraction'] = float(fraction)
    info['xpu_allocator_limit_gib'] = round(total * fraction / 2**30, 3)


def prepare_worker_device(requested, check_cpu=False, xpu_memory_fraction=None):
    info = resolve_worker_device(requested)
    device = info['torch_device']
    if info['backend'] in ('cuda', 'rocm'):
        torch.cuda.set_device(device)
        info['gpu_name'] = torch.cuda.get_device_name(device)
    elif info['backend'] == 'xpu':
        torch.xpu.set_device(device)
        info['gpu_name'] = torch.xpu.get_device_name(device)
        info['gpu_memory_kind'] = xpu_memory_kind(torch.xpu.get_device_properties(device))
    elif info['backend'] == 'mps':
        info['gpu_name'] = 'Metal (MPS)'
    configure_xpu_memory(info, xpu_memory_fraction)
    if check_cpu or info['backend'] != 'cpu':
        try:
            info['check_losses'] = device_smoke_test(device)
        except Exception as e:
            raise RuntimeError(f'{info["device"]} failed the offline training check: {e}. '
                               'Check your GPU/driver/PyTorch support before joining; see README.md.') from e
    return info


def check_device_cmd(a):
    torch.set_num_threads(max(1, a.threads))
    try:
        info = prepare_worker_device(a.device, check_cpu=True,
                                     xpu_memory_fraction=getattr(a, 'xpu_memory_fraction', None))
    except (ValueError, RuntimeError) as e:
        raise SystemExit(f'[check-device] {e}') from e
    print(json.dumps(dict(ok=True, torch_version=torch.__version__, hip_version=torch.version.hip,
                          cuda_version=torch.version.cuda, xpu_version=getattr(torch.version, 'xpu', None), **info), indent=2))


def report_version(value):
    """Only a numeric release/prerelease, never a local build suffix, path or arbitrary text."""
    match = re.match(r'^(\d+\.\d+(?:\.\d+)?(?:(?:a|b|rc|dev)\d+)?)', str(value))
    return match.group(1)[:32] if match else 'unknown'


def report_device(value):
    spec = str(value).strip().lower()
    return spec if re.fullmatch(r'(?:cpu|auto|mps(?::0)?|(?:cuda|rocm|hip|xpu|intel)(?::[0-9]{1,8})?)', spec) else 'invalid'


def report_hardware(info):
    """Explicit opt-in model/capacity only. No arbitrary property or diagnostic dumps."""
    backend, device = info['backend'], info['torch_device']
    if backend == 'cpu':
        return dict(gpu='none')
    if backend == 'mps':
        return dict(gpu_model='Metal (MPS)', memory_kind='shared_or_unknown')
    api = torch.xpu if backend == 'xpu' else torch.cuda
    try:
        name = api.get_device_name(device)
        props = api.get_device_properties(device)
        # Driver model strings are reviewed by the volunteer before publication.
        # Reject multiline/path/email-shaped values instead of partially copying them.
        model = name if isinstance(name, str) and len(name) <= 80 and re.fullmatch(r'[A-Za-z0-9 ()+.,_-]+', name) else 'withheld'
        out = dict(gpu_model=model, memory_kind=xpu_memory_kind(props) if backend == 'xpu' else 'dedicated_or_unknown')
        total = props.total_memory
        if type(total) in (int, float) and math.isfinite(total) and 0 < total <= 65536 * 2**30:
            out['total_device_memory_gib'] = round(total / 2**30, 1)
        return out
    except Exception:
        return dict(available=False)


def hardware_report(requested, threads=2, include_hardware=False, xpu_memory_fraction=None):
    """Run one offline check and build a local, reviewable report. Never send it anywhere."""
    os_name, arch = platform.system(), platform.machine().lower()
    report = dict(schema='dbmoe.hardware-report.v1', report_id=secrets.token_hex(6),
                  generated_at_utc=time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime()),
                  dbmoe_version=VERSION, test_scope='tiny_synthetic_offline',
                  self_reported=True, independently_verified=False,
                  full_model_capacity_tested=False, network_training_tested=False,
                  requested_device=report_device(requested),
                  environment=dict(os_family=os_name if os_name in ('Linux', 'Windows', 'Darwin') else 'other',
                                   architecture=arch if arch in ('x86_64', 'amd64', 'aarch64', 'arm64', 'x86', 'i386',
                                                               'i686', 'armv7l', 'ppc64le', 'riscv64') else 'other',
                                   torch_version=report_version(torch.__version__)),
                  hardware_opt_in=bool(include_hardware), result='failed', stage='device_selection')
    try:
        with open(__file__, 'rb') as source:
            report['source_sha256'] = hashlib.sha256(source.read()).hexdigest()
    except OSError:
        report['source_sha256'] = 'unavailable'
    selected = None
    try:
        if type(threads) is not int or not 1 <= threads <= 4096:
            raise ValueError('invalid thread count')
        torch.set_num_threads(threads)
        selected = resolve_worker_device(requested)
        report['selected_device'] = selected['device']
        report['backend'] = selected['backend']
        report['stage'] = 'offline_preflight'
        runtime = getattr(torch.version, 'hip' if selected['backend'] == 'rocm' else
                          'cuda' if selected['backend'] == 'cuda' else 'xpu', None)
        if selected['backend'] in ('cuda', 'rocm', 'xpu'):
            report['environment']['runtime_version'] = report_version(runtime)
        info = prepare_worker_device(selected['torch_device'], check_cpu=True,
                                     xpu_memory_fraction=xpu_memory_fraction)
        report.update(result='passed', stage='complete', checks=['forward_backward', 'adamw', 'bounded_cpu_deltas', 'safe_serialization'])
        if 'xpu_memory_fraction' in info:
            report['xpu_memory_fraction'] = info['xpu_memory_fraction']
            if include_hardware:
                report['xpu_allocator_limit_gib'] = info['xpu_allocator_limit_gib']
    except ValueError:
        report['failure_category'] = 'invalid_device_or_options'
    except RuntimeError:
        report['failure_category'] = 'backend_unavailable' if report['stage'] == 'device_selection' else 'runtime_or_training_failure'
    except Exception:
        report['failure_category'] = 'unexpected_error'
    if include_hardware and selected is not None:
        report['hardware'] = report_hardware(selected)
    return report


def hardware_report_cmd(a):
    report = hardware_report(a.device, threads=a.threads, include_hardware=a.include_hardware,
                             xpu_memory_fraction=a.xpu_memory_fraction)
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if a.output == '-':
        print(payload, end='')
    else:
        try:
            with open(os.path.expanduser(a.output), 'x', encoding='utf-8') as destination:
                destination.write(payload)
        except OSError:
            raise SystemExit('[hardware-report] Could not create output; choose an unused writable filename.') from None
        print('[hardware-report] Saved locally. Review before sharing; nothing was uploaded.')
    if report['result'] != 'passed':
        raise SystemExit(1)


def token_path(base):
    return os.path.join(os.path.expanduser('~'), '.dbmoe', hashlib.sha256(base.encode()).hexdigest()[:12] + '.token')


def solve_challenge(ch):
    h, limit, n = hashlib.sha256(f"{ch['challenge']}.{ch['sig']}.".encode()), 1 << (256 - int(ch['bits'])), 0
    while True:
        m = h.copy()
        m.update(str(n).encode())
        if int.from_bytes(m.digest(), 'big') < limit:
            return str(n)
        n += 1


def mint(base):
    ch = json.loads(http('GET', f'{base}/join', '', timeout=60))
    body = json.dumps(dict(challenge=ch['challenge'], sig=ch['sig'], nonce=solve_challenge(ch))).encode()
    return json.loads(http('POST', f'{base}/join', '', body, timeout=60))


def enrol_cmd(a):
    """Optional: mint your own token and keep it in a file.  The worker does this by itself if you skip this."""
    base, path = a.url.rstrip('/'), os.path.expanduser(a.token_file)
    if os.path.exists(path):
        sys.exit(f'[enrol] {path} already exists; use it or move it aside before minting another')
    try:
        r = mint(base)
    except urllib.error.HTTPError as e:
        sys.exit(f"[enrol] HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(r['token'] + '\n')
    print(f"[enrol] you are {r['participant']}; token saved in {path} (mode 0600, not printed)", flush=True)
    print(f'[enrol] start with: python3 dbmoe.py worker --url {base} --token @{path}', flush=True)


def leave_cmd(a):
    """Leave for good: your workers come off the public list and your credential stops working.  What you already
    contributed stays in the model, because it cannot be separated out.  You can rejoin at any time."""
    base, path = a.url.rstrip('/'), None
    if a.token:
        token = open(os.path.expanduser(a.token[1:])).read().strip() if a.token.startswith('@') else a.token
    else:
        path = token_path(base)
        if not os.path.exists(path):
            sys.exit(f'[leave] no saved credential at {path}; if you minted your own, use --token @worker.token')
        with open(path) as f:
            token = f.read().strip()
    try:
        r = json.loads(http('POST', f'{base}/leave', token, json.dumps(dict(name=a.name) if a.name else {}).encode(), timeout=60))
    except urllib.error.HTTPError as e:
        if e.code != 410:
            sys.exit(f"[leave] HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}")
        r = dict(removed=0, credential_ended=True)
    if path and not a.name:
        os.remove(path)
        try:
            os.remove(os.path.join(os.path.dirname(path), 'worker-name'))
        except OSError:
            pass
    print(f"[leave] {r['removed']} worker record(s) removed from the public list" + ('; credential ended' if r['credential_ended'] else ''), flush=True)
    print('[leave] Any worker still running with this credential stops by itself. Your updates so far stay in the model.', flush=True)


def stable_worker_name():
    """A random label kept on this machine, so a restart keeps the same card on the list.  Never the hostname."""
    path = os.path.join(os.path.expanduser('~'), '.dbmoe', 'worker-name')
    try:
        with open(path) as f:
            name = f.read().strip()
        if name:
            return name[:40]
    except OSError:
        pass
    name = 'worker-' + secrets.token_hex(3)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(name)
    except OSError:
        pass
    return name


def load_or_join(base, force=False):
    """Our own credential: the saved one, or a new one.  No person is involved."""
    path = token_path(base)
    if not force and os.path.exists(path):
        with open(path) as f:
            tok = f.read().strip()
        if tok:
            return tok
    print('[worker] getting a credential (a moment of proof of work; nobody has to approve it)', flush=True)
    r = mint(base)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(r['token'])
    print(f"[worker] joined as {r['participant']}; credential saved in {path}", flush=True)
    return r['token']


class Keep:
    """One persistent HTTP(S) connection, so a hop does not pay for a new TLS handshake."""
    def __init__(self, base, a):
        u = urllib.parse.urlparse(base)
        self.cls = httpc.HTTPSConnection if u.scheme == 'https' else httpc.HTTPConnection
        self.host, self.prefix, self.a, self.c = u.netloc, u.path.rstrip('/'), a, None

    def close(self):
        if self.c:
            self.c.close()
        self.c = None

    def post(self, path, obj, timeout=60):
        body, url = json.dumps(obj).encode(), f'{self.prefix}/{path}'
        headers = {'Authorization': f'Bearer {self.a.token}', 'Content-Type': 'application/json', 'User-Agent': VERSION}
        for attempt in (0, 1):
            try:
                if self.c is None:
                    self.c = self.cls(self.host, timeout=timeout)
                self.c.request('POST', url, body, headers)
                r = self.c.getresponse()
                data = r.read()
                if r.status != 200:
                    raise urllib.error.HTTPError(url, r.status, data[:200].decode('utf-8', 'replace'), r.headers, None)
                return json.loads(data)
            except (httpc.HTTPException, OSError) as e:
                self.close()
                if attempt or isinstance(e, urllib.error.HTTPError):
                    raise


def load_inference_block(payload, assignment):
    """Validate the pinned placement before replacing a worker's current block.

    This checks assignment consistency, not authenticity: transport security and
    coordinator result audits remain necessary for untrusted participants.
    """
    if not isinstance(payload, dict) or not isinstance(assignment, dict):
        raise ValueError('invalid inference assignment or payload')
    for key in ('snap', 'block'):
        if (type(assignment.get(key)) is not int or type(payload.get(key)) is not int
                or payload[key] != assignment[key]):
            raise ValueError('inference payload does not match assigned snapshot and block')
    cfg = payload['cfg']
    if (not isinstance(cfg, dict) or type(cfg.get('blocks')) is not int
            or not 0 <= payload['block'] < cfg['blocks']):
        raise ValueError('inference block outside model configuration')
    sh, bl = Shared(cfg), Block(cfg)
    sh.load_state_dict(payload['shared'])
    bl.load_state_dict(payload['block_state'])
    return dict(snap=payload['snap'], block=payload['block'], shared=sh.eval(), module=bl.eval(), d=cfg['d'])


def serve_job(held, job):
    t0 = time.time()
    try:
        if job['block'] != held['block'] or job['snap'] != held['snap']:
            raise ValueError('job is not for the block this worker holds')
        zl, lg = run_segment(held['shared'], held['module'], base64.b64decode(job['ids']), int(job['seed']),
                             decode_floats(job['z'], held['d']).view(1, 1, -1), [float(x) for x in job['grid']], bool(job['final']),
                             float(job.get('exit_conf') or 0.0))
        out = dict(id=job['id'], z=encode_floats(zl), ms=round((time.time() - t0) * 1000, 1))
        if lg is not None:
            out['logits'] = encode_floats(lg)
        return out
    except Exception as e:
        return dict(id=job.get('id'), error=repr(e)[:200])


def serve_hops(a, base):
    """Every worker's second duty, beside training: hold one block of the current snapshot and run its part of other
    people's generations on request.  Long polling works from behind NAT; the prompt's bytes are visible to this worker."""
    conn, session, held, result, backoff, served = Keep(base, a), secrets.token_hex(6), None, None, 2, 0
    while True:
        try:
            r = conn.post('ipoll', dict(name=a.name, session=session, snap=held['snap'] if held else None,
                                        block=held['block'] if held else None, result=result))
            result, backoff = None, 2
            if r.get('assign'):
                asg = r['assign']
                L = from_bytes(http('GET', f"{base}/iblock?snap={int(asg['snap'])}&block={int(asg['block'])}", a.token, timeout=300))
                held = load_inference_block(L, asg)
                print(f"[serve] holding block {held['block']} of snapshot {held['snap']}", flush=True)
            elif r.get('job'):
                result = serve_job(held, r['job']) if held else dict(id=r['job'].get('id'), error='no block held')
                served += 1
                if served % 500 == 0:
                    print(f'[serve] {served} hops served', flush=True)
            elif r.get('retry_s'):
                time.sleep(min(600.0, float(r['retry_s'])))
        except urllib.error.HTTPError as e:
            result = None
            if e.code == 410:
                return
            print(f'[serve] HTTP {e.code}; retry shortly', flush=True)
            time.sleep(300 if e.code in (401, 403, 404) else backoff)
            backoff = min(60, backoff * 2)
        except Exception as e:
            result = None
            print(f'[serve] error {e!r}; retry in {backoff}s', flush=True)
            time.sleep(backoff)
            backoff = min(60, backoff * 2)


def post_update(base, token, body, tries=4):
    """Training work is worth a few retries: a busy coordinator or a dropped connection should not throw it away.  A
    lease merges at most once, so a retry after a lost reply gets a clean refusal rather than merging twice."""
    for attempt in range(tries):
        try:
            return json.loads(http('POST', f'{base}/update', token, body, content_type=ST_UPDATE_TYPE))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 502, 503, 504) or attempt == tries - 1:
                raise
        except (urllib.error.URLError, OSError):
            if attempt == tries - 1:
                raise
        time.sleep(5 * 2 ** attempt + random.random() * 5)


def train_lease(L, device, opt_cache=None):
    """Train the leased block (and the shared params) for L['steps'] steps; return the deltas.  opt_cache (dict) keeps
    the AdamW state per block between leases, so a worker that gets the same block again continues its optimizer
    instead of restarting Adam (a fresh Adam's first steps are sign-SGD-sized jolts on a trained block)."""
    cfg, b = L['cfg'], L['block']
    lo, hi = block_range(cfg, L['sigmas'], b)
    shared, block = Shared(cfg), Block(cfg)
    shared.load_state_dict(L['shared'])
    block.load_state_dict(L['block_state'])
    shared.to(device).train()
    block.to(device).train()
    s0 = {k: v.detach().clone() for k, v in shared.state_dict().items()}
    b0 = {k: v.detach().clone() for k, v in block.state_dict().items()}
    params = list(shared.parameters()) + list(block.parameters())
    opt = torch.optim.AdamW([dict(params=[p for p in params if p.ndim >= 2], weight_decay=0.01),
                             dict(params=[p for p in params if p.ndim < 2], weight_decay=0.0)], lr=L['lr'], betas=(0.9, 0.95))
    cached = (opt_cache or {}).get(b)
    warm = 0 if cached and 0 <= L.get('block_version', 0) - cached[1] <= 4 else 10
    if not warm:
        try:
            opt.load_state_dict(cached[0])
            for st in opt.state.values():               # keep each parameter's scale, drop the momentum: it points where
                st['exp_avg'].zero_()                   # the old weights wanted to go, and others have moved them since
        except Exception:
            warm = 10
    data = L['data'].long().to(device)
    S, bs = L['steps'], L['batch']
    settle = min(8, S // 4) if warm else 0              # a new Adam first watches a few steps at lr 0, so its first real steps are not sign-sized jolts
    ces, t0 = [], time.perf_counter()
    for s in range(S):
        for g in opt.param_groups:
            g['lr'] = L['lr'] * (0.0 if s < settle else min(1.0, (s + 1 - settle) / warm) if warm else 1.0)
        x = data[s * bs:(s + 1) * bs]
        loss, ce, _ = dblock_loss(shared, block, lo, hi, x, cfg['aux'], sigma=sample_sigma(cfg, lo, hi, x.size(0), device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        ces.append(round(float(ce), 4))
    sec = time.perf_counter() - t0
    if opt_cache is not None:
        opt_cache.pop(b, None)
        opt_cache[b] = (opt.state_dict(), L.get('block_version', 0) + 1)
        while len(opt_cache) > 2:                          # keep the two most recent blocks
            opt_cache.pop(next(iter(opt_cache)))
    # Keep the reported 10ms resolution positive for tiny/fast leases; zero is invalid wire metadata.
    return dict(lease=L['lease'], block=b, losses=ces, seconds=max(0.01, round(sec, 2)), tokens=int(S * bs * cfg['ctx']),
                fresh_optimizer=bool(warm),
                d_shared={k: (v.detach().float().cpu() - s0[k].float().cpu()) for k, v in shared.state_dict().items()},
                d_block={k: (v.detach().float().cpu() - b0[k].float().cpu()) for k, v in block.state_dict().items()})


def worker(a):
    torch.set_num_threads(max(1, a.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        device_info = prepare_worker_device(a.device, xpu_memory_fraction=getattr(a, 'xpu_memory_fraction', None))
    except (ValueError, RuntimeError) as e:
        sys.exit(f'[worker] {e}')                       # before any network or credential, so a bad device costs nothing
    device = device_info['torch_device']
    base = a.url.rstrip('/')
    auto = not a.token
    if auto:                                            # nobody to ask: use or get our own credential
        a.token = load_or_join(base)
    elif a.token.startswith('@'):                       # --token @FILE keeps the token out of `ps` output
        a.token = open(os.path.expanduser(a.token[1:])).read().strip()
    a.name = a.name or stable_worker_name()
    hardware = None
    if not getattr(a, 'no_share_hardware', False):
        try:
            hardware = worker_hardware(device_info, a.threads)
        except Exception:       # a report is never worth stopping for
            pass
    print(f'[worker] {VERSION} {a.name} -> {base}  device={device} threads={a.threads}', flush=True)
    if hardware:
        print('[worker] shown on the public worker list: ' + ', '.join(f'{k}={v}' for k, v in hardware.items() if k not in ('schema', 'self_reported'))
              + '.  No hostname or address.  --no-share-hardware turns this off.', flush=True)
    if 'xpu_memory_fraction' in device_info:
        print(f'[worker] XPU allocator cap {device_info["xpu_allocator_limit_gib"]:.3f} GiB '
              '(not reserved; host RAM and runtime overhead are outside this cap)', flush=True)
    if getattr(a, 'serve_only', False):
        return serve_hops(a, base)
    if not getattr(a, 'no_serve', True):
        threading.Thread(target=serve_hops, args=(a, base), daemon=True).start()
    backoff, done, opt_cache = 5, 0, {}
    t_end = time.time() + a.hours * 3600 if a.hours else None
    q = urllib.parse.urlencode(dict(name=a.name, steps=a.steps or '', batch=a.batch or '', session=secrets.token_hex(6)))
    while t_end is None or time.time() < t_end:
        try:
            t0 = time.time()
            L = from_bytes(http('GET', f'{base}/lease?{q}', a.token))
            t1 = time.time()
            up = train_lease(L, device, opt_cache)
            up.update(name=a.name, device=device_info['device'])
            if hardware:
                up['hardware'] = hardware
            body = update_to_safetensors(up)
            t2 = time.time()
            r = post_update(base, a.token, body)
            tail = up['losses'][-5:]
            print(f"[worker] block {L['block']}{' (fresh Adam)' if up['fresh_optimizer'] else ''}  ce {up['losses'][0]:.3f} -> {sum(tail) / len(tail):.3f}  "
                  f"{up['tokens'] / max(up['seconds'], 1e-6):.0f} bytes/s  train {up['seconds']:.0f}s  "
                  f"net {t1 - t0 + time.time() - t2:.1f}s  -> {r}", flush=True)
            backoff, done = 5, done + 1
            if a.once and done >= a.once:
                return
        except urllib.error.HTTPError as e:
            if e.code == 410:
                print('[worker] this credential has left the run; stopping', flush=True)
                return
            print(f'[worker] HTTP {e.code} {e.read()[:300]!r}; retry in {backoff}s', flush=True)
            if e.code == 401 and auto:                  # the coordinator no longer knows this credential: get a new one
                try:
                    a.token = load_or_join(base, force=True)
                except Exception as e2:
                    print(f'[worker] could not join: {e2!r}', flush=True)
                    backoff = 300
            elif e.code == 401:
                backoff = 300
            elif e.code == 403:
                backoff = 900
            time.sleep(backoff)
            backoff = min(300, backoff * 2)
        except Exception as e:
            print(f'[worker] error {e!r}; retry in {backoff}s', flush=True)
            time.sleep(backoff)
            backoff = min(300, backoff * 2)


def sample_cmd(a):
    torch.set_num_threads(a.threads)
    m, _ = load_ckpt(a.ckpt)
    print(generate(m, a.prompt, n=a.n, temp=a.temp, seed=a.seed, mode=a.mode))


def eval_cmd(a):
    torch.set_num_threads(a.threads)
    m, ck = load_ckpt(a.ckpt)
    x = torch.load(a.heldout, weights_only=True)
    print(json.dumps(dict(updates=ck['meta']['updates'], **evaluate(m, x), **evaluate_chain(m, x))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest='cmd', required=True)
    c = sp.add_parser('coord')
    c.add_argument('--dir', required=True)
    c.add_argument('--host', default='127.0.0.1')
    c.add_argument('--port', type=int, default=8790)
    c.add_argument('--tokens', default=None, help='JSON {token: participant}; re-read on change. Omit only for local tests (no auth).')
    c.add_argument('--hf', default=None, help='Hugging Face dataset REPO:GLOB (parquet files, e.g. HuggingFaceFW/fineweb-edu:sample/10BT/*.parquet) or REPO[:CONFIG] (datasets library)')
    c.add_argument('--text-file', default=None)
    c.add_argument('--cfg', default=None, help='JSON overrides of the model config (new runs only)')
    c.add_argument('--steps', type=int, default=60)
    c.add_argument('--batch', type=int, default=8)
    c.add_argument('--max-steps', type=int, default=400)
    c.add_argument('--max-batch', type=int, default=64)
    c.add_argument('--lr', type=float, default=3e-4, help="the paper's rate for text")
    c.add_argument('--lease-ttl', type=float, default=1800)
    c.add_argument('--eval-every', type=int, default=24)
    c.add_argument('--snap-every', type=int, default=200)
    c.add_argument('--eval-threads', type=int, default=4)
    c.add_argument('--heldout', type=int, default=32)
    c.add_argument('--max-upload-mb', type=int, default=256)
    c.add_argument('--pickle-updates', choices=('guarded', 'reject'),
                   default=os.environ.get('DBMOE_PICKLE_UPDATES', 'guarded'),
                   help='torch.save (pickle) /update bodies from older Python workers: "guarded" accepts them only '
                        'through the bounded ZIP gate (load_update_pickle), "reject" refuses them with 415 before '
                        'reading the body. Switch to reject once /status update_formats shows no recent pickle uploads.')
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--no-open-join', dest='open_join', action='store_false', help='only tokens from --tokens may take part')
    c.add_argument('--join-bits', type=int, default=18, help='proof-of-work difficulty for a new credential')
    c.add_argument('--join-per-client-day', type=int, default=0, help='0: no artificial daily participant limit; capacity backpressure remains')
    c.add_argument('--join-per-day', type=int, default=0, help='0: no artificial daily participant limit')
    c.add_argument('--gate-windows', type=int, default=12)
    c.add_argument('--gate-tol', type=float, default=0.03, help='refuse an update that makes held-out loss this fraction worse')
    c.add_argument('--no-gate-all', dest='gate_all', action='store_false', help='test only open participants, not tokens.json ones')
    c.add_argument('--min-steps', type=int, default=10, help='shortest lease; shorter ones change too little to test')
    c.add_argument('--learn-margin', type=float, default=0.002, help='how much an update must lower the loss on text from its own lease (fraction)')
    c.add_argument('--net-refresh', type=float, default=600, help='seconds between inference snapshots')
    t = sp.add_parser('enrol', aliases=['token'], help='optional: mint your own token now and save it privately (the worker also does this by itself)')
    t.add_argument('--url', default=DEFAULT_URL)
    t.add_argument('--token-file', default='worker.token')
    l = sp.add_parser('leave', help='leave for good: take your workers off the public list and end your credential')
    l.add_argument('--url', default=DEFAULT_URL)
    l.add_argument('--token', default=None, help='bearer token or @FILE, if not the saved one')
    l.add_argument('--name', default=None, help='remove only this worker label and keep your credential')
    w = sp.add_parser('worker')
    w.add_argument('--url', default=DEFAULT_URL)
    w.add_argument('--token', default=None, help='optional bearer token or @FILE; without one the worker enrols itself')
    w.add_argument('--name', default='', help='label for this worker (default: worker, or the hostname with --token)')
    w.add_argument('--no-share-hardware', action='store_true', help='keep CPU/GPU model, RAM, thread count and PyTorch version off the public list')
    w.add_argument('--share-hardware', action='store_true', help=argparse.SUPPRESS)    # the default now; accepted for older commands
    w.add_argument('--no-serve', action='store_true', help='train only: do not serve a block for network inference')
    w.add_argument('--serve-only', action='store_true', help='serve network inference only: no training')
    w.add_argument('--device', default='cpu', help='cpu (default), auto, cuda[:N], rocm[:N] for AMD, xpu[:N] for Intel, or mps for Mac; '
                   'explicit GPU requests fail if unavailable, rather than falling back to CPU')
    d = sp.add_parser('check-device', help='offline tiny training and serialization test; no token or network')
    d.add_argument('--device', default='cpu', help='cpu, auto, cuda[:N], rocm[:N], xpu[:N] (intel alias), or mps')
    d.add_argument('--threads', type=int, default=2)
    r = sp.add_parser('hardware-report', help='opt-in offline compatibility report; local output only, no upload')
    r.add_argument('--device', default='cpu', help='cpu, auto, cuda[:N], rocm[:N], xpu[:N], or mps')
    r.add_argument('--threads', type=int, default=2)
    r.add_argument('--output', default='-', help='new local JSON file, or - for stdout; existing files are never overwritten')
    r.add_argument('--include-hardware', action='store_true', help='include GPU model and total memory for your review; never uploads them')
    for parser in (w, d, r):
        parser.add_argument('--xpu-memory-fraction', type=float, default=None,
                            help='optional XPU allocator cap as a fraction of reported total device memory '
                                 '(0 < fraction <= 1); not free memory or a total host-RAM limit')
    w.add_argument('--threads', type=int, default=2)
    w.add_argument('--steps', type=int, default=0, help='steps per lease (0 = coordinator default)')
    w.add_argument('--batch', type=int, default=0, help='windows per step (0 = coordinator default)')
    w.add_argument('--hours', type=float, default=0, help='stop after this many hours (0 = run forever)')
    w.add_argument('--once', type=int, default=0, help='stop after this many leases (tests)')
    for name in ('sample', 'eval'):
        s = sp.add_parser(name)
        s.add_argument('--ckpt', required=True)
        s.add_argument('--threads', type=int, default=4)
        if name == 'sample':
            s.add_argument('--prompt', default='The ')
            s.add_argument('--n', type=int, default=200)
            s.add_argument('--temp', type=float, default=0.8)
            s.add_argument('--seed', type=int, default=0)
            s.add_argument('--mode', default='chain', choices=['chain', 'top'])
        else:
            s.add_argument('--heldout', required=True)
    a = p.parse_args()
    if a.cmd == 'worker' and not a.name:
        a.name = stable_worker_name()
    {'coord': serve, 'enrol': enrol_cmd, 'token': enrol_cmd, 'leave': leave_cmd, 'worker': worker, 'sample': sample_cmd, 'eval': eval_cmd,
     'check-device': check_device_cmd, 'hardware-report': hardware_report_cmd}[a.cmd](a)


if __name__ == '__main__':
    main()

