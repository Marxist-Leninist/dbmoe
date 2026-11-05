#!/usr/bin/env python3
"""dbmoe.py -- a small DiffusionBlocks-MoE byte-level language model trained across many machines.

OpenTransformers / AGILLM-4 lineage (block sigma partition, EDM preconditioning and loss weight ported from
AGILLM-4 dblocks_train.py).  One file; workers need only PyTorch (CPU or CUDA).

  coordinator  python dbmoe.py coord --dir RUN --host 127.0.0.1 --port 8790 --tokens tokens.json \
                   --hf "HuggingFaceFW/fineweb-edu:sample/10BT/*.parquet"
  worker       python dbmoe.py worker --url https://codex.opentransformers.online/dbmoe --token TOKEN --name NAME \
                   [--device cpu|cuda] [--threads 2] [--steps 40] [--batch 8]
  sample       python dbmoe.py sample --ckpt RUN/master.pt --prompt "The " [--n 200] [--mode chain|top]
  eval         python dbmoe.py eval --ckpt RUN/master.pt --heldout RUN/heldout.pt

MODEL.  Bytes (vocab 256) -> one shared embedding table (rows renormalised to RMS SD=0.5; also the tied output head)
plus learned positions, then B independent blocks.  Each block = L pre-LN causal transformer layers whose FFN is a
top-1 mixture of experts (Switch-style balance loss), a noise-level embedding and an output projection.

DIFFUSIONBLOCKS OBJECTIVE.  What is diffused is the embedding of the NEXT byte.  Block i owns the noise range
[s_i, s_{i+1}] (equal-probability slices of a log-normal noise distribution, AGILLM-4 _block_sigmas).  Training step
for block i: sigma ~ logU[s_i, s_{i+1}], z = e(x[t+1]) + sigma*eps; the block reads the CLEAN causal context
e(x[<=t]) + pos and c_in*z, outputs D = c_skip*z + c_out*F_i(.) (EDM preconditioning), and the loss is
w(sigma) * CE(head(D), x[t+1]) + 0.01 * balance.  A block never needs another block's activations or gradients,
so each machine trains a different block and ships back only that block plus a small delta of the shared
embedding / position / norm parameters.  (Unlike the EDM image defaults, the log-normal is re-centred on
sigma* = SD*sqrt(d/ln V), the noise level at which one byte's embedding stops being readable; with the image
defaults 5 of 6 blocks would sit where the target can simply be read off z.)

GENERATION chains the blocks as a diffusion sampler for the next byte's embedding: z ~ N(0, s_B^2), then blocks
B-1 ... 0 each take Euler steps of the EDM probability-flow ODE across their own range, with the x0 estimate
softmax(logits) @ table (CDCD score interpolation).  Already-known positions are kept at their true embedding plus
the current noise level, exactly as in training.  `--mode top` instead samples from the top block alone (one pass).

NETWORK.  HTTP + bearer token.  GET lease -> {lease, block, cfg, sigmas, shared, block_state, data, steps, batch, lr};
POST update <- {lease, block, d_block, d_shared, losses, tokens, seconds}.  Payloads are torch.save dicts of
tensors and plain values and are always loaded with weights_only=True (no pickle code execution).  Each update is
checked (keys, shapes, finite, |delta| <= 3.5*lr*steps, the Adam bound) before it is merged.  Merging is
asynchronous delta averaging: block += d_block / (leases that overlapped on that block), shared += d_shared /
(all overlapping leases).  New people join by running the worker with a token; tokens.json is re-read when it
changes.  GET status (JSON), GET sample?prompt=..&n=..&mode=chain|top (text), GET code (this file), GET / (join page).
"""
import argparse, io, json, math, os, random, secrets, socket, sys, threading, time, urllib.error, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn as nn
import torch.nn.functional as F

SD = 0.5
VERSION = 'dbmoe-v2-20261105'
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
        frac = torch.bincount(top_i, minlength=self.E).float() / max(1, top_i.numel())
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


def dblock_loss(shared, block, lo, hi, x, aux_w, sigma=None):
    """x: [batch, T+1] byte ids.  DiffusionBlocks loss for one block at sigma ~ logU[lo, hi] (or a given sigma)."""
    ids, tgt = x[:, :-1], x[:, 1:]
    if sigma is None:
        sigma = torch.exp(torch.empty(x.size(0), device=x.device).uniform_(math.log(lo), math.log(hi)))
    e = shared.embed(tgt)
    z = e + sigma[:, None, None] * torch.randn_like(e)
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


@torch.no_grad()
def next_byte_logits(model, ids, mode='chain', steps_per_block=2, gen=None):
    """ids: LongTensor [1, T] -> logits [vocab] for byte T+1."""
    sh, cfg, sig = model.shared, model.cfg, model.sigmas
    ctx, tab = sh.context(ids), sh.table()
    known = sh.embed(ids[:, 1:])                     # the next byte is known for positions 0..T-2
    ek = torch.randn(known.shape, generator=gen)
    zl = torch.randn(1, 1, cfg['d'], generator=gen) * sig[-1]

    def run(i, s, zl):
        D, _, _ = model.blocks[i](ctx, torch.cat([known + s * ek, zl], 1), torch.full((1,), float(s)))
        return sh.logits(D[:, -1:])

    if mode == 'top':
        return run(cfg['blocks'] - 1, sig[-1], zl)[0, -1]
    for i in reversed(range(cfg['blocks'])):
        lo, hi = sig[i], sig[i + 1]
        grid = [hi * (lo / hi) ** (k / steps_per_block) for k in range(steps_per_block + 1)]
        for a, b in zip(grid[:-1], grid[1:]):
            x0 = run(i, a, zl).softmax(-1) @ tab
            zl = zl + (b - a) * (zl - x0) / a
    return run(0, sig[0], zl)[0, -1]


@torch.no_grad()
def generate(model, prompt, n=200, temp=0.8, top_k=40, seed=0, mode='chain', ctx_len=None):
    gen = torch.Generator().manual_seed(seed)
    ids = encode(prompt) or [10]
    L = min(ctx_len or model.cfg['ctx'], model.cfg['ctx'])
    for _ in range(n):
        lg = next_byte_logits(model, torch.tensor([ids[-L:]]), mode=mode, gen=gen) / max(temp, 1e-4)
        if top_k:
            v, _ = lg.topk(min(top_k, lg.numel()))
            lg[lg < v[-1]] = -float('inf')
        ids.append(int(torch.multinomial(lg.softmax(-1), 1, generator=gen)))
    return decode(ids)


@torch.no_grad()
def evaluate(model, x):
    """x: [n, T+1] uint8/long held-out windows.  Per-block CE at each block's mid sigma, and the top block's CE at
    sigma_max (the next-byte prediction from context alone: a proper conditional distribution, reported as bits/byte)."""
    x = x.long()
    sig, B = model.sigmas, model.cfg['blocks']
    torch.manual_seed(0)
    per = []
    for i in range(B):
        s = torch.full((x.size(0),), math.sqrt(sig[i] * sig[i + 1]))
        per.append(round(float(dblock_loss(model.shared, model.blocks[i], 0, 0, x, 0, sigma=s)[1]), 4))
    _, ce, fr = dblock_loss(model.shared, model.blocks[-1], 0, 0, x, 0, sigma=torch.full((x.size(0),), sig[-1]))
    return dict(ar_ce=round(float(ce), 4), ar_bits_per_byte=round(float(ce) / LN2, 4), per_block_ce=per,
                top_block_expert_frac=[[round(float(v), 3) for v in f] for f in fr])


# ------------------------------------------------------------------------------------------------------------- helpers
def to_bytes(obj):
    b = io.BytesIO()
    torch.save(obj, b)
    return b.getvalue()


def from_bytes(b):
    return torch.load(io.BytesIO(b), map_location='cpu', weights_only=True)


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

Join (any computer with Python 3.9+ and PyTorch; a GPU is optional):
  python3 -m pip install torch            # CPU build is fine:  --index-url https://download.pytorch.org/whl/cpu
  curl -fsSL {base}/code -o dbmoe.py
  python3 dbmoe.py worker --url {base} --token YOUR_TOKEN --name YOUR_NAME [--device cuda] [--threads 4]
Ask the run owner for a token.  Live status: {base}/status     Sample text: {base}/sample?prompt=The%20&n=120
"""


class Coordinator:
    def __init__(self, a):
        self.dir = a.dir
        os.makedirs(os.path.join(self.dir, 'snapshots'), exist_ok=True)
        self.path = os.path.join(self.dir, 'master.pt')
        self.log_path = os.path.join(self.dir, 'events.jsonl')
        self.lock = threading.RLock()
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
        self.leases = {}
        self.data = TextSource(hf=a.hf, text_file=a.text_file, skip_docs=self.meta.get('data_docs', 0))
        hp = os.path.join(self.dir, 'heldout.pt')
        if os.path.exists(hp):
            self.heldout = torch.load(hp, weights_only=True)
        else:
            self.heldout = self.data.windows(a.heldout, self.model.cfg['ctx'])      # first windows of the stream
            torch.save(self.heldout, hp)
        self.last_save, self.eval_busy, self.eval_started, self.sample_lock = time.time(), False, 0, threading.Lock()
        self.save()

    # --- persistence / logging / auth
    def save(self, snapshot=False):
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
        return self.tokens.get(tok) if tok else None

    # --- leases
    def expire(self):
        t = time.time()
        for k in [k for k, v in self.leases.items() if t - v['t0'] > v['ttl']]:
            v = self.leases.pop(k)
            self.log(ev='lease_expired', lease=k, block=v['block'], worker=v['worker'])

    def lease(self, worker, steps, batch, session=''):
        steps = max(1, min(self.max_steps, steps or self.steps))
        batch = max(1, min(self.max_batch, batch or self.batch))
        data = self.data.windows(steps * batch, self.model.cfg['ctx'])      # before taking the lock (may wait on HF)
        with self.lock:
            self.expire()
            for k in [k for k, v in self.leases.items() if v['worker'] == worker and v['session'] == session]:
                self.log(ev='lease_abandoned', lease=k, block=self.leases.pop(k)['block'], worker=worker)
            B, bu = self.model.cfg['blocks'], self.meta['block_updates']
            active = [0] * B
            for v in self.leases.values():
                active[v['block']] += 1
            cand = [i for i in range(B) if active[i] == min(active)]
            w = self.meta['workers'].get(worker) or {}
            last, low = w.get('last_block'), min(bu[i] for i in cand)
            if last in cand and bu[last] <= low + 2:
                b = last                     # sticky: the worker keeps its optimizer state for this block
            else:
                b = min(cand, key=lambda i: (bu[i], random.random()))
            for v in self.leases.values():
                v['all'] += 1
                v['same'] += v['block'] == b
            wall = w.get('wall_s')
            ttl = self.ttl if not wall else max(300.0, min(self.ttl, 8 * wall * steps / max(1, w.get('steps', steps)) + 120))
            lid = secrets.token_hex(8)
            self.leases[lid] = dict(block=b, worker=worker, session=session, t0=time.time(), ttl=ttl, steps=steps, batch=batch,
                                    same=active[b] + 1, all=len(self.leases) + 1)
            payload = dict(version=VERSION, lease=lid, block=b, block_version=bu[b], cfg=self.model.cfg, sigmas=self.model.sigmas,
                           shared={k: v.clone() for k, v in self.model.shared.state_dict().items()},
                           block_state={k: v.clone() for k, v in self.model.blocks[b].state_dict().items()},
                           steps=steps, batch=batch, lr=self.lr, ttl=ttl, data=data)
        self.log(ev='lease', lease=lid, block=b, worker=worker, steps=steps, batch=batch, ttl=round(ttl))
        return payload

    def _check(self, up, L):
        lim = 3.5 * self.lr * L['steps'] + 1e-6
        b = L['block']
        for name, ref, d in (('block', self.model.blocks[b].state_dict(), up.get('d_block')),
                             ('shared', self.model.shared.state_dict(), up.get('d_shared'))):
            if not isinstance(d, dict) or set(d) != set(ref):
                return f'{name}: parameter names differ'
            for k, t in d.items():
                if not isinstance(t, torch.Tensor) or t.shape != ref[k].shape or not t.is_floating_point():
                    return f'{name}.{k}: wrong shape/type'
                if not bool(torch.isfinite(t).all()):
                    return f'{name}.{k}: not finite'
                m = float(t.abs().max()) if t.numel() else 0.0
                if m > lim:
                    return f'{name}.{k}: max|delta| {m:.3g} > {lim:.3g}'
        return None

    def update(self, worker, up):
        with self.lock:
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
                    p.add_(up['d_block'][k].to(p.dtype), alpha=1.0 / L['same'])
                for k, p in self.model.shared.state_dict().items():
                    p.add_(up['d_shared'][k].to(p.dtype), alpha=1.0 / L['all'])
            m = self.meta
            toks = int(L['steps'] * L['batch'] * self.model.cfg['ctx'])
            m['updates'] += 1
            m['tokens'] += toks
            m['block_updates'][b] += 1
            losses = [float(v) for v in (up.get('losses') or [])][:10000]
            lo = round(sum(losses[-5:]) / len(losses[-5:]), 4) if losses else None
            prev = m['block_ce'][b]
            m['block_ce'][b] = lo if prev is None or lo is None else round(0.7 * prev + 0.3 * lo, 4)
            sec = float(up.get('seconds') or 0)
            wall = time.time() - L['t0']
            w = m['workers'].setdefault(worker, dict(updates=0, tokens=0, seconds=0.0))
            w.update(updates=w['updates'] + 1, tokens=w['tokens'] + toks, seconds=round(w['seconds'] + sec, 1), last=now(),
                     last_block=b, host=str(up.get('host', ''))[:64], device=str(up.get('device', ''))[:32],
                     tok_s=round(toks / max(sec, 1e-6), 1), steps=L['steps'],
                     wall_s=round(wall if not w.get('wall_s') else 0.7 * w['wall_s'] + 0.3 * wall, 1))
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
        if do_save:
            self.save(snapshot=snap)
        if do_eval:
            threading.Thread(target=self.run_eval, daemon=True).start()
        return 200, dict(ok=True, updates=n)

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
            ev['sample'] = generate(m, 'The ', n=100, mode='chain', ctx_len=128)
            ev['eval_seconds'] = round(time.time() - t0, 1)
            with self.lock:
                self.meta['evals'] = (self.meta['evals'] + [ev])[-100:]
            self.log(ev='eval', **ev)
            print(f'[coord] eval @{n0}: AR {ev["ar_bits_per_byte"]} bits/byte  per-block CE {ev["per_block_ce"]}  {ev["sample"]!r}', flush=True)
        except Exception as e:
            print(f'[coord] eval failed: {e!r}', flush=True)
        finally:
            self.eval_busy = False

    def status(self):
        with self.lock:
            self.expire()
            m = self.meta
            return dict(version=VERSION, cfg=self.model.cfg, sigmas=[round(s, 4) for s in self.model.sigmas],
                        params=sum(p.numel() for p in self.model.parameters()), updates=m['updates'], tokens=m['tokens'],
                        rejected=m['rejected'], block_updates=m['block_updates'], block_ce=m['block_ce'],
                        active_leases=[dict(block=v['block'], worker=v['worker'], age_s=round(time.time() - v['t0']))
                                       for v in self.leases.values()],
                        workers=m['workers'], last_eval=(m['evals'][-1] if m['evals'] else None),
                        eval_history=[dict(updates=e['updates'], t=e.get('t'), ar_bpb=e['ar_bits_per_byte']) for e in m['evals']],
                        data=dict(source=m.get('data_source'), docs=self.data.docs, buffered_mb=round(len(self.data.buf) / 2**20, 1),
                                  error=self.data.err),
                        lease_defaults=dict(steps=self.steps, batch=self.batch, lr=self.lr, max_steps=self.max_steps,
                                            max_batch=self.max_batch, ttl_s=self.ttl))


def serve(a):
    C = Coordinator(a)
    src = open(os.path.abspath(__file__), 'rb').read()
    block_mb = 4 * sum(p.numel() for p in C.model.blocks[0].parameters()) / 1e6

    class H(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

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

        def do_GET(self):
            try:
                u = urllib.parse.urlparse(self.path)
                p = u.path.rstrip('/').split('/')[-1]
                q = dict(urllib.parse.parse_qsl(u.query))
                if p == 'status':
                    return self._send(200, C.status())
                if p == 'code':
                    return self._send(200, src, 'text/x-python; charset=utf-8')
                if p == 'sample':
                    if not C.sample_lock.acquire(blocking=False):
                        return self._send(503, dict(error='a sample is already being generated; try again shortly'))
                    try:
                        torch.set_num_threads(a.eval_threads)
                        txt = generate(C.copy_model(), q.get('prompt', 'The ')[:200], n=max(1, min(240, int(q.get('n', 120)))),
                                       temp=float(q.get('temp', 0.8)), seed=int(q.get('seed', 0)),
                                       mode='top' if q.get('mode') == 'top' else 'chain', ctx_len=128)
                    finally:
                        C.sample_lock.release()
                    return self._send(200, txt, 'text/plain; charset=utf-8')
                if p == 'lease':
                    who = C.auth(self.headers)
                    if not who:
                        return self._send(401, dict(error='missing or unknown token'))
                    name = f"{who}/{q.get('name', 'worker')[:40]}"
                    body = to_bytes(C.lease(name, int(q.get('steps') or 0), int(q.get('batch') or 0), q.get('session', '')[:32]))
                    return self._send(200, body, 'application/octet-stream')
                if p in ('', 'dbmoe', 'index.html'):
                    s = C.status()
                    ev = s['last_eval'] or {}
                    page = JOIN_PAGE.format(version=VERSION, params=s['params'], blocks=s['cfg']['blocks'], lpb=s['cfg']['layers_per_block'],
                                            experts=s['cfg']['experts'], d=s['cfg']['d'], ctx=s['cfg']['ctx'], data=s['data']['source'],
                                            mb=block_mb, updates=s['updates'], tokens=s['tokens'], active=len(s['active_leases']),
                                            bpb=ev.get('ar_bits_per_byte', 'n/a'), base=self._base())
                    return self._send(200, page, 'text/plain; charset=utf-8')
                return self._send(404, dict(error='not found', endpoints=['', 'status', 'sample', 'code', 'lease', 'update']))
            except Exception as e:
                return self._send(500, dict(error=repr(e)[:300]))

        def do_POST(self):
            try:
                p = urllib.parse.urlparse(self.path).path.rstrip('/').split('/')[-1]
                n = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(n) if 0 < n <= a.max_upload_mb << 20 else None   # always drain before replying
                who = C.auth(self.headers)
                if not who:
                    return self._send(401, dict(error='missing or unknown token'))
                if p != 'update':
                    return self._send(404, dict(error='not found'))
                if body is None:
                    return self._send(413, dict(error='bad size'))
                up = from_bytes(body)
                if not isinstance(up, dict):
                    return self._send(400, dict(error='bad payload'))
                name = f"{who}/{str(up.get('name', 'worker'))[:40]}"
                code, r = C.update(name, up)
                return self._send(code, r)
            except Exception as e:
                return self._send(400, dict(error=repr(e)[:300]))

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
def http(method, url, token, body=None, timeout=900):
    req = urllib.request.Request(url, data=body, method=method, headers={'Authorization': f'Bearer {token}',
                                                                          'Content-Type': 'application/octet-stream',
                                                                          'User-Agent': VERSION})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def train_lease(L, device, opt_cache=None):
    """Train the leased block (and the shared params) for L['steps'] steps; return the deltas.  opt_cache (dict) keeps
    the AdamW state per block between leases, so a worker that gets the same block again continues its optimizer
    instead of restarting Adam (a fresh Adam's first steps are sign-SGD-sized jolts on a trained block)."""
    cfg, b = L['cfg'], L['block']
    lo, hi = L['sigmas'][b], L['sigmas'][b + 1]
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
        except Exception:
            warm = 10
    data = L['data'].long().to(device)
    S, bs = L['steps'], L['batch']
    ces, t0 = [], time.time()
    for s in range(S):
        for g in opt.param_groups:
            g['lr'] = L['lr'] * (min(1.0, (s + 1) / warm) if warm else 1.0)
        loss, ce, _ = dblock_loss(shared, block, lo, hi, data[s * bs:(s + 1) * bs], cfg['aux'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        ces.append(round(float(ce), 4))
    sec = time.time() - t0
    if opt_cache is not None:
        opt_cache.pop(b, None)
        opt_cache[b] = (opt.state_dict(), L.get('block_version', 0) + 1)
        while len(opt_cache) > 2:                          # keep the two most recent blocks
            opt_cache.pop(next(iter(opt_cache)))
    return dict(lease=L['lease'], block=b, losses=ces, seconds=round(sec, 2), tokens=int(S * bs * cfg['ctx']),
                fresh_optimizer=bool(warm),
                d_shared={k: (v.detach().float().cpu() - s0[k].float().cpu()) for k, v in shared.state_dict().items()},
                d_block={k: (v.detach().float().cpu() - b0[k].float().cpu()) for k, v in block.state_dict().items()})


def worker(a):
    torch.set_num_threads(max(1, a.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    base = a.url.rstrip('/')
    if a.token.startswith('@'):                         # --token @FILE keeps the token out of `ps` output
        a.token = open(os.path.expanduser(a.token[1:])).read().strip()
    device = a.device if (a.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
    host = socket.gethostname()
    print(f'[worker] {VERSION} {a.name}@{host} -> {base}  device={device} threads={a.threads}', flush=True)
    backoff, done, opt_cache = 5, 0, {}
    t_end = time.time() + a.hours * 3600 if a.hours else None
    q = urllib.parse.urlencode(dict(name=a.name, steps=a.steps or '', batch=a.batch or '', session=secrets.token_hex(6)))
    while t_end is None or time.time() < t_end:
        try:
            t0 = time.time()
            L = from_bytes(http('GET', f'{base}/lease?{q}', a.token))
            t1 = time.time()
            up = train_lease(L, device, opt_cache)
            up.update(name=a.name, host=host, device=device)
            body = to_bytes(up)
            t2 = time.time()
            r = json.loads(http('POST', f'{base}/update', a.token, body))
            tail = up['losses'][-5:]
            print(f"[worker] block {L['block']}{' (fresh Adam)' if up['fresh_optimizer'] else ''}  ce {up['losses'][0]:.3f} -> {sum(tail) / len(tail):.3f}  "
                  f"{up['tokens'] / max(up['seconds'], 1e-6):.0f} bytes/s  train {up['seconds']:.0f}s  "
                  f"net {t1 - t0 + time.time() - t2:.1f}s  -> {r}", flush=True)
            backoff, done = 5, done + 1
            if a.once and done >= a.once:
                return
        except urllib.error.HTTPError as e:
            print(f'[worker] HTTP {e.code} {e.read()[:300]!r}; retry in {backoff}s', flush=True)
            if e.code == 401:
                backoff = 300
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
    print(json.dumps(dict(updates=ck['meta']['updates'], **evaluate(m, torch.load(a.heldout, weights_only=True)))))


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
    c.add_argument('--lr', type=float, default=1e-3)
    c.add_argument('--lease-ttl', type=float, default=1800)
    c.add_argument('--eval-every', type=int, default=24)
    c.add_argument('--snap-every', type=int, default=200)
    c.add_argument('--eval-threads', type=int, default=4)
    c.add_argument('--heldout', type=int, default=32)
    c.add_argument('--max-upload-mb', type=int, default=256)
    c.add_argument('--seed', type=int, default=0)
    w = sp.add_parser('worker')
    w.add_argument('--url', required=True)
    w.add_argument('--token', required=True, help='bearer token, or @FILE to read it from a file')
    w.add_argument('--name', default=socket.gethostname())
    w.add_argument('--device', default='cpu')
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
    dict(coord=serve, worker=worker, sample=sample_cmd, eval=eval_cmd)[a.cmd](a)


if __name__ == '__main__':
    main()
