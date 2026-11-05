#!/usr/bin/env python3
"""Generate an offline, deterministic training contract for a future native port.

This is a reference fixture generator, not an iOS training runtime or wire endpoint.
The output contains only synthetic tensors and has no token/network access.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest import mock

import torch


def tensor(value):
    value = value.detach().cpu()
    return dict(dtype=str(value.dtype).removeprefix('torch.'), shape=list(value.shape),
                data=value.reshape(-1).tolist())


def generate(module_path, skewed=False):
    module_path = Path(module_path).resolve()
    spec = importlib.util.spec_from_file_location('dbmoe_native_reference', module_path)
    db = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db)
    torch.set_num_threads(1)
    torch.manual_seed(20261105)
    cfg = db.make_cfg(dict(d=8, heads=2, layers_per_block=1, blocks=2, experts=2, ffn=12, ctx=4))
    shared, block = db.Shared(cfg), db.Block(cfg)
    # Nonzero attention projection ensures the fixture exercises Q/K/V gradients,
    # instead of testing only the default zero-projection initialization.
    with torch.no_grad():
        block.layers[0].proj.weight.normal_(std=.02)
    named = [('shared.' + k, p) for k, p in shared.named_parameters()]
    named += [('block.' + k, p) for k, p in block.named_parameters()]
    before = {k: p.detach().clone() for k, p in named}
    params = [p for _, p in named]
    batch = torch.tensor([[7, 65, 110, 116, 33]], dtype=torch.long)
    sigma = torch.tensor([.75], dtype=torch.float32)
    epsilon = torch.linspace(-1.2, 1.3, 32, dtype=torch.float32).reshape(1, 4, 8)
    sigmas = db.cfg_sigmas(cfg)
    if skewed:
        # Deterministically select a 3:1 routing case to exercise the otherwise
        # zero auxiliary-load gradient of the balanced 2:2 reference.
        router = block.layers[0].moe.router.weight
        for seed in range(64):
            with torch.no_grad():
                router.copy_(torch.randn(router.shape, generator=torch.Generator().manual_seed(seed)) * .2)
                with mock.patch.object(db.torch, 'randn_like', return_value=epsilon):
                    _, _, fractions = db.dblock_loss(shared, block, sigmas[0], sigmas[1], batch, cfg['aux'], sigma=sigma)
            if sorted(fractions[0].tolist()) == [.25, .75]:
                break
        else:
            raise RuntimeError('could not construct deterministic skewed routing fixture')
        before = {k: p.detach().clone() for k, p in named}
    lr = 1e-3
    optimizer = torch.optim.AdamW(
        [dict(params=[p for p in params if p.ndim >= 2], weight_decay=.01),
         dict(params=[p for p in params if p.ndim < 2], weight_decay=0.)],
        lr=lr / 10, betas=(.9, .95), eps=1e-8, foreach=False)
    with mock.patch.object(db.torch, 'randn_like', return_value=epsilon):
        loss, ce, fracs = db.dblock_loss(shared, block, sigmas[0], sigmas[1], batch, cfg['aux'], sigma=sigma)
    with torch.no_grad():
        target = batch[:, 1:]
        z = shared.embed(target) + sigma[:, None, None] * epsilon
        denoised, auxiliary, _ = block(shared.context(batch[:, :-1]), z, sigma)
        logits = shared.logits(denoised)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = {k: tensor(p.grad) if p.grad is not None else None for k, p in named}
    norm = torch.nn.utils.clip_grad_norm_(params, 1., foreach=False)
    clipped = {k: tensor(p.grad) if p.grad is not None else None for k, p in named}
    optimizer.step()
    return dict(
        format='dbmoe.native.parity.v1',
        fixture_case='skewed-routing' if skewed else 'balanced-routing',
        source_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
        torch_version=str(torch.__version__), seed=20261105, cfg=cfg,
        note='Synthetic numerical reference only; no mobile app, coordinator endpoint or device validation.',
        batch=tensor(batch), sigma=tensor(sigma), epsilon=tensor(epsilon),
        optimizer=dict(name='AdamW', base_lr=lr, lr=lr / 10, betas=[.9, .95], eps=1e-8,
                       weight_decay_matrix=.01, weight_decay_vector=0., clip_norm=1., step=1),
        initial={k: tensor(v) for k, v in before.items()},
        expected=dict(loss=float(loss.detach()), ce=float(ce), auxiliary=float(auxiliary),
                      expert_fractions=[tensor(v) for v in fracs], denoised=tensor(denoised),
                      logits=tensor(logits), gradient_norm_before_clip=float(norm),
                      gradients=gradients, clipped_gradients=clipped,
                      after_step={k: tensor(p) for k, p in named},
                      deltas={k: tensor(p.detach() - before[k]) for k, p in named}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, default=Path(__file__).resolve().parents[1] / 'dbmoe.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skewed', action='store_true', help='exercise unequal expert usage and its auxiliary gradient')
    args = parser.parse_args()
    result = generate(args.module, args.skewed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False, separators=(',', ':')) + '\n')
    print(json.dumps(dict(output=str(args.output), loss=result['expected']['loss'],
                          ce=result['expected']['ce'], parameters=len(result['initial']))))


if __name__ == '__main__':
    main()
