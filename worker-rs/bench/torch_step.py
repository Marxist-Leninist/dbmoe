#!/usr/bin/env python3
"""Time dbmoe.py's training step (PyTorch) on the live model configuration, to compare with `dbmoe-worker bench`.

Same work per step as the worker: dblock_loss on [batch, ctx+1] random bytes, backward, clip_grad_norm_(1.0), and an
AdamW step.  Prints one JSON line: median and mean seconds per step and bytes per second (batch x ctx / step time).

    python worker-rs/bench/torch_step.py --threads 2
    python worker-rs/bench/torch_step.py --device cuda
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
import dbmoe  # noqa: E402

LIVE = dict(d=256, heads=4, layers_per_block=4, blocks=6, experts=4, ffn=528, ctx=256)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='cpu')
    p.add_argument('--cfg', default=None, help='JSON overrides of the live configuration')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(1)
    cfg = dbmoe.make_cfg(dict(LIVE, **json.loads(a.cfg or '{}')))
    dev = torch.device(a.device)
    shared, block = dbmoe.Shared(cfg).to(dev).train(), dbmoe.Block(cfg).to(dev).train()
    params = list(shared.parameters()) + list(block.parameters())
    opt = torch.optim.AdamW([dict(params=[q for q in params if q.ndim >= 2], weight_decay=0.01),
                             dict(params=[q for q in params if q.ndim < 2], weight_decay=0.0)], lr=1e-4, betas=(0.9, 0.95))
    times = []
    for s in range(a.warmup + a.steps):
        x = torch.randint(0, cfg['vocab'], (a.batch, cfg['ctx'] + 1), device=dev)
        sigma = (torch.rand(a.batch) * 3 + 0.5).to(dev)
        t0 = time.perf_counter()
        loss, ce, _ = dbmoe.dblock_loss(shared, block, 0.5, 3.5, x, cfg['aux'], sigma=sigma)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        float(loss)                                     # waits for the device
        if s >= a.warmup:
            times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    print(json.dumps(dict(worker=f'pytorch {torch.__version__}', device=a.device, threads=a.threads, batch=a.batch,
                          steps=a.steps, cfg=cfg, median_step_s=med, mean_step_s=statistics.mean(times),
                          bytes_per_s=a.batch * cfg['ctx'] / med)))


if __name__ == '__main__':
    main()
