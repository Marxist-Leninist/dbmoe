# DiffusionBlocks-MoE (DBMoE)

An experimental, byte-level language model trained asynchronously across participating CPU and CUDA machines. Each worker trains one noise-range block of a small mixture-of-experts model and returns bounded parameter deltas to a coordinator.

- Live progress: https://codex.opentransformers.online/dbmoe/progress
- Coordinator endpoint: https://codex.opentransformers.online/dbmoe
- Status: https://codex.opentransformers.online/dbmoe/status

This is an early research project, not a production-ready language model. No quality or benchmark claims are made. This repository contains code, not trained weights or training data.

## Contribute compute

Install Python 3.10+ and PyTorch 2.x in a virtual environment. For CPU-only machines:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA, use the PyTorch installation command appropriate for your GPU and driver from https://pytorch.org/get-started/locally/.

Ask the run owner for a participant token. Store it in a private local file called `worker.token`; do not commit it or share it. Choose a public pseudonym for your worker name, because participation statistics may be visible on the progress page.

```sh
python dbmoe.py worker \
  --url https://codex.opentransformers.online/dbmoe \
  --token @worker.token --name YOUR_PUBLIC_PSEUDONYM \
  --device cpu --threads 2 --hours 1
```

Use `--device cuda` for a supported GPU. `--hours 1` limits participation to approximately one hour, checked between leases. Workers use your machine's compute, electricity and network. Run only on machines you own or are authorized to use. Keep your token private and request revocation from the owner if exposed.

## How it works

- Byte vocabulary of 256 with tied embedding/output weights.
- Six blocks by default, each with two causal transformer layers and a top-1, four-expert feed-forward network.
- Width 256, four attention heads and context length 256 by default.
- Each block learns a separate diffusion-noise interval; workers need not exchange intermediate activations.
- Workers receive leases over HTTPS and return parameter deltas; the coordinator validates tensor keys, shapes, finite values and delta bounds before merging.
- Payload loading uses `torch.load(..., weights_only=True)`.

Validation is not a guarantee against malicious training updates. The current authenticated coordinator should be treated as experimental infrastructure.

## Local coordinator

The worker needs PyTorch. Streaming coordinator data additionally needs `huggingface_hub` and `pyarrow`, or `datasets` for dataset-library streaming. You can instead use a local text file you have permission to train on.

```sh
python dbmoe.py coord --dir run --host 127.0.0.1 \
  --text-file training.txt --tokens tokens.json
```

Create `tokens.json` privately as a JSON object mapping randomly generated participant tokens to participant labels. Never commit it. Do not expose an unauthenticated coordinator. Internet-facing deployments require TLS, access controls and operational hardening.

## Sample a checkpoint

```sh
python dbmoe.py sample --ckpt run/master.pt --prompt "The " --n 200
```

Only load checkpoints from trusted sources. No checkpoint is included in this initial source release.

## Attribution and licensing

The implementation identifies its lineage as OpenTransformers / AGILLM-4, including block-sigma partitioning and EDM preconditioning from `dblocks_train.py`. Related projects:

- https://github.com/Marxist-Leninist/AGILLM4-diffusionblocks
- https://github.com/Marxist-Leninist/DiffusionBlocks

The related DiffusionBlocks repository declares Apache-2.0. A license for this DBMoE source release has not yet been declared by its owner; public availability does not itself grant a general redistribution license. Dataset licenses and terms are separate and must be checked for any selected data source.
