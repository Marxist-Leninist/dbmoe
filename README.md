# DiffusionBlocks-MoE (DBMoE)

An experimental, byte-level language model trained asynchronously across participating CPU, NVIDIA CUDA and AMD ROCm machines. Each worker trains one noise-range block of a small mixture-of-experts model and returns bounded parameter deltas to a coordinator.

- Live progress: https://codex.opentransformers.online/dbmoe/progress
- Coordinator endpoint: https://codex.opentransformers.online/dbmoe
- Status: https://codex.opentransformers.online/dbmoe/status

This is an early research project, not a production-ready language model. No quality or benchmark claims are made. This repository contains code, not trained weights or training data.

## Contribute compute

Install PyTorch 2.x and a Python version supported by your chosen PyTorch release in a virtual environment (Python 3.11+ is a good starting point for current releases). For CPU-only machines:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA, use the PyTorch installation command appropriate for your GPU and driver from https://pytorch.org/get-started/locally/.

Self-service enrolment is open: no owner approval is needed. On Linux or macOS, the one-command installer downloads the current worker, enrols automatically and starts a bounded one-hour run:

```sh
curl -fsSL https://codex.opentransformers.online/dbmoe/join.sh | sh -s -- --threads 2 --hours 1
```

For manual setup, download the exact currently deployed worker into a new directory. Optionally mint your own token immediately with `enrol`; it is saved privately in `worker.token` with mode `0600` and is not printed. Alternatively, omit `--token` from the worker command and it will enrol itself.

```sh
curl -fsSL https://codex.opentransformers.online/dbmoe/code -o dbmoe.py
# Optional: mint your own token now.
python dbmoe.py enrol --url https://codex.opentransformers.online/dbmoe
python dbmoe.py worker \
  --url https://codex.opentransformers.online/dbmoe \
  --token @worker.token --name YOUR_PUBLIC_PSEUDONYM \
  --device cpu --threads 2 --hours 1
```

New enrolment uses a short proof-of-work challenge and per-address limits; the coordinator stores only a hash of the credential. Keep the token private and request revocation if exposed. Choose a public pseudonym for your worker name, because participation statistics may be visible on the progress page. Workers use your machine's compute, electricity and network. Run only on machines you own or are authorized to use.

Use `--device cuda` for NVIDIA or `--device rocm` for a supported AMD GPU. The default remains CPU; `--device auto` explicitly opts in to selecting an available GPU, otherwise CPU. An explicit GPU request now fails early if unavailable instead of silently training on CPU. `--hours 1` limits participation to approximately one hour, checked between leases.

## AMD GPUs (ROCm)

The worker uses standard PyTorch FP32 operations, with no NVIDIA-only custom kernels, CUDA extensions, or NCCL requirement. CPU, NVIDIA and AMD workers exchange the same CPU-tensor update format with the coordinator. **AMD support is implemented but has not yet been validated on physical AMD hardware for this release.** Mocked backend tests are not GPU validation.

1. Check your exact GPU, OS, driver and supported framework combination in [AMD's current compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html). Not every Radeon, Ryzen integrated GPU or older AMD card supports ROCm. This project does not bypass that matrix or recommend architecture-spoofing overrides.
2. Use a supported Linux configuration for training. Follow [AMD's PyTorch installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html) for your GPU, or the [PyTorch installation selector](https://pytorch.org/get-started/locally/) with Linux and the compatible ROCm compute platform. Run its current installation command inside a fresh virtual environment. Install **ROCm-enabled torch**, not the CPU or NVIDIA CUDA wheel; installing requirements alone does not select the right GPU build. DBMoE itself only needs `torch`, not torchvision/torchaudio.
3. Run the local check before requesting a training lease:

```sh
python -c "import torch; print('torch:', torch.__version__, 'HIP:', torch.version.hip, 'GPU available:', torch.cuda.is_available())"
python dbmoe.py check-device --device rocm
```

Expect a nonempty HIP version, GPU availability `True`, and a successful `check-device` result. This runs a tiny synthetic DBMoE forward/backward/AdamW update and checks finite, bounded CPU deltas through safe serialization. It makes no network request, reads no token, and changes no training checkpoint. GPU workers also run this check automatically before their first lease. It does not establish full-size-model memory capacity or long-run stability.

Then join normally:

```sh
python dbmoe.py worker \
  --url https://codex.opentransformers.online/dbmoe \
  --token @worker.token --name YOUR_PUBLIC_PSEUDONYM \
  --device rocm --threads 2 --batch 1 --hours 1
```

Use `--device rocm:1` for the second visible GPU, or `--device cuda:1` on NVIDIA. Each worker uses one GPU; `--threads` controls PyTorch CPU threads. Start with batch 1 when available VRAM is uncertain. A coordinator can limit requested batch sizes, and full-size model weights plus optimizer state must still fit. Reduce the batch if training runs out of memory; a successful tiny check is not a guarantee that a live lease will fit.

[PyTorch deliberately uses its `torch.cuda` API and `cuda:N` tensor devices on ROCm](https://docs.pytorch.org/docs/stable/notes/hip.html). DBMoE's `rocm`/`hip` aliases map to that API and require a HIP-enabled build. Existing `--device cuda` commands still work on ROCm and report the backend honestly as `rocm:N`. A CUDA wheel does not add AMD support.

Native Windows and WSL have separate, hardware- and release-specific AMD support. Do not assume an inference-capable Windows package supports this training worker. DBMoE has not validated AMD training on native Windows, WSL, or macOS; use the documented Linux training stack, or `--device cpu`. This change does not add DirectML, Vulkan, or Apple Metal/MPS backends.

If the check fails, confirm the installed wheel, supported GPU/OS, driver and device visibility/permissions against AMD's guide. An AMD GPU name appearing in an OS utility is not enough: PyTorch must load its kernels and complete backward propagation. `--device rocm` never silently falls back to CPU.

## Optional public hardware specs

Add `--share-hardware` to opt in to publishing self-reported hardware details with your worker statistics: GPU model and rounded total GPU memory, CPU architecture, the number of PyTorch CPU threads assigned to this worker, PyTorch version, and CUDA/ROCm runtime version. Thread count is not the machine's physical core count; GPU memory is not free memory. These reports are supplied by participants, not independently verified inventory.

Without this flag, no detailed hardware report is sent. Device/backend and participation statistics are still sent as before. New workers do not transmit machine hostnames, IP fields, serial numbers or local paths in update payloads; the coordinator/network provider necessarily sees connection IPs. Choose an explicit public pseudonym, or omit `--name` for a random worker label. The coordinator strips legacy host/IP fields from public status and accepts only the documented hardware fields. Restart without the flag to withdraw the report on the next accepted update; prior logs/backups may still retain earlier reports. Older coordinators may ignore this optional metadata, and website display requires coordinator/page support.

## Tests

```sh
python -m unittest -v
python dbmoe.py check-device --device cpu
# On a supported AMD machine after installing ROCm PyTorch:
python dbmoe.py check-device --device rocm
```

The unit suite covers CPU training, optimizer-state reuse, coordinator delta validation, portable serialization, CPU/CUDA/ROCm device selection and privacy behavior. GPU selection is mocked where hardware is absent. The real-GPU test is explicitly skipped if PyTorch cannot access a GPU. CPU tests passed on PyTorch 2.6.0+cpu; actual AMD and NVIDIA hardware validation remains pending.

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

