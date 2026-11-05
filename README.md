# DiffusionBlocks-MoE (DBMoE)

An experimental, byte-level language model designed for asynchronous training, with CPU, NVIDIA CUDA, AMD ROCm and Apple Mac MPS worker paths. Each worker trains one noise-range block of a small mixture-of-experts model and returns bounded parameter deltas to a coordinator.

- Live progress: https://codex.opentransformers.online/dbmoe/progress
- Coordinator endpoint: https://codex.opentransformers.online/dbmoe
- Status: https://codex.opentransformers.online/dbmoe/status

This is an early research project, not a production-ready language model. No quality or benchmark claims are made. This repository contains code, not trained weights or training data.

## Contribute compute

Install PyTorch 2.x and a Python version supported by your chosen PyTorch release in a virtual environment (Python 3.11+ is a good starting point for current releases). For Linux/Windows CPU-only machines (Mac users: use the [macOS instructions](#apple-mac-macos) below):

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA, use the PyTorch installation command appropriate for your GPU and driver from https://pytorch.org/get-started/locally/.

Anonymous joining is open. No account, manually supplied token, or owner approval is needed. The worker obtains and stores its own private credential automatically.

On Linux or macOS, first download and review the installer. It installs the reviewed worker and CPU PyTorch if needed, then starts participating until you stop it:

```sh
curl -fsSL https://codex.opentransformers.online/dbmoe/join.sh -o join.sh
less join.sh
sh join.sh --threads 2
```

Or download the exact deployed worker and start it directly:

```sh
curl -fsSL https://codex.opentransformers.online/dbmoe/code -o dbmoe.py
python dbmoe.py worker --url https://codex.opentransformers.online/dbmoe --device cpu --threads 2
```

Use `py` and `curl.exe` on Windows where appropriate. Install a compatible PyTorch build first. Ctrl+C stops participation. You can choose your own session budget with `--hours 1`; it is optional, not a network-imposed cutoff.

Active anonymous access renews automatically. There is no daily participant count or lifetime job limit. Work can wait when compute or validation capacity is full. After long inactivity, the worker can rejoin automatically. Credentials are private and should never be posted, committed or copied into screenshots.

The worker trains and can serve an inference block. This uses your machine's compute, electricity and network; only join on machines you own or are authorized to use. Inference workers can see prompt bytes routed through their block, so do not send private prompts to a public volunteer network. `--no-serve` opts out of serving. Submitted training updates are checked before merging, but those checks do not guarantee language improvement or protection against every malicious update.

**Release synchronization:** the live worker at `/dbmoe/code` and this repository now carry the same v4 protocol: anonymous joining, lease/learning gates, bounded downloads, split block/shared-round validation, and measured `chain`/`top` sampling. CPU, NVIDIA CUDA, AMD ROCm and Apple MPS code paths are included. Physical AMD and Apple-hardware validation remains pending; no capability claim is made for those devices yet.

In this repository, `--device cpu` remains the default; `cuda` selects NVIDIA, `rocm` selects a supported AMD GPU, and `mps` selects a compatible Mac GPU. `auto` explicitly opts into CUDA/ROCm first, then MPS, otherwise CPU. Explicit GPU requests fail early when unavailable. Physical backend validation is a separate gate and remains pending on AMD and Apple hardware.


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

The AMD device check is integrated into the deployed anonymous worker. Physical AMD training validation is still required before claiming a tested AMD release; the mocked backend suite is not hardware proof.

Use `--device rocm:1` for the second visible GPU, or `--device cuda:1` on NVIDIA. Each worker uses one GPU; `--threads` controls PyTorch CPU threads. Start with batch 1 when available VRAM is uncertain. A coordinator can limit requested batch sizes, and full-size model weights plus optimizer state must still fit. Reduce the batch if training runs out of memory; a successful tiny check is not a guarantee that a live lease will fit.

[PyTorch deliberately uses its `torch.cuda` API and `cuda:N` tensor devices on ROCm](https://docs.pytorch.org/docs/stable/notes/hip.html). DBMoE's `rocm`/`hip` aliases map to that API and require a HIP-enabled build. Existing `--device cuda` commands still work on ROCm and report the backend honestly as `rocm:N`. A CUDA wheel does not add AMD support.

Native Windows and WSL have separate, hardware- and release-specific AMD support. Do not assume an inference-capable Windows package supports this training worker. DBMoE has not validated AMD training on native Windows, WSL, or macOS; use the documented Linux training stack, or `--device cpu`. This change does not add DirectML or Vulkan. Mac GPU participation uses the separate MPS path below, not ROCm.

If the check fails, confirm the installed wheel, supported GPU/OS, driver and device visibility/permissions against AMD's guide. An AMD GPU name appearing in an OS utility is not enough: PyTorch must load its kernels and complete backward propagation. `--device rocm` never silently falls back to CPU.

## Apple Mac (macOS)

MacBook Air/Pro, iMac, Mac mini, Mac Studio and Mac Pro can use the Python worker **when their OS, architecture and PyTorch build are compatible**. Apple silicon uses `--device mps` for Metal GPU training or `--device cpu`. Intel Macs require a compatible x86_64 PyTorch release and can use the CPU route; do not assume a current Apple-silicon wheel, an AMD GPU inside an older Mac, or the newest PyTorch release supports that machine.

**MPS worker support is implemented and regression-tested on Linux with mocked device selection. It has not yet been validated on physical Apple hardware.** The runtime check below must pass on the particular Mac before it joins. It tests a tiny synthetic model, not full-model memory capacity or long-running stability.

Use [Apple's PyTorch installation guide](https://developer.apple.com/metal/pytorch/) and the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for the selected release. Requirements change by release: Apple's guide currently lists Apple silicon and macOS 14+ for its documented stable build. Older releases have different requirements. Do not use the Linux CPU wheel index on macOS, install CUDA/ROCm packages on a Mac, or bypass unsupported-OS checks.

For the MPS checks, use the current `dbmoe.py` from this repository or `/code`; both include the MPS path and anonymous protocol. The offline check runs before any credential read or network request. Physical Apple-hardware validation remains pending; do not infer support from mocked tests alone.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch
python dbmoe.py check-device --device cpu
python dbmoe.py check-device --device mps
```

For an isolated coordinator running this same repository revision (see [Local coordinator](#local-coordinator)), a bounded Mac session looks like this. Use its privately configured participant token:

```sh
python dbmoe.py worker \
  --url http://127.0.0.1:8790 \
  --token @worker.token --name YOUR_PUBLIC_PSEUDONYM \
  --device mps --threads 2 --steps 2 --batch 1 --hours 0.25
```

To train on CPU, change `--device mps` to `--device cpu`. An explicit MPS request fails rather than silently selecting CPU. `mps:0` is accepted as the single MPS device; other MPS indexes are rejected. MPS workers automatically run the offline forward/backward/AdamW/CPU-delta check before reading a token or requesting a lease. The implementation avoids `bincount` on MPS releases without that operator. It does not enable PyTorch's CPU fallback environment switch or disable MPS memory safeguards.

Keep the machine ventilated, use external power when practical, start with batch 1, and stop with Ctrl-C if the machine gets hot or memory pressure increases. On Apple silicon the GPU shares system memory: the model, gradients, optimizer state and cached optimizers can consume much more memory than the downloaded weights. A tiny successful check does not prove a large lease fits. `--hours` is checked between leases, not a hard timer, so use short steps for a responsive stop. There is no battery/thermal monitor or automatic launch-at-login behavior in this CLI. Running it is an explicit choice to use compute, electricity and network.

With `--share-hardware`, MPS reports the backend, generic `Metal (MPS)` name, CPU architecture, worker thread count, PyTorch version and macOS version. It does not report a guessed chip model or pretend unified memory is dedicated/free VRAM.

## iPhone, iPad and other Apple products

**The macOS Python command is not an iOS/iPadOS app. This release does not ship a working native iPhone/iPad training app, signed IPA or App Store/TestFlight build.** Merely opening the progress page does not contribute compute. There is no claim of universal Apple-device training support.

A genuine native path is [MLX Swift](https://github.com/ml-explore/mlx-swift), whose official examples include an iOS/macOS training app. The verified [0.31.4 package targets](https://github.com/ml-explore/mlx-swift/blob/0.31.4/Package.swift) are iOS/iPadOS 17+, macOS 14+, tvOS 17+ and visionOS 1+. These dependency targets are not a DBMoE support or memory guarantee. DBMoE still needs a port of its actual causal transformer/MoE, EDM loss, gradients and AdamW, a language-neutral lease/delta adapter instead of Python `torch.save`, and numerical parity plus real-device tests. Core ML conversion or an inference-only runtime is not a training implementation. See [the native Apple implementation checklist](apple/README.md).

The initial native worker must be foreground-only, start only after an explicit compute/network consent, pause on backgrounding, low battery, Low Power Mode or thermal pressure, and offer a visible Stop action with short bounded jobs. [iOS/iPadOS 26 continued processing](https://developer.apple.com/documentation/backgroundtasks/bgcontinuedprocessingtask) can allow user-initiated finite work to continue in the background; [GPU access additionally needs its capability](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.background-tasks.continued-processing.gpu) and supported hardware/runtime checks. That is not permission or a guarantee for indefinite unattended training. No background entitlement is enabled by this repository.

Apple TV and Apple Vision Pro would need separate native UI/runtime/device validation. Apple Watch, HomePod, AirPods and accessories are not supported training workers; companion/control use would be a separate feature. No jailbreak, sideloading bypass, developer-account creation or paid Apple membership is required or performed by this change.

## Optional public hardware specs

With `--share-hardware` present in the deployed worker, add it to opt in to publishing self-reported hardware details with your worker statistics: GPU model and rounded total GPU memory, CPU architecture, the number of PyTorch CPU threads assigned to this worker, PyTorch version, and CUDA/ROCm runtime version (macOS version for MPS). MPS reports a generic Metal name and no VRAM figure. Thread count is not the machine's physical core count; GPU memory is not free memory. These reports are supplied by participants, not independently verified inventory.

Without this flag, no detailed hardware report is sent. Device/backend and participation statistics are still sent as before. New workers do not transmit machine hostnames, IP fields, serial numbers or local paths in update payloads; the coordinator/network provider necessarily sees connection IPs. Choose an explicit public pseudonym, or omit `--name` for a random worker label. The coordinator strips legacy host/IP fields from public status and accepts only the documented hardware fields. Restart without the flag to withdraw the report on the next accepted update; prior logs/backups may still retain earlier reports. Older coordinators may ignore this optional metadata, and website display requires coordinator/page support.

## Tests

```sh
python -m unittest -v
python dbmoe.py check-device --device cpu
# On a supported AMD machine after installing ROCm PyTorch:
python dbmoe.py check-device --device rocm
# On a compatible Mac after installing macOS PyTorch:
python dbmoe.py check-device --device mps
```

The current suite has 40 passing tests, including CPU training, optimizer-state reuse, coordinator delta validation, portable serialization, CPU/CUDA/ROCm/MPS device selection, privacy and Apple numerical fixtures. A separate 31-check attack/recovery suite plus a serve-only identity check passed: saturation and NaN rejection, honest updates, full-chain and one-hop inference, liar detection, restart and snapshot rollover. Physical CUDA/ROCm/MPS tests are skipped when that hardware is unavailable, so those paths remain unvalidated on real devices.

## How it works

- Byte vocabulary of 256 with tied embedding/output weights.
- The reference run has **1,611,827,201 parameters** in 24 blocks: width 1024, 16 heads, two transformer layers per block, eight experts and FFN 1728, with context 256.
- Each block learns a separate diffusion-noise interval; workers need not exchange intermediate activations.
- Workers receive leases over HTTPS and return parameter deltas; the coordinator validates keys, shapes, finite values, step/token counts, saturation and delta bounds before merging.
- Immediate block updates are checked against a retained lease sample and held-out text. Shared deltas are banked into rounds and checked separately at their actual commit.
- Generation uses probability-flow Euler steps. The coordinator measures `chain` and `top` on real text and serves the better-scoring mode by default; both remain explicitly selectable.
- One inference snapshot and one optimizer state are retained per worker to cap memory. Lease and block downloads are bounded to prevent large-model stampedes.
- Payload loading uses `torch.load(..., weights_only=True)`.

Validation is not a guarantee against malicious training updates. The current authenticated coordinator should be treated as experimental infrastructure.

## 1.6B reference coordinator

The reference configuration is 24 blocks, about **268.5 MB of FP32 block weights per lease** and a **6.45 GB** full checkpoint. Start it only with resources for that checkpoint and its gate evaluations. The public run keeps the previous 16M coordinator as a rollback lane.

```sh
python dbmoe.py coord --dir run-1p6b --host 127.0.0.1 --port 8791 \
  --tokens tokens.json \
  --hf "HuggingFaceFW/fineweb-edu:sample/10BT/*.parquet" \
  --cfg '{"vocab":256,"d":1024,"heads":16,"layers_per_block":2,"blocks":24,"experts":8,"ffn":1728,"ctx":256,"aux":0.01}' \
  --steps 4 --batch 1 --max-steps 16 --max-batch 4 \
  --heldout 4 --eval-every 0 --snap-every 0 --max-upload-mb 768
```

Increase upload limits only as needed for the block size. Retained snapshots are disabled in this example because the checkpoint is large; use Hugging Face for verified backups. Every participant is gated by default, including trusted tokens. Daily participant and lifetime job caps are disabled; capacity backpressure, proof of work, lease slots and quality checks remain active.

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


