# DiffusionBlocks-MoE (DBMoE)

A byte-level language model trained and run by computers anywhere. Each worker trains one noise-range block of a mixture-of-experts model and serves that block for inference, following the recipe of [DiffusionBlocks](https://arxiv.org/abs/2506.14202) (Shing, Koyama and Akiba, ICLR 2026). A block needs no other block's activations or gradients, so machines never have to talk to each other. It is experimental; workers can train on CPU, NVIDIA CUDA, AMD ROCm, Intel XPU or Apple Mac MPS.

- Live progress: https://codex.opentransformers.online/dbmoe/progress
- Coordinator: https://codex.opentransformers.online/dbmoe
- Status (JSON): https://codex.opentransformers.online/dbmoe/status

This repository holds the code, not trained weights or training data. `dbmoe.py` here is the same file the coordinator serves at `/code`.

## Join

One command, on Linux or macOS with Python 3.9+. No account, no token, nobody to ask:

```sh
curl -fsSL https://codex.opentransformers.online/dbmoe/join.sh | sh
```

It installs what it needs in `~/dbmoe-worker` and starts. Your worker gets its own credential, saved in `~/.dbmoe`. Ctrl+C stops it and running it again carries on. Options go after `sh -s --`, for example `curl -fsSL https://codex.opentransformers.online/dbmoe/join.sh | sh -s -- --threads 4 --hours 1`. To read the script first, download it with `-o join.sh`.

By hand, and on Windows (`py` instead of `python3`, `curl.exe` in PowerShell):

```sh
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
curl -fsSL https://codex.opentransformers.online/dbmoe/code -o dbmoe.py
python3 dbmoe.py worker --threads 4
```

| Option | What it does |
| --- | --- |
| `--device cuda`, `rocm`, `xpu`, `mps` or `auto` | Train on an NVIDIA, AMD, Intel or Mac GPU, or pick what's there. The default is CPU. See the sections below. |
| `--threads N` | CPU threads to use (default 2) |
| `--hours N` | Stop after about this long, checked between jobs |
| `--no-serve` | Train only, no inference |
| `--serve-only` | Inference only, no training |
| `--no-share-hardware` | Keep your CPU, RAM and GPU off the public worker list |

### Many machines, one identity

Mint one token and use it everywhere, so all your machines count as one participant:

```sh
python3 dbmoe.py enrol       # writes worker.token, readable only by you
# copy worker.token to each machine, then on each one:
python3 dbmoe.py worker --token @worker.token --name machine-1
```

Give each machine its own `--name`. Keep the token out of GitHub, screenshots and public messages.

### Leave

```sh
curl -fsSL https://codex.opentransformers.online/dbmoe/leave.sh | sh
```

Your workers come off the public list and your credential stops working; a worker still running stops by itself. What you already contributed stays in the model, because it can't be separated out. You can rejoin at any time. If you minted your own token, use `python3 dbmoe.py leave --token @worker.token`. Add `--name machine-1` to remove one machine and keep the token.

### What runs on your machine

The worker uses your compute, power and bandwidth, so only join on machines you own or are authorized to use. Workers that serve inference see the bytes of other people's prompts, so nobody should send anything private through the public network.

## How updates are tested

Every update from every computer, including the run owner's, gets the same automatic test before it merges:

1. **It must have learned something.** At full weight, it must lower the loss on text from its own job. Noise and empty uploads fail here.
2. **It must not do harm.** At the weight it would merge with, it must not make held-out text worse. No worker is ever given that text.
3. **Damage can't build up.** A block that has drifted more than 5% above its best takes only improvements, however many participants are sending small steps, and a running total catches one participant who keeps doing slight harm.

Updates on a block are tested one at a time, each on the model it would merge into and at the weight it would merge with. Shared parameters go in as a round every 12 updates, and the round gets its own held-out test first.

A refused update isn't merged or counted. Open participants who keep failing are suspended for 6 hours. Inference is checked too: a new volunteer's first hops are recomputed before anyone sees them, a quarter of everyone's hops are spot-checked after that, and workers caught returning wrong numbers stop being used.

The live counts are on the [progress page](https://codex.opentransformers.online/dbmoe/progress) and in `gate` at `/status`.

## Network inference

```sh
# Automatic: measure both real routes on this snapshot and use the faster one
curl -N "https://codex.opentransformers.online/dbmoe/sample?net=1&route=auto&prompt=The%20&n=32"

# Explicit alternatives: single, distributed (strict), or network (existing fallback behaviour)
curl -N "https://codex.opentransformers.online/dbmoe/sample?net=1&route=distributed&prompt=The%20&n=32"
```

Each response is pinned to one checkpoint snapshot, including across a snapshot rollover. `route=auto` performs one bounded byte of fixed public text on the single coordinator and on strict distributed inference, then caches the faster choice for that snapshot and sampler. It never uses the caller's prompt for calibration, and the selected route and probe timings are reported in the stream. `route=single` uses the coordinator; `route=distributed` requires the necessary workers and forbids coordinator substitution; `route=network` preserves the existing network-with-coordinator-fallback behaviour.

Each byte's diffusion chain goes from worker to worker, one hop per block. Workers long-poll for hops, so this works from behind NAT. The reply is a stream of JSON lines naming the anonymous worker behind every hop. If nobody holds a block, or a worker vanishes, the hop is rerouted or the coordinator runs it and says so unless strict distributed mode was requested. A chain stops early once a block is already certain of the next byte. `sampler=top` uses only the top block, one hop per byte.

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

Then join with `python3 dbmoe.py worker --device rocm --threads 2 --batch 1`.

The AMD device check is integrated into the deployed anonymous worker. Physical AMD training validation is still required before claiming a tested AMD release; the mocked backend suite is not hardware proof.

Use `--device rocm:1` for the second visible GPU, or `--device cuda:1` on NVIDIA. Each worker uses one GPU; `--threads` controls PyTorch CPU threads. Start with batch 1 when available VRAM is uncertain. A coordinator can limit requested batch sizes, and full-size model weights plus optimizer state must still fit. Reduce the batch if training runs out of memory; a successful tiny check is not a guarantee that a live lease will fit.

[PyTorch deliberately uses its `torch.cuda` API and `cuda:N` tensor devices on ROCm](https://docs.pytorch.org/docs/stable/notes/hip.html). DBMoE's `rocm`/`hip` aliases map to that API and require a HIP-enabled build. Existing `--device cuda` commands still work on ROCm and report the backend honestly as `rocm:N`. A CUDA wheel does not add AMD support.

Native Windows and WSL have separate, hardware- and release-specific AMD support. Do not assume an inference-capable Windows package supports this training worker. DBMoE has not validated AMD training on native Windows, WSL, or macOS; use the documented Linux training stack, or `--device cpu`. This change does not add DirectML or Vulkan. Mac GPU participation uses the separate MPS path below, not ROCm.

If the check fails, confirm the installed wheel, supported GPU/OS, driver and device visibility/permissions against AMD's guide. An AMD GPU name appearing in an OS utility is not enough: PyTorch must load its kernels and complete backward propagation. `--device rocm` never silently falls back to CPU.

## Intel GPUs (XPU)

**Intel support is implemented through native PyTorch `torch.xpu`, but has not been validated on physical Intel hardware for this release.** The worker uses its existing FP32 training and CPU-tensor update format. It does not require Intel Extension for PyTorch, CUDA, a custom kernel, or a different coordinator protocol.

Check your exact GPU and OS against [PyTorch's Intel GPU guide](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) and follow the matching [Intel GPU prerequisites](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html). Supported Arc discrete GPUs, selected Core Ultra integrated GPUs and Data Center GPU Max systems have different requirements. Do not assume every Intel iGPU, an older Intel Mac, or an Intel NPU supports this path. Linux, Windows and WSL2 support depends on the hardware/release matrix.

In a fresh virtual environment, after installing the supported Intel driver/runtime, install the official stable XPU wheel (DBMoE needs only `torch`):

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/xpu
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available()); print(torch.xpu.device_count())"
python dbmoe.py check-device --device xpu
```

Use a Python version supported by that wheel. A CPU/CUDA/ROCm wheel or `pip install -r requirements.txt` alone does not select Intel GPU support. Follow the official binary-install instructions, not source-build-only environment setup, when using wheels.

The offline check runs the real tiny DBMoE forward pass, backward pass and AdamW update on the selected XPU, then verifies finite bounded FP32 deltas copied to CPU and serialized. It reads no token, makes no network request and changes no training checkpoint. Every XPU worker runs the same check before reading a credential or joining. If it fails, fix the driver/wheel/device compatibility first; `--device xpu` never silently switches to CPU.

After the check passes, using the reviewed `dbmoe.py` from this branch/release:

```bash
python dbmoe.py worker --url https://codex.opentransformers.online/dbmoe --device xpu --steps 2 --batch 1 --threads 2
```

Use `--device xpu:1` for the second visible Intel GPU; `intel` and `intel:N` are aliases and report `xpu:N`. Each worker uses one GPU. A tiny check does not prove a full lease fits in memory or establish sustained throughput. Start with batch 1, monitor system/GPU memory and temperature, and stop with Ctrl-C. Integrated GPUs share system memory, and PyTorch's reported total device memory is not free or necessarily dedicated VRAM. CPU remains available with `--device cpu`.

`--share-hardware` keeps the existing opt-in behavior and can include the Intel GPU name, rounded total device memory and XPU build version when PyTorch exposes it. It does not send serials, hostnames or paths. Older coordinators may ignore the new optional `xpu` hardware label while accepting the unchanged CPU-delta wire format. The inference-serving path remains CPU-based; selecting XPU accelerates worker training only.

### Intel integrated graphics and shared-memory budgeting

Supported Intel integrated GPUs use the same `--device xpu` / `xpu:N` path. There is no separate iGPU kernel or name-based hardware override. **The following is PyTorch's validated hardware/OS matrix, not DBMoE physical-device test results.** The [PyTorch 2.15 guide](https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html), checked November 5, 2026, lists:

- Core Ultra Arc graphics: Meteor Lake-H, Arrow Lake-H and Lunar Lake on Windows 11, Ubuntu 24.04/26.04 and WSL2 Ubuntu 24.04/26.04.
- Core Ultra Series 3 / Panther Lake on Windows 11, Ubuntu 26.04 and WSL2 Ubuntu 24.04/26.04.

Legacy Intel UHD/Iris Xe graphics and macOS are not in that validated XPU matrix. This feature does not promise support for them, Intel NPUs, or every processor carrying the Core Ultra name. An Intel CPU alone does not imply XPU GPU availability. Use `--device cpu` for a compatible CPU-only installation, or the separate MPS path on a supported Mac.

Follow the [release-matched Intel prerequisites](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-15.html): Linux needs the supported Intel GPU driver/runtime; native Windows needs the supported Windows GPU driver; WSL2 uses the Windows host GPU driver and the Linux guest's user-mode compute runtime. Do not install a Linux kernel GPU driver inside WSL2 or assume the host's Python packages also install the guest environment. Official wheel users should follow the binary instructions rather than separately sourcing source-build toolkit environments. DBMoE does not require `torch.compile`.

Integrated graphics share system RAM with the CPU. For a cautious first run, explicitly limit the XPU caching allocator and turn off additional CPU inference serving:

```bash
python dbmoe.py check-device --device xpu --xpu-memory-fraction 0.5
python dbmoe.py worker --url https://codex.opentransformers.online/dbmoe --device xpu --xpu-memory-fraction 0.5 --steps 2 --batch 1 --threads 2 --no-serve
```

`--xpu-memory-fraction` applies PyTorch's [per-process allocator limit](https://docs.pytorch.org/docs/stable/generated/torch.xpu.memory.set_per_process_memory_fraction.html) to the selected GPU **before the offline training check or any credential/network access**. The value must be finite, greater than zero and at most one. It multiplies PyTorch's reported total device memory, not currently free memory. The local check prints the configured limit in GiB. If the API is missing or applying the cap fails, the worker stops instead of ignoring the requested budget. The flag is rejected on a CPU/CUDA/ROCm/MPS selection. Without it, existing PyTorch allocation behavior is unchanged.

The example fraction is a conservative starting policy, not an Intel guarantee or a reservation. The cap covers PyTorch's XPU caching allocator, not CPU copies of leases, optimizer/checkpoint data, driver allocations, other applications, or total process/system RAM. Do not add reported iGPU memory and CPU RAM as separate pools. Leave headroom for the OS, reduce batch size or the fraction when needed, and stop if memory pressure or temperature rises. A full lease can exceed the cap even when the tiny check passes; that requires a smaller lease/model or more memory, not bypassing the cap. A supported iGPU may not expose a working free-memory query, so startup does not depend on `torch.xpu.mem_get_info()`.

Local checks and optional hardware reports use PyTorch's `is_integrated_gpu` property to label memory as `shared` or `dedicated`; older builds without a boolean property report `unknown`. No GPU-name guessing is used. Shared capacity is neither free memory nor dedicated VRAM. These additions are tested with mocks and CPU training, not Intel GPU emulation or physical Intel hardware. They are repository changes, not a live-service deployment.

## Apple Mac (macOS)

MacBook Air/Pro, iMac, Mac mini, Mac Studio and Mac Pro can use the Python worker **when their OS, architecture and PyTorch build are compatible**. Apple silicon uses `--device mps` for Metal GPU training or `--device cpu`. Intel Macs require a compatible x86_64 PyTorch release and can use the CPU route; do not assume a current Apple-silicon wheel, an AMD GPU inside an older Mac, or the newest PyTorch release supports that machine.

**MPS worker support is implemented and regression-tested on Linux with mocked device selection. It has not yet been validated on physical Apple hardware.** The runtime check below must pass on the particular Mac before it joins. It tests a tiny synthetic model, not full-model memory capacity or long-running stability.

Use [Apple's PyTorch installation guide](https://developer.apple.com/metal/pytorch/) and the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for the selected release. Requirements change by release: Apple's guide currently lists Apple silicon and macOS 14+ for its documented stable build. Older releases have different requirements. Do not use the Linux CPU wheel index on macOS, install CUDA/ROCm packages on a Mac, or bypass unsupported-OS checks.

For the MPS checks, use the current `dbmoe.py` from this repository or `/code`; both include the MPS path and anonymous protocol. The offline check runs before any credential read or network request. Physical Apple-hardware validation remains pending; do not infer support from mocked tests alone.

Run the checks first:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch
python dbmoe.py check-device --device cpu
python dbmoe.py check-device --device mps
```

Then join:

```sh
python dbmoe.py worker --device mps --threads 2 --batch 1 --hours 0.25
```

To train on CPU, change `--device mps` to `--device cpu`. An explicit MPS request fails rather than silently selecting CPU. `mps:0` is accepted as the single MPS device; other MPS indexes are rejected. MPS workers automatically run the offline forward/backward/AdamW/CPU-delta check before reading a credential or requesting work. The implementation avoids `bincount` on MPS releases without that operator. It does not enable PyTorch's CPU fallback environment switch or disable MPS memory safeguards.

Keep the machine ventilated, use external power when practical, start with batch 1, and stop with Ctrl-C if the machine gets hot or memory pressure increases. On Apple silicon the GPU shares system memory: the model, gradients, optimizer state and cached optimizers can consume much more memory than the downloaded weights. A tiny successful check does not prove a large lease fits. `--hours` is checked between leases, not a hard timer, so use short steps for a responsive stop. There is no battery/thermal monitor or automatic launch-at-login behavior in this CLI. Running it is an explicit choice to use compute, electricity and network.

On a Mac the worker list shows the generic `Metal (MPS)` name and no VRAM figure, because the GPU shares system memory.

## iPhone, iPad and other Apple products

**The macOS Python command is not an iOS/iPadOS app. This release does not ship a working native iPhone/iPad training app, signed IPA or App Store/TestFlight build.** Merely opening the progress page does not contribute compute. There is no claim of universal Apple-device training support.

A genuine native path is [MLX Swift](https://github.com/ml-explore/mlx-swift), whose official examples include an iOS/macOS training app. The verified [0.31.4 package targets](https://github.com/ml-explore/mlx-swift/blob/0.31.4/Package.swift) are iOS/iPadOS 17+, macOS 14+, tvOS 17+ and visionOS 1+. These dependency targets are not a DBMoE support or memory guarantee. DBMoE still needs a port of its actual causal transformer/MoE, EDM loss, gradients and AdamW, a language-neutral lease/delta adapter instead of Python `torch.save`, and numerical parity plus real-device tests. Core ML conversion or an inference-only runtime is not a training implementation. See [the native Apple implementation checklist](apple/README.md).

The initial native worker must be foreground-only, start only after an explicit compute/network consent, pause on backgrounding, low battery, Low Power Mode or thermal pressure, and offer a visible Stop action with short bounded jobs. [iOS/iPadOS 26 continued processing](https://developer.apple.com/documentation/backgroundtasks/bgcontinuedprocessingtask) can allow user-initiated finite work to continue in the background; [GPU access additionally needs its capability](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.background-tasks.continued-processing.gpu) and supported hardware/runtime checks. That is not permission or a guarantee for indefinite unattended training. No background entitlement is enabled by this repository.

Apple TV and Apple Vision Pro would need separate native UI/runtime/device validation. Apple Watch, HomePod, AirPods and accessories are not supported training workers; companion/control use would be a separate feature. No jailbreak, sideloading bypass, developer-account creation or paid Apple membership is required or performed by this change.

## What the public worker list shows

Each worker's card shows its CPU model, core and thread counts, RAM, GPU model and memory, operating system and PyTorch version, as the worker reports them, plus the speed the coordinator measures. The worker prints exactly what it is sharing when it starts. No hostname, address, serial number or path is sent, and workers appear under an anonymous ID. `--no-share-hardware` keeps the hardware off the list, and the next accepted update removes an earlier report.

These reports come from participants and are not independently verified. The thread count is what PyTorch was given, not the machine's physical cores, and GPU memory is the total, not what is free (on Intel integrated graphics it may be shared system memory). The coordinator and network provider necessarily see connection IP addresses.

## Voluntary hardware reports

If you test hardware the project has not validated, generate a local report with `python dbmoe.py hardware-report --device xpu --output hardware-report.json` (choose your backend). Add `--include-hardware` only if you want GPU model/memory in the file. Nothing is uploaded or joined automatically. Failures produce fixed safe categories instead of raw logs. Review the JSON, then optionally submit it through the [public hardware-report form](https://github.com/Marxist-Leninist/dbmoe/issues/new?template=hardware-compatibility.yml), associated with your GitHub account.

[Reporting/privacy details and opt-out](docs/hardware-compatibility.md) explain the fields, how to update the same issue after a meaningful change, and why volunteer passes remain self-reported rather than independently verified. [Recent hardware reports](https://github.com/Marxist-Leninist/dbmoe/issues?q=is%3Aissue+in%3Atitle+%22%5BHardware+report%5D%22&s=updated&o=desc) track published findings. This is a manual tracker, not an automatic monitoring or telemetry service.

## Tests

```sh
python -m unittest -v
python dbmoe.py check-device --device cpu
# On a supported AMD machine after installing ROCm PyTorch:
python dbmoe.py check-device --device rocm
# On a compatible Mac after installing macOS PyTorch:
python dbmoe.py check-device --device mps
# On a supported Intel machine after installing XPU PyTorch:
python dbmoe.py check-device --device xpu
```

The unit suite covers CPU training, optimizer-state reuse, the update test, coordinator delta validation, portable serialization, CPU/CUDA/ROCm/XPU/MPS device selection and privacy behavior. Intel tests also cover missing XPU APIs, invalid GPU indexes, explicit-device failure before credential or network access, and selected-device preflight. Integrated-GPU tests cover shared or unknown memory labels, allocator caps applied before training, invalid budgets, missing or failing allocator APIs, and startup without free-memory queries. The hardware-report tests cover the offline, opt-in compatibility report. GPU selection is mocked where hardware is absent. Real CUDA/ROCm/XPU and MPS tests are explicitly skipped when that backend is unavailable. CPU tests passed on PyTorch 2.6.0+cpu; actual Apple, AMD, Intel and NVIDIA hardware validation remains pending.

## How it works

- Byte vocabulary of 256 with tied embedding and output weights. By default, six blocks of two causal transformer layers with a top-1, four-expert feed-forward network, width 256, four attention heads and context length 256.
- The 1.6B reference configuration (`CFG_1P6B`) has 1,611,827,201 parameters in 24 blocks: width 1024, 16 heads, two transformer layers per block, eight experts and FFN 1728, with context 256.
- What is noised is the embedding of the next byte. Each block learns its own slice of a log-normal noise distribution, widened by the paper's overlap, with EDM preconditioning and weighting.
- Generation runs the blocks as a diffusion sampler for the next byte, one Euler step per block, so a byte costs one pass through the layers. The coordinator measures `chain` and `top` sampling on real text and serves the better-scoring mode by default; both stay selectable.
- Workers receive jobs over HTTPS and return parameter deltas. Merging is asynchronous averaging. Shared parameters (embedding, positions) define the target every block denoises toward, so their deltas are banked and go in together every 12 updates.
- Before merging, the coordinator validates keys, shapes, finite values, step and token counts, saturation and delta bounds. Block updates are checked against a retained lease sample and held-out text; shared deltas are banked into rounds and checked at their actual commit.
- Compute goes where it's needed: low-noise blocks can read the target off their input, so more updates teach them little, and the blocks with most left to learn get the most work.
- Workers keep Adam's per-parameter scale between jobs on a block and drop its momentum, which points where the old weights wanted to go.
- Evaluation measures the chain at each depth and scores generated text against the statistics of real text, because block losses alone can't show whether generation works.
- One inference snapshot and one optimizer state are kept per worker to cap memory, and lease and block downloads are bounded so large models don't cause download stampedes.
- Payloads load with `torch.load(..., weights_only=True)`, so nothing a participant sends can run code.

Validation is not a guarantee against malicious training updates. Treat the coordinator as experimental infrastructure.

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

The worker needs PyTorch. The coordinator also requires NumPy for immutable snapshot fingerprints. After selecting the appropriate PyTorch build for your hardware, install the declared dependencies with `python -m pip install -r requirements.txt` before starting a coordinator. Streaming coordinator data additionally needs `huggingface_hub` and `pyarrow`, or `datasets` for dataset-library streaming. You can instead use a local text file you have permission to train on.

```sh
python dbmoe.py coord --dir run --host 127.0.0.1 \
  --text-file training.txt --tokens tokens.json
```

Create `tokens.json` privately as a JSON object mapping randomly generated participant tokens to participant labels, and never commit it. Anyone can enrol through `/join` unless you pass `--no-open-join`. Put TLS in front of anything facing the internet.

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



