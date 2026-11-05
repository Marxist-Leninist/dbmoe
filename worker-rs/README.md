# dbmoe-worker (Rust)

The DBMoE worker as one binary: lease a block from a coordinator, train it, send the change back. It is a port of the
training maths in [`dbmoe.py`](../dbmoe.py) to [Burn](https://burn.dev) 0.22, so it needs no Python and no PyTorch:

- **CPU** everywhere, through Burn's pure-Rust Flex backend (SIMD, multi-threaded). The Linux build is fully static.
- **Any GPU** through wgpu (`--features wgpu`): NVIDIA, AMD, Intel, Qualcomm Adreno and Apple GPUs via Vulkan,
  DirectX 12 or Metal, with no CUDA or ROCm install. Windows ARM64/Adreno uses DirectX 12, not OpenCL.
- **Browsers**, compiled to WebAssembly and running on WebGPU ([`web/`](web/)).

It is experimental. The numbers match PyTorch (below), but it has only trained against local coordinators so far.

## What it talks to

The Python coordinator speaks `torch.save` pickles, which only PyTorch can read. This crate needs the coordinator's
opt-in **safetensors** wire format, added to `dbmoe.py` together with this crate:

- `GET lease?...&fmt=safetensors` returns one safetensors file: tensors `shared.*`, `block.*` (F32) and `data`
  (U8 `[steps*batch, ctx+1]`), and every other lease field as JSON in the `dbmoe.lease` metadata entry.
- `POST update` with `Content-Type: application/vnd.dbmoe.update+safetensors`: tensors `d_shared.*` and `d_block.*`
  (F32), the other fields (`lease`, `block`, `losses`, `seconds`, `tokens`, `fresh_optimizer`, `name`, `device`) as
  JSON in `dbmoe.update`. The coordinator runs exactly the checks it runs on a pickle upload.

A coordinator without this change (the live one, until it is deployed) ignores `fmt=safetensors` and answers with a
pickle; the Rust worker recognises that, says so and stops, rather than taking lease after lease it cannot use.

## Build

Rust 1.95 or newer ([rustup.rs](https://rustup.rs)); Burn 0.22 needs it.

```sh
cargo build --release --manifest-path worker-rs/Cargo.toml                    # CPU
cargo build --release --manifest-path worker-rs/Cargo.toml --features wgpu     # CPU and GPU
```

The binary is `worker-rs/target/release/dbmoe-worker` (`.exe` on Windows). CI also builds downloadable binaries for
Linux x86_64 (static, CPU), Linux x86_64 (GPU), Windows x86_64 and Windows ARM64 (static C runtime, GPU, including
Adreno through DirectX 12) and macOS arm64 (GPU): see the `worker-rs` workflow's artifacts. The Windows ARM64 job
runs natively on `windows-11-arm` and repeats the CPU parity check there; an Adreno GPU itself is not yet tested.

## Run against a local coordinator

Everything below stays on your machine.

```sh
mkdir -p /tmp/dbmoe-local && cd /tmp/dbmoe-local
python3 -c "print('Some local text to train on. ' * 2000)" > train.txt
echo '{"local-test-token": "me"}' > tokens.json
python3 /path/to/dbmoe.py coord --dir run --host 127.0.0.1 --port 8790 --tokens tokens.json --text-file train.txt \
    --cfg '{"d": 64, "heads": 2, "layers_per_block": 2, "blocks": 4, "experts": 2, "ffn": 128, "ctx": 64}' \
    --steps 20 --batch 4 --min-steps 1 --no-gate-all --eval-every 0
```

In a second terminal:

```sh
worker-rs/target/release/dbmoe-worker worker --url http://127.0.0.1:8790 --token local-test-token --once 3
```

Each finished lease prints a line like the Python worker's (block, loss at the start and end, training bytes per
second, and the coordinator's reply `{"ok":true,"updates":N}`); `curl -s http://127.0.0.1:8790/status` shows the
updates merged. Leave out `--once` to keep going, add `--device gpu` for a wgpu build.

| Option | What it does |
| --- | --- |
| `--url URL` | The coordinator (default: the public one, which needs the safetensors format first) |
| `--token TOKEN` or `--token @FILE` | A token from the coordinator's `tokens.json`. Without one, the worker enrols itself by proof of work and keeps the credential in `~/.dbmoe`, the same file the Python worker uses |
| `--device cpu`, `gpu`, `igpu`, `dgpu`, `adreno` | CPU (default), or the best/integrated/discrete GPU through wgpu; `adreno` selects the integrated Qualcomm GPU (DirectX 12 on Windows ARM64); `dx12`, `vulkan`, `metal` or `opengl` pin a graphics API |
| `--threads N` | CPU threads (default 2) |
| `--steps N`, `--batch N` | Ask for this lease size (0 = the coordinator's default) |
| `--hours H`, `--once N` | Stop after about H hours, or after N leases |
| `--name NAME` | Worker label (default: a random `worker-rs-xxxxxx`, kept in `~/.dbmoe/worker-name-rs`) |
| `--seed N` | Reproducible noise and sigma draws |

Not yet in the Rust worker: serving blocks for network inference (`ipoll`), and the opt-in hardware report.

## Numerical parity with PyTorch

[`apple/parity.json`](../apple/parity.json) holds one training step computed by PyTorch: inputs, loss, logits,
every gradient before and after clipping, and every parameter after one AdamW step. `dbmoe-worker parity` repeats it:

```sh
worker-rs/target/release/dbmoe-worker parity --fixture apple/parity.json
```

On the CPU all 87 checks pass with a largest absolute error of 8.3e-7 (the test requires 1e-5): the loss to 4.8e-7,
logits to 8.3e-7, gradients to 4.6e-7 or better, and the parameters after AdamW are bit-identical for 13 of 20
tensors and within 4.7e-10 (one float32 rounding step) for the rest. CI repeats this on Linux x86_64, Windows x86_64
and macOS arm64, also with fixtures regenerated from the current `dbmoe.py` in the balanced and the `--skewed`
routing case (which exercises the load-balancing gradient); the largest error there is 1.1e-6. On a GPU (Intel UHD
620 through DirectX 12) the step passes with a largest error of 9.5e-7. `cargo test` runs the CPU check.

## Speed

Seconds per training step of the live model (d=256, 4 layers, 4 experts, ffn 528; batch 8 x 256 bytes; forward,
backward, clipping and AdamW), median of 3 to 5 steps, 2 CPU threads, both measured on the same machine:

| Machine | PyTorch | Rust, CPU | Rust, GPU |
| --- | --- | --- | --- |
| Latitude 5400 laptop (Core i5-8365U, Intel UHD 620), Windows | 0.94 s (torch 2.14) | 2.68 s | 6.0 s (UHD 620, DirectX 12) |
| GitHub runner, Linux x86_64 | 0.29 s (torch 2.15) | 0.78 s | |
| GitHub runner, Windows x86_64 | 0.43 s | 1.10 s | |
| GitHub runner, macOS arm64 | 0.26 s | 1.10 s | |

Plainly: on CPUs PyTorch is 2.5 to 4.3 times faster than this worker today, and on the laptop the integrated GPU
is slower than the CPU. The GPU's first steps also compile shaders (344 s and 78 s on the UHD 620 before steps
settled near 6 s, and one 29 s step later, when the expert loads produced tensor sizes it had not seen). No discrete
GPU has run it yet. What the Rust worker offers now is reach: an 8 MB static binary, any GPU through DirectX,
Vulkan or Metal without a CUDA or ROCm install, and browsers. Known places to make it faster: 2 and 4 threads take
the same time on the CPU, so the time goes somewhere other than the parallel matrix products (profiling is next);
routing reads the router's choices back once per layer; and per-expert batch sizes change every step, which makes GPU
backends compile new kernels (padding expert batches to a few fixed sizes would bound that).

```sh
worker-rs/target/release/dbmoe-worker bench --threads 2                 # add --device gpu with a wgpu build
python worker-rs/bench/torch_step.py --threads 2
```

## GPUs

| Build | Runs on | Tested |
| --- | --- | --- |
| `--features wgpu` | NVIDIA, AMD, Intel, Qualcomm Adreno and Apple GPUs through DirectX 12 (Windows), Vulkan (Linux) or Metal (macOS) with `--device gpu`; `adreno` selects an integrated Qualcomm GPU; `dx12`, `vulkan`, `metal`, `opengl` pin one | Intel UHD 620 on Windows: parity passes through DirectX 12 and OpenGL runs; Intel's Vulkan driver crashed, which is why `gpu` means DirectX 12 on Windows. Built in CI for Linux, Windows x86_64/ARM64 and macOS; ARM64 CPU parity runs natively, but no Adreno GPU has run it yet |
| `--features cuda` | NVIDIA through CUDA (`--device cuda`); needs the NVIDIA driver and CUDA runtime | not yet built or run |
| `--features rocm` | AMD through ROCm/HIP (`--device rocm`); needs ROCm | not yet built or run |

OpenCL is not offered: Burn has no OpenCL backend. On Windows ARM64, Adreno is reached through DirectX 12 rather
than OpenCL. The outside contributor's reported OpenCL optimization is a separate implementation, not code or a
benchmark incorporated here.

## Container

[`Dockerfile`](Dockerfile) puts the static CPU worker on an empty base image of 8 MB (CI builds it and runs the
parity check inside). On Linux a container shares the host kernel, so CPU training runs at native speed; Docker Desktop on
Windows and macOS runs a Linux VM. For a GPU, run the native binary.

```sh
docker build -f worker-rs/Dockerfile -t dbmoe-worker-rs .
docker run --rm dbmoe-worker-rs parity --fixture /parity.json
docker run --rm --network host -v dbmoe-worker:/data dbmoe-worker-rs worker --url http://127.0.0.1:8790 --token local-test-token
```

## Browser

`web/` wraps the training step for JavaScript. The page runs the parity step and a benchmark on WebGPU or on the
WebAssembly CPU backend; nothing is sent anywhere. Build it with:

```sh
rustup target add wasm32-unknown-unknown
cd worker-rs/web
cargo build --release                                  # .cargo/config.toml selects wasm32 and getrandom's backend
cargo install wasm-bindgen-cli --version <the wasm-bindgen version in web/Cargo.lock>
wasm-bindgen --target web --no-typescript --out-dir pkg target/wasm32-unknown-unknown/release/dbmoe_web.wasm
cp index.html ../../apple/parity.json pkg/ && python3 -m http.server -d pkg 8000
```

then open http://localhost:8000 in a browser with WebGPU (Chrome, Edge, or Safari 26+). CI builds the same `pkg/` as
an artifact. Training from a browser against a coordinator needs CORS on the coordinator, which this change does not add.

## Layout

| File | What |
| --- | --- |
| `src/model.rs` | `Cfg`, parameters, `dblock_loss` (embedding table, EDM preconditioning, causal attention, top-1 MoE, weighted cross-entropy) |
| `src/optim.rs` | `clip_grad_norm_` and AdamW in PyTorch's operation order |
| `src/train.rs` | One training step; one lease (`train_lease`: warm-up, per-block optimizer cache, deltas) |
| `src/sigma.rs` | Noise schedule (`block_range`, `sample_sigma`) and a seeded random generator |
| `src/st.rs` | Strict safetensors codec |
| `src/net.rs` | Coordinator protocol: enrolment by proof of work, lease, update, the worker loop |
| `src/fixture.rs` | Parity fixture loader and comparison |
| `src/bench.rs` | Step timing |
| `src/main.rs` | The `dbmoe-worker` command |
| `web/` | WebAssembly + WebGPU wrapper and test page |
| `bench/torch_step.py` | The same timing for PyTorch |

Every tensor readback is `async` (`into_data_async`), because browsers only allow asynchronous GPU reads; native code
drives the futures with `pollster`. Expert routing reads the router's choices back once per layer, so each expert runs
only on its own tokens, as in PyTorch.
