# Native Apple training: implementation contract

Status: **research and numerical-parity groundwork, not a working iOS/iPadOS app**.
The macOS Python CPU/MPS worker is documented in the main README. Nothing in this
directory installs software, enrolls a developer account, signs an app, creates a
background entitlement, obtains a training lease, or starts work on a device.

## Practical runtime choice

[MLX Swift](https://github.com/ml-explore/mlx-swift) supplies automatic
differentiation, neural-network layers and optimizers, including AdamW, and links
an actual iOS/macOS MNIST training example. Its verified
[0.31.4 package](https://github.com/ml-explore/mlx-swift/blob/0.31.4/Package.swift)
declares iOS/iPadOS 17+, macOS 14+, tvOS 17+ and visionOS 1+. Those are dependency
deployment floors, not proof that DBMoE fits or runs on every corresponding device.
There is no watchOS target in that package.

An MLX port must reproduce the byte embedding normalization, tied output head,
causal attention, top-1 expert routing/gating and detached load-balance counts,
EDM preconditioning and weighted cross-entropy, AdamW parameter groups, gradient
clipping, and CPU FP32 delta layout. It must not substitute inference, a classifier,
or an unrelated toy loss for training this model.

[ExecuTorch's training extension](https://github.com/pytorch/executorch/blob/main/extension/training/README.md)
is an alternative worth testing. It has ahead-of-time forward/backward graphs and
optimizers, but documents experimental status and ongoing delegate integration.
An iOS inference delegate does not establish mobile GPU training support.
[Core ML's updatable-model path](https://apple.github.io/coremltools/docs-guides/source/updatable-neural-network-classifier-on-mnist-dataset.html)
has constrained trainable layers/losses and is not a drop-in replacement for this
PyTorch autograd and AdamW worker.

## Tested first increment: deterministic parity fixture

`generate_parity_fixture.py` exports synthetic, named FP32 parameters, explicit
byte IDs, sigma and epsilon, expected logits/denoised values/CE/total loss, every
gradient before and after clipping, and one AdamW step and its deltas. It exercises
both experts and nonzero attention-projection gradients. This is useful to a Swift,
C++ or browser implementation without shipping private weights or training data.

From the repository root with its PyTorch dependency installed:

```sh
python apple/generate_parity_fixture.py --output apple/parity.json
python apple/generate_parity_fixture.py --skewed --output /tmp/parity-skewed.json
python -m unittest -v test_apple_native
```

The checked-in fixture uses the small configuration d=8, heads=2, one layer,
two experts, FFN=12 and context=4, batch=1. Named linear weights use PyTorch's
`[output, input]` layout; arrays are flattened in row-major order, including scalar
shape `[]`. AdamW starts with fresh zero state, uses learning rate 0.001/10,
betas (0.9, 0.95), epsilon 1e-8, weight decay 0.01 only for parameters of rank at
least two, and global gradient norm limit 1.0. Gradient clipping includes PyTorch's
1e-6 norm denominator offset. The `initial`, `gradients`, `clipped_gradients`,
`after_step`, and `deltas` maps use the same names.

The optional `--skewed` case deterministically sets 3:1 expert usage, exercising the
load-balance gradient that is zero in the balanced 2:2 case. Both experts remain
active. This second case is also used in the separate browser port's parity tests.

CPU regeneration/replay tests are a numerical reference, not a passing Swift,
Metal, iPhone or iPad test. The source SHA-256 and PyTorch version record fixture
provenance. Do not silently regenerate expected values to make a different model
pass: review numerical changes and update the versioned contract deliberately.

## Remaining release gates

1. Implement a bounded, versioned native lease/update transport alongside existing
   `torch.save`, with dtype/shape/key/byte-count/finite checks, model revision and
   exact lease identity. Never load executable Python pickle in a native client.
   Checksum integrity does not replace HTTPS authentication or server validation.
2. Port the exact model/objective/optimizer and compare this fixture, then several
   seeds, batches, routing patterns and optimizer-reuse cases against PyTorch.
   Record tolerances rather than claiming bitwise GPU equivalence.
3. Reject oversized models before allocation; account for weights, activations,
   gradients, optimizer state and cached blocks. Bound steps, session duration,
   network transfer and simultaneous work. Do not disable OS memory protections.
4. Build with Xcode, test Simulator where applicable, then test real Apple silicon
   Macs, iPhones and iPads for forward/backward/AdamW parity and an isolated
   coordinator lease/update roundtrip. Simulator success is not device GPU proof.
5. Provide explicit Start/Stop, informed compute/electricity/network consent,
   conservative charging/Wi-Fi defaults and visible progress. Do not collect
   private device identifiers or personal data; detailed hardware sharing stays
   opt-in. Tokens need app-private secure storage and must not appear in logs.
6. Start foreground-only. Pause and preserve only safe state on backgrounding,
   low battery, Low Power Mode, thermal pressure or cancellation. Never submit
   partially computed or expired deltas as complete work.
7. Only add finite user-initiated background sessions after implementing the actual
   [continued-processing lifecycle](https://developer.apple.com/documentation/backgroundtasks/bgcontinuedprocessingtask).
   iOS/iPadOS 26 can support this, but background GPU also requires the
   [GPU capability](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.background-tasks.continued-processing.gpu),
   runtime `supportedResources` checks, visible progress, cancellation and
   expiration handling. Unsupported OS/devices must pause. It is not an
   unrestricted or guaranteed background service.
8. Signing, TestFlight/App Store distribution and any new agreements/accounts are
   separate steps requiring the owner's appropriate authorization and credentials.

tvOS and visionOS require their own app and device tests. Watch/HomePod/accessory
training is not implemented; possible companion/control features are separate work.
