//! A DBMoE block-training worker in Rust.
//!
//! This crate ports the training maths of `dbmoe.py` (`Shared`, `Block`, `Layer`, `MoE`, `edm_pre`, `edm_w`,
//! `dblock_loss`, the noise schedule) and PyTorch's `clip_grad_norm_` and AdamW step to [Burn]. Gradients come from
//! Burn's autodiff. The backend is chosen at run time by the [`Device`]: Flex (pure-Rust CPU, the numerical
//! reference checked against `apple/parity.json`) is always built in; the `wgpu` feature adds one GPU path for
//! NVIDIA, AMD, Intel and Apple GPUs, and WebGPU in browsers.
//!
//! Every function that has to read a tensor back to the host is `async`, because browsers (WebGPU) only allow
//! asynchronous readback. Native code drives them with [`block_on`].
//!
//! [Burn]: https://burn.dev

pub mod bench;
pub mod fixture;
pub mod model;
pub mod optim;
pub mod sigma;
pub mod st;
pub mod train;

#[cfg(not(target_arch = "wasm32"))]
pub mod net;

pub use burn::tensor::Device;
pub use model::{dblock_loss, edm_pre, edm_w, Batch, Cfg, Forward, HostTensor, Named, Params, LN_EPS, SD};
pub use optim::{clip_grad_norm, AdamW, OptState};
pub use train::{train_lease, train_step, Lease, OptCache, StepOut, Update};

/// Sent as `User-Agent` and as the worker's version string.
pub const VERSION: &str = concat!("dbmoe-worker-rs/", env!("CARGO_PKG_VERSION"));

/// Runs a future to completion on the current thread (native only; browsers use their own event loop).
#[cfg(not(target_arch = "wasm32"))]
pub fn block_on<F: core::future::Future>(f: F) -> F::Output {
    pollster::block_on(f)
}

/// The device for a name given on the command line or by a web page.
///
/// `cpu` (or `flex`) is Burn's pure-Rust CPU backend, always available. `gpu` (or `wgpu`) is the best GPU that
/// wgpu finds (Vulkan, DirectX 12 or Metal; WebGPU in a browser), `igpu` an integrated one. `vulkan`, `metal`,
/// `webgpu`, `cuda` and `rocm` pin one API. GPU names need the matching cargo feature.
pub fn device_from_name(name: &str) -> Result<Device, String> {
    let n = name.trim().to_ascii_lowercase();
    match n.as_str() {
        "" | "cpu" | "flex" => Ok(Device::flex()),
        #[cfg(feature = "wgpu")]
        "gpu" | "wgpu" | "auto" => Ok(Device::wgpu(burn::tensor::DeviceKind::DefaultDevice)),
        #[cfg(feature = "wgpu")]
        "dgpu" | "discrete" => Ok(Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0))),
        #[cfg(feature = "wgpu")]
        "igpu" | "integrated" => Ok(Device::wgpu(burn::tensor::DeviceKind::IntegratedGpu(0))),
        #[cfg(feature = "vulkan")]
        "vulkan" => Ok(Device::vulkan(burn::tensor::DeviceKind::DefaultDevice)),
        #[cfg(feature = "metal")]
        "metal" | "mps" => Ok(Device::metal(burn::tensor::DeviceKind::DefaultDevice)),
        #[cfg(feature = "webgpu")]
        "webgpu" => Ok(Device::webgpu(burn::tensor::DeviceKind::DefaultDevice)),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Device::cuda(0usize)),
        #[cfg(feature = "rocm")]
        "rocm" | "hip" => Ok(Device::rocm(0usize)),
        other => Err(format!(
            "unknown or not-built device {other:?}: this build knows {}",
            known_devices().join(", ")
        )),
    }
}

/// Device names this build accepts.
pub fn known_devices() -> Vec<&'static str> {
    let mut v = vec!["cpu"];
    if cfg!(feature = "wgpu") {
        v.extend(["gpu", "dgpu", "igpu"]);
    }
    if cfg!(feature = "vulkan") {
        v.push("vulkan");
    }
    if cfg!(feature = "metal") {
        v.push("metal");
    }
    if cfg!(feature = "webgpu") {
        v.push("webgpu");
    }
    if cfg!(feature = "cuda") {
        v.push("cuda");
    }
    if cfg!(feature = "rocm") {
        v.push("rocm");
    }
    v
}
