//! The DBMoE worker's training step in a browser: WebAssembly, on WebGPU or on the CPU backend.
//!
//! JavaScript calls [`parity`] (one reference step compared with PyTorch's numbers) and [`bench`] (timed training
//! steps). Both return JSON strings. The page in `index.html` drives them.

use std::cell::RefCell;

use dbmoe_worker::fixture::{self, Tolerance};
use dbmoe_worker::{Cfg, Device};
use wasm_bindgen::prelude::*;

thread_local! {
    /// The WebGPU device, initialized once (Burn's wgpu runtime cannot be initialized twice).
    static GPU: RefCell<Option<Device>> = const { RefCell::new(None) };
}

fn js(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// `cpu` is Burn's pure-Rust backend compiled to WebAssembly; anything else is the browser's WebGPU.
async fn device(name: &str) -> Result<Device, JsValue> {
    if name == "cpu" {
        return Ok(Device::flex());
    }
    if let Some(d) = GPU.with(|g| g.borrow().clone()) {
        return Ok(d);
    }
    let d = Device::wgpu_options().init_async().await.map_err(|e| js(format!("WebGPU is not available: {e}")))?;
    GPU.with(|g| *g.borrow_mut() = Some(d.clone()));
    Ok(d)
}

/// The worker version string.
#[wasm_bindgen]
pub fn version() -> String {
    dbmoe_worker::VERSION.to_string()
}

/// Runs the training step of a parity fixture (the text of `apple/parity.json`) and returns the JSON report.
#[wasm_bindgen]
pub async fn parity(fixture_json: String, device_name: String, atol: f64, rtol: f64) -> Result<String, JsValue> {
    let fx = fixture::load(&fixture_json).map_err(js)?;
    let dev = device(&device_name).await?;
    let report = fixture::run(&fx, &dev, Tolerance { atol, rtol }).await.map_err(js)?;
    serde_json::to_string(&report).map_err(js)
}

/// Times `steps` training steps (after one warm-up) of the live configuration, or of `cfg_json` overrides of it.
#[wasm_bindgen]
pub async fn bench(device_name: String, steps: u32, batch: u32, cfg_json: Option<String>) -> Result<String, JsValue> {
    let mut cfg = Cfg::live();
    if let Some(j) = cfg_json.filter(|j| !j.trim().is_empty()) {
        let mut v = serde_json::to_value(&cfg).map_err(js)?;
        let over: serde_json::Value = serde_json::from_str(&j).map_err(js)?;
        for (k, x) in over.as_object().ok_or_else(|| js("cfg must be a JSON object"))? {
            v[k] = x.clone();
        }
        cfg = serde_json::from_value(v).map_err(js)?;
    }
    let dev = device(&device_name).await?;
    let r = dbmoe_worker::bench::run(&cfg, &dev, steps as usize, 1, batch as usize, 1, |_, _, _, _| {}).await.map_err(js)?;
    serde_json::to_string(&serde_json::json!({ "device": device_name, "cfg": cfg, "result": r })).map_err(js)
}
