//! Numerical parity with PyTorch: one full training step of `apple/parity.json` on the CPU backend.

use dbmoe_worker::fixture::{self, Tolerance};
use dbmoe_worker::Device;

fn fixture_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("apple").join("parity.json")
}

#[test]
fn cpu_step_matches_pytorch() {
    let text = std::fs::read_to_string(fixture_path()).expect("apple/parity.json");
    let fx = fixture::load(&text).expect("fixture parses");
    // 1e-5 absolute on every number (losses, logits, every gradient, the parameters after AdamW), 1e-4 relative.
    let report = dbmoe_worker::block_on(fixture::run(&fx, &Device::flex(), Tolerance { atol: 1e-5, rtol: 1e-4 })).unwrap();
    for c in report.checks.iter().filter(|c| !c.ok) {
        eprintln!("FAIL {} max|err| {:e} {}", c.name, c.max_abs, c.worst);
    }
    assert!(report.ok, "parity failed; max|err| {:e}", report.max_abs);
    assert!(report.checks.len() > 80, "every parameter is compared");
    eprintln!("parity ok: {} checks, max|err| {:e}", report.checks.len(), report.max_abs);
}
