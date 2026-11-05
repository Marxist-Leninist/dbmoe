//! `dbmoe-worker`: the DBMoE worker as one binary.

use std::collections::BTreeMap;
use std::process::ExitCode;

use dbmoe_worker::fixture::{self, Tolerance};
use dbmoe_worker::sigma::Rng;
use dbmoe_worker::{device_from_name, known_devices, Batch, Cfg, Device, VERSION};

const HELP: &str = "dbmoe-worker: trains DBMoE blocks for a coordinator, on CPU or any GPU

USAGE
  dbmoe-worker worker  [--url URL] [--token TOKEN|@FILE] [--name NAME] [--device cpu|gpu|...]
                       [--steps N] [--batch N] [--threads N] [--hours H] [--once N] [--seed N]
  dbmoe-worker parity  [--fixture apple/parity.json] [--device cpu] [--atol 1e-5] [--rtol 1e-4] [--json]
  dbmoe-worker bench   [--device cpu] [--threads N] [--steps 10] [--warmup 2] [--batch 8] [--cfg JSON] [--json]
  dbmoe-worker devices
  dbmoe-worker version

worker   Lease a block, train it, send the change back, repeat. Without --token the worker enrols itself
         (a few seconds of proof of work) and keeps the credential in ~/.dbmoe, as the Python worker does.
         The coordinator must offer the safetensors wire format (GET lease?fmt=safetensors).
parity   Repeat the reference training step in apple/parity.json and compare every number with PyTorch.
bench    Time training steps of the live model configuration on random bytes.

--device  cpu (default) is Burn's pure-Rust CPU backend. gpu picks the best GPU through wgpu (Vulkan,
          DirectX 12 or Metal); igpu/dgpu ask for an integrated/discrete one. Needs a build with --features wgpu.
--threads CPU threads (default 2 for worker, all cores for bench).";

struct Opts {
    values: BTreeMap<String, String>,
    flags: Vec<String>,
}

impl Opts {
    fn parse(args: &[String], flags: &[&str]) -> Result<Self, String> {
        let (mut values, mut set) = (BTreeMap::new(), vec![]);
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            let Some(key) = a.strip_prefix("--") else { return Err(format!("unexpected argument {a:?}")) };
            if let Some((k, v)) = key.split_once('=') {
                values.insert(k.to_string(), v.to_string());
            } else if flags.contains(&key) {
                set.push(key.to_string());
            } else {
                let v = args.get(i + 1).ok_or(format!("--{key} needs a value"))?;
                values.insert(key.to_string(), v.clone());
                i += 1;
            }
            i += 1;
        }
        Ok(Self { values, flags: set })
    }

    fn get(&self, k: &str) -> Option<&str> {
        self.values.get(k).map(String::as_str)
    }

    fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> Result<T, String> {
        match self.get(k) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("--{k}: not a number: {v:?}")),
        }
    }

    fn flag(&self, k: &str) -> bool {
        self.flags.iter().any(|f| f == k)
    }

    fn only(&self, allowed: &[&str]) -> Result<(), String> {
        for k in self.values.keys().chain(self.flags.iter()) {
            if !allowed.contains(&k.as_str()) {
                return Err(format!("unknown option --{k}"));
            }
        }
        Ok(())
    }
}

fn set_threads(n: usize) {
    if n > 0 {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    }
}

fn device(o: &Opts) -> Result<(String, Device), String> {
    let name = o.get("device").unwrap_or("cpu").to_string();
    Ok((name.clone(), device_from_name(&name)?))
}

fn parity(o: &Opts) -> Result<bool, String> {
    o.only(&["fixture", "device", "atol", "rtol", "json", "threads"])?;
    set_threads(o.num("threads", 0)?);
    let path = o.get("fixture").unwrap_or("apple/parity.json");
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let fx = fixture::load(&text)?;
    let (name, dev) = device(o)?;
    let tol = Tolerance { atol: o.num("atol", Tolerance::default().atol)?, rtol: o.num("rtol", Tolerance::default().rtol)? };
    let report = dbmoe_worker::block_on(fixture::run(&fx, &dev, tol))?;
    if o.flag("json") {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        for c in &report.checks {
            println!("{} {:<52} max|err| {:9.3e}  rel {:9.3e}  {}", if c.ok { "ok  " } else { "FAIL" }, c.name, c.max_abs, c.max_rel_to_max, c.worst);
        }
        println!(
            "parity {} on {name}: case {}, {} checks, max|err| {:.3e} (atol {:e}, rtol {:e})",
            if report.ok { "PASSED" } else { "FAILED" },
            report.case,
            report.checks.len(),
            report.max_abs,
            tol.atol,
            tol.rtol
        );
    }
    Ok(report.ok)
}

fn bench(o: &Opts) -> Result<bool, String> {
    o.only(&["device", "threads", "steps", "warmup", "batch", "cfg", "json", "seed"])?;
    set_threads(o.num("threads", 0)?);
    let (name, dev) = device(o)?;
    let mut cfg = Cfg::live();
    if let Some(j) = o.get("cfg") {
        let mut v = serde_json::to_value(&cfg).unwrap();
        let over: serde_json::Value = serde_json::from_str(j).map_err(|e| format!("--cfg: {e}"))?;
        for (k, x) in over.as_object().ok_or("--cfg must be a JSON object")? {
            v[k] = x.clone();
        }
        cfg = serde_json::from_value(v).map_err(|e| format!("--cfg: {e}"))?;
    }
    cfg.validate()?;
    let (steps, warmup, bs): (usize, usize, usize) = (o.num("steps", 10)?, o.num("warmup", 2)?, o.num("batch", 8)?);
    let quiet = o.flag("json");
    let r = dbmoe_worker::block_on(dbmoe_worker::bench::run(&cfg, &dev, steps, warmup, bs, o.num("seed", 1)?, |s, dt, loss, g| {
        if !quiet {
            println!("step {s:3}{}  {dt:.3}s  loss {loss:.4}  |g| {g:.3}", if s < warmup { " (warm-up)" } else { "" });
        }
    }))?;
    let threads = rayon::current_num_threads();
    if quiet {
        let summary = serde_json::json!({
            "worker": VERSION, "device": name, "threads": threads, "cfg": cfg, "batch": bs, "steps": steps,
            "median_step_s": r.median_step_s, "mean_step_s": r.mean_step_s, "bytes_per_s": r.bytes_per_s,
        });
        println!("{summary}");
    } else {
        println!(
            "bench on {name} ({threads} CPU threads): median {:.3}s/step, mean {:.3}s, {:.0} bytes/s (batch {bs} x ctx {})",
            r.median_step_s, r.mean_step_s, r.bytes_per_s, cfg.ctx
        );
    }
    Ok(true)
}

fn worker(o: &Opts) -> Result<bool, String> {
    o.only(&["url", "token", "name", "device", "steps", "batch", "threads", "hours", "once", "seed"])?;
    let hours: f64 = o.num("hours", 0.0)?;
    if !(hours.is_finite() && hours >= 0.0) {
        return Err("--hours must be a non-negative number".into());
    }
    set_threads(o.num("threads", 2)?);
    let (name, dev) = device(o)?;
    // a bad device costs nothing: fail before any network or credential, as the Python worker does
    smoke_test(&dev)?;
    let a = dbmoe_worker::net::WorkerArgs {
        url: o.get("url").unwrap_or("https://codex.opentransformers.online/dbmoe").to_string(),
        token: o.get("token").map(str::to_string),
        name: o.get("name").map(str::to_string),
        device: name,
        steps: o.num("steps", 0)?,
        batch: o.num("batch", 0)?,
        once: o.num("once", 0)?,
        hours,
        seed: o.get("seed").map(|s| s.parse()).transpose().map_err(|_| "--seed: not a number".to_string())?,
    };
    dbmoe_worker::net::run_worker(&a, &dev)?;
    Ok(true)
}

/// One tiny training step, so a broken device fails at start-up.
fn smoke_test(dev: &Device) -> Result<(), String> {
    let cfg = Cfg { d: 8, heads: 2, layers_per_block: 1, blocks: 2, experts: 2, ffn: 12, ctx: 4, ..Cfg::live() };
    let mut rng = Rng::new(3);
    let named = cfg
        .param_specs()
        .into_iter()
        .map(|(k, shape)| {
            let n: usize = shape.iter().product::<usize>().max(1);
            let data = (0..n).map(|_| (rng.uniform() - 0.5) as f32).collect();
            (k, dbmoe_worker::HostTensor { shape, data })
        })
        .collect();
    let mut params = dbmoe_worker::Params::from_host(&cfg, &named, dev)?;
    let batch = Batch { ids: vec![7, 65, 110, 116, 33], b: 1, t1: 5, sigma: vec![0.75] };
    let eps = burn::tensor::Tensor::<3>::zeros([1, 4, 8], dev);
    let out = dbmoe_worker::block_on(dbmoe_worker::train_step(
        &mut params,
        &mut dbmoe_worker::OptState::default(),
        &dbmoe_worker::AdamW::default(),
        &cfg,
        &batch,
        eps,
        1e-4,
        1.0,
    ));
    if out.loss.is_finite() {
        Ok(())
    } else {
        Err("device smoke test: the loss is not finite".into())
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    let rest = if args.is_empty() { &[][..] } else { &args[1..] };
    if rest.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return ExitCode::SUCCESS;
    }
    let result = Opts::parse(rest, &["json"]).and_then(|o| match cmd {
        "worker" => worker(&o),
        "parity" => parity(&o),
        "bench" => bench(&o),
        "devices" => {
            println!("{}", known_devices().join("\n"));
            Ok(true)
        }
        "version" | "--version" | "-V" => {
            println!("{VERSION}");
            Ok(true)
        }
        "help" | "--help" | "-h" => {
            println!("{HELP}");
            Ok(true)
        }
        other => Err(format!("unknown command {other:?}\n\n{HELP}")),
    });
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("dbmoe-worker: {e}");
            ExitCode::from(2)
        }
    }
}
