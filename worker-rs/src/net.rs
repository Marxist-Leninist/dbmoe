//! The coordinator protocol (native builds): enrolment by proof of work, `GET lease?fmt=safetensors`, training,
//! `POST update` as safetensors. Same endpoints, checks and credential files as the Python worker.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::model::{Cfg, HostTensor, Named};
use crate::optim::AdamW;
use crate::sigma::Rng;
use crate::st;
use crate::train::{train_lease, Lease, OptCache, Update};
use crate::{block_on, Device, VERSION};

/// `Content-Type` of a safetensors update.
pub const UPDATE_CONTENT_TYPE: &str = "application/vnd.dbmoe.update+safetensors";
/// Largest lease accepted (bytes).
pub const MAX_LEASE_BYTES: u64 = 2 << 30;

/// A failed request.
#[derive(Debug)]
pub enum NetError {
    /// The coordinator answered with this HTTP status and message.
    Status(u16, String),
    /// No answer (connection, TLS or timeout).
    Transport(String),
    /// An answer this worker cannot use.
    Protocol(String),
    /// The coordinator cannot serve this worker at all; retrying will not help.
    Unsupported(String),
}

impl std::fmt::Display for NetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetError::Status(c, m) => write!(f, "HTTP {c} {m}"),
            NetError::Transport(m) => write!(f, "network: {m}"),
            NetError::Protocol(m) => write!(f, "protocol: {m}"),
            NetError::Unsupported(m) => write!(f, "{m}"),
        }
    }
}

/// The proxy for `url` from the environment, as Python's urllib picks it: `https_proxy`/`HTTPS_PROXY` or
/// `http_proxy`/`HTTP_PROXY` by scheme, and none for loopback hosts or hosts listed in `no_proxy`/`NO_PROXY`.
pub fn proxy_for(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split(['/', '?', '#']).next()?;
    let hostport = authority.rsplit('@').next()?;
    let host = if let Some(v6) = hostport.strip_prefix('[') {
        v6.split(']').next()?.to_string()
    } else {
        hostport.split(':').next()?.to_string()
    }
    .to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".localhost") || host == "::1" || host.starts_with("127.") {
        return None;
    }
    let env = |names: &[&str]| names.iter().find_map(|n| std::env::var(n).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty()));
    if let Some(no) = env(&["no_proxy", "NO_PROXY"]) {
        for entry in no.split(',').map(|e| e.trim().trim_start_matches('.').to_ascii_lowercase()) {
            let entry = entry.split(':').next().unwrap_or("").to_string();
            if entry == "*" || (!entry.is_empty() && (host == entry || host.ends_with(&format!(".{entry}")))) {
                return None;
            }
        }
    }
    match scheme.to_ascii_lowercase().as_str() {
        "https" => env(&["https_proxy", "HTTPS_PROXY"]),
        "http" => env(&["http_proxy", "HTTP_PROXY"]),
        _ => None,
    }
}

fn agent(url: &str) -> ureq::Agent {
    let mut b = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(900))
        .timeout_write(Duration::from_secs(900))
        .user_agent(VERSION);
    if let Some(p) = proxy_for(url) {
        match ureq::Proxy::new(&p) {
            Ok(proxy) => b = b.proxy(proxy),
            Err(e) => eprintln!("[worker] ignoring unusable proxy setting: {e}"),
        }
    }
    b.build()
}

fn read_body(r: ureq::Response, limit: u64) -> Result<Vec<u8>, NetError> {
    let mut buf = Vec::new();
    r.into_reader().take(limit + 1).read_to_end(&mut buf).map_err(|e| NetError::Transport(e.to_string()))?;
    if buf.len() as u64 > limit {
        return Err(NetError::Protocol(format!("response larger than {limit} bytes")));
    }
    Ok(buf)
}

fn call(req: Result<ureq::Response, ureq::Error>, limit: u64) -> Result<Vec<u8>, NetError> {
    match req {
        Ok(r) => read_body(r, limit),
        Err(ureq::Error::Status(code, r)) => {
            let msg = r.into_string().unwrap_or_default();
            Err(NetError::Status(code, msg.chars().take(300).collect()))
        }
        Err(e) => Err(NetError::Transport(e.to_string())),
    }
}

/// Percent-encodes a query value.
pub fn enc(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// A logged-in connection to one coordinator.
pub struct Client {
    pub base: String,
    token: String,
    agent: ureq::Agent,
}

impl Client {
    pub fn new(base: &str, token: &str) -> Self {
        let base = base.trim_end_matches('/').to_string();
        Self { agent: agent(&base), base, token: token.to_string() }
    }

    fn auth(&self) -> String {
        format!("Bearer {}", self.token)
    }

    /// `GET lease` in the safetensors wire format.
    pub fn lease(&self, name: &str, steps: usize, batch: usize, session: &str) -> Result<Lease, NetError> {
        let opt = |n: usize| if n == 0 { String::new() } else { n.to_string() };
        let url = format!(
            "{}/lease?name={}&steps={}&batch={}&session={}&fmt=safetensors",
            self.base,
            enc(name),
            opt(steps),
            opt(batch),
            enc(session)
        );
        let body = call(self.agent.get(&url).set("Authorization", &self.auth()).call(), MAX_LEASE_BYTES)?;
        if body.starts_with(b"PK\x03\x04") {
            // a torch.save zip: this coordinator ignores fmt=safetensors, and every retry would only take another lease
            return Err(NetError::Unsupported(
                "this coordinator does not offer the safetensors wire format yet (it sent a PyTorch pickle); \
                 use the Python worker there, or a coordinator with the safetensors change"
                    .into(),
            ));
        }
        decode_lease(&body).map_err(NetError::Protocol)
    }

    /// `POST update`, retried a few times when the coordinator is busy or the connection drops (a lease merges at
    /// most once, so a retry after a lost reply gets a clean refusal rather than merging twice).
    pub fn update(&self, up: &Update, name: &str, device: &str, rng: &mut Rng) -> Result<Value, NetError> {
        let body = encode_update(up, name, device);
        let url = format!("{}/update", self.base);
        let tries = 4;
        for attempt in 0..tries {
            let r = self
                .agent
                .post(&url)
                .set("Authorization", &self.auth())
                .set("Content-Type", UPDATE_CONTENT_TYPE)
                .send_bytes(&body);
            match call(r, 1 << 20) {
                Ok(b) => return serde_json::from_slice(&b).map_err(|e| NetError::Protocol(format!("update reply: {e}"))),
                Err(NetError::Status(c, m)) if !matches!(c, 429 | 502 | 503 | 504) || attempt == tries - 1 => {
                    return Err(NetError::Status(c, m))
                }
                Err(e @ NetError::Transport(_)) if attempt == tries - 1 => return Err(e),
                Err(e @ NetError::Protocol(_)) => return Err(e),
                Err(_) => {}
            }
            std::thread::sleep(Duration::from_secs_f64(5.0 * 2f64.powi(attempt) + rng.uniform() * 5.0));
        }
        unreachable!()
    }
}

fn host_f32(t: &st::Tensor, what: &str) -> Result<HostTensor, String> {
    Ok(HostTensor { shape: t.shape.clone(), data: t.f32s().map_err(|e| format!("{what}: {e}"))? })
}

/// Parses a safetensors lease: tensors `shared.*`, `block.*` (F32) and `data` (U8 `[rows, ctx+1]`), and the plain
/// fields as JSON in the `dbmoe.lease` metadata entry.
pub fn decode_lease(buf: &[u8]) -> Result<Lease, String> {
    let (tensors, meta) = st::decode(buf)?;
    let m: Value = serde_json::from_str(meta.get("dbmoe.lease").ok_or("lease: no dbmoe.lease metadata")?)
        .map_err(|e| format!("lease metadata: {e}"))?;
    let cfg: Cfg = serde_json::from_value(m["cfg"].clone()).map_err(|e| format!("lease cfg: {e}"))?;
    let u = |k: &str| m[k].as_u64().map(|x| x as usize).ok_or(format!("lease: {k} missing"));
    let f = |k: &str| m[k].as_f64().ok_or(format!("lease: {k} missing"));
    let sigmas = m["sigmas"]
        .as_array()
        .ok_or("lease: sigmas missing")?
        .iter()
        .map(|x| x.as_f64().ok_or("lease: bad sigma".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let (mut shared, mut block_state, mut data) = (Named::new(), Named::new(), None);
    for (name, t) in &tensors {
        if name == "data" {
            if t.dtype != "U8" || t.shape.len() != 2 {
                return Err(format!("lease: data is {} {:?}, expected U8 [rows, ctx+1]", t.dtype, t.shape));
            }
            data = Some(t);
        } else if let Some(k) = name.strip_prefix("shared.") {
            shared.insert(k.to_string(), host_f32(t, name)?);
        } else if let Some(k) = name.strip_prefix("block.") {
            block_state.insert(k.to_string(), host_f32(t, name)?);
        } else {
            return Err(format!("lease: unexpected tensor {name}"));
        }
    }
    let data = data.ok_or("lease: no data")?;
    let lease = Lease {
        version: m["version"].as_str().unwrap_or("").to_string(),
        lease: m["lease"].as_str().ok_or("lease: id missing")?.to_string(),
        block: u("block")?,
        block_version: m["block_version"].as_i64().unwrap_or(0),
        cfg,
        sigmas,
        steps: u("steps")?,
        batch: u("batch")?,
        lr: f("lr")?,
        ttl: m["ttl"].as_f64().unwrap_or(0.0),
        shared,
        block_state,
        data: data.bytes.clone(),
        data_rows: data.shape[0],
        data_cols: data.shape[1],
    };
    lease.validate()?;
    Ok(lease)
}

/// Encodes an update: tensors `d_shared.*` and `d_block.*` (F32), the rest as JSON in `dbmoe.update` metadata.
pub fn encode_update(up: &Update, name: &str, device: &str) -> Vec<u8> {
    let mut tensors = BTreeMap::new();
    for (group, map) in [("d_shared", &up.d_shared), ("d_block", &up.d_block)] {
        for (k, t) in map {
            tensors.insert(format!("{group}.{k}"), st::Tensor::from_f32(t.shape.clone(), &t.data));
        }
    }
    let meta = json!({
        "lease": up.lease,
        "block": up.block,
        "losses": up.losses,
        "seconds": up.seconds,
        "tokens": up.tokens,
        "fresh_optimizer": up.fresh_optimizer,
        "name": name,
        "device": device,
    });
    let mut m = BTreeMap::new();
    m.insert("dbmoe.update".to_string(), meta.to_string());
    st::encode(&tensors, &m)
}

/// Finds the proof-of-work nonce: `sha256("{challenge}.{sig}.{n}")` must start with `bits` zero bits.
pub fn solve_challenge(challenge: &str, sig: &str, bits: u32) -> String {
    let mut base = Sha256::new();
    base.update(format!("{challenge}.{sig}.").as_bytes());
    let mut n: u64 = 0;
    loop {
        let mut h = base.clone();
        h.update(n.to_string().as_bytes());
        let d = h.finalize();
        let mut zeros = 0u32;
        for byte in d.iter() {
            if *byte == 0 {
                zeros += 8;
            } else {
                zeros += byte.leading_zeros();
                break;
            }
        }
        if zeros >= bits {
            return n.to_string();
        }
        n += 1;
    }
}

/// Mints a credential: `GET join`, proof of work, `POST join`. Returns `(token, participant)`.
pub fn mint(base: &str) -> Result<(String, String), NetError> {
    let a = agent(base);
    let ch: Value = serde_json::from_slice(&call(a.get(&format!("{base}/join")).call(), 1 << 20)?)
        .map_err(|e| NetError::Protocol(format!("join: {e}")))?;
    let (c, s) = (ch["challenge"].as_str().unwrap_or(""), ch["sig"].as_str().unwrap_or(""));
    let bits = ch["bits"].as_u64().ok_or(NetError::Protocol("join: no bits".into()))?;
    if bits > 40 {
        return Err(NetError::Protocol(format!("join: difficulty {bits} bits is unreasonable")));
    }
    let nonce = solve_challenge(c, s, bits as u32);
    let body = json!({"challenge": c, "sig": s, "nonce": nonce}).to_string();
    let r: Value = serde_json::from_slice(&call(
        a.post(&format!("{base}/join")).set("Content-Type", "application/json").send_string(&body),
        1 << 20,
    )?)
    .map_err(|e| NetError::Protocol(format!("join reply: {e}")))?;
    let token = r["token"].as_str().ok_or(NetError::Protocol("join: no token".into()))?.to_string();
    Ok((token, r["participant"].as_str().unwrap_or("").to_string()))
}

/// The home directory as Python's `os.path.expanduser('~')` finds it: `HOME` on Unix; on Windows `USERPROFILE`
/// (never `HOME`, which Git Bash and MSYS set), so both workers find the same `~/.dbmoe`.
fn home() -> PathBuf {
    #[allow(deprecated)] // un-deprecated in Rust 1.86 with exactly this Windows behaviour
    std::env::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

/// Where the credential for `base` lives: the same file the Python worker uses, so both share one identity.
pub fn token_path(base: &str) -> PathBuf {
    let h = Sha256::digest(base.as_bytes());
    let hex: String = h.iter().map(|b| format!("{b:02x}")).collect();
    home().join(".dbmoe").join(format!("{}.token", &hex[..12]))
}

fn write_private(path: &PathBuf, text: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    use std::io::Write;
    o.open(path)?.write_all(text.as_bytes())
}

/// The saved credential, or a new one.
pub fn load_or_join(base: &str, force: bool) -> Result<String, NetError> {
    let path = token_path(base);
    if !force {
        if let Ok(t) = std::fs::read_to_string(&path) {
            let t = t.trim();
            if !t.is_empty() {
                return Ok(t.to_string());
            }
        }
    }
    println!("[worker] getting a credential (a moment of proof of work; nobody has to approve it)");
    let (token, participant) = mint(base)?;
    write_private(&path, &token).map_err(|e| NetError::Protocol(format!("saving credential: {e}")))?;
    println!("[worker] joined as {participant}; credential saved in {}", path.display());
    Ok(token)
}

/// A random label kept on this machine (never the hostname); separate from the Python worker's label.
pub fn stable_worker_name(rng: &mut Rng) -> String {
    let path = home().join(".dbmoe").join("worker-name-rs");
    if let Ok(n) = std::fs::read_to_string(&path) {
        let n = n.trim();
        if !n.is_empty() {
            return n.chars().take(40).collect();
        }
    }
    let name = format!("worker-rs-{}", rng.hex(3));
    let _ = write_private(&path, &name);
    name
}

/// Worker options (see `dbmoe-worker worker --help`).
#[derive(Clone, Debug)]
pub struct WorkerArgs {
    pub url: String,
    pub token: Option<String>,
    pub name: Option<String>,
    pub device: String,
    pub steps: usize,
    pub batch: usize,
    pub once: usize,
    pub hours: f64,
    pub seed: Option<u64>,
}

/// The training loop: lease, train, update, until `hours` or `once` leases are done.
pub fn run_worker(a: &WorkerArgs, device: &Device) -> Result<(), String> {
    let base = a.url.trim_end_matches('/').to_string();
    // --seed makes the noise reproducible; the label and session must still differ between machines
    let mut ids = Rng::from_entropy();
    let mut rng = a.seed.map(Rng::new).unwrap_or_else(Rng::from_entropy);
    let auto = a.token.is_none();
    let mut token = match &a.token {
        Some(t) if t.starts_with('@') => std::fs::read_to_string(&t[1..]).map_err(|e| format!("token file {}: {e}", &t[1..]))?.trim().to_string(),
        Some(t) => t.clone(),
        None => load_or_join(&base, false).map_err(|e| e.to_string())?,
    };
    let name = match a.name.as_deref().map(str::trim) {
        Some(n) if !n.is_empty() => n.to_string(),
        _ => stable_worker_name(&mut ids),
    };
    let session = ids.hex(6);
    println!("[worker] {VERSION} {name} -> {base}  device={}", a.device);
    let opt = AdamW::default();
    let mut cache = OptCache::new();
    let deadline = if a.hours > 0.0 {
        Duration::try_from_secs_f64(a.hours * 3600.0).ok().and_then(|d| web_time::Instant::now().checked_add(d))
    } else {
        None
    };
    let (mut backoff, mut done) = (5u64, 0usize);
    let mut client = Client::new(&base, &token);
    while deadline.map_or(true, |d| web_time::Instant::now() < d) {
        let t0 = web_time::Instant::now();
        let result = client.lease(&name, a.steps, a.batch, &session).and_then(|lease| {
            let t1 = web_time::Instant::now();
            let up = block_on(train_lease(&lease, device, &mut cache, &mut rng, &opt)).map_err(NetError::Protocol)?;
            let t2 = web_time::Instant::now();
            let r = client.update(&up, &name, &a.device, &mut rng)?;
            let tail = &up.losses[up.losses.len().saturating_sub(5)..];
            println!(
                "[worker] block {}{}  ce {:.3} -> {:.3}  {:.0} bytes/s  train {:.0}s  net {:.1}s  -> {}",
                lease.block,
                if up.fresh_optimizer { " (fresh Adam)" } else { "" },
                up.losses[0],
                tail.iter().sum::<f64>() / tail.len() as f64,
                up.tokens as f64 / up.seconds.max(1e-6),
                up.seconds,
                (t1 - t0).as_secs_f64() + t2.elapsed().as_secs_f64(),
                r
            );
            Ok(())
        });
        match result {
            Ok(()) => {
                backoff = 5;
                done += 1;
                if a.once > 0 && done >= a.once {
                    return Ok(());
                }
            }
            Err(NetError::Status(410, _)) => {
                println!("[worker] this credential has left the run; stopping");
                return Ok(());
            }
            Err(NetError::Unsupported(m)) => return Err(m),
            Err(e) => {
                println!("[worker] {e}; retry in {backoff}s");
                if let NetError::Status(code, _) = e {
                    if code == 401 && auto {
                        match load_or_join(&base, true) {
                            Ok(t) => {
                                token = t;
                                client = Client::new(&base, &token);
                            }
                            Err(e2) => {
                                println!("[worker] could not join: {e2}");
                                backoff = 300;
                            }
                        }
                    } else if code == 401 {
                        backoff = 300;
                    } else if code == 403 {
                        backoff = 900;
                    }
                }
                std::thread::sleep(Duration::from_secs(backoff));
                backoff = (backoff * 2).min(300);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_of_work_matches_python() {
        // int.from_bytes(sha256(f"{c}.{s}.{n}").digest(), 'big') < 1 << (256 - bits)
        let n = solve_challenge("abc", "def", 12);
        let d = Sha256::digest(format!("abc.def.{n}").as_bytes());
        assert_eq!(d[0], 0);
        assert!(d[1] < 16);
        assert_eq!(solve_challenge("x", "y", 0), "0");
    }

    #[test]
    fn query_encoding() {
        assert_eq!(enc("worker-rs a/b+c"), "worker-rs%20a%2Fb%2Bc");
    }

    #[test]
    fn proxies_follow_the_environment_like_urllib() {
        // one test owns these variables (tests run in threads, and nothing else here reads them)
        std::env::set_var("HTTPS_PROXY", "http://proxy.example:3128");
        std::env::set_var("HTTP_PROXY", "http://plain.example:8080");
        std::env::set_var("NO_PROXY", "internal.example, .corp");
        assert_eq!(proxy_for("https://codex.opentransformers.online/dbmoe").as_deref(), Some("http://proxy.example:3128"));
        assert_eq!(proxy_for("http://example.org:8790").as_deref(), Some("http://plain.example:8080"));
        assert_eq!(proxy_for("http://127.0.0.1:8790"), None);
        assert_eq!(proxy_for("http://localhost:8790"), None);
        assert_eq!(proxy_for("http://[::1]:8790"), None);
        assert_eq!(proxy_for("https://internal.example/x"), None);
        assert_eq!(proxy_for("https://a.b.corp/x"), None);
        for v in ["HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY"] {
            std::env::remove_var(v);
        }
    }

    #[test]
    fn update_round_trip() {
        let mut d = Named::new();
        d.insert("pos".into(), HostTensor { shape: vec![2, 2], data: vec![1.0, -1.0, 0.5, 0.0] });
        let up = Update {
            lease: "abc".into(),
            block: 1,
            losses: vec![1.5, 1.25],
            seconds: 0.5,
            tokens: 32,
            fresh_optimizer: true,
            d_shared: d.clone(),
            d_block: d,
        };
        let (t, m) = st::decode(&encode_update(&up, "w", "cpu")).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t["d_block.pos"].f32s().unwrap(), vec![1.0, -1.0, 0.5, 0.0]);
        let v: Value = serde_json::from_str(&m["dbmoe.update"]).unwrap();
        assert_eq!(v["tokens"], 32);
        assert_eq!(v["block"], 1);
        assert_eq!(v["losses"][1], 1.25);
    }
}
