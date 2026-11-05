//! The noise schedule: which sigma range a block trains on and how sigma is drawn within it (`block_range`,
//! `sample_sigma` in dbmoe.py), plus a small seeded random generator.

use std::f64::consts::{PI, SQRT_2};

use crate::model::Cfg;

/// Standard normal CDF, `_cdf` in dbmoe.py.
pub fn cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / SQRT_2))
}

/// Standard normal quantile (inverse CDF): Acklam's rational approximation refined by one Halley step, accurate to
/// double precision. `sqrt(2) * erfinv(2p - 1)` in dbmoe.py.
pub fn ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] =
        [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let tail = |q: f64| {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    let x = if p < 0.02425 {
        tail((-2.0 * p.ln()).sqrt())
    } else if p <= 1.0 - 0.02425 {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    };
    let e = 0.5 * libm::erfc(-x / SQRT_2) - p;
    let u = e * (2.0 * PI).sqrt() * (x * x / 2.0).exp();
    x - u / (1.0 + x * u / 2.0)
}

/// Block `b`'s training range: its slice of the schedule widened by the overlap (`block_range` in dbmoe.py).
pub fn block_range(cfg: &Cfg, sigmas: &[f64], b: usize) -> (f64, f64) {
    let (lo, hi) = (sigmas[b], sigmas[b + 1]);
    let a = (hi / lo).powf(cfg.overlap);
    ((lo / a).max(sigmas[0]), (hi * a).min(sigmas[sigmas.len() - 1]))
}

/// `n` noise levels from the log-normal noise distribution restricted to `[lo, hi]` (`sample_sigma`).
pub fn sample_sigma(cfg: &Cfg, lo: f64, hi: f64, n: usize, rng: &mut Rng) -> Vec<f32> {
    let (pm, ps) = (cfg.sig_pm, cfg.sig_ps);
    let a = cdf((lo.ln() - pm) / ps);
    let b = cdf((hi.ln() - pm) / ps);
    (0..n)
        .map(|_| {
            let q = (a + (b - a) * rng.uniform()).clamp(1e-6, 1.0 - 1e-6);
            (pm + ps * ppf(q)).exp() as f32
        })
        .collect()
}

/// xoshiro256++ seeded through SplitMix64: small, fast and reproducible across platforms.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self { s: [next(), next(), next(), next()] }
    }

    /// A seed from the clock and the process, for when reproducibility is not wanted.
    pub fn from_entropy() -> Self {
        let t = web_time::SystemTime::now().duration_since(web_time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
        let local = 0u8;
        Self::new(t ^ (std::process::id() as u64).rotate_left(32) ^ (&local as *const u8 as u64))
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let out = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// Uniform in [0, 1) with 53 random bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// `n` standard-normal values from this seeded stream, using paired Box–Muller draws.
    /// Keeping this off backend RNGs makes training replay consistent across CPU, GPU and browser workers.
    pub fn standard_normals(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            // uniform() is [0, 1), so 1 - uniform() is (0, 1] and its logarithm is finite.
            let radius = (-2.0 * (1.0 - self.uniform()).ln()).sqrt();
            let angle = 2.0 * PI * self.uniform();
            out.push((radius * angle.cos()) as f32);
            if out.len() < n {
                out.push((radius * angle.sin()) as f32);
            }
        }
        out
    }

    /// Lower-case hex of `n` random bytes.
    pub fn hex(&mut self, n: usize) -> String {
        (0..n).map(|_| format!("{:02x}", self.next_u64() as u8)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppf_inverts_cdf() {
        for &p in &[1e-9, 1e-6, 0.001, 0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.975, 0.999, 1.0 - 1e-6] {
            let x = ppf(p);
            assert!((cdf(x) - p).abs() <= 1e-12 * p.max(1e-3), "p={p} x={x} cdf={}", cdf(x));
        }
        assert!((ppf(0.975) - 1.959963984540054).abs() < 1e-12);
    }

    #[test]
    fn sigma_stays_in_range() {
        let cfg = Cfg::live();
        let sigmas = [0.3397, 1.2, 2.4, 3.9, 6.5, 12.0, 80.0];
        let mut rng = Rng::new(7);
        for b in 0..6 {
            let (lo, hi) = block_range(&cfg, &sigmas, b);
            assert!(lo <= sigmas[b] && hi >= sigmas[b + 1] && lo >= sigmas[0] && hi <= 80.0);
            for s in sample_sigma(&cfg, lo, hi, 256, &mut rng) {
                assert!((s as f64) >= lo * 0.9999 && (s as f64) <= hi * 1.0001, "block {b}: {s} outside [{lo}, {hi}]");
            }
        }
    }
}
