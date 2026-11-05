//! A minimal, strict [safetensors](https://github.com/huggingface/safetensors) codec: an 8-byte little-endian
//! header length, a JSON header (`{name: {dtype, shape, data_offsets}, "__metadata__": {str: str}}`), then the
//! raw little-endian tensor bytes, contiguous and in header order. It is the opt-in wire format between the
//! coordinator and workers that do not run PyTorch.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

/// Largest header accepted. A lease or an update header is a few KB.
pub const MAX_HEADER: usize = 1 << 20;
/// Most tensors accepted in one file. A lease or an update has a few dozen.
pub const MAX_TENSORS: usize = 4096;

/// One tensor: dtype name (`"F32"`, `"U8"`, `"I64"`, ...), shape and raw little-endian bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// Bytes per element of a safetensors dtype.
pub fn dtype_size(dtype: &str) -> Option<usize> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
        "I16" | "U16" | "F16" | "BF16" => 2,
        "I32" | "U32" | "F32" => 4,
        "I64" | "U64" | "F64" => 8,
        _ => return None,
    })
}

impl Tensor {
    pub fn from_f32(shape: Vec<usize>, data: &[f32]) -> Self {
        Self { dtype: "F32".into(), shape, bytes: data.iter().flat_map(|v| v.to_le_bytes()).collect() }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// The values as `f32` (only for `F32` tensors).
    pub fn f32s(&self) -> Result<Vec<f32>, String> {
        if self.dtype != "F32" {
            return Err(format!("expected F32, got {}", self.dtype));
        }
        Ok(self.bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

/// Encodes tensors (written in name order) and string metadata.
pub fn encode(tensors: &BTreeMap<String, Tensor>, metadata: &BTreeMap<String, String>) -> Vec<u8> {
    let mut header = Map::new();
    if !metadata.is_empty() {
        let m: Map<String, Value> = metadata.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect();
        header.insert("__metadata__".into(), Value::Object(m));
    }
    let mut offset = 0usize;
    for (name, t) in tensors {
        assert_eq!(Some(t.bytes.len()), dtype_size(&t.dtype).map(|s| s * t.numel()), "tensor {name}: size mismatch");
        let mut entry = Map::new();
        entry.insert("dtype".into(), Value::String(t.dtype.clone()));
        entry.insert("shape".into(), Value::Array(t.shape.iter().map(|&s| Value::from(s as u64)).collect()));
        entry.insert("data_offsets".into(), Value::Array(vec![Value::from(offset as u64), Value::from((offset + t.bytes.len()) as u64)]));
        header.insert(name.clone(), Value::Object(entry));
        offset += t.bytes.len();
    }
    let mut h = serde_json::to_vec(&Value::Object(header)).expect("json");
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + h.len() + offset);
    out.extend_from_slice(&(h.len() as u64).to_le_bytes());
    out.extend_from_slice(&h);
    for t in tensors.values() {
        out.extend_from_slice(&t.bytes);
    }
    out
}

/// Decodes and validates a safetensors buffer: known dtypes, sizes that match shapes, at most [`MAX_TENSORS`]
/// tensors, and offsets that tile the data section exactly (no gaps, overlaps or trailing bytes). The whole layout
/// is checked before any tensor data is copied, so the copies never exceed the buffer's size.
pub fn decode(buf: &[u8]) -> Result<(BTreeMap<String, Tensor>, BTreeMap<String, String>), String> {
    if buf.len() < 8 {
        return Err("safetensors: shorter than its header length".into());
    }
    let n = u64::from_le_bytes(buf[..8].try_into().unwrap());
    if n > MAX_HEADER as u64 || n > (buf.len() - 8) as u64 {
        return Err(format!("safetensors: bad header length {n}"));
    }
    let n = n as usize;
    let header: Value = serde_json::from_slice(&buf[8..8 + n]).map_err(|e| format!("safetensors: header: {e}"))?;
    let Value::Object(header) = header else { return Err("safetensors: header is not an object".into()) };
    if header.len() > MAX_TENSORS + 1 {
        return Err(format!("safetensors: more than {MAX_TENSORS} tensors"));
    }
    let data = &buf[8 + n..];
    let mut metadata = BTreeMap::new();
    let mut entries = Vec::new();
    for (name, entry) in header {
        if name == "__metadata__" {
            let Value::Object(m) = entry else { return Err("safetensors: __metadata__ is not an object".into()) };
            for (k, v) in m {
                let Value::String(s) = v else { return Err(format!("safetensors: metadata {k} is not a string")) };
                metadata.insert(k, s);
            }
            continue;
        }
        let bad = |why: &str| format!("safetensors: tensor {name:?}: {why}");
        let Value::Object(e) = entry else { return Err(bad("entry is not an object")) };
        if e.len() != 3 {
            return Err(bad("expected exactly dtype, shape and data_offsets"));
        }
        let dtype = e.get("dtype").and_then(Value::as_str).ok_or_else(|| bad("no dtype"))?.to_string();
        let size = dtype_size(&dtype).ok_or_else(|| bad("unknown dtype"))?;
        let shape: Vec<usize> = e
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("no shape"))?
            .iter()
            .map(|v| v.as_u64().map(|x| x as usize).ok_or_else(|| bad("shape entries must be non-negative integers")))
            .collect::<Result<_, _>>()?;
        let offs = e.get("data_offsets").and_then(Value::as_array).ok_or_else(|| bad("no data_offsets"))?;
        if offs.len() != 2 {
            return Err(bad("data_offsets must have two entries"));
        }
        let (a, b) = match (offs[0].as_u64(), offs[1].as_u64()) {
            (Some(a), Some(b)) if a <= b && b <= data.len() as u64 => (a as usize, b as usize),
            _ => return Err(bad("data_offsets out of range")),
        };
        let numel = shape.iter().try_fold(1usize, |acc, &s| acc.checked_mul(s)).ok_or_else(|| bad("shape overflows"))?;
        if numel.checked_mul(size) != Some(b - a) {
            return Err(bad("byte length does not match shape and dtype"));
        }
        entries.push((a, b, name, dtype, shape));
    }
    if entries.len() > MAX_TENSORS {
        return Err(format!("safetensors: more than {MAX_TENSORS} tensors"));
    }
    let mut spans: Vec<(usize, usize)> = entries.iter().map(|e| (e.0, e.1)).collect();
    spans.sort_unstable();
    let mut pos = 0;
    for (a, b) in spans {
        if a != pos {
            return Err("safetensors: tensor data has gaps or overlaps".into());
        }
        pos = b;
    }
    if pos != data.len() {
        return Err("safetensors: trailing bytes after the last tensor".into());
    }
    // the spans tile the data section, so these copies add up to its size
    let tensors = entries.into_iter().map(|(a, b, name, dtype, shape)| (name, Tensor { dtype, shape, bytes: data[a..b].to_vec() })).collect();
    Ok((tensors, metadata))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut t = BTreeMap::new();
        t.insert("b".to_string(), Tensor::from_f32(vec![2, 3], &[1.0, -2.0, 3.5, 0.0, 1e-8, -0.0]));
        t.insert("a".to_string(), Tensor { dtype: "U8".into(), shape: vec![3], bytes: vec![1, 2, 255] });
        t.insert("s".to_string(), Tensor::from_f32(vec![], &[7.0]));
        t.insert("e".to_string(), Tensor { dtype: "I64".into(), shape: vec![0, 4], bytes: vec![] });
        let mut m = BTreeMap::new();
        m.insert("dbmoe.test".to_string(), "{\"x\": 1}".to_string());
        let buf = encode(&t, &m);
        assert_eq!((u64::from_le_bytes(buf[..8].try_into().unwrap()) % 8), 0);
        let (t2, m2) = decode(&buf).unwrap();
        assert_eq!(t, t2);
        assert_eq!(m, m2);
        assert_eq!(t2["b"].f32s().unwrap()[2], 3.5);
    }

    #[test]
    fn rejects_malformed() {
        let mut t = BTreeMap::new();
        t.insert("x".to_string(), Tensor::from_f32(vec![2], &[1.0, 2.0]));
        let good = encode(&t, &BTreeMap::new());
        assert!(decode(&good[..7]).is_err());
        assert!(decode(&good[..good.len() - 1]).is_err(), "truncated data");
        let mut extra = good.clone();
        extra.push(0);
        assert!(decode(&extra).is_err(), "trailing byte");
        let mut huge = good.clone();
        huge[..8].copy_from_slice(&(u64::MAX).to_le_bytes());
        assert!(decode(&huge).is_err());
        let forge = |h: &str, data: &[u8]| {
            let mut v = (h.len() as u64).to_le_bytes().to_vec();
            v.extend_from_slice(h.as_bytes());
            v.extend_from_slice(data);
            decode(&v)
        };
        assert!(forge(r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#, &[0; 8]).is_ok());
        assert!(forge(r#"{"x":{"dtype":"F31","shape":[2],"data_offsets":[0,8]}}"#, &[0; 8]).is_err());
        assert!(forge(r#"{"x":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#, &[0; 8]).is_err());
        assert!(forge(r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, &[0; 8]).is_err());
        assert!(forge(r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#, &[0; 8]).is_err());
        assert!(forge(r#"{"x":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#, &[0; 4]).is_err());
        assert!(forge(r#"{"__metadata__":{"k":1}}"#, &[]).is_err());
        assert!(forge(r#"[1,2]"#, &[]).is_err());
        // many names over one span: refused by the layout check, before anything is copied
        let span = 1 << 20;
        let many: Vec<String> =
            (0..200).map(|i| format!(r#""t{i}":{{"dtype":"U8","shape":[{span}],"data_offsets":[0,{span}]}}"#)).collect();
        assert!(forge(&format!("{{{}}}", many.join(",")), &vec![0; span]).is_err());
        let empties: Vec<String> = (0..MAX_TENSORS + 1).map(|i| format!(r#""t{i}":{{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#)).collect();
        assert!(forge(&format!("{{{}}}", empties.join(",")), &[]).is_err());
        assert!(forge(&format!(r#"{{"__metadata__":{{"k":"{}"}}}}"#, "x".repeat(MAX_HEADER)), &[]).is_err());
    }
}
