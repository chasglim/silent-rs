use crate::error::ParamError;
use crate::scheme::SchemeFamily;
use crate::security::SecurityLevel;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParamsId(pub [u8; 32]);

impl ParamsId {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            use std::fmt::Write as _;
            let _ = write!(&mut out, "{byte:02x}");
        }
        out
    }
}

pub trait ParameterSet {
    fn name(&self) -> &'static str;
    fn scheme_family(&self) -> SchemeFamily;
    fn security_level(&self) -> SecurityLevel;
    fn params_id(&self) -> ParamsId;
    fn validate(&self) -> Result<(), ParamError>;
}

pub(crate) trait CanonicalParamEncoding {
    fn encode_canonical(&self, out: &mut Vec<u8>);
}

pub(crate) fn params_id_from_canonical<T: CanonicalParamEncoding>(
    domain: &'static [u8],
    value: &T,
) -> ParamsId {
    let mut encoded = Vec::with_capacity(256);
    encoded.extend_from_slice(b"silent-params/v1/");
    encoded.extend_from_slice(domain);
    encoded.push(0);
    value.encode_canonical(&mut encoded);

    let digest = Sha256::digest(&encoded);
    let mut id = [0u8; 32];
    id.copy_from_slice(&digest);
    ParamsId(id)
}

pub(crate) fn push_bool(out: &mut Vec<u8>, value: bool) {
    out.push(u8::from(value));
}

pub(crate) fn push_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

pub(crate) fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn push_usize(out: &mut Vec<u8>, value: usize) {
    push_u64(out, value as u64);
}

pub(crate) fn push_str(out: &mut Vec<u8>, value: &'static str) {
    push_u16(out, value.len() as u16);
    out.extend_from_slice(value.as_bytes());
}

pub(crate) fn push_opt_u8(out: &mut Vec<u8>, value: Option<u8>) {
    match value {
        Some(v) => {
            push_bool(out, true);
            push_u8(out, v);
        }
        None => push_bool(out, false),
    }
}

pub(crate) fn push_opt_u16(out: &mut Vec<u8>, value: Option<u16>) {
    match value {
        Some(v) => {
            push_bool(out, true);
            push_u16(out, v);
        }
        None => push_bool(out, false),
    }
}

pub(crate) fn push_slice_u16(out: &mut Vec<u8>, values: &[u16]) {
    push_u32(out, values.len() as u32);
    for value in values {
        push_u16(out, *value);
    }
}
