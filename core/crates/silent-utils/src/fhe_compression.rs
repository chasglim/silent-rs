//! Lossless compression primitives for large FHE/RLWE limb buffers.
//!
//! The first supported codec is intentionally scheme-independent: it compresses
//! `u64` coefficient limbs with zero-run encoding and falls back to raw little
//! endian bytes when the input is dense.  Scheme crates can wrap this byte
//! format with their own object headers in `silent-io`.

use thiserror::Error;

const MAGIC: &[u8; 4] = b"SFC1";
const HEADER_LEN: usize = 4 + 1 + 8 + 8;
const TAG_ZERO_RUN: u8 = 0;
const TAG_LITERAL_RUN: u8 = 1;

/// Compression mode stored in the byte header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CompressionMode {
    /// Raw `u64` little-endian limbs.
    RawU64Le = 0,
    /// Zero-run encoding over `u64` little-endian limbs.
    ZeroRunU64Le = 1,
}

impl CompressionMode {
    fn from_u8(value: u8) -> Result<Self, CompressionError> {
        match value {
            0 => Ok(Self::RawU64Le),
            1 => Ok(Self::ZeroRunU64Le),
            other => Err(CompressionError::UnsupportedMode(other)),
        }
    }
}

/// Errors returned while decoding compressed limb buffers.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CompressionError {
    /// The byte stream is shorter than the SILENT compression header.
    #[error("compressed buffer is too short")]
    TooShort,
    /// The stream does not start with the SILENT compression magic.
    #[error("compressed buffer has invalid magic")]
    InvalidMagic,
    /// The stream advertises an unknown compression mode.
    #[error("unsupported compression mode {0}")]
    UnsupportedMode(u8),
    /// The encoded payload length does not match the surrounding byte buffer.
    #[error("compressed payload length is inconsistent with the buffer")]
    LengthMismatch,
    /// The payload ended in the middle of a run.
    #[error("compressed payload is truncated")]
    Truncated,
    /// A run has zero length.
    #[error("compressed payload contains an empty run")]
    EmptyRun,
    /// The decoded limb count differs from the header.
    #[error("decoded limb count does not match the header")]
    DecodedLenMismatch,
}

/// Compress a `u64` limb buffer.
///
/// The function chooses the smaller of raw little-endian bytes and zero-run
/// encoding.  The result is self-describing and can be decoded with
/// [`decompress_u64_limbs`].
pub fn compress_u64_limbs(limbs: &[u64]) -> Vec<u8> {
    let raw_payload = raw_payload(limbs);
    let rle_payload = zero_run_payload(limbs);
    let (mode, payload) = if rle_payload.len() < raw_payload.len() {
        (CompressionMode::ZeroRunU64Le, rle_payload)
    } else {
        (CompressionMode::RawU64Le, raw_payload)
    };

    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(MAGIC);
    out.push(mode as u8);
    out.extend_from_slice(&(limbs.len() as u64).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

/// Decode a buffer produced by [`compress_u64_limbs`].
pub fn decompress_u64_limbs(bytes: &[u8]) -> Result<Vec<u64>, CompressionError> {
    if bytes.len() < HEADER_LEN {
        return Err(CompressionError::TooShort);
    }
    if &bytes[..4] != MAGIC {
        return Err(CompressionError::InvalidMagic);
    }

    let mode = CompressionMode::from_u8(bytes[4])?;
    let limb_count = u64::from_le_bytes(bytes[5..13].try_into().unwrap()) as usize;
    let payload_len = u64::from_le_bytes(bytes[13..21].try_into().unwrap()) as usize;
    let payload_end = HEADER_LEN
        .checked_add(payload_len)
        .ok_or(CompressionError::LengthMismatch)?;
    if payload_end != bytes.len() {
        return Err(CompressionError::LengthMismatch);
    }
    let payload = &bytes[HEADER_LEN..payload_end];

    match mode {
        CompressionMode::RawU64Le => decode_raw_payload(payload, limb_count),
        CompressionMode::ZeroRunU64Le => decode_zero_run_payload(payload, limb_count),
    }
}

fn raw_payload(limbs: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(limbs.len() * 8);
    for &limb in limbs {
        out.extend_from_slice(&limb.to_le_bytes());
    }
    out
}

fn decode_raw_payload(payload: &[u8], limb_count: usize) -> Result<Vec<u64>, CompressionError> {
    if payload.len()
        != limb_count
            .checked_mul(8)
            .ok_or(CompressionError::LengthMismatch)?
    {
        return Err(CompressionError::LengthMismatch);
    }
    let mut out = Vec::with_capacity(limb_count);
    for chunk in payload.chunks_exact(8) {
        out.push(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

fn zero_run_payload(limbs: &[u64]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < limbs.len() {
        if limbs[index] == 0 {
            let start = index;
            while index < limbs.len() && limbs[index] == 0 && index - start < u32::MAX as usize {
                index += 1;
            }
            push_run_header(&mut out, TAG_ZERO_RUN, index - start);
        } else {
            let start = index;
            while index < limbs.len() && limbs[index] != 0 && index - start < u32::MAX as usize {
                index += 1;
            }
            push_run_header(&mut out, TAG_LITERAL_RUN, index - start);
            for &limb in &limbs[start..index] {
                out.extend_from_slice(&limb.to_le_bytes());
            }
        }
    }
    out
}

fn push_run_header(out: &mut Vec<u8>, tag: u8, len: usize) {
    debug_assert!(len > 0);
    debug_assert!(u32::try_from(len).is_ok());
    out.push(tag);
    out.extend_from_slice(&(len as u32).to_le_bytes());
}

fn decode_zero_run_payload(
    payload: &[u8],
    limb_count: usize,
) -> Result<Vec<u64>, CompressionError> {
    let mut out = Vec::with_capacity(limb_count);
    let mut cursor = 0usize;
    while cursor < payload.len() {
        if cursor + 5 > payload.len() {
            return Err(CompressionError::Truncated);
        }
        let tag = payload[cursor];
        cursor += 1;
        let run_len = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if run_len == 0 {
            return Err(CompressionError::EmptyRun);
        }
        if out.len() + run_len > limb_count {
            return Err(CompressionError::DecodedLenMismatch);
        }

        match tag {
            TAG_ZERO_RUN => out.resize(out.len() + run_len, 0),
            TAG_LITERAL_RUN => {
                let byte_len = run_len
                    .checked_mul(8)
                    .ok_or(CompressionError::LengthMismatch)?;
                if cursor + byte_len > payload.len() {
                    return Err(CompressionError::Truncated);
                }
                for chunk in payload[cursor..cursor + byte_len].chunks_exact(8) {
                    out.push(u64::from_le_bytes(chunk.try_into().unwrap()));
                }
                cursor += byte_len;
            }
            other => return Err(CompressionError::UnsupportedMode(other)),
        }
    }

    if out.len() == limb_count {
        Ok(out)
    } else {
        Err(CompressionError::DecodedLenMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_payload_roundtrips_as_raw() {
        let limbs = (0..64)
            .map(|i| 0xfeed_0000_0000_0000u64 | i)
            .collect::<Vec<_>>();
        let encoded = compress_u64_limbs(&limbs);
        assert_eq!(
            CompressionMode::from_u8(encoded[4]).unwrap(),
            CompressionMode::RawU64Le
        );
        assert_eq!(decompress_u64_limbs(&encoded).unwrap(), limbs);
    }

    #[test]
    fn sparse_payload_roundtrips_smaller_than_raw() {
        let mut limbs = vec![0u64; 256];
        limbs[7] = 1;
        limbs[128] = u64::MAX;
        let encoded = compress_u64_limbs(&limbs);
        assert_eq!(
            CompressionMode::from_u8(encoded[4]).unwrap(),
            CompressionMode::ZeroRunU64Le
        );
        assert!(encoded.len() < HEADER_LEN + limbs.len() * 8);
        assert_eq!(decompress_u64_limbs(&encoded).unwrap(), limbs);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut encoded = compress_u64_limbs(&[1, 2, 3]);
        encoded[0] = b'X';
        assert_eq!(
            decompress_u64_limbs(&encoded),
            Err(CompressionError::InvalidMagic)
        );
    }

    #[test]
    fn rejects_truncated_run() {
        let mut encoded = compress_u64_limbs(&[0; 16]);
        encoded.truncate(encoded.len() - 1);
        assert_eq!(
            decompress_u64_limbs(&encoded),
            Err(CompressionError::LengthMismatch)
        );
    }
}
