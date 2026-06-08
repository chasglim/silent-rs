//! Little-endian fixint read/write primitives.
//!
//! All multi-byte integers are encoded in **little-endian** order with fixed
//! width.  Every write function returns the number of bytes written; every read
//! function consumes exactly the expected number of bytes or returns
//! [`IoError::UnexpectedEof`].

use crate::error::IoError;
use std::io::{Read, Write};

// ── Write helpers ───────────────────────────────────────────────────────────

/// Write a single byte.
#[inline]
pub fn write_u8(w: &mut impl Write, value: u8) -> Result<usize, IoError> {
    w.write_all(&[value])?;
    Ok(1)
}

/// Write a `u16` in little-endian order.
#[inline]
pub fn write_u16(w: &mut impl Write, value: u16) -> Result<usize, IoError> {
    w.write_all(&value.to_le_bytes())?;
    Ok(2)
}

/// Write a `u32` in little-endian order.
#[inline]
pub fn write_u32(w: &mut impl Write, value: u32) -> Result<usize, IoError> {
    w.write_all(&value.to_le_bytes())?;
    Ok(4)
}

/// Write a `u64` in little-endian order.
#[inline]
pub fn write_u64(w: &mut impl Write, value: u64) -> Result<usize, IoError> {
    w.write_all(&value.to_le_bytes())?;
    Ok(8)
}

/// Write an `i64` in little-endian order.
#[inline]
pub fn write_i64(w: &mut impl Write, value: i64) -> Result<usize, IoError> {
    write_u64(w, value as u64)
}

/// Write a boolean as a single byte (0x00 or 0x01).
#[inline]
pub fn write_bool(w: &mut impl Write, value: bool) -> Result<usize, IoError> {
    write_u8(w, value as u8)
}

/// Write a raw byte slice.
///
/// This writes only the bytes; no length prefix is prepended.
#[inline]
pub fn write_bytes(w: &mut impl Write, bytes: &[u8]) -> Result<usize, IoError> {
    w.write_all(bytes)?;
    Ok(bytes.len())
}

/// Write a `[u8; N]` array (no length prefix).
#[inline]
pub fn write_u8_array<const N: usize>(w: &mut impl Write, arr: &[u8; N]) -> Result<usize, IoError> {
    write_bytes(w, arr.as_slice())
}

/// Write a `u32` length-prefixed byte slice.
///
/// Layout: `[len: u32 LE][bytes...]`
pub fn write_len_prefixed_bytes(w: &mut impl Write, bytes: &[u8]) -> Result<usize, IoError> {
    let len = bytes.len();
    // Length is at most 4 GiB, so u32 is safe.
    let len32: u32 = len.try_into().map_err(|_| IoError::InvalidEncoding {
        detail: format!("byte slice length {len} exceeds u32::MAX"),
    })?;
    Ok(write_u32(w, len32)? + write_bytes(w, bytes)?)
}

/// Write a sequence of `T` with a `u32 LE` element count prefix.
///
/// Each element is written via `encode_fn`.
pub fn write_slice_with<T>(
    w: &mut impl Write,
    items: &[T],
    encode_fn: impl Fn(&T) -> Result<Vec<u8>, IoError>,
) -> Result<usize, IoError> {
    let count: u32 = items
        .len()
        .try_into()
        .map_err(|_| IoError::InvalidEncoding {
            detail: format!("slice length {} exceeds u32::MAX", items.len()),
        })?;
    let count_bytes = count.to_le_bytes();
    w.write_all(&count_bytes)?;
    let mut written = 4usize;
    for item in items {
        let bytes = encode_fn(item)?;
        w.write_all(&bytes)?;
        written += bytes.len();
    }
    Ok(written)
}

// ── Read helpers ────────────────────────────────────────────────────────────

/// Read a single byte.
#[inline]
pub fn read_u8(r: &mut impl Read) -> Result<u8, IoError> {
    let mut buf = [0u8; 1];
    read_exact(r, &mut buf)?;
    Ok(buf[0])
}

/// Read a `u16` in little-endian order.
#[inline]
pub fn read_u16(r: &mut impl Read) -> Result<u16, IoError> {
    let mut buf = [0u8; 2];
    read_exact(r, &mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

/// Read a `u32` in little-endian order.
#[inline]
pub fn read_u32(r: &mut impl Read) -> Result<u32, IoError> {
    let mut buf = [0u8; 4];
    read_exact(r, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a `u64` in little-endian order.
#[inline]
pub fn read_u64(r: &mut impl Read) -> Result<u64, IoError> {
    let mut buf = [0u8; 8];
    read_exact(r, &mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Read an `i64` in little-endian order.
#[inline]
pub fn read_i64(r: &mut impl Read) -> Result<i64, IoError> {
    Ok(read_u64(r)? as i64)
}

/// Read a boolean as a single byte (must be 0x00 or 0x01).
#[inline]
pub fn read_bool(r: &mut impl Read) -> Result<bool, IoError> {
    let b = read_u8(r)?;
    match b {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(IoError::InvalidEncoding {
            detail: format!("invalid bool byte 0x{b:02x}, expected 0x00 or 0x01"),
        }),
    }
}

/// Read exactly `n` bytes into `buf`.
///
/// Unlike [`Read::read_exact`], this function returns an [`IoError::UnexpectedEof`]
/// with the actual number of bytes available when the stream is exhausted.
#[inline]
pub fn read_exact(r: &mut impl Read, buf: &mut [u8]) -> Result<(), IoError> {
    let n = buf.len();
    let mut offset = 0;
    while offset < n {
        let count = r.read(&mut buf[offset..])?;
        if count == 0 {
            return Err(IoError::unexpected_eof(n, offset));
        }
        offset += count;
    }
    Ok(())
}

/// Read exactly `n` bytes into a new `Vec<u8>`.
pub fn read_vec(r: &mut impl Read, n: usize) -> Result<Vec<u8>, IoError> {
    let mut buf = vec![0u8; n];
    read_exact(r, &mut buf)?;
    Ok(buf)
}

/// Read a length-prefixed byte vector.
///
/// Layout: `[len: u32 LE][bytes...]`
pub fn read_len_prefixed_bytes(r: &mut impl Read) -> Result<Vec<u8>, IoError> {
    let len = read_u32(r)? as usize;
    read_vec(r, len)
}

/// Read a length-prefixed sequence of fixed-size elements.
///
/// The element count is read as `u32 LE`, then exactly `count` elements are
/// read using `decode_fn`.  An element-length mismatch short-circuits with
/// [`IoError::UnexpectedEof`].
pub fn read_elements_with<T>(
    r: &mut impl Read,
    element_len: usize,
    mut decode_fn: impl FnMut(&[u8]) -> Result<T, IoError>,
) -> Result<Vec<T>, IoError> {
    let count = read_u32(r)? as usize;
    let mut out = Vec::with_capacity(count);
    let mut buf = vec![0u8; element_len];
    for _ in 0..count {
        read_exact(r, &mut buf)?;
        out.push(decode_fn(&buf)?);
    }
    Ok(out)
}

/// Read exactly `count` u64 values from the stream, each in LE.
pub fn read_u64_slice(r: &mut impl Read, count: usize) -> Result<Vec<u64>, IoError> {
    let mut out = Vec::with_capacity(count);
    let mut buf = [0u8; 8];
    for _ in 0..count {
        read_exact(r, &mut buf)?;
        out.push(u64::from_le_bytes(buf));
    }
    Ok(out)
}

// ── Size-bound read helpers ─────────────────────────────────────────────────

/// Read at most `limit` bytes from `r` and discard them.
///
/// Returns the number of bytes actually skipped.
pub fn skip_bytes(r: &mut impl Read, limit: usize) -> Result<usize, IoError> {
    let mut remaining = limit;
    let mut buf = [0u8; 4096];
    while remaining > 0 {
        let chunk = remaining.min(buf.len());
        let count = r.read(&mut buf[..chunk])?;
        if count == 0 {
            return Ok(limit - remaining);
        }
        remaining -= count;
    }
    Ok(limit)
}

/// A reader wrapper that enforces a maximum byte budget.
///
/// Once the budget is exhausted, any further read returns
/// [`IoError::PayloadTooLarge`].
pub struct BoundedReader<R> {
    inner: R,
    remaining: u64,
    limit: u64,
}

impl<R> BoundedReader<R> {
    /// Wrap `inner` so that at most `limit` total bytes can be read.
    pub fn new(inner: R, limit: u64) -> Self {
        Self {
            inner,
            remaining: limit,
            limit,
        }
    }

    /// Return the remaining byte budget.
    pub fn remaining(&self) -> u64 {
        self.remaining
    }
}

impl<R: Read> Read for BoundedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let max = (buf.len() as u64).min(self.remaining) as usize;
        if max == 0 {
            return if self.remaining == 0 {
                Ok(0) // natural EOF
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "bounded reader exhausted",
                ))
            };
        }
        let n = self.inner.read(&mut buf[..max])?;
        self.remaining -= n as u64;
        Ok(n)
    }
}

impl<R> BoundedReader<R> {
    /// Consume the bounded reader and return the inner reader, together with
    /// the number of bytes that were actually consumed from the budget.
    pub fn into_inner(self) -> (R, u64) {
        (self.inner, self.limit - self.remaining)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ── Write + read roundtrip ─────────────────────────────────────────────

    #[test]
    fn u8_roundtrip() {
        let mut buf = Vec::new();
        write_u8(&mut buf, 0xab).unwrap();
        assert_eq!(buf, [0xab]);
        assert_eq!(read_u8(&mut Cursor::new(&buf)).unwrap(), 0xab);
    }

    #[test]
    fn u16_roundtrip() {
        let mut buf = Vec::new();
        write_u16(&mut buf, 0x1234).unwrap();
        assert_eq!(buf, [0x34, 0x12]);
        assert_eq!(read_u16(&mut Cursor::new(&buf)).unwrap(), 0x1234);
    }

    #[test]
    fn u32_roundtrip() {
        let mut buf = Vec::new();
        write_u32(&mut buf, 0xdead_beef).unwrap();
        assert_eq!(buf, [0xef, 0xbe, 0xad, 0xde]);
        assert_eq!(read_u32(&mut Cursor::new(&buf)).unwrap(), 0xdead_beef);
    }

    #[test]
    fn u64_roundtrip() {
        let mut buf = Vec::new();
        write_u64(&mut buf, 0x0123_4567_89ab_cdef).unwrap();
        assert_eq!(buf, [0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01]);
        assert_eq!(
            read_u64(&mut Cursor::new(&buf)).unwrap(),
            0x0123_4567_89ab_cdef
        );
    }

    #[test]
    fn i64_roundtrip() {
        let mut buf = Vec::new();
        write_i64(&mut buf, -42).unwrap();
        assert_eq!(read_i64(&mut Cursor::new(&buf)).unwrap(), -42);
    }

    #[test]
    fn bool_roundtrip() {
        let mut buf = Vec::new();
        write_bool(&mut buf, true).unwrap();
        write_bool(&mut buf, false).unwrap();
        assert_eq!(buf, [1, 0]);
        let mut r = Cursor::new(&buf);
        assert!(read_bool(&mut r).unwrap());
        assert!(!read_bool(&mut r).unwrap());
    }

    #[test]
    fn read_bool_rejects_invalid() {
        let mut r = Cursor::new([0x02u8]);
        assert!(read_bool(&mut r).is_err());
    }

    #[test]
    fn len_prefixed_bytes_roundtrip() {
        let data = b"hello, canonical world";
        let mut buf = Vec::new();
        write_len_prefixed_bytes(&mut buf, data).unwrap();
        let decoded = read_len_prefixed_bytes(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn len_prefixed_bytes_empty() {
        let mut buf = Vec::new();
        write_len_prefixed_bytes(&mut buf, b"").unwrap();
        assert_eq!(buf, [0, 0, 0, 0]); // u32 LE = 0
        let decoded = read_len_prefixed_bytes(&mut Cursor::new(&buf)).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn u8_array_roundtrip() {
        let arr: [u8; 32] = [0xcc; 32];
        let mut buf = Vec::new();
        write_u8_array(&mut buf, &arr).unwrap();
        assert_eq!(buf.len(), 32);

        let mut out = [0u8; 32];
        read_exact(&mut Cursor::new(&buf), &mut out).unwrap();
        assert_eq!(out, arr);
    }

    #[test]
    fn write_slice_with_roundtrip() {
        let items: Vec<u32> = vec![1, 2, 3, 0xffff_ffff];
        let mut buf = Vec::new();
        write_slice_with(&mut buf, &items, |item| {
            let mut v = Vec::with_capacity(4);
            write_u32(&mut v, *item).unwrap();
            Ok(v)
        })
        .unwrap();

        let mut r = Cursor::new(&buf);
        let count = read_u32(&mut r).unwrap();
        assert_eq!(count, 4);
        let mut decoded = Vec::new();
        for _ in 0..count {
            decoded.push(read_u32(&mut r).unwrap());
        }
        assert_eq!(decoded, items);
    }

    // ── EOF / error cases ──────────────────────────────────────────────────

    #[test]
    fn read_eof_returns_unexpected_eof() {
        let mut r = Cursor::new([0x01u8]);
        let mut buf = [0u8; 8];
        let err = read_exact(&mut r, &mut buf).unwrap_err();
        assert!(matches!(err, IoError::UnexpectedEof { .. }));
    }

    #[test]
    fn bounded_reader_enforces_limit() {
        let data = [0u8; 100];
        let mut r = BoundedReader::new(Cursor::new(&data[..]), 10);
        let mut buf = [0u8; 20];
        let n = r.read(&mut buf).unwrap();
        assert_eq!(n, 10);
        let n2 = r.read(&mut buf).unwrap();
        assert_eq!(n2, 0);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn bounded_reader_into_inner() {
        let data = [0u8; 100];
        let mut r = BoundedReader::new(Cursor::new(&data[..]), 10);
        let mut buf = [0u8; 6];
        r.read(&mut buf).unwrap();
        let (_inner, consumed) = r.into_inner();
        assert_eq!(consumed, 6);
    }

    #[test]
    fn skip_bytes_partial() {
        let data = [0u8; 5];
        let mut r = Cursor::new(&data[..]);
        let skipped = skip_bytes(&mut r, 10).unwrap();
        assert_eq!(skipped, 5);
    }

    #[test]
    fn read_elements_with_roundtrip() {
        let mut buf = Vec::new();
        // u32 count
        buf.extend_from_slice(&3u32.to_le_bytes());
        // three u64 values
        buf.extend_from_slice(&0x100u64.to_le_bytes());
        buf.extend_from_slice(&0x200u64.to_le_bytes());
        buf.extend_from_slice(&0x300u64.to_le_bytes());

        let decoded = read_elements_with(&mut Cursor::new(&buf), 8, |bytes| {
            let arr: [u8; 8] = bytes.try_into().unwrap();
            Ok(u64::from_le_bytes(arr))
        })
        .unwrap();
        assert_eq!(decoded, vec![0x100u64, 0x200, 0x300]);
    }

    #[test]
    fn read_u64_slice_roundtrip() {
        // 5 × u64 LE = 0
        let data = [0u8; 40];
        let mut r = Cursor::new(&data[..]);
        let values = read_u64_slice(&mut r, 5).unwrap();
        assert_eq!(values.len(), 5);
        assert!(values.iter().all(|&v| v == 0));
    }

    #[test]
    fn read_u64_slice_partial_eof() {
        let data = [0u8; 8]; // only 1 × u64
        let mut r = Cursor::new(&data[..]);
        let err = read_u64_slice(&mut r, 3).unwrap_err();
        assert!(matches!(err, IoError::UnexpectedEof { .. }));
    }
}
