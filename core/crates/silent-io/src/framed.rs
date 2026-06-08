//! Length-delimited stream framing for network transport.
//!
//! Each frame carries a [`FrameHeader`] (object type, flags, payload length)
//! followed by the raw payload bytes.  Frames are designed for use over TCP
//! or other byte-stream transports where individual messages are separated
//! by length prefixes.
//!
//! # Size limits
//!
//! A [`FramedReader`] enforces a configurable maximum frame payload length
//! (default: 64 KiB).  Frames declaring a larger payload are rejected at the
//! header stage without reading the payload.

use crate::error::IoError;
use crate::header::{DEFAULT_MAX_FRAME_PAYLOAD, FrameHeader};
use crate::primitives;
use crate::{CanonicalDecode, CanonicalEncode, DecodeWithParams, EncodeContext, ObjectKind};
use std::io::{Read, Write};

/// A builder-style writer that sends length-delimited frames over a byte sink.
pub struct FramedWriter<W: Write> {
    inner: W,
}

impl<W: Write> FramedWriter<W> {
    /// Wrap an existing writer.
    pub fn new(inner: W) -> Self {
        Self { inner }
    }

    /// Send an object as a single frame.
    ///
    /// The frame header carries `T::OBJECT_TYPE` and `ctx.flags`.
    pub fn send_object<T>(&mut self, object: &T, ctx: &EncodeContext) -> Result<usize, IoError>
    where
        T: CanonicalEncode + ObjectKind,
    {
        let payload = object.encode_to_vec()?;
        self.send_raw(T::OBJECT_TYPE, ctx.flags, &payload)
    }

    /// Send a raw payload with explicit object type and flags.
    ///
    /// This is the lowest-level write; it constructs the frame header and
    /// writes it followed by the payload bytes.
    pub fn send_raw(
        &mut self,
        object_type: crate::ObjectType,
        flags: u8,
        payload: &[u8],
    ) -> Result<usize, IoError> {
        let payload_len: u32 = payload
            .len()
            .try_into()
            .map_err(|_| IoError::InvalidEncoding {
                detail: format!("frame payload length {} exceeds u32::MAX", payload.len()),
            })?;

        let header = FrameHeader {
            object_type,
            flags,
            payload_len,
        };

        let mut written = header.encode_to(&mut self.inner)?;
        self.inner.write_all(payload)?;
        written += payload.len();
        Ok(written)
    }

    /// Flush the underlying writer.
    pub fn flush(&mut self) -> Result<(), IoError> {
        self.inner.flush()?;
        Ok(())
    }

    /// Consume and return the inner writer.
    pub fn into_inner(self) -> W {
        self.inner
    }
}

/// A reader that consumes length-delimited frames from a byte source.
pub struct FramedReader<R: Read> {
    inner: R,
    max_payload_len: u32,
}

impl<R: Read> FramedReader<R> {
    /// Wrap an existing reader with the default maximum frame payload (64 KiB).
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            max_payload_len: DEFAULT_MAX_FRAME_PAYLOAD,
        }
    }

    /// Set the maximum frame payload length (in bytes).
    pub fn with_max_payload(mut self, limit: u32) -> Self {
        self.max_payload_len = limit;
        self
    }

    /// Read the next frame header.
    ///
    /// The header payload length is validated against the configured maximum.
    /// Returns `Err` for oversized frames or I/O failures.
    pub fn read_header(&mut self) -> Result<FrameHeader, IoError> {
        let header = FrameHeader::decode_from(&mut self.inner)?;
        if header.payload_len > self.max_payload_len {
            return Err(IoError::PayloadTooLarge {
                declared: header.payload_len as u64,
                limit: self.max_payload_len as u64,
            });
        }
        Ok(header)
    }

    /// Read the payload for a previously-read header.
    ///
    /// The allocated buffer size is exactly `header.payload_len`.
    pub fn read_payload(&mut self, header: &FrameHeader) -> Result<Vec<u8>, IoError> {
        primitives::read_vec(&mut self.inner, header.payload_len as usize)
    }

    /// Read a full frame and decode the payload as `T`.
    ///
    /// Convenience wrapper around [`read_header`] + [`read_payload`] +
    /// [`CanonicalDecode::decode_from`].
    pub fn recv_object<T: CanonicalDecode + ObjectKind>(&mut self) -> Result<T, IoError> {
        let header = self.read_header()?;
        if header.object_type != T::OBJECT_TYPE {
            return Err(IoError::ObjectTypeMismatch {
                expected: T::OBJECT_TYPE,
                actual: header.object_type,
            });
        }
        let payload = self.read_payload(&header)?;
        T::decode_from(&mut std::io::Cursor::new(&payload))
    }

    /// Read a full frame and decode the payload as `T` with parameter context.
    pub fn recv_object_with_params<T, P>(&mut self, params: &P) -> Result<T, IoError>
    where
        T: DecodeWithParams<P> + ObjectKind,
    {
        let header = self.read_header()?;
        if header.object_type != T::OBJECT_TYPE {
            return Err(IoError::ObjectTypeMismatch {
                expected: T::OBJECT_TYPE,
                actual: header.object_type,
            });
        }
        let payload = self.read_payload(&header)?;
        T::decode_with_params(params, &mut std::io::Cursor::new(&payload))
    }

    /// Consume and return the inner reader.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalDecode, CanonicalEncode, IoError, ObjectKind, ObjectType};

    // A tiny roundtrippable type for testing.
    #[derive(Debug, PartialEq, Eq)]
    struct WireU32(u32);

    impl CanonicalEncode for WireU32 {
        fn encoded_len(&self) -> usize {
            4
        }
        fn encode_to<W: Write>(&self, writer: &mut W) -> Result<usize, IoError> {
            primitives::write_u32(writer, self.0)
        }
    }

    impl CanonicalDecode for WireU32 {
        fn decode_from<R: Read>(reader: &mut R) -> Result<Self, IoError> {
            primitives::read_u32(reader).map(WireU32)
        }
    }

    impl ObjectKind for WireU32 {
        const OBJECT_TYPE: ObjectType = ObjectType::POLY; // re-use tag for test
        const SERIALIZED_VERSION: u16 = 1;
    }

    #[test]
    fn framed_send_recv_object() {
        let mut buf = Vec::new();
        let mut writer = FramedWriter::new(&mut buf);

        let obj = WireU32(0xdead_beef);
        let ctx = EncodeContext::default();
        writer.send_object(&obj, &ctx).unwrap();
        writer.flush().unwrap();

        let mut reader = FramedReader::new(std::io::Cursor::new(&buf));
        let decoded: WireU32 = reader.recv_object().unwrap();
        assert_eq!(decoded, obj);
    }

    #[test]
    fn framed_recv_object_type_mismatch() {
        // Write a frame with a different object type.
        let mut buf = Vec::new();
        let header = FrameHeader {
            object_type: ObjectType::LWE_CIPHERTEXT,
            flags: 0,
            payload_len: 4,
        };
        header.encode_to(&mut buf).unwrap();
        buf.extend_from_slice(&42u32.to_le_bytes());

        let mut reader = FramedReader::new(std::io::Cursor::new(&buf));
        let err = reader.recv_object::<WireU32>().unwrap_err();
        assert!(matches!(err, IoError::ObjectTypeMismatch { .. }));
    }

    #[test]
    fn framed_rejects_oversized_frame() {
        let mut buf = Vec::new();
        let header = FrameHeader {
            object_type: ObjectType::POLY,
            flags: 0,
            payload_len: 128 * 1024, // 128 KiB > default 64 KiB
        };
        header.encode_to(&mut buf).unwrap();

        let mut reader = FramedReader::new(std::io::Cursor::new(&buf));
        let err = reader.read_header().unwrap_err();
        assert!(matches!(err, IoError::PayloadTooLarge { .. }));
    }

    #[test]
    fn framed_custom_max_payload() {
        let mut buf = Vec::new();
        let header = FrameHeader {
            object_type: ObjectType::POLY,
            flags: 0,
            payload_len: 128 * 1024,
        };
        header.encode_to(&mut buf).unwrap();

        let mut reader = FramedReader::new(std::io::Cursor::new(&buf)).with_max_payload(256 * 1024);
        let decoded_header = reader.read_header().unwrap();
        assert_eq!(decoded_header.payload_len, 128 * 1024);
    }

    #[test]
    fn framed_send_raw() {
        let mut buf = Vec::new();
        let mut writer = FramedWriter::new(&mut buf);
        writer
            .send_raw(ObjectType::RLWE_CIPHERTEXT, 0, b"payload")
            .unwrap();

        let mut reader = FramedReader::new(std::io::Cursor::new(&buf));
        let header = reader.read_header().unwrap();
        assert_eq!(header.object_type, ObjectType::RLWE_CIPHERTEXT);
        assert_eq!(header.payload_len, 7);
        let payload = reader.read_payload(&header).unwrap();
        assert_eq!(payload, b"payload");
    }

    #[test]
    fn framed_empty_payload() {
        let mut buf = Vec::new();
        let mut writer = FramedWriter::new(&mut buf);
        writer.send_raw(ObjectType::POLY, 0, b"").unwrap();

        let mut reader = FramedReader::new(std::io::Cursor::new(&buf));
        let header = reader.read_header().unwrap();
        assert_eq!(header.payload_len, 0);
        let payload = reader.read_payload(&header).unwrap();
        assert!(payload.is_empty());
    }
}
