//! File container I/O with magic bytes, versioning, and XXH3 checksum.
//!
//! A SILENT file container is a self-describing envelope:
//!
//! ```text
//! ┌────────────────┐
//! │ ContainerHeader│ 56 bytes
//! ├────────────────┤
//! │ payload        │  variable
//! ├────────────────┤
//! │ XXH3 checksum  │  8 bytes (covers header + payload)
//! └────────────────┘
//! ```
//!
//! The checksum covers the serialised header bytes followed by the payload
//! bytes.  The checksum field itself is excluded from the hash input.

use crate::error::IoError;
use crate::header::{ContainerHeader, DEFAULT_MAX_CONTAINER_PAYLOAD};
use crate::primitives;
use crate::validate;
use crate::xxh3::Xxh3State;
use crate::{CanonicalDecode, CanonicalEncode, DecodeWithParams, EncodeContext, ObjectKind};
use std::io::{Read, Write};

/// Writes a single object into a SILENT container.
///
/// # Example
///
/// ```ignore
/// let mut file = std::fs::File::create("key.sfir")?;
/// let mut writer = ContainerWriter::new(&mut file);
/// writer.write_object(&key, &ctx)?;
/// writer.finish()?;
/// ```
pub struct ContainerWriter<W: Write> {
    inner: W,
    hasher: Xxh3State,
    bytes_written: u64,
}

impl<W: Write> ContainerWriter<W> {
    /// Create a new container writer wrapping `inner`.
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Xxh3State::new(0),
            bytes_written: 0,
        }
    }

    /// Write an object as the payload of a new container.
    ///
    /// The container header is constructed from `T::OBJECT_TYPE`,
    /// `T::SERIALIZED_VERSION`, and the supplied `ctx`.  The payload is the
    /// canonical encoding of `object`.  An XXH3 checksum is appended.
    ///
    /// Returns the total number of bytes written (header + payload + checksum).
    pub fn write_object<T>(&mut self, object: &T, ctx: &EncodeContext) -> Result<usize, IoError>
    where
        T: CanonicalEncode + ObjectKind,
    {
        let payload = object.encode_to_vec()?;
        self.write_raw(T::OBJECT_TYPE, T::SERIALIZED_VERSION, ctx, &payload)
    }

    /// Write a raw payload with explicit type metadata.
    ///
    /// This is the lowest-level write operation.  It constructs the header,
    /// writes header + payload through the hasher, then appends the checksum.
    pub fn write_raw(
        &mut self,
        object_type: crate::ObjectType,
        object_version: u16,
        ctx: &EncodeContext,
        payload: &[u8],
    ) -> Result<usize, IoError> {
        let payload_len: u64 = payload
            .len()
            .try_into()
            .map_err(|_| IoError::InvalidEncoding {
                detail: format!(
                    "container payload length {} exceeds u64 range",
                    payload.len()
                ),
            })?;

        let header = ContainerHeader {
            format_version: crate::FORMAT_VERSION,
            object_type,
            scheme_id: ctx.scheme_id,
            params_id: ctx.params_id,
            flags: ctx.flags,
            object_version,
            payload_len,
        };

        let header_bytes = header.encode_to_vec()?;

        // Feed header into hasher.
        self.hasher.update(&header_bytes);
        // Feed payload into hasher.
        self.hasher.update(payload);
        // Finalise *before* writing to avoid hashing the checksum itself.
        let checksum = self.hasher.finish();

        let mut total = 0usize;

        // Write header.
        self.inner.write_all(&header_bytes)?;
        total += header_bytes.len();
        self.bytes_written += header_bytes.len() as u64;

        // Write payload.
        self.inner.write_all(payload)?;
        total += payload.len();
        self.bytes_written += payload.len() as u64;

        // Write checksum.
        let checksum_bytes = checksum.to_le_bytes();
        self.inner.write_all(&checksum_bytes)?;
        total += checksum_bytes.len();
        self.bytes_written += checksum_bytes.len() as u64;

        Ok(total)
    }

    /// Return the total number of bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
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

/// Reads a single object from a SILENT container.
///
/// # Example
///
/// ```ignore
/// let mut file = std::fs::File::open("key.sfir")?;
/// let mut reader = ContainerReader::new(&mut file);
/// let header = reader.read_header()?;
/// let key: SecretKey = reader.read_payload_with_params(&header, &params)?;
/// ```
pub struct ContainerReader<R: Read> {
    inner: R,
    max_payload_len: u64,
}

impl<R: Read> ContainerReader<R> {
    /// Create a new container reader with the default payload limit (8 MiB).
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            max_payload_len: DEFAULT_MAX_CONTAINER_PAYLOAD,
        }
    }

    /// Override the maximum payload length.
    pub fn with_max_payload(mut self, limit: u64) -> Self {
        self.max_payload_len = limit;
        self
    }

    /// Read and validate the container header.
    ///
    /// Validates magic bytes, format version, and payload length against the
    /// configured maximum.
    pub fn read_header(&mut self) -> Result<ContainerHeader, IoError> {
        let header = ContainerHeader::decode_from(&mut self.inner)?;
        validate::validate_payload_len(header.payload_len, self.max_payload_len)?;
        Ok(header)
    }

    /// Read the payload bytes (governed by `header.payload_len`).
    pub fn read_payload(&mut self, header: &ContainerHeader) -> Result<Vec<u8>, IoError> {
        let len: usize = header
            .payload_len
            .try_into()
            .map_err(|_| IoError::InvalidEncoding {
                detail: format!(
                    "container payload length {} exceeds usize on this platform",
                    header.payload_len
                ),
            })?;
        primitives::read_vec(&mut self.inner, len)
    }

    /// Verify the container checksum.
    ///
    /// Must be called after [`read_payload`].  The checksum is computed over
    /// the serialised header bytes followed by the payload bytes.
    pub fn verify_checksum(
        &mut self,
        header: &ContainerHeader,
        payload: &[u8],
    ) -> Result<(), IoError> {
        let expected = primitives::read_u64(&mut self.inner)?;

        let mut hasher = Xxh3State::new(0);
        hasher.update(&header.encode_to_vec()?);
        hasher.update(payload);
        let actual = hasher.finish();

        if expected != actual {
            return Err(IoError::ChecksumMismatch { expected, actual });
        }
        Ok(())
    }

    /// High-level: read header, payload, verify checksum, decode as `T`.
    pub fn read_object<T: CanonicalDecode + ObjectKind>(&mut self) -> Result<T, IoError> {
        let header = self.read_header()?;
        if header.object_type != T::OBJECT_TYPE {
            return Err(IoError::ObjectTypeMismatch {
                expected: T::OBJECT_TYPE,
                actual: header.object_type,
            });
        }
        if header.object_version > T::SERIALIZED_VERSION {
            return Err(IoError::ObjectVersionTooNew {
                data_version: header.object_version,
                lib_version: T::SERIALIZED_VERSION,
            });
        }
        let payload = self.read_payload(&header)?;
        self.verify_checksum(&header, &payload)?;
        T::decode_from(&mut std::io::Cursor::new(&payload))
    }

    /// High-level: read header, payload, verify checksum, decode with params.
    pub fn read_object_with_params<T, P>(&mut self, params: &P) -> Result<T, IoError>
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
        if header.object_version > T::SERIALIZED_VERSION {
            return Err(IoError::ObjectVersionTooNew {
                data_version: header.object_version,
                lib_version: T::SERIALIZED_VERSION,
            });
        }
        let payload = self.read_payload(&header)?;
        self.verify_checksum(&header, &payload)?;
        T::decode_with_params(params, &mut std::io::Cursor::new(&payload))
    }

    /// High-level: like [`read_object_with_params`] but additionally validates
    /// that the container header's `params_id` matches the caller's parameter
    /// set.
    ///
    /// If the header carries a non-zero `params_id` that differs from
    /// `params.params_id()`, [`IoError::ParamsIdMismatch`] is returned
    /// **before** the payload is decoded.
    pub fn read_object_with_params_checked<T, P>(&mut self, params: &P) -> Result<T, IoError>
    where
        T: DecodeWithParams<P> + ObjectKind,
        P: crate::HasParamsId,
    {
        let header = self.read_header()?;
        if header.object_type != T::OBJECT_TYPE {
            return Err(IoError::ObjectTypeMismatch {
                expected: T::OBJECT_TYPE,
                actual: header.object_type,
            });
        }
        if header.object_version > T::SERIALIZED_VERSION {
            return Err(IoError::ObjectVersionTooNew {
                data_version: header.object_version,
                lib_version: T::SERIALIZED_VERSION,
            });
        }
        // Validate params_id: skip if header carries the all-zero sentinel
        // (meaning "no parameter binding").
        if !header.params_id.is_zero() {
            let expected_id = params.params_id();
            let io_expected = crate::ParamsId(expected_id.0);
            if header.params_id != io_expected {
                return Err(IoError::ParamsIdMismatch {
                    expected: format!("{io_expected}"),
                    actual: format!("{}", header.params_id),
                });
            }
        }
        let payload = self.read_payload(&header)?;
        self.verify_checksum(&header, &payload)?;
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
    use crate::primitives;
    use crate::{CanonicalDecode, CanonicalEncode, IoError, ObjectKind, ObjectType};

    #[derive(Clone, Debug, PartialEq, Eq)]
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
        const OBJECT_TYPE: ObjectType = ObjectType::POLY; // reuse tag for test
        const SERIALIZED_VERSION: u16 = 1;
    }

    // ── Roundtrip ──────────────────────────────────────────────────────────

    #[test]
    fn container_roundtrip() {
        let mut buf = Vec::new();
        let ctx = EncodeContext::default();

        // Write
        {
            let mut writer = ContainerWriter::new(&mut buf);
            writer.write_object(&WireU32(0xdead_beef), &ctx).unwrap();
        }

        // Read
        let mut reader = ContainerReader::new(std::io::Cursor::new(&buf));
        let decoded: WireU32 = reader.read_object().unwrap();
        assert_eq!(decoded, WireU32(0xdead_beef));
    }

    #[test]
    fn container_empty_payload() {
        let mut buf = Vec::new();
        let ctx = EncodeContext::default();

        {
            let mut writer = ContainerWriter::new(&mut buf);
            writer.write_raw(ObjectType::POLY, 1, &ctx, b"").unwrap();
        }

        let mut reader = ContainerReader::new(std::io::Cursor::new(&buf));
        let header = reader.read_header().unwrap();
        assert_eq!(header.payload_len, 0);
        let payload = reader.read_payload(&header).unwrap();
        assert!(payload.is_empty());
        reader.verify_checksum(&header, &payload).unwrap();
    }

    // ── Error cases ────────────────────────────────────────────────────────

    #[test]
    fn container_bad_magic() {
        let buf = vec![0u8; 64]; // all zeros, bad magic
        let mut reader = ContainerReader::new(std::io::Cursor::new(&buf));
        let err = reader.read_header().unwrap_err();
        assert!(matches!(err, IoError::InvalidMagic { .. }));
    }

    #[test]
    fn container_bad_checksum() {
        let mut buf = Vec::new();
        let ctx = EncodeContext::default();

        {
            let mut writer = ContainerWriter::new(&mut buf);
            writer.write_object(&WireU32(42), &ctx).unwrap();
        }

        // Corrupt the payload byte.
        let header_len = crate::header::CONTAINER_HEADER_SIZE;
        buf[header_len] ^= 0xff;

        let mut reader = ContainerReader::new(std::io::Cursor::new(&buf));
        let header = reader.read_header().unwrap();
        let payload = reader.read_payload(&header).unwrap();
        let err = reader.verify_checksum(&header, &payload).unwrap_err();
        assert!(matches!(err, IoError::ChecksumMismatch { .. }));
    }

    #[test]
    fn container_oversized_payload() {
        let mut buf = Vec::new();
        let ctx = EncodeContext::default();

        // Write a 256-byte payload.
        let payload = vec![0xabu8; 256];
        {
            let mut writer = ContainerWriter::new(&mut buf);
            writer
                .write_raw(ObjectType::POLY, 1, &ctx, &payload)
                .unwrap();
        }

        // Read with a 128-byte limit.
        let mut reader = ContainerReader::new(std::io::Cursor::new(&buf)).with_max_payload(128);
        let err = reader.read_header().unwrap_err();
        assert!(matches!(err, IoError::PayloadTooLarge { .. }));
    }

    #[test]
    fn container_object_version_too_new() {
        let mut buf = Vec::new();
        let ctx = EncodeContext::default();

        {
            let mut writer = ContainerWriter::new(&mut buf);
            // Write with version 99 (newer than WireU32::SERIALIZED_VERSION = 1).
            writer
                .write_raw(ObjectType::POLY, 99, &ctx, &42u32.to_le_bytes())
                .unwrap();
        }

        let mut reader = ContainerReader::new(std::io::Cursor::new(&buf));
        let err = reader.read_object::<WireU32>().unwrap_err();
        assert!(matches!(err, IoError::ObjectVersionTooNew { .. }));
    }

    #[test]
    fn container_bytes_written_tracking() {
        let mut buf = Vec::new();
        let mut writer = ContainerWriter::new(&mut buf);
        let ctx = EncodeContext::default();
        writer.write_object(&WireU32(0), &ctx).unwrap();
        let total = writer.bytes_written();
        // header (56) + payload (4) + checksum (8) = 68
        assert_eq!(total, 68);
    }

    // ── params_id validation ──────────────────────────────────────────────

    /// A test "param set" that implements DecodeWithParams and HasParamsId.
    struct TestParams {
        id: crate::ParamsId,
    }

    impl crate::HasParamsId for TestParams {
        fn params_id(&self) -> crate::ParamsId {
            self.id
        }
    }

    impl DecodeWithParams<TestParams> for WireU32 {
        fn decode_with_params<R: Read>(
            _params: &TestParams,
            reader: &mut R,
        ) -> Result<Self, IoError> {
            primitives::read_u32(reader).map(WireU32)
        }
    }

    #[test]
    fn container_params_id_checked_accepts_matching() {
        let id = crate::ParamsId([0xab; 32]);
        let params = TestParams { id };
        let ctx = EncodeContext {
            params_id: id,
            ..Default::default()
        };

        let mut buf = Vec::new();
        {
            let mut writer = ContainerWriter::new(&mut buf);
            writer.write_object(&WireU32(42), &ctx).unwrap();
        }

        let mut reader = ContainerReader::new(std::io::Cursor::new(&buf));
        let decoded: WireU32 = reader.read_object_with_params_checked(&params).unwrap();
        assert_eq!(decoded, WireU32(42));
    }

    #[test]
    fn container_params_id_checked_rejects_mismatch() {
        let id_a = crate::ParamsId([0xab; 32]);
        let id_b = crate::ParamsId([0xcd; 32]);
        let params = TestParams { id: id_b };
        let ctx = EncodeContext {
            params_id: id_a,
            ..Default::default()
        };

        let mut buf = Vec::new();
        {
            let mut writer = ContainerWriter::new(&mut buf);
            writer.write_object(&WireU32(42), &ctx).unwrap();
        }

        let mut reader = ContainerReader::new(std::io::Cursor::new(&buf));
        let err = reader
            .read_object_with_params_checked::<WireU32, _>(&params)
            .unwrap_err();
        assert!(matches!(err, IoError::ParamsIdMismatch { .. }));
    }

    #[test]
    fn container_params_id_checked_allows_zero_header() {
        // Zero params_id in header means "unbound" — should always pass.
        let params = TestParams {
            id: crate::ParamsId([0xab; 32]),
        };
        let ctx = EncodeContext::default(); // params_id = ZERO

        let mut buf = Vec::new();
        {
            let mut writer = ContainerWriter::new(&mut buf);
            writer.write_object(&WireU32(99), &ctx).unwrap();
        }

        let mut reader = ContainerReader::new(std::io::Cursor::new(&buf));
        let decoded: WireU32 = reader.read_object_with_params_checked(&params).unwrap();
        assert_eq!(decoded, WireU32(99));
    }
}
