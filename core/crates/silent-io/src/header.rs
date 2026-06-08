//! Container and frame header structures.
//!
//! These headers are written before the payload and carry the metadata needed
//! to identify, version, and validate the object.

use crate::error::IoError;
use crate::primitives;
use crate::{FORMAT_VERSION, ObjectType, ParamsId};
use std::io::{Read, Write};

// ── Container header ────────────────────────────────────────────────────────

/// The 56-byte header that prefixes every SILENT file container.
///
/// Layout:
///
/// ```text
/// Offset  Size   Field
/// ──────  ────   ─────
/// 0       4      magic: b"SFIR"
/// 4       2      format_version: u16 LE
/// 6       4      object_type: u32 LE
/// 10      2      scheme_id: u16 LE
/// 12      32     params_id: [u8; 32]
/// 44      1      flags: u8
/// 45      2      object_version: u16 LE
/// 47      1      reserved: u8
/// 48      8      payload_len: u64 LE
/// ──────  ────   ─────  header: 56 bytes
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContainerHeader {
    /// Format version of the container itself.
    pub format_version: u16,
    /// Numeric type tag identifying the payload object.
    pub object_type: ObjectType,
    /// Scheme family discriminant (interpreted by `silent-params`).
    pub scheme_id: u16,
    /// Parameter-set identifier; all-zero when no parameter binding applies.
    pub params_id: ParamsId,
    /// Encoding flags bitfield.
    pub flags: u8,
    /// Serialization format version of the payload object type.
    pub object_version: u16,
    /// Byte length of the payload that follows the header.
    pub payload_len: u64,
}

/// Size of the container header on the wire, in bytes.
pub const CONTAINER_HEADER_SIZE: usize = 56;

impl ContainerHeader {
    /// Encode this header to `writer` in canonical little-endian format.
    ///
    /// Returns the number of bytes written (always [`CONTAINER_HEADER_SIZE`]).
    pub fn encode_to<W: Write>(&self, writer: &mut W) -> Result<usize, IoError> {
        let mut written = 0usize;
        written += primitives::write_u8_array(writer, crate::CONTAINER_MAGIC)?;
        written += primitives::write_u16(writer, self.format_version)?;
        written += primitives::write_u32(writer, self.object_type.to_u32())?;
        written += primitives::write_u16(writer, self.scheme_id)?;
        written += primitives::write_u8_array(writer, self.params_id.as_bytes())?;
        written += primitives::write_u8(writer, self.flags)?;
        written += primitives::write_u16(writer, self.object_version)?;
        written += primitives::write_u8(writer, 0u8)?; // reserved
        written += primitives::write_u64(writer, self.payload_len)?;
        debug_assert_eq!(written, CONTAINER_HEADER_SIZE);
        Ok(written)
    }

    /// Encode this header into a freshly allocated `Vec<u8>`.
    pub fn encode_to_vec(&self) -> Result<Vec<u8>, IoError> {
        let mut buf = Vec::with_capacity(CONTAINER_HEADER_SIZE);
        self.encode_to(&mut buf)?;
        Ok(buf)
    }

    /// Decode a [`ContainerHeader`] from `reader`.
    ///
    /// Validates magic bytes and format version.  Unknown or forward-version
    /// headers are rejected with [`IoError::InvalidMagic`] or
    /// [`IoError::UnsupportedVersion`].
    pub fn decode_from<R: Read>(reader: &mut R) -> Result<Self, IoError> {
        // ── magic ──
        let mut magic = [0u8; 4];
        primitives::read_exact(reader, &mut magic)?;
        if &magic != crate::CONTAINER_MAGIC {
            return Err(IoError::InvalidMagic { actual: magic });
        }

        // ── format_version ──
        let format_version = primitives::read_u16(reader)?;
        if format_version > FORMAT_VERSION {
            return Err(IoError::UnsupportedVersion {
                found: format_version,
                max: FORMAT_VERSION,
            });
        }

        // ── object_type ──
        let object_type = ObjectType::from_u32(primitives::read_u32(reader)?);

        // ── scheme_id ──
        let scheme_id = primitives::read_u16(reader)?;

        // ── params_id ──
        let mut pid_bytes = [0u8; 32];
        primitives::read_exact(reader, &mut pid_bytes)?;
        let params_id = ParamsId(pid_bytes);

        // ── flags ──
        let flags = primitives::read_u8(reader)?;

        // ── object_version ──
        let object_version = primitives::read_u16(reader)?;

        // ── reserved ──
        let _reserved = primitives::read_u8(reader)?;

        // ── payload_len ──
        let payload_len = primitives::read_u64(reader)?;

        Ok(Self {
            format_version,
            object_type,
            scheme_id,
            params_id,
            flags,
            object_version,
            payload_len,
        })
    }
}

// ── Stream frame header ─────────────────────────────────────────────────────

/// The 12-byte header preceding each length-delimited stream frame.
///
/// Layout:
///
/// ```text
/// Offset  Size   Field
/// ──────  ────   ─────
/// 0       4      payload_len: u32 LE    (excludes these 12 header bytes)
/// 4       4      object_type: u32 LE
/// 8       1      flags: u8
/// 9       3      reserved: [u8; 3]
/// ──────  ────   ─────  frame header: 12 bytes
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    /// Numeric type tag identifying the payload object.
    pub object_type: ObjectType,
    /// Encoding flags bitfield.
    pub flags: u8,
    /// Byte length of the payload that follows the frame header.
    pub payload_len: u32,
}

/// Size of a frame header on the wire, in bytes.
pub const FRAME_HEADER_SIZE: usize = 12;

impl FrameHeader {
    /// Encode this frame header to `writer`.
    pub fn encode_to<W: Write>(&self, writer: &mut W) -> Result<usize, IoError> {
        let mut written = 0usize;
        written += primitives::write_u32(writer, self.payload_len)?;
        written += primitives::write_u32(writer, self.object_type.to_u32())?;
        written += primitives::write_u8(writer, self.flags)?;
        written += primitives::write_bytes(writer, &[0u8; 3])?;
        debug_assert_eq!(written, FRAME_HEADER_SIZE);
        Ok(written)
    }

    /// Decode a [`FrameHeader`] from `reader`.
    pub fn decode_from<R: Read>(reader: &mut R) -> Result<Self, IoError> {
        let payload_len = primitives::read_u32(reader)?;
        let object_type = ObjectType::from_u32(primitives::read_u32(reader)?);
        let flags = primitives::read_u8(reader)?;
        let mut _reserved = [0u8; 3];
        primitives::read_exact(reader, &mut _reserved)?;
        Ok(Self {
            object_type,
            flags,
            payload_len,
        })
    }
}

// ── Header utility constants ────────────────────────────────────────────────

/// Maximum payload length for a file container (8 MiB by default).
///
/// Override via [`crate::container::ContainerReader::with_max_payload`].
pub const DEFAULT_MAX_CONTAINER_PAYLOAD: u64 = 8 * 1024 * 1024;

/// Maximum frame payload length for a stream frame (64 KiB by default).
///
/// Override via [`crate::framed::FramedReader::with_max_payload`].
pub const DEFAULT_MAX_FRAME_PAYLOAD: u32 = 64 * 1024;

/// Bit 0 in the flags byte: the payload is in NTT domain.
pub const FLAG_IS_NTT: u8 = 0x01;

/// Bit 1 in the flags byte: the payload uses seeded-a compact encoding.
pub const FLAG_IS_SEEDED: u8 = 0x02;

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_container_header() -> ContainerHeader {
        ContainerHeader {
            format_version: FORMAT_VERSION,
            object_type: ObjectType::RLWE_CIPHERTEXT,
            scheme_id: 1,
            params_id: ParamsId([0xabu8; 32]),
            flags: FLAG_IS_NTT,
            object_version: 1,
            payload_len: 1024,
        }
    }

    #[test]
    fn container_header_roundtrip() {
        let h = sample_container_header();
        let buf = h.encode_to_vec().unwrap();
        assert_eq!(buf.len(), CONTAINER_HEADER_SIZE);

        let decoded = ContainerHeader::decode_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, h);
    }

    #[test]
    fn container_header_rejects_bad_magic() {
        let mut buf = [0u8; CONTAINER_HEADER_SIZE];
        buf[0] = 0xff; // break magic
        let err = ContainerHeader::decode_from(&mut Cursor::new(&buf)).unwrap_err();
        assert!(matches!(err, IoError::InvalidMagic { .. }));
    }

    #[test]
    fn container_header_rejects_future_version() {
        let h = ContainerHeader {
            format_version: FORMAT_VERSION + 1,
            ..sample_container_header()
        };
        let buf = h.encode_to_vec().unwrap();
        let err = ContainerHeader::decode_from(&mut Cursor::new(&buf)).unwrap_err();
        assert!(matches!(err, IoError::UnsupportedVersion { .. }));
    }

    #[test]
    fn container_header_eof_detection() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"SFIR"); // correct magic
        buf.extend_from_slice(&[0u8; 6]); // too short — only 10 bytes total
        let err = ContainerHeader::decode_from(&mut Cursor::new(&buf)).unwrap_err();
        assert!(matches!(err, IoError::UnexpectedEof { .. }));
    }

    #[test]
    fn frame_header_roundtrip() {
        let h = FrameHeader {
            object_type: ObjectType::LWE_CIPHERTEXT,
            flags: 0,
            payload_len: 256,
        };
        let mut buf = Vec::new();
        h.encode_to(&mut buf).unwrap();
        assert_eq!(buf.len(), FRAME_HEADER_SIZE);

        let decoded = FrameHeader::decode_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, h);
    }

    #[test]
    fn frame_header_zero_payload() {
        let h = FrameHeader {
            object_type: ObjectType::RLWE_CIPHERTEXT,
            flags: 0,
            payload_len: 0,
        };
        let mut buf = Vec::new();
        h.encode_to(&mut buf).unwrap();
        let decoded = FrameHeader::decode_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded.payload_len, 0);
    }
}
