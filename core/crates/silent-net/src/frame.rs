//! SILENT network frame: wire-format encode/decode with size limits and routing metadata.

use silent_io::error::IoError;
use silent_io::primitives;
use silent_io::{ObjectType, ParamsId};
use std::io::{Read, Write};

use crate::transport::Stream;

/// Current wire protocol version.
pub const WIRE_VERSION: u16 = 1;

/// Fixed size of the network frame header for wire version 1, in bytes.
///
/// ```text
/// Offset  Size   Field
///   0      2     wire_version: u16 LE
///   2      1     message_kind: u8
///   3      1     flags: u8
///   4      8     session_id: u64 LE
///  12      8     task_id: u64 LE
///  20      8     gate_id: u64 LE
///  28      4     object_type: u32 LE
///  32      2     object_version: u16 LE
///  34      2     scheme_id: u16 LE
///  36      32    params_id: [u8; 32]
///  68      4     payload_len: u32 LE
/// ──────── ────  total header: 72 bytes
/// ```
pub const FRAME_HEADER_SIZE: usize = 72;

/// Default maximum frame payload size (1 MiB).
pub const DEFAULT_MAX_FRAME_PAYLOAD: u32 = 1024 * 1024;

/// Discriminator for the kind of message carried by a network frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MessageKind {
    /// Regular MPC data: ciphertexts, shares, evaluation keys.
    Data = 0,
    /// Control-plane message: handshake, ack, nack, ping.
    Control = 1,
    /// Setup-phase message: key generation, parameter exchange.
    Setup = 2,
    /// Error or diagnostic message.
    Error = 3,
}

impl MessageKind {
    /// Parse from a wire byte.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Data),
            1 => Some(Self::Control),
            2 => Some(Self::Setup),
            3 => Some(Self::Error),
            _ => None,
        }
    }
}

/// Errors returned by [`NetworkFrame`] validation and receive operations.
#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    /// The wire version in the frame is newer than what this library supports.
    #[error("unsupported wire version: {found} (max {max})")]
    UnsupportedWireVersion { found: u16, max: u16 },

    /// The `message_kind` byte does not correspond to a known variant.
    #[error("unknown message kind: {0}")]
    UnknownMessageKind(u8),

    /// The `object_type` tag is not in the known registry.
    #[error("unknown object type: {0}")]
    UnknownObjectType(ObjectType),

    /// The serialized object version is newer than the caller expects.
    #[error("object version too new: {found} (max {max})")]
    ObjectVersionTooNew { found: u16, max: u16 },

    /// The payload length exceeds the configured maximum.
    #[error("payload too large: declared {declared} exceeds limit of {limit}")]
    PayloadTooLarge { declared: u64, limit: u64 },

    /// The `params_id` in the frame does not match the expected value.
    #[error("params_id mismatch: expected {expected}, got {actual}")]
    ParamsIdMismatch {
        expected: ParamsId,
        actual: ParamsId,
    },

    /// Transport-level I/O error.
    #[error("transport error: {0}")]
    Transport(String),

    /// Underlying I/O or encoding error.
    #[error(transparent)]
    Io(#[from] IoError),
}

impl FrameError {
    pub(crate) fn transport(e: impl std::fmt::Display) -> Self {
        Self::Transport(e.to_string())
    }
}

/// A SILENT network frame with routing metadata and a canonical payload.
///
/// # Wire layout
///
/// The fixed-size header (72 bytes for wire version 1) carries all routing
/// and type metadata.  The variable-length payload follows immediately and
/// contains the canonical bytes for the cryptographic object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkFrame {
    pub session_id: u64,
    pub task_id: u64,
    pub gate_id: u64,
    pub message_kind: MessageKind,
    pub object_type: ObjectType,
    pub object_version: u16,
    pub scheme_id: u16,
    pub params_id: ParamsId,
    pub flags: u8,
    pub payload: Vec<u8>,
}

impl NetworkFrame {
    /// Encode the full frame (header + payload) into `writer`.
    ///
    /// Returns the total number of bytes written.
    pub fn encode_to(&self, writer: &mut impl Write) -> Result<usize, IoError> {
        let mut written = 0usize;

        written += primitives::write_u16(writer, WIRE_VERSION)?;
        written += primitives::write_u8(writer, self.message_kind as u8)?;
        written += primitives::write_u8(writer, self.flags)?;
        written += primitives::write_u64(writer, self.session_id)?;
        written += primitives::write_u64(writer, self.task_id)?;
        written += primitives::write_u64(writer, self.gate_id)?;
        written += primitives::write_u32(writer, self.object_type.to_u32())?;
        written += primitives::write_u16(writer, self.object_version)?;
        written += primitives::write_u16(writer, self.scheme_id)?;
        written += primitives::write_u8_array(writer, self.params_id.as_bytes())?;

        let payload_len: u32 =
            self.payload
                .len()
                .try_into()
                .map_err(|_| IoError::InvalidEncoding {
                    detail: format!("payload length {} exceeds u32::MAX", self.payload.len()),
                })?;
        written += primitives::write_u32(writer, payload_len)?;

        debug_assert_eq!(written, FRAME_HEADER_SIZE);

        primitives::write_bytes(writer, &self.payload)?;
        written += self.payload.len();

        Ok(written)
    }

    /// Encode the full frame into a `Vec<u8>`.
    pub fn encode_to_vec(&self) -> Result<Vec<u8>, IoError> {
        let mut buf = Vec::with_capacity(FRAME_HEADER_SIZE + self.payload.len());
        self.encode_to(&mut buf)?;
        Ok(buf)
    }

    /// Decode a frame from `reader` with a payload size limit.
    ///
    /// Validates:
    /// - `wire_version` ≤ [`WIRE_VERSION`]
    /// - `message_kind` is a known variant
    /// - `payload_len` ≤ `max_payload_len`
    ///
    /// Caller is responsible for checking `object_type`, `object_version`,
    /// and `params_id` against expectations via the `validate_*` methods.
    pub fn decode_from(reader: &mut impl Read, max_payload_len: u32) -> Result<Self, FrameError> {
        let wire_version = primitives::read_u16(reader)?;
        if wire_version > WIRE_VERSION {
            return Err(FrameError::UnsupportedWireVersion {
                found: wire_version,
                max: WIRE_VERSION,
            });
        }

        let message_kind = primitives::read_u8(reader)?;
        let message_kind = MessageKind::from_u8(message_kind)
            .ok_or(FrameError::UnknownMessageKind(message_kind))?;

        let flags = primitives::read_u8(reader)?;
        let session_id = primitives::read_u64(reader)?;
        let task_id = primitives::read_u64(reader)?;
        let gate_id = primitives::read_u64(reader)?;
        let object_type = ObjectType::from_u32(primitives::read_u32(reader)?);
        let object_version = primitives::read_u16(reader)?;
        let scheme_id = primitives::read_u16(reader)?;

        let mut pid_bytes = [0u8; 32];
        primitives::read_exact(reader, &mut pid_bytes)?;
        let params_id = ParamsId(pid_bytes);

        let payload_len = primitives::read_u32(reader)?;
        if payload_len > max_payload_len {
            return Err(FrameError::PayloadTooLarge {
                declared: payload_len as u64,
                limit: max_payload_len as u64,
            });
        }

        let payload = primitives::read_vec(reader, payload_len as usize)?;

        Ok(Self {
            session_id,
            task_id,
            gate_id,
            message_kind,
            object_type,
            object_version,
            scheme_id,
            params_id,
            flags,
            payload,
        })
    }

    /// Validate that `self.object_type` is a known type.
    pub fn validate_object_type(&self) -> Result<&Self, FrameError> {
        if !self.object_type.is_known() {
            return Err(FrameError::UnknownObjectType(self.object_type));
        }
        Ok(self)
    }

    /// Validate that `self.object_version` is not newer than `max_version`.
    pub fn validate_object_version(&self, max_version: u16) -> Result<&Self, FrameError> {
        if self.object_version > max_version {
            return Err(FrameError::ObjectVersionTooNew {
                found: self.object_version,
                max: max_version,
            });
        }
        Ok(self)
    }

    /// Validate that `self.params_id` matches `expected`.
    pub fn validate_params_id(&self, expected: ParamsId) -> Result<&Self, FrameError> {
        if self.params_id != expected {
            return Err(FrameError::ParamsIdMismatch {
                expected,
                actual: self.params_id,
            });
        }
        Ok(self)
    }

    /// The number of bytes this frame occupies on the wire.
    pub fn wire_len(&self) -> usize {
        FRAME_HEADER_SIZE + self.payload.len()
    }

    /// Encode only the header (72 bytes) into `writer`.
    ///
    /// Writes all header fields except the payload.  The payload length is
    /// still written so the receiver can size its read buffer.
    pub fn encode_header_to(&self, writer: &mut impl Write) -> Result<usize, IoError> {
        let mut written = 0usize;

        written += primitives::write_u16(writer, WIRE_VERSION)?;
        written += primitives::write_u8(writer, self.message_kind as u8)?;
        written += primitives::write_u8(writer, self.flags)?;
        written += primitives::write_u64(writer, self.session_id)?;
        written += primitives::write_u64(writer, self.task_id)?;
        written += primitives::write_u64(writer, self.gate_id)?;
        written += primitives::write_u32(writer, self.object_type.to_u32())?;
        written += primitives::write_u16(writer, self.object_version)?;
        written += primitives::write_u16(writer, self.scheme_id)?;
        written += primitives::write_u8_array(writer, self.params_id.as_bytes())?;

        let payload_len: u32 =
            self.payload
                .len()
                .try_into()
                .map_err(|_| IoError::InvalidEncoding {
                    detail: format!("payload length {} exceeds u32::MAX", self.payload.len()),
                })?;
        written += primitives::write_u32(writer, payload_len)?;

        debug_assert_eq!(written, FRAME_HEADER_SIZE);
        Ok(written)
    }
}

// ── Async transport helpers ─────────────────────────────────────────────────

/// Extract the payload length from a raw header buffer.
///
/// The payload length is stored as a u32 LE at offset 68 (the last 4 bytes
/// of the 72-byte header).
fn extract_payload_len(header_bytes: &[u8]) -> u32 {
    let offset = FRAME_HEADER_SIZE - 4;
    u32::from_le_bytes(header_bytes[offset..offset + 4].try_into().unwrap())
}

/// Send a network frame over a transport [`Stream`].
///
/// Writes the frame header followed by the payload.
pub async fn send_frame(
    stream: &mut Box<dyn Stream>,
    frame: &NetworkFrame,
) -> Result<(), FrameError> {
    let mut header_buf = vec![0u8; FRAME_HEADER_SIZE];
    frame.encode_header_to(&mut std::io::Cursor::new(&mut header_buf[..]))?;
    stream
        .write_all(&header_buf)
        .await
        .map_err(FrameError::transport)?;

    if !frame.payload.is_empty() {
        stream
            .write_all(&frame.payload)
            .await
            .map_err(FrameError::transport)?;
    }
    Ok(())
}

/// Receive a network frame from a transport [`Stream`].
///
/// Reads the fixed-size header, extracts `payload_len`, reads the payload,
/// and validates the frame.
pub async fn recv_frame(
    stream: &mut Box<dyn Stream>,
    max_payload: u32,
) -> Result<NetworkFrame, FrameError> {
    let mut header_buf = vec![0u8; FRAME_HEADER_SIZE];
    stream
        .read_exact(&mut header_buf)
        .await
        .map_err(FrameError::transport)?;

    let payload_len = extract_payload_len(&header_buf);
    if payload_len > max_payload {
        return Err(FrameError::PayloadTooLarge {
            declared: payload_len as u64,
            limit: max_payload as u64,
        });
    }

    let mut payload_buf = vec![0u8; payload_len as usize];
    if payload_len > 0 {
        stream
            .read_exact(&mut payload_buf)
            .await
            .map_err(FrameError::transport)?;
    }

    // Reconstruct header bytes + payload into a full frame buffer and decode.
    let mut full_buf = Vec::with_capacity(FRAME_HEADER_SIZE + payload_len as usize);
    full_buf.extend_from_slice(&header_buf);
    full_buf.extend_from_slice(&payload_buf);

    NetworkFrame::decode_from(&mut std::io::Cursor::new(&full_buf), max_payload)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::Connection;
    use std::io::Cursor;

    fn sample_frame() -> NetworkFrame {
        NetworkFrame {
            session_id: 1,
            task_id: 42,
            gate_id: 7,
            message_kind: MessageKind::Data,
            object_type: ObjectType::RLWE_CIPHERTEXT,
            object_version: 1,
            scheme_id: 2,
            params_id: ParamsId([0xabu8; 32]),
            flags: 0,
            payload: b"hello, silent".to_vec(),
        }
    }

    // ── Encode/decode roundtrip ────────────────────────────────────────────

    #[test]
    fn encode_decode_roundtrip() {
        let frame = sample_frame();
        let buf = frame.encode_to_vec().unwrap();
        assert_eq!(buf.len(), FRAME_HEADER_SIZE + frame.payload.len());

        let decoded =
            NetworkFrame::decode_from(&mut Cursor::new(&buf), DEFAULT_MAX_FRAME_PAYLOAD).unwrap();

        assert_eq!(decoded, frame);
    }

    #[test]
    fn encode_decode_empty_payload() {
        let mut frame = sample_frame();
        frame.payload = vec![];
        let buf = frame.encode_to_vec().unwrap();
        assert_eq!(buf.len(), FRAME_HEADER_SIZE);

        let decoded =
            NetworkFrame::decode_from(&mut Cursor::new(&buf), DEFAULT_MAX_FRAME_PAYLOAD).unwrap();
        assert_eq!(decoded.payload, b"");
    }

    #[test]
    fn encode_decode_large_payload() {
        let mut frame = sample_frame();
        frame.payload = vec![0xccu8; 4096];

        let buf = frame.encode_to_vec().unwrap();
        let decoded = NetworkFrame::decode_from(&mut Cursor::new(&buf), 8192).unwrap();
        assert_eq!(decoded.payload.len(), 4096);
        assert!(decoded.payload.iter().all(|&b| b == 0xcc));
    }

    #[test]
    fn all_message_kinds_roundtrip() {
        for kind in [
            MessageKind::Data,
            MessageKind::Control,
            MessageKind::Setup,
            MessageKind::Error,
        ] {
            let frame = NetworkFrame {
                message_kind: kind,
                ..sample_frame()
            };
            let buf = frame.encode_to_vec().unwrap();
            let decoded =
                NetworkFrame::decode_from(&mut Cursor::new(&buf), DEFAULT_MAX_FRAME_PAYLOAD)
                    .unwrap();
            assert_eq!(decoded.message_kind, kind);
        }
    }

    #[test]
    fn params_id_values_roundtrip() {
        for pid in [
            ParamsId::ZERO,
            ParamsId([0xffu8; 32]),
            ParamsId([0xabu8; 32]),
        ] {
            let frame = NetworkFrame {
                params_id: pid,
                ..sample_frame()
            };
            let buf = frame.encode_to_vec().unwrap();
            let decoded =
                NetworkFrame::decode_from(&mut Cursor::new(&buf), DEFAULT_MAX_FRAME_PAYLOAD)
                    .unwrap();
            assert_eq!(decoded.params_id, pid);
        }
    }

    // ── Malformed frame tests ──────────────────────────────────────────────

    #[test]
    fn rejects_future_wire_version() {
        let frame = sample_frame();
        let mut buf = frame.encode_to_vec().unwrap();
        buf[0] = 0xff;
        buf[1] = 0xff;

        let err = NetworkFrame::decode_from(&mut Cursor::new(&buf), DEFAULT_MAX_FRAME_PAYLOAD)
            .unwrap_err();

        assert!(matches!(err, FrameError::UnsupportedWireVersion { .. }));
    }

    #[test]
    fn rejects_unknown_message_kind() {
        let frame = sample_frame();
        let mut buf = frame.encode_to_vec().unwrap();
        buf[2] = 0xff;

        let err = NetworkFrame::decode_from(&mut Cursor::new(&buf), DEFAULT_MAX_FRAME_PAYLOAD)
            .unwrap_err();

        assert!(matches!(err, FrameError::UnknownMessageKind(0xff)));
    }

    #[test]
    fn rejects_oversized_payload() {
        let frame = sample_frame();
        let buf = frame.encode_to_vec().unwrap();

        let err = NetworkFrame::decode_from(&mut Cursor::new(&buf), 1).unwrap_err();

        assert!(matches!(err, FrameError::PayloadTooLarge { .. }));
    }

    #[test]
    fn validate_rejects_unknown_object_type() {
        let frame = NetworkFrame {
            object_type: ObjectType::from_u32(0xdead),
            ..sample_frame()
        };

        let err = frame.validate_object_type().unwrap_err();
        assert!(matches!(err, FrameError::UnknownObjectType(_)));
    }

    #[test]
    fn validate_rejects_future_object_version() {
        let frame = NetworkFrame {
            object_version: 5,
            ..sample_frame()
        };

        let err = frame.validate_object_version(3).unwrap_err();
        assert!(matches!(
            err,
            FrameError::ObjectVersionTooNew { found: 5, max: 3 }
        ));
    }

    #[test]
    fn validate_accepts_current_object_version() {
        let frame = NetworkFrame {
            object_version: 3,
            ..sample_frame()
        };
        assert!(frame.validate_object_version(3).is_ok());
    }

    #[test]
    fn validate_rejects_params_id_mismatch() {
        let frame = sample_frame();
        let expected = ParamsId([0xffu8; 32]);

        let err = frame.validate_params_id(expected).unwrap_err();
        assert!(matches!(err, FrameError::ParamsIdMismatch { .. }));
    }

    #[test]
    fn validate_accepts_matching_params_id() {
        let frame = sample_frame();
        assert!(frame.validate_params_id(ParamsId([0xabu8; 32])).is_ok());
    }

    #[test]
    fn rejects_truncated_header() {
        let buf = vec![0u8; 10];
        let err = NetworkFrame::decode_from(&mut Cursor::new(&buf), DEFAULT_MAX_FRAME_PAYLOAD)
            .unwrap_err();

        assert!(matches!(err, FrameError::Io(IoError::UnexpectedEof { .. })));
    }

    #[test]
    fn rejects_truncated_payload() {
        let frame = sample_frame();
        let mut buf = frame.encode_to_vec().unwrap();
        let new_len = buf.len() - 5;
        buf.truncate(new_len);

        let err = NetworkFrame::decode_from(&mut Cursor::new(&buf), DEFAULT_MAX_FRAME_PAYLOAD)
            .unwrap_err();

        assert!(matches!(err, FrameError::Io(IoError::UnexpectedEof { .. })));
    }

    #[test]
    fn wire_len_is_correct() {
        let frame = sample_frame();
        assert_eq!(frame.wire_len(), FRAME_HEADER_SIZE + frame.payload.len());
    }

    #[test]
    fn extract_payload_len_from_header() {
        let frame = NetworkFrame {
            payload: vec![0u8; 42],
            ..sample_frame()
        };
        let mut header_buf = vec![0u8; FRAME_HEADER_SIZE];
        frame
            .encode_header_to(&mut std::io::Cursor::new(header_buf.as_mut_slice()))
            .unwrap();

        assert_eq!(extract_payload_len(&header_buf), 42);
    }

    // ── Memory transport roundtrip tests ───────────────────────────────────

    #[tokio::test]
    async fn memory_transport_send_recv_roundtrip() {
        let (alice, bob) = crate::transport::memory::MemoryConnection::pair();

        let frame = sample_frame();
        let frame_send = frame.clone();

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_frame(&mut stream, &frame_send).await.unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_frame(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .unwrap()
        });

        alice_handle.await.unwrap();
        let received = bob_handle.await.unwrap();

        assert_eq!(received, frame);
    }

    #[tokio::test]
    async fn memory_transport_multiple_frames() {
        let (alice, bob) = crate::transport::memory::MemoryConnection::pair();

        let frames: Vec<NetworkFrame> = (0..5)
            .map(|i| NetworkFrame {
                session_id: i,
                task_id: i * 10,
                payload: format!("frame_{i}").into_bytes(),
                ..sample_frame()
            })
            .collect();

        let frames_clone = frames.clone();

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            for f in &frames_clone {
                send_frame(&mut stream, f).await.unwrap();
            }
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            let mut received = Vec::new();
            for _ in 0..5 {
                received.push(
                    recv_frame(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD)
                        .await
                        .unwrap(),
                );
            }
            received
        });

        alice_handle.await.unwrap();
        let received = bob_handle.await.unwrap();

        assert_eq!(received, frames);
    }

    #[tokio::test]
    async fn memory_transport_empty_payload_frame() {
        let (alice, bob) = crate::transport::memory::MemoryConnection::pair();

        let frame = NetworkFrame {
            payload: vec![],
            ..sample_frame()
        };
        let frame_send = frame.clone();

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_frame(&mut stream, &frame_send).await.unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_frame(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .unwrap()
        });

        alice_handle.await.unwrap();
        let received = bob_handle.await.unwrap();

        assert_eq!(received, frame);
        assert!(received.payload.is_empty());
    }

    #[tokio::test]
    async fn memory_transport_large_payload_frame() {
        let (alice, bob) = crate::transport::memory::MemoryConnection::pair();

        let frame = NetworkFrame {
            payload: vec![0xabu8; 65_536],
            ..sample_frame()
        };
        let frame_send = frame.clone();

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_frame(&mut stream, &frame_send).await.unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_frame(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .unwrap()
        });

        alice_handle.await.unwrap();
        let received = bob_handle.await.unwrap();

        assert_eq!(received.payload.len(), 65536);
        assert!(received.payload.iter().all(|&b| b == 0xab));
    }

    #[tokio::test]
    async fn memory_transport_rejects_oversized_payload() {
        let (alice, bob) = crate::transport::memory::MemoryConnection::pair();

        let frame = NetworkFrame {
            payload: vec![0u8; 1024],
            ..sample_frame()
        };

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_frame(&mut stream, &frame).await.unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            // Set max_payload to 100 bytes, but frame has 1024
            recv_frame(&mut stream, 100).await
        });

        alice_handle.await.unwrap();
        let result = bob_handle.await.unwrap();
        let err = result.unwrap_err();

        assert!(matches!(err, FrameError::PayloadTooLarge { .. }));
    }

    #[tokio::test]
    async fn memory_transport_bidirectional() {
        let (alice, bob) = crate::transport::memory::MemoryConnection::pair();

        let alice_frame = NetworkFrame {
            session_id: 100,
            payload: b"alice_to_bob".to_vec(),
            ..sample_frame()
        };
        let bob_frame = NetworkFrame {
            session_id: 200,
            payload: b"bob_to_alice".to_vec(),
            ..sample_frame()
        };

        let af = alice_frame.clone();
        let bf = bob_frame.clone();

        let alice_handle = tokio::spawn(async move {
            let mut send_stream = alice.open_stream().await.unwrap();
            send_frame(&mut send_stream, &af).await.unwrap();

            let mut recv_stream = alice.accept_stream().await.unwrap();
            recv_frame(&mut recv_stream, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .unwrap()
        });

        let bob_handle = tokio::spawn(async move {
            let mut recv_stream = bob.accept_stream().await.unwrap();
            let received = recv_frame(&mut recv_stream, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .unwrap();

            let mut send_stream = bob.open_stream().await.unwrap();
            send_frame(&mut send_stream, &bf).await.unwrap();

            received
        });

        let bob_received = bob_handle.await.unwrap();
        let alice_received = alice_handle.await.unwrap();

        assert_eq!(bob_received.payload, b"alice_to_bob");
        assert_eq!(alice_received.payload, b"bob_to_alice");
    }
}
