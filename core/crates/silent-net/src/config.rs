//! Connection handshake and capability negotiation.
//!
//! On connection establishment, both peers exchange a [`Handshake`] message
//! to agree on:
//!
//! - Protocol version
//! - Node identity
//! - Maximum frame payload size
//! - Supported payload formats (bitset)
//! - Supported compression algorithms
//!
//! The negotiation selects the minimum (most restrictive) value for
//! numeric fields and the intersection for set-like fields.

use crate::frame::{DEFAULT_MAX_FRAME_PAYLOAD, MessageKind, NetworkFrame, recv_frame, send_frame};
use crate::transport::Stream;
use silent_io::ParamsId;
use silent_io::error::IoError;
use silent_io::primitives;
use std::io::{Read, Write};

/// Current handshake protocol version.
pub const HANDSHAKE_VERSION: u16 = 1;

/// Special object type used for handshake frames.
pub const HANDSHAKE_OBJECT_TYPE: u32 = 0x2001;

/// A capability advertisement sent at connection start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Handshake {
    /// Handshake protocol version (not the wire version).
    pub protocol_version: u16,
    /// Human-readable node identifier.
    pub node_id: String,
    /// Maximum frame payload size this peer will accept (in bytes).
    pub max_frame_size: u32,
    /// Bitset of supported payload formats.
    ///
    /// Bit 0: Canonical (always set).
    pub payload_formats: u32,
    /// Compression algorithm IDs this peer supports (empty = none).
    pub compression_algos: Vec<u8>,
}

impl Handshake {
    /// Create a default handshake advertising conservative defaults.
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            protocol_version: HANDSHAKE_VERSION,
            node_id: node_id.into(),
            max_frame_size: DEFAULT_MAX_FRAME_PAYLOAD,
            payload_formats: 0x01, // canonical only
            compression_algos: vec![],
        }
    }

    /// Set the maximum frame size.
    pub fn with_max_frame_size(mut self, size: u32) -> Self {
        self.max_frame_size = size;
        self
    }

    /// Add a payload format bit.
    pub fn with_payload_format(mut self, bit: u32) -> Self {
        self.payload_formats |= bit;
        self
    }

    /// Canonical encode to a byte vector.
    ///
    /// Layout:
    /// ```text
    /// 0      2   protocol_version: u16 LE
    /// 2      2   node_id_len: u16 LE
    /// 4      ..  node_id: UTF-8 bytes
    /// ..     4   max_frame_size: u32 LE
    /// ..     4   payload_formats: u32 LE
    /// ..     1   num_compression_algos: u8
    /// ..     ..  compression_algos: [u8; num_compression_algos]
    /// ```
    pub fn encode_to_vec(&self) -> Result<Vec<u8>, IoError> {
        let mut buf = Vec::new();
        self.encode_to(&mut buf)?;
        Ok(buf)
    }

    fn encode_to(&self, w: &mut impl Write) -> Result<usize, IoError> {
        let mut n = 0;
        n += primitives::write_u16(w, self.protocol_version)?;

        let id_bytes = self.node_id.as_bytes();
        if id_bytes.len() > u16::MAX as usize {
            return Err(IoError::InvalidEncoding {
                detail: "node_id too long".into(),
            });
        }
        n += primitives::write_u16(w, id_bytes.len() as u16)?;
        n += primitives::write_bytes(w, id_bytes)?;
        n += primitives::write_u32(w, self.max_frame_size)?;
        n += primitives::write_u32(w, self.payload_formats)?;

        if self.compression_algos.len() > u8::MAX as usize {
            return Err(IoError::InvalidEncoding {
                detail: "too many compression algos".into(),
            });
        }
        n += primitives::write_u8(w, self.compression_algos.len() as u8)?;
        n += primitives::write_bytes(w, &self.compression_algos)?;

        Ok(n)
    }

    /// Canonical decode from a byte reader.
    pub fn decode_from(r: &mut impl Read) -> Result<Self, IoError> {
        let protocol_version = primitives::read_u16(r)?;
        if protocol_version > HANDSHAKE_VERSION {
            return Err(IoError::UnsupportedVersion {
                found: protocol_version,
                max: HANDSHAKE_VERSION,
            });
        }

        let id_len = primitives::read_u16(r)? as usize;
        let id_bytes = primitives::read_vec(r, id_len)?;
        let node_id = String::from_utf8(id_bytes).map_err(|e| IoError::InvalidEncoding {
            detail: format!("invalid UTF-8 in node_id: {e}"),
        })?;

        let max_frame_size = primitives::read_u32(r)?;
        let payload_formats = primitives::read_u32(r)?;

        let num_algos = primitives::read_u8(r)? as usize;
        let compression_algos = primitives::read_vec(r, num_algos)?;

        Ok(Self {
            protocol_version,
            node_id,
            max_frame_size,
            payload_formats,
            compression_algos,
        })
    }
}

/// Result of negotiating two handshakes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandshakeResult {
    pub agreed_protocol_version: u16,
    pub initiator_id: String,
    pub responder_id: String,
    pub max_frame_size: u32,
    pub payload_formats: u32,
    pub compression_algos: Vec<u8>,
}

/// Negotiate between two handshakes.
///
/// Returns `None` if no mutually acceptable parameters were found.
pub fn negotiate(initiator: &Handshake, responder: &Handshake) -> Option<HandshakeResult> {
    let version = initiator.protocol_version.min(responder.protocol_version);

    let payload_formats = initiator.payload_formats & responder.payload_formats;
    if payload_formats == 0 {
        return None; // no common payload format
    }

    let max_frame_size = initiator.max_frame_size.min(responder.max_frame_size);

    let compression_algos: Vec<u8> = initiator
        .compression_algos
        .iter()
        .filter(|a| responder.compression_algos.contains(a))
        .copied()
        .collect();

    Some(HandshakeResult {
        agreed_protocol_version: version,
        initiator_id: initiator.node_id.clone(),
        responder_id: responder.node_id.clone(),
        max_frame_size,
        payload_formats,
        compression_algos,
    })
}

/// Perform a two-way handshake over a transport stream.
///
/// The initiator sends first, then receives the responder's handshake.
/// Returns the negotiated result.
pub async fn perform_handshake_as_initiator(
    stream: &mut Box<dyn Stream>,
    our_handshake: &Handshake,
) -> Result<HandshakeResult, crate::frame::FrameError> {
    send_handshake(stream, our_handshake).await?;
    let their_handshake = recv_handshake(stream).await?;

    negotiate(our_handshake, &their_handshake).ok_or_else(|| {
        crate::frame::FrameError::Io(IoError::InvalidEncoding {
            detail: "handshake negotiation failed: no common parameters".into(),
        })
    })
}

/// Perform a two-way handshake as the responder.
///
/// The responder receives first, then sends its own handshake.
pub async fn perform_handshake_as_responder(
    stream: &mut Box<dyn Stream>,
    our_handshake: &Handshake,
) -> Result<HandshakeResult, crate::frame::FrameError> {
    let their_handshake = recv_handshake(stream).await?;

    send_handshake(stream, our_handshake).await?;

    negotiate(&their_handshake, our_handshake).ok_or_else(|| {
        crate::frame::FrameError::Io(IoError::InvalidEncoding {
            detail: "handshake negotiation failed: no common parameters".into(),
        })
    })
}

/// Send a handshake frame over a stream.
async fn send_handshake(
    stream: &mut Box<dyn Stream>,
    hs: &Handshake,
) -> Result<(), crate::frame::FrameError> {
    let payload = hs.encode_to_vec()?;

    let frame = NetworkFrame {
        session_id: 0,
        task_id: 0,
        gate_id: 0,
        message_kind: MessageKind::Control,
        object_type: silent_io::ObjectType::from_u32(HANDSHAKE_OBJECT_TYPE),
        object_version: 1,
        scheme_id: 0,
        params_id: ParamsId::ZERO,
        flags: 0,
        payload,
    };

    send_frame(stream, &frame).await
}

/// Receive a handshake frame from a stream.
async fn recv_handshake(
    stream: &mut Box<dyn Stream>,
) -> Result<Handshake, crate::frame::FrameError> {
    let frame = recv_frame(stream, DEFAULT_MAX_FRAME_PAYLOAD).await?;

    if frame.object_type.to_u32() != HANDSHAKE_OBJECT_TYPE {
        return Err(crate::frame::FrameError::Io(IoError::ObjectTypeMismatch {
            expected: silent_io::ObjectType::from_u32(HANDSHAKE_OBJECT_TYPE),
            actual: frame.object_type,
        }));
    }

    Handshake::decode_from(&mut std::io::Cursor::new(&frame.payload))
        .map_err(crate::frame::FrameError::from)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::Connection;
    use crate::transport::memory::MemoryConnection;

    fn sample_handshake() -> Handshake {
        Handshake::new("node-001")
            .with_max_frame_size(256 * 1024)
            .with_payload_format(0x02)
    }

    #[test]
    fn handshake_encode_decode_roundtrip() {
        let hs = sample_handshake();
        let buf = hs.encode_to_vec().unwrap();
        let decoded = Handshake::decode_from(&mut std::io::Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, hs);
    }

    #[test]
    fn handshake_default_values() {
        let hs = Handshake::new("test");
        assert_eq!(hs.protocol_version, HANDSHAKE_VERSION);
        assert_eq!(hs.node_id, "test");
        assert_eq!(hs.payload_formats, 0x01);
        assert!(hs.compression_algos.is_empty());
    }

    #[test]
    fn negotiate_compatible_handshakes() {
        let a = Handshake::new("alice").with_max_frame_size(128 * 1024);
        let b = Handshake::new("bob").with_max_frame_size(256 * 1024);

        let result = negotiate(&a, &b).unwrap();
        assert_eq!(result.max_frame_size, 128 * 1024); // min
        assert_eq!(result.payload_formats, 0x01); // intersection
        assert_eq!(result.initiator_id, "alice");
        assert_eq!(result.responder_id, "bob");
    }

    #[test]
    fn negotiate_incompatible_payload_formats() {
        let a = Handshake::new("alice");
        let b = Handshake {
            payload_formats: 0x02, // different from alice's 0x01
            ..Handshake::new("bob")
        };

        assert!(negotiate(&a, &b).is_none());
    }

    #[test]
    fn negotiate_picks_minimum_protocol_version() {
        let a = Handshake {
            protocol_version: 1,
            ..Handshake::new("alice")
        };
        let b = Handshake {
            protocol_version: 2,
            ..Handshake::new("bob")
        };

        let result = negotiate(&a, &b).unwrap();
        assert_eq!(result.agreed_protocol_version, 1);
    }

    #[test]
    fn negotiate_intersection_of_compression_algos() {
        let a = Handshake {
            compression_algos: vec![1, 2, 3],
            ..Handshake::new("alice")
        };
        let b = Handshake {
            compression_algos: vec![2, 3, 4],
            ..Handshake::new("bob")
        };

        let result = negotiate(&a, &b).unwrap();
        assert_eq!(result.compression_algos, vec![2, 3]);
    }

    #[test]
    fn handshake_null_node_id() {
        let hs = Handshake::new("");
        let buf = hs.encode_to_vec().unwrap();
        let decoded = Handshake::decode_from(&mut std::io::Cursor::new(&buf)).unwrap();
        assert_eq!(decoded.node_id, "");
    }

    #[test]
    fn handshake_unicode_node_id() {
        let hs = Handshake::new("节点-🐱");
        let buf = hs.encode_to_vec().unwrap();
        let decoded = Handshake::decode_from(&mut std::io::Cursor::new(&buf)).unwrap();
        assert_eq!(decoded.node_id, "节点-🐱");
    }

    #[test]
    fn handshake_rejects_future_version() {
        let hs = Handshake {
            protocol_version: HANDSHAKE_VERSION + 1,
            ..sample_handshake()
        };
        let buf = hs
            .encode_to_vec()
            .expect("encode should succeed even for future version");
        let err = Handshake::decode_from(&mut std::io::Cursor::new(&buf)).unwrap_err();
        assert!(matches!(err, IoError::UnsupportedVersion { .. }));
    }

    // ── Integration: handshake over memory transport ─────────────────────

    #[tokio::test]
    async fn handshake_over_memory_transport() {
        let (alice, bob) = MemoryConnection::pair();

        let alice_hs = Handshake::new("alice");
        let bob_hs = Handshake::new("bob");

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            perform_handshake_as_initiator(&mut stream, &alice_hs)
                .await
                .unwrap()
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            perform_handshake_as_responder(&mut stream, &bob_hs)
                .await
                .unwrap()
        });

        let alice_result = alice_handle.await.unwrap();
        let bob_result = bob_handle.await.unwrap();

        assert_eq!(alice_result.initiator_id, "alice");
        assert_eq!(alice_result.responder_id, "bob");
        assert_eq!(alice_result.max_frame_size, DEFAULT_MAX_FRAME_PAYLOAD);

        assert_eq!(
            alice_result.agreed_protocol_version,
            bob_result.agreed_protocol_version
        );
        assert_eq!(alice_result.max_frame_size, bob_result.max_frame_size);
        assert_eq!(alice_result.payload_formats, bob_result.payload_formats);
    }

    #[tokio::test]
    async fn handshake_with_custom_limits() {
        let (alice, bob) = MemoryConnection::pair();

        let alice_hs = Handshake::new("alice-limited").with_max_frame_size(64 * 1024);
        let bob_hs = Handshake::new("bob-generous").with_max_frame_size(256 * 1024);

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            perform_handshake_as_initiator(&mut stream, &alice_hs)
                .await
                .unwrap()
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            perform_handshake_as_responder(&mut stream, &bob_hs)
                .await
                .unwrap()
        });

        let result = alice_handle.await.unwrap();
        let _ = bob_handle.await.unwrap();

        assert_eq!(result.max_frame_size, 64 * 1024); // min(64K, 256K) = 64K
        assert_eq!(result.initiator_id, "alice-limited");
        assert_eq!(result.responder_id, "bob-generous");
    }
}
