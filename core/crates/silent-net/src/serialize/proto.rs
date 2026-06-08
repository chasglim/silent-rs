//! Protobuf encode/decode for SILENT control-plane messages.
//!
//! This module is **only** enabled behind the `proto` feature.  It must
//! never be used for cryptographic payloads — those are always canonical.
//!
//! Currently supported messages:
//! - [`Handshake`](crate::config::Handshake) (capability negotiation)

use crate::config::Handshake;
use bytes::Bytes;
use prost::Message;
use std::io;

// Auto-generated protobuf code from `proto/message.proto`.
pub mod generated {
    include!(concat!(env!("OUT_DIR"), "/silent.net.rs"));
}

/// Encode a [`Handshake`] to protobuf bytes.
pub fn encode_handshake(hs: &Handshake) -> Bytes {
    let msg = generated::Handshake {
        node_id: hs.node_id.clone(),
        supported_features: vec!["canonical".to_string()],
    };

    let mut buf = Vec::with_capacity(msg.encoded_len());
    msg.encode(&mut buf)
        .expect("protobuf encode is infallible for Handshake");
    buf.into()
}

/// Decode a [`Handshake`] from protobuf bytes.
pub fn decode_handshake(data: &[u8]) -> io::Result<Handshake> {
    let msg = generated::Handshake::decode(data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    Ok(Handshake::new(msg.node_id))
}
