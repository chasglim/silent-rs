use std::collections::HashMap;
use std::io::Cursor;

use silent_io::{ObjectType, primitives};
use silent_net::frame::{FRAME_HEADER_SIZE, NetworkFrame, recv_frame, send_frame};
use silent_net::transport::Stream;

use crate::error::RuntimeError;
use crate::ids::{GateId, PartyId, RuntimeRoute, SessionId, TaskId, WireContext};
use crate::value::{Value, decode_value_payload, encode_value_payload};

/// Application-reserved object tag for runtime control/data payloads.
pub const RUNTIME_PAYLOAD_OBJECT: ObjectType = ObjectType::from_u32(0x3000);

/// Wrap a payload in a SILENT network frame using runtime route metadata.
pub fn frame_payload(route: RuntimeRoute, wire: &WireContext, payload: Vec<u8>) -> NetworkFrame {
    NetworkFrame {
        session_id: route.session.0,
        task_id: route.task.0,
        gate_id: route.gate.0,
        message_kind: route.kind,
        object_type: RUNTIME_PAYLOAD_OBJECT,
        object_version: 1,
        scheme_id: wire.encode.scheme_id,
        params_id: wire.encode.params_id,
        flags: wire.encode.flags,
        payload,
    }
}

/// Runtime channel counters measured at SILENT frame boundaries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeStats {
    pub frames_sent: u64,
    pub frames_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
}

impl RuntimeStats {
    fn record_sent(&mut self, payload_len: usize) {
        self.frames_sent = self.frames_sent.saturating_add(1);
        self.bytes_sent = self
            .bytes_sent
            .saturating_add((FRAME_HEADER_SIZE + payload_len) as u64);
    }

    fn record_received(&mut self, payload_len: usize) {
        self.frames_received = self.frames_received.saturating_add(1);
        self.bytes_received = self
            .bytes_received
            .saturating_add((FRAME_HEADER_SIZE + payload_len) as u64);
    }

    pub fn merge(&mut self, rhs: RuntimeStats) {
        self.frames_sent = self.frames_sent.saturating_add(rhs.frames_sent);
        self.frames_received = self.frames_received.saturating_add(rhs.frames_received);
        self.bytes_sent = self.bytes_sent.saturating_add(rhs.bytes_sent);
        self.bytes_received = self.bytes_received.saturating_add(rhs.bytes_received);
    }
}

/// Runtime communication endpoint backed by `silent-net`.
pub struct RuntimeChannel {
    stream: Box<dyn Stream>,
    wire: WireContext,
    stats: RuntimeStats,
}

impl RuntimeChannel {
    pub fn new(stream: Box<dyn Stream>, wire: WireContext) -> Self {
        Self {
            stream,
            wire,
            stats: RuntimeStats::default(),
        }
    }

    pub fn wire(&self) -> &WireContext {
        &self.wire
    }

    pub fn stats(&self) -> RuntimeStats {
        self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = RuntimeStats::default();
    }

    pub async fn send_payload(
        &mut self,
        route: RuntimeRoute,
        payload: Vec<u8>,
    ) -> Result<(), RuntimeError> {
        if payload.len() > self.wire.max_payload as usize {
            return Err(RuntimeError::Network(format!(
                "payload length {} exceeds max frame payload {}",
                payload.len(),
                self.wire.max_payload
            )));
        }
        let payload_len = payload.len();
        let frame = frame_payload(route, &self.wire, payload);
        send_frame(&mut self.stream, &frame).await?;
        self.stats.record_sent(payload_len);
        Ok(())
    }

    pub async fn recv_payload(&mut self) -> Result<(RuntimeRoute, Vec<u8>), RuntimeError> {
        let frame = recv_frame(&mut self.stream, self.wire.max_payload).await?;
        if frame.object_type != RUNTIME_PAYLOAD_OBJECT {
            return Err(RuntimeError::Network(format!(
                "unexpected runtime object type: {}",
                frame.object_type
            )));
        }
        frame.validate_object_version(1)?;
        if frame.scheme_id != self.wire.encode.scheme_id {
            return Err(RuntimeError::Network(format!(
                "scheme id mismatch: expected {}, got {}",
                self.wire.encode.scheme_id, frame.scheme_id
            )));
        }
        frame.validate_params_id(self.wire.encode.params_id)?;
        let route = RuntimeRoute::new(
            SessionId(frame.session_id),
            TaskId(frame.task_id),
            GateId(frame.gate_id),
            frame.message_kind,
        );
        self.stats.record_received(frame.payload.len());
        Ok((route, frame.payload))
    }

    pub async fn recv_expected_payload(
        &mut self,
        expected: RuntimeRoute,
    ) -> Result<Vec<u8>, RuntimeError> {
        let (actual, payload) = self.recv_payload().await?;
        expected.ensure_matches(actual)?;
        Ok(payload)
    }

    pub async fn send_u64_vec(
        &mut self,
        route: RuntimeRoute,
        values: &[u64],
    ) -> Result<(), RuntimeError> {
        let mut payload = Vec::with_capacity(8 + values.len() * 8);
        primitives::write_u64(&mut payload, values.len() as u64)
            .map_err(|err| RuntimeError::Network(err.to_string()))?;
        for &value in values {
            primitives::write_u64(&mut payload, value)
                .map_err(|err| RuntimeError::Network(err.to_string()))?;
        }
        self.send_payload(route, payload).await
    }

    pub async fn recv_u64_vec(&mut self) -> Result<(RuntimeRoute, Vec<u64>), RuntimeError> {
        let (route, payload) = self.recv_payload().await?;
        let mut cursor = Cursor::new(payload.as_slice());
        let len = primitives::read_u64(&mut cursor)
            .map_err(|err| RuntimeError::Network(err.to_string()))? as usize;
        let expected = 8usize
            .checked_add(
                len.checked_mul(8)
                    .ok_or(RuntimeError::InvalidValue("u64 vector length overflow"))?,
            )
            .ok_or(RuntimeError::InvalidValue(
                "u64 vector payload length overflow",
            ))?;
        if payload.len() != expected {
            return Err(RuntimeError::InvalidValue(
                "u64 vector payload length does not match header",
            ));
        }
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(
                primitives::read_u64(&mut cursor)
                    .map_err(|err| RuntimeError::Network(err.to_string()))?,
            );
        }
        Ok((route, values))
    }

    pub async fn recv_expected_u64_vec(
        &mut self,
        expected: RuntimeRoute,
    ) -> Result<Vec<u64>, RuntimeError> {
        let (actual, values) = self.recv_u64_vec().await?;
        expected.ensure_matches(actual)?;
        Ok(values)
    }

    pub async fn send_value(
        &mut self,
        route: RuntimeRoute,
        value: &Value,
    ) -> Result<(), RuntimeError> {
        self.send_payload(route, encode_value_payload(value)?).await
    }

    pub async fn recv_value(&mut self) -> Result<(RuntimeRoute, Value), RuntimeError> {
        let (route, payload) = self.recv_payload().await?;
        Ok((route, decode_value_payload(&payload)?))
    }

    pub async fn recv_expected_value(
        &mut self,
        expected: RuntimeRoute,
    ) -> Result<Value, RuntimeError> {
        let (actual, value) = self.recv_value().await?;
        expected.ensure_matches(actual)?;
        Ok(value)
    }
}

/// Peer-indexed communication mesh for one local runtime party.
pub struct RuntimeMesh {
    local_party: PartyId,
    peers: HashMap<PartyId, RuntimeChannel>,
}

impl RuntimeMesh {
    pub fn new(local_party: PartyId) -> Self {
        Self {
            local_party,
            peers: HashMap::new(),
        }
    }

    pub fn local_party(&self) -> PartyId {
        self.local_party
    }

    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    pub fn insert_peer(
        &mut self,
        peer: PartyId,
        channel: RuntimeChannel,
    ) -> Result<(), RuntimeError> {
        if peer == self.local_party {
            return Err(RuntimeError::InvalidConfig(
                "runtime mesh peer cannot be the local party",
            ));
        }
        if self.peers.contains_key(&peer) {
            return Err(RuntimeError::InvalidConfig(
                "runtime mesh peer channel already exists",
            ));
        }
        self.peers.insert(peer, channel);
        Ok(())
    }

    pub fn contains_peer(&self, peer: PartyId) -> bool {
        self.peers.contains_key(&peer)
    }

    pub fn peer_stats(&self, peer: PartyId) -> Result<RuntimeStats, RuntimeError> {
        self.peers
            .get(&peer)
            .map(RuntimeChannel::stats)
            .ok_or(RuntimeError::UnknownPeer(peer))
    }

    pub fn total_stats(&self) -> RuntimeStats {
        let mut stats = RuntimeStats::default();
        for channel in self.peers.values() {
            stats.merge(channel.stats());
        }
        stats
    }

    pub async fn send_payload(
        &mut self,
        peer: PartyId,
        route: RuntimeRoute,
        payload: Vec<u8>,
    ) -> Result<(), RuntimeError> {
        self.peer_mut(peer)?.send_payload(route, payload).await
    }

    pub async fn recv_payload(
        &mut self,
        peer: PartyId,
    ) -> Result<(RuntimeRoute, Vec<u8>), RuntimeError> {
        self.peer_mut(peer)?.recv_payload().await
    }

    pub async fn recv_expected_payload(
        &mut self,
        peer: PartyId,
        expected: RuntimeRoute,
    ) -> Result<Vec<u8>, RuntimeError> {
        self.peer_mut(peer)?.recv_expected_payload(expected).await
    }

    pub async fn send_u64_vec(
        &mut self,
        peer: PartyId,
        route: RuntimeRoute,
        values: &[u64],
    ) -> Result<(), RuntimeError> {
        self.peer_mut(peer)?.send_u64_vec(route, values).await
    }

    pub async fn recv_u64_vec(
        &mut self,
        peer: PartyId,
    ) -> Result<(RuntimeRoute, Vec<u64>), RuntimeError> {
        self.peer_mut(peer)?.recv_u64_vec().await
    }

    pub async fn recv_expected_u64_vec(
        &mut self,
        peer: PartyId,
        expected: RuntimeRoute,
    ) -> Result<Vec<u64>, RuntimeError> {
        self.peer_mut(peer)?.recv_expected_u64_vec(expected).await
    }

    pub async fn send_value(
        &mut self,
        peer: PartyId,
        route: RuntimeRoute,
        value: &Value,
    ) -> Result<(), RuntimeError> {
        self.peer_mut(peer)?.send_value(route, value).await
    }

    pub async fn recv_value(
        &mut self,
        peer: PartyId,
    ) -> Result<(RuntimeRoute, Value), RuntimeError> {
        self.peer_mut(peer)?.recv_value().await
    }

    pub async fn recv_expected_value(
        &mut self,
        peer: PartyId,
        expected: RuntimeRoute,
    ) -> Result<Value, RuntimeError> {
        self.peer_mut(peer)?.recv_expected_value(expected).await
    }

    fn peer_mut(&mut self, peer: PartyId) -> Result<&mut RuntimeChannel, RuntimeError> {
        self.peers
            .get_mut(&peer)
            .ok_or(RuntimeError::UnknownPeer(peer))
    }
}
