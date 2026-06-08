#![forbid(unsafe_code)]

//! Shared protocol runtime skeleton for SILENT protocol.
//!
//! This crate owns SILENT's lightweight execution boundary: parties,
//! sessions, typed values, operator dispatch, tracing, and framed transport.
//! Cryptographic kernels stay in `core` and reusable encrypted
//! operators stay in `silent-operators`; this layer only coordinates them
//! with SILENT's canonical `silent-net` and `silent-io` boundaries.

mod channel;
mod context;
mod error;
mod ids;
mod io;
mod program;
mod session;
mod value;

pub use channel::{
    RUNTIME_PAYLOAD_OBJECT, RuntimeChannel, RuntimeMesh, RuntimeStats, frame_payload,
};
pub use context::{RuntimeContext, TraceEvent};
pub use error::RuntimeError;
pub use ids::{
    FieldType, GateCursor, GateId, PartyId, ProtocolKind, RuntimeConfig, RuntimeRoute,
    RuntimeTopology, SessionId, TaskId, Visibility, WireContext,
};
pub use io::RuntimeIo;
pub use program::{
    NamedRuntimeProgram, Operator, OperatorRegistry, RuntimeInstruction, RuntimeProgram,
    RuntimeStep,
};
pub use session::RuntimeSession;
pub use value::{
    SymbolTable, Value, ValueData, ValueType, decode_value_payload, encode_value_payload,
};

pub mod prelude {
    pub use crate::{
        FieldType, GateCursor, GateId, NamedRuntimeProgram, Operator, OperatorRegistry, PartyId,
        ProtocolKind, RuntimeChannel, RuntimeConfig, RuntimeContext, RuntimeError,
        RuntimeInstruction, RuntimeIo, RuntimeMesh, RuntimeProgram, RuntimeRoute, RuntimeSession,
        RuntimeStats, RuntimeStep, RuntimeTopology, SessionId, SymbolTable, TaskId, TraceEvent,
        Value, ValueData, ValueType, Visibility, WireContext, decode_value_payload,
        encode_value_payload, frame_payload,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use silent_net::frame::{FRAME_HEADER_SIZE, MessageKind};
    use silent_net::transport::Connection;
    use silent_net::transport::memory::MemoryConnection;
    use silent_net::transport::tcp::TcpTransport;

    struct EchoLen;

    impl Operator for EchoLen {
        fn name(&self) -> &'static str {
            "echo_len"
        }

        fn eval(
            &self,
            _ctx: &mut RuntimeContext,
            inputs: &[Value],
        ) -> Result<Vec<Value>, RuntimeError> {
            Value::public_u64(vec![inputs.len() as u64], FieldType::Ring64, vec![1])
                .map(|v| vec![v])
        }
    }

    #[test]
    fn config_validates_party_uniqueness() {
        let mut cfg = RuntimeConfig::two_party_inference();
        cfg.parties = vec![PartyId(0), PartyId(0)];
        assert!(matches!(
            cfg.validate(),
            Err(RuntimeError::InvalidConfig(_))
        ));
    }

    #[test]
    fn topology_sorts_and_lists_peers() {
        let topology =
            RuntimeTopology::new(vec![PartyId(2), PartyId(0), PartyId(1)]).expect("topology");
        assert_eq!(topology.parties(), &[PartyId(0), PartyId(1), PartyId(2)]);
        assert_eq!(topology.index_of(PartyId(1)), Some(1));
        assert_eq!(
            topology.peers(PartyId(1)).expect("peers"),
            vec![PartyId(0), PartyId(2)]
        );
        assert!(matches!(
            RuntimeTopology::new(vec![PartyId(0), PartyId(0)]),
            Err(RuntimeError::InvalidConfig(_))
        ));
    }

    #[test]
    fn registry_dispatch_records_trace() {
        let mut registry = OperatorRegistry::default();
        registry.register(EchoLen).expect("register");
        let mut cfg = RuntimeConfig::two_party_inference();
        cfg.enable_operator_trace = true;
        let mut ctx = RuntimeContext::new(cfg, PartyId(0), SessionId(7), WireContext::default())
            .expect("context");

        let out = registry.eval("echo_len", &mut ctx, &[]).expect("eval");
        assert_eq!(out[0].data, ValueData::PublicU64(vec![0]));
        assert_eq!(ctx.trace()[0].op, "echo_len");
        assert_eq!(registry.names(), vec!["echo_len"]);
        assert!(registry.contains("echo_len"));
    }

    #[test]
    fn runtime_program_appends_step_outputs() {
        let mut registry = OperatorRegistry::default();
        registry.register(EchoLen).expect("register");
        let mut cfg = RuntimeConfig::two_party_inference();
        cfg.enable_operator_trace = true;
        let mut ctx = RuntimeContext::new(cfg, PartyId(0), SessionId(7), WireContext::default())
            .expect("context");
        let mut program = RuntimeProgram::new();
        program.push_step(RuntimeStep::new("echo_len", vec![]));
        program.push_step(RuntimeStep::new("echo_len", vec![0]));

        let values = program.eval(&registry, &mut ctx, vec![]).expect("eval");

        assert_eq!(values.len(), 2);
        assert_eq!(values[0].data, ValueData::PublicU64(vec![0]));
        assert_eq!(values[1].data, ValueData::PublicU64(vec![1]));
        assert_eq!(ctx.trace().len(), 2);
    }

    #[test]
    fn symbol_table_sets_gets_and_deletes_values() {
        let mut symbols = SymbolTable::new();
        let value = Value::public_u64(vec![9], FieldType::Ring64, vec![1]).expect("value");

        symbols.set_var("x", value.clone()).expect("set");
        assert!(symbols.has_var("x"));
        assert_eq!(symbols.get_var("x").expect("get"), &value);
        assert_eq!(symbols.names(), vec!["x"]);
        assert_eq!(symbols.del_var("x"), Some(value));
        assert!(matches!(
            symbols.get_var("x"),
            Err(RuntimeError::UnknownSymbol(name)) if name == "x"
        ));
    }

    #[test]
    fn named_runtime_program_executes_against_symbols() {
        let mut registry = OperatorRegistry::default();
        registry.register(EchoLen).expect("register");
        let mut cfg = RuntimeConfig::two_party_inference();
        cfg.enable_operator_trace = true;
        let mut ctx = RuntimeContext::new(cfg, PartyId(0), SessionId(7), WireContext::default())
            .expect("context");
        let mut symbols = SymbolTable::new();
        let mut program = NamedRuntimeProgram::new();
        program.push_instruction(
            RuntimeInstruction::new("echo_len", std::iter::empty::<&str>(), ["x"])
                .expect("instruction"),
        );
        program.push_instruction(
            RuntimeInstruction::new("echo_len", ["x"], ["y"]).expect("instruction"),
        );

        program
            .eval(&registry, &mut ctx, &mut symbols)
            .expect("program eval");

        assert_eq!(
            &symbols.get_var("x").expect("x").data,
            &ValueData::PublicU64(vec![0])
        );
        assert_eq!(
            &symbols.get_var("y").expect("y").data,
            &ValueData::PublicU64(vec![1])
        );
        assert_eq!(ctx.trace().len(), 2);
    }

    #[test]
    fn route_converts_to_net_route_key() {
        let route = RuntimeRoute::new(SessionId(1), TaskId(2), GateId(3), MessageKind::Data);
        let key = route.route_key();
        assert_eq!(key.session_id, 1);
        assert_eq!(key.task_id, 2);
        assert_eq!(key.gate_id, 3);
        assert_eq!(key.message_kind, MessageKind::Data);
    }

    #[test]
    fn gate_cursor_allocates_monotonic_routes() {
        let mut cursor = GateCursor::new(SessionId(3), TaskId(8));
        let first = cursor.next_route(MessageKind::Setup).expect("first");
        let second = cursor.next_route(MessageKind::Data).expect("second");

        assert_eq!(
            first,
            RuntimeRoute::new(SessionId(3), TaskId(8), GateId(0), MessageKind::Setup)
        );
        assert_eq!(
            second,
            RuntimeRoute::new(SessionId(3), TaskId(8), GateId(1), MessageKind::Data)
        );
        assert_eq!(cursor.next_gate(), GateId(2));
    }

    #[test]
    fn runtime_value_payload_roundtrips_and_rejects_trailing_bytes() {
        let public = Value::public_u64(vec![7, 8], FieldType::Prime(97), vec![2]).expect("public");
        let decoded = decode_value_payload(&encode_value_payload(&public).expect("encode"))
            .expect("decode public");
        assert_eq!(decoded, public);

        let secret =
            Value::secret_bytes(vec![1, 2, 3], FieldType::Ring64, vec![3]).expect("secret");
        let decoded = decode_value_payload(&encode_value_payload(&secret).expect("encode"))
            .expect("decode secret");
        assert_eq!(decoded, secret);

        let handle = Value::handle(
            "backend:42",
            Visibility::Private(PartyId(1)),
            FieldType::Ring64,
            vec![1],
        )
        .expect("handle");
        let mut payload = encode_value_payload(&handle).expect("encode handle");
        let decoded = decode_value_payload(&payload).expect("decode handle");
        assert_eq!(decoded, handle);

        payload.push(0);
        assert!(matches!(
            decode_value_payload(&payload),
            Err(RuntimeError::Codec(_))
        ));
    }

    #[test]
    fn runtime_io_sets_public_symbols_and_rejects_secret_outfeed() {
        let io = RuntimeIo::new(RuntimeConfig::two_party_inference(), WireContext::default())
            .expect("io");
        let mut symbols = SymbolTable::new();

        io.set_public_u64(&mut symbols, "x", vec![1, 2], vec![2])
            .expect("set public");
        io.set_secret_bytes(&mut symbols, "s", vec![7, 8], vec![2])
            .expect("set secret");

        assert_eq!(
            io.get_public_u64(&symbols, "x").expect("get public"),
            vec![1, 2]
        );
        assert!(matches!(
            io.get_public_u64(&symbols, "s"),
            Err(RuntimeError::InvalidValue(_))
        ));
    }

    #[test]
    fn runtime_io_container_roundtrips_values() {
        let io = RuntimeIo::new(RuntimeConfig::two_party_inference(), WireContext::default())
            .expect("io");
        let value = Value::public_u64(vec![3, 4, 5], FieldType::Ring64, vec![3]).expect("value");
        let mut encoded = Vec::new();

        io.write_value_container(&mut encoded, &value)
            .expect("write container");
        let decoded = io
            .read_value_container(&mut std::io::Cursor::new(&encoded))
            .expect("read container");
        assert_eq!(decoded, value);

        let mut wrong_cfg = RuntimeConfig::two_party_inference();
        wrong_cfg.field = FieldType::Prime(97);
        let wrong_io = RuntimeIo::new(wrong_cfg, WireContext::default()).expect("wrong io");
        assert!(matches!(
            wrong_io.read_value_container(&mut std::io::Cursor::new(&encoded)),
            Err(RuntimeError::InvalidValue(_))
        ));
    }

    #[tokio::test]
    async fn runtime_channel_roundtrips_u64_vec_over_silent_net() {
        let (a, b) = MemoryConnection::pair();
        let stream_a = a.open_stream().await.expect("open stream");
        let stream_b = b.accept_stream().await.expect("accept stream");
        let mut chan_a = RuntimeChannel::new(stream_a, WireContext::default());
        let mut chan_b = RuntimeChannel::new(stream_b, WireContext::default());
        let route = RuntimeRoute::new(SessionId(9), TaskId(4), GateId(2), MessageKind::Data);

        chan_a
            .send_u64_vec(route, &[10, 20, 30])
            .await
            .expect("send");
        let (got_route, got_values) = chan_b.recv_u64_vec().await.expect("recv");

        assert_eq!(got_route, route);
        assert_eq!(got_values, vec![10, 20, 30]);
        assert_eq!(chan_a.stats().frames_sent, 1);
        assert!(chan_a.stats().bytes_sent >= FRAME_HEADER_SIZE as u64 + 32);
        assert_eq!(chan_b.stats().frames_received, 1);
    }

    #[tokio::test]
    async fn runtime_channel_roundtrips_values_and_checks_routes() {
        let (a, b) = MemoryConnection::pair();
        let stream_a = a.open_stream().await.expect("open stream");
        let stream_b = b.accept_stream().await.expect("accept stream");
        let mut chan_a = RuntimeChannel::new(stream_a, WireContext::default());
        let mut chan_b = RuntimeChannel::new(stream_b, WireContext::default());
        let route = RuntimeRoute::new(SessionId(12), TaskId(2), GateId(0), MessageKind::Data);
        let value = Value::public_u64(vec![5, 6], FieldType::Ring64, vec![2]).expect("value");

        chan_a.send_value(route, &value).await.expect("send value");
        let got = chan_b
            .recv_expected_value(route)
            .await
            .expect("recv expected value");
        assert_eq!(got, value);

        let actual = RuntimeRoute::new(SessionId(12), TaskId(2), GateId(1), MessageKind::Data);
        let expected = RuntimeRoute::new(SessionId(12), TaskId(2), GateId(2), MessageKind::Data);
        chan_a
            .send_u64_vec(actual, &[1])
            .await
            .expect("send mismatch payload");
        assert!(matches!(
            chan_b.recv_expected_u64_vec(expected).await,
            Err(RuntimeError::RouteMismatch { expected: e, actual: a })
                if e == expected && a == actual
        ));
    }

    #[tokio::test]
    async fn runtime_mesh_routes_by_peer_over_silent_net() {
        let (a, b) = MemoryConnection::pair();
        let stream_a = a.open_stream().await.expect("open stream");
        let stream_b = b.accept_stream().await.expect("accept stream");
        let mut mesh_a = RuntimeMesh::new(PartyId(0));
        let mut mesh_b = RuntimeMesh::new(PartyId(1));
        mesh_a
            .insert_peer(
                PartyId(1),
                RuntimeChannel::new(stream_a, WireContext::default()),
            )
            .expect("insert peer a");
        mesh_b
            .insert_peer(
                PartyId(0),
                RuntimeChannel::new(stream_b, WireContext::default()),
            )
            .expect("insert peer b");
        let route = RuntimeRoute::new(SessionId(9), TaskId(4), GateId(3), MessageKind::Data);

        mesh_a
            .send_u64_vec(PartyId(1), route, &[42, 99])
            .await
            .expect("mesh send");
        let (got_route, got_values) = mesh_b.recv_u64_vec(PartyId(0)).await.expect("mesh recv");

        assert_eq!(got_route, route);
        assert_eq!(got_values, vec![42, 99]);
        assert_eq!(mesh_a.peer_stats(PartyId(1)).expect("stats").frames_sent, 1);
        assert_eq!(
            mesh_b
                .peer_stats(PartyId(0))
                .expect("stats")
                .frames_received,
            1
        );
        assert!(matches!(
            mesh_b.recv_u64_vec(PartyId(2)).await,
            Err(RuntimeError::UnknownPeer(PartyId(2)))
        ));
    }

    #[tokio::test]
    async fn runtime_session_sends_values_with_auto_routes() {
        let (a, b) = MemoryConnection::pair();
        let stream_a = a.open_stream().await.expect("open stream");
        let stream_b = b.accept_stream().await.expect("accept stream");
        let cfg = RuntimeConfig::two_party_inference();
        let ctx_a = RuntimeContext::new(
            cfg.clone(),
            PartyId(0),
            SessionId(55),
            WireContext::default(),
        )
        .expect("ctx a");
        let ctx_b = RuntimeContext::new(cfg, PartyId(1), SessionId(55), WireContext::default())
            .expect("ctx b");
        let mut mesh_a = RuntimeMesh::new(PartyId(0));
        let mut mesh_b = RuntimeMesh::new(PartyId(1));
        mesh_a
            .insert_peer(
                PartyId(1),
                RuntimeChannel::new(stream_a, WireContext::default()),
            )
            .expect("peer a");
        mesh_b
            .insert_peer(
                PartyId(0),
                RuntimeChannel::new(stream_b, WireContext::default()),
            )
            .expect("peer b");
        let mut session_a = RuntimeSession::with_mesh(ctx_a, mesh_a).expect("session a");
        let mut session_b = RuntimeSession::with_mesh(ctx_b, mesh_b).expect("session b");
        let value = Value::public_u64(vec![1, 2, 3], FieldType::Ring64, vec![3]).expect("value");

        let route = session_a
            .send_value(PartyId(1), TaskId(9), MessageKind::Data, &value)
            .await
            .expect("send value");
        let got = session_b
            .recv_expected_value(PartyId(0), route)
            .await
            .expect("recv value");

        assert_eq!(route.gate, GateId(0));
        assert_eq!(got, value);
        assert_eq!(session_a.stats().frames_sent, 1);
        assert_eq!(session_b.stats().frames_received, 1);

        let route2 = session_a
            .send_u64_vec(PartyId(1), TaskId(9), MessageKind::Data, &[7])
            .await
            .expect("send u64");
        assert_eq!(route2.gate, GateId(1));
        assert_eq!(
            session_b
                .recv_expected_u64_vec(PartyId(0), route2)
                .await
                .expect("recv u64"),
            vec![7]
        );
    }

    #[tokio::test]
    async fn runtime_session_roundtrips_values_over_tcp_loopback() {
        let listener = TcpTransport::new()
            .bind("127.0.0.1:0")
            .await
            .expect("bind tcp listener");
        let addr = listener.local_addr().expect("local addr");
        let accept_task = tokio::spawn(async move {
            let conn = listener.accept().await.expect("accept tcp");
            conn.open_stream().await.expect("server stream")
        });
        let client_conn = TcpTransport::new()
            .connect(&addr.to_string())
            .await
            .expect("connect tcp");
        let stream_a = client_conn.open_stream().await.expect("client stream");
        let stream_b = accept_task.await.expect("accept task");

        let cfg = RuntimeConfig::two_party_inference();
        let wire = WireContext::default();
        let ctx_a = RuntimeContext::new(cfg.clone(), PartyId(0), SessionId(77), wire.clone())
            .expect("ctx a");
        let ctx_b =
            RuntimeContext::new(cfg, PartyId(1), SessionId(77), wire.clone()).expect("ctx b");
        let mut mesh_a = RuntimeMesh::new(PartyId(0));
        let mut mesh_b = RuntimeMesh::new(PartyId(1));
        mesh_a
            .insert_peer(PartyId(1), RuntimeChannel::new(stream_a, wire.clone()))
            .expect("peer a");
        mesh_b
            .insert_peer(PartyId(0), RuntimeChannel::new(stream_b, wire))
            .expect("peer b");
        let mut session_a = RuntimeSession::with_mesh(ctx_a, mesh_a).expect("session a");
        let mut session_b = RuntimeSession::with_mesh(ctx_b, mesh_b).expect("session b");
        let value =
            Value::public_u64(vec![11, 13, 17, 19], FieldType::Ring64, vec![4]).expect("value");

        let route = session_a
            .send_value(PartyId(1), TaskId(21), MessageKind::Data, &value)
            .await
            .expect("send value");
        let got = session_b
            .recv_expected_value(PartyId(0), route)
            .await
            .expect("recv value");

        assert_eq!(got, value);
        assert_eq!(session_a.stats().frames_sent, 1);
        assert_eq!(session_b.stats().frames_received, 1);
        assert!(session_a.stats().bytes_sent > FRAME_HEADER_SIZE as u64);
        assert_eq!(
            session_a.stats().bytes_sent,
            session_b.stats().bytes_received
        );
    }
}
