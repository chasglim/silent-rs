//! QUIC transport using `quinn`.
//!
//! Each QUIC connection supports multiple independent bidirectional streams,
//! allowing independent MPC tasks to avoid head-of-line blocking at the
//! transport level.
//!
//! Enable with `cargo build --features quic`.

use super::{Connection, Stream};
use async_trait::async_trait;
use bytes::Bytes;
use quinn::{Endpoint, RecvStream, SendStream};
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;

/// A bidirectional QUIC stream wrapping a `(SendStream, RecvStream)` pair.
pub struct QuicStream {
    send: SendStream,
    recv: RecvStream,
}

impl QuicStream {
    pub fn new(send: SendStream, recv: RecvStream) -> Self {
        Self { send, recv }
    }
}

#[async_trait]
impl Stream for QuicStream {
    async fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.recv
            .read_exact(buf)
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::ConnectionReset, e.to_string()))
    }

    async fn read_chunk(&mut self, max_len: usize) -> io::Result<Option<Bytes>> {
        let mut buf = vec![0u8; max_len];
        match self.recv.read(&mut buf).await {
            Ok(Some(n)) => {
                buf.truncate(n);
                Ok(Some(buf.into()))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(io::Error::new(io::ErrorKind::Other, e.to_string())),
        }
    }

    async fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.send
            .write_all(buf)
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e.to_string()))
    }
}

/// A QUIC connection that can open and accept multiplexed streams.
///
/// In quinn 0.10, `accept_bi()` returns one stream per call;
/// `open_bi()` creates a new outgoing stream.
pub struct QuicConnection {
    connection: quinn::Connection,
}

impl QuicConnection {
    pub fn new(connection: quinn::Connection) -> Self {
        Self { connection }
    }
}

#[async_trait]
impl Connection for QuicConnection {
    async fn open_stream(&self) -> io::Result<Box<dyn Stream>> {
        let (send, recv) = self
            .connection
            .open_bi()
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::ConnectionRefused, e.to_string()))?;

        Ok(Box::new(QuicStream::new(send, recv)))
    }

    async fn accept_stream(&self) -> io::Result<Box<dyn Stream>> {
        let (send, recv) = self
            .connection
            .accept_bi()
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::ConnectionReset, e.to_string()))?;

        Ok(Box::new(QuicStream::new(send, recv)))
    }
}

/// Configuration for a QUIC server.
pub struct QuicServerConfig {
    pub bind_addr: SocketAddr,
    pub cert_chain: Vec<rustls::Certificate>,
    pub private_key: rustls::PrivateKey,
    pub alpn_protocols: Vec<Vec<u8>>,
}

impl QuicServerConfig {
    pub fn new(
        bind_addr: SocketAddr,
        cert_chain: Vec<rustls::Certificate>,
        private_key: rustls::PrivateKey,
    ) -> Self {
        Self {
            bind_addr,
            cert_chain,
            private_key,
            alpn_protocols: vec![b"silent".to_vec()],
        }
    }
}

/// A QUIC server that accepts connections.
pub struct QuicServer {
    endpoint: Endpoint,
}

impl QuicServer {
    /// Create a new QUIC server.
    ///
    /// Returns the server and the actual `SocketAddr` it is bound to.
    pub async fn new(config: QuicServerConfig) -> io::Result<(Self, SocketAddr)> {
        let mut server_crypto = rustls::ServerConfig::builder()
            .with_safe_defaults()
            .with_no_client_auth()
            .with_single_cert(config.cert_chain, config.private_key)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;

        server_crypto.alpn_protocols = config.alpn_protocols;

        let server_config = quinn::ServerConfig::with_crypto(Arc::new(server_crypto));
        let endpoint = Endpoint::server(server_config, config.bind_addr)
            .map_err(|e| io::Error::new(io::ErrorKind::AddrInUse, e.to_string()))?;

        let local_addr = endpoint.local_addr()?;
        Ok((Self { endpoint }, local_addr))
    }

    /// Accept an incoming connection.
    pub async fn accept(&self) -> io::Result<QuicConnection> {
        match self.endpoint.accept().await {
            Some(connecting) => {
                let connection = connecting
                    .await
                    .map_err(|e| io::Error::new(io::ErrorKind::ConnectionRefused, e.to_string()))?;
                Ok(QuicConnection::new(connection))
            }
            None => Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "server endpoint closed",
            )),
        }
    }
}

/// A QUIC client for connecting to servers.
pub struct QuicClient {
    endpoint: Endpoint,
}

/// A certificate verifier that accepts any server certificate.
/// For localhost testing only.
#[cfg(test)]
struct NoServerVerification;

#[cfg(test)]
impl rustls::client::ServerCertVerifier for NoServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::Certificate,
        _intermediates: &[rustls::Certificate],
        _server_name: &rustls::ServerName,
        _scts: &mut dyn Iterator<Item = &[u8]>,
        _ocsp_response: &[u8],
        _now: std::time::SystemTime,
    ) -> Result<rustls::client::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::ServerCertVerified::assertion())
    }
}

impl QuicClient {
    /// Create a QUIC client that trusts the given server certificate(s).
    pub fn with_server_cert(certs: Vec<rustls::Certificate>) -> io::Result<Self> {
        let mut roots = rustls::RootCertStore::empty();
        for cert in &certs {
            roots
                .add(cert)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
        }

        let client_crypto = rustls::ClientConfig::builder()
            .with_safe_defaults()
            .with_root_certificates(roots)
            .with_no_client_auth();

        Self::build_endpoint(client_crypto)
    }

    /// Create a QUIC client that skips server certificate verification.
    /// For localhost testing only.
    #[cfg(test)]
    fn new_insecure() -> io::Result<Self> {
        let client_crypto = rustls::ClientConfig::builder()
            .with_safe_defaults()
            .with_custom_certificate_verifier(std::sync::Arc::new(NoServerVerification))
            .with_no_client_auth();

        Self::build_endpoint(client_crypto)
    }

    fn build_endpoint(mut client_crypto: rustls::ClientConfig) -> io::Result<Self> {
        client_crypto.alpn_protocols = vec![b"silent".to_vec()];

        let mut client_cfg = quinn::ClientConfig::new(Arc::new(client_crypto));

        let mut transport = quinn::TransportConfig::default();
        transport.max_concurrent_bidi_streams(256u32.into());
        transport.keep_alive_interval(Some(std::time::Duration::from_secs(5)));
        client_cfg.transport_config(Arc::new(transport));

        let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        endpoint.set_default_client_config(client_cfg);

        Ok(Self { endpoint })
    }

    /// Connect to a QUIC server.
    pub async fn connect(&self, addr: SocketAddr, server_name: &str) -> io::Result<QuicConnection> {
        let connection = self
            .endpoint
            .connect(addr, server_name)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::ConnectionRefused, e.to_string()))?;

        Ok(QuicConnection::new(connection))
    }

    /// Gracefully close the client endpoint.
    pub fn close(self) {
        self.endpoint.close(0u32.into(), b"client shutdown");
    }
}

// ── Self-signed certificate for localhost testing ──────────────────────────

/// Generate a self-signed CA certificate and private key for localhost testing.
#[cfg(test)]
pub fn generate_self_signed_cert() -> (Vec<rustls::Certificate>, rustls::PrivateKey) {
    let mut params =
        rcgen::CertificateParams::new(vec!["localhost".to_string()]).expect("cert params");
    params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    let key_pair = rcgen::KeyPair::generate().expect("key generation");
    let cert = params.self_signed(&key_pair).expect("self-sign");

    let cert_der = cert.der().to_vec();
    let key_der = key_pair.serialize_der();

    (
        vec![rustls::Certificate(cert_der)],
        rustls::PrivateKey(key_der),
    )
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{
        DEFAULT_MAX_FRAME_PAYLOAD, MessageKind, NetworkFrame, recv_frame, send_frame,
    };
    use silent_io::{ObjectType, ParamsId};

    fn make_frame(payload: &str) -> NetworkFrame {
        NetworkFrame {
            session_id: 1,
            task_id: 2,
            gate_id: 3,
            message_kind: MessageKind::Data,
            object_type: ObjectType::RLWE_CIPHERTEXT,
            object_version: 1,
            scheme_id: 0,
            params_id: ParamsId::ZERO,
            flags: 0,
            payload: payload.as_bytes().to_vec(),
        }
    }

    async fn setup_server_client() -> (QuicServer, QuicClient, SocketAddr) {
        let (certs, key) = generate_self_signed_cert();
        let (server, addr) = QuicServer::new(QuicServerConfig::new(
            "127.0.0.1:0".parse().unwrap(),
            certs,
            key,
        ))
        .await
        .unwrap();

        let client = QuicClient::new_insecure().unwrap();
        (server, client, addr)
    }

    #[tokio::test]
    async fn quic_send_recv_single_frame() {
        let (server, client, addr) = setup_server_client().await;

        let frame = make_frame("hello from quic");
        let frame_clone = frame.clone();

        // Start server accept first so it's ready.
        let server_handle = tokio::spawn(async move {
            let conn = server.accept().await.expect("server accept");
            let mut stream = conn.accept_stream().await.expect("server accept_stream");
            let received = recv_frame(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .expect("server recv_frame");
            received
        });

        // Brief yield to let server spawn start.
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;

        let conn = client
            .connect(addr, "localhost")
            .await
            .expect("client connect");
        let mut stream = conn.open_stream().await.expect("client open_stream");
        send_frame(&mut stream, &frame)
            .await
            .expect("client send_frame");

        // Don't drop conn until server is done.
        let received = server_handle.await.expect("server join");
        assert_eq!(received, frame_clone);
        drop(conn); // explicit
    }

    #[tokio::test]
    async fn quic_multiple_streams() {
        let (server, client, addr) = setup_server_client().await;

        let frame_a = make_frame("stream_a");
        let frame_b = make_frame("stream_b");
        let fa = frame_a.clone();
        let fb = frame_b.clone();

        let server_handle = tokio::spawn(async move {
            let conn = server.accept().await.expect("server accept");

            let mut stream1 = conn.accept_stream().await.expect("server accept stream1");
            let mut stream2 = conn.accept_stream().await.expect("server accept stream2");

            let r1 = recv_frame(&mut stream1, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .expect("recv1");
            let r2 = recv_frame(&mut stream2, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .expect("recv2");

            (r1, r2)
        });

        tokio::time::sleep(std::time::Duration::from_millis(5)).await;

        let conn = client
            .connect(addr, "localhost")
            .await
            .expect("client connect");
        let mut s1 = conn.open_stream().await.expect("client open_stream 1");
        let mut s2 = conn.open_stream().await.expect("client open_stream 2");

        send_frame(&mut s1, &frame_a).await.expect("send1");
        send_frame(&mut s2, &frame_b).await.expect("send2");

        let (r1, r2) = server_handle.await.expect("server join");

        assert_eq!(r1, fa);
        assert_eq!(r2, fb);
        drop(conn);
    }
}
