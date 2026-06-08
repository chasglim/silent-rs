//! TCP transport using `tokio::net::TcpStream`.
//!
//! TCP is positioned as a LAN/debug fallback.  For WAN head-of-line resistant
//! transport, use the QUIC transport (`quic` feature) instead.
//!
//! # Architecture
//!
//! Each TCP connection carries a single bidirectional [`Stream`].  The
//! sender frames messages with [`crate::frame::send_frame`]; the receiver
//! reads the fixed-size header first to determine the payload length.

use super::{Connection, Stream, Transport};
use async_trait::async_trait;
use bytes::Bytes;
use std::io;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream as TokioTcpStream};

/// A bidirectional byte stream over a TCP socket.
pub struct TcpStream {
    inner: TokioTcpStream,
}

impl TcpStream {
    /// Wrap an existing TCP stream.
    pub fn new(inner: TokioTcpStream) -> Self {
        Self { inner }
    }

    /// Consume and return the inner Tokio TCP stream.
    pub fn into_inner(self) -> TokioTcpStream {
        self.inner
    }
}

#[async_trait]
impl Stream for TcpStream {
    async fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.inner.read_exact(buf).await.map(|_| ())
    }

    async fn read_chunk(&mut self, max_len: usize) -> io::Result<Option<Bytes>> {
        let mut buf = vec![0u8; max_len];
        match self.inner.read(&mut buf).await {
            Ok(0) => Ok(None),
            Ok(n) => {
                buf.truncate(n);
                Ok(Some(buf.into()))
            }
            Err(e) => Err(e),
        }
    }

    async fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.inner.write_all(buf).await
    }
}

/// A TCP connection that wraps a single bidirectional stream.
///
/// Since TCP does not natively multiplex streams, `open_stream()` returns
/// a new `TcpStream` that writes to the same underlying socket.  For proper
/// multiplexing, use the QUIC transport.
pub struct TcpConnection {
    stream: tokio::sync::Mutex<Option<TcpStream>>,
}

impl TcpConnection {
    /// Wrap an existing connected TCP stream.
    pub fn new(stream: TokioTcpStream) -> Self {
        Self {
            stream: tokio::sync::Mutex::new(Some(TcpStream::new(stream))),
        }
    }

    /// Helper: take the stream out of the mutex, returning a None for future calls.
    async fn take_stream(&self) -> io::Result<Box<dyn Stream>> {
        let mut guard = self.stream.lock().await;
        guard
            .take()
            .map(|s| Box::new(s) as Box<dyn Stream>)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotConnected, "TCP connection closed"))
    }
}

#[async_trait]
impl Connection for TcpConnection {
    async fn open_stream(&self) -> io::Result<Box<dyn Stream>> {
        self.take_stream().await
    }

    async fn accept_stream(&self) -> io::Result<Box<dyn Stream>> {
        self.take_stream().await
    }
}

/// A simple TCP transport that can connect to peers or listen for incoming
/// connections.
pub struct TcpTransport;

impl TcpTransport {
    pub fn new() -> Self {
        Self
    }

    /// Connect to a TCP endpoint and return the connection.
    pub async fn connect(&self, addr: &str) -> io::Result<TcpConnection> {
        let stream = TokioTcpStream::connect(addr).await?;
        Ok(TcpConnection::new(stream))
    }

    /// Bind and listen on a TCP address, returning a listener.
    pub async fn bind(&self, addr: &str) -> io::Result<TcpListenTransport> {
        let listener = TcpListener::bind(addr).await?;
        Ok(TcpListenTransport { listener })
    }
}

impl Default for TcpTransport {
    fn default() -> Self {
        Self::new()
    }
}

/// A bound TCP listener that accepts incoming connections.
pub struct TcpListenTransport {
    listener: TcpListener,
}

impl TcpListenTransport {
    /// Accept an incoming connection.
    pub async fn accept(&self) -> io::Result<TcpConnection> {
        let (stream, _addr) = self.listener.accept().await?;
        Ok(TcpConnection::new(stream))
    }

    /// Return the local address the listener is bound to.
    pub fn local_addr(&self) -> io::Result<std::net::SocketAddr> {
        self.listener.local_addr()
    }
}

#[async_trait]
impl Transport for TcpTransport {
    async fn connect(&self, addr: &str) -> io::Result<Box<dyn Connection>> {
        let stream = TokioTcpStream::connect(addr).await?;
        Ok(Box::new(TcpConnection::new(stream)))
    }

    async fn accept(&self) -> io::Result<Box<dyn Connection>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "TcpTransport does not listen; use TcpListenTransport for accept",
        ))
    }
}

#[async_trait]
impl Transport for TcpListenTransport {
    async fn connect(&self, _addr: &str) -> io::Result<Box<dyn Connection>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "TcpListenTransport does not connect; use TcpTransport::connect",
        ))
    }

    async fn accept(&self) -> io::Result<Box<dyn Connection>> {
        let (stream, _addr) = self.listener.accept().await?;
        Ok(Box::new(TcpConnection::new(stream)))
    }
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

    #[tokio::test]
    async fn tcp_send_recv_single_frame() {
        let transport = TcpTransport::new();
        let listener = transport.bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let frame = make_frame("hello from tcp");
        let frame_clone = frame.clone();

        let server_handle = tokio::spawn(async move {
            let conn = listener.accept().await.unwrap();
            let mut stream = conn.open_stream().await.unwrap();
            recv_frame(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .unwrap()
        });

        let client_handle = tokio::spawn(async move {
            let conn = transport.connect(&addr.to_string()).await.unwrap();
            let mut stream = conn.open_stream().await.unwrap();
            send_frame(&mut stream, &frame).await.unwrap();
        });

        client_handle.await.unwrap();
        let received = server_handle.await.unwrap();

        assert_eq!(received, frame_clone);
    }

    #[tokio::test]
    async fn tcp_send_recv_multiple_frames() {
        let transport = TcpTransport::new();
        let listener = transport.bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let frames: Vec<NetworkFrame> =
            (0..10).map(|i| make_frame(&format!("frame_{i}"))).collect();
        let frames_clone = frames.clone();

        let server_handle = tokio::spawn(async move {
            let conn = listener.accept().await.unwrap();
            let mut stream = conn.open_stream().await.unwrap();
            let mut received = Vec::new();
            for _ in 0..10 {
                received.push(
                    recv_frame(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD)
                        .await
                        .unwrap(),
                );
            }
            received
        });

        let client_handle = tokio::spawn(async move {
            let conn = transport.connect(&addr.to_string()).await.unwrap();
            let mut stream = conn.open_stream().await.unwrap();
            for f in &frames_clone {
                send_frame(&mut stream, f).await.unwrap();
            }
        });

        client_handle.await.unwrap();
        let received = server_handle.await.unwrap();

        assert_eq!(received, frames);
    }

    #[tokio::test]
    async fn tcp_send_recv_large_payload() {
        let transport = TcpTransport::new();
        let listener = transport.bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let payload_len = 128_000;
        let frame = NetworkFrame {
            payload: vec![0x42u8; payload_len],
            ..make_frame("")
        };

        let server_handle = tokio::spawn(async move {
            let conn = listener.accept().await.unwrap();
            let mut stream = conn.open_stream().await.unwrap();
            recv_frame(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .unwrap()
        });

        let client_handle = tokio::spawn(async move {
            let conn = transport.connect(&addr.to_string()).await.unwrap();
            let mut stream = conn.open_stream().await.unwrap();
            send_frame(&mut stream, &frame).await.unwrap();
        });

        client_handle.await.unwrap();
        let received = server_handle.await.unwrap();

        assert_eq!(received.payload.len(), payload_len);
        assert!(received.payload.iter().all(|&b| b == 0x42));
    }

    #[tokio::test]
    async fn tcp_bidirectional() {
        let transport = TcpTransport::new();
        let listener = transport.bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move {
            let conn = transport.connect(&addr.to_string()).await.unwrap();
            let mut stream = conn.open_stream().await.unwrap();

            // Send a frame
            send_frame(&mut stream, &make_frame("client->server"))
                .await
                .unwrap();

            // Receive a frame
            let reply = recv_frame(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .unwrap();

            assert_eq!(reply.payload, b"server->client");
        });

        let server_handle = tokio::spawn(async move {
            let conn = listener.accept().await.unwrap();
            let mut stream = conn.open_stream().await.unwrap();

            // Receive a frame
            let request = recv_frame(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .unwrap();

            assert_eq!(request.payload, b"client->server");

            // Send a reply
            send_frame(&mut stream, &make_frame("server->client"))
                .await
                .unwrap();
        });

        client_handle.await.unwrap();
        server_handle.await.unwrap();
    }
}
