use async_trait::async_trait;
use bytes::Bytes;
use std::io;

pub mod memory;
#[cfg(feature = "quic")]
pub mod quic;
pub mod tcp;

/// A bi-directional, asynchronous byte stream (e.g., a QUIC stream or TCP connection).
#[async_trait]
pub trait Stream: Send + Sync + 'static {
    /// Reads exactly `buf.len()` bytes into `buf`.
    async fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()>;

    /// Reads a chunk of bytes.
    async fn read_chunk(&mut self, max_len: usize) -> io::Result<Option<Bytes>>;

    /// Writes the entire `buf`.
    async fn write_all(&mut self, buf: &[u8]) -> io::Result<()>;
}

/// A multiplexed connection between two nodes.
#[async_trait]
pub trait Connection: Send + Sync + 'static {
    /// Open a new unidirectional or bidirectional stream.
    async fn open_stream(&self) -> io::Result<Box<dyn Stream>>;

    /// Accept an incoming stream from the peer.
    async fn accept_stream(&self) -> io::Result<Box<dyn Stream>>;
}

/// A transport layer that can listen for incoming connections or connect to peers.
#[async_trait]
pub trait Transport: Send + Sync + 'static {
    /// Connects to a remote peer.
    async fn connect(&self, addr: &str) -> io::Result<Box<dyn Connection>>;

    /// Accepts a new connection from a peer.
    async fn accept(&self) -> io::Result<Box<dyn Connection>>;
}
