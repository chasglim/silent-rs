use super::{Connection, Stream, Transport};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use std::io::{self, Error, ErrorKind};
use tokio::sync::mpsc;

/// An in-memory stream using tokio MPSC channels.
pub struct MemoryStream {
    rx: mpsc::Receiver<Bytes>,
    tx: mpsc::Sender<Bytes>,
    buffer: BytesMut,
}

impl MemoryStream {
    pub fn new(rx: mpsc::Receiver<Bytes>, tx: mpsc::Sender<Bytes>) -> Self {
        Self {
            rx,
            tx,
            buffer: BytesMut::new(),
        }
    }
}

#[async_trait]
impl Stream for MemoryStream {
    async fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        let mut needed = buf.len();
        let mut offset = 0;

        while needed > 0 {
            if self.buffer.is_empty() {
                match self.rx.recv().await {
                    Some(data) => self.buffer.extend_from_slice(&data),
                    None => return Err(Error::new(ErrorKind::UnexpectedEof, "Channel closed")),
                }
            }

            let take = std::cmp::min(needed, self.buffer.len());
            let chunk = self.buffer.split_to(take);
            buf[offset..offset + take].copy_from_slice(&chunk);
            offset += take;
            needed -= take;
        }
        Ok(())
    }

    async fn read_chunk(&mut self, max_len: usize) -> io::Result<Option<Bytes>> {
        if !self.buffer.is_empty() {
            let take = std::cmp::min(max_len, self.buffer.len());
            return Ok(Some(self.buffer.split_to(take).freeze()));
        }
        match self.rx.recv().await {
            Some(mut data) => {
                if data.len() > max_len {
                    let rest = data.split_off(max_len);
                    self.buffer.extend_from_slice(&rest);
                }
                Ok(Some(data))
            }
            None => Ok(None),
        }
    }

    async fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        let bytes = Bytes::copy_from_slice(buf);
        self.tx
            .send(bytes)
            .await
            .map_err(|_| Error::new(ErrorKind::BrokenPipe, "Failed to send on memory channel"))
    }
}

/// An in-memory connection simulating QUIC's ability to open multiple streams.
type BiStreamPair = (mpsc::Sender<Bytes>, mpsc::Receiver<Bytes>);

/// An in-memory connection simulating QUIC's ability to open multiple streams.
pub struct MemoryConnection {
    peer_tx: mpsc::Sender<BiStreamPair>,
    incoming_rx: tokio::sync::Mutex<mpsc::Receiver<BiStreamPair>>,
}

impl MemoryConnection {
    pub fn pair() -> (Self, Self) {
        let (tx1, rx1) = mpsc::channel(32);
        let (tx2, rx2) = mpsc::channel(32);
        (
            Self {
                peer_tx: tx1,
                incoming_rx: tokio::sync::Mutex::new(rx2),
            },
            Self {
                peer_tx: tx2,
                incoming_rx: tokio::sync::Mutex::new(rx1),
            },
        )
    }
}

#[async_trait]
impl Connection for MemoryConnection {
    async fn open_stream(&self) -> io::Result<Box<dyn Stream>> {
        let (tx_to_peer, rx_from_me) = mpsc::channel(32);
        let (tx_to_me, rx_from_peer) = mpsc::channel(32);

        self.peer_tx
            .send((tx_to_me, rx_from_me))
            .await
            .map_err(|_| Error::new(ErrorKind::BrokenPipe, "Connection closed"))?;

        Ok(Box::new(MemoryStream::new(rx_from_peer, tx_to_peer)))
    }

    async fn accept_stream(&self) -> io::Result<Box<dyn Stream>> {
        let mut rx = self.incoming_rx.lock().await;
        match rx.recv().await {
            Some((tx_to_peer, rx_from_peer)) => {
                Ok(Box::new(MemoryStream::new(rx_from_peer, tx_to_peer)))
            }
            None => Err(Error::new(ErrorKind::UnexpectedEof, "Connection closed")),
        }
    }
}

// ── MemoryTransport ─────────────────────────────────────────────────────────

/// A paired memory transport that implements the [`Transport`] trait.
///
/// `connect("")` creates a new in-memory connection and sends the paired
/// endpoint to the peer transport; `accept()` receives it.
pub struct MemoryTransport {
    conn_tx: mpsc::Sender<MemoryConnection>,
    conn_rx: tokio::sync::Mutex<mpsc::Receiver<MemoryConnection>>,
}

impl MemoryTransport {
    /// Create a pair of linked transports.
    ///
    /// Connections opened via `transport_a.connect("")` arrive on
    /// `transport_b.accept()`, and vice-versa.
    pub fn pair() -> (Self, Self) {
        let (tx_a, rx_a) = mpsc::channel(32);
        let (tx_b, rx_b) = mpsc::channel(32);
        (
            Self {
                conn_tx: tx_b,
                conn_rx: tokio::sync::Mutex::new(rx_a),
            },
            Self {
                conn_tx: tx_a,
                conn_rx: tokio::sync::Mutex::new(rx_b),
            },
        )
    }
}

#[async_trait]
impl Transport for MemoryTransport {
    async fn connect(&self, _addr: &str) -> io::Result<Box<dyn Connection>> {
        let (client_conn, server_conn) = MemoryConnection::pair();
        self.conn_tx
            .send(server_conn)
            .await
            .map_err(|_| Error::new(ErrorKind::BrokenPipe, "peer transport closed"))?;
        Ok(Box::new(client_conn))
    }

    async fn accept(&self) -> io::Result<Box<dyn Connection>> {
        let mut rx = self.conn_rx.lock().await;
        match rx.recv().await {
            Some(conn) => Ok(Box::new(conn)),
            None => Err(Error::new(ErrorKind::UnexpectedEof, "transport closed")),
        }
    }
}
