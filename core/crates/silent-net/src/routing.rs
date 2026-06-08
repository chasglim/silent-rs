//! Frame demuxer keyed by `(session_id, task_id, gate_id, message_kind)`.
//!
//! Each registered route gets a bounded `mpsc` channel.  Incoming frames are
//! dispatched to the matching route; frames for unregistered routes are dropped.
//!
//! # Backpressure
//!
//! [`Router::try_dispatch`] drops frames when a route's channel is full.
//! Use [`Router::dispatch_blocking`] to wait until space is available.

use crate::frame::{MessageKind, NetworkFrame};
use std::collections::HashMap;
use tokio::sync::mpsc;

/// Uniquely identifies a route within the router.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RouteKey {
    pub session_id: u64,
    pub task_id: u64,
    pub gate_id: u64,
    pub message_kind: MessageKind,
}

impl RouteKey {
    pub const fn new(
        session_id: u64,
        task_id: u64,
        gate_id: u64,
        message_kind: MessageKind,
    ) -> Self {
        Self {
            session_id,
            task_id,
            gate_id,
            message_kind,
        }
    }
}

/// Receiver endpoint for a registered route.
pub type RouteReceiver = mpsc::Receiver<NetworkFrame>;

/// A bounded frame demuxer that routes incoming frames to the correct
/// destination based on the frame's routing metadata.
pub struct Router {
    routes: HashMap<RouteKey, mpsc::Sender<NetworkFrame>>,
    capacity: usize,
}

impl Router {
    /// Create a new router where every route's channel has the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            routes: HashMap::new(),
            capacity,
        }
    }

    /// Register a route and return a receiver for frames dispatched to it.
    ///
    /// If the key is already registered, the old channel is closed and a new
    /// one is created.
    pub fn register(&mut self, key: RouteKey) -> RouteReceiver {
        let (tx, rx) = mpsc::channel(self.capacity);
        self.routes.insert(key, tx);
        rx
    }

    /// Remove a route, closing its channel.
    pub fn deregister(&mut self, key: RouteKey) -> bool {
        self.routes.remove(&key).is_some()
    }

    /// Check whether a route is registered.
    pub fn contains(&self, key: RouteKey) -> bool {
        self.routes.contains_key(&key)
    }

    /// Number of currently registered routes.
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    /// Try to dispatch a frame, returning immediately.
    ///
    /// Returns `false` if no route is registered or the channel is full.
    pub fn try_dispatch(&mut self, frame: NetworkFrame) -> bool {
        let key = route_key_from_frame(&frame);
        match self.routes.get(&key) {
            Some(tx) => tx.try_send(frame).is_ok(),
            None => false,
        }
    }

    /// Dispatch a frame, waiting for channel capacity if needed.
    ///
    /// Returns `Err(frame)` when no route is registered.
    pub async fn dispatch_blocking(&mut self, frame: NetworkFrame) -> Result<(), NetworkFrame> {
        let key = route_key_from_frame(&frame);
        match self.routes.get(&key) {
            Some(tx) => {
                tx.send(frame).await.map_err(|e| e.0)?;
                Ok(())
            }
            None => Err(frame),
        }
    }

    /// Dispatch a frame, waiting for capacity but dropping if no route exists.
    pub async fn dispatch_or_drop(&mut self, frame: NetworkFrame) -> bool {
        let key = route_key_from_frame(&frame);
        match self.routes.get(&key) {
            Some(tx) => tx.send(frame).await.is_ok(),
            None => false,
        }
    }

    /// Remove all routes.
    pub fn clear(&mut self) {
        self.routes.clear();
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new(64)
    }
}

/// Extract a [`RouteKey`] from a frame's routing metadata.
pub fn route_key_from_frame(frame: &NetworkFrame) -> RouteKey {
    RouteKey {
        session_id: frame.session_id,
        task_id: frame.task_id,
        gate_id: frame.gate_id,
        message_kind: frame.message_kind,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use silent_io::{ObjectType, ParamsId};

    fn make_frame(
        session: u64,
        task: u64,
        gate: u64,
        kind: MessageKind,
        payload: &str,
    ) -> NetworkFrame {
        NetworkFrame {
            session_id: session,
            task_id: task,
            gate_id: gate,
            message_kind: kind,
            object_type: ObjectType::RLWE_CIPHERTEXT,
            object_version: 1,
            scheme_id: 0,
            params_id: ParamsId::ZERO,
            flags: 0,
            payload: payload.as_bytes().to_vec(),
        }
    }

    #[tokio::test]
    async fn register_and_dispatch_single_route() {
        let mut router = Router::new(8);
        let key = RouteKey::new(0, 0, 0, MessageKind::Data);
        let mut rx = router.register(key);

        assert!(router.try_dispatch(make_frame(0, 0, 0, MessageKind::Data, "hello")));

        assert_eq!(rx.recv().await.unwrap().payload, b"hello");
    }

    #[tokio::test]
    async fn dispatch_to_multiple_routes() {
        let mut router = Router::new(8);

        let key_a = RouteKey::new(0, 0, 1, MessageKind::Data);
        let key_b = RouteKey::new(0, 0, 2, MessageKind::Data);
        let key_c = RouteKey::new(1, 0, 0, MessageKind::Control);

        let mut rx_a = router.register(key_a);
        let mut rx_b = router.register(key_b);
        let mut rx_c = router.register(key_c);

        assert_eq!(router.route_count(), 3);

        assert!(router.try_dispatch(make_frame(0, 0, 1, MessageKind::Data, "A")));
        assert!(router.try_dispatch(make_frame(1, 0, 0, MessageKind::Control, "C")));
        assert!(router.try_dispatch(make_frame(0, 0, 2, MessageKind::Data, "B")));

        assert_eq!(rx_a.recv().await.unwrap().payload, b"A");
        assert_eq!(rx_c.recv().await.unwrap().payload, b"C");
        assert_eq!(rx_b.recv().await.unwrap().payload, b"B");
    }

    #[tokio::test]
    async fn out_of_order_dispatch() {
        let mut router = Router::new(8);

        let key_x = RouteKey::new(1, 1, 1, MessageKind::Data);
        let key_y = RouteKey::new(2, 2, 2, MessageKind::Data);

        let mut rx_x = router.register(key_x);
        let mut rx_y = router.register(key_y);

        for i in 0..5 {
            router.try_dispatch(make_frame(1, 1, 1, MessageKind::Data, &format!("x{i}")));
            router.try_dispatch(make_frame(2, 2, 2, MessageKind::Data, &format!("y{i}")));
        }

        for i in 0..5 {
            assert_eq!(
                rx_x.recv().await.unwrap().payload,
                format!("x{i}").as_bytes()
            );
            assert_eq!(
                rx_y.recv().await.unwrap().payload,
                format!("y{i}").as_bytes()
            );
        }
    }

    #[test]
    fn dispatch_returns_false_for_unregistered_route() {
        let mut router = Router::new(8);
        assert!(!router.try_dispatch(make_frame(99, 99, 99, MessageKind::Data, "orphan")));
    }

    #[tokio::test]
    async fn dispatch_blocking_returns_err_for_unregistered_route() {
        let mut router = Router::new(8);
        let result = router
            .dispatch_blocking(make_frame(99, 99, 99, MessageKind::Data, "orphan"))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().payload, b"orphan");
    }

    #[test]
    fn backpressure_drops_when_channel_full() {
        let mut router = Router::new(2);

        let key = RouteKey::new(0, 0, 0, MessageKind::Data);
        let _rx = router.register(key);

        assert!(router.try_dispatch(make_frame(0, 0, 0, MessageKind::Data, "1")));
        assert!(router.try_dispatch(make_frame(0, 0, 0, MessageKind::Data, "2")));
        assert!(!router.try_dispatch(make_frame(0, 0, 0, MessageKind::Data, "3")));
    }

    #[tokio::test]
    async fn dispatch_blocking_waits_for_capacity() {
        let mut router = Router::new(2);

        let key = RouteKey::new(0, 0, 0, MessageKind::Data);
        let mut rx = router.register(key);

        router.try_dispatch(make_frame(0, 0, 0, MessageKind::Data, "a"));
        router.try_dispatch(make_frame(0, 0, 0, MessageKind::Data, "b"));

        let mut router_handle = router;
        let handle = tokio::spawn(async move {
            router_handle
                .dispatch_blocking(make_frame(0, 0, 0, MessageKind::Data, "c"))
                .await
        });

        // Drain one item to unblock the sender.
        assert_eq!(rx.recv().await.unwrap().payload, b"a");

        let result = handle.await.unwrap();
        assert!(result.is_ok());

        assert_eq!(rx.recv().await.unwrap().payload, b"b");
        assert_eq!(rx.recv().await.unwrap().payload, b"c");
    }

    #[tokio::test]
    async fn deregister_closes_channel() {
        let mut router = Router::new(8);

        let key = RouteKey::new(0, 0, 0, MessageKind::Data);
        let mut rx = router.register(key);

        router.try_dispatch(make_frame(0, 0, 0, MessageKind::Data, "x"));

        assert!(router.deregister(key));
        assert!(!router.contains(key));

        assert_eq!(rx.recv().await.unwrap().payload, b"x");
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn message_kind_routes_separately() {
        let mut router = Router::new(8);

        let data_key = RouteKey::new(0, 0, 0, MessageKind::Data);
        let control_key = RouteKey::new(0, 0, 0, MessageKind::Control);

        let mut rx_data = router.register(data_key);
        let mut rx_ctrl = router.register(control_key);

        router.try_dispatch(make_frame(0, 0, 0, MessageKind::Data, "d"));
        router.try_dispatch(make_frame(0, 0, 0, MessageKind::Control, "c"));

        assert_eq!(rx_data.recv().await.unwrap().payload, b"d");
        assert_eq!(rx_ctrl.recv().await.unwrap().payload, b"c");
    }

    #[tokio::test]
    async fn clear_removes_all_routes() {
        let mut router = Router::new(8);

        let key = RouteKey::new(0, 0, 0, MessageKind::Data);
        let mut rx = router.register(key);

        router.try_dispatch(make_frame(0, 0, 0, MessageKind::Data, "before"));

        router.clear();
        assert_eq!(router.route_count(), 0);

        assert_eq!(rx.recv().await.unwrap().payload, b"before");
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn re_registration_replaces_old_channel() {
        let mut router = Router::new(8);

        let key = RouteKey::new(0, 0, 0, MessageKind::Data);
        let mut rx1 = router.register(key);

        router.try_dispatch(make_frame(0, 0, 0, MessageKind::Data, "old"));

        let mut rx2 = router.register(key);

        assert_eq!(rx1.recv().await.unwrap().payload, b"old");
        assert!(rx1.recv().await.is_none());

        router.try_dispatch(make_frame(0, 0, 0, MessageKind::Data, "new"));
        assert_eq!(rx2.recv().await.unwrap().payload, b"new");
    }

    #[tokio::test]
    async fn dispatch_or_drop_waits_for_capacity() {
        let mut router = Router::new(8);

        let key = RouteKey::new(0, 0, 0, MessageKind::Data);
        let _rx = router.register(key);

        assert!(
            router
                .dispatch_or_drop(make_frame(0, 0, 0, MessageKind::Data, "ok"))
                .await
        );
        assert!(
            !router
                .dispatch_or_drop(make_frame(99, 99, 99, MessageKind::Data, "nope"))
                .await
        );
    }

    #[test]
    fn route_key_equality() {
        let a = RouteKey::new(1, 2, 3, MessageKind::Data);
        let b = RouteKey::new(1, 2, 3, MessageKind::Data);
        let c = RouteKey::new(1, 2, 3, MessageKind::Control);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
