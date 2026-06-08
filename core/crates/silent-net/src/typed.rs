//! Typed send/receive helpers that enforce object tags, versions, and
//! parameter binding using `silent-io` traits.
//!
//! These wrappers around [`super::frame::send_frame`] /
//! [`super::frame::recv_frame`] ensure that:
//! - The object type tag ([`ObjectKind::OBJECT_TYPE`]) is written into the
//!   frame header and verified on receive.
//! - The serialized object version ([`ObjectKind::SERIALIZED_VERSION`]) is
//!   recorded in the frame header.
//! - Canonical encoding ([`CanonicalEncode`] / [`DecodeWithParams`]) is the
//!   only data-path codec.
//! - `params_id` binding is enforced when the parameter type implements
//!   [`HasParamsId`].

use crate::frame::{FrameError, MessageKind, NetworkFrame, recv_frame, send_frame};
use crate::transport::Stream;
use silent_io::error::IoError;
use silent_io::traits::{
    CanonicalDecode, CanonicalEncode, DecodeWithParams, EncodeContext, HasParamsId, ObjectKind,
};

/// Send a canonically-encoded object over a transport stream.
///
/// The object is encoded via [`CanonicalEncode::encode_to_vec`] and wrapped
/// in a [`NetworkFrame`] with the supplied routing metadata and context.
pub async fn send_object<T: CanonicalEncode + ObjectKind>(
    stream: &mut Box<dyn Stream>,
    object: &T,
    session_id: u64,
    task_id: u64,
    gate_id: u64,
    ctx: &EncodeContext,
) -> Result<(), FrameError> {
    let payload = object.encode_to_vec()?;

    let frame = NetworkFrame {
        session_id,
        task_id,
        gate_id,
        message_kind: MessageKind::Data,
        object_type: T::OBJECT_TYPE,
        object_version: T::SERIALIZED_VERSION,
        scheme_id: ctx.scheme_id,
        params_id: ctx.params_id,
        flags: ctx.flags,
        payload,
    };

    send_frame(stream, &frame).await
}

/// Receive an object using [`CanonicalDecode`] (parameter-free decode).
///
/// Only types that implement `CanonicalDecode` can use this path.  Crypto
/// types that depend on modulus/degree require [`recv_object_with_params`].
pub async fn recv_object<T: CanonicalDecode + ObjectKind>(
    stream: &mut Box<dyn Stream>,
    max_payload: u32,
) -> Result<T, FrameError> {
    let frame = recv_frame(stream, max_payload).await?;

    if frame.object_type != T::OBJECT_TYPE {
        return Err(FrameError::from(IoError::ObjectTypeMismatch {
            expected: T::OBJECT_TYPE,
            actual: frame.object_type,
        }));
    }

    frame.validate_object_version(T::SERIALIZED_VERSION)?;

    T::decode_from(&mut std::io::Cursor::new(&frame.payload)).map_err(FrameError::from)
}

/// Receive an object using [`DecodeWithParams`].
///
/// Validates:
/// - Object type matches `T::OBJECT_TYPE`
/// - Serialized version ≤ `T::SERIALIZED_VERSION`
pub async fn recv_object_with_params<T, P>(
    stream: &mut Box<dyn Stream>,
    max_payload: u32,
    params: &P,
) -> Result<T, FrameError>
where
    T: DecodeWithParams<P> + ObjectKind,
{
    let frame = recv_frame(stream, max_payload).await?;

    if frame.object_type != T::OBJECT_TYPE {
        return Err(FrameError::from(IoError::ObjectTypeMismatch {
            expected: T::OBJECT_TYPE,
            actual: frame.object_type,
        }));
    }

    frame.validate_object_version(T::SERIALIZED_VERSION)?;

    T::decode_with_params(params, &mut std::io::Cursor::new(&frame.payload))
        .map_err(FrameError::from)
}

/// Receive an object using [`DecodeWithParams`] **and** check that the
/// frame's `params_id` matches the given parameter set's [`HasParamsId`].
///
/// This is the preferred receive path for all modulus-dependent crypto
/// types because it prevents cross-parameter-set misinterpretation.
pub async fn recv_object_with_params_checked<T, P>(
    stream: &mut Box<dyn Stream>,
    max_payload: u32,
    params: &P,
) -> Result<T, FrameError>
where
    T: DecodeWithParams<P> + ObjectKind,
    P: HasParamsId,
{
    let frame = recv_frame(stream, max_payload).await?;

    if frame.object_type != T::OBJECT_TYPE {
        return Err(FrameError::from(IoError::ObjectTypeMismatch {
            expected: T::OBJECT_TYPE,
            actual: frame.object_type,
        }));
    }

    frame.validate_object_version(T::SERIALIZED_VERSION)?;
    frame.validate_params_id(params.params_id())?;

    T::decode_with_params(params, &mut std::io::Cursor::new(&frame.payload))
        .map_err(FrameError::from)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::DEFAULT_MAX_FRAME_PAYLOAD;
    use crate::transport::Connection;
    use crate::transport::memory::MemoryConnection;
    use silent_io::ParamsId;
    use silent_io::object_type::ObjectType;
    use silent_io::primitives;
    use std::io::{Read, Write};

    // ── Mock types for testing ────────────────────────────────────────────

    // A simple CanonicalDecode-able type (no parameters needed).
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestValue(u32);

    impl CanonicalEncode for TestValue {
        fn encoded_len(&self) -> usize {
            4
        }
        fn encode_to<W: Write>(&self, writer: &mut W) -> Result<usize, IoError> {
            primitives::write_u32(writer, self.0)
        }
    }

    impl CanonicalDecode for TestValue {
        fn decode_from<R: Read>(reader: &mut R) -> Result<Self, IoError> {
            primitives::read_u32(reader).map(TestValue)
        }
    }

    impl ObjectKind for TestValue {
        const OBJECT_TYPE: ObjectType = ObjectType::POLY;
        const SERIALIZED_VERSION: u16 = 1;
    }

    // A parameter-set with HasParamsId support.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestParams {
        id: ParamsId,
        multiplier: u32,
    }

    impl HasParamsId for TestParams {
        fn params_id(&self) -> ParamsId {
            self.id
        }
    }

    // A type that requires TestParams for decoding.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ParamValue(u32);

    impl CanonicalEncode for ParamValue {
        fn encoded_len(&self) -> usize {
            4
        }
        fn encode_to<W: Write>(&self, writer: &mut W) -> Result<usize, IoError> {
            primitives::write_u32(writer, self.0)
        }
    }

    impl DecodeWithParams<TestParams> for ParamValue {
        fn decode_with_params<R: Read>(
            params: &TestParams,
            reader: &mut R,
        ) -> Result<Self, IoError> {
            let raw = primitives::read_u32(reader)?;
            Ok(ParamValue(raw * params.multiplier))
        }
    }

    impl ObjectKind for ParamValue {
        const OBJECT_TYPE: ObjectType = ObjectType::RLWE_CIPHERTEXT;
        const SERIALIZED_VERSION: u16 = 2;
    }

    // ── Roundtrip tests ──────────────────────────────────────────────────

    #[tokio::test]
    async fn send_recv_object_roundtrip() {
        let (alice, bob) = MemoryConnection::pair();

        let value = TestValue(0xdead_beef);
        let ctx = EncodeContext::default();

        let value_clone = value.clone();
        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_object(&mut stream, &value_clone, 1, 2, 3, &ctx)
                .await
                .unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_object::<TestValue>(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .unwrap()
        });

        alice_handle.await.unwrap();
        let received = bob_handle.await.unwrap();

        assert_eq!(received, value);
    }

    #[tokio::test]
    async fn send_recv_object_with_params_roundtrip() {
        let (alice, bob) = MemoryConnection::pair();

        let value = ParamValue(42);
        let params = TestParams {
            id: ParamsId::ZERO,
            multiplier: 1, // identity: decoded value = raw * 1 = raw
        };
        let ctx = EncodeContext::default();

        let value_clone = value.clone();
        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_object(&mut stream, &value_clone, 1, 2, 3, &ctx)
                .await
                .unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_object_with_params::<ParamValue, TestParams>(
                &mut stream,
                DEFAULT_MAX_FRAME_PAYLOAD,
                &params,
            )
            .await
            .unwrap()
        });

        alice_handle.await.unwrap();
        let received = bob_handle.await.unwrap();

        assert_eq!(received, value);
    }

    #[tokio::test]
    async fn recv_object_with_params_checked_passes() {
        let (alice, bob) = MemoryConnection::pair();

        let params = TestParams {
            id: ParamsId([0xabu8; 32]),
            multiplier: 1,
        };
        let ctx = EncodeContext {
            params_id: params.params_id(),
            ..Default::default()
        };

        let value = ParamValue(7);
        let value_clone = value.clone();

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_object(&mut stream, &value_clone, 1, 2, 3, &ctx)
                .await
                .unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_object_with_params_checked::<ParamValue, TestParams>(
                &mut stream,
                DEFAULT_MAX_FRAME_PAYLOAD,
                &params,
            )
            .await
            .unwrap()
        });

        alice_handle.await.unwrap();
        let received = bob_handle.await.unwrap();
        assert_eq!(received, value);
    }

    #[tokio::test]
    async fn recv_object_with_params_checked_rejects_mismatch() {
        let (alice, bob) = MemoryConnection::pair();

        // Sender uses a different params_id than what receiver expects.
        let sender_params_id = ParamsId([0xabu8; 32]);
        let receiver_params = TestParams {
            id: ParamsId([0xffu8; 32]), // different
            multiplier: 1,
        };
        let ctx = EncodeContext {
            params_id: sender_params_id,
            ..Default::default()
        };

        let value = ParamValue(7);

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_object(&mut stream, &value, 1, 2, 3, &ctx)
                .await
                .unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_object_with_params_checked::<ParamValue, TestParams>(
                &mut stream,
                DEFAULT_MAX_FRAME_PAYLOAD,
                &receiver_params,
            )
            .await
        });

        alice_handle.await.unwrap();
        let err = bob_handle.await.unwrap().unwrap_err();

        assert!(matches!(err, FrameError::ParamsIdMismatch { .. }));
    }

    #[tokio::test]
    async fn recv_object_rejects_type_mismatch() {
        let (alice, bob) = MemoryConnection::pair();

        // Send a ParamValue (RLWE_CIPHERTEXT) but expect TestValue (POLY).
        let value = ParamValue(42);
        let ctx = EncodeContext::default();
        let value_clone = value.clone();

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_object(&mut stream, &value_clone, 1, 2, 3, &ctx)
                .await
                .unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_object::<TestValue>(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD).await
        });

        alice_handle.await.unwrap();
        let err = bob_handle.await.unwrap().unwrap_err();

        assert!(matches!(
            err,
            FrameError::Io(IoError::ObjectTypeMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn recv_object_rejects_future_version() {
        let (alice, bob) = MemoryConnection::pair();

        // Send a ParamValue with SERIALIZED_VERSION=2, but expect TestValue with version=1
        // (same object_type, different type). Already tested above with type mismatch.
        // Here we test version rejection directly: send with version 99.

        let ctx = EncodeContext::default();
        let value = TestValue(1);
        let value_clone = value.clone();

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            // Manually create a frame with an inflated version.
            let frame = NetworkFrame {
                session_id: 1,
                task_id: 2,
                gate_id: 3,
                message_kind: MessageKind::Data,
                object_type: TestValue::OBJECT_TYPE,
                object_version: 99, // future version
                scheme_id: ctx.scheme_id,
                params_id: ctx.params_id,
                flags: ctx.flags,
                payload: value_clone.encode_to_vec().unwrap(),
            };
            send_frame(&mut stream, &frame).await.unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_object::<TestValue>(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD).await
        });

        alice_handle.await.unwrap();
        let err = bob_handle.await.unwrap().unwrap_err();

        assert!(matches!(err, FrameError::ObjectVersionTooNew { .. }));
    }

    #[tokio::test]
    async fn send_object_preserves_envelope_in_header() {
        let (alice, bob) = MemoryConnection::pair();

        let value = TestValue(42);
        let ctx = EncodeContext {
            scheme_id: 5,
            params_id: ParamsId([0xccu8; 32]),
            flags: 0x03,
        };

        let value_clone = value.clone();
        let ctx_check = ctx.clone();
        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_object(&mut stream, &value_clone, 10, 20, 30, &ctx)
                .await
                .unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_frame(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .unwrap()
        });

        alice_handle.await.unwrap();
        let frame = bob_handle.await.unwrap();

        assert_eq!(frame.session_id, 10);
        assert_eq!(frame.task_id, 20);
        assert_eq!(frame.gate_id, 30);
        assert_eq!(frame.message_kind, MessageKind::Data);
        assert_eq!(frame.object_type, TestValue::OBJECT_TYPE);
        assert_eq!(frame.object_version, TestValue::SERIALIZED_VERSION);
        assert_eq!(frame.scheme_id, ctx_check.scheme_id);
        assert_eq!(frame.params_id, ctx_check.params_id);
        assert_eq!(frame.flags, ctx_check.flags);
    }
}
