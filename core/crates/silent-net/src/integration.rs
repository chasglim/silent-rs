//! Integration tests with real cryptographic types.
//!
//! Enabled via `--features crypto-tests`:
//!
//! ```bash
//! cargo test -p silent-net --features crypto-tests
//! ```
//!
//! Tests covered:
//! - `NativePoly` (silent-ring): encode/decode via `send_object` / `recv_object_with_params`
//! - `LweCiphertext` (silent-rlwe): encode/decode with parameter validation
//! - `ShortintCiphertext` (silent-fhe): encode/decode with `recv_object_with_params_checked`
//! - Non-canonical parameter rejection over the network path
//! - Coefficient-out-of-range rejection over the network path

#[cfg(test)]
mod tests {
    use crate::frame::DEFAULT_MAX_FRAME_PAYLOAD;
    use crate::transport::Connection;
    use crate::transport::memory::MemoryConnection;
    use crate::typed::{recv_object_with_params, recv_object_with_params_checked, send_object};
    use silent_fhe::schemes::shortint::{Degree, NoiseLevel, ShortintCiphertext};
    use silent_io::traits::{
        CanonicalEncode, DecodeWithParams, EncodeContext, HasParamsId, ObjectKind,
    };
    use silent_params::{CiphertextModulusLog, LweDimension};
    use silent_ring::NativePoly;
    use silent_rlwe::lwe::LweCiphertext;

    // ── NativePoly roundtrip ───────────────────────────────────────────────────

    #[tokio::test]
    async fn send_recv_native_poly_roundtrip() {
        let (alice, bob) = MemoryConnection::pair();

        let degree = 8usize;
        let log_modulus: u8 = 32;
        let poly = NativePoly::zeros(degree, log_modulus);
        let ctx = EncodeContext::default();

        let poly_encoded = poly.encode_to_vec().unwrap();
        let poly_clone = poly.clone();

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_object(&mut stream, &poly_clone, 1, 2, 3, &ctx)
                .await
                .unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_object_with_params::<NativePoly, _>(
                &mut stream,
                DEFAULT_MAX_FRAME_PAYLOAD,
                &(degree, log_modulus),
            )
            .await
            .unwrap()
        });

        alice_handle.await.unwrap();
        let decoded = bob_handle.await.unwrap();

        assert_eq!(decoded.encode_to_vec().unwrap(), poly_encoded);
    }

    // ── LweCiphertext roundtrip ───────────────────────────────────────────────

    #[tokio::test]
    async fn send_recv_lwe_ciphertext_roundtrip() {
        let (alice, bob) = MemoryConnection::pair();

        let ctx = EncodeContext::default();
        let dim = LweDimension(4);
        let log_mod = CiphertextModulusLog(32);
        let ct = LweCiphertext::zeros(dim, log_mod);

        let ct_encoded = ct.encode_to_vec().unwrap();
        let ct_clone = ct.clone();

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_object(&mut stream, &ct_clone, 1, 2, 3, &ctx)
                .await
                .unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_object_with_params::<LweCiphertext, _>(
                &mut stream,
                DEFAULT_MAX_FRAME_PAYLOAD,
                &(dim, log_mod),
            )
            .await
            .unwrap()
        });

        alice_handle.await.unwrap();
        let decoded = bob_handle.await.unwrap();

        assert_eq!(decoded.encode_to_vec().unwrap(), ct_encoded);
    }

    // ── ShortintCiphertext with params_checked ─────────────────────────────────

    #[tokio::test]
    async fn send_recv_shortint_ciphertext_with_params_checked() {
        let (alice, bob) = MemoryConnection::pair();

        // Use toy TFHE parameters for a simple test.
        let params = silent_fhe::schemes::tfhe::params::TfheParameters::new(
            silent_params::presets::toy::toy_tfhe_n512(),
        )
        .expect("toy params should be valid");

        let dim = params.lwe_dimension();
        let log_mod = params.ciphertext_modulus_log();
        let mm = params.message_modulus().0;
        let cm = params.carry_modulus().0;

        let inner = LweCiphertext::zeros(dim, log_mod);
        let ct = ShortintCiphertext::from_parts(inner, Degree(4), NoiseLevel(0), mm, cm);

        let ct_encoded = ct.encode_to_vec().unwrap();
        let ct_clone = ct.clone();
        let params_clone = params.clone();
        let ctx = EncodeContext {
            params_id: params.params_id(),
            ..Default::default()
        };

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_object(&mut stream, &ct_clone, 1, 2, 3, &ctx)
                .await
                .unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_object_with_params_checked::<ShortintCiphertext, _>(
                &mut stream,
                DEFAULT_MAX_FRAME_PAYLOAD,
                &params_clone,
            )
            .await
            .unwrap()
        });

        alice_handle.await.unwrap();
        let decoded = bob_handle.await.unwrap();

        assert_eq!(decoded.encode_to_vec().unwrap(), ct_encoded);
    }

    // ── Non-canonical parameter rejection ─────────────────────────────────────

    #[tokio::test]
    async fn native_poly_rejects_wrong_degree_over_network() {
        let (alice, bob) = MemoryConnection::pair();

        let poly = NativePoly::zeros(8, 32);
        let ctx = EncodeContext::default();

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_object(&mut stream, &poly, 1, 2, 3, &ctx)
                .await
                .unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_object_with_params::<NativePoly, _>(
                &mut stream,
                DEFAULT_MAX_FRAME_PAYLOAD,
                &(16usize, 32u8),
            )
            .await
        });

        alice_handle.await.unwrap();
        assert!(
            bob_handle.await.unwrap().is_err(),
            "should reject wrong degree"
        );
    }

    #[tokio::test]
    async fn lwe_ciphertext_rejects_wrong_dimension_over_network() {
        let (alice, bob) = MemoryConnection::pair();

        let dim = LweDimension(4);
        let log_mod = CiphertextModulusLog(32);
        let ct = LweCiphertext::zeros(dim, log_mod);
        let ctx = EncodeContext::default();

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_object(&mut stream, &ct, 1, 2, 3, &ctx).await.unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_object_with_params::<LweCiphertext, _>(
                &mut stream,
                DEFAULT_MAX_FRAME_PAYLOAD,
                &(LweDimension(8), log_mod),
            )
            .await
        });

        alice_handle.await.unwrap();
        assert!(
            bob_handle.await.unwrap().is_err(),
            "should reject wrong LWE dimension"
        );
    }

    #[tokio::test]
    async fn native_poly_rejects_wrong_log_modulus_over_network() {
        let (alice, bob) = MemoryConnection::pair();

        let poly = NativePoly::zeros(8, 32);
        let ctx = EncodeContext::default();

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            send_object(&mut stream, &poly, 1, 2, 3, &ctx)
                .await
                .unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            recv_object_with_params::<NativePoly, _>(
                &mut stream,
                DEFAULT_MAX_FRAME_PAYLOAD,
                &(8usize, 64u8),
            )
            .await
        });

        alice_handle.await.unwrap();
        assert!(
            bob_handle.await.unwrap().is_err(),
            "should reject wrong log_modulus"
        );
    }

    // ── Coefficient out of range over network path ────────────────────────────

    #[tokio::test]
    async fn native_poly_rejects_overflow_coefficient_over_network() {
        let (alice, bob) = MemoryConnection::pair();

        // Manually craft a NetworkFrame whose payload is a valid-looking
        // NativePoly header but with a coefficient ≥ 2^32 (out of range for
        // a 32-bit modulus).

        // Valid NativePoly payload layout:
        //   u8  log_modulus = 32
        //   u32 degree = 1
        //   u64 coeff_0 = < 2^32  (valid)
        // We replace coeff_0 with 0x1_0000_0000 (≥ 2^32).

        let mut payload = Vec::new();
        // log_modulus u8
        payload.push(32u8);
        // degree u32 LE
        payload.extend_from_slice(&1u32.to_le_bytes());
        // coefficient u64 LE — overflow value
        payload.extend_from_slice(&0x1_0000_0000u64.to_le_bytes());

        let frame = crate::frame::NetworkFrame {
            session_id: 1,
            task_id: 2,
            gate_id: 3,
            message_kind: crate::frame::MessageKind::Data,
            object_type: silent_ring::NativePoly::OBJECT_TYPE,
            object_version: silent_ring::NativePoly::SERIALIZED_VERSION,
            scheme_id: 0,
            params_id: silent_io::ParamsId::ZERO,
            flags: 0,
            payload,
        };

        let frame_clone = frame.clone();

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            crate::frame::send_frame(&mut stream, &frame_clone)
                .await
                .unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            let raw = crate::frame::recv_frame(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .unwrap();
            NativePoly::decode_with_params(&(1usize, 32u8), &mut std::io::Cursor::new(&raw.payload))
        });

        alice_handle.await.unwrap();
        let err = bob_handle.await.unwrap().unwrap_err();

        assert!(
            matches!(
                err,
                silent_io::error::IoError::CoefficientOutOfRange { .. }
            ),
            "expected CoefficientOutOfRange, got {:?}",
            err
        );
    }

    #[tokio::test]
    async fn native_poly_rejects_overflow_coefficient_in_second_position() {
        let (alice, bob) = MemoryConnection::pair();

        // degree = 2, log_modulus = 32, second coefficient overflows.
        let mut payload = Vec::new();
        payload.push(32u8);
        payload.extend_from_slice(&2u32.to_le_bytes());
        // coeff 0 — valid
        payload.extend_from_slice(&42u64.to_le_bytes());
        // coeff 1 — overflow
        payload.extend_from_slice(&0xFFFF_FFFF_0000u64.to_le_bytes());

        let frame = crate::frame::NetworkFrame {
            session_id: 1,
            task_id: 2,
            gate_id: 3,
            message_kind: crate::frame::MessageKind::Data,
            object_type: silent_ring::NativePoly::OBJECT_TYPE,
            object_version: silent_ring::NativePoly::SERIALIZED_VERSION,
            scheme_id: 0,
            params_id: silent_io::ParamsId::ZERO,
            flags: 0,
            payload,
        };

        let frame_clone = frame.clone();

        let alice_handle = tokio::spawn(async move {
            let mut stream = alice.open_stream().await.unwrap();
            crate::frame::send_frame(&mut stream, &frame_clone)
                .await
                .unwrap();
        });

        let bob_handle = tokio::spawn(async move {
            let mut stream = bob.accept_stream().await.unwrap();
            let raw = crate::frame::recv_frame(&mut stream, DEFAULT_MAX_FRAME_PAYLOAD)
                .await
                .unwrap();
            NativePoly::decode_with_params(&(2usize, 32u8), &mut std::io::Cursor::new(&raw.payload))
        });

        alice_handle.await.unwrap();
        let err = bob_handle.await.unwrap().unwrap_err();

        assert!(
            matches!(
                err,
                silent_io::error::IoError::CoefficientOutOfRange { .. }
            ),
            "expected CoefficientOutOfRange at index 1, got {:?}",
            err
        );
    }
}
