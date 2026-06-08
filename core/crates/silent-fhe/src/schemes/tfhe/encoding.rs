//! Plaintext encoding for TFHE.
//!
//! TFHE works on **scalar** plaintexts (one integer per ciphertext) drawn
//! from `Z_{message_modulus}`.  Encoding shifts the message into the most
//! significant bits of a native-modulus word so that a small additive noise
//! does not corrupt it; decoding rounds back to the integer slot.
//!
//! The encoder is *not* yet wired into the existing
//! [`crate::core::encoder::HeEncoder`] trait because that trait was modelled
//! after BFV's batched-slot semantics (it returns a polynomial-shaped
//! [`silent_rlwe::Plaintext`]).  M6 of the implementation plan widens the
//! trait with associated types so a TFHE-friendly variant can be plugged in
//! without affecting BFV; until then we expose a small, focused TFHE API.

use silent_math::torus::{decode_msb_u64, encode_msb_u64};

use super::params::TfheParameters;
use silent_io::HasParamsId;

/// Scalar plaintext produced by [`TfheEncoder::encode_message`].
///
/// Implemented as a tiny struct rather than a bare `u64` so that the type
/// system pins the active modulus down and prevents accidental misuse from
/// other schemes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TfhePlaintext {
    pub value: u64,
    pub log_modulus: u8,
}

impl TfhePlaintext {
    pub fn new(value: u64, log_modulus: u8) -> Self {
        Self { value, log_modulus }
    }
}

/// Encoder for TFHE scalar messages.
///
/// Internally maintains both
///   * `full_plaintext_modulus = message_modulus · carry_modulus`,
///     the *user-facing* plaintext space, and
///   * `padded_plaintext_modulus = 2 · full_plaintext_modulus`,
///     the *padded* modulus used to compute the encoding scale `Δ = q / 2T`.
///
/// The padding bit is required by TFHE's blind-rotation arithmetic:
/// it shifts the rotation lattice so that messages and their negacyclic
/// images do not collide at the boundary slot.  All M4 round-trip tests still
/// pass because additions inside the carry buffer remain inside the upper
/// half of the padded range and never cross the negacyclic mid-point.
#[derive(Clone, Debug)]
pub struct TfheEncoder {
    params: TfheParameters,
    full_plaintext_modulus: u64,
    padded_plaintext_modulus: u64,
}

impl TfheEncoder {
    pub fn new(params: TfheParameters) -> Self {
        let full = params.total_message_modulus();
        let padded = full
            .checked_mul(2)
            .expect("padded modulus must fit in u64; validated by TfheParameters");
        Self {
            params,
            full_plaintext_modulus: full,
            padded_plaintext_modulus: padded,
        }
    }

    pub fn params(&self) -> &TfheParameters {
        &self.params
    }

    /// Encode a *message* `m ∈ [0, message_modulus)` into an LWE plaintext.
    ///
    /// The carry slot is left empty; subsequent homomorphic additions can
    /// fill it without overflowing into the message bits.
    pub fn encode_message(&self, message: u64) -> TfhePlaintext {
        debug_assert!(message < self.params.message_modulus().0);
        TfhePlaintext::new(
            encode_msb_u64(
                message,
                self.padded_plaintext_modulus,
                self.params.ciphertext_modulus_log().0,
            ),
            self.params.ciphertext_modulus_log().0,
        )
    }

    /// Encode a *plaintext slot value* `v ∈ [0, total_message_modulus)`
    /// directly into the MSBs.  Useful for testing and for encoding lookup
    /// table outputs.
    pub fn encode_full(&self, value: u64) -> TfhePlaintext {
        debug_assert!(value < self.full_plaintext_modulus);
        TfhePlaintext::new(
            encode_msb_u64(
                value,
                self.padded_plaintext_modulus,
                self.params.ciphertext_modulus_log().0,
            ),
            self.params.ciphertext_modulus_log().0,
        )
    }

    /// Decode an LWE plaintext back into the *message* slot, dropping the
    /// carry information.
    pub fn decode_message(&self, plain: TfhePlaintext) -> u64 {
        let full = decode_msb_u64(
            plain.value,
            self.padded_plaintext_modulus,
            self.params.ciphertext_modulus_log().0,
        );
        // Strip the padding bit and the carry slots.
        (full % self.full_plaintext_modulus) % self.params.message_modulus().0
    }

    /// Decode the full plaintext slot (message + carry).
    pub fn decode_full(&self, plain: TfhePlaintext) -> u64 {
        let raw = decode_msb_u64(
            plain.value,
            self.padded_plaintext_modulus,
            self.params.ciphertext_modulus_log().0,
        );
        raw % self.full_plaintext_modulus
    }

    /// Returns `Δ = q / (2 · total_message_modulus)`.
    pub fn delta(&self) -> u64 {
        let log_q = self.params.ciphertext_modulus_log().0;
        let plain_log = self.padded_plaintext_modulus.trailing_zeros() as u8;
        if log_q - plain_log >= 64 {
            0
        } else {
            1u64 << (log_q - plain_log)
        }
    }

    /// Effective unpadded plaintext modulus
    /// (`message_modulus * carry_modulus`).
    pub fn full_plaintext_modulus(&self) -> u64 {
        self.full_plaintext_modulus
    }

    /// Padded plaintext modulus used internally to derive `Δ`.  Useful for
    /// LUT construction where the actual rotation lattice has `2T` slots.
    pub fn padded_plaintext_modulus(&self) -> u64 {
        self.padded_plaintext_modulus
    }
}

impl HasParamsId for TfheEncoder {
    fn params_id(&self) -> silent_io::ParamsId {
        silent_io::ParamsId(silent_params::ParameterSet::params_id(self.params.canonical()).0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use silent_params::presets;

    fn encoder() -> TfheEncoder {
        let params = TfheParameters::new(presets::toy::toy_tfhe_n512()).unwrap();
        TfheEncoder::new(params)
    }

    #[test]
    fn encode_decode_roundtrip() {
        let enc = encoder();
        for m in 0..enc.params().message_modulus().0 {
            let p = enc.encode_message(m);
            assert_eq!(enc.decode_message(p), m);
        }
    }

    #[test]
    fn delta_matches_q_div_padded_plaintext() {
        let enc = encoder();
        let log_q = enc.params().ciphertext_modulus_log().0;
        let padded_log = enc.padded_plaintext_modulus().trailing_zeros() as u8;
        let expected = 1u64 << (log_q - padded_log);
        assert_eq!(enc.delta(), expected);
    }
}
