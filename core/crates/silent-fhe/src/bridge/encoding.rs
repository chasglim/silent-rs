//! Bridge-side message encoding helpers.
//!
//! Maps between:
//!   * BFV plaintext: `m ∈ Z_t` (signed or unsigned small integer)
//!   * TFHE torus:  `μ = Δ_T · m` where `Δ_T = q / (2 · 2^k)`

use super::params::BridgeParams;

#[derive(Clone, Debug)]
pub struct BridgeEncoder {
    params: BridgeParams,
    delta: u64,
    half_modulus: u64,
}

impl BridgeEncoder {
    pub fn new(params: BridgeParams) -> Self {
        let delta = params.tfhe_delta();
        let half_modulus = params.message_modulus() >> 1;
        Self {
            params,
            delta,
            half_modulus,
        }
    }

    pub fn params(&self) -> &BridgeParams {
        &self.params
    }

    /// Convert a BFV-signed integer `m ∈ [-M/2, M/2)` to unsigned
    /// `m' ∈ [0, M)` for TFHE consumption.
    pub fn bfv_to_tfhe_message(&self, m: i64) -> u64 {
        let m_mod = m.rem_euclid(self.params.message_modulus() as i64) as u64;
        m_mod
    }

    /// Convert an unsigned TFHE message `m' ∈ [0, M)` back to a
    /// BFV-signed integer `m ∈ [-M/2, M/2)`.
    pub fn tfhe_to_bfv_message(&self, m: u64) -> i64 {
        let m = m & (self.params.message_modulus() - 1);
        if m >= self.half_modulus {
            m as i64 - self.params.message_modulus() as i64
        } else {
            m as i64
        }
    }

    /// Scale an unsigned message `m` to a TFHE torus value `Δ_T·m`.
    pub fn encode_torus(&self, message: u64) -> u64 {
        let m = message & (self.params.message_modulus() - 1);
        m.wrapping_mul(self.delta)
    }

    /// Decode a TFHE torus value back to an unsigned message.
    pub fn decode_torus(&self, torus_value: u64) -> u64 {
        let log_q = self.params.tfhe_params.ciphertext_modulus_log().0;
        let padded_bits = self.params.message_bits as u32 + 1;
        let shift = log_q as u32 - padded_bits;
        let rounded = if shift >= 64 {
            0
        } else {
            (torus_value.wrapping_add(self.delta >> 1)) >> shift
        };
        rounded & (self.params.padded_message_modulus() - 1) % self.params.message_modulus()
    }

    /// Encode a signed BFV integer directly to torus.
    pub fn encode_signed_torus(&self, m: i64) -> u64 {
        self.encode_torus(self.bfv_to_tfhe_message(m))
    }
}
