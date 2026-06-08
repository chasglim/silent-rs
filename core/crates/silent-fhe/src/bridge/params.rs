use crate::schemes::bfv::params::{BfvMulMethod, BfvParameters};
use crate::schemes::tfhe::encoding::TfheEncoder;
use crate::schemes::tfhe::params::TfheParameters;

use super::error::BridgeError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BridgeMode {
    DiscreteInteger,
    FixedPoint,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BridgePlaintextModulus {
    NativeU64(u64),
    Crt(Vec<u64>),
}

/// Layout of the BFV plaintext that participates in the bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BridgeLayout {
    Coefficient,
    BatchSlots,
}

/// Parameters governing the BFV <-> TFHE bridge.
#[derive(Clone, Debug)]
pub struct BridgeParams {
    pub mode: BridgeMode,
    pub bfv_params: BfvParameters,
    pub tfhe_params: TfheParameters,
    pub message_bits: usize,
    pub num_slots: usize,
    pub layout: BridgeLayout,
    pub noise_margin_bits: usize,
}

impl BridgeParams {
    pub fn new(
        bfv_params: BfvParameters,
        tfhe_params: TfheParameters,
        message_bits: usize,
        num_slots: usize,
        layout: BridgeLayout,
    ) -> Result<Self, BridgeError> {
        if num_slots == 0 || num_slots > bfv_params.degree() {
            return Err(BridgeError::BatchTooLarge(num_slots, bfv_params.degree()));
        }
        let total = tfhe_params.total_message_modulus();
        let expected = 1u64.checked_shl(message_bits as u32).ok_or_else(|| {
            BridgeError::IncompatibleParameters("message_bits must be smaller than 64".into())
        })?;
        if total != expected {
            return Err(BridgeError::PlaintextModulusMismatch(expected, total));
        }

        Ok(Self {
            mode: BridgeMode::DiscreteInteger, // Default mode
            bfv_params,
            tfhe_params,
            message_bits,
            num_slots,
            layout,
            noise_margin_bits: 4, // Default margin
        })
    }

    pub fn message_modulus(&self) -> u64 {
        1u64 << self.message_bits
    }

    pub fn padded_message_modulus(&self) -> u64 {
        self.message_modulus().saturating_mul(2)
    }

    pub fn tfhe_delta(&self) -> u64 {
        TfheEncoder::new(self.tfhe_params.clone()).delta()
    }

    pub fn bfv_first_modulus(&self) -> u64 {
        self.bfv_params.ring().rns().moduli()[0].value()
    }

    /// Bridge-side BFV eval/key-switching currently relies on the HPS path.
    /// Keep this local to bridge code so the original BFV defaults stay intact.
    pub fn bridge_bfv_eval_params(&self) -> BfvParameters {
        self.bfv_params.clone().with_mul_method(BfvMulMethod::Hps)
    }

    pub fn bfv_delta_first_limb(&self) -> u64 {
        let q0 = self.bfv_first_modulus() as u128;
        let t = self.bfv_params.plain_modulus() as u128;
        ((q0 + (t / 2)) / t) as u64
    }

    pub fn switched_bfv_delta(&self) -> u64 {
        let q0 = self.bfv_first_modulus() as u128;
        let delta_bfv = self.bfv_delta_first_limb() as u128;
        (((delta_bfv << 64) + (q0 / 2)) / q0) as u64
    }

    pub fn bfv_to_tfhe_scale(&self) -> u64 {
        1
    }
}
