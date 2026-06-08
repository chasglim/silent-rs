#[cfg(test)]
use super::repack_naive_homomorphic::repack_with_secret;
use crate::bridge::keys::BridgeKeys;
use crate::bridge::{BridgeError, BridgeLayout, BridgeParams};
#[cfg(test)]
use crate::schemes::tfhe::crypto::TfheDecryptor;
use silent_params::NoiseDistribution;
use silent_rlwe::{Ciphertext, LweCiphertext};
#[cfg(test)]
use silent_utils::rng::SecureRng;

pub struct TfheToBfvConverter<'a> {
    params: &'a BridgeParams,
    keys: &'a BridgeKeys,
}

impl<'a> TfheToBfvConverter<'a> {
    pub fn new(params: &'a BridgeParams, keys: &'a BridgeKeys) -> Self {
        Self { params, keys }
    }

    pub fn convert_homomorphic(
        &self,
        tfhe_cts: &[LweCiphertext],
    ) -> Result<Ciphertext, BridgeError> {
        validate_tfhe_inputs(tfhe_cts, self.params)?;
        ensure_full_tfhe_to_bfv_preconditions(self.params)?;
        super::torus_decode::bfv_decode_lwe1_zero_noise_to_coefficients(
            tfhe_cts,
            self.keys,
            self.params,
        )
    }

    /// Convert exact torus-native `1/p` torus-native LWE1 inputs into BFV
    /// coefficient messages.
    ///
    /// This path is intentionally narrow: it assumes TFHE PBS/functional
    /// bootstrapping has already normalized every TLWE plaintext to an exact
    /// multiple of `1/p`. General noisy/high-dimensional TFHE inputs are still
    /// rejected by `ensure_torus_native_tfhe_to_bfv_preconditions`.
    pub fn convert_torus_native_lwe1_exact(
        &self,
        tfhe_cts: &[LweCiphertext],
    ) -> Result<Ciphertext, BridgeError> {
        validate_tfhe_inputs(tfhe_cts, self.params)?;
        ensure_torus_native_tfhe_to_bfv_preconditions(self.params)?;
        super::torus_decode::bfv_decode_lwe1_torus_native_to_coefficients(
            tfhe_cts,
            self.keys,
            self.params,
        )
    }

    /// Repack TLWE ciphertext phases into BFV coefficient layout.
    ///
    /// This is the real homomorphic partial-decryption/repacking stage of
    /// TFHE -> BFV. The output encrypts raw torus phases, not decoded messages;
    /// callers still need a reviewed Fmod/TorusDecode backend unless they use
    /// the exact LWE1 public converter.
    pub fn convert_to_phase_coefficients(
        &self,
        tfhe_cts: &[LweCiphertext],
    ) -> Result<Ciphertext, BridgeError> {
        validate_tfhe_inputs(tfhe_cts, self.params)?;
        super::repack_naive_homomorphic::repack_naive(tfhe_cts, &self.keys.tfhe_to_bfv, self.params)
    }

    /// Decode an already-repacked BFV phase ciphertext into BFV messages.
    ///
    /// The current packed backend is center-only: the plaintext coefficients
    /// must be exact TFHE centers `Δ_T * m`. Lifted `k*q_T` offsets are handled
    /// only by `convert_homomorphic`, which decodes each phase before packing.
    pub fn convert_phase_to_messages(
        &self,
        phase_ct: &Ciphertext,
    ) -> Result<Ciphertext, BridgeError> {
        super::torus_decode::bfv_fmod_torus_decode(phase_ct, self.keys, self.params)
    }
}

pub fn tfhe_to_bfv(
    tfhe_cts: &[LweCiphertext],
    params: &BridgeParams,
    keys: &BridgeKeys,
) -> Result<Ciphertext, BridgeError> {
    let converter = TfheToBfvConverter::new(params, keys);
    converter.convert_homomorphic(tfhe_cts)
}

pub fn tfhe_to_bfv_phase_coefficients(
    tfhe_cts: &[LweCiphertext],
    params: &BridgeParams,
    keys: &BridgeKeys,
) -> Result<Ciphertext, BridgeError> {
    let converter = TfheToBfvConverter::new(params, keys);
    converter.convert_to_phase_coefficients(tfhe_cts)
}

pub fn tfhe_to_bfv_torus_native_lwe1_exact(
    tfhe_cts: &[LweCiphertext],
    params: &BridgeParams,
    keys: &BridgeKeys,
) -> Result<Ciphertext, BridgeError> {
    let converter = TfheToBfvConverter::new(params, keys);
    converter.convert_torus_native_lwe1_exact(tfhe_cts)
}

pub fn bfv_phase_to_bfv_messages(
    phase_ct: &Ciphertext,
    params: &BridgeParams,
    keys: &BridgeKeys,
) -> Result<Ciphertext, BridgeError> {
    let converter = TfheToBfvConverter::new(params, keys);
    converter.convert_phase_to_messages(phase_ct)
}

pub fn ensure_full_tfhe_to_bfv_preconditions(params: &BridgeParams) -> Result<(), BridgeError> {
    if params.layout != BridgeLayout::Coefficient {
        return Err(BridgeError::IncompatibleParameters(
            "TFHE->BFV currently supports only coefficient layout".into(),
        ));
    }
    super::torus_decode::ensure_lwe1_direct_decode_preconditions(params)?;
    ensure_zero_lwe_noise(params)?;
    Ok(())
}

pub fn ensure_torus_native_tfhe_to_bfv_preconditions(
    params: &BridgeParams,
) -> Result<(), BridgeError> {
    if params.layout != BridgeLayout::Coefficient {
        return Err(BridgeError::IncompatibleParameters(
            "torus-native TFHE->BFV currently supports only coefficient layout".into(),
        ));
    }
    super::torus_decode::ensure_lwe1_torus_native_preconditions(params)?;
    ensure_zero_lwe_noise(params)?;
    Ok(())
}

fn ensure_zero_lwe_noise(params: &BridgeParams) -> Result<(), BridgeError> {
    match params.tfhe_params.lwe_noise() {
        NoiseDistribution::Gaussian { stddev } if stddev == 0.0 => Ok(()),
        NoiseDistribution::Gaussian { stddev } => {
            Err(BridgeError::IncompatibleParameters(format!(
                "exact TFHE->BFV requires zero-noise TFHE LWE inputs; got Gaussian stddev {stddev}. Noisy inputs require an audited interval Fmod/noise proof"
            )))
        }
        NoiseDistribution::TUniform { bound_log2 } => {
            Err(BridgeError::IncompatibleParameters(format!(
                "exact TFHE->BFV requires zero-noise TFHE LWE inputs; got TUniform bound_log2 {bound_log2}. Noisy inputs require an audited interval Fmod/noise proof"
            )))
        }
    }
}

fn validate_tfhe_inputs(
    tfhe_cts: &[LweCiphertext],
    params: &BridgeParams,
) -> Result<(), BridgeError> {
    if tfhe_cts.is_empty() {
        return Err(BridgeError::IncompatibleParameters(
            "TFHE->BFV conversion requires at least one TFHE ciphertext".into(),
        ));
    }

    let expected_dimension = params.tfhe_params.lwe_dimension().0;
    let expected_log_modulus = params.tfhe_params.ciphertext_modulus_log().0;
    for (index, ct) in tfhe_cts.iter().enumerate() {
        if ct.dimension().0 != expected_dimension {
            return Err(BridgeError::IncompatibleParameters(format!(
                "TFHE ciphertext {index} has LWE dimension {}, expected {}",
                ct.dimension().0,
                expected_dimension
            )));
        }
        if ct.log_modulus() != expected_log_modulus {
            return Err(BridgeError::IncompatibleParameters(format!(
                "TFHE ciphertext {index} has log modulus {}, expected {}",
                ct.log_modulus(),
                expected_log_modulus
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
pub fn tfhe_to_bfv_trusted_oracle(
    tfhe_cts: &[LweCiphertext],
    params: &BridgeParams,
    tfhe_dec: &TfheDecryptor,
    bfv_sk: &silent_rlwe::SecretKey,
    rng: SecureRng,
) -> Result<Ciphertext, BridgeError> {
    repack_with_secret(tfhe_cts, params, tfhe_dec, bfv_sk, rng)
}
