//! BFV -> TFHE conversion pipeline.

#[cfg(test)]
use crate::schemes::bfv::crypto::BfvDecryptor;
#[cfg(test)]
use crate::schemes::tfhe::crypto::TfheEncryptor;
#[cfg(test)]
use crate::schemes::tfhe::encoding::TfheEncoder;
use crate::schemes::tfhe::keyswitch::LweKeyswitcher;
#[cfg(test)]
use rand_core::SeedableRng;
use silent_params::NoiseDistribution;
use silent_rlwe::{Ciphertext, LweCiphertext};
#[cfg(test)]
use silent_utils::rng::SecureRng;
use std::thread;

pub mod extract;
pub mod keyswitch;
pub mod modswitch;
pub mod slot_to_coeff;

use super::keys::BridgeKeys;
use super::{BridgeError, BridgeParams};
use extract::{extract_coefficient_lwe, extract_lwe_from_bfv};
use modswitch::{modswitch_bfv_lwe_to_tfhe, modswitch_bfv_lwe_to_torus_native_tfhe};

pub struct BfvToTfheConverter<'a> {
    params: &'a BridgeParams,
    keys: &'a BridgeKeys,
}

impl<'a> BfvToTfheConverter<'a> {
    pub fn new(params: &'a BridgeParams, keys: &'a BridgeKeys) -> Self {
        Self { params, keys }
    }

    pub fn convert(&self, ct: &Ciphertext) -> Result<Vec<LweCiphertext>, BridgeError> {
        if !matches!(self.params.layout, super::BridgeLayout::Coefficient) {
            return Err(BridgeError::IncompatibleParameters(
                "BFV->TFHE currently supports only coefficient layout; BatchSlots requires SlotToCoeff first"
                    .into(),
            ));
        }
        self.convert_via_keyswitch(ct)
    }

    pub fn convert_via_keyswitch(
        &self,
        ct: &Ciphertext,
    ) -> Result<Vec<LweCiphertext>, BridgeError> {
        ensure_public_bfv_to_tfhe_preconditions(self.params)?;
        let switcher = LweKeyswitcher::new(&self.keys.bfv_to_tfhe.ksk);
        let extracted = extract_lwe_from_bfv(ct, self.params)?;

        Ok(extracted
            .into_iter()
            .map(|lwe_q| {
                let lwe_t = modswitch_bfv_lwe_to_tfhe(&lwe_q, self.params)?;
                Ok(switcher.keyswitch(&lwe_t))
            })
            .collect::<Result<Vec<_>, BridgeError>>()?)
    }

    /// Convert BFV coefficient plaintexts to TLWE ciphertexts under the
    /// SILENT torus-native contract.
    ///
    /// BFV coefficient `z in Z_p` is interpreted as the torus point `z/p`.
    /// The returned TFHE ciphertexts therefore decrypt to raw torus phases
    /// `round(q_T * z / p)`, not to values encoded by `TfheEncoder::encode_full`.
    pub fn convert_torus_native(&self, ct: &Ciphertext) -> Result<Vec<LweCiphertext>, BridgeError> {
        if !matches!(self.params.layout, super::BridgeLayout::Coefficient) {
            return Err(BridgeError::IncompatibleParameters(
                "BFV->TFHE torus-native conversion currently supports only coefficient layout; BatchSlots requires SlotToCoeff first"
                    .into(),
            ));
        }
        ensure_torus_native_bfv_to_tfhe_preconditions(self.params, self.keys)?;

        self.convert_torus_native_coefficients(ct, self.params.num_slots)
    }

    /// Convert BFV SIMD slots to TLWE ciphertexts under the SILENT
    /// torus-native contract.
    ///
    /// This follows the Pegasus/CHIMERA production order:
    /// `RLWE slots -> SlotsToCoefficients -> coefficient extraction ->
    /// modulus/key switching`.
    pub fn convert_slots_torus_native(
        &self,
        ct: &Ciphertext,
    ) -> Result<Vec<LweCiphertext>, BridgeError> {
        if !matches!(
            self.params.layout,
            super::BridgeLayout::Coefficient | super::BridgeLayout::BatchSlots
        ) {
            return Err(BridgeError::IncompatibleParameters(
                "BFV slot->TFHE torus-native conversion expects a coefficient-domain BFV ciphertext containing SIMD slots"
                    .into(),
            ));
        }
        ensure_torus_native_bfv_to_tfhe_preconditions(self.params, self.keys)?;

        let coeff_ct = slot_to_coeff::slot_to_coeff(ct, self.keys, self.params)?;
        self.convert_torus_native_coefficients(&coeff_ct, self.params.num_slots)
    }

    fn convert_torus_native_coefficients(
        &self,
        ct: &Ciphertext,
        count: usize,
    ) -> Result<Vec<LweCiphertext>, BridgeError> {
        if count < 64 {
            return self.convert_torus_native_coefficient_range(ct, 0, count);
        }

        let workers = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(count);
        if workers <= 1 {
            return self.convert_torus_native_coefficient_range(ct, 0, count);
        }

        let chunk = count.div_ceil(workers);
        let mut chunks = Vec::with_capacity(workers);
        thread::scope(|scope| -> Result<(), BridgeError> {
            let mut handles = Vec::new();
            for start in (0..count).step_by(chunk) {
                let end = (start + chunk).min(count);
                handles.push(scope.spawn(move || {
                    self.convert_torus_native_coefficient_range(ct, start, end)
                        .map(|values| (start, values))
                }));
            }
            for handle in handles {
                let chunk = handle.join().map_err(|_| {
                    BridgeError::IncompatibleParameters(
                        "BFV->TFHE coefficient extraction worker panicked".into(),
                    )
                })??;
                chunks.push(chunk);
            }
            Ok(())
        })?;

        chunks.sort_by_key(|(start, _)| *start);
        let mut out = Vec::with_capacity(count);
        for (_, mut values) in chunks {
            out.append(&mut values);
        }
        Ok(out)
    }

    fn convert_torus_native_coefficient_range(
        &self,
        ct: &Ciphertext,
        start: usize,
        end: usize,
    ) -> Result<Vec<LweCiphertext>, BridgeError> {
        let switcher = LweKeyswitcher::new(&self.keys.bfv_to_tfhe.ksk);
        (start..end)
            .map(|index| {
                let lwe_q = extract_coefficient_lwe(ct, index, self.params)?;
                let lwe_t = modswitch_bfv_lwe_to_torus_native_tfhe(&lwe_q, self.params)?;
                Ok(switcher.keyswitch(&lwe_t))
            })
            .collect::<Result<Vec<_>, BridgeError>>()
    }
}

fn ensure_public_bfv_to_tfhe_preconditions(params: &BridgeParams) -> Result<(), BridgeError> {
    let padded_plaintext_modulus = params
        .tfhe_params
        .total_message_modulus()
        .checked_mul(2)
        .ok_or_else(|| {
            BridgeError::IncompatibleParameters(
                "TFHE padded plaintext modulus overflows u64".into(),
            )
        })?;
    if params.bfv_params.plain_modulus() != padded_plaintext_modulus {
        return Err(BridgeError::IncompatibleParameters(format!(
            "public BFV->TFHE conversion currently requires BFV plaintext modulus t={} to equal TFHE padded plaintext modulus 2*(message*carry)={}; non-native scaling needs an explicit BFV noise budget proof",
            params.bfv_params.plain_modulus(),
            padded_plaintext_modulus
        )));
    }
    Ok(())
}

fn ensure_torus_native_bfv_to_tfhe_preconditions(
    params: &BridgeParams,
    keys: &BridgeKeys,
) -> Result<(), BridgeError> {
    let ksk = &keys.bfv_to_tfhe.ksk;
    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    if ksk.log_modulus() != log_q {
        return Err(BridgeError::IncompatibleParameters(format!(
            "BFV->TFHE torus-native KSK modulus log {} does not match TFHE modulus log {}",
            ksk.log_modulus(),
            log_q
        )));
    }
    if ksk.input_dimension().0 != params.bfv_params.degree() {
        return Err(BridgeError::IncompatibleParameters(format!(
            "BFV->TFHE torus-native KSK input dimension {} does not match BFV degree {}",
            ksk.input_dimension().0,
            params.bfv_params.degree()
        )));
    }

    let precision_bits = u16::from(ksk.base_log()) * u16::from(ksk.level());
    if precision_bits < u16::from(log_q) {
        return Err(BridgeError::IncompatibleParameters(format!(
            "BFV->TFHE torus-native conversion requires full q-bit key-switch decomposition for audited 1/p cells; got base_log*level={} bits for q=2^{}",
            precision_bits, log_q
        )));
    }

    let q_t = 1u128.checked_shl(log_q as u32).ok_or_else(|| {
        BridgeError::IncompatibleParameters("TFHE ciphertext modulus exceeds u128".into())
    })?;
    let plain_modulus = params.bfv_params.plain_modulus() as u128;
    let half_cell = q_t / (2 * plain_modulus.max(1));
    let ksk_noise = estimate_key_switch_noise_bound(
        params.tfhe_params.lwe_noise(),
        ksk.input_dimension().0,
        ksk.base_log(),
        ksk.level(),
        log_q,
    )?;
    let amplified_noise = ksk_noise
        .checked_shl(params.noise_margin_bits as u32)
        .ok_or_else(|| {
            BridgeError::IncompatibleParameters(
                "BFV->TFHE torus-native noise margin overflows u128".into(),
            )
        })?;
    if amplified_noise >= half_cell {
        return Err(BridgeError::IncompatibleParameters(format!(
            "BFV->TFHE torus-native key-switch noise bound {} with {} margin bits exceeds 1/p half-cell {}; reduce TFHE key-switch noise/increase precision or use a smaller BFV plaintext modulus p={}",
            ksk_noise,
            params.noise_margin_bits,
            half_cell,
            params.bfv_params.plain_modulus()
        )));
    }

    Ok(())
}

fn estimate_key_switch_noise_bound(
    noise: NoiseDistribution,
    input_dimension: usize,
    base_log: u8,
    level: u8,
    log_q: u8,
) -> Result<u128, BridgeError> {
    let base = 1u128.checked_shl(base_log as u32).ok_or_else(|| {
        BridgeError::IncompatibleParameters("key-switch base exceeds u128".into())
    })?;
    let term_count = (input_dimension as u128)
        .checked_mul(level as u128)
        .ok_or_else(|| {
            BridgeError::IncompatibleParameters("KSK term count overflows u128".into())
        })?;

    match noise {
        NoiseDistribution::Gaussian { stddev } if stddev == 0.0 => Ok(0),
        NoiseDistribution::Gaussian { stddev } => {
            let q = if log_q == 64 {
                2f64.powi(64)
            } else {
                (1u64 << log_q) as f64
            };
            let base_minus_one = (base - 1) as f64;
            let weighted_terms = (term_count as f64) * base_minus_one * base_minus_one;
            let bound = 8.0 * stddev * q * weighted_terms.sqrt();
            if !bound.is_finite() || bound < 0.0 || bound > u128::MAX as f64 {
                return Err(BridgeError::IncompatibleParameters(
                    "BFV->TFHE torus-native Gaussian key-switch noise bound is not finite".into(),
                ));
            }
            Ok(bound.ceil() as u128)
        }
        NoiseDistribution::TUniform { bound_log2 } => {
            let per_sample = 1u128.checked_shl(bound_log2 as u32).ok_or_else(|| {
                BridgeError::IncompatibleParameters(
                    "key-switch TUniform noise bound exceeds u128".into(),
                )
            })?;
            term_count
                .checked_mul(base - 1)
                .and_then(|x| x.checked_mul(per_sample))
                .ok_or_else(|| {
                    BridgeError::IncompatibleParameters(
                        "BFV->TFHE torus-native TUniform key-switch noise bound overflows u128"
                            .into(),
                    )
                })
        }
    }
}

pub fn bfv_to_tfhe(
    bfv_ct: &Ciphertext,
    keys: &BridgeKeys,
    params: &BridgeParams,
) -> Result<Vec<LweCiphertext>, BridgeError> {
    BfvToTfheConverter::new(params, keys).convert(bfv_ct)
}

pub fn bfv_to_tfhe_torus_native(
    bfv_ct: &Ciphertext,
    keys: &BridgeKeys,
    params: &BridgeParams,
) -> Result<Vec<LweCiphertext>, BridgeError> {
    BfvToTfheConverter::new(params, keys).convert_torus_native(bfv_ct)
}

pub fn bfv_slots_to_tfhe_torus_native(
    bfv_ct: &Ciphertext,
    keys: &BridgeKeys,
    params: &BridgeParams,
) -> Result<Vec<LweCiphertext>, BridgeError> {
    BfvToTfheConverter::new(params, keys).convert_slots_torus_native(bfv_ct)
}

#[cfg(test)]
pub fn bfv_to_tfhe_trusted_oracle(
    bfv_ct: &Ciphertext,
    params: &BridgeParams,
    bfv_sk: &silent_rlwe::SecretKey,
    tfhe_sk: &silent_rlwe::LweSecretKey,
) -> Result<Vec<LweCiphertext>, BridgeError> {
    let encoder = TfheEncoder::new(params.tfhe_params.clone());
    let mut encryptor = TfheEncryptor::new(params.tfhe_params.clone(), SecureRng::from_entropy());
    let values = decrypt_coefficients_with_secret(bfv_ct, params, bfv_sk)?;
    let mut out = Vec::with_capacity(params.num_slots);
    for value in values.iter().take(params.num_slots) {
        out.push(encryptor.encrypt_symmetric(
            tfhe_sk,
            encoder.encode_full(*value % params.message_modulus()),
        ));
    }
    Ok(out)
}

#[cfg(test)]
fn decrypt_coefficients_with_secret(
    ct: &Ciphertext,
    params: &BridgeParams,
    sk: &silent_rlwe::SecretKey,
) -> Result<Vec<u64>, BridgeError> {
    if ct.data.len() != 2 {
        return Err(BridgeError::InvalidBfvCiphertext(ct.data.len()));
    }
    let decryptor = BfvDecryptor::new(params.bfv_params.clone(), sk.clone());
    let plain = decryptor.decrypt(ct);
    Ok(plain.value.limb(0)[..params.num_slots].to_vec())
}
