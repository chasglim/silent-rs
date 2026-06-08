use crate::bridge::{BridgeError, BridgeParams};
use crate::schemes::tfhe::crypto::TfheDecryptor;
use silent_rlwe::{LweCiphertext, LweSecretKey};

/// Convention check: determine whether `phase = b - <a, s>` or `b + <a, s>`.
pub fn get_phase_convention(
    cts: &[LweCiphertext],
    params: &BridgeParams,
    tfhe_dec: &TfheDecryptor,
) -> Result<(), BridgeError> {
    let expected_dimension = params.tfhe_params.lwe_dimension().0;
    let expected_log_modulus = params.tfhe_params.ciphertext_modulus_log().0;

    for (index, ct) in cts.iter().enumerate() {
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

        let raw = tfhe_dec.decrypt_raw(ct);
        let minus = phase_b_minus_as(ct, tfhe_dec.secret_key());
        if raw != minus {
            let plus = phase_b_plus_as(ct, tfhe_dec.secret_key());
            return Err(BridgeError::IncompatibleParameters(format!(
                "TFHE phase convention mismatch at ciphertext {index}: decrypt_raw={raw}, b-<a,s>={minus}, b+<a,s>={plus}"
            )));
        }
    }

    Ok(())
}

pub fn phase_b_minus_as(ct: &LweCiphertext, sk: &LweSecretKey) -> u64 {
    phase_with_sign(ct, sk, true)
}

pub fn phase_b_plus_as(ct: &LweCiphertext, sk: &LweSecretKey) -> u64 {
    phase_with_sign(ct, sk, false)
}

fn phase_with_sign(ct: &LweCiphertext, sk: &LweSecretKey, subtract: bool) -> u64 {
    debug_assert_eq!(ct.dimension().0, sk.dimension().0);
    let mut inner = 0u64;
    for (a, s) in ct.mask().iter().zip(sk.data().iter()) {
        inner = inner.wrapping_add(a.wrapping_mul(*s));
    }

    let mut value = if subtract {
        ct.body().wrapping_sub(inner)
    } else {
        ct.body().wrapping_add(inner)
    };
    if ct.log_modulus() < 64 {
        value &= (1u64 << ct.log_modulus()) - 1;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::{phase_b_minus_as, phase_b_plus_as};
    use crate::schemes::tfhe::crypto::{TfheDecryptor, TfheEncryptor};
    use crate::schemes::tfhe::encoding::TfheEncoder;
    use crate::schemes::tfhe::keys::TfheKeyGenerator;
    use crate::schemes::tfhe::params::TfheParameters;
    use rand_core::SeedableRng;
    use silent_params::presets;
    use silent_utils::rng::SecureRng;

    #[test]
    fn convention_matches_tfhe_decrypt_raw() {
        let mut raw = presets::toy::toy_tfhe_n512();
        raw.ciphertext_modulus_log = silent_params::CiphertextModulusLog(32);
        let params = TfheParameters::new(raw).unwrap();
        let mut keygen =
            TfheKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([91u8; 32]));
        let sk = keygen.generate_lwe_secret_key();
        let mut enc = TfheEncryptor::new(params.clone(), SecureRng::from_seed([92u8; 32]));
        let dec = TfheDecryptor::new(params.clone(), sk.clone());
        let encoder = TfheEncoder::new(params);
        let ct = enc.encrypt_symmetric(&sk, encoder.encode_full(3));

        assert_eq!(phase_b_minus_as(&ct, &sk), dec.decrypt_raw(&ct));
        assert_ne!(phase_b_plus_as(&ct, &sk), dec.decrypt_raw(&ct));
    }
}
