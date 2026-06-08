//! TFHE-side LUT operators exposed through the bridge.
//!
//! Each operator is a programmable bootstrap (PBS) with a specific lookup
//! table.  The bridge provides convenience wrappers so callers do not need
//! to construct `LweBootstrapKey` / `LweKeyswitchKey` manually.

use silent_rlwe::LweCiphertext;

use crate::schemes::tfhe::bootstrap::TfheBootstrapper;
use crate::schemes::tfhe::encoding::TfheEncoder;
use crate::schemes::tfhe::params::TfheParameters;

/// Bridge-side LUT specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lut {
    Identity,
    Sign,
    ReLU,
}

impl Lut {
    /// Apply the LUT to a single TFHE LWE ciphertext.
    pub fn apply(
        &self,
        ct: &LweCiphertext,
        params: &TfheParameters,
        encoder: &TfheEncoder,
        bootstrapper: &TfheBootstrapper,
    ) -> LweCiphertext {
        match self {
            Lut::Identity => bootstrapper.apply_lookup_table(encoder, ct, |x| x),
            Lut::Sign => {
                // total = message_modulus * carry_modulus
                let total = params.total_message_modulus();
                let half = total / 2;
                bootstrapper.apply_lookup_table(encoder, ct, |x| {
                    if x >= half {
                        // Negative interval → -1 mapped to total-1
                        (total - 1) % total
                    } else if x == 0 {
                        0
                    } else {
                        1
                    }
                })
            }
            Lut::ReLU => {
                let total = params.total_message_modulus();
                let half = total / 2;
                bootstrapper.apply_lookup_table(encoder, ct, |x| {
                    if x >= half {
                        // Negative → 0
                        0
                    } else {
                        x % total
                    }
                })
            }
        }
    }
}

/// Evaluate a [`Lut`] on a vector of TFHE LWE ciphertexts.
pub fn eval_lut(
    cts: &[LweCiphertext],
    lut: Lut,
    params: &TfheParameters,
    encoder: &TfheEncoder,
    bootstrapper: &TfheBootstrapper,
) -> Vec<LweCiphertext> {
    cts.iter()
        .map(|ct| lut.apply(ct, params, encoder, bootstrapper))
        .collect()
}
