//! Semantic wrapper around [`crate::ggsw::GgswCiphertextList`] dedicated to
//! the LWE bootstrap-key role.
//!
//! TFHE's `LweBootstrapKey` is structurally a list of GGSW ciphertexts — one
//! per coordinate of the input LWE secret key — that encrypt the secret bits
//! under a *different* GLWE key.  The wrapper here only enforces the
//! invariants the blind-rotation step relies on (matching dimensions etc.).

use silent_params::{
    CiphertextModulusLog, DecompositionBaseLog, DecompositionLevelCount, GlweDimension,
    LweDimension, PolynomialSize,
};

use crate::ggsw::{GgswCiphertext, GgswCiphertextList};

/// Bootstrap key used by TFHE's PBS.
///
/// Wraps a [`GgswCiphertextList`] together with a small descriptor of the
/// input LWE dimension and the decomposition shape.  The list contains
/// exactly `input_lwe_dimension` GGSW ciphertexts.
#[derive(Clone, Debug)]
pub struct LweBootstrapKey {
    list: GgswCiphertextList,
    input_lwe_dimension: LweDimension,
    glwe_dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    log_modulus: u8,
}

impl LweBootstrapKey {
    pub fn new(
        list: GgswCiphertextList,
        input_lwe_dimension: LweDimension,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
        base_log: DecompositionBaseLog,
        level: DecompositionLevelCount,
        log_modulus: CiphertextModulusLog,
    ) -> Self {
        assert_eq!(
            list.len(),
            input_lwe_dimension.0,
            "bootstrap key must hold one GGSW per input LWE coordinate"
        );
        for ggsw in list.iter() {
            assert_eq!(ggsw.glwe_dimension(), glwe_dimension);
            assert_eq!(ggsw.polynomial_size(), polynomial_size);
            assert_eq!(ggsw.base_log(), base_log);
            assert_eq!(ggsw.level_count(), level);
            assert_eq!(ggsw.log_modulus(), log_modulus.0);
        }
        Self {
            list,
            input_lwe_dimension,
            glwe_dimension,
            polynomial_size,
            base_log,
            level,
            log_modulus: log_modulus.0,
        }
    }

    pub fn input_lwe_dimension(&self) -> LweDimension {
        self.input_lwe_dimension
    }

    pub fn glwe_dimension(&self) -> GlweDimension {
        self.glwe_dimension
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        self.polynomial_size
    }

    pub fn base_log(&self) -> DecompositionBaseLog {
        self.base_log
    }

    pub fn level(&self) -> DecompositionLevelCount {
        self.level
    }

    pub fn log_modulus(&self) -> u8 {
        self.log_modulus
    }

    pub fn list(&self) -> &GgswCiphertextList {
        &self.list
    }

    pub fn list_mut(&mut self) -> &mut GgswCiphertextList {
        &mut self.list
    }

    pub fn get(&self, input_index: usize) -> &GgswCiphertext {
        self.list.get(input_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cm(log: u8) -> CiphertextModulusLog {
        CiphertextModulusLog(log)
    }

    #[test]
    fn bootstrap_key_validates_shape() {
        let g = GgswCiphertext::zeros(
            GlweDimension(1),
            PolynomialSize(8),
            DecompositionBaseLog(8),
            DecompositionLevelCount(2),
            cm(64),
        );
        let list = GgswCiphertextList::new(vec![g.clone(), g.clone(), g]);
        let bsk = LweBootstrapKey::new(
            list,
            LweDimension(3),
            GlweDimension(1),
            PolynomialSize(8),
            DecompositionBaseLog(8),
            DecompositionLevelCount(2),
            cm(64),
        );
        assert_eq!(bsk.input_lwe_dimension().0, 3);
        assert_eq!(bsk.glwe_dimension().0, 1);
        assert_eq!(bsk.polynomial_size().0, 8);
    }
}
