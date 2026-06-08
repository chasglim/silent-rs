//! GGSW ciphertext containers used as the bootstrap-key building block.
//!
//! A GGSW ciphertext encrypting a polynomial `M` is, conceptually, a list of
//! `(k+1)` "rows" of `level` GLWE encryptions: row `j ∈ 0..k` encrypts
//! `-S_j · M · 2^(log_q - (i+1)*base_log)` for `i ∈ 0..level`, and row `k`
//! encrypts `M · 2^(log_q - (i+1)*base_log)`.  The CMUX gate consumed by the
//! TFHE blind rotation operates on this representation directly.
//!
//! The container only carries shape and storage; encryption belongs in
//! `silent-fhe::schemes::tfhe::keys`.

use silent_params::{
    CiphertextModulusLog, DecompositionBaseLog, DecompositionLevelCount, GlweDimension,
    PolynomialSize,
};

use crate::glwe::GlweCiphertext;

/// One GGSW level matrix: `(k+1)` GLWE ciphertexts at decomposition level `i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgswLevelMatrix {
    rows: Vec<GlweCiphertext>,
}

impl GgswLevelMatrix {
    pub fn from_rows(rows: Vec<GlweCiphertext>) -> Self {
        assert!(!rows.is_empty(), "level matrix needs at least one row");
        Self { rows }
    }

    pub fn rows(&self) -> &[GlweCiphertext] {
        &self.rows
    }

    pub fn rows_mut(&mut self) -> &mut [GlweCiphertext] {
        &mut self.rows
    }
}

/// Full GGSW ciphertext: `level` level matrices, each holding `(k+1)` rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgswCiphertext {
    levels: Vec<GgswLevelMatrix>,
    base_log: u8,
    log_modulus: u8,
}

impl GgswCiphertext {
    pub fn zeros(
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
        base_log: DecompositionBaseLog,
        level: DecompositionLevelCount,
        log_modulus: CiphertextModulusLog,
    ) -> Self {
        let row_count = glwe_dimension.glwe_size();
        let levels = (0..level.0)
            .map(|_| {
                let rows = (0..row_count)
                    .map(|_| GlweCiphertext::zeros(glwe_dimension, polynomial_size, log_modulus))
                    .collect();
                GgswLevelMatrix { rows }
            })
            .collect();
        Self {
            levels,
            base_log: base_log.0,
            log_modulus: log_modulus.0,
        }
    }

    pub fn level_count(&self) -> DecompositionLevelCount {
        DecompositionLevelCount(self.levels.len() as u8)
    }

    pub fn base_log(&self) -> DecompositionBaseLog {
        DecompositionBaseLog(self.base_log)
    }

    pub fn glwe_dimension(&self) -> GlweDimension {
        self.levels[0].rows[0].glwe_dimension()
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        self.levels[0].rows[0].polynomial_size()
    }

    pub fn log_modulus(&self) -> u8 {
        self.log_modulus
    }

    pub fn levels(&self) -> &[GgswLevelMatrix] {
        &self.levels
    }

    pub fn levels_mut(&mut self) -> &mut [GgswLevelMatrix] {
        &mut self.levels
    }
}

/// Ordered list of GGSW ciphertexts.  Serves as the storage shape of the
/// LWE bootstrap key (`bsk[i] = GGSW(s_i)` for the input LWE secret).
#[derive(Clone, Debug)]
pub struct GgswCiphertextList {
    inner: Vec<GgswCiphertext>,
}

impl GgswCiphertextList {
    pub fn new(items: Vec<GgswCiphertext>) -> Self {
        Self { inner: items }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GgswCiphertext> {
        self.inner.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut GgswCiphertext> {
        self.inner.iter_mut()
    }

    pub fn get(&self, index: usize) -> &GgswCiphertext {
        &self.inner[index]
    }

    pub fn get_mut(&mut self, index: usize) -> &mut GgswCiphertext {
        &mut self.inner[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cm(log: u8) -> CiphertextModulusLog {
        CiphertextModulusLog(log)
    }

    #[test]
    fn ggsw_zero_layout() {
        let g = GgswCiphertext::zeros(
            GlweDimension(2),
            PolynomialSize(8),
            DecompositionBaseLog(8),
            DecompositionLevelCount(3),
            cm(64),
        );
        assert_eq!(g.level_count().0, 3);
        assert_eq!(g.base_log().0, 8);
        assert_eq!(g.glwe_dimension().0, 2);
        assert_eq!(g.polynomial_size().0, 8);
        for level in g.levels() {
            assert_eq!(level.rows().len(), 3); // glwe_size = k + 1 = 3
            for row in level.rows() {
                assert_eq!(row.polys().len(), 3);
                for p in row.polys() {
                    assert_eq!(p.degree(), 8);
                    assert!(p.coeffs().iter().all(|&c| c == 0));
                }
            }
        }
    }

    #[test]
    fn ggsw_list_iteration() {
        let g1 = GgswCiphertext::zeros(
            GlweDimension(1),
            PolynomialSize(4),
            DecompositionBaseLog(4),
            DecompositionLevelCount(2),
            cm(64),
        );
        let g2 = g1.clone();
        let list = GgswCiphertextList::new(vec![g1, g2]);
        assert_eq!(list.len(), 2);
        assert_eq!(list.iter().count(), 2);
    }
}
