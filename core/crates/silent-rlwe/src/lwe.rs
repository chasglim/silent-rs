//! TFHE-style LWE containers over a native power-of-two ciphertext modulus.
//!
//! These types are intentionally storage-only: TFHE-specific encryption,
//! decryption and key-switch logic live in `silent-fhe::schemes::tfhe`.
//! Keeping the containers here lets multiple higher-level crates share the
//! same shape (e.g. a future protocol LWE-aggregation operator) without each
//! one redefining its own polymorphic record.

use silent_params::{CiphertextModulusLog, LweDimension};

/// LWE ciphertext: `(a_0, ..., a_{n-1}, b)` where `b = ⟨a, s⟩ + Δ·m + e` and
/// `n = lwe_dimension`.
///
/// The body sits at the *last* index, mirroring tfhe-rs's storage layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LweCiphertext {
    /// Length is `lwe_dimension + 1`; mask is `data[..n]`, body is `data[n]`.
    data: Vec<u64>,
    log_modulus: u8,
}

impl LweCiphertext {
    /// Construct a freshly-zeroed ciphertext.
    pub fn zeros(dimension: LweDimension, log_modulus: CiphertextModulusLog) -> Self {
        Self {
            data: vec![0u64; dimension.lwe_size()],
            log_modulus: log_modulus.0,
        }
    }

    /// Build directly from a flat container.
    pub fn from_data(data: Vec<u64>, log_modulus: CiphertextModulusLog) -> Self {
        assert!(data.len() >= 1, "LWE ciphertext must contain a body");
        Self {
            data,
            log_modulus: log_modulus.0,
        }
    }

    /// LWE secret-key dimension `n`.
    pub fn dimension(&self) -> LweDimension {
        LweDimension(self.data.len() - 1)
    }

    /// Base-2 log of the ciphertext modulus.
    pub fn log_modulus(&self) -> u8 {
        self.log_modulus
    }

    /// `a` part (length `n`).
    pub fn mask(&self) -> &[u64] {
        &self.data[..self.dimension().0]
    }

    /// `a` part, mutable.
    pub fn mask_mut(&mut self) -> &mut [u64] {
        let n = self.dimension().0;
        &mut self.data[..n]
    }

    /// Body coefficient `b`.
    pub fn body(&self) -> u64 {
        *self.data.last().expect("body always present")
    }

    /// Body coefficient `b`, mutable.
    pub fn body_mut(&mut self) -> &mut u64 {
        self.data.last_mut().expect("body always present")
    }

    /// Mask + body as a single contiguous slice (mask first, body last).
    pub fn as_slice(&self) -> &[u64] {
        &self.data
    }

    /// Mask + body as a single contiguous mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        &mut self.data
    }

    /// Add another LWE ciphertext (same dimension and modulus).  Wrapping is
    /// the modular reduction since the modulus is power-of-two.
    pub fn add_assign(&mut self, other: &LweCiphertext) {
        debug_assert_eq!(self.log_modulus, other.log_modulus);
        debug_assert_eq!(self.data.len(), other.data.len());
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = a.wrapping_add(*b);
        }
        self.reduce();
    }

    /// Subtract another LWE ciphertext.
    pub fn sub_assign(&mut self, other: &LweCiphertext) {
        debug_assert_eq!(self.log_modulus, other.log_modulus);
        debug_assert_eq!(self.data.len(), other.data.len());
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = a.wrapping_sub(*b);
        }
        self.reduce();
    }

    /// Negate every entry.
    pub fn negate_assign(&mut self) {
        for a in self.data.iter_mut() {
            *a = a.wrapping_neg();
        }
        self.reduce();
    }

    /// Multiply every entry by a (wrapping) scalar.
    pub fn scalar_mul_assign(&mut self, scalar: u64) {
        for a in self.data.iter_mut() {
            *a = a.wrapping_mul(scalar);
        }
        self.reduce();
    }

    /// Add a scalar to the body — useful for adding plaintext constants.
    pub fn body_add_assign(&mut self, value: u64) {
        let body = self.body_mut();
        *body = body.wrapping_add(value);
        self.reduce_body();
    }

    /// Mask everything back into `[0, 2^log_modulus)`.  No-op for q = 2^64.
    pub fn reduce(&mut self) {
        if self.log_modulus == 64 {
            return;
        }
        let mask = (1u64 << self.log_modulus) - 1;
        for a in self.data.iter_mut() {
            *a &= mask;
        }
    }

    fn reduce_body(&mut self) {
        if self.log_modulus == 64 {
            return;
        }
        let mask = (1u64 << self.log_modulus) - 1;
        let body = self.body_mut();
        *body &= mask;
    }
}

/// LWE secret key: a vector of secret integers (typically uniform binary or
/// uniform ternary).  We store them in `Vec<u64>` regardless of the chosen
/// distribution; sampling is the producer's responsibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LweSecretKey {
    data: Vec<u64>,
    log_modulus: u8,
}

impl LweSecretKey {
    pub fn from_data(data: Vec<u64>, log_modulus: CiphertextModulusLog) -> Self {
        Self {
            data,
            log_modulus: log_modulus.0,
        }
    }

    pub fn dimension(&self) -> LweDimension {
        LweDimension(self.data.len())
    }

    pub fn log_modulus(&self) -> u8 {
        self.log_modulus
    }

    pub fn data(&self) -> &[u64] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [u64] {
        &mut self.data
    }
}

/// LWE public key: a list of `count` independent LWE encryptions of `0`
/// under the same secret key, all in the canonical mask-first body-last
/// storage layout.
///
/// Encryption with this public key uses the standard "binary subset-sum"
/// trick: pick `r ∈ {0, 1}^count` uniformly at random and compute
///
/// ```text
/// Enc(μ) = Σᵢ rᵢ · zero_iᵉ + (0, …, 0, μ̃)
/// ```
///
/// where `μ̃` is the encoded plaintext and `zero_iᵉ` is the i-th zero
/// encryption.  Increasing `count` gives more randomness per ciphertext at
/// the cost of a larger public key.
#[derive(Clone, Debug)]
pub struct LwePublicKey {
    /// `count` LWE ciphertexts, each of length `dimension + 1`.
    ciphertexts: Vec<LweCiphertext>,
    dimension: LweDimension,
    log_modulus: u8,
}

impl LwePublicKey {
    /// Build a public key from a pre-computed list of LWE encryptions of `0`.
    pub fn from_zero_encryptions(
        ciphertexts: Vec<LweCiphertext>,
        dimension: LweDimension,
        log_modulus: CiphertextModulusLog,
    ) -> Self {
        for ct in &ciphertexts {
            debug_assert_eq!(ct.dimension(), dimension);
            debug_assert_eq!(ct.log_modulus(), log_modulus.0);
        }
        Self {
            ciphertexts,
            dimension,
            log_modulus: log_modulus.0,
        }
    }

    pub fn dimension(&self) -> LweDimension {
        self.dimension
    }

    pub fn log_modulus(&self) -> u8 {
        self.log_modulus
    }

    /// Number of stored zero encryptions.
    pub fn count(&self) -> usize {
        self.ciphertexts.len()
    }

    pub fn ciphertexts(&self) -> &[LweCiphertext] {
        &self.ciphertexts
    }

    pub fn ciphertext(&self, index: usize) -> &LweCiphertext {
        &self.ciphertexts[index]
    }
}

/// LWE→LWE key-switch key: for each input secret coordinate `s_in[i]`,
/// `level` LWE encryptions of `s_in[i] · 2^(log_q - (j+1)·base_log)` under
/// `s_out`.
#[derive(Clone, Debug)]
pub struct LweKeyswitchKey {
    /// `input_dimension * level` ciphertexts, all of size `output_dimension+1`.
    ciphertexts: Vec<LweCiphertext>,
    input_dimension: LweDimension,
    output_dimension: LweDimension,
    base_log: u8,
    level: u8,
    log_modulus: u8,
}

impl LweKeyswitchKey {
    pub fn new(
        input_dimension: LweDimension,
        output_dimension: LweDimension,
        base_log: u8,
        level: u8,
        log_modulus: CiphertextModulusLog,
    ) -> Self {
        let count = input_dimension.0 * level as usize;
        let ciphertexts = (0..count)
            .map(|_| LweCiphertext::zeros(output_dimension, log_modulus))
            .collect();
        Self {
            ciphertexts,
            input_dimension,
            output_dimension,
            base_log,
            level,
            log_modulus: log_modulus.0,
        }
    }

    pub fn input_dimension(&self) -> LweDimension {
        self.input_dimension
    }

    pub fn output_dimension(&self) -> LweDimension {
        self.output_dimension
    }

    pub fn base_log(&self) -> u8 {
        self.base_log
    }

    pub fn level(&self) -> u8 {
        self.level
    }

    pub fn log_modulus(&self) -> u8 {
        self.log_modulus
    }

    /// Returns the `level` LWE encryptions associated with the `input`-th
    /// coordinate of the source secret key.
    pub fn block(&self, input: usize) -> &[LweCiphertext] {
        let start = input * self.level as usize;
        let end = start + self.level as usize;
        &self.ciphertexts[start..end]
    }

    /// Mutable variant of [`LweKeyswitchKey::block`].
    pub fn block_mut(&mut self, input: usize) -> &mut [LweCiphertext] {
        let start = input * self.level as usize;
        let end = start + self.level as usize;
        &mut self.ciphertexts[start..end]
    }

    pub fn iter_blocks(&self) -> impl Iterator<Item = &[LweCiphertext]> {
        (0..self.input_dimension.0).map(|i| self.block(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cm(log: u8) -> CiphertextModulusLog {
        CiphertextModulusLog(log)
    }

    #[test]
    fn lwe_basic_layout() {
        let mut ct = LweCiphertext::zeros(LweDimension(3), cm(64));
        ct.mask_mut().copy_from_slice(&[10, 20, 30]);
        *ct.body_mut() = 99;
        assert_eq!(ct.dimension().0, 3);
        assert_eq!(ct.mask(), &[10, 20, 30]);
        assert_eq!(ct.body(), 99);
        assert_eq!(ct.as_slice(), &[10, 20, 30, 99]);
    }

    #[test]
    fn lwe_arithmetic_q32_wraps() {
        let mut a = LweCiphertext::from_data(vec![u32::MAX as u64, 0, 1], cm(32));
        let b = LweCiphertext::from_data(vec![1, 0, 1], cm(32));
        a.add_assign(&b);
        assert_eq!(a.as_slice(), &[0, 0, 2]); // wraps within 32-bit modulus
        a.negate_assign();
        // -[0, 0, 2] mod 2^32 = [0, 0, 2^32-2]
        assert_eq!(a.as_slice(), &[0, 0, (1u64 << 32) - 2]);
    }

    #[test]
    fn pubkey_records_count_and_shape() {
        let zeros: Vec<_> = (0..5)
            .map(|_| LweCiphertext::zeros(LweDimension(7), cm(64)))
            .collect();
        let pk = LwePublicKey::from_zero_encryptions(zeros, LweDimension(7), cm(64));
        assert_eq!(pk.count(), 5);
        assert_eq!(pk.dimension().0, 7);
        assert_eq!(pk.log_modulus(), 64);
        assert_eq!(pk.ciphertexts().len(), 5);
        assert_eq!(pk.ciphertext(2).dimension().0, 7);
    }

    #[test]
    fn ksk_block_layout() {
        let ksk = LweKeyswitchKey::new(LweDimension(4), LweDimension(2), 8, 3, cm(64));
        assert_eq!(ksk.iter_blocks().count(), 4);
        for block in ksk.iter_blocks() {
            assert_eq!(block.len(), 3);
            for ct in block {
                assert_eq!(ct.dimension().0, 2);
            }
        }
    }
}
