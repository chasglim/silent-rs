//! Shortint server key: homomorphic arithmetic and programmable bootstrapping.

use std::sync::Arc;

use silent_math::fft64::{with_fft_mul, with_torus_fft};
use silent_params::CiphertextModulusLog;
use silent_ring::NativePoly;
use silent_rlwe::{LweBootstrapKey, LweCiphertext, LweKeyswitchKey};

use super::ciphertext::ShortintCiphertext;
use super::error::ShortintError;
use super::types::{CiphertextNoiseDegree, Degree, MaxDegree, MaxNoiseLevel, NoiseLevel};
use crate::schemes::tfhe::bootstrap::{TfheBootstrapper, build_lookup_table_polynomial};
use crate::schemes::tfhe::bootstrap_fft::{LweBootstrapKeyFft, LweBootstrapKeyTorusFft};
use crate::schemes::tfhe::encoding::TfheEncoder;
use crate::schemes::tfhe::params::TfheParameters;

/// Pre-computed programmable-bootstrap polynomial for a packed bivariate LUT.
///
/// The server evaluates `f(left, right)` by packing `left · p + right` into one
/// shortint slot (with `p = pack_factor`, typically `message_modulus`), then
/// running a single PBS with the univariate table stored here.
pub struct BivariateLookupTable {
    pack_factor: u64,
    lut: Arc<NativePoly>,
}

impl BivariateLookupTable {
    #[inline]
    pub fn pack_factor(&self) -> u64 {
        self.pack_factor
    }
}

/// Cached frequency-domain bootstrap key, selected based on `ciphertext_modulus_log`.
///
/// - `Torus64`: compact N/2 Torus-Fourier backend — only valid for `q = 2^64`.
/// - `Exact`: split-precision 4-bucket backend — works for any supported `q`.
#[derive(Clone)]
enum PbsCache {
    /// Compact Torus-Fourier cache (only valid for `ciphertext_modulus_log == 64`).
    Torus64(LweBootstrapKeyTorusFft),
    /// Exact split-precision cache (works for any `log_modulus`).
    Exact(LweBootstrapKeyFft),
}

/// Server key material: bootstrap + keyswitch, matching [`ShortintClientKey`] parameters.
#[derive(Clone)]
pub struct ShortintServerKey {
    params: TfheParameters,
    bsk: Arc<LweBootstrapKey>,
    ksk: Arc<LweKeyswitchKey>,
    pbs_cache: Arc<PbsCache>,
    encoder: TfheEncoder,
    max_degree: MaxDegree,
    max_noise: MaxNoiseLevel,
}

impl ShortintServerKey {
    /// Validate noise bounds and construct the server key.
    pub fn try_new(
        params: TfheParameters,
        bsk: LweBootstrapKey,
        ksk: LweKeyswitchKey,
        encoder: TfheEncoder,
    ) -> Result<Self, ShortintError> {
        let mm = params.message_modulus().0;
        let cm = params.carry_modulus().0;
        if mm < 2 {
            return Err(ShortintError::MessageModulusTooSmall);
        }
        let bsk = Arc::new(bsk);
        let ksk = Arc::new(ksk);
        let n = params.polynomial_size().0;
        let log_q = params.ciphertext_modulus_log().0;
        let bs = TfheBootstrapper::new(params.clone(), bsk.as_ref(), ksk.as_ref());
        let pbs_cache = if log_q == 64 {
            // Torus-Fourier backend: compact N/2 representation, ~30% faster.
            // Only valid when the torus scale is exactly 2^64.
            PbsCache::Torus64(with_torus_fft(n, |fft| bs.build_torus_fft_bsk_cache(fft)))
        } else {
            // Exact split-precision backend: works for any q = 2^k.
            PbsCache::Exact(with_fft_mul(n, |fft| bs.build_fft_bsk_cache(fft)))
        };
        Ok(Self {
            params,
            bsk,
            ksk,
            pbs_cache: Arc::new(pbs_cache),
            encoder,
            max_degree: MaxDegree::from_msg_carry(mm, cm),
            max_noise: MaxNoiseLevel::from_msg_carry(mm, cm)?,
        })
    }

    #[inline]
    pub fn parameters(&self) -> &TfheParameters {
        &self.params
    }

    #[inline]
    pub fn encoder(&self) -> &TfheEncoder {
        &self.encoder
    }

    #[inline]
    fn apply_lookup_table_poly_cached(
        &self,
        ct: &LweCiphertext,
        lut: &NativePoly,
    ) -> LweCiphertext {
        let bs = TfheBootstrapper::new(self.params.clone(), self.bsk.as_ref(), self.ksk.as_ref());
        let n = self.params.polynomial_size().0;
        match self.pbs_cache.as_ref() {
            PbsCache::Torus64(cache) => with_torus_fft(n, |fft| {
                bs.apply_lookup_table_torus_fft_with_polynomial(cache, fft, ct, lut)
            }),
            PbsCache::Exact(cache) => with_fft_mul(n, |fft| {
                bs.apply_lookup_table_fft_with_polynomial(cache, fft, ct, lut)
            }),
        }
    }

    /// Build a bivariate lookup table with `pack_factor = message_modulus` (left
    /// operand scaled then added to the right before a single PBS).
    pub fn generate_lookup_table_bivariate<F>(&self, f: F) -> BivariateLookupTable
    where
        F: FnMut(u64, u64) -> u64,
    {
        self.generate_lookup_table_bivariate_with_factor(f, self.params.message_modulus().0)
    }

    /// Build a bivariate LUT with a custom left-hand packing radix (advanced).
    pub fn generate_lookup_table_bivariate_with_factor<F>(
        &self,
        mut f: F,
        pack_factor: u64,
    ) -> BivariateLookupTable
    where
        F: FnMut(u64, u64) -> u64,
    {
        let mm = self.params.message_modulus().0;
        let total = self.params.total_message_modulus();
        let factor = pack_factor;
        let mut wrapped = move |packed: u64| {
            let lhs = (packed / factor) % mm;
            let rhs = (packed % factor) % mm;
            f(lhs, rhs) % total
        };
        let poly = build_lookup_table_polynomial(
            &self.encoder,
            self.params.polynomial_size(),
            self.params.ciphertext_modulus_log(),
            &mut wrapped,
        );
        BivariateLookupTable {
            pack_factor: factor,
            lut: Arc::new(poly),
        }
    }

    /// Univariate PBS LUT for `f: ℤ_total → ℤ_total` (one shortint input).
    pub fn generate_lookup_table<F>(&self, mut f: F) -> Arc<NativePoly>
    where
        F: FnMut(u64) -> u64,
    {
        let poly = build_lookup_table_polynomial(
            &self.encoder,
            self.params.polynomial_size(),
            self.params.ciphertext_modulus_log(),
            &mut f,
        );
        Arc::new(poly)
    }

    fn is_bivariate_pbs_possible(
        &self,
        lhs: CiphertextNoiseDegree,
        rhs: CiphertextNoiseDegree,
        factor: u64,
    ) -> Result<(), ShortintError> {
        if rhs.degree.get() >= factor {
            return Err(ShortintError::UnscaledScaledOverlap(rhs.degree, factor));
        }
        let final_d = lhs
            .degree
            .saturating_mul_scalar(factor)
            .saturating_add(rhs.degree);
        let final_n = lhs
            .noise
            .saturating_mul_scalar(factor)
            .saturating_add(rhs.noise);
        self.max_degree.validate(final_d)?;
        self.max_noise.validate(final_n)?;
        Ok(())
    }

    /// Homomorphic addition with automatic carry management (refresh PBS when the
    /// carry slot is non-zero).
    pub fn add(
        &self,
        ct_left: &ShortintCiphertext,
        ct_right: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let mut out = ct_left.clone();
        self.add_assign(&mut out, ct_right)?;
        Ok(out)
    }

    pub fn add_assign(
        &self,
        ct_left: &mut ShortintCiphertext,
        ct_right: &ShortintCiphertext,
    ) -> Result<(), ShortintError> {
        ct_left.ensure_params(&self.params)?;
        ct_right.ensure_params(&self.params)?;
        if !ct_left.carry_is_empty() {
            self.message_extract_assign(ct_left)?;
        }
        let mut tmp;
        let rhs: &ShortintCiphertext = if ct_right.carry_is_empty() {
            ct_right
        } else {
            tmp = ct_right.clone();
            self.message_extract_assign(&mut tmp)?;
            &tmp
        };
        unchecked_add_assign(ct_left, rhs, self.max_degree, self.max_noise)?;
        self.message_extract_assign(ct_left)?;
        Ok(())
    }

    pub fn unchecked_add(
        &self,
        ct_left: &ShortintCiphertext,
        ct_right: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let mut out = ct_left.clone();
        self.unchecked_add_assign(&mut out, ct_right)?;
        Ok(out)
    }

    pub fn unchecked_add_assign(
        &self,
        ct_left: &mut ShortintCiphertext,
        ct_right: &ShortintCiphertext,
    ) -> Result<(), ShortintError> {
        ct_left.ensure_params(&self.params)?;
        ct_right.ensure_params(&self.params)?;
        ct_left.ensure_compatible(ct_right)?;
        unchecked_add_assign(ct_left, ct_right, self.max_degree, self.max_noise)
    }

    pub fn sub(
        &self,
        ct_left: &ShortintCiphertext,
        ct_right: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let mut out = ct_left.clone();
        self.sub_assign(&mut out, ct_right)?;
        Ok(out)
    }

    pub fn sub_assign(
        &self,
        ct_left: &mut ShortintCiphertext,
        ct_right: &ShortintCiphertext,
    ) -> Result<(), ShortintError> {
        ct_left.ensure_params(&self.params)?;
        ct_right.ensure_params(&self.params)?;
        if !ct_left.carry_is_empty() {
            self.message_extract_assign(ct_left)?;
        }
        let mut tmp;
        let rhs: &ShortintCiphertext = if ct_right.carry_is_empty() {
            ct_right
        } else {
            tmp = ct_right.clone();
            self.message_extract_assign(&mut tmp)?;
            &tmp
        };
        let mut neg = rhs.clone();
        neg.inner_mut().negate_assign();
        unchecked_add_assign(ct_left, &neg, self.max_degree, self.max_noise)?;
        self.message_extract_assign(ct_left)?;
        Ok(())
    }

    pub fn unchecked_sub_assign(
        &self,
        ct_left: &mut ShortintCiphertext,
        ct_right: &ShortintCiphertext,
    ) -> Result<(), ShortintError> {
        ct_left.ensure_params(&self.params)?;
        ct_right.ensure_params(&self.params)?;
        ct_left.ensure_compatible(ct_right)?;
        let mut neg = ct_right.clone();
        neg.inner_mut().negate_assign();
        unchecked_add_assign(ct_left, &neg, self.max_degree, self.max_noise)
    }

    /// Scalar multiplication `ct · k` (exact integer factor, wrapping LWE limbs).
    pub fn scalar_mul(
        &self,
        ct: &ShortintCiphertext,
        scalar: u64,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let mut out = ct.clone();
        self.scalar_mul_assign(&mut out, scalar)?;
        Ok(out)
    }

    pub fn scalar_mul_assign(
        &self,
        ct: &mut ShortintCiphertext,
        scalar: u64,
    ) -> Result<(), ShortintError> {
        ct.ensure_params(&self.params)?;
        if !ct.carry_is_empty() {
            self.message_extract_assign(ct)?;
        }
        unchecked_scalar_mul_assign(ct, scalar, self.max_degree, self.max_noise)?;
        self.message_extract_assign(ct)?;
        Ok(())
    }

    pub fn unchecked_scalar_mul_assign(
        &self,
        ct: &mut ShortintCiphertext,
        scalar: u64,
    ) -> Result<(), ShortintError> {
        ct.ensure_params(&self.params)?;
        unchecked_scalar_mul_assign(ct, scalar, self.max_degree, self.max_noise)
    }

    /// PBS that clears the carry slot, leaving a clean message encoding.
    pub fn message_extract(
        &self,
        ct: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let mut out = ct.clone();
        self.message_extract_assign(&mut out)?;
        Ok(out)
    }

    pub fn message_extract_assign(&self, ct: &mut ShortintCiphertext) -> Result<(), ShortintError> {
        ct.ensure_params(&self.params)?;
        if ct.carry_is_empty() {
            return Ok(());
        }
        let mm = self.params.message_modulus().0;
        let poly = build_lookup_table_polynomial(
            &self.encoder,
            self.params.polynomial_size(),
            self.params.ciphertext_modulus_log(),
            move |m: u64| m % mm,
        );
        let new_inner = self.apply_lookup_table_poly_cached(ct.lwe_ciphertext(), &poly);
        *ct.inner_mut() = new_inner;
        ct.set_degree_noise(Degree::new(mm.saturating_sub(1)), NoiseLevel::NOMINAL);
        Ok(())
    }

    /// Apply a univariate PBS using a pre-built [`NativePoly`] LUT.
    pub fn apply_lookup_table(
        &self,
        ct: &ShortintCiphertext,
        lut: &NativePoly,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let mut out = ct.clone();
        self.apply_lookup_table_assign(&mut out, lut)?;
        Ok(out)
    }

    pub fn apply_lookup_table_assign(
        &self,
        ct: &mut ShortintCiphertext,
        lut: &NativePoly,
    ) -> Result<(), ShortintError> {
        ct.ensure_params(&self.params)?;
        let new_inner = self.apply_lookup_table_poly_cached(ct.lwe_ciphertext(), lut);
        *ct.inner_mut() = new_inner;
        let mm = self.params.message_modulus().0;
        ct.set_degree_noise(Degree::new(mm.saturating_sub(1)), NoiseLevel::NOMINAL);
        Ok(())
    }

    pub fn unchecked_apply_lookup_table_bivariate(
        &self,
        ct_left: &ShortintCiphertext,
        ct_right: &ShortintCiphertext,
        table: &BivariateLookupTable,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let mut out = ct_left.clone();
        self.unchecked_apply_lookup_table_bivariate_assign(&mut out, ct_right, table)?;
        Ok(out)
    }

    pub fn unchecked_apply_lookup_table_bivariate_assign(
        &self,
        ct_left: &mut ShortintCiphertext,
        ct_right: &ShortintCiphertext,
        table: &BivariateLookupTable,
    ) -> Result<(), ShortintError> {
        ct_left.ensure_params(&self.params)?;
        ct_right.ensure_params(&self.params)?;
        ct_left.ensure_compatible(ct_right)?;
        if table.pack_factor() != self.params.message_modulus().0 {
            return Err(ShortintError::BivariatePbsPrecondition(
                "BivariateLookupTable.pack_factor must equal parameter message_modulus (use generate_lookup_table_bivariate)",
            ));
        }
        let factor = table.pack_factor();
        self.is_bivariate_pbs_possible(ct_left.noise_degree(), ct_right.noise_degree(), factor)?;
        unchecked_scalar_mul_assign(ct_left, factor, self.max_degree, self.max_noise)?;
        let rd = ct_right.degree();
        let rn = ct_right.noise_level();
        ct_left.inner_mut().add_assign(ct_right.lwe_ciphertext());
        let new_d = ct_left.degree().saturating_add(rd);
        let new_n = ct_left.noise_level().saturating_add(rn);
        self.max_degree.validate(new_d)?;
        self.max_noise.validate(new_n)?;
        ct_left.set_degree_noise(new_d, new_n);

        let new_inner =
            self.apply_lookup_table_poly_cached(ct_left.lwe_ciphertext(), table.lut.as_ref());
        *ct_left.inner_mut() = new_inner;
        let mm = self.params.message_modulus().0;
        ct_left.set_degree_noise(Degree::new(mm.saturating_sub(1)), NoiseLevel::NOMINAL);
        Ok(())
    }

    pub fn apply_lookup_table_bivariate(
        &self,
        ct_left: &ShortintCiphertext,
        ct_right: &ShortintCiphertext,
        table: &BivariateLookupTable,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let mut left = ct_left.clone();
        self.apply_lookup_table_bivariate_assign(&mut left, ct_right, table)?;
        Ok(left)
    }

    pub fn apply_lookup_table_bivariate_assign(
        &self,
        ct_left: &mut ShortintCiphertext,
        ct_right: &ShortintCiphertext,
        table: &BivariateLookupTable,
    ) -> Result<(), ShortintError> {
        if table.pack_factor() != self.params.message_modulus().0 {
            return Err(ShortintError::BivariatePbsPrecondition(
                "pack_factor mismatch with message_modulus",
            ));
        }
        let factor = table.pack_factor();
        let mut tmp_r;
        let right_ref: &ShortintCiphertext = if self
            .is_bivariate_pbs_possible(ct_left.noise_degree(), ct_right.noise_degree(), factor)
            .is_err()
        {
            self.message_extract_assign(ct_left)?;
            tmp_r = ct_right.clone();
            self.message_extract_assign(&mut tmp_r)?;
            &tmp_r
        } else {
            ct_right
        };
        self.is_bivariate_pbs_possible(ct_left.noise_degree(), right_ref.noise_degree(), factor)?;
        self.unchecked_apply_lookup_table_bivariate_assign(ct_left, right_ref, table)?;
        Ok(())
    }

    // --- Boolean / comparison helpers (message space must support the output) ---

    /// Boolean AND for inputs restricted to `{0, 1}`: returns `(a·b) mod 2`.
    pub fn bitand(
        &self,
        ct_a: &ShortintCiphertext,
        ct_b: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let mm = self.params.message_modulus().0;
        let table = self.generate_lookup_table_bivariate(|x, y| (x & 1).wrapping_mul(y & 1) % mm);
        self.apply_lookup_table_bivariate(ct_a, ct_b, &table)
    }

    /// Boolean XOR: `(a + b) mod 2` for binary inputs.
    pub fn bitxor(
        &self,
        ct_a: &ShortintCiphertext,
        ct_b: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let table = self.generate_lookup_table_bivariate(|x, y| (x + y) % 2);
        self.apply_lookup_table_bivariate(ct_a, ct_b, &table)
    }

    /// Boolean OR on `{0,1}` via `(a + b - a·b) mod 2`.
    pub fn bitor(
        &self,
        ct_a: &ShortintCiphertext,
        ct_b: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let mm = self.params.message_modulus().0;
        let table = self.generate_lookup_table_bivariate(|x, y| {
            let xb = x & 1;
            let yb = y & 1;
            xb.wrapping_add(yb).saturating_sub(xb.wrapping_mul(yb)) % mm
        });
        self.apply_lookup_table_bivariate(ct_a, ct_b, &table)
    }

    /// `1` iff `a > b`, else `0` (outputs live in the message modulus).
    pub fn gt(
        &self,
        ct_a: &ShortintCiphertext,
        ct_b: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let table = self.generate_lookup_table_bivariate(|a, b| if a > b { 1 } else { 0 });
        self.apply_lookup_table_bivariate(ct_a, ct_b, &table)
    }

    /// `1` iff `a == b`, else `0`.
    pub fn eq(
        &self,
        ct_a: &ShortintCiphertext,
        ct_b: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let table = self.generate_lookup_table_bivariate(|a, b| if a == b { 1 } else { 0 });
        self.apply_lookup_table_bivariate(ct_a, ct_b, &table)
    }

    /// `1` iff `a < b`, else `0`.
    pub fn lt(
        &self,
        ct_a: &ShortintCiphertext,
        ct_b: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let table = self.generate_lookup_table_bivariate(|a, b| if a < b { 1 } else { 0 });
        self.apply_lookup_table_bivariate(ct_a, ct_b, &table)
    }

    /// `1` iff `a >= b`, else `0`.
    pub fn ge(
        &self,
        ct_a: &ShortintCiphertext,
        ct_b: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let table = self.generate_lookup_table_bivariate(|a, b| if a >= b { 1 } else { 0 });
        self.apply_lookup_table_bivariate(ct_a, ct_b, &table)
    }

    /// `1` iff `a <= b`, else `0`.
    pub fn le(
        &self,
        ct_a: &ShortintCiphertext,
        ct_b: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let table = self.generate_lookup_table_bivariate(|a, b| if a <= b { 1 } else { 0 });
        self.apply_lookup_table_bivariate(ct_a, ct_b, &table)
    }

    /// `1` iff `a != b`, else `0`.
    pub fn ne(
        &self,
        ct_a: &ShortintCiphertext,
        ct_b: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let table = self.generate_lookup_table_bivariate(|a, b| if a != b { 1 } else { 0 });
        self.apply_lookup_table_bivariate(ct_a, ct_b, &table)
    }

    /// Homomorphic product `(a · b) mod message_modulus` via one bivariate PBS.
    pub fn mul(
        &self,
        ct_a: &ShortintCiphertext,
        ct_b: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let mut out = ct_a.clone();
        self.mul_assign(&mut out, ct_b)?;
        Ok(out)
    }

    /// Write `(ct_left · ct_right) mod message_modulus` into `ct_left`.
    pub fn mul_assign(
        &self,
        ct_left: &mut ShortintCiphertext,
        ct_right: &ShortintCiphertext,
    ) -> Result<(), ShortintError> {
        let mm = self.params.message_modulus().0;
        let table = self.generate_lookup_table_bivariate(|a, b| a.wrapping_mul(b) % mm);
        self.apply_lookup_table_bivariate_assign(ct_left, ct_right, &table)
    }

    /// Map `a ↦ (-a) mod message_modulus` using scalar multiplication by `message_modulus − 1`.
    pub fn neg(&self, ct: &ShortintCiphertext) -> Result<ShortintCiphertext, ShortintError> {
        let mut out = ct.clone();
        self.neg_assign(&mut out)?;
        Ok(out)
    }

    pub fn neg_assign(&self, ct: &mut ShortintCiphertext) -> Result<(), ShortintError> {
        let mm = self.params.message_modulus().0;
        if mm < 2 {
            return Err(ShortintError::MessageModulusTooSmall);
        }
        self.scalar_mul_assign(ct, mm - 1)
    }

    /// Element-wise maximum in `message_modulus` (values interpreted mod `mm` before compare).
    pub fn max(
        &self,
        ct_a: &ShortintCiphertext,
        ct_b: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let table = self.generate_lookup_table_bivariate(|a, b| if a >= b { a } else { b });
        self.apply_lookup_table_bivariate(ct_a, ct_b, &table)
    }

    /// Element-wise minimum.
    pub fn min(
        &self,
        ct_a: &ShortintCiphertext,
        ct_b: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let table = self.generate_lookup_table_bivariate(|a, b| if a <= b { a } else { b });
        self.apply_lookup_table_bivariate(ct_a, ct_b, &table)
    }

    /// Boolean NOT for ciphertexts whose **message slot** is in `{0, 1}`.
    ///
    /// Implemented as a univariate PBS: `v ↦ (1 − (v mod message_modulus)) mod message_modulus`.
    pub fn bitnot(&self, ct: &ShortintCiphertext) -> Result<ShortintCiphertext, ShortintError> {
        let mm = self.params.message_modulus().0;
        if mm < 2 {
            return Err(ShortintError::MessageModulusTooSmall);
        }
        let lut = self.generate_lookup_table(|v| (1u64 + mm - (v % mm)) % mm);
        self.apply_lookup_table(ct, lut.as_ref())
    }

    pub fn unchecked_sub(
        &self,
        ct_left: &ShortintCiphertext,
        ct_right: &ShortintCiphertext,
    ) -> Result<ShortintCiphertext, ShortintError> {
        let mut out = ct_left.clone();
        self.unchecked_sub_assign(&mut out, ct_right)?;
        Ok(out)
    }
}

fn unchecked_add_assign(
    ct_left: &mut ShortintCiphertext,
    ct_right: &ShortintCiphertext,
    max_degree: MaxDegree,
    max_noise: MaxNoiseLevel,
) -> Result<(), ShortintError> {
    ct_left.ensure_compatible(ct_right)?;
    ct_left.inner_mut().add_assign(ct_right.lwe_ciphertext());
    let new_d = ct_left.degree().saturating_add(ct_right.degree());
    let new_n = ct_left.noise_level().saturating_add(ct_right.noise_level());
    max_degree.validate(new_d)?;
    max_noise.validate(new_n)?;
    ct_left.set_degree_noise(new_d, new_n);
    Ok(())
}

fn unchecked_scalar_mul_assign(
    ct: &mut ShortintCiphertext,
    scalar: u64,
    max_degree: MaxDegree,
    max_noise: MaxNoiseLevel,
) -> Result<(), ShortintError> {
    if scalar == 0 {
        *ct.inner_mut() = LweCiphertext::zeros(
            ct.lwe_ciphertext().dimension(),
            CiphertextModulusLog(ct.lwe_ciphertext().log_modulus()),
        );
        ct.set_degree_noise(Degree::new(0), NoiseLevel::NOMINAL);
        return Ok(());
    }
    let new_d = ct.degree().saturating_mul_scalar(scalar);
    let new_n = ct.noise_level().saturating_mul_scalar(scalar);
    max_degree.validate(new_d)?;
    max_noise.validate(new_n)?;
    ct.inner_mut().scalar_mul_assign(scalar);
    ct.set_degree_noise(new_d, new_n);
    Ok(())
}

#[cfg(test)]
mod tests {
    use rand_core::SeedableRng;
    use silent_params::{
        CarryModulus, CiphertextModulusLog, DecompositionBaseLog, DecompositionLevelCount,
        EncryptionKeyChoice, GlweDimension, Log2PFail, LweDimension, MessageModulus,
        NoiseDistribution, PolynomialSize, SecurityLevel, TfheParams,
    };
    use silent_utils::rng::SecureRng;

    use crate::generate_shortint_keys;
    use crate::schemes::tfhe::params::TfheParameters;

    fn two_bit_radix_limb_params() -> TfheParameters {
        TfheParameters::new(TfheParams {
            name: "shortint-two-bit-radix-limb-test",
            lwe_dimension: LweDimension(8),
            glwe_dimension: GlweDimension(1),
            polynomial_size: PolynomialSize(64),
            ciphertext_modulus_log: CiphertextModulusLog(64),
            message_modulus: MessageModulus(4),
            carry_modulus: CarryModulus(4),
            pbs_base_log: DecompositionBaseLog(8),
            pbs_level: DecompositionLevelCount(4),
            ks_base_log: DecompositionBaseLog(2),
            ks_level: DecompositionLevelCount(7),
            lwe_noise: NoiseDistribution::Gaussian { stddev: 1e-12 },
            glwe_noise: NoiseDistribution::Gaussian { stddev: 1e-15 },
            log2_p_fail: Log2PFail(-40.0),
            encryption_key_choice: EncryptionKeyChoice::Big,
            security_level: SecurityLevel::Toy,
        })
        .unwrap()
    }

    #[test]
    fn bivariate_compare_and_max_support_two_bit_radix_limbs() {
        let params = two_bit_radix_limb_params();
        let mut rng = SecureRng::from_seed([19u8; 32]);
        let (mut client, server) = generate_shortint_keys(params, &mut rng).unwrap();

        let a = client.encrypt(3).unwrap();
        let b = client.encrypt(2).unwrap();
        assert_eq!(client.decrypt(&server.gt(&a, &b).unwrap()).unwrap(), 1);
        assert_eq!(client.decrypt(&server.gt(&b, &a).unwrap()).unwrap(), 0);
        assert_eq!(client.decrypt(&server.eq(&a, &a).unwrap()).unwrap(), 1);
        assert_eq!(client.decrypt(&server.eq(&a, &b).unwrap()).unwrap(), 0);
        assert_eq!(client.decrypt(&server.max(&a, &b).unwrap()).unwrap(), 3);

        let zero = client.encrypt(0).unwrap();
        let one = client.encrypt(1).unwrap();
        let two = client.encrypt(2).unwrap();
        assert_eq!(client.decrypt(&server.mul(&one, &two).unwrap()).unwrap(), 2);
        assert_eq!(
            client
                .decrypt(&server.bitand(&zero, &zero).unwrap())
                .unwrap(),
            0
        );
        assert_eq!(
            client
                .decrypt(&server.bitand(&one, &zero).unwrap())
                .unwrap(),
            0
        );
        assert_eq!(
            client.decrypt(&server.bitand(&one, &one).unwrap()).unwrap(),
            1
        );
        assert_eq!(
            client
                .decrypt(&server.bitor(&zero, &zero).unwrap())
                .unwrap(),
            0
        );
        assert_eq!(
            client.decrypt(&server.bitor(&one, &zero).unwrap()).unwrap(),
            1
        );
        assert_eq!(
            client.decrypt(&server.bitor(&one, &one).unwrap()).unwrap(),
            1
        );
        assert_eq!(client.decrypt(&server.bitnot(&zero).unwrap()).unwrap(), 1);
        assert_eq!(client.decrypt(&server.bitnot(&one).unwrap()).unwrap(), 0);
    }
}
