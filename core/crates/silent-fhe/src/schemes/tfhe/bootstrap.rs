//! Programmable bootstrapping pipeline.
//!
//! The flow follows the canonical CGGI / TFHE recipe:
//!
//! ```text
//!   Input  : LWE c under sk_small  (encrypts message m, plus noise)
//!   Stages :  1. modulus switch q → 2N
//!             2. blind rotation: divide the test GLWE by X^{ms(body)}, then
//!                CMUX ladder using (acc · X^{ms(a_i)} - acc) per mask digit
//!                (TFHE/CGGI monomial conventions; mask digit 0 skips a step)
//!             3. sample extraction (GLWE → big LWE)
//!             4. LWE→LWE key switch back to sk_small
//!   Output : LWE c' under sk_small (encrypts f(m))
//! ```
//!
//! Blind rotation / PBS hot paths use the `O(N log N)` split-precision
//! [`silent_math::fft64::FftMul`] backend by default (thread-local plan
//! cache via [`silent_math::fft64::with_fft_mul`]).  Pass
//! [`silent_math::fft64::SchoolbookMul`] or [`silent_math::fft64::KaratsubaMul`]
//! to [`TfheBootstrapper::apply_lookup_table_with`] for regression tests.
//! Some auxiliary steps (e.g. GLWE symmetric encryption during keygen) still
//! use [`SchoolbookMul`] for a small, obvious baseline.

use silent_math::fft64::{NegacyclicMul, SchoolbookMul, with_fft_mul};
use silent_math::torus::modulus_switch_u64;
use silent_params::{
    CiphertextModulusLog, DecompositionBaseLog, DecompositionLevelCount, GlweDimension,
    NoiseDistribution, PolynomialSize,
};
use silent_ring::{GadgetDecomposition, NativePoly};
use silent_rlwe::{
    GgswCiphertext, GgswCiphertextList, GlweCiphertext, GlweSecretKey, LweBootstrapKey,
    LweCiphertext, LweKeyswitchKey, LweSecretKey,
};
use silent_utils::rng::SecureRng;

use super::encoding::TfheEncoder;
use super::keyswitch::{LweKeyswitcher, build_lwe_keyswitch_key};
use super::params::TfheParameters;
use super::sampling::{fill_noise, fill_uniform_mod};

// ---------------------------------------------------------------------------
// GLWE encryption / GGSW construction
// ---------------------------------------------------------------------------

/// Symmetric GLWE encryption of a plaintext polynomial under `sk_glwe`.
///
/// The plaintext is interpreted as a polynomial whose coefficients have
/// already been scaled (i.e. `Δ·μ` if you think of it message-by-message).
/// Used internally by GGSW row construction.
pub fn encrypt_glwe_under_sk(
    rng: &mut SecureRng,
    sk_glwe: &GlweSecretKey,
    plaintext: &NativePoly,
    noise: NoiseDistribution,
    log_modulus: CiphertextModulusLog,
) -> GlweCiphertext {
    let n = sk_glwe.polynomial_size().0;
    let glwe_dim = sk_glwe.glwe_dimension();

    let mut glwe = GlweCiphertext::zeros(glwe_dim, sk_glwe.polynomial_size(), log_modulus);

    // a uniformly random.
    for poly in glwe.mask_mut() {
        let mut buf = vec![0u64; n];
        fill_uniform_mod(rng, log_modulus.0, &mut buf);
        *poly = NativePoly::from_u64(&buf, log_modulus.0);
    }

    // body = sum_i a_i * s_i + plaintext + e.
    let body = glwe.body_mut();
    *body = plaintext.clone();
    let mut noise_buf = vec![0u64; n];
    fill_noise(rng, noise, log_modulus.0, &mut noise_buf);
    let noise_poly = NativePoly::from_u64(&noise_buf, log_modulus.0);
    body.add_assign(&noise_poly);

    // Workaround for borrow checker: we need to read mask while writing body.
    // Recompute on a snapshot.
    let mask_snapshot: Vec<NativePoly> = glwe.mask().to_vec();
    let backend = SchoolbookMul;
    for (a_i, s_i) in mask_snapshot.iter().zip(sk_glwe.polys().iter()) {
        a_i.fma_with(s_i, glwe.body_mut(), &backend);
    }

    glwe
}

/// Encrypt the *constant polynomial* `value * X^0` as a GLWE.  Useful for
/// trivial encryptions (no noise, no mask) when the test polynomial does not
/// need to hide the LUT.
pub fn trivial_glwe_constant(
    glwe_dim: GlweDimension,
    polynomial_size: PolynomialSize,
    log_modulus: CiphertextModulusLog,
    value_poly: NativePoly,
) -> GlweCiphertext {
    debug_assert_eq!(value_poly.degree(), polynomial_size.0);
    let mut glwe = GlweCiphertext::zeros(glwe_dim, polynomial_size, log_modulus);
    *glwe.body_mut() = value_poly;
    glwe
}

/// Encrypt a single integer message `m ∈ {0, 1}` (binary key) as a GGSW under
/// `sk_glwe`.  The value `m` is encoded into each level matrix according to
/// the gadget decomposition `(base_log, level)`.
pub fn encrypt_ggsw_binary(
    rng: &mut SecureRng,
    sk_glwe: &GlweSecretKey,
    message: u64,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    noise: NoiseDistribution,
    log_modulus: CiphertextModulusLog,
) -> GgswCiphertext {
    let polynomial_size = sk_glwe.polynomial_size();
    let glwe_dim = sk_glwe.glwe_dimension();

    let mut ggsw = GgswCiphertext::zeros(glwe_dim, polynomial_size, base_log, level, log_modulus);

    for j in 0..level.0 as usize {
        let shift = (log_modulus.0 as u32) - (j as u32 + 1) * (base_log.0 as u32);
        let scale = if shift >= 64 { 0 } else { 1u64 << shift };
        // For each row r in 0..k+1: encrypt the appropriate scaled message.
        // Layout: mask rows encrypt `-s_r · m · scale`, the last row encrypts
        // `m · scale` in the constant coefficient (standard GGSW encoding for a
        // binary secret with gadget decomposition).
        for r in 0..glwe_dim.glwe_size() {
            let scaled_value = message.wrapping_mul(scale);
            let plaintext_poly = if r < glwe_dim.0 {
                // -s_r * (message * scale)
                let mut p = sk_glwe.polys()[r].clone();
                p.scalar_mul_assign(scaled_value);
                p.negate_assign();
                p
            } else {
                // message * scale (as a constant polynomial)
                let mut p = NativePoly::zeros(polynomial_size.0, log_modulus.0);
                p.coeffs_mut()[0] = scaled_value;
                p
            };
            let glwe_row = encrypt_glwe_under_sk(rng, sk_glwe, &plaintext_poly, noise, log_modulus);
            ggsw.levels_mut()[j].rows_mut()[r] = glwe_row;
        }
    }

    ggsw
}

// ---------------------------------------------------------------------------
// Bootstrap key generation
// ---------------------------------------------------------------------------

/// Generate the LWE bootstrap key: one GGSW encryption of `s_i ∈ {0, 1}`
/// under `sk_glwe` per coordinate of `sk_lwe`.
pub fn build_lwe_bootstrap_key(
    rng: &mut SecureRng,
    sk_lwe: &LweSecretKey,
    sk_glwe: &GlweSecretKey,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    noise: NoiseDistribution,
    log_modulus: CiphertextModulusLog,
) -> LweBootstrapKey {
    assert_eq!(sk_lwe.log_modulus(), log_modulus.0);
    assert_eq!(sk_glwe.log_modulus(), log_modulus.0);

    let ggsw_list: Vec<_> = sk_lwe
        .data()
        .iter()
        .map(|&s_i| encrypt_ggsw_binary(rng, sk_glwe, s_i, base_log, level, noise, log_modulus))
        .collect();

    LweBootstrapKey::new(
        GgswCiphertextList::new(ggsw_list),
        sk_lwe.dimension(),
        sk_glwe.glwe_dimension(),
        sk_glwe.polynomial_size(),
        base_log,
        level,
        log_modulus,
    )
}

// ---------------------------------------------------------------------------
// External product, CMUX, blind rotation
// ---------------------------------------------------------------------------

/// Decompose a polynomial coefficient-wise into `level` signed-digit polys.
fn decompose_poly(poly: &NativePoly, gadget: &GadgetDecomposition, out: &mut [NativePoly]) {
    debug_assert_eq!(out.len(), gadget.level() as usize);
    let n = poly.degree();
    let mut digits = vec![0u64; gadget.level() as usize];
    for k in 0..n {
        gadget.decompose_into(poly.coeffs()[k], &mut digits);
        for (j, d) in digits.iter().enumerate() {
            out[j].coeffs_mut()[k] = *d;
        }
    }
}

/// Compute `ggsw ⊠ glwe`: the external product producing a GLWE encrypting
/// `ggsw_message · glwe_message`.
///
/// Implementation sketch:
///
/// ```text
///   acc := 0
///   for poly_index r in 0..k+1:
///     decompose glwe.polys[r] into level signed digit polys
///     for level j in 0..level:
///       digit_poly := decomposition[j]
///       row_glwe   := ggsw.levels[j].rows[r]
///       acc.poly[i] += digit_poly * row_glwe.poly[i]      ∀ i ∈ 0..k+1
///   return acc
/// ```
pub fn build_external_product<M: NegacyclicMul>(
    ggsw: &GgswCiphertext,
    glwe: &GlweCiphertext,
    backend: &M,
) -> GlweCiphertext {
    debug_assert_eq!(ggsw.glwe_dimension(), glwe.glwe_dimension());
    debug_assert_eq!(ggsw.polynomial_size(), glwe.polynomial_size());
    debug_assert_eq!(ggsw.log_modulus(), glwe.log_modulus());

    let log_modulus = CiphertextModulusLog(glwe.log_modulus());
    let polynomial_size = glwe.polynomial_size();
    let glwe_dim = glwe.glwe_dimension();

    let gadget = GadgetDecomposition::new(log_modulus.0, ggsw.base_log().0, ggsw.level_count().0);

    // Scratch storage for the decomposition of one input poly.
    let level = ggsw.level_count().0 as usize;
    let mut decomposed: Vec<NativePoly> = (0..level)
        .map(|_| NativePoly::zeros(polynomial_size.0, log_modulus.0))
        .collect();

    let mut acc = GlweCiphertext::zeros(glwe_dim, polynomial_size, log_modulus);

    for (r, input_poly) in glwe.polys().iter().enumerate() {
        decompose_poly(input_poly, &gadget, &mut decomposed);
        for j in 0..level {
            let digit_poly = &decomposed[j];
            let row_glwe = &ggsw.levels()[j].rows()[r];
            // acc.polys[i] += digit_poly * row_glwe.polys[i]   ∀ i.
            for (i, row_poly) in row_glwe.polys().iter().enumerate() {
                digit_poly.fma_with(row_poly, &mut acc.polys_mut()[i], backend);
            }
        }
    }

    acc
}

/// CMUX gate: `cmux(GGSW(s), c0, c1) = c0 + s · (c1 − c0)`.
///
/// `s ∈ {0, 1}` (we sample binary GGSW keys, so this matches the BSK shape).
pub fn cmux<M: NegacyclicMul>(
    ggsw: &GgswCiphertext,
    c0: &GlweCiphertext,
    c1: &GlweCiphertext,
    backend: &M,
) -> GlweCiphertext {
    let mut diff = c1.clone();
    diff.sub_assign(c0);
    let mut out = build_external_product(ggsw, &diff, backend);
    out.add_assign(c0);
    out
}

/// Run blind rotation of the accumulator `acc` driven by an LWE ciphertext.
///
/// Matches the CGGI / TFHE layout: divide the accumulator by `X^{ms_b}` on the
/// modulus-switched body, then for each mask digit `ms_a` (skipped when zero)
/// CMUX against the branch that multiplies by `X^{ms_a}` using the fused
/// monomial map `(acc · X^{ms_a} - acc)` inside [`cmux`].
///
/// ```text
///   ms_b = round(b · 2N / q)
///   acc := acc / X^{ms_b}   (monic monomial division, each RLWE limb)
///   for each i in 0..n:
///     ms_a = round(a_i · 2N / q)
///     if ms_a == 0: continue
///     acc := cmux( bsk[i], acc, acc · X^{ms_a} )   (equivalently via diff poly)
/// ```
pub fn blind_rotate_assign<M: NegacyclicMul>(
    acc: &mut GlweCiphertext,
    lwe: &LweCiphertext,
    bsk: &LweBootstrapKey,
    backend: &M,
) {
    debug_assert_eq!(lwe.dimension(), bsk.input_lwe_dimension());
    debug_assert_eq!(acc.glwe_dimension(), bsk.glwe_dimension());
    debug_assert_eq!(acc.polynomial_size(), bsk.polynomial_size());
    debug_assert_eq!(acc.log_modulus(), bsk.log_modulus());

    let polynomial_size = acc.polynomial_size().0;
    let two_n = polynomial_size << 1;
    let log_two_n = (two_n as u64).trailing_zeros() as u8;
    let log_q = lwe.log_modulus();

    let ms_b = modulus_switch_u64(lwe.body(), log_q, log_two_n) as usize;
    for p in acc.polys_mut() {
        p.wrapping_monic_monomial_div_assign(ms_b);
    }

    let n = polynomial_size;
    let log_poly = acc.log_modulus();
    let mut diff_poly = NativePoly::zeros(n, log_poly);
    let mut rotated_acc = acc.clone();

    for (i, &a_i) in lwe.mask().iter().enumerate() {
        let ms_a = modulus_switch_u64(a_i, log_q, log_two_n) as usize;
        if ms_a == 0 {
            continue;
        }

        for (dst, src) in rotated_acc.polys_mut().iter_mut().zip(acc.polys().iter()) {
            diff_poly.wrapping_monic_monomial_mul_and_subtract_from(src, ms_a);
            dst.clone_from(src);
            dst.add_assign(&diff_poly);
        }

        let new_acc = cmux(bsk.get(i), acc, &rotated_acc, backend);
        *acc = new_acc;
    }
}

// ---------------------------------------------------------------------------
// Sample extraction (GLWE → big LWE)
// ---------------------------------------------------------------------------

/// Extract the LWE encryption of `glwe.body.coeffs[position]` under the
/// flat sample-extracted GLWE secret key.
pub fn sample_extract(glwe: &GlweCiphertext, position: usize) -> LweCiphertext {
    let n = glwe.polynomial_size().0;
    let k = glwe.glwe_dimension().0;
    debug_assert!(position < n);

    let log_modulus = CiphertextModulusLog(glwe.log_modulus());
    let mut data = vec![0u64; k * n + 1];

    // body = glwe.body.coeffs[position].
    *data.last_mut().unwrap() = glwe.body().coeffs()[position];

    // mask: see the negacyclic-extraction formula.
    for (i, mask_poly) in glwe.mask().iter().enumerate() {
        let coeffs = mask_poly.coeffs();
        let block = &mut data[i * n..(i + 1) * n];
        for j in 0..n {
            block[j] = if j <= position {
                coeffs[position - j]
            } else {
                coeffs[n + position - j].wrapping_neg()
            };
        }
    }

    let mut ct = LweCiphertext::from_data(data, log_modulus);
    ct.reduce();
    ct
}

// ---------------------------------------------------------------------------
// Lookup-table construction & full PBS
// ---------------------------------------------------------------------------

/// Build a TFHE accumulator polynomial for programmable bootstrapping using the
/// standard “redundant box per plaintext value” layout on the negacyclic ring
/// dimension `N`: each of the `T = message_modulus · carry_modulus` slots is
/// replicated across `N / T` coefficients so modulus switching lands inside the
/// intended box.
///
/// For the decoded output to equal `f(m)` on the tested inputs, `f` must
/// respect the negacyclic constraints of the scheme (padding doubles the
/// effective plain modulus). After filling boxes with `encode(f(m))`, apply
/// the usual half-box fixup for programmable PBS: negate the leading
/// `box_size / 2` coefficients of the body, then rotate the full coefficient
/// vector left by that same offset so blind rotation addresses LUT entries
/// consistently. A different negate/rotate composition applies a fixed offset
/// to every table output.
///
/// When validating `f` only on `m ∈ [0, T/2)`, the rounded decode matches
/// `f(m)` without exercising upper-slot boundary behaviour.
pub fn build_lookup_table_polynomial<F>(
    encoder: &TfheEncoder,
    polynomial_size: PolynomialSize,
    log_modulus: CiphertextModulusLog,
    mut f: F,
) -> NativePoly
where
    F: FnMut(u64) -> u64,
{
    let total = encoder.full_plaintext_modulus();
    assert!(total > 0);
    let n = polynomial_size.0;
    let box_size = n.checked_div(total as usize).expect("total > 0").max(1);
    assert!(
        box_size > 0,
        "polynomial size {n} cannot host {total} message boxes"
    );
    let half_box = box_size / 2;

    // Step 1: fill the polynomial with constant blocks of `encode(f(m))`.
    let mut raw = vec![0u64; n];
    for m in 0..total {
        let value = f(m) % total;
        let encoded = encoder.encode_full(value).value;
        let start = (m as usize) * box_size;
        let end = (start + box_size).min(n);
        for slot in &mut raw[start..end] {
            *slot = encoded;
        }
    }

    // Half-box negacyclic fixup (see module docs above).
    for slot in &mut raw[0..half_box] {
        *slot = slot.wrapping_neg();
    }
    raw.rotate_left(half_box);

    NativePoly::from_u64(&raw, log_modulus.0)
}

/// All-in-one PBS bootstrapper.  Holds the bootstrap key and the LWE→LWE
/// key-switch key together with a [`TfheEncoder`] for LUT construction.
pub struct TfheBootstrapper<'a> {
    params: TfheParameters,
    bsk: &'a LweBootstrapKey,
    ksk: &'a LweKeyswitchKey,
}

impl<'a> TfheBootstrapper<'a> {
    pub fn new(params: TfheParameters, bsk: &'a LweBootstrapKey, ksk: &'a LweKeyswitchKey) -> Self {
        debug_assert_eq!(bsk.log_modulus(), params.ciphertext_modulus_log().0);
        debug_assert_eq!(ksk.log_modulus(), params.ciphertext_modulus_log().0);
        Self { params, bsk, ksk }
    }

    pub fn params(&self) -> &TfheParameters {
        &self.params
    }

    /// Borrowed access to the underlying bootstrap key.  Used by the
    /// FFT-cache constructor in [`super::bootstrap_fft`].
    pub fn bsk(&self) -> &'a LweBootstrapKey {
        self.bsk
    }

    /// Borrowed access to the underlying key-switching key.  Used by the
    /// FFT-cached PBS path in [`super::bootstrap_fft`].
    pub fn ksk(&self) -> &'a LweKeyswitchKey {
        self.ksk
    }

    /// Apply the lookup table `f` to an LWE ciphertext encrypting `m`.
    /// Returns an LWE ciphertext encrypting `f(m)` under the small key.
    ///
    /// Uses the [`silent_math::fft64::FftMul`] backend (thread-local cache).
    /// Call [`Self::apply_lookup_table_with`] to select
    /// [`SchoolbookMul`], [`silent_math::fft64::KaratsubaMul`], or another
    /// [`NegacyclicMul`] (e.g. for bit-exact regression against the reference).
    pub fn apply_lookup_table<F: FnMut(u64) -> u64>(
        &self,
        encoder: &TfheEncoder,
        ct: &LweCiphertext,
        f: F,
    ) -> LweCiphertext {
        let n = self.params.polynomial_size().0;
        with_fft_mul(n, |fft| self.apply_lookup_table_with(encoder, ct, f, fft))
    }

    /// Like [`Self::apply_lookup_table`] but lets callers select the
    /// [`NegacyclicMul`] backend driving every polynomial multiplication
    /// inside the blind rotation.
    pub fn apply_lookup_table_with<F, M>(
        &self,
        encoder: &TfheEncoder,
        ct: &LweCiphertext,
        f: F,
        backend: &M,
    ) -> LweCiphertext
    where
        F: FnMut(u64) -> u64,
        M: NegacyclicMul,
    {
        let lut_poly = build_lookup_table_polynomial(
            encoder,
            self.params.polynomial_size(),
            self.params.ciphertext_modulus_log(),
            f,
        );
        let mut acc = trivial_glwe_constant(
            self.params.glwe_dimension(),
            self.params.polynomial_size(),
            self.params.ciphertext_modulus_log(),
            lut_poly,
        );
        blind_rotate_assign(&mut acc, ct, self.bsk, backend);
        let big_lwe = sample_extract(&acc, 0);
        LweKeyswitcher::new(self.ksk).keyswitch(&big_lwe)
    }

    /// PBS with a caller-built accumulator polynomial (see
    /// [`build_lookup_table_polynomial`]).  Used to amortise LUT
    /// construction across many bootstraps (shortint bivariate tables, custom
    /// programme logic, etc.).
    pub fn apply_lookup_table_with_polynomial<M: NegacyclicMul>(
        &self,
        ct: &LweCiphertext,
        lut_poly: NativePoly,
        backend: &M,
    ) -> LweCiphertext {
        let mut acc = trivial_glwe_constant(
            self.params.glwe_dimension(),
            self.params.polynomial_size(),
            self.params.ciphertext_modulus_log(),
            lut_poly,
        );
        blind_rotate_assign(&mut acc, ct, self.bsk, backend);
        let big_lwe = sample_extract(&acc, 0);
        LweKeyswitcher::new(self.ksk).keyswitch(&big_lwe)
    }
}

// ---------------------------------------------------------------------------
// Convenience: in-one-shot PBS keygen
// ---------------------------------------------------------------------------

/// Generate (sk_small, sk_glwe, bsk, ksk) ready to drive a [`TfheBootstrapper`].
///
/// Uses the supplied RNG for *all* randomness so callers can deterministically
/// reproduce the keys for tests.
pub fn generate_pbs_keys(
    params: TfheParameters,
    rng: &mut SecureRng,
) -> (
    LweSecretKey,
    GlweSecretKey,
    LweBootstrapKey,
    LweKeyswitchKey,
) {
    let mut keygen = super::keys::TfheKeyGenerator::with_rng(params.clone(), clone_rng_state(rng));

    let sk_small = keygen.generate_lwe_secret_key();
    let sk_glwe = keygen.generate_glwe_secret_key();
    let sk_big = keygen.lwe_secret_from_glwe(&sk_glwe);

    // Reuse `rng` (the parameter) for BSK + KSK to keep the public surface small.
    let bsk = build_lwe_bootstrap_key(
        rng,
        &sk_small,
        &sk_glwe,
        params.pbs_base_log(),
        params.pbs_level(),
        params.glwe_noise(),
        params.ciphertext_modulus_log(),
    );
    let ksk = build_lwe_keyswitch_key(
        rng,
        &sk_big,
        &sk_small,
        params.ks_base_log(),
        params.ks_level(),
        params.lwe_noise(),
        params.ciphertext_modulus_log(),
    );

    (sk_small, sk_glwe, bsk, ksk)
}

/// Internal helper: produce an independent `SecureRng` snapshot for the
/// auxiliary keygen step.  Uses a fresh seed mixed with the input RNG so the
/// auxiliary stream cannot collide with the caller's BSK/KSK randomness.
fn clone_rng_state(rng: &mut SecureRng) -> SecureRng {
    use rand::RngCore;
    use rand_core::SeedableRng;
    let mut seed = [0u8; 32];
    rng.fill_bytes(&mut seed);
    SecureRng::from_seed(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemes::tfhe::{
        crypto::{TfheDecryptor, TfheEncryptor},
        encoding::TfheEncoder,
    };
    use rand_core::SeedableRng;
    use silent_params::{
        CiphertextModulusLog, DecompositionBaseLog, DecompositionLevelCount, GlweDimension,
        Log2PFail, LweDimension, MessageModulus, NoiseDistribution, PolynomialSize, presets,
    };
    use silent_utils::rng::SecureRng;

    /// Very small (insecure!) parameter set used only to keep blind rotation
    /// fast enough for unit tests on the schoolbook backend.
    fn micro_params() -> TfheParameters {
        let mut p = presets::toy::toy_tfhe_n512();
        p.lwe_dimension = LweDimension(8);
        p.polynomial_size = PolynomialSize(64);
        p.glwe_dimension = GlweDimension(1);
        p.message_modulus = MessageModulus(4);
        p.carry_modulus = silent_params::CarryModulus(4);
        p.pbs_base_log = DecompositionBaseLog(8);
        p.pbs_level = DecompositionLevelCount(4);
        p.ks_base_log = DecompositionBaseLog(2);
        p.ks_level = DecompositionLevelCount(7);
        p.lwe_noise = NoiseDistribution::Gaussian { stddev: 1e-12 };
        p.glwe_noise = NoiseDistribution::Gaussian { stddev: 1e-15 };
        p.log2_p_fail = Log2PFail(-40.0);
        TfheParameters::new(p).unwrap()
    }

    #[test]
    fn external_product_of_zero_ggsw_yields_zero() {
        let params = micro_params();
        let log_q = params.ciphertext_modulus_log();
        let mut rng = SecureRng::from_seed([99u8; 32]);
        let mut keygen = crate::schemes::tfhe::keys::TfheKeyGenerator::with_rng(
            params.clone(),
            SecureRng::from_seed([100u8; 32]),
        );
        let sk_glwe = keygen.generate_glwe_secret_key();

        let zero_ggsw = encrypt_ggsw_binary(
            &mut rng,
            &sk_glwe,
            0,
            params.pbs_base_log(),
            params.pbs_level(),
            params.glwe_noise(),
            log_q,
        );

        // Trivial GLWE encryption of a non-zero polynomial.
        let mut p = NativePoly::zeros(params.polynomial_size().0, log_q.0);
        p.coeffs_mut()[0] = 0xDEAD_BEEFu64;
        let glwe =
            trivial_glwe_constant(params.glwe_dimension(), params.polynomial_size(), log_q, p);

        let backend = SchoolbookMul;
        let prod = build_external_product(&zero_ggsw, &glwe, &backend);

        // Decrypt: body - <mask, sk_glwe> should be a polynomial of small noise.
        let mut residual = prod.body().clone();
        for (a, s) in prod.mask().iter().zip(sk_glwe.polys().iter()) {
            let mut t = a.mul(s);
            residual.sub_assign(&t);
            let _ = &mut t;
        }
        // Each coefficient should be small (only noise).  We use a generous
        // bound because there are many noise sources.
        let bound = 1u64 << 50;
        for &c in residual.coeffs() {
            let signed = c as i64;
            let abs = signed.unsigned_abs();
            assert!(abs < bound || abs > u64::MAX - bound, "residual = {c:#x}");
        }
    }

    #[test]
    fn sample_extract_recovers_constant_term() {
        // Build a trivial GLWE encryption of polynomial `body`, then extract.
        let log_q = CiphertextModulusLog(64);
        let mut body = NativePoly::zeros(8, 64);
        body.coeffs_mut().copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let glwe = trivial_glwe_constant(GlweDimension(2), PolynomialSize(8), log_q, body);

        let lwe = sample_extract(&glwe, 0);
        // body = 1 (constant term of the body polynomial). mask is all-zero
        // because trivial GLWE has zero mask.
        assert_eq!(lwe.body(), 1);
        assert!(lwe.mask().iter().all(|&v| v == 0));
    }

    #[test]
    fn full_pbs_identity_lookup_recovers_message() {
        let params = micro_params();
        let encoder = TfheEncoder::new(params.clone());
        let mut keygen_rng = SecureRng::from_seed([21u8; 32]);
        let (sk_small, _sk_glwe, bsk, ksk) = generate_pbs_keys(params.clone(), &mut keygen_rng);

        let mut enc = TfheEncryptor::new(params.clone(), SecureRng::from_seed([22u8; 32]));
        let dec = TfheDecryptor::new(params.clone(), sk_small.clone());
        let bootstrapper = TfheBootstrapper::new(params.clone(), &bsk, &ksk);

        // Restrict to lower-half messages so the negacyclic LUT returns the
        // identity (the upper half would return -m, which is also testable).
        for m in 0..(encoder.params().total_message_modulus() / 2) {
            let ct = enc.encrypt_symmetric(&sk_small, encoder.encode_full(m));
            let bootstrapped = bootstrapper.apply_lookup_table(&encoder, &ct, |x| x);
            let decoded = dec.decrypt_full(&bootstrapped, &encoder);
            assert_eq!(decoded, m, "PBS identity failed for m={m}");
        }
    }

    #[test]
    fn full_pbs_double_lookup() {
        let params = micro_params();
        let encoder = TfheEncoder::new(params.clone());
        let mut keygen_rng = SecureRng::from_seed([31u8; 32]);
        let (sk_small, _sk_glwe, bsk, ksk) = generate_pbs_keys(params.clone(), &mut keygen_rng);

        let mut enc = TfheEncryptor::new(params.clone(), SecureRng::from_seed([32u8; 32]));
        let dec = TfheDecryptor::new(params.clone(), sk_small.clone());
        let bootstrapper = TfheBootstrapper::new(params.clone(), &bsk, &ksk);

        // f(m) = (2 * m) mod total.  Restrict m so 2m stays inside the
        // negacyclic lower half.
        let total = encoder.params().total_message_modulus();
        let max_m = total / 4; // 2*m < total/2
        for m in 0..max_m {
            let ct = enc.encrypt_symmetric(&sk_small, encoder.encode_full(m));
            let bootstrapped = bootstrapper.apply_lookup_table(&encoder, &ct, |x| (2 * x) % total);
            let decoded = dec.decrypt_full(&bootstrapped, &encoder);
            assert_eq!(decoded, 2 * m, "PBS doubling failed for m={m}");
        }
    }

    #[test]
    fn full_pbs_increment_mod_total_lower_half() {
        let params = micro_params();
        let encoder = TfheEncoder::new(params.clone());
        let mut keygen_rng = SecureRng::from_seed([41u8; 32]);
        let (sk_small, _sk_glwe, bsk, ksk) = generate_pbs_keys(params.clone(), &mut keygen_rng);

        let mut enc = TfheEncryptor::new(params.clone(), SecureRng::from_seed([42u8; 32]));
        let dec = TfheDecryptor::new(params.clone(), sk_small.clone());
        let bootstrapper = TfheBootstrapper::new(params.clone(), &bsk, &ksk);

        let total = encoder.params().total_message_modulus();
        let half = total / 2;
        for m in 0..(half.saturating_sub(1)) {
            let ct = enc.encrypt_symmetric(&sk_small, encoder.encode_full(m));
            let bootstrapped = bootstrapper.apply_lookup_table(&encoder, &ct, |x| (x + 1) % total);
            assert_eq!(
                dec.decrypt_full(&bootstrapped, &encoder),
                (m + 1) % total,
                "PBS increment failed for m={m}"
            );
        }
    }

    #[test]
    fn lut_increment_box_index_matches_trivial_pbs_semantics() {
        let params = micro_params();
        let encoder = TfheEncoder::new(params.clone());
        let total = encoder.full_plaintext_modulus();
        let n = params.polynomial_size().0;
        let box_size = n / total as usize;

        let poly_inc = build_lookup_table_polynomial(
            &encoder,
            params.polynomial_size(),
            params.ciphertext_modulus_log(),
            |x| (x + 1) % total,
        );
        let poly_id = build_lookup_table_polynomial(
            &encoder,
            params.polynomial_size(),
            params.ciphertext_modulus_log(),
            |x| x,
        );
        let coeffs_inc = poly_inc.coeffs();
        let coeffs_id = poly_id.coeffs();
        for m in [0u64, 1u64] {
            let idx = m as usize * box_size;
            assert_eq!(
                coeffs_id[idx],
                encoder.encode_full(m).value,
                "identity LUT at box start m={m}"
            );
            assert_eq!(
                coeffs_inc[idx],
                encoder.encode_full((m + 1) % total).value,
                "increment LUT at box start m={m}"
            );
        }
    }
}
