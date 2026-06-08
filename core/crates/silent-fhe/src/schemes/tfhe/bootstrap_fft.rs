//! Frequency-domain bootstrap-key cache and the matching FFT-driven PBS
//! pipeline.
//!
//! Background
//! ==========
//!
//! The [`super::bootstrap`] module ships a fully working PBS that calls
//! [`silent_math::fft64::FftMul`] (or any [`NegacyclicMul`]) one polynomial
//! multiplication at a time.  Each `fma_with` call internally:
//!
//! 1. forward-FFTs the LHS (the gadget-decomposed accumulator chunk),
//! 2. forward-FFTs the RHS (the GGSW polynomial coming from the BSK),
//! 3. accumulates the 16-bit-split pointwise products,
//! 4. inverse-FFTs each of the 4 buckets,
//! 5. recombines the result into a `u64` polynomial.
//!
//! For a production-shape parameter set (e.g. `dev_tfhe_n2048`,
//! `lwe_dim = 918`, `level = 1`, `k+1 = 2`) one PBS runs
//! `lwe_dim * (k+1) * level * (k+1) = 918 * 2 * 1 * 2 = 3672` of these calls.
//! That works out to **3672 forward FFTs of BSK polynomials** — and the BSK
//! never changes!  Worse still, the LHS digit polynomial gets re-FFT'd
//! `(k+1) = 2` times per `(r, j)` pair because the inner loop iterates over
//! `i ∈ 0..k+1` while the LHS only depends on `(r, j)`.
//!
//! This module addresses both issues by:
//!
//! * Pre-computing the forward FFT of every BSK polynomial *once*, packaging
//!   the result as [`LweBootstrapKeyFft`].  Subsequent PBS calls simply read
//!   the cached frequency-domain representation.
//! * Restructuring [`build_external_product_fft`] so that:
//!     - the LHS digit polynomial is FFT'd exactly once per `(r, j)`,
//!     - all `level * (k+1) * (k+1)` GGSW × digit pointwise products
//!       accumulate into a *single* frequency-domain accumulator per output
//!       polynomial (`(k+1)` accumulators total),
//!     - the inverse FFT (4 bucket IFFTs + recombine) runs once per output
//!       polynomial of the resulting GLWE — i.e. `(k+1)` IFFT chains per
//!       external product instead of `level * (k+1)^2`.
//!
//! On `dev_tfhe_n2048` this drops the per-PBS FFT count from 22 032 to
//! ~ 11 016 (a ~2× reduction) and removes the ~480 MB of redundant BSK
//! transforms that the naïve path was implicitly performing every PBS.
//!
//! The output of [`TfheBootstrapper::apply_lookup_table_fft`] is bit-exact
//! against [`TfheBootstrapper::apply_lookup_table_with`] using the same
//! [`FftMul`], because the only changes are *order of operations* — the
//! arithmetic is identical (4 buckets, 16-bit chunks, ψ-twist, IFFT,
//! round-and-recombine).

use silent_math::fft64::{
    FftBuckets, FftHalfBuckets, FftHalfPoly, FftMul, FftPoly, FftSignedHalfPoly, FftSignedPoly,
    TorusFft, TorusFftPoly,
};
use silent_math::torus::modulus_switch_u64;
use silent_params::{
    CiphertextModulusLog, DecompositionBaseLog, DecompositionLevelCount, GlweDimension,
    LweDimension, PolynomialSize,
};
use silent_ring::{GadgetDecomposition, NativePoly};
use silent_rlwe::{GgswCiphertext, GlweCiphertext, LweBootstrapKey, LweCiphertext};

use super::bootstrap::{
    TfheBootstrapper, build_lookup_table_polynomial, sample_extract, trivial_glwe_constant,
};
use super::encoding::TfheEncoder;
use super::keyswitch::LweKeyswitcher;

// ---------------------------------------------------------------------------
// Pre-FFT'd bootstrap key
// ---------------------------------------------------------------------------

/// A bootstrap key with every polynomial already in frequency domain.
///
/// Built once via [`Self::from_bsk`] and reused across PBS calls.  Memory
/// footprint is roughly `8 ×` the original [`LweBootstrapKey`] (each 8-byte
/// `u64` coefficient maps to 4 chunks of 16-byte complex pairs).  For
/// `dev_tfhe_n2048` that is ~480 MB — fine on a developer laptop, large
/// enough that production servers may want to control when the cache lives.
#[derive(Clone, Debug)]
pub struct LweBootstrapKeyFft {
    /// `storage[lwe_idx]` is a flat block of length `level * (k+1) * (k+1)`, indexed by
    /// `j * (k+1)*(k+1) + r * (k+1) + i`.
    storage: BootstrapKeyFftStorage,
    input_lwe_dimension: LweDimension,
    glwe_dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    log_modulus: u8,
}

#[derive(Clone, Debug)]
enum BootstrapKeyFftStorage {
    Full(Vec<Vec<FftPoly>>),
    Half(Vec<Vec<FftHalfPoly>>),
}

#[derive(Clone, Copy, Debug)]
enum GgswFftBlock<'a> {
    Full(&'a [FftPoly]),
    Half(&'a [FftHalfPoly]),
}

impl LweBootstrapKeyFft {
    /// Forward-transform every polynomial in `bsk` into split-precision
    /// frequency domain.  `fft.n()` must equal `bsk.polynomial_size()`.
    pub fn from_bsk(bsk: &LweBootstrapKey, fft: &FftMul) -> Self {
        let n = bsk.polynomial_size().0;
        assert_eq!(fft.n(), n, "FftMul size must match BSK polynomial size");
        let glwe_dim = bsk.glwe_dimension();
        let glwe_size = glwe_dim.glwe_size();
        let level = bsk.level().0 as usize;
        let use_half_storage =
            use_half_signed_external_product(bsk.base_log(), bsk.polynomial_size(), glwe_size);

        let storage = if use_half_storage {
            BootstrapKeyFftStorage::Half(
                (0..bsk.input_lwe_dimension().0)
                    .map(|i| {
                        let ggsw = bsk.get(i);
                        Self::transform_ggsw_half(ggsw, glwe_size, level, fft)
                    })
                    .collect(),
            )
        } else {
            BootstrapKeyFftStorage::Full(
                (0..bsk.input_lwe_dimension().0)
                    .map(|i| {
                        let ggsw = bsk.get(i);
                        Self::transform_ggsw(ggsw, glwe_size, level, fft)
                    })
                    .collect(),
            )
        };

        Self {
            storage,
            input_lwe_dimension: bsk.input_lwe_dimension(),
            glwe_dimension: glwe_dim,
            polynomial_size: bsk.polynomial_size(),
            base_log: bsk.base_log(),
            level: bsk.level(),
            log_modulus: bsk.log_modulus(),
        }
    }

    fn transform_ggsw(
        ggsw: &GgswCiphertext,
        glwe_size: usize,
        level: usize,
        fft: &FftMul,
    ) -> Vec<FftPoly> {
        let mut polys = Vec::with_capacity(level * glwe_size * glwe_size);
        for j in 0..level {
            for r in 0..glwe_size {
                let glwe_row = &ggsw.levels()[j].rows()[r];
                for poly in glwe_row.polys() {
                    polys.push(fft.forward_poly(poly.coeffs()));
                }
            }
        }
        polys
    }

    fn transform_ggsw_half(
        ggsw: &GgswCiphertext,
        glwe_size: usize,
        level: usize,
        fft: &FftMul,
    ) -> Vec<FftHalfPoly> {
        let mut polys = Vec::with_capacity(level * glwe_size * glwe_size);
        for j in 0..level {
            for r in 0..glwe_size {
                let glwe_row = &ggsw.levels()[j].rows()[r];
                for poly in glwe_row.polys() {
                    polys.push(fft.forward_poly_half(poly.coeffs()));
                }
            }
        }
        polys
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

    /// All entries belonging to the GGSW at `lwe_idx`, in
    /// `(j, r, i)`-row-major order.  Borrowed slice of length
    /// `level * (k+1) * (k+1)`.
    fn ggsw_block(&self, lwe_idx: usize) -> GgswFftBlock<'_> {
        match &self.storage {
            BootstrapKeyFftStorage::Full(entries) => GgswFftBlock::Full(&entries[lwe_idx]),
            BootstrapKeyFftStorage::Half(entries) => GgswFftBlock::Half(&entries[lwe_idx]),
        }
    }
}

fn signed_digit_external_product_is_precise(
    base_log: DecompositionBaseLog,
    polynomial_size: PolynomialSize,
) -> bool {
    (base_log.0 as u32) + polynomial_size.0.trailing_zeros() <= 37
}

fn use_half_signed_external_product(
    base_log: DecompositionBaseLog,
    polynomial_size: PolynomialSize,
    glwe_size: usize,
) -> bool {
    signed_digit_external_product_is_precise(base_log, polynomial_size) && glwe_size == 2
}

// ---------------------------------------------------------------------------
// External product / cmux / blind rotation in frequency domain
// ---------------------------------------------------------------------------

struct ExternalProductFftScratch {
    decomposed: Vec<NativePoly>,
    decomposed_pair: Vec<NativePoly>,
    digits: Vec<u64>,
    freq_acc: Vec<FftBuckets>,
    half_freq_acc: Vec<FftHalfBuckets>,
    digit_fft: FftPoly,
    signed_digit_fft: FftSignedPoly,
    signed_digit_half_fft: FftSignedHalfPoly,
    signed_digit_half_fft_pair: FftSignedHalfPoly,
}

impl ExternalProductFftScratch {
    fn new(
        fft: &FftMul,
        glwe_dim: GlweDimension,
        polynomial_size: PolynomialSize,
        base_log: DecompositionBaseLog,
        level: DecompositionLevelCount,
        log_modulus: CiphertextModulusLog,
    ) -> Self {
        let n = polynomial_size.0;
        let level_count = level.0 as usize;
        let glwe_size = glwe_dim.glwe_size();
        let use_half_signed_acc =
            use_half_signed_external_product(base_log, polynomial_size, glwe_size);
        Self {
            decomposed: (0..level_count)
                .map(|_| NativePoly::zeros(n, log_modulus.0))
                .collect(),
            decomposed_pair: (0..level_count)
                .map(|_| NativePoly::zeros(n, log_modulus.0))
                .collect(),
            digits: vec![0u64; level_count],
            freq_acc: if use_half_signed_acc {
                Vec::new()
            } else {
                (0..glwe_size).map(|_| fft.buckets_zero()).collect()
            },
            half_freq_acc: if use_half_signed_acc {
                (0..glwe_size).map(|_| fft.half_buckets_zero()).collect()
            } else {
                Vec::new()
            },
            digit_fft: if use_half_signed_acc {
                FftPoly::zeros(0)
            } else {
                fft.empty_poly()
            },
            signed_digit_fft: if use_half_signed_acc {
                FftSignedPoly::zeros(0)
            } else {
                fft.empty_signed_poly()
            },
            signed_digit_half_fft: fft.empty_signed_half_poly(),
            signed_digit_half_fft_pair: fft.empty_signed_half_poly(),
        }
    }
}

fn decompose_poly_into(
    poly: &NativePoly,
    gadget: &GadgetDecomposition,
    digits: &mut [u64],
    out: &mut [NativePoly],
) {
    debug_assert_eq!(out.len(), gadget.level() as usize);
    debug_assert_eq!(digits.len(), gadget.level() as usize);
    let n = poly.degree();

    if gadget.level() == 1 {
        let shift = gadget.level_shift(0);
        let base = gadget.base();
        let half_base = base >> 1;
        let mask = base - 1;
        let coeffs = poly.coeffs();
        let out_coeffs = out[0].coeffs_mut();
        if shift == 0 {
            for k in 0..n {
                let raw = coeffs[k] & mask;
                out_coeffs[k] = if raw > half_base {
                    raw.wrapping_sub(base)
                } else {
                    raw
                };
            }
        } else {
            let rounding = 1u64 << (shift - 1);
            for k in 0..n {
                let raw = (coeffs[k].wrapping_add(rounding) >> shift) & mask;
                out_coeffs[k] = if raw > half_base {
                    raw.wrapping_sub(base)
                } else {
                    raw
                };
            }
        }
        return;
    }

    for k in 0..n {
        gadget.decompose_into(poly.coeffs()[k], digits);
        for (j, d) in digits.iter().enumerate() {
            out[j].coeffs_mut()[k] = *d;
        }
    }
}

/// Compute `bsk_fft_block ⊠ glwe`: external product of one pre-FFT'd GGSW
/// against an in-the-clear GLWE ciphertext.
///
/// Output shape matches [`super::bootstrap::build_external_product`].  All
/// frequency-domain accumulation happens inside this function; the only
/// `u64` polynomial operations performed are gadget decomposition and the
/// final inverse-recombine.
fn build_external_product_fft_into(
    bsk_fft_block: GgswFftBlock<'_>,
    glwe: &GlweCiphertext,
    addend: &GlweCiphertext,
    fft: &FftMul,
    glwe_dim: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    log_modulus: CiphertextModulusLog,
    scratch: &mut ExternalProductFftScratch,
    result: &mut GlweCiphertext,
) {
    debug_assert_eq!(glwe.glwe_dimension(), glwe_dim);
    debug_assert_eq!(glwe.polynomial_size(), polynomial_size);
    debug_assert_eq!(glwe.log_modulus(), log_modulus.0);
    debug_assert_eq!(addend.glwe_dimension(), glwe_dim);
    debug_assert_eq!(addend.polynomial_size(), polynomial_size);
    debug_assert_eq!(addend.log_modulus(), log_modulus.0);
    debug_assert_eq!(result.glwe_dimension(), glwe_dim);
    debug_assert_eq!(result.polynomial_size(), polynomial_size);
    debug_assert_eq!(result.log_modulus(), log_modulus.0);

    let glwe_size = glwe_dim.glwe_size();
    let level_count = level.0 as usize;
    let signed_digit_is_precise =
        signed_digit_external_product_is_precise(base_log, polynomial_size);
    let use_half_signed_acc =
        use_half_signed_external_product(base_log, polynomial_size, glwe_size);
    let bsk_fft_block_len = match bsk_fft_block {
        GgswFftBlock::Full(block) => block.len(),
        GgswFftBlock::Half(block) => block.len(),
    };
    debug_assert_eq!(
        bsk_fft_block_len,
        level_count * glwe_size * glwe_size,
        "BSK block shape mismatch"
    );

    let gadget = GadgetDecomposition::new(log_modulus.0, base_log.0, level.0);
    if use_half_signed_acc {
        for acc in scratch.half_freq_acc.iter_mut() {
            acc.clear();
        }
    } else {
        for acc in scratch.freq_acc.iter_mut() {
            acc.clear();
        }
    }

    if use_half_signed_acc {
        let (full_bsk_fft_block, half_bsk_fft_block) = match bsk_fft_block {
            GgswFftBlock::Full(block) => (Some(block), None),
            GgswFftBlock::Half(block) => (None, Some(block)),
        };
        decompose_poly_into(
            &glwe.polys()[0],
            &gadget,
            &mut scratch.digits,
            &mut scratch.decomposed,
        );
        decompose_poly_into(
            &glwe.polys()[1],
            &gadget,
            &mut scratch.digits,
            &mut scratch.decomposed_pair,
        );
        for j in 0..level_count {
            fft.forward_signed_poly_pair_half_into(
                scratch.decomposed[j].coeffs(),
                scratch.decomposed_pair[j].coeffs(),
                &mut scratch.signed_digit_half_fft,
                &mut scratch.signed_digit_half_fft_pair,
            );

            let row0_base = j * glwe_size * glwe_size;
            let row1_base = row0_base + glwe_size;
            if let Some(block) = half_bsk_fft_block {
                for i in 0..glwe_size {
                    fft.fma_signed_half_lhs_half_rhs_freq_half_into(
                        &scratch.signed_digit_half_fft,
                        &block[row0_base + i],
                        &mut scratch.half_freq_acc[i],
                    );
                    fft.fma_signed_half_lhs_half_rhs_freq_half_into(
                        &scratch.signed_digit_half_fft_pair,
                        &block[row1_base + i],
                        &mut scratch.half_freq_acc[i],
                    );
                }
            } else if let Some(block) = full_bsk_fft_block {
                for i in 0..glwe_size {
                    fft.fma_signed_half_lhs_freq_half_into(
                        &scratch.signed_digit_half_fft,
                        &block[row0_base + i],
                        &mut scratch.half_freq_acc[i],
                    );
                    fft.fma_signed_half_lhs_freq_half_into(
                        &scratch.signed_digit_half_fft_pair,
                        &block[row1_base + i],
                        &mut scratch.half_freq_acc[i],
                    );
                }
            }
        }
    } else {
        let GgswFftBlock::Full(bsk_fft_block) = bsk_fft_block else {
            panic!("half-spectrum BSK cache can only be used by the signed GLWE-size-2 path");
        };
        for r in 0..glwe_size {
            decompose_poly_into(
                &glwe.polys()[r],
                &gadget,
                &mut scratch.digits,
                &mut scratch.decomposed,
            );
            for j in 0..level_count {
                // Stride: at fixed (j, r), entry index is j*glwe_size^2 + r*glwe_size + i.
                let row_base = j * glwe_size * glwe_size + r * glwe_size;
                if signed_digit_is_precise {
                    fft.forward_signed_poly_into(
                        scratch.decomposed[j].coeffs(),
                        &mut scratch.signed_digit_fft,
                    );
                    for i in 0..glwe_size {
                        let ggsw_fft = &bsk_fft_block[row_base + i];
                        fft.fma_signed_lhs_freq_into(
                            &scratch.signed_digit_fft,
                            ggsw_fft,
                            &mut scratch.freq_acc[i],
                        );
                    }
                } else {
                    fft.forward_poly_into(scratch.decomposed[j].coeffs(), &mut scratch.digit_fft);
                    for i in 0..glwe_size {
                        let ggsw_fft = &bsk_fft_block[row_base + i];
                        fft.fma_freq_into(&scratch.digit_fft, ggsw_fft, &mut scratch.freq_acc[i]);
                    }
                }
            }
        }
    }

    for i in 0..glwe_size {
        let poly = &mut result.polys_mut()[i];
        let addend_coeffs = addend.polys()[i].coeffs();
        if use_half_signed_acc {
            fft.inverse_combine_half_write_with_addend_into(
                &scratch.half_freq_acc[i],
                addend_coeffs,
                poly.coeffs_mut(),
            );
        } else {
            fft.inverse_combine_write_with_addend_into(
                &mut scratch.freq_acc[i],
                addend_coeffs,
                poly.coeffs_mut(),
            );
        }
        poly.reduce();
    }
}

/// FFT-cached blind rotation.  Equivalent to [`super::bootstrap::blind_rotate_assign`]
/// but operates on a [`LweBootstrapKeyFft`].
pub fn blind_rotate_assign_fft(
    acc: &mut GlweCiphertext,
    lwe: &LweCiphertext,
    bsk_fft: &LweBootstrapKeyFft,
    fft: &FftMul,
) {
    debug_assert_eq!(lwe.dimension(), bsk_fft.input_lwe_dimension());
    debug_assert_eq!(acc.glwe_dimension(), bsk_fft.glwe_dimension());
    debug_assert_eq!(acc.polynomial_size(), bsk_fft.polynomial_size());
    debug_assert_eq!(acc.log_modulus(), bsk_fft.log_modulus());

    let glwe_dim = bsk_fft.glwe_dimension();
    let polynomial_size = bsk_fft.polynomial_size();
    let base_log = bsk_fft.base_log();
    let level = bsk_fft.level();
    let log_modulus = CiphertextModulusLog(bsk_fft.log_modulus());

    let n = polynomial_size.0;
    let two_n = n << 1;
    let log_two_n = (two_n as u64).trailing_zeros() as u8;
    let log_q = lwe.log_modulus();

    let ms_b = modulus_switch_u64(lwe.body(), log_q, log_two_n) as usize;
    for p in acc.polys_mut() {
        p.wrapping_monic_monomial_div_assign(ms_b);
    }

    let mut cmux_diff = GlweCiphertext::zeros(glwe_dim, polynomial_size, log_modulus);
    let mut cmux_out = GlweCiphertext::zeros(glwe_dim, polynomial_size, log_modulus);
    let mut scratch = ExternalProductFftScratch::new(
        fft,
        glwe_dim,
        polynomial_size,
        base_log,
        level,
        log_modulus,
    );

    for (i, &a_i) in lwe.mask().iter().enumerate() {
        let ms_a = modulus_switch_u64(a_i, log_q, log_two_n) as usize;
        if ms_a == 0 {
            continue;
        }

        for (dst, src) in cmux_diff.polys_mut().iter_mut().zip(acc.polys().iter()) {
            dst.wrapping_monic_monomial_mul_and_subtract_from(src, ms_a);
        }
        build_external_product_fft_into(
            bsk_fft.ggsw_block(i),
            &cmux_diff,
            acc,
            fft,
            glwe_dim,
            polynomial_size,
            base_log,
            level,
            log_modulus,
            &mut scratch,
            &mut cmux_out,
        );
        std::mem::swap(acc, &mut cmux_out);
    }
}

// ---------------------------------------------------------------------------
// Torus Fourier PBS Implementation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LweBootstrapKeyTorusFft {
    storage: Vec<Vec<TorusFftPoly>>,
    input_lwe_dimension: LweDimension,
    glwe_dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    log_modulus: u8,
}

impl LweBootstrapKeyTorusFft {
    pub fn from_bsk(bsk: &LweBootstrapKey, fft: &TorusFft) -> Self {
        assert_eq!(
            bsk.log_modulus(),
            64,
            "TorusFft backend requires ciphertext_modulus_log == 64, got {}",
            bsk.log_modulus()
        );
        let glwe_dim = bsk.glwe_dimension();
        let glwe_size = glwe_dim.glwe_size();
        let level = bsk.level().0 as usize;

        let storage = (0..bsk.input_lwe_dimension().0)
            .map(|i| {
                let ggsw = bsk.get(i);
                let mut polys = Vec::with_capacity(level * glwe_size * glwe_size);
                for j in 0..level {
                    for r in 0..glwe_size {
                        let glwe_row = &ggsw.levels()[j].rows()[r];
                        for poly in glwe_row.polys() {
                            let mut torus_poly = fft.empty_poly();
                            fft.forward_torus_poly_into(poly.coeffs(), &mut torus_poly);
                            polys.push(torus_poly);
                        }
                    }
                }
                polys
            })
            .collect();

        Self {
            storage,
            input_lwe_dimension: bsk.input_lwe_dimension(),
            glwe_dimension: glwe_dim,
            polynomial_size: bsk.polynomial_size(),
            base_log: bsk.base_log(),
            level: bsk.level(),
            log_modulus: bsk.log_modulus(),
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

    fn ggsw_block(&self, lwe_idx: usize) -> &[TorusFftPoly] {
        &self.storage[lwe_idx]
    }
}

struct ExternalProductTorusFftScratch {
    decomposed: Vec<NativePoly>,
    digits: Vec<u64>,
    freq_acc: Vec<TorusFftPoly>,
    digit_fft: TorusFftPoly,
}

impl ExternalProductTorusFftScratch {
    fn new(
        fft: &TorusFft,
        glwe_dim: GlweDimension,
        polynomial_size: PolynomialSize,
        level: DecompositionLevelCount,
        log_modulus: CiphertextModulusLog,
    ) -> Self {
        let n = polynomial_size.0;
        let level_count = level.0 as usize;
        let glwe_size = glwe_dim.glwe_size();
        Self {
            decomposed: (0..level_count)
                .map(|_| NativePoly::zeros(n, log_modulus.0))
                .collect(),
            digits: vec![0u64; level_count],
            freq_acc: (0..glwe_size).map(|_| fft.empty_poly()).collect(),
            digit_fft: fft.empty_poly(),
        }
    }
}

fn build_external_product_torus_fft_into(
    bsk_fft_block: &[TorusFftPoly],
    glwe: &GlweCiphertext,
    addend: &GlweCiphertext,
    fft: &TorusFft,
    glwe_dim: GlweDimension,
    _polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    log_modulus: CiphertextModulusLog,
    scratch: &mut ExternalProductTorusFftScratch,
    result: &mut GlweCiphertext,
) {
    let glwe_size = glwe_dim.glwe_size();
    let level_count = level.0 as usize;
    let gadget = GadgetDecomposition::new(log_modulus.0, base_log.0, level.0);

    for acc in scratch.freq_acc.iter_mut() {
        acc.clear();
    }

    for r in 0..glwe_size {
        decompose_poly_into(
            &glwe.polys()[r],
            &gadget,
            &mut scratch.digits,
            &mut scratch.decomposed,
        );
        for j in 0..level_count {
            let row_base = j * glwe_size * glwe_size + r * glwe_size;
            fft.forward_signed_integer_poly_into(
                scratch.decomposed[j].coeffs(),
                &mut scratch.digit_fft,
            );
            for i in 0..glwe_size {
                fft.fma_freq_into(
                    &scratch.digit_fft,
                    &bsk_fft_block[row_base + i],
                    &mut scratch.freq_acc[i],
                );
            }
        }
    }

    for i in 0..glwe_size {
        let poly = &mut result.polys_mut()[i];
        let addend_coeffs = addend.polys()[i].coeffs();
        fft.inverse_torus_write_with_addend_into(
            &mut scratch.freq_acc[i],
            addend_coeffs,
            poly.coeffs_mut(),
        );
        poly.reduce();
    }
}

pub fn blind_rotate_assign_torus_fft(
    acc: &mut GlweCiphertext,
    lwe: &LweCiphertext,
    bsk_fft: &LweBootstrapKeyTorusFft,
    fft: &TorusFft,
) {
    let glwe_dim = bsk_fft.glwe_dimension();
    let polynomial_size = bsk_fft.polynomial_size();
    let base_log = bsk_fft.base_log();
    let level = bsk_fft.level();
    let log_modulus = CiphertextModulusLog(bsk_fft.log_modulus());

    let n = polynomial_size.0;
    let two_n = n << 1;
    let log_two_n = (two_n as u64).trailing_zeros() as u8;
    let log_q = lwe.log_modulus();

    let ms_b = modulus_switch_u64(lwe.body(), log_q, log_two_n) as usize;
    for p in acc.polys_mut() {
        p.wrapping_monic_monomial_div_assign(ms_b);
    }

    let mut cmux_diff = GlweCiphertext::zeros(glwe_dim, polynomial_size, log_modulus);
    let mut cmux_out = GlweCiphertext::zeros(glwe_dim, polynomial_size, log_modulus);
    let mut scratch =
        ExternalProductTorusFftScratch::new(fft, glwe_dim, polynomial_size, level, log_modulus);

    for (i, &a_i) in lwe.mask().iter().enumerate() {
        let ms_a = modulus_switch_u64(a_i, log_q, log_two_n) as usize;
        if ms_a == 0 {
            continue;
        }

        for (dst, src) in cmux_diff.polys_mut().iter_mut().zip(acc.polys().iter()) {
            dst.wrapping_monic_monomial_mul_and_subtract_from(src, ms_a);
        }
        build_external_product_torus_fft_into(
            bsk_fft.ggsw_block(i),
            &cmux_diff,
            acc,
            fft,
            glwe_dim,
            polynomial_size,
            base_log,
            level,
            log_modulus,
            &mut scratch,
            &mut cmux_out,
        );
        std::mem::swap(acc, &mut cmux_out);
    }
}

// ---------------------------------------------------------------------------
// TfheBootstrapper integration
// ---------------------------------------------------------------------------

impl<'a> TfheBootstrapper<'a> {
    /// Pre-compute the frequency-domain bootstrap-key cache.  Construct
    /// once per (params, bsk) pair and reuse across many PBS calls — the
    /// resulting [`LweBootstrapKeyFft`] is what makes
    /// [`Self::apply_lookup_table_fft`] much faster than re-FFTing the BSK
    /// inside every external product.
    pub fn build_fft_bsk_cache(&self, fft: &FftMul) -> LweBootstrapKeyFft {
        LweBootstrapKeyFft::from_bsk(self.bsk(), fft)
    }

    /// FFT-driven PBS using a pre-computed [`LweBootstrapKeyFft`].
    ///
    /// Bit-exact equivalent of [`Self::apply_lookup_table_with`] passing the
    /// same [`FftMul`], but skips the redundant forward FFTs of every BSK
    /// polynomial that the naïve path performs `lwe_dim` times each PBS.
    pub fn apply_lookup_table_fft<F>(
        &self,
        encoder: &TfheEncoder,
        bsk_fft: &LweBootstrapKeyFft,
        fft: &FftMul,
        ct: &LweCiphertext,
        f: F,
    ) -> LweCiphertext
    where
        F: FnMut(u64) -> u64,
    {
        debug_assert_eq!(bsk_fft.polynomial_size(), self.params().polynomial_size());
        debug_assert_eq!(
            bsk_fft.log_modulus(),
            self.params().ciphertext_modulus_log().0
        );
        debug_assert_eq!(fft.n(), self.params().polynomial_size().0);

        let lut_poly = build_lookup_table_polynomial(
            encoder,
            self.params().polynomial_size(),
            self.params().ciphertext_modulus_log(),
            f,
        );
        let mut acc = trivial_glwe_constant(
            self.params().glwe_dimension(),
            self.params().polynomial_size(),
            self.params().ciphertext_modulus_log(),
            lut_poly,
        );
        blind_rotate_assign_fft(&mut acc, ct, bsk_fft, fft);
        let big_lwe = sample_extract(&acc, 0);
        LweKeyswitcher::new(self.ksk()).keyswitch(&big_lwe)
    }

    /// FFT-driven PBS using a caller-built LUT polynomial.
    ///
    /// Mirrors [`Self::apply_lookup_table_with_polynomial`] but runs on the
    /// cached Fourier bootstrap key so repeated shortint PBS calls do not
    /// re-FFT the BSK.
    pub fn apply_lookup_table_fft_with_polynomial(
        &self,
        bsk_fft: &LweBootstrapKeyFft,
        fft: &FftMul,
        ct: &LweCiphertext,
        lut_poly: &NativePoly,
    ) -> LweCiphertext {
        debug_assert_eq!(bsk_fft.polynomial_size(), self.params().polynomial_size());
        debug_assert_eq!(
            bsk_fft.log_modulus(),
            self.params().ciphertext_modulus_log().0
        );
        debug_assert_eq!(fft.n(), self.params().polynomial_size().0);

        let mut acc = GlweCiphertext::zeros(
            self.params().glwe_dimension(),
            self.params().polynomial_size(),
            self.params().ciphertext_modulus_log(),
        );
        acc.body_mut().clone_from(lut_poly);
        blind_rotate_assign_fft(&mut acc, ct, bsk_fft, fft);
        let big_lwe = sample_extract(&acc, 0);
        LweKeyswitcher::new(self.ksk()).keyswitch(&big_lwe)
    }

    pub fn build_torus_fft_bsk_cache(&self, fft: &TorusFft) -> LweBootstrapKeyTorusFft {
        LweBootstrapKeyTorusFft::from_bsk(self.bsk(), fft)
    }

    pub fn apply_lookup_table_torus_fft<F>(
        &self,
        encoder: &TfheEncoder,
        bsk_fft: &LweBootstrapKeyTorusFft,
        fft: &TorusFft,
        ct: &LweCiphertext,
        f: F,
    ) -> LweCiphertext
    where
        F: FnMut(u64) -> u64,
    {
        assert_eq!(bsk_fft.polynomial_size(), self.params().polynomial_size());
        assert_eq!(
            bsk_fft.log_modulus(),
            self.params().ciphertext_modulus_log().0
        );
        assert_eq!(fft.n(), self.params().polynomial_size().0);

        let lut_poly = build_lookup_table_polynomial(
            encoder,
            self.params().polynomial_size(),
            self.params().ciphertext_modulus_log(),
            f,
        );
        let mut acc = trivial_glwe_constant(
            self.params().glwe_dimension(),
            self.params().polynomial_size(),
            self.params().ciphertext_modulus_log(),
            lut_poly,
        );
        blind_rotate_assign_torus_fft(&mut acc, ct, bsk_fft, fft);
        let big_lwe = sample_extract(&acc, 0);
        LweKeyswitcher::new(self.ksk()).keyswitch(&big_lwe)
    }

    pub fn apply_lookup_table_torus_fft_with_polynomial(
        &self,
        bsk_fft: &LweBootstrapKeyTorusFft,
        fft: &TorusFft,
        ct: &LweCiphertext,
        lut_poly: &NativePoly,
    ) -> LweCiphertext {
        assert_eq!(bsk_fft.polynomial_size(), self.params().polynomial_size());
        assert_eq!(
            bsk_fft.log_modulus(),
            self.params().ciphertext_modulus_log().0
        );
        assert_eq!(fft.n(), self.params().polynomial_size().0);

        let mut acc = GlweCiphertext::zeros(
            self.params().glwe_dimension(),
            self.params().polynomial_size(),
            self.params().ciphertext_modulus_log(),
        );
        acc.body_mut().clone_from(lut_poly);
        blind_rotate_assign_torus_fft(&mut acc, ct, bsk_fft, fft);
        let big_lwe = sample_extract(&acc, 0);
        LweKeyswitcher::new(self.ksk()).keyswitch(&big_lwe)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemes::tfhe::bootstrap::generate_pbs_keys;
    use crate::schemes::tfhe::crypto::TfheEncryptor;
    use crate::schemes::tfhe::encoding::TfheEncoder;
    use crate::schemes::tfhe::params::TfheParameters;
    use rand_core::SeedableRng;
    use silent_params::{
        CarryModulus, DecompositionBaseLog, DecompositionLevelCount, GlweDimension, Log2PFail,
        LweDimension, MessageModulus, NoiseDistribution, PolynomialSize, presets,
    };
    use silent_utils::rng::SecureRng;

    /// Micro parameter set mirrors the one used by the existing bootstrap
    /// tests so the bit-exact comparison is meaningful.
    fn micro_params() -> TfheParameters {
        let mut p = presets::toy::toy_tfhe_n512();
        p.lwe_dimension = LweDimension(8);
        p.polynomial_size = PolynomialSize(64);
        p.glwe_dimension = GlweDimension(1);
        p.message_modulus = MessageModulus(4);
        p.carry_modulus = CarryModulus(4);
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
    fn fft_bsk_cache_pbs_matches_uncached_fft_bit_exact() {
        let params = micro_params();
        let mut keygen_rng = SecureRng::from_seed([21u8; 32]);
        let (sk_small, _sk_glwe, bsk, ksk) = generate_pbs_keys(params.clone(), &mut keygen_rng);
        let encoder = TfheEncoder::new(params.clone());
        let fft = FftMul::new(params.polynomial_size().0);

        let mut enc = TfheEncryptor::new(params.clone(), SecureRng::from_seed([22u8; 32]));
        let bs = TfheBootstrapper::new(params.clone(), &bsk, &ksk);
        let bsk_fft = bs.build_fft_bsk_cache(&fft);

        let identity = |m: u64| m;

        let n_slots = encoder.params().total_message_modulus();
        for v in 0..n_slots {
            let ct = enc.encrypt_symmetric(&sk_small, encoder.encode_full(v));

            let cached = bs.apply_lookup_table_fft(&encoder, &bsk_fft, &fft, &ct, identity);
            let uncached = bs.apply_lookup_table_with(&encoder, &ct, identity, &fft);

            assert_eq!(
                cached.as_slice(),
                uncached.as_slice(),
                "FFT-cached vs uncached PBS must be bit-exact (v = {v})"
            );
        }
    }

    #[test]
    fn torus_fft_bsk_cache_pbs_decrypt_correct() {
        let params = micro_params();
        let mut keygen_rng = SecureRng::from_seed([31u8; 32]);
        let (sk_small, _sk_glwe, bsk, ksk) = generate_pbs_keys(params.clone(), &mut keygen_rng);
        let encoder = TfheEncoder::new(params.clone());
        let n = params.polynomial_size().0;

        let mut enc = TfheEncryptor::new(params.clone(), SecureRng::from_seed([32u8; 32]));
        let bs = TfheBootstrapper::new(params.clone(), &bsk, &ksk);
        let dec =
            crate::schemes::tfhe::crypto::TfheDecryptor::new(params.clone(), sk_small.clone());

        let torus_fft = TorusFft::new(n);
        let bsk_torus = bs.build_torus_fft_bsk_cache(&torus_fft);

        let identity = |m: u64| m;
        let n_slots = encoder.params().total_message_modulus();

        for v in 0..n_slots {
            let ct = enc.encrypt_symmetric(&sk_small, encoder.encode_full(v));
            let result =
                bs.apply_lookup_table_torus_fft(&encoder, &bsk_torus, &torus_fft, &ct, identity);
            let decrypted = dec.decrypt_full(&result, &encoder);
            assert_eq!(
                decrypted, v,
                "Torus-FFT PBS decrypt mismatch: expected {v}, got {decrypted}"
            );
        }
    }

    #[test]
    fn torus_fft_vs_exact_fft_decrypt_agree() {
        let params = micro_params();
        let mut keygen_rng = SecureRng::from_seed([41u8; 32]);
        let (sk_small, _sk_glwe, bsk, ksk) = generate_pbs_keys(params.clone(), &mut keygen_rng);
        let encoder = TfheEncoder::new(params.clone());
        let n = params.polynomial_size().0;

        let mut enc = TfheEncryptor::new(params.clone(), SecureRng::from_seed([42u8; 32]));
        let bs = TfheBootstrapper::new(params.clone(), &bsk, &ksk);
        let dec =
            crate::schemes::tfhe::crypto::TfheDecryptor::new(params.clone(), sk_small.clone());

        let fft_exact = FftMul::new(n);
        let bsk_exact = bs.build_fft_bsk_cache(&fft_exact);

        let torus_fft = TorusFft::new(n);
        let bsk_torus = bs.build_torus_fft_bsk_cache(&torus_fft);

        let identity = |m: u64| m;
        let n_slots = encoder.params().total_message_modulus();

        for v in 0..n_slots {
            let ct = enc.encrypt_symmetric(&sk_small, encoder.encode_full(v));

            let result_exact =
                bs.apply_lookup_table_fft(&encoder, &bsk_exact, &fft_exact, &ct, identity);
            let result_torus =
                bs.apply_lookup_table_torus_fft(&encoder, &bsk_torus, &torus_fft, &ct, identity);

            let dec_exact = dec.decrypt_full(&result_exact, &encoder);
            let dec_torus = dec.decrypt_full(&result_torus, &encoder);
            assert_eq!(
                dec_exact, dec_torus,
                "Exact-FFT vs Torus-FFT decrypt mismatch at v={v}: exact={dec_exact}, torus={dec_torus}"
            );
        }
    }
}
