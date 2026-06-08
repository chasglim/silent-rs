//! TFHE -> BFV torus-native functional switching.
//!
//! This module implements the torus-native functional-switching path for
//! toy/audit examples:
//!
//! * `KSi = TRGSW_K(S_i)` is represented by SILENT's GGSW container, with
//!   target GLWE key `K = -s_BFV` so the resulting GLWE can be modulus-switched
//!   directly into SILENT's BFV convention `c0 + c1 * s_BFV`.
//! * PBS normalization maps SILENT TFHE integer messages to exact BFV
//!   plaintext lattice points `z/p in T`.
//! * Functional switching applies the public `VDM^{-1} * G` transform to TLWE
//!   bodies/masks and uses external products with the TRGSW key to remove the
//!   TFHE secret-key terms.

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};
use silent_math::fft64::SchoolbookMul;
use silent_params::{CiphertextModulusLog, GlweDimension, PolynomialSize};
use silent_ring::{NativePoly, Poly};
use silent_rlwe::{
    Ciphertext, GgswCiphertextList, GlweCiphertext, GlweSecretKey, LweBootstrapKey, LweCiphertext,
    LweSecretKey, SecretKey,
};
use silent_utils::rng::SecureRng;

use crate::bridge::keys::bfv_secret_key_as_lwe;
use crate::bridge::tfhe_to_bfv::matrix;
use crate::bridge::{BridgeError, BridgeParams};
use crate::schemes::tfhe::bootstrap::{
    TfheBootstrapper, build_external_product, encrypt_ggsw_binary,
};
use crate::schemes::tfhe::encoding::TfheEncoder;

#[derive(Clone, Debug)]
pub struct TfheToBfvFunctionalKey {
    trgsw: LweBootstrapKey,
}

impl TfheToBfvFunctionalKey {
    pub fn new(trgsw: LweBootstrapKey) -> Self {
        Self { trgsw }
    }

    pub fn trgsw(&self) -> &LweBootstrapKey {
        &self.trgsw
    }
}

/// Generate TFHE-to-BFV functional switching key material:
/// `KSi = TRGSW_K(S_i)` for every TFHE LWE secret-key coefficient `S_i`.
pub fn generate_tfhe_to_bfv_functional_key(
    rng: &mut SecureRng,
    params: &BridgeParams,
    bfv_sk: &SecretKey,
    tfhe_sk: &LweSecretKey,
) -> Result<TfheToBfvFunctionalKey, BridgeError> {
    ensure_functional_switch_ring_shape(params)?;
    if tfhe_sk.dimension().0 != params.tfhe_params.lwe_dimension().0 {
        return Err(BridgeError::IncompatibleParameters(format!(
            "SILENT functional key expected TFHE LWE dimension {}, got {}",
            params.tfhe_params.lwe_dimension().0,
            tfhe_sk.dimension().0
        )));
    }

    let target_glwe_sk = bfv_secret_key_as_negated_glwe(bfv_sk, params)?;
    let mut trgsw_list = Vec::with_capacity(tfhe_sk.dimension().0);
    for &secret_bit in tfhe_sk.data() {
        if secret_bit > 1 {
            return Err(BridgeError::IncompatibleParameters(format!(
                "SILENT functional switching requires binary TFHE secret bits, got {secret_bit}"
            )));
        }
        trgsw_list.push(encrypt_ggsw_binary(
            rng,
            &target_glwe_sk,
            secret_bit,
            params.tfhe_params.pbs_base_log(),
            params.tfhe_params.pbs_level(),
            params.tfhe_params.glwe_noise(),
            params.tfhe_params.ciphertext_modulus_log(),
        ));
    }

    Ok(TfheToBfvFunctionalKey::new(LweBootstrapKey::new(
        GgswCiphertextList::new(trgsw_list),
        tfhe_sk.dimension(),
        GlweDimension(1),
        PolynomialSize(params.bfv_params.degree()),
        params.tfhe_params.pbs_base_log(),
        params.tfhe_params.pbs_level(),
        params.tfhe_params.ciphertext_modulus_log(),
    )))
}

/// PBS-normalize a TFHE LWE ciphertext into an exact BFV `1/p` torus lattice.
///
/// The input is decoded according to SILENT's existing TFHE integer encoder.
/// The output remains under the TFHE LWE key and has raw torus message
/// `round(q_T * map(m) / p)`, suitable as input to TFHE-to-BFV functional switching.
pub fn normalize_lwe_to_bfv_torus_native_with_pbs<F>(
    ct: &LweCiphertext,
    params: &BridgeParams,
    encoder: &TfheEncoder,
    bootstrapper: &TfheBootstrapper<'_>,
    map: F,
) -> Result<LweCiphertext, BridgeError>
where
    F: FnMut(u64) -> u64,
{
    let lut = build_torus_native_normalization_lut(
        encoder,
        params.tfhe_params.polynomial_size(),
        params.tfhe_params.ciphertext_modulus_log(),
        params.bfv_params.plain_modulus(),
        map,
    )?;
    let backend = SchoolbookMul;
    Ok(bootstrapper.apply_lookup_table_with_polynomial(ct, lut, &backend))
}

pub fn build_torus_native_normalization_lut<F>(
    encoder: &TfheEncoder,
    polynomial_size: PolynomialSize,
    log_modulus: CiphertextModulusLog,
    bfv_plaintext_modulus: u64,
    mut map: F,
) -> Result<NativePoly, BridgeError>
where
    F: FnMut(u64) -> u64,
{
    if bfv_plaintext_modulus <= 1 {
        return Err(BridgeError::IncompatibleParameters(format!(
            "SILENT PBS normalization requires BFV plaintext modulus p > 1, got {bfv_plaintext_modulus}"
        )));
    }

    let total = encoder.full_plaintext_modulus();
    let n = polynomial_size.0;
    if n < total as usize {
        return Err(BridgeError::IncompatibleParameters(format!(
            "SILENT PBS normalization LUT needs polynomial_size >= TFHE full plaintext modulus; got N={n}, T={total}"
        )));
    }
    let box_size = n / total as usize;
    let half_box = box_size / 2;

    let mut raw = vec![0u64; n];
    for m in 0..total {
        let value = map(m) % bfv_plaintext_modulus;
        let encoded = torus_native_encode(value, bfv_plaintext_modulus, log_modulus.0)?;
        let start = m as usize * box_size;
        let end = (start + box_size).min(n);
        for slot in &mut raw[start..end] {
            *slot = encoded;
        }
    }

    for slot in &mut raw[0..half_box] {
        *slot = slot.wrapping_neg();
    }
    raw.rotate_left(half_box);

    Ok(NativePoly::from_u64(&raw, log_modulus.0))
}

/// TFHE-to-BFV functional switching for coefficient-layout BFV messages.
pub fn tfhe_to_bfv_functional_coefficients(
    tfhe_cts: &[LweCiphertext],
    params: &BridgeParams,
    key: &TfheToBfvFunctionalKey,
) -> Result<Ciphertext, BridgeError> {
    let degree = params.bfv_params.degree();
    let mut coeff_map = vec![vec![0u64; tfhe_cts.len()]; degree];
    for i in 0..tfhe_cts.len().min(degree) {
        coeff_map[i][i] = 1;
    }
    tfhe_to_bfv_with_functional_coefficient_map(tfhe_cts, params, key, &coeff_map)
}

/// TFHE-to-BFV functional switching for BFV SIMD slots.
///
/// This uses `G = [I_k; 0]` and `G' = VDM^{-1} * G`, so the first `k` BFV
/// slots decode to the input TLWE messages.
pub fn tfhe_to_bfv_functional_slots(
    tfhe_cts: &[LweCiphertext],
    params: &BridgeParams,
    key: &TfheToBfvFunctionalKey,
) -> Result<Ciphertext, BridgeError> {
    let degree = params.bfv_params.degree();
    let plaintext_modulus = params.bfv_params.plain_modulus();
    let slot_to_coeff = matrix::generate_slot_to_coeff_matrix(degree, plaintext_modulus)
        .ok_or_else(|| {
            BridgeError::IncompatibleParameters(format!(
                "SILENT slot switching requires X^N+1 to split modulo p; got N={degree}, p={plaintext_modulus}"
            ))
        })?;
    let k = tfhe_cts.len();
    if k > degree {
        return Err(BridgeError::IncompatibleParameters(format!(
            "SILENT slot switching received {k} TLWE ciphertexts but BFV degree is {degree}"
        )));
    }
    let coeff_map = slot_to_coeff
        .iter()
        .map(|row| row[..k].to_vec())
        .collect::<Vec<_>>();
    tfhe_to_bfv_with_functional_coefficient_map(tfhe_cts, params, key, &coeff_map)
}

pub fn tfhe_to_bfv_with_functional_coefficient_map(
    tfhe_cts: &[LweCiphertext],
    params: &BridgeParams,
    key: &TfheToBfvFunctionalKey,
    coeff_map: &[Vec<u64>],
) -> Result<Ciphertext, BridgeError> {
    validate_functional_switch_inputs(tfhe_cts, params, key, coeff_map)?;

    let degree = params.bfv_params.degree();
    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    let plaintext_modulus = params.bfv_params.plain_modulus();
    let mut switched = GlweCiphertext::zeros(
        GlweDimension(1),
        PolynomialSize(degree),
        params.tfhe_params.ciphertext_modulus_log(),
    );

    for row in 0..degree {
        let mut acc = 0u64;
        for (col, ct) in tfhe_cts.iter().enumerate() {
            torus_add_scaled_mod_p(
                &mut acc,
                ct.body(),
                coeff_map[row][col],
                plaintext_modulus,
                log_q,
            );
        }
        switched.body_mut().coeffs_mut()[row] = acc;
    }

    let backend = SchoolbookMul;
    for lwe_index in 0..key.trgsw.input_lwe_dimension().0 {
        let mut a_poly = NativePoly::zeros(degree, log_q);
        for row in 0..degree {
            let mut acc = 0u64;
            for (col, ct) in tfhe_cts.iter().enumerate() {
                torus_add_scaled_mod_p(
                    &mut acc,
                    ct.mask()[lwe_index],
                    coeff_map[row][col],
                    plaintext_modulus,
                    log_q,
                );
            }
            a_poly.coeffs_mut()[row] = acc;
        }

        if a_poly.coeffs().iter().all(|&x| x == 0) {
            continue;
        }

        let a_glwe = GlweCiphertext::from_polys(
            vec![NativePoly::zeros(degree, log_q), a_poly],
            params.tfhe_params.ciphertext_modulus_log(),
        );
        let product = build_external_product(key.trgsw.get(lwe_index), &a_glwe, &backend);
        switched.sub_assign(&product);
    }

    glwe_torus_to_bfv_ciphertext(&switched, params)
}

fn validate_functional_switch_inputs(
    tfhe_cts: &[LweCiphertext],
    params: &BridgeParams,
    key: &TfheToBfvFunctionalKey,
    coeff_map: &[Vec<u64>],
) -> Result<(), BridgeError> {
    ensure_functional_switch_ring_shape(params)?;
    if tfhe_cts.is_empty() {
        return Err(BridgeError::IncompatibleParameters(
            "SILENT TFHE->BFV switching requires at least one TLWE ciphertext".into(),
        ));
    }
    if coeff_map.len() != params.bfv_params.degree() {
        return Err(BridgeError::IncompatibleParameters(format!(
            "SILENT coefficient map has {} rows, expected BFV degree {}",
            coeff_map.len(),
            params.bfv_params.degree()
        )));
    }
    for (row, coeffs) in coeff_map.iter().enumerate() {
        if coeffs.len() != tfhe_cts.len() {
            return Err(BridgeError::IncompatibleParameters(format!(
                "SILENT coefficient map row {row} has {} columns, expected {}",
                coeffs.len(),
                tfhe_cts.len()
            )));
        }
    }

    let expected_dimension = key.trgsw.input_lwe_dimension().0;
    let expected_log_q = params.tfhe_params.ciphertext_modulus_log().0;
    for (index, ct) in tfhe_cts.iter().enumerate() {
        if ct.dimension().0 != expected_dimension {
            return Err(BridgeError::IncompatibleParameters(format!(
                "SILENT TLWE {index} has dimension {}, expected {}",
                ct.dimension().0,
                expected_dimension
            )));
        }
        if ct.log_modulus() != expected_log_q {
            return Err(BridgeError::IncompatibleParameters(format!(
                "SILENT TLWE {index} has log modulus {}, expected {}",
                ct.log_modulus(),
                expected_log_q
            )));
        }
    }

    if key.trgsw.glwe_dimension().0 != 1 {
        return Err(BridgeError::IncompatibleParameters(format!(
            "SILENT TFHE->BFV switching currently targets one BFV RLWE key polynomial, got GLWE dimension {}",
            key.trgsw.glwe_dimension().0
        )));
    }
    if key.trgsw.polynomial_size().0 != params.bfv_params.degree() {
        return Err(BridgeError::IncompatibleParameters(format!(
            "SILENT TRGSW key polynomial size {} does not match BFV degree {}",
            key.trgsw.polynomial_size().0,
            params.bfv_params.degree()
        )));
    }
    if key.trgsw.log_modulus() != expected_log_q {
        return Err(BridgeError::IncompatibleParameters(format!(
            "SILENT TRGSW key log modulus {} does not match TFHE log modulus {}",
            key.trgsw.log_modulus(),
            expected_log_q
        )));
    }
    let precision_bits = u16::from(key.trgsw.base_log().0) * u16::from(key.trgsw.level().0);
    if precision_bits < u16::from(expected_log_q) {
        return Err(BridgeError::IncompatibleParameters(format!(
            "SILENT functional switching requires full q-bit TRGSW decomposition for the toy exact backend; got base_log*level={} for q=2^{}",
            precision_bits, expected_log_q
        )));
    }

    Ok(())
}

fn ensure_functional_switch_ring_shape(params: &BridgeParams) -> Result<(), BridgeError> {
    let degree = params.bfv_params.degree();
    let polynomial_size = params.tfhe_params.polynomial_size().0;
    if degree != polynomial_size {
        return Err(BridgeError::IncompatibleParameters(format!(
            "SILENT TFHE->BFV switching requires BFV degree to equal target TRLWE polynomial size; got BFV degree {degree}, TFHE polynomial size {polynomial_size}"
        )));
    }
    if params.bfv_params.plain_modulus() <= 1 {
        return Err(BridgeError::IncompatibleParameters(format!(
            "SILENT TFHE->BFV switching requires BFV plaintext modulus p > 1, got {}",
            params.bfv_params.plain_modulus()
        )));
    }
    Ok(())
}

fn bfv_secret_key_as_negated_glwe(
    bfv_sk: &SecretKey,
    params: &BridgeParams,
) -> Result<GlweSecretKey, BridgeError> {
    let lwe = bfv_secret_key_as_lwe(bfv_sk, params);
    if lwe.dimension().0 != params.bfv_params.degree() {
        return Err(BridgeError::IncompatibleParameters(format!(
            "BFV secret converted to LWE dimension {}, expected degree {}",
            lwe.dimension().0,
            params.bfv_params.degree()
        )));
    }
    let poly = NativePoly::from_u64(lwe.data(), params.tfhe_params.ciphertext_modulus_log().0);
    Ok(GlweSecretKey::from_polys(
        vec![poly],
        params.tfhe_params.ciphertext_modulus_log(),
    ))
}

fn glwe_torus_to_bfv_ciphertext(
    glwe: &GlweCiphertext,
    params: &BridgeParams,
) -> Result<Ciphertext, BridgeError> {
    if glwe.glwe_dimension().0 != 1 {
        return Err(BridgeError::IncompatibleParameters(format!(
            "can convert only one-dimensional GLWE to BFV ciphertext, got {}",
            glwe.glwe_dimension().0
        )));
    }
    let c0 = torus_poly_to_bfv_poly(glwe.body(), params)?;
    let c1 = torus_poly_to_bfv_poly(&glwe.mask()[0], params)?;
    Ok(Ciphertext::new(
        vec![c0, c1],
        params.bfv_params.runtime_params_arc().as_ref().clone(),
        false,
    ))
}

fn torus_poly_to_bfv_poly(poly: &NativePoly, params: &BridgeParams) -> Result<Poly, BridgeError> {
    let degree = params.bfv_params.degree();
    if poly.degree() != degree {
        return Err(BridgeError::IncompatibleParameters(format!(
            "torus polynomial degree {} does not match BFV degree {degree}",
            poly.degree()
        )));
    }

    let ring = params.bfv_params.ring();
    let mut q_bfv = BigUint::one();
    for modulus in ring.rns().moduli() {
        q_bfv *= BigUint::from(modulus.value());
    }
    let q_t = BigUint::one() << params.tfhe_params.ciphertext_modulus_log().0;
    let half_q_t = &q_t >> 1usize;
    let mut out = Poly::new(degree, ring.rns().len());

    for coeff_index in 0..degree {
        let coeff = reduce_torus(poly.coeffs()[coeff_index], poly.log_modulus());
        let lifted = (BigUint::from(coeff) * &q_bfv + &half_q_t) / &q_t;
        for (limb_index, modulus) in ring.rns().moduli().iter().enumerate() {
            let residue = (&lifted % BigUint::from(modulus.value()))
                .to_u64()
                .ok_or_else(|| {
                    BridgeError::IncompatibleParameters(
                        "SILENT torus->BFV residue does not fit in u64".into(),
                    )
                })?;
            out.limb_mut(limb_index)[coeff_index] = residue;
        }
    }

    Ok(out)
}

fn torus_native_encode(value: u64, plain_modulus: u64, log_q: u8) -> Result<u64, BridgeError> {
    let q = BigUint::one() << log_q;
    let p = BigUint::from(plain_modulus);
    let encoded = (BigUint::from(value % plain_modulus) * &q + (&p >> 1usize)) / &p;
    if log_q == 64 {
        encoded.to_u64().ok_or_else(|| {
            BridgeError::IncompatibleParameters("torus native encoding exceeds u64".into())
        })
    } else {
        let modulus = BigUint::one() << log_q;
        (&encoded % modulus).to_u64().ok_or_else(|| {
            BridgeError::IncompatibleParameters("torus native encoding exceeds u64".into())
        })
    }
}

fn torus_add_scaled_mod_p(dst: &mut u64, value: u64, scalar_mod_p: u64, p: u64, log_q: u8) {
    let scalar = scalar_mod_p % p;
    if scalar == 0 {
        return;
    }
    let (positive, magnitude) = if scalar <= p / 2 {
        (true, scalar)
    } else {
        (false, p - scalar)
    };
    let term = torus_mul_small(value, magnitude, log_q);
    *dst = if positive {
        dst.wrapping_add(term)
    } else {
        dst.wrapping_sub(term)
    };
    *dst = reduce_torus(*dst, log_q);
}

fn torus_mul_small(value: u64, scalar: u64, log_q: u8) -> u64 {
    let product = value.wrapping_mul(scalar);
    reduce_torus(product, log_q)
}

fn reduce_torus(value: u64, log_q: u8) -> u64 {
    if log_q == 64 {
        value
    } else if log_q == 0 {
        0
    } else {
        value & ((1u64 << log_q) - 1)
    }
}
