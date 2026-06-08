use crate::bridge::bfv_to_tfhe::extract::{ExtractedBfvLwe, extract_lwe_from_bfv};
use crate::bridge::bfv_to_tfhe::modswitch::{
    modswitch_bfv_lwe_to_tfhe, modswitch_bfv_lwe_to_torus_native_tfhe,
};
use crate::bridge::keys::bfv_secret_key_as_lwe;
use crate::core::encoder::HeEncoder;
use crate::core::keys::KeyGenerator;
use crate::schemes::bfv::crypto::{BfvDecryptor, BfvEncryptor};
use crate::schemes::bfv::encoding::BatchEncoder;
use crate::schemes::bfv::keys::BfvKeyGenerator;
use crate::schemes::bfv::params::BfvParameters;
use crate::schemes::tfhe::bootstrap::{build_lwe_bootstrap_key, generate_pbs_keys};
use crate::schemes::tfhe::crypto::{TfheDecryptor, TfheEncryptor};
use crate::schemes::tfhe::encoding::{TfheEncoder, TfhePlaintext};
use crate::schemes::tfhe::keys::TfheKeyGenerator;
use crate::schemes::tfhe::keyswitch::build_lwe_keyswitch_key;
use crate::schemes::tfhe::params::TfheParameters;
use rand_core::SeedableRng;
use silent_params::{
    CarryModulus, DecompositionBaseLog, DecompositionLevelCount, GlweDimension, Log2PFail,
    LweDimension, MessageModulus, NoiseDistribution, PolynomialSize, presets,
};
use silent_rlwe::Plaintext;
use silent_utils::rng::SecureRng;
use std::sync::Arc;

use super::bfv_to_tfhe::{
    bfv_slots_to_tfhe_torus_native, bfv_to_tfhe, bfv_to_tfhe_torus_native,
    bfv_to_tfhe_trusted_oracle, slot_to_coeff::slot_to_coeff,
};
use super::tfhe_ops::{
    eval_relu,
    minmax::{eval_max, eval_min},
    mux::eval_mux,
};
use super::tfhe_to_bfv::{
    bfv_phase_to_bfv_messages, ensure_full_tfhe_to_bfv_preconditions, tfhe_to_bfv,
    tfhe_to_bfv_phase_coefficients, tfhe_to_bfv_torus_native_lwe1_exact,
    tfhe_to_bfv_trusted_oracle,
};
use super::{BridgeKeyGenerator, BridgeLayout, BridgeParams};

fn bridge_params() -> BridgeParams {
    bridge_params_with_plain_modulus(12289)
}

fn bridge_params_b2t_native_scale() -> BridgeParams {
    bridge_params_with_plain_modulus(32)
}

fn bridge_params_t2b_native_phase() -> BridgeParams {
    bridge_params_custom(1u64 << 32, 128, 64, 64, vec![50, 50], Vec::new())
}

fn bridge_params_fmod_center() -> BridgeParams {
    let degree = 128;
    let plain_modulus = silent_math::numth::next_ntt_prime((1u64 << 32) + 1, (2 * degree) as u64)
        .expect("test plaintext modulus");
    bridge_params_custom(plain_modulus, degree, 64, 64, vec![50, 50], vec![50])
}

fn bridge_params_t2b_lwe1_binary_exact() -> BridgeParams {
    let degree = 64;
    let lwe_dimension = 1;
    let message_bits = 1;
    let full_message_modulus = 1u64 << message_bits;
    let q_t = 1u64 << 32;
    let delta = q_t / (2 * full_message_modulus);
    let plain_modulus = 49 * delta + 1;
    assert!(silent_math::numth::is_prime(plain_modulus));

    bridge_params_custom_with_tfhe_plaintext(
        plain_modulus,
        degree,
        lwe_dimension,
        64,
        vec![50, 50],
        vec![50],
        2,
        1,
        message_bits,
        Some(NoiseDistribution::Gaussian { stddev: 0.0 }),
        None,
        None,
        None,
        None,
        None,
    )
}

fn bridge_params_t2b_lwe1_torus_native_exact() -> BridgeParams {
    bridge_params_custom_with_tfhe_plaintext(
        12289,
        64,
        1,
        64,
        vec![50, 50],
        vec![50],
        4,
        4,
        4,
        Some(NoiseDistribution::Gaussian { stddev: 0.0 }),
        None,
        None,
        None,
        None,
        None,
    )
}

fn bridge_params_b2t_torus_native_exact_large_plaintext() -> BridgeParams {
    bridge_params_custom_with_tfhe_plaintext(
        12289,
        128,
        64,
        64,
        vec![50, 50],
        Vec::new(),
        4,
        4,
        4,
        Some(NoiseDistribution::Gaussian { stddev: 0.0 }),
        Some(NoiseDistribution::Gaussian { stddev: 0.0 }),
        Some(2),
        Some(16),
        Some(2),
        Some(16),
    )
}

fn bridge_params_with_plain_modulus(plain_modulus: u64) -> BridgeParams {
    bridge_params_custom(plain_modulus, 512, 512, 512, vec![27], Vec::new())
}

fn tfhe_ops_test_params() -> TfheParameters {
    let mut tfhe_raw = presets::toy::toy_tfhe_n512();
    tfhe_raw.lwe_dimension = LweDimension(8);
    tfhe_raw.glwe_dimension = GlweDimension(1);
    tfhe_raw.polynomial_size = PolynomialSize(64);
    tfhe_raw.message_modulus = MessageModulus(4);
    tfhe_raw.carry_modulus = CarryModulus(4);
    tfhe_raw.pbs_base_log = DecompositionBaseLog(8);
    tfhe_raw.pbs_level = DecompositionLevelCount(4);
    tfhe_raw.ks_base_log = DecompositionBaseLog(2);
    tfhe_raw.ks_level = DecompositionLevelCount(7);
    tfhe_raw.lwe_noise = NoiseDistribution::Gaussian { stddev: 1e-12 };
    tfhe_raw.glwe_noise = NoiseDistribution::Gaussian { stddev: 1e-15 };
    tfhe_raw.log2_p_fail = Log2PFail(-40.0);
    TfheParameters::new(tfhe_raw).unwrap()
}

fn bridge_params_custom(
    plain_modulus: u64,
    degree: usize,
    tfhe_lwe_dimension: usize,
    tfhe_polynomial_size: usize,
    bfv_ciphertext_modulus_bits: Vec<u16>,
    bfv_special_modulus_bits: Vec<u16>,
) -> BridgeParams {
    bridge_params_custom_with_tfhe_plaintext(
        plain_modulus,
        degree,
        tfhe_lwe_dimension,
        tfhe_polynomial_size,
        bfv_ciphertext_modulus_bits,
        bfv_special_modulus_bits,
        4,
        4,
        4,
        None,
        None,
        None,
        None,
        None,
        None,
    )
}

fn bridge_params_custom_with_tfhe_plaintext(
    plain_modulus: u64,
    degree: usize,
    tfhe_lwe_dimension: usize,
    tfhe_polynomial_size: usize,
    bfv_ciphertext_modulus_bits: Vec<u16>,
    bfv_special_modulus_bits: Vec<u16>,
    tfhe_message_modulus: u64,
    tfhe_carry_modulus: u64,
    bridge_message_bits: usize,
    lwe_noise: Option<NoiseDistribution>,
    glwe_noise: Option<NoiseDistribution>,
    pbs_base_log: Option<u8>,
    pbs_level: Option<u8>,
    ks_base_log: Option<u8>,
    ks_level: Option<u8>,
) -> BridgeParams {
    let mut bfv_raw = presets::toy::toy_bfv_1024();
    bfv_raw.rlwe.ring.ring_dim = silent_params::newtypes::RingDim(degree);
    bfv_raw.rlwe.ring.log_n = silent_params::newtypes::LogN(degree.trailing_zeros() as u8);
    bfv_raw.rlwe.ciphertext_modulus_bits = bfv_ciphertext_modulus_bits
        .into_iter()
        .map(silent_params::ModulusBits)
        .collect();
    bfv_raw.rlwe.special_modulus_bits = bfv_special_modulus_bits
        .into_iter()
        .map(silent_params::ModulusBits)
        .collect();
    bfv_raw.plaintext_modulus = silent_params::newtypes::PlaintextModulus(plain_modulus);

    // Create RnsTool with scaling enabled to avoid OpenFheDisabled error
    let mut config = bfv_raw
        .rlwe
        .to_rns_tool_config(bfv_raw.plaintext_modulus)
        .unwrap();
    config = config.enable_openfhe_scale();

    // If we have special moduli, enable HPS/Switch tables
    if !bfv_raw.rlwe.special_modulus_bits.is_empty() {
        config = config.enable_openfhe_switch(false);
        config = config.enable_openfhe_approx_scale();
        // Use special base as R for HPS
        let generated = bfv_raw.rlwe.gen_moduli().unwrap();
        let base_p = silent_math::rns::RnsBase::from_values(generated.special_moduli).unwrap();
        config = config.with_base_r(base_p);
        config = config.enable_openfhe_hps();
    }

    let rlwe_params =
        Arc::new(silent_rlwe::EncryptionParams::from_rns_config(degree, config).unwrap());
    let bfv = BfvParameters::from_runtime_parts(bfv_raw, rlwe_params).unwrap();

    let mut tfhe_raw = presets::toy::toy_tfhe_n512();
    tfhe_raw.lwe_dimension = silent_params::newtypes::LweDimension(tfhe_lwe_dimension);
    tfhe_raw.polynomial_size = silent_params::newtypes::PolynomialSize(tfhe_polynomial_size);
    tfhe_raw.ciphertext_modulus_log = silent_params::newtypes::CiphertextModulusLog(32);
    tfhe_raw.message_modulus = silent_params::newtypes::MessageModulus(tfhe_message_modulus);
    tfhe_raw.carry_modulus = silent_params::newtypes::CarryModulus(tfhe_carry_modulus);
    if let Some(noise) = lwe_noise {
        tfhe_raw.lwe_noise = noise;
    }
    if let Some(noise) = glwe_noise {
        tfhe_raw.glwe_noise = noise;
    }
    if let Some(base_log) = pbs_base_log {
        tfhe_raw.pbs_base_log = silent_params::newtypes::DecompositionBaseLog(base_log);
    }
    if let Some(level) = pbs_level {
        tfhe_raw.pbs_level = silent_params::newtypes::DecompositionLevelCount(level);
    }
    if let Some(base_log) = ks_base_log {
        tfhe_raw.ks_base_log = silent_params::newtypes::DecompositionBaseLog(base_log);
    }
    if let Some(level) = ks_level {
        tfhe_raw.ks_level = silent_params::newtypes::DecompositionLevelCount(level);
    }

    let tfhe = TfheParameters::new(tfhe_raw).unwrap();

    BridgeParams::new(bfv, tfhe, bridge_message_bits, 8, BridgeLayout::Coefficient).unwrap()
}

fn decrypt_bfv_coefficients(
    ct: &silent_rlwe::Ciphertext,
    params: &BridgeParams,
    sk: &silent_rlwe::SecretKey,
) -> Vec<u64> {
    let decryptor = BfvDecryptor::new(params.bfv_params.clone(), sk.clone());
    decryptor.decrypt(ct).value.limb(0).to_vec()
}

fn decrypt_bfv_coefficients_exact(
    ct: &silent_rlwe::Ciphertext,
    params: &BridgeParams,
    sk: &silent_rlwe::SecretKey,
) -> Vec<u64> {
    let ring = params.bfv_params.ring();
    let degree = ring.degree();
    let ct_is_ntt = ct.is_ntt;

    let mut m_poly = ct.data[0].clone();
    if !ct_is_ntt {
        m_poly.ntt_forward(ring);
    }

    let ct_moduli_count = m_poly.num_moduli();
    let sk_prepared = if sk.value.num_moduli() > ct_moduli_count {
        let mut s_reduced = silent_ring::Poly::new(degree, ct_moduli_count);
        s_reduced
            .data_mut()
            .copy_from_slice(&sk.value.data()[..degree * ct_moduli_count]);
        s_reduced
    } else {
        sk.value.clone()
    };

    let mut s_pow = sk_prepared.clone();
    for i in 1..ct.data.len() {
        let mut term = ct.data[i].clone();
        if !ct_is_ntt {
            term.ntt_forward(ring);
        }
        term.mul_assign(&s_pow, ring);
        m_poly.add_assign(&term, ring);

        if i < ct.data.len() - 1 {
            s_pow.mul_assign(&sk_prepared, ring);
        }
    }

    m_poly.ntt_inverse(ring);
    params
        .bfv_params
        .rns_tool()
        .scale_and_round(&m_poly.data(), degree)
        .expect("exact BFV scale-and-round should succeed")
}

fn torus_distance_u64(a: u64, b: u64, log_q: u8) -> u64 {
    if log_q == 64 {
        let forward = a.wrapping_sub(b);
        let backward = b.wrapping_sub(a);
        forward.min(backward)
    } else {
        let mask = (1u64 << log_q) - 1;
        let forward = a.wrapping_sub(b) & mask;
        let backward = b.wrapping_sub(a) & mask;
        forward.min(backward)
    }
}

fn torus_native_encode(message: u64, plain_modulus: u64, log_q: u8) -> u64 {
    let q = 1u128 << log_q;
    let numerator = (message as u128) * q + (plain_modulus as u128 / 2);
    let encoded = numerator / plain_modulus as u128;
    if log_q == 64 {
        encoded as u64
    } else {
        (encoded as u64) & ((1u64 << log_q) - 1)
    }
}

fn trivial_bfv_ciphertext_from_coeffs(
    params: &BridgeParams,
    coeffs: &[u64],
) -> silent_rlwe::Ciphertext {
    let degree = params.bfv_params.degree();
    let ring = params.bfv_params.ring();
    let rns = ring.rns();
    let q_bfv = rns.base_prod_u128().unwrap();
    let t = params.bfv_params.plain_modulus() as u128;
    let delta_bfv = q_bfv / t;

    let mut c0 = silent_ring::Poly::new(degree, rns.moduli().len());
    for (j, &value) in coeffs.iter().enumerate() {
        let target_val = delta_bfv * (value as u128);
        for (i, modulus) in rns.moduli().iter().enumerate() {
            c0.limb_mut(i)[j] = (target_val % modulus.value() as u128) as u64;
        }
    }
    let c1 = silent_ring::Poly::new(degree, rns.moduli().len());
    silent_rlwe::Ciphertext::new(
        vec![c0, c1],
        params.bfv_params.runtime_params_arc().as_ref().clone(),
        false,
    )
}

#[test]
fn bfv_to_tfhe_roundtrip_via_trusted_repack() {
    let params = bridge_params();
    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([1u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();
    let mut bfv_enc = BfvEncryptor::new(params.bfv_params.clone(), SecureRng::from_seed([2u8; 32]));

    let mut tfhe_keygen =
        TfheKeyGenerator::with_rng(params.tfhe_params.clone(), SecureRng::from_seed([3u8; 32]));
    let tfhe_sk = tfhe_keygen.generate_lwe_secret_key();
    let tfhe_dec = TfheDecryptor::new(params.tfhe_params.clone(), tfhe_sk.clone());

    let mut bridge_keygen =
        BridgeKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([4u8; 32]));
    let _bridge_keys = bridge_keygen.generate(&bfv_sk, &tfhe_sk);

    let messages = [0u64, 1, 2, 3, 4, 5, 6, 7];
    let mut poly = silent_ring::Poly::new(params.bfv_params.degree(), 1);
    for (index, value) in messages.iter().enumerate() {
        poly.limb_mut(0)[index] = *value;
    }
    let ct_bfv = bfv_enc.encrypt_symmetric(&bfv_sk, &Plaintext { value: poly });

    let tfhe_cts = bfv_to_tfhe_trusted_oracle(&ct_bfv, &params, &bfv_sk, &tfhe_sk).unwrap();
    let encoder = TfheEncoder::new(params.tfhe_params.clone());
    let recovered: Vec<u64> = tfhe_cts
        .iter()
        .map(|ct| tfhe_dec.decrypt_full(ct, &encoder))
        .collect();
    assert_eq!(recovered, messages);

    let repacked = tfhe_to_bfv_trusted_oracle(
        &tfhe_cts,
        &params,
        &tfhe_dec,
        &bfv_sk,
        SecureRng::from_seed([5u8; 32]),
    )
    .unwrap();
    let recovered_bfv = decrypt_bfv_coefficients(&repacked, &params, &bfv_sk);
    assert_eq!(&recovered_bfv[..params.num_slots], &messages);
}

#[test]
fn bridge_relu_roundtrip_via_trusted_repack() {
    let params = bridge_params();
    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([11u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();
    let mut bfv_enc =
        BfvEncryptor::new(params.bfv_params.clone(), SecureRng::from_seed([12u8; 32]));

    let mut tfhe_keygen =
        TfheKeyGenerator::with_rng(params.tfhe_params.clone(), SecureRng::from_seed([13u8; 32]));
    let tfhe_sk = tfhe_keygen.generate_lwe_secret_key();
    let glwe_sk = tfhe_keygen.generate_glwe_secret_key();
    let tfhe_dec = TfheDecryptor::new(params.tfhe_params.clone(), tfhe_sk.clone());

    let big_sk = tfhe_keygen.lwe_secret_from_glwe(&glwe_sk);
    let bsk = build_lwe_bootstrap_key(
        &mut SecureRng::from_seed([14u8; 32]),
        &tfhe_sk,
        &glwe_sk,
        params.tfhe_params.pbs_base_log(),
        params.tfhe_params.pbs_level(),
        params.tfhe_params.glwe_noise(),
        params.tfhe_params.ciphertext_modulus_log(),
    );
    let ksk = build_lwe_keyswitch_key(
        &mut SecureRng::from_seed([15u8; 32]),
        &big_sk,
        &tfhe_sk,
        params.tfhe_params.ks_base_log(),
        params.tfhe_params.ks_level(),
        params.tfhe_params.lwe_noise(),
        params.tfhe_params.ciphertext_modulus_log(),
    );

    let mut bridge_keygen =
        BridgeKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([16u8; 32]));
    let _bridge_keys = bridge_keygen.generate(&bfv_sk, &tfhe_sk);

    // Signed values encoded in Z_16: [-3, -1, 0, 2, 5, -7, 3, 1]
    let messages = [13u64, 15, 0, 2, 5, 9, 3, 1];
    let expected = [0u64, 0, 0, 2, 5, 0, 3, 1];
    let mut poly = silent_ring::Poly::new(params.bfv_params.degree(), 1);
    for (index, value) in messages.iter().enumerate() {
        poly.limb_mut(0)[index] = *value;
    }
    let ct_bfv = bfv_enc.encrypt_symmetric(&bfv_sk, &Plaintext { value: poly });

    let tfhe_cts = bfv_to_tfhe_trusted_oracle(&ct_bfv, &params, &bfv_sk, &tfhe_sk).unwrap();
    let encoder = TfheEncoder::new(params.tfhe_params.clone());
    let relu_cts = tfhe_cts
        .iter()
        .map(|ct| eval_relu(&params.tfhe_params, &encoder, &bsk, &ksk, ct).unwrap())
        .collect::<Vec<_>>();

    let repacked = tfhe_to_bfv_trusted_oracle(
        &relu_cts,
        &params,
        &tfhe_dec,
        &bfv_sk,
        SecureRng::from_seed([17u8; 32]),
    )
    .unwrap();
    let recovered_bfv = decrypt_bfv_coefficients(&repacked, &params, &bfv_sk);
    assert_eq!(&recovered_bfv[..params.num_slots], &expected);
}

#[test]
fn bridge_mux_min_max_run_pbs_selector_gate() {
    let params = tfhe_ops_test_params();
    let encoder = TfheEncoder::new(params.clone());
    let mut pbs_rng = SecureRng::from_seed([61u8; 32]);
    let (sk, _glwe_sk, bsk, ksk) = generate_pbs_keys(params.clone(), &mut pbs_rng);
    let mut encryptor = TfheEncryptor::new(params.clone(), SecureRng::from_seed([62u8; 32]));
    let decryptor = TfheDecryptor::new(params.clone(), sk.clone());

    let selector_zero = encryptor.encrypt_symmetric(&sk, encoder.encode_full(0));
    let selector_one = encryptor.encrypt_symmetric(&sk, encoder.encode_full(1));
    let x = encryptor.encrypt_symmetric(&sk, encoder.encode_full(6));
    let y = encryptor.encrypt_symmetric(&sk, encoder.encode_full(2));

    let choose_x = eval_mux(&params, &encoder, &bsk, &ksk, &selector_one, &x, &y).unwrap();
    assert_eq!(decryptor.decrypt_full(&choose_x, &encoder), 6);

    let choose_y = eval_mux(&params, &encoder, &bsk, &ksk, &selector_zero, &x, &y).unwrap();
    assert_eq!(decryptor.decrypt_full(&choose_y, &encoder), 2);

    let min = eval_min(&params, &encoder, &bsk, &ksk, &x, &y).unwrap();
    assert_eq!(decryptor.decrypt_full(&min, &encoder), 2);

    let max = eval_max(&params, &encoder, &bsk, &ksk, &x, &y).unwrap();
    assert_eq!(decryptor.decrypt_full(&max, &encoder), 6);
}

#[test]
fn public_tfhe_to_bfv_rejects_trusted_oracle_path() {
    let params = bridge_params();
    let mut tfhe_keygen =
        TfheKeyGenerator::with_rng(params.tfhe_params.clone(), SecureRng::from_seed([21u8; 32]));
    let tfhe_sk = tfhe_keygen.generate_lwe_secret_key();

    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([22u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();

    let mut bridge_keygen =
        BridgeKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([23u8; 32]));
    let bridge_keys = bridge_keygen.generate(&bfv_sk, &tfhe_sk);

    // Verify public tfhe_to_bfv (homomorphic) returns error for now
    let err = tfhe_to_bfv(&[], &params, &bridge_keys).unwrap_err();
    assert!(matches!(err, super::BridgeError::IncompatibleParameters(_)));
}

#[test]
fn public_bfv_to_tfhe_rejects_non_native_plain_modulus() {
    let params = bridge_params();
    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([31u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();
    let mut tfhe_keygen =
        TfheKeyGenerator::with_rng(params.tfhe_params.clone(), SecureRng::from_seed([32u8; 32]));
    let tfhe_sk = tfhe_keygen.generate_lwe_secret_key();
    let mut bridge_keygen =
        BridgeKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([33u8; 32]));
    let bridge_keys = bridge_keygen.generate(&bfv_sk, &tfhe_sk);

    let ct_bfv = trivial_bfv_ciphertext_from_coeffs(&params, &[0]);
    let err = bfv_to_tfhe(&ct_bfv, &bridge_keys, &params).unwrap_err();
    assert!(matches!(err, super::BridgeError::IncompatibleParameters(_)));
}

#[test]
fn test_modswitch_scaling_perfect_input() {
    let params = bridge_params_b2t_native_scale();
    let ring = params.bfv_params.ring();
    let rns = ring.rns();
    let q_bfv = rns.base_prod_u128().unwrap();
    let t = params.bfv_params.plain_modulus() as u128;
    let encoder = TfheEncoder::new(params.tfhe_params.clone());

    // Create a perfect BFV constant ciphertext: c0 = round(Q/t) * message, c1 = 0
    let message = 3u64;
    let delta_bfv = q_bfv / t;
    let target_c0 = delta_bfv * message as u128;

    let size_q = rns.moduli().len();
    let dimension = params.bfv_params.degree();
    let count = dimension + 1;
    let mut residues_q = vec![0u64; count * size_q];

    for (limb_idx, qi) in rns.moduli().iter().enumerate() {
        // Only set the body (index = dimension) for the constant message
        residues_q[limb_idx * count + dimension] = (target_c0 % qi.value() as u128) as u64;
    }

    let lwe_q = ExtractedBfvLwe::new(dimension, residues_q);

    let lwe_t = modswitch_bfv_lwe_to_tfhe(&lwe_q, &params).unwrap();
    let phase = lwe_t.body();
    let expected_phase = encoder.encode_full(message).value;

    let dist = torus_distance_u64(
        phase,
        expected_phase,
        params.tfhe_params.ciphertext_modulus_log().0,
    );
    assert!(
        dist < 100,
        "Modswitch scaling is off even for perfect input! dist={}",
        dist
    );
}

#[test]
fn modswitch_general_plain_modulus_perfect_input() {
    let params = bridge_params_custom(12289, 128, 64, 64, vec![50, 50], Vec::new());
    let ring = params.bfv_params.ring();
    let rns = ring.rns();
    let q_bfv = rns.base_prod_u128().unwrap();
    let t = params.bfv_params.plain_modulus() as u128;
    let encoder = TfheEncoder::new(params.tfhe_params.clone());

    let message = 7u64;
    let delta_bfv = q_bfv / t;
    let target_c0 = delta_bfv * message as u128;

    let size_q = rns.moduli().len();
    let dimension = params.bfv_params.degree();
    let count = dimension + 1;
    let mut residues_q = vec![0u64; count * size_q];

    for (limb_idx, qi) in rns.moduli().iter().enumerate() {
        residues_q[limb_idx * count + dimension] = (target_c0 % qi.value() as u128) as u64;
    }

    let lwe_q = ExtractedBfvLwe::new(dimension, residues_q);
    let lwe_t = modswitch_bfv_lwe_to_tfhe(&lwe_q, &params).unwrap();
    let phase = lwe_t.body();
    let expected_phase = encoder.encode_full(message).value;

    let dist = torus_distance_u64(
        phase,
        expected_phase,
        params.tfhe_params.ciphertext_modulus_log().0,
    );
    assert!(
        dist < 100,
        "general BFV->TFHE perfect-input scaling is off: dist={dist}"
    );
}

#[test]
fn modswitch_torus_native_general_plain_modulus_perfect_input() {
    let params = bridge_params_custom(12289, 128, 64, 64, vec![50, 50], Vec::new());
    let ring = params.bfv_params.ring();
    let rns = ring.rns();
    let q_bfv = rns.base_prod_u128().unwrap();
    let t = params.bfv_params.plain_modulus() as u128;
    let log_q = params.tfhe_params.ciphertext_modulus_log().0;

    let message = 4096u64;
    let delta_bfv = q_bfv / t;
    let target_c0 = delta_bfv * message as u128;

    let size_q = rns.moduli().len();
    let dimension = params.bfv_params.degree();
    let count = dimension + 1;
    let mut residues_q = vec![0u64; count * size_q];

    for (limb_idx, qi) in rns.moduli().iter().enumerate() {
        residues_q[limb_idx * count + dimension] = (target_c0 % qi.value() as u128) as u64;
    }

    let lwe_q = ExtractedBfvLwe::new(dimension, residues_q);
    let lwe_t = modswitch_bfv_lwe_to_torus_native_tfhe(&lwe_q, &params).unwrap();
    let phase = lwe_t.body();
    let expected_phase = torus_native_encode(message, params.bfv_params.plain_modulus(), log_q);

    let dist = torus_distance_u64(phase, expected_phase, log_q);
    assert!(
        dist < 100,
        "torus-native BFV->TFHE perfect-input scaling is off: dist={dist}, phase={phase}, expected={expected_phase}"
    );
}

#[test]
fn bfv_to_tfhe_pre_keyswitch_phase_matches_full_encoding() {
    let params = bridge_params_b2t_native_scale();
    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([41u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();
    let mut bfv_enc =
        BfvEncryptor::new(params.bfv_params.clone(), SecureRng::from_seed([42u8; 32]));
    let encoder = TfheEncoder::new(params.tfhe_params.clone());
    let lwe_sk = bfv_secret_key_as_lwe(&bfv_sk, &params);
    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    let delta = encoder.delta();

    let tfhe_decryptor = TfheDecryptor::new(params.tfhe_params.clone(), lwe_sk);
    let messages = [0u64, 1, 2, 3, 4, 5, 6, 7];
    let mut poly = silent_ring::Poly::new(params.bfv_params.degree(), 1);
    for (index, value) in messages.iter().enumerate() {
        poly.limb_mut(0)[index] = *value;
    }
    let ct_bfv = bfv_enc.encrypt_symmetric(&bfv_sk, &Plaintext { value: poly });

    let extracted = extract_lwe_from_bfv(&ct_bfv, &params).unwrap();
    let mut mismatches = Vec::new();
    for (i, lwe_q) in extracted.iter().enumerate().take(messages.len()) {
        let lwe_t = modswitch_bfv_lwe_to_tfhe(lwe_q, &params).unwrap();
        let phase = tfhe_decryptor.decrypt_raw(&lwe_t);

        let decoded = encoder.decode_full(TfhePlaintext::new(phase, log_q));
        let expected_raw = encoder.encode_full(messages[i]).value;
        let distance = torus_distance_u64(phase, expected_raw, log_q);

        if decoded != messages[i] || distance >= (delta / 2) {
            mismatches.push((i, messages[i], decoded, phase, expected_raw, distance));
        }
    }

    assert!(
        mismatches.is_empty(),
        "BFV->TFHE pre-keyswitch phase mismatch(s): {:?}",
        mismatches
    );
}

#[test]
fn bfv_to_tfhe_trivial_body_path_matches_full_encoding() {
    let params = bridge_params_b2t_native_scale();
    let encoder = TfheEncoder::new(params.tfhe_params.clone());
    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    let delta = encoder.delta();
    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([43u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();

    let messages = [0u64, 1, 2, 3, 4, 5, 6, 7];
    let ct_bfv = trivial_bfv_ciphertext_from_coeffs(&params, &messages);
    let recovered = decrypt_bfv_coefficients(&ct_bfv, &params, &bfv_sk);
    assert_eq!(&recovered[..messages.len()], &messages);
    let extracted = extract_lwe_from_bfv(&ct_bfv, &params).unwrap();
    let zero_sk = silent_rlwe::LweSecretKey::from_data(
        vec![0u64; params.bfv_params.degree()],
        params.tfhe_params.ciphertext_modulus_log(),
    );

    let tfhe_dec = TfheDecryptor::new(params.tfhe_params.clone(), zero_sk);
    let mut mismatches = Vec::new();
    for (index, lwe_q) in extracted.iter().enumerate().take(messages.len()) {
        let lwe_t = modswitch_bfv_lwe_to_tfhe(lwe_q, &params).unwrap();

        let phase = tfhe_dec.decrypt_raw(&lwe_t);
        let decoded = encoder.decode_full(TfhePlaintext::new(phase, log_q));
        let expected_raw = encoder.encode_full(messages[index]).value;
        let distance = torus_distance_u64(phase, expected_raw, log_q);
        if decoded != messages[index] || distance >= (delta / 2) {
            mismatches.push((
                index,
                messages[index],
                decoded,
                phase,
                expected_raw,
                distance,
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "BFV->TFHE trivial body-path mismatch(s): {:?}",
        mismatches
    );
}

#[test]
fn public_bfv_to_tfhe_roundtrip_native_scale_coefficient_layout() {
    let params = bridge_params_b2t_native_scale();
    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([51u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();
    let mut bfv_enc =
        BfvEncryptor::new(params.bfv_params.clone(), SecureRng::from_seed([52u8; 32]));

    let mut tfhe_keygen =
        TfheKeyGenerator::with_rng(params.tfhe_params.clone(), SecureRng::from_seed([53u8; 32]));
    let tfhe_sk = tfhe_keygen.generate_lwe_secret_key();
    let tfhe_dec = TfheDecryptor::new(params.tfhe_params.clone(), tfhe_sk.clone());

    let mut bridge_keygen =
        BridgeKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([54u8; 32]));
    let bridge_keys = bridge_keygen.generate(&bfv_sk, &tfhe_sk);

    let messages = [0u64, 1, 2, 3, 4, 5, 6, 7];
    let mut poly = silent_ring::Poly::new(params.bfv_params.degree(), 1);
    for (index, value) in messages.iter().enumerate() {
        poly.limb_mut(0)[index] = *value;
    }
    let ct_bfv = bfv_enc.encrypt_symmetric(&bfv_sk, &Plaintext { value: poly });
    let tfhe_cts = bfv_to_tfhe(&ct_bfv, &bridge_keys, &params).unwrap();
    let encoder = TfheEncoder::new(params.tfhe_params.clone());
    let recovered = tfhe_cts
        .iter()
        .map(|ct| tfhe_dec.decrypt_full(ct, &encoder))
        .collect::<Vec<_>>();
    assert_eq!(recovered, messages);
}

#[test]
fn public_bfv_to_tfhe_torus_native_roundtrip_general_plain_modulus() {
    let params = bridge_params_b2t_torus_native_exact_large_plaintext();
    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([55u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();
    let mut bfv_enc =
        BfvEncryptor::new(params.bfv_params.clone(), SecureRng::from_seed([56u8; 32]));

    let mut tfhe_keygen =
        TfheKeyGenerator::with_rng(params.tfhe_params.clone(), SecureRng::from_seed([57u8; 32]));
    let tfhe_sk = tfhe_keygen.generate_lwe_secret_key();
    let tfhe_dec = TfheDecryptor::new(params.tfhe_params.clone(), tfhe_sk.clone());

    let mut bridge_keygen =
        BridgeKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([58u8; 32]));
    let bridge_keys = bridge_keygen.generate(&bfv_sk, &tfhe_sk);

    let messages = [0u64, 1, 2, 3, 255, 2048, 4096, 8192];
    let mut poly = silent_ring::Poly::new(params.bfv_params.degree(), 1);
    for (index, value) in messages.iter().enumerate() {
        poly.limb_mut(0)[index] = *value;
    }
    let ct_bfv = bfv_enc.encrypt_symmetric(&bfv_sk, &Plaintext { value: poly });
    let tfhe_cts = bfv_to_tfhe_torus_native(&ct_bfv, &bridge_keys, &params).unwrap();

    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    let half_cell = ((1u128 << log_q) / (2 * params.bfv_params.plain_modulus() as u128)) as u64;
    let mut mismatches = Vec::new();
    for (index, ct) in tfhe_cts.iter().enumerate() {
        let phase = tfhe_dec.decrypt_raw(ct);
        let expected =
            torus_native_encode(messages[index], params.bfv_params.plain_modulus(), log_q);
        let distance = torus_distance_u64(phase, expected, log_q);
        if distance >= half_cell {
            mismatches.push((index, messages[index], phase, expected, distance, half_cell));
        }
    }

    assert!(
        mismatches.is_empty(),
        "torus-native public BFV->TFHE phase mismatch(s): {:?}",
        mismatches
    );
}

#[test]
fn public_bfv_slot_to_coeff_decrypts_batch_encoder_slots() {
    let params = bridge_params_b2t_torus_native_exact_large_plaintext();
    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([91u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();
    let mut bfv_enc =
        BfvEncryptor::new(params.bfv_params.clone(), SecureRng::from_seed([92u8; 32]));

    let mut tfhe_keygen =
        TfheKeyGenerator::with_rng(params.tfhe_params.clone(), SecureRng::from_seed([93u8; 32]));
    let tfhe_sk = tfhe_keygen.generate_lwe_secret_key();

    let mut bridge_keygen =
        BridgeKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([94u8; 32]));
    let bridge_keys = bridge_keygen.generate(&bfv_sk, &tfhe_sk);

    let messages = [0u64, 1, 2, 3, 17, 255, 2048, 8192];
    let mut slots = vec![0u64; params.bfv_params.degree()];
    slots[..messages.len()].copy_from_slice(&messages);
    let encoder = BatchEncoder::new(params.bfv_params.clone());
    let ct_bfv = bfv_enc.encrypt_symmetric(&bfv_sk, &encoder.encode(&slots));

    let coeff_ct = slot_to_coeff(&ct_bfv, &bridge_keys, &params).unwrap();
    let recovered = decrypt_bfv_coefficients(&coeff_ct, &params, &bfv_sk);
    assert_eq!(&recovered[..messages.len()], &messages);
}

#[test]
fn public_bfv_slots_to_tfhe_torus_native_roundtrip_batch_encoder_slots() {
    let params = bridge_params_b2t_torus_native_exact_large_plaintext();
    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([95u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();
    let mut bfv_enc =
        BfvEncryptor::new(params.bfv_params.clone(), SecureRng::from_seed([96u8; 32]));

    let mut tfhe_keygen =
        TfheKeyGenerator::with_rng(params.tfhe_params.clone(), SecureRng::from_seed([97u8; 32]));
    let tfhe_sk = tfhe_keygen.generate_lwe_secret_key();
    let tfhe_dec = TfheDecryptor::new(params.tfhe_params.clone(), tfhe_sk.clone());

    let mut bridge_keygen =
        BridgeKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([98u8; 32]));
    let bridge_keys = bridge_keygen.generate(&bfv_sk, &tfhe_sk);

    let messages = [0u64, 1, 2, 3, 17, 255, 2048, 8192];
    let mut slots = vec![0u64; params.bfv_params.degree()];
    slots[..messages.len()].copy_from_slice(&messages);
    let encoder = BatchEncoder::new(params.bfv_params.clone());
    let ct_bfv = bfv_enc.encrypt_symmetric(&bfv_sk, &encoder.encode(&slots));

    let tfhe_cts = bfv_slots_to_tfhe_torus_native(&ct_bfv, &bridge_keys, &params).unwrap();
    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    let half_cell = ((1u128 << log_q) / (2 * params.bfv_params.plain_modulus() as u128)) as u64;
    let mut mismatches = Vec::new();
    for (index, ct) in tfhe_cts.iter().take(messages.len()).enumerate() {
        let phase = tfhe_dec.decrypt_raw(ct);
        let expected =
            torus_native_encode(messages[index], params.bfv_params.plain_modulus(), log_q);
        let distance = torus_distance_u64(phase, expected, log_q);
        if distance >= half_cell {
            mismatches.push((index, messages[index], phase, expected, distance, half_cell));
        }
    }

    assert!(
        mismatches.is_empty(),
        "BFV slot->TFHE public phase mismatch(s): {:?}",
        mismatches
    );
}

#[test]
fn public_bfv_to_tfhe_torus_native_rejects_insufficient_key_switch_precision() {
    let params = bridge_params_custom(12289, 128, 64, 64, vec![50, 50], Vec::new());
    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([59u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();

    let mut tfhe_keygen =
        TfheKeyGenerator::with_rng(params.tfhe_params.clone(), SecureRng::from_seed([60u8; 32]));
    let tfhe_sk = tfhe_keygen.generate_lwe_secret_key();

    let mut bridge_keygen =
        BridgeKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([61u8; 32]));
    let bridge_keys = bridge_keygen.generate(&bfv_sk, &tfhe_sk);

    let ct_bfv = trivial_bfv_ciphertext_from_coeffs(&params, &[1]);
    let err = bfv_to_tfhe_torus_native(&ct_bfv, &bridge_keys, &params).unwrap_err();
    match err {
        super::BridgeError::IncompatibleParameters(message) => {
            assert!(message.contains("full q-bit key-switch decomposition"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn torus_decode_scalar_matches_tfhe_full_encoding() {
    let params = bridge_params_b2t_native_scale();
    let encoder = TfheEncoder::new(params.tfhe_params.clone());

    for message in 0..params.message_modulus() {
        let phase = encoder.encode_full(message).value;
        let decoded =
            super::tfhe_to_bfv::torus_decode::decode_torus_scalar_for_test(phase, &params).unwrap();
        assert_eq!(decoded, message);
    }
}

#[test]
fn tfhe_to_bfv_native_phase_repack_matches_raw_tfhe_phase() {
    let params = bridge_params_t2b_native_phase();
    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([61u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();

    let mut tfhe_keygen =
        TfheKeyGenerator::with_rng(params.tfhe_params.clone(), SecureRng::from_seed([62u8; 32]));
    let tfhe_sk = tfhe_keygen.generate_lwe_secret_key();
    let tfhe_dec = TfheDecryptor::new(params.tfhe_params.clone(), tfhe_sk.clone());
    let encoder = TfheEncoder::new(params.tfhe_params.clone());
    let mut tfhe_enc = crate::schemes::tfhe::crypto::TfheEncryptor::new(
        params.tfhe_params.clone(),
        SecureRng::from_seed([63u8; 32]),
    );

    let mut bridge_keygen =
        BridgeKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([64u8; 32]));
    let bridge_keys = bridge_keygen.generate(&bfv_sk, &tfhe_sk);

    let messages = [0u64, 1, 2, 3, 4, 5, 6, 7];
    let tfhe_cts = messages
        .iter()
        .map(|&message| tfhe_enc.encrypt_symmetric(&tfhe_sk, encoder.encode_full(message)))
        .collect::<Vec<_>>();

    let phase_ct = tfhe_to_bfv_phase_coefficients(&tfhe_cts, &params, &bridge_keys).unwrap();
    let recovered = decrypt_bfv_coefficients_exact(&phase_ct, &params, &bfv_sk);
    let mask = (1u64 << params.tfhe_params.ciphertext_modulus_log().0) - 1;
    let expected = tfhe_cts
        .iter()
        .map(|ct| tfhe_dec.decrypt_raw(ct) & mask)
        .collect::<Vec<_>>();

    assert_eq!(&recovered[..messages.len()], expected.as_slice());
}

#[test]
fn fmod_center_decode_decodes_trivial_phase_centers() {
    let params = bridge_params_fmod_center();
    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([71u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();

    let mut tfhe_keygen =
        TfheKeyGenerator::with_rng(params.tfhe_params.clone(), SecureRng::from_seed([72u8; 32]));
    let tfhe_sk = tfhe_keygen.generate_lwe_secret_key();

    let mut bridge_keygen =
        BridgeKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([73u8; 32]));
    let bridge_keys = bridge_keygen.generate(&bfv_sk, &tfhe_sk);
    assert!(!bridge_keys.fmod_keys.coeffs.is_empty());
    assert!(bridge_keys.fmod_keys.center_decode_scalar.is_some());

    let messages = [0u64, 1, 2, 3, 4, 5, 6, 7];
    let phases = messages
        .iter()
        .map(|&message| super::tfhe_to_bfv::fmod::encode_torus(message, &params))
        .collect::<Vec<_>>();
    let phase_ct = trivial_bfv_ciphertext_from_coeffs(&params, &phases);

    let decoded_ct = bfv_phase_to_bfv_messages(&phase_ct, &params, &bridge_keys).unwrap();
    let decoded = decrypt_bfv_coefficients_exact(&decoded_ct, &params, &bfv_sk);

    assert_eq!(&decoded[..messages.len()], &messages);
}

#[test]
fn full_tfhe_to_bfv_preconditions_accept_only_lwe1_exact_backend() {
    let phase_params = bridge_params_t2b_native_phase();
    let phase_err = ensure_full_tfhe_to_bfv_preconditions(&phase_params).unwrap_err();
    assert!(matches!(
        phase_err,
        super::BridgeError::IncompatibleParameters(_)
    ));

    let fmod_params = bridge_params_fmod_center();
    let fmod_err = ensure_full_tfhe_to_bfv_preconditions(&fmod_params).unwrap_err();
    assert!(matches!(
        fmod_err,
        super::BridgeError::IncompatibleParameters(_)
    ));

    let lwe1_params = bridge_params_t2b_lwe1_binary_exact();
    assert!(ensure_full_tfhe_to_bfv_preconditions(&lwe1_params).is_ok());
}

#[test]
fn torus_native_tfhe_to_bfv_rejects_high_dimensional_without_functional_switching() {
    let params = bridge_params();
    let err =
        super::tfhe_to_bfv::ensure_torus_native_tfhe_to_bfv_preconditions(&params).unwrap_err();
    match err {
        super::BridgeError::IncompatibleParameters(message) => {
            assert!(
                message.contains("Higher-dimensional inputs require SILENT functional switching")
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn public_tfhe_to_bfv_rejects_without_plaintext_lifting_backend() {
    let params = bridge_params();
    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([1u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();

    let mut tfhe_keygen =
        TfheKeyGenerator::with_rng(params.tfhe_params.clone(), SecureRng::from_seed([3u8; 32]));
    let tfhe_sk = tfhe_keygen.generate_lwe_secret_key();
    let encoder = TfheEncoder::new(params.tfhe_params.clone());
    let mut tfhe_encryptor = crate::schemes::tfhe::crypto::TfheEncryptor::new(
        params.tfhe_params.clone(),
        SecureRng::from_seed([10u8; 32]),
    );

    let mut bridge_keygen =
        BridgeKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([4u8; 32]));
    let bridge_keys = bridge_keygen.generate(&bfv_sk, &tfhe_sk);

    // Encrypt some TFHE values
    let messages = [0u64, 1, 2, 3, 0, 1, 2, 3];
    let mut tfhe_cts = Vec::new();
    for &m in messages.iter() {
        let pt = encoder.encode_full(m);
        tfhe_cts.push(tfhe_encryptor.encrypt_symmetric(&tfhe_sk, pt));
    }

    let err = tfhe_to_bfv(&tfhe_cts, &params, &bridge_keys).unwrap_err();
    assert!(matches!(err, super::BridgeError::IncompatibleParameters(_)));
}

#[test]
fn public_tfhe_to_bfv_lwe1_exact_binary_roundtrip() {
    let params = bridge_params_t2b_lwe1_binary_exact();
    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([81u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();

    let mut tfhe_keygen =
        TfheKeyGenerator::with_rng(params.tfhe_params.clone(), SecureRng::from_seed([82u8; 32]));
    let tfhe_sk = tfhe_keygen.generate_lwe_secret_key();
    let encoder = TfheEncoder::new(params.tfhe_params.clone());
    let mut tfhe_encryptor = crate::schemes::tfhe::crypto::TfheEncryptor::new(
        params.tfhe_params.clone(),
        SecureRng::from_seed([83u8; 32]),
    );

    let mut bridge_keygen =
        BridgeKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([84u8; 32]));
    let bridge_keys = bridge_keygen.generate(&bfv_sk, &tfhe_sk);

    let messages = [0u64, 1, 1, 0, 1, 0, 0, 1];
    let tfhe_cts = messages
        .iter()
        .map(|&message| tfhe_encryptor.encrypt_symmetric(&tfhe_sk, encoder.encode_full(message)))
        .collect::<Vec<_>>();

    let decoded_ct = tfhe_to_bfv(&tfhe_cts, &params, &bridge_keys).unwrap();
    let decoded = decrypt_bfv_coefficients_exact(&decoded_ct, &params, &bfv_sk);

    assert_eq!(&decoded[..messages.len()], &messages);
}

#[test]
fn public_tfhe_to_bfv_lwe1_torus_native_exact_roundtrip() {
    let params = bridge_params_t2b_lwe1_torus_native_exact();
    let mut bfv_keygen =
        BfvKeyGenerator::with_rng(params.bfv_params.clone(), SecureRng::from_seed([85u8; 32]));
    let bfv_sk = bfv_keygen.generate_secret_key();

    let mut tfhe_keygen =
        TfheKeyGenerator::with_rng(params.tfhe_params.clone(), SecureRng::from_seed([86u8; 32]));
    let tfhe_sk = tfhe_keygen.generate_lwe_secret_key();
    let mut tfhe_encryptor =
        TfheEncryptor::new(params.tfhe_params.clone(), SecureRng::from_seed([87u8; 32]));

    let mut bridge_keygen =
        BridgeKeyGenerator::with_rng(params.clone(), SecureRng::from_seed([88u8; 32]));
    let bridge_keys = bridge_keygen.generate(&bfv_sk, &tfhe_sk);

    let log_q = params.tfhe_params.ciphertext_modulus_log().0;
    let p = params.bfv_params.plain_modulus();
    let messages = [0u64, 1, 2, 3, 255, 2048, 4096, 8192];
    let tfhe_cts = messages
        .iter()
        .map(|&message| {
            let phase = torus_native_encode(message, p, log_q);
            tfhe_encryptor.encrypt_symmetric(&tfhe_sk, TfhePlaintext::new(phase, log_q))
        })
        .collect::<Vec<_>>();

    let decoded_ct = tfhe_to_bfv_torus_native_lwe1_exact(&tfhe_cts, &params, &bridge_keys).unwrap();
    let decoded = decrypt_bfv_coefficients_exact(&decoded_ct, &params, &bfv_sk);

    assert_eq!(&decoded[..messages.len()], &messages);
}
