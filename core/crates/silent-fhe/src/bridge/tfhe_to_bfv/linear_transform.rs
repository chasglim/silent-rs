use super::bfv_helpers::{add_plain_to_ciphertext, mul_ciphertext_by_plain, trivial_constant_bfv};
use crate::bridge::{BridgeError, BridgeKeys, BridgeParams};
use crate::core::encoder::HeEncoder;
use crate::core::evaluator::HeEvaluator;
use crate::schemes::bfv::encoding::BatchEncoder;
use crate::schemes::bfv::ops::BfvEvaluator;
use silent_rlwe::Ciphertext;

/// BSGS Linear Transform for homomorphic repacking.
/// Evaluates ct -> M * ct + bias
pub fn bsgs_linear_transform(
    ct: &Ciphertext,
    matrix: &[Vec<u64>],
    bias: &[u64],
    keys: &BridgeKeys,
    params: &BridgeParams,
) -> Result<Ciphertext, BridgeError> {
    let n = params.bfv_params.degree();
    let evaluator = BfvEvaluator::new(params.bridge_bfv_eval_params());
    let encoder = BatchEncoder::new(params.bfv_params.clone());

    // BSGS parameters: n1 * n2 = n
    let n1 = (n as f64).sqrt().ceil() as usize;
    let n2 = (n + n1 - 1) / n1;

    // 1. Precompute rotations for baby steps: rotate(ct, i) for i in 0..n1
    // Note: SIMD rotation by i uses galois element 3^i mod 2n
    let gk = keys.bfv_galois_keys.as_ref().ok_or_else(|| {
        BridgeError::IncompatibleParameters("Galois keys missing for linear transform".into())
    })?;

    let mut baby_steps = Vec::with_capacity(n1);
    baby_steps.push(ct.clone());
    for i in 1..n1 {
        let g = silent_math::numth::mod_pow(3, i as u64, (2 * n) as u64);
        baby_steps.push(evaluator.rotate(ct, g as u32, gk));
    }

    let mut outer_sum = trivial_constant_bfv(params, 0);
    let mut has_outer = false;

    for j in 0..n2 {
        let mut inner_sum = trivial_constant_bfv(params, 0);
        let mut has_inner = false;

        for i in 0..n1 {
            let k = i + j * n1;
            if k >= n {
                break;
            }

            let diag_slots = (0..n)
                .map(|l| matrix[l][(l + k) % n] % params.bfv_params.plain_modulus())
                .collect::<Vec<_>>();
            let diag_plain = encoder.encode(&diag_slots);

            let term = mul_ciphertext_by_plain(&baby_steps[i], &diag_plain, params);
            if !has_inner {
                inner_sum = term;
                has_inner = true;
            } else {
                inner_sum = evaluator.add(&inner_sum, &term);
            }
        }

        if has_inner {
            let giant_step_k = (j * n1) as u64;
            let term = if giant_step_k == 0 {
                inner_sum
            } else {
                let g = silent_math::numth::mod_pow(3, giant_step_k, (2 * n) as u64);
                evaluator.rotate(&inner_sum, g as u32, gk)
            };

            if !has_outer {
                outer_sum = term;
                has_outer = true;
            } else {
                outer_sum = evaluator.add(&outer_sum, &term);
            }
        }
    }

    let mut bias_slots = vec![0u64; n];
    for (i, &b) in bias.iter().enumerate().take(n) {
        bias_slots[i] = b % params.bfv_params.plain_modulus();
    }
    let bias_plain = encoder.encode(&bias_slots);

    Ok(add_plain_to_ciphertext(&outer_sum, &bias_plain, params))
}

/// Dense but exact slot-domain linear transform using the actual Galois
/// automorphism permutations induced by SILENT's `BatchEncoder`.
///
/// This is the correctness-first path used for BFV SlotsToCoefficients. It
/// implements `y = matrix * x + bias` over batching slots as
/// `sum_g diag(d_g) * Gal_g(x)`, where each diagonal is derived from the
/// concrete `automorphism_map(g)` rather than assuming a cyclic slot order.
pub fn galois_linear_transform(
    ct: &Ciphertext,
    matrix: &[Vec<u64>],
    bias: &[u64],
    keys: &BridgeKeys,
    params: &BridgeParams,
) -> Result<Ciphertext, BridgeError> {
    let n = params.bfv_params.degree();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return Err(BridgeError::IncompatibleParameters(format!(
            "slot linear transform expects an {n}x{n} matrix"
        )));
    }

    let evaluator = BfvEvaluator::new(params.bridge_bfv_eval_params());
    let encoder = BatchEncoder::new(params.bfv_params.clone());
    let gk = keys.bfv_galois_keys.as_ref().ok_or_else(|| {
        BridgeError::IncompatibleParameters("Galois keys missing for slot linear transform".into())
    })?;

    let t = params.bfv_params.plain_modulus();
    let mut outer_sum = trivial_constant_bfv(params, 0);
    let mut has_outer = false;

    for g in all_galois_elements(n) {
        if g != 1 && !gk.keys.contains_key(&g) {
            return Err(BridgeError::IncompatibleParameters(format!(
                "Galois key for automorphism {g} missing from bridge key material"
            )));
        }

        let map = params.bfv_params.ring().automorphism_map(g as u64);
        let mut diag_slots = vec![0u64; n];
        let mut nonzero = false;
        for row in 0..n {
            let value = matrix[row][map[row]] % t;
            diag_slots[row] = value;
            nonzero |= value != 0;
        }
        if !nonzero {
            continue;
        }

        let rotated = if g == 1 {
            ct.clone()
        } else {
            evaluator.rotate(ct, g, gk)
        };
        let diag_plain = encoder.encode(&diag_slots);
        let term = mul_ciphertext_by_plain(&rotated, &diag_plain, params);
        if !has_outer {
            outer_sum = term;
            has_outer = true;
        } else {
            outer_sum = evaluator.add(&outer_sum, &term);
        }
    }

    let mut bias_slots = vec![0u64; n];
    for (i, &b) in bias.iter().enumerate().take(n) {
        bias_slots[i] = b % t;
    }
    let bias_plain = encoder.encode(&bias_slots);

    Ok(add_plain_to_ciphertext(&outer_sum, &bias_plain, params))
}

fn all_galois_elements(n: usize) -> impl Iterator<Item = u32> {
    (1..(2 * n) as u32).step_by(2)
}
