use crate::bridge::BridgeParams;
use crate::core::evaluator::HeEvaluator;
use crate::schemes::bfv::ops::BfvEvaluator;
use silent_ring::Poly;
use silent_rlwe::{Ciphertext, Plaintext};

pub(super) fn trivial_constant_bfv(params: &BridgeParams, value: u64) -> Ciphertext {
    let degree = params.bfv_params.degree();
    let mut c0 = Poly::new(degree, params.bfv_params.ring().rns().len());
    let delta = bfv_delta_rns(params);
    for (i, modulus) in params.bfv_params.ring().rns().moduli().iter().enumerate() {
        let qi = modulus.value();
        let scaled = silent_math::arith::mul_mod_u64(value % qi, delta[i], qi);
        c0.limb_mut(i)[0] = scaled;
    }
    let c1 = Poly::new(degree, params.bfv_params.ring().rns().len());
    Ciphertext::new(
        vec![c0, c1],
        params.bfv_params.runtime_params_arc().as_ref().clone(),
        false,
    )
}

pub(super) fn scalar_mul_bfv(ct: &Ciphertext, scalar: u64, params: &BridgeParams) -> Ciphertext {
    let ring = params.bfv_params.ring();
    let mut out = ct.clone();
    for poly in out.data.iter_mut() {
        for (i, modulus) in ring.rns().moduli().iter().enumerate() {
            let qi = modulus.value();
            let ratio = modulus.const_ratio();
            let limb = poly.limb_mut(i);
            for coeff in limb.iter_mut() {
                *coeff = silent_math::arith::mul_mod_barrett_u64(
                    *coeff,
                    scalar % qi,
                    qi,
                    ratio[0],
                    ratio[1],
                );
            }
        }
    }
    out
}

pub(super) fn add_constant_bfv(
    ct: &Ciphertext,
    constant: u64,
    params: &BridgeParams,
) -> Ciphertext {
    let evaluator = BfvEvaluator::new(params.bridge_bfv_eval_params());
    evaluator.add(ct, &trivial_constant_bfv(params, constant))
}

pub(super) fn shift_ciphertext_by_monomial(
    ct: &Ciphertext,
    power: usize,
    params: &BridgeParams,
) -> Ciphertext {
    let degree = params.bfv_params.degree();
    let num_moduli = params.bfv_params.ring().rns().len();
    let mut out = ct.clone();
    for poly in out.data.iter_mut() {
        for limb_idx in 0..num_moduli {
            let src = poly.limb(limb_idx).to_vec();
            let dst = poly.limb_mut(limb_idx);
            dst.fill(0);
            for (j, value) in src.iter().copied().enumerate() {
                let target = j + power;
                if target < degree {
                    dst[target] = value;
                } else {
                    let wrapped = target - degree;
                    let qi = params.bfv_params.ring().rns().moduli()[limb_idx].value();
                    dst[wrapped] = if value == 0 { 0 } else { qi - value };
                }
            }
        }
    }
    out
}

fn encode_plaintext_to_bfv(params: &BridgeParams, plain: &Plaintext) -> Plaintext {
    let degree = params.bfv_params.degree();
    let ring = params.bfv_params.ring();
    let mut out = Poly::new(degree, ring.rns().len());
    let delta = bfv_delta_rns(params);

    for (i, modulus) in ring.rns().moduli().iter().enumerate() {
        let qi = modulus.value();
        let limb_in = if plain.value.num_moduli() == 1 {
            plain.value.limb(0)
        } else {
            plain.value.limb(i)
        };
        let limb_out = out.limb_mut(i);
        for (c_out, &c_in) in limb_out.iter_mut().zip(limb_in.iter()) {
            *c_out = silent_math::arith::mul_mod_u64(c_in % qi, delta[i], qi);
        }
    }

    Plaintext { value: out }
}

fn bfv_delta_rns(params: &BridgeParams) -> Vec<u64> {
    let q_moduli = params.bfv_params.ring().rns().moduli();
    let t = params.bfv_params.plain_modulus();
    let q_mod_t = q_moduli.iter().fold(1u64, |acc, modulus| {
        ((acc as u128 * (modulus.value() % t) as u128) % t as u128) as u64
    });

    q_moduli
        .iter()
        .map(|modulus| {
            let qi = modulus.value();
            let t_inv = silent_math::numth::mod_inverse(t % qi, qi)
                .expect("BFV plaintext modulus must be coprime with every ciphertext modulus");
            let q_mod_t_mod_qi = q_mod_t % qi;
            let neg_q_mod_t = if q_mod_t_mod_qi == 0 {
                0
            } else {
                qi - q_mod_t_mod_qi
            };
            silent_math::arith::mul_mod_u64(neg_q_mod_t, t_inv, qi)
        })
        .collect()
}

pub(super) fn mul_ciphertext_by_plain(
    ct: &Ciphertext,
    plain: &Plaintext,
    params: &BridgeParams,
) -> Ciphertext {
    let mut res = ct.clone();
    let ring = params.bfv_params.ring();

    let mut p = plain.value.clone();
    if p.num_moduli() == 1 && res.data[0].num_moduli() > 1 {
        let mut expanded = Poly::new(p.degree(), res.data[0].num_moduli());
        let limb0 = p.limb(0).to_vec();
        for i in 0..expanded.num_moduli() {
            expanded.limb_mut(i).copy_from_slice(&limb0);
        }
        p = expanded;
    }

    if !res.is_ntt {
        for poly in res.data.iter_mut() {
            poly.ntt_forward(ring);
        }
        res.is_ntt = true;
    }
    p.ntt_forward(ring);

    for poly in res.data.iter_mut() {
        poly.mul_assign(&p, ring);
    }
    res
}

pub(super) fn add_plain_to_ciphertext(
    ct: &Ciphertext,
    plain: &Plaintext,
    params: &BridgeParams,
) -> Ciphertext {
    let mut res = ct.clone();
    let ring = params.bfv_params.ring();
    let encoded = encode_plaintext_to_bfv(params, plain);
    let mut p = encoded.value;
    if res.is_ntt {
        p.ntt_forward(ring);
    }
    res.data[0].add_assign(&p, ring);
    res
}
