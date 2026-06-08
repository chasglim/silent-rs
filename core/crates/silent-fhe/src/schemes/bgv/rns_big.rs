use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{Signed, ToPrimitive, Zero};
use silent_math::numth;
use silent_ring::RingContext;

pub(super) fn ciphertext_modulus(ring: &RingContext) -> BigUint {
    ring.rns()
        .moduli()
        .iter()
        .fold(BigUint::from(1u8), |acc, modulus| {
            acc * BigUint::from(modulus.value())
        })
}

pub(super) fn compose_residues(residues: &[u64], ring: &RingContext, q: &BigUint) -> BigUint {
    assert_eq!(residues.len(), ring.rns().len());
    let mut acc = BigUint::zero();

    for (limb_idx, &residue) in residues.iter().enumerate() {
        let qi = ring.rns().moduli()[limb_idx].value();
        let q_hat = q / BigUint::from(qi);
        let q_hat_mod_qi = (&q_hat % qi).to_u64().expect("q_hat modulo qi fits in u64");
        let inv = numth::mod_inverse(q_hat_mod_qi, qi).expect("RNS base moduli are coprime");
        let scaled = (u128::from(residue) * u128::from(inv)) % u128::from(qi);
        acc += &q_hat * BigUint::from(scaled);
    }

    acc % q
}

pub(super) fn compose_centered(residues: &[u64], ring: &RingContext, q: &BigUint) -> BigInt {
    let value = compose_residues(residues, ring, q);
    let q_half = q >> 1usize;
    if value > q_half {
        BigInt::from_biguint(Sign::Plus, value) - BigInt::from_biguint(Sign::Plus, q.clone())
    } else {
        BigInt::from_biguint(Sign::Plus, value)
    }
}

pub(super) fn signed_mod_u64(value: &BigInt, modulus: u64) -> u64 {
    let modulus_big = BigInt::from(modulus);
    let mut residue = value % &modulus_big;
    if residue.is_negative() {
        residue += modulus_big;
    }
    residue
        .to_u64()
        .expect("residue modulo u64 modulus fits in u64")
}

pub(super) fn bit_width(value: &BigUint) -> usize {
    if value.is_zero() {
        0
    } else {
        value.bits() as usize
    }
}
