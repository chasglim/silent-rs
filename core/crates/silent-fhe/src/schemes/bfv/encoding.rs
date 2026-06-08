use crate::core::encoder::HeEncoder;
use crate::schemes::bfv::params::BfvParameters;
use silent_math::modulus::Modulus;
use silent_math::rns::RnsBase;
use silent_ring::{Poly, RingContext};
use silent_rlwe::Plaintext;

pub struct BatchEncoder {
    params: BfvParameters,
    // Context for the plaintext ring R_t = Z_t[X]/(X^N + 1)
    plain_ring: RingContext,
}

impl HeEncoder for BatchEncoder {
    type Context = BfvParameters;

    fn new(context: BfvParameters) -> Self {
        let degree = context.degree();
        let t = context.plain_modulus();

        let _mod_t = Modulus::new(t).expect("Invalid plaintext modulus");
        // Ensure t is prime and NTT friendly? RingContext construction checks this or NTT table creation will fail.
        let rns_t = RnsBase::from_values(vec![t]).expect("Failed to create RNS base for t");

        let plain_ring = RingContext::new(degree, rns_t);

        Self {
            params: context,
            plain_ring,
        }
    }

    fn encode(&self, values: &[u64]) -> Plaintext {
        let degree = self.params.degree();
        let mut coeffs = values.to_vec();

        // Pad with zeros if necessary
        if coeffs.len() < degree {
            coeffs.resize(degree, 0);
        } else if coeffs.len() > degree {
            // Truncate? Or panic? Typically panic or take first N.
            coeffs.truncate(degree);
        }

        // Input `values` are effectively in the NTT domain (slots)
        // We put them into a Poly and perform Inverse NTT to get coefficients.
        let mut poly = Poly::new(degree, 1);

        // Since Poly uses flat storage (RNS first), and we have 1 modulus:
        // data layout is just [c0, c1, ..., cN-1].
        let limb = poly.limb_mut(0);
        limb.copy_from_slice(&coeffs);

        // Perform Inverse NTT
        poly.ntt_inverse(&self.plain_ring);

        Plaintext { value: poly }
    }

    fn decode(&self, plain: &Plaintext) -> Vec<u64> {
        // Decode logic: Coeff -> NTT (Forward)
        // Clone to avoid modifying input Plaintext (which is typically immutable conceptualy)
        let mut poly = plain.value.clone();

        // Ensure poly is in same ring context or compatible?
        // Ideally verify. Here we assume it matches.

        poly.ntt_forward(&self.plain_ring);

        // Extract values
        poly.limb(0).to_vec()
    }
}
