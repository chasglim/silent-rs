use crate::core::encoder::HeEncoder;
use crate::schemes::bgv::params::BgvParameters;
use silent_math::modulus::Modulus;
use silent_math::rns::RnsBase;
use silent_ring::{Poly, RingContext};
use silent_rlwe::Plaintext;

pub struct BgvBatchEncoder {
    params: BgvParameters,
    plain_ring: RingContext,
}

impl HeEncoder for BgvBatchEncoder {
    type Context = BgvParameters;

    fn new(context: BgvParameters) -> Self {
        let degree = context.degree();
        let t = context.plain_modulus();

        let _mod_t = Modulus::new(t).expect("Invalid plaintext modulus");
        let rns_t = RnsBase::from_values(vec![t]).expect("Failed to create plaintext RNS base");
        let plain_ring = RingContext::new(degree, rns_t);

        Self {
            params: context,
            plain_ring,
        }
    }

    fn encode(&self, values: &[u64]) -> Plaintext {
        let degree = self.params.degree();
        let t = self.params.plain_modulus();
        let mut coeffs = values.iter().map(|value| value % t).collect::<Vec<_>>();

        if coeffs.len() < degree {
            coeffs.resize(degree, 0);
        } else if coeffs.len() > degree {
            coeffs.truncate(degree);
        }

        let mut poly = Poly::new(degree, 1);
        poly.limb_mut(0).copy_from_slice(&coeffs);
        poly.ntt_inverse(&self.plain_ring);

        Plaintext { value: poly }
    }

    fn decode(&self, plain: &Plaintext) -> Vec<u64> {
        let mut poly = plain.value.clone();
        poly.ntt_forward(&self.plain_ring);
        poly.limb(0).to_vec()
    }
}
