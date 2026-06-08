use rand::{RngCore, SeedableRng};
use silent_hss::{HssBatchEncoder, HssContext, HssEvaluator, HssShare};
use silent_rlwe::PublicKey;
use silent_utils::rng::SecureRng;

use crate::domain::DomainConverter;
use crate::error::OperatorError;
use crate::shares::AdditiveShares;

#[derive(Clone)]
pub struct HssSlotEngine {
    context: HssContext,
    public_key: PublicKey,
    key_share0: HssShare,
    key_share1: HssShare,
}

impl HssSlotEngine {
    pub fn new(
        context: HssContext,
        public_key: PublicKey,
        key_share0: HssShare,
        key_share1: HssShare,
    ) -> Self {
        Self {
            context,
            public_key,
            key_share0,
            key_share1,
        }
    }

    pub fn context(&self) -> &HssContext {
        &self.context
    }

    pub fn mul_slots<R: RngCore + ?Sized>(
        &self,
        lhs: &AdditiveShares,
        rhs: &AdditiveShares,
        rng: &mut R,
    ) -> Result<AdditiveShares, OperatorError> {
        if lhs.modulus() != self.context.plain_modulus()
            || rhs.modulus() != self.context.plain_modulus()
        {
            return Err(OperatorError::InvalidParams(
                "HSS slot multiplication share modulus must match HSS plaintext modulus",
            ));
        }
        if lhs.len() != rhs.len() {
            return Err(OperatorError::InvalidParams(
                "HSS slot multiplication lengths must match",
            ));
        }
        if lhs.len() > self.context.degree() {
            return Err(OperatorError::InvalidParams(
                "HSS slot multiplication length exceeds ring degree",
            ));
        }

        let mut rng_lhs0 = derive_secure_rng(rng);
        let mut rng_lhs1 = derive_secure_rng(rng);
        let ct_lhs = DomainConverter::share_slots_to_he(
            &self.context,
            &self.public_key,
            lhs.party0(),
            lhs.party1(),
            &mut rng_lhs0,
            &mut rng_lhs1,
        )?;

        let mut d2s_rng = derive_secure_rng(rng);
        let lhs_mem = DomainConverter::he_to_hss_shares(
            &self.context,
            &self.key_share0,
            &self.key_share1,
            &ct_lhs,
            &mut d2s_rng,
        )?;

        let mut rng_rhs0 = derive_secure_rng(rng);
        let mut rng_rhs1 = derive_secure_rng(rng);
        let ct_rhs = DomainConverter::share_slots_to_he(
            &self.context,
            &self.public_key,
            rhs.party0(),
            rhs.party1(),
            &mut rng_rhs0,
            &mut rng_rhs1,
        )?;

        let mut mult_rng = derive_secure_rng(rng);
        let (prod0, prod1) = HssEvaluator::mult_pair_exact(
            &self.context,
            &lhs_mem.party0,
            &lhs_mem.party1,
            &ct_rhs,
            &self.key_share0,
            &self.key_share1,
            &mut mult_rng,
        )?;

        let slots0 = self.memory_share_to_slots(&prod0, lhs.len())?;
        let slots1 = self.memory_share_to_slots(&prod1, lhs.len())?;
        AdditiveShares::new(self.context.plain_modulus(), slots0, slots1)
    }

    fn memory_share_to_slots(
        &self,
        share: &HssShare,
        len: usize,
    ) -> Result<Vec<u64>, OperatorError> {
        let coeffs = HssEvaluator::output_mem(&self.context, share, self.context.plain_modulus())?;
        let mut poly = silent_ring::Poly::new(self.context.degree(), 1);
        poly.limb_mut(0).copy_from_slice(&coeffs);
        let encoder = HssBatchEncoder::from_context(&self.context);
        let slots = encoder.decode(&poly);
        Ok(slots.into_iter().take(len).collect())
    }
}

fn derive_secure_rng<R: RngCore + ?Sized>(rng: &mut R) -> SecureRng {
    let mut seed = [0u8; 32];
    rng.fill_bytes(&mut seed);
    SecureRng::from_seed(seed)
}
