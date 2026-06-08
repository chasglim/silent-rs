use rand::Rng;

use crate::error::OperatorError;
use crate::linear_map::LinearMap;

/// Transport-neutral state for converting additive shares in `Z_q` to shares in
/// `Z_p` with a semi-honest masked opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QToPMaskedOpenState {
    q_modulus: u64,
    p_modulus: u64,
    masks_p: Vec<u64>,
    masked_q: Vec<u64>,
}

impl QToPMaskedOpenState {
    pub fn q_modulus(&self) -> u64 {
        self.q_modulus
    }

    pub fn p_modulus(&self) -> u64 {
        self.p_modulus
    }

    pub fn masked_q(&self) -> &[u64] {
        &self.masked_q
    }

    pub fn len(&self) -> usize {
        self.masked_q.len()
    }

    pub fn is_empty(&self) -> bool {
        self.masked_q.is_empty()
    }
}

pub struct ShareConverter;

impl ShareConverter {
    pub fn mask_q_to_p<R: Rng + ?Sized>(
        shares_q: &[u64],
        q_modulus: u64,
        p_modulus: u64,
        rng: &mut R,
    ) -> Result<QToPMaskedOpenState, OperatorError> {
        validate_q_to_p_inputs(shares_q, q_modulus, p_modulus)?;
        if shares_q.is_empty() {
            return Ok(QToPMaskedOpenState {
                q_modulus,
                p_modulus,
                masks_p: Vec::new(),
                masked_q: Vec::new(),
            });
        }

        let scale = q_modulus / p_modulus;
        let mut masks_p = Vec::with_capacity(shares_q.len());
        let mut masked_q = Vec::with_capacity(shares_q.len());
        for &share in shares_q {
            let mask_p = rng.gen_range(0..p_modulus);
            let mask_q = ((mask_p as u128) * (scale as u128) % q_modulus as u128) as u64;
            masks_p.push(mask_p);
            masked_q.push(add_mod_u64(share, mask_q, q_modulus));
        }

        Ok(QToPMaskedOpenState {
            q_modulus,
            p_modulus,
            masks_p,
            masked_q,
        })
    }

    pub fn finish_q_to_p(
        party_index: usize,
        state: &QToPMaskedOpenState,
        peer_masked_q: &[u64],
    ) -> Result<Vec<u64>, OperatorError> {
        if party_index > 1 {
            return Err(OperatorError::InvalidParams("party_index must be 0 or 1"));
        }
        if peer_masked_q.len() != state.len() {
            return Err(OperatorError::Protocol(
                "q->p conversion received malformed payload",
            ));
        }
        if peer_masked_q.iter().any(|&value| value >= state.q_modulus) {
            return Err(OperatorError::Protocol(
                "q->p conversion peer payload is out of Z_q range",
            ));
        }

        let opened_q = state
            .masked_q
            .iter()
            .zip(peer_masked_q.iter())
            .map(|(&local, &peer)| add_mod_u64(local, peer, state.q_modulus))
            .collect::<Vec<_>>();
        let opened_p = LinearMap::decode_q_to_p(&opened_q, state.q_modulus, state.p_modulus);

        let mut out = Vec::with_capacity(state.len());
        for (idx, &mask_p) in state.masks_p.iter().enumerate() {
            let share = if party_index == 0 {
                add_mod_u64(opened_p[idx], state.p_modulus - mask_p, state.p_modulus)
            } else if mask_p == 0 {
                0
            } else {
                state.p_modulus - mask_p
            };
            out.push(share);
        }
        Ok(out)
    }

    pub fn convert_pair_local<R0: Rng + ?Sized, R1: Rng + ?Sized>(
        party0_q: &[u64],
        party1_q: &[u64],
        q_modulus: u64,
        p_modulus: u64,
        rng0: &mut R0,
        rng1: &mut R1,
    ) -> Result<(Vec<u64>, Vec<u64>), OperatorError> {
        if party0_q.len() != party1_q.len() {
            return Err(OperatorError::InvalidParams(
                "q->p share vectors must have equal length",
            ));
        }
        let state0 = Self::mask_q_to_p(party0_q, q_modulus, p_modulus, rng0)?;
        let state1 = Self::mask_q_to_p(party1_q, q_modulus, p_modulus, rng1)?;
        let party0_p = Self::finish_q_to_p(0, &state0, state1.masked_q())?;
        let party1_p = Self::finish_q_to_p(1, &state1, state0.masked_q())?;
        Ok((party0_p, party1_p))
    }
}

fn validate_q_to_p_inputs(
    shares_q: &[u64],
    q_modulus: u64,
    p_modulus: u64,
) -> Result<(), OperatorError> {
    if q_modulus < p_modulus || p_modulus < 2 {
        return Err(OperatorError::InvalidParams(
            "invalid moduli for q->p conversion",
        ));
    }
    if q_modulus / p_modulus == 0 {
        return Err(OperatorError::InvalidParams("q_modulus/p_modulus is zero"));
    }
    if shares_q.iter().any(|&value| value >= q_modulus) {
        return Err(OperatorError::InvalidParams(
            "shares_q entries must be canonical elements in Z_q",
        ));
    }
    Ok(())
}

#[inline]
fn add_mod_u64(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    ((lhs as u128 + rhs as u128) % modulus as u128) as u64
}
