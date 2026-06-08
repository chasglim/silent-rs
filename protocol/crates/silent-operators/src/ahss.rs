use serde::{Deserialize, Serialize};

use crate::coefficient_cert::{CoeffCert, PrimitiveBound, apply_primitive_bound};
use crate::error::OperatorError;

#[derive(Clone, Debug, PartialEq)]
pub struct AhssFullState<Ct, Share> {
    pub input_material: OkdmMaterial<Ct>,
    pub memory_shares: [AhssMemoryShare<Share>; 2],
    pub cert: CoeffCert,
}

impl<Ct, Share> AhssFullState<Ct, Share> {
    pub fn validate(&self) -> Result<(), OperatorError> {
        self.cert.validate()?;
        self.input_material.validate()?;
        if self.memory_shares[0].party_id == self.memory_shares[1].party_id {
            return Err(OperatorError::InvalidParams(
                "HARMONY AHSS full state requires distinct party ids",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AhssMemoryShare<Share> {
    pub party_id: usize,
    pub value_times_secret_share: Share,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OkdmMaterial<Ct> {
    pub coordinates: Vec<Ct>,
    pub coordinate_tags: Vec<String>,
    pub noise_bound: f64,
}

impl<Ct> OkdmMaterial<Ct> {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.coordinates.is_empty() {
            return Err(OperatorError::InvalidParams(
                "HARMONY OKDM material must contain at least one coordinate",
            ));
        }
        if self.coordinate_tags.len() != self.coordinates.len() {
            return Err(OperatorError::InvalidParams(
                "HARMONY OKDM material coordinate tag count must match coordinates",
            ));
        }
        if !self.noise_bound.is_finite() || self.noise_bound < 0.0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY OKDM material noise bound must be finite and non-negative",
            ));
        }
        Ok(())
    }

    pub fn coordinate_count(&self) -> usize {
        self.coordinates.len()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AhssEvalStats {
    pub okdm_count: usize,
    pub ddec_count: usize,
    pub ring_add_count: usize,
    pub ring_mul_count: usize,
    pub communication_bytes: usize,
    pub elapsed_ms: f64,
}

impl AhssEvalStats {
    pub fn merge(&mut self, rhs: &Self) {
        self.okdm_count += rhs.okdm_count;
        self.ddec_count += rhs.ddec_count;
        self.ring_add_count += rhs.ring_add_count;
        self.ring_mul_count += rhs.ring_mul_count;
        self.communication_bytes += rhs.communication_bytes;
        self.elapsed_ms += rhs.elapsed_ms;
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearOpPlan {
    pub op_name: String,
    pub primitive: PrimitiveBound,
    pub coordinate_tags: Vec<String>,
    pub communication_bytes: usize,
    pub ring_add_count: usize,
    pub ring_mul_count: usize,
}

impl LinearOpPlan {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.op_name.is_empty() {
            return Err(OperatorError::InvalidParams(
                "HARMONY AHSS linear op name must not be empty",
            ));
        }
        if self.coordinate_tags.is_empty() {
            return Err(OperatorError::InvalidParams(
                "HARMONY AHSS linear op must keep at least one coordinate",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestrictedMulPlan {
    pub op_name: String,
    pub input_tag: String,
    pub communication_bytes: usize,
    pub ring_mul_count: usize,
}

impl RestrictedMulPlan {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.op_name.is_empty() || self.input_tag.is_empty() {
            return Err(OperatorError::InvalidParams(
                "HARMONY restricted multiplication plan requires op and input tags",
            ));
        }
        Ok(())
    }
}

pub trait OkdmOracle<Ct, Pt> {
    fn okdm(&self, value: &Pt, coordinate_index: usize) -> Result<Ct, OperatorError>;
}

pub trait DistributedDecrypt<Ct, Share, Pt> {
    fn ddec(&self, party_id: usize, key_share: &Share, ct: &Ct) -> Result<Pt, OperatorError>;
}

pub trait AhssBackend<Ct, Pt, Share>:
    OkdmOracle<Ct, Pt> + DistributedDecrypt<Ct, Share, Pt>
{
    fn load_share(
        &self,
        party_id: usize,
        material: &OkdmMaterial<Ct>,
        cert: &CoeffCert,
    ) -> Result<Share, OperatorError>;

    fn eval_linear_material(
        &self,
        material: &OkdmMaterial<Ct>,
        op: &LinearOpPlan,
    ) -> Result<OkdmMaterial<Ct>, OperatorError>;

    fn eval_linear_share(
        &self,
        share: &AhssMemoryShare<Share>,
        op: &LinearOpPlan,
    ) -> Result<AhssMemoryShare<Share>, OperatorError>;

    fn restricted_mul_share(
        &self,
        input_material: &OkdmMaterial<Ct>,
        memory_share: &AhssMemoryShare<Share>,
        plan: &RestrictedMulPlan,
    ) -> Result<AhssMemoryShare<Share>, OperatorError>;

    fn add_ct(&self, a: &Ct, b: &Ct) -> Result<Ct, OperatorError>;

    fn add_pt(&self, a: &Pt, b: &Pt) -> Result<Pt, OperatorError>;
}

pub trait AhssShareReconstruct<Pt>: Sized {
    fn reconstruct_pair(lhs: &Self, rhs: &Self, modulus: u128) -> Result<Pt, OperatorError>;
}

impl AhssShareReconstruct<u128> for u128 {
    fn reconstruct_pair(lhs: &Self, rhs: &Self, modulus: u128) -> Result<u128, OperatorError> {
        if modulus < 2 {
            return Err(OperatorError::InvalidParams(
                "HARMONY reconstruction modulus must be at least two",
            ));
        }
        Ok((lhs % modulus + rhs % modulus) % modulus)
    }
}

impl AhssShareReconstruct<Vec<u128>> for Vec<u128> {
    fn reconstruct_pair(lhs: &Self, rhs: &Self, modulus: u128) -> Result<Vec<u128>, OperatorError> {
        if modulus < 2 {
            return Err(OperatorError::InvalidParams(
                "HARMONY reconstruction modulus must be at least two",
            ));
        }
        if lhs.len() != rhs.len() {
            return Err(OperatorError::InvalidParams(
                "HARMONY reconstruction vector share lengths must match",
            ));
        }
        Ok(lhs
            .iter()
            .zip(rhs.iter())
            .map(|(&a, &b)| (a % modulus + b % modulus) % modulus)
            .collect())
    }
}

pub fn load_full<Ct, Pt, Share, B>(
    backend: &B,
    material: OkdmMaterial<Ct>,
    cert: CoeffCert,
) -> Result<AhssFullState<Ct, Share>, OperatorError>
where
    B: AhssBackend<Ct, Pt, Share>,
{
    material.validate()?;
    cert.validate()?;
    let party0 = backend.load_share(0, &material, &cert)?;
    let party1 = backend.load_share(1, &material, &cert)?;
    Ok(AhssFullState {
        input_material: material,
        memory_shares: [
            AhssMemoryShare {
                party_id: 0,
                value_times_secret_share: party0,
            },
            AhssMemoryShare {
                party_id: 1,
                value_times_secret_share: party1,
            },
        ],
        cert,
    })
}

pub fn eval_linear<Ct, Pt, Share, B>(
    backend: &B,
    state: &AhssFullState<Ct, Share>,
    op: &LinearOpPlan,
) -> Result<(AhssFullState<Ct, Share>, AhssEvalStats), OperatorError>
where
    B: AhssBackend<Ct, Pt, Share>,
{
    state.validate()?;
    op.validate()?;
    let output_bound = apply_primitive_bound(&[state.cert.coeff_bound], &op.primitive)?;
    let mut cert = state.cert.with_bound(output_bound, &op.op_name);
    cert.noise_bound += state.input_material.noise_bound;

    let input_material = backend.eval_linear_material(&state.input_material, op)?;
    let memory_shares = [
        backend.eval_linear_share(&state.memory_shares[0], op)?,
        backend.eval_linear_share(&state.memory_shares[1], op)?,
    ];
    let stats = AhssEvalStats {
        okdm_count: 0,
        ddec_count: 0,
        ring_add_count: op.ring_add_count,
        ring_mul_count: op.ring_mul_count,
        communication_bytes: op.communication_bytes,
        elapsed_ms: 0.0,
    };
    Ok((
        AhssFullState {
            input_material,
            memory_shares,
            cert,
        },
        stats,
    ))
}

pub fn restricted_mul<Ct, Pt, Share, B>(
    backend: &B,
    input: &AhssFullState<Ct, Share>,
    memory: &[AhssMemoryShare<Share>; 2],
    plan: &RestrictedMulPlan,
) -> Result<([AhssMemoryShare<Share>; 2], AhssEvalStats), OperatorError>
where
    B: AhssBackend<Ct, Pt, Share>,
{
    input.validate()?;
    plan.validate()?;
    if memory[0].party_id == memory[1].party_id {
        return Err(OperatorError::InvalidParams(
            "HARMONY restricted multiplication requires distinct party shares",
        ));
    }
    let out = [
        backend.restricted_mul_share(&input.input_material, &memory[0], plan)?,
        backend.restricted_mul_share(&input.input_material, &memory[1], plan)?,
    ];
    let stats = AhssEvalStats {
        okdm_count: input.input_material.coordinate_count(),
        ddec_count: 2,
        ring_add_count: 0,
        ring_mul_count: plan.ring_mul_count,
        communication_bytes: plan.communication_bytes,
        elapsed_ms: 0.0,
    };
    Ok((out, stats))
}

pub fn reconstruct<Pt, Share>(
    shares: &[AhssMemoryShare<Share>; 2],
    modulus: u128,
) -> Result<Pt, OperatorError>
where
    Share: AhssShareReconstruct<Pt>,
{
    if shares[0].party_id == shares[1].party_id {
        return Err(OperatorError::InvalidParams(
            "HARMONY reconstruction requires distinct party shares",
        ));
    }
    Share::reconstruct_pair(
        &shares[0].value_times_secret_share,
        &shares[1].value_times_secret_share,
        modulus,
    )
}
