use serde::{Deserialize, Serialize};

use crate::ahss::AhssFullState;
use crate::coefficient_cert::{CoeffCert, PrimitiveBound, apply_primitive_bound};
use crate::error::OperatorError;

#[derive(Clone, Debug, PartialEq)]
pub struct BridgeState<Ct> {
    pub ciphertext: Ct,
    pub cert: CoeffCert,
    pub compatible_with_bks_coordinate_1: bool,
    pub he_key_id: String,
}

impl<Ct> BridgeState<Ct> {
    pub fn validate(&self) -> Result<(), OperatorError> {
        self.cert.validate()?;
        if self.he_key_id.is_empty() {
            return Err(OperatorError::InvalidParams(
                "HARMONY bridge state requires a non-empty HE key id",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterializationPolicy {
    EagerFull,
    BridgeDeferred,
    BridgeImmediate,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BridgeStats {
    pub bridge_states: usize,
    pub full_materializations: usize,
    pub ciphertext_count: usize,
    pub okdm_coordinate_count: usize,
    pub storage_bytes: usize,
    pub communication_bytes: usize,
    pub latency_ms: f64,
}

impl BridgeStats {
    pub fn saved_percent_vs(&self, eager: &Self) -> f64 {
        if eager.storage_bytes == 0 {
            return 0.0;
        }
        let saved = eager.storage_bytes.saturating_sub(self.storage_bytes);
        saved as f64 * 100.0 / eager.storage_bytes as f64
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BridgePolicyInput {
    pub policy: MaterializationPolicy,
    pub bridge_state_count: usize,
    pub bridge_state_bytes: usize,
    pub full_state_bytes: usize,
    pub full_okdm_coordinate_count: usize,
    pub communication_bytes_per_materialization: usize,
    pub materialization_batch: usize,
    pub can_reuse_bridge_first_coordinate: bool,
}

impl BridgePolicyInput {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.bridge_state_count == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY bridge policy requires at least one bridge state",
            ));
        }
        if self.bridge_state_bytes == 0 || self.full_state_bytes == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY bridge policy byte sizes must be positive",
            ));
        }
        if self.full_okdm_coordinate_count == 0 || self.materialization_batch == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY bridge policy coordinate count and batch must be positive",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BridgePolicyEstimate {
    pub policy: MaterializationPolicy,
    pub bridge_states: usize,
    pub full_materializations: usize,
    pub okdm_coordinate_count: usize,
    pub storage_bytes: usize,
    pub peak_storage_bytes: usize,
    pub communication_bytes: usize,
    pub saved_percent_vs_eager_storage: f64,
    pub saved_percent_vs_eager_communication: f64,
}

pub fn estimate_bridge_policy(
    input: &BridgePolicyInput,
) -> Result<BridgePolicyEstimate, OperatorError> {
    input.validate()?;
    let eager_storage = input
        .bridge_state_count
        .saturating_mul(input.full_state_bytes);
    let eager_communication = input
        .bridge_state_count
        .saturating_mul(input.communication_bytes_per_materialization);
    let materializations = match input.policy {
        MaterializationPolicy::EagerFull | MaterializationPolicy::BridgeImmediate => {
            input.bridge_state_count
        }
        MaterializationPolicy::BridgeDeferred => {
            ceil_div_usize(input.bridge_state_count, input.materialization_batch)
        }
    };
    let resident_bridge_states = match input.policy {
        MaterializationPolicy::EagerFull => 0,
        MaterializationPolicy::BridgeImmediate | MaterializationPolicy::BridgeDeferred => {
            input.bridge_state_count
        }
    };
    let okdm_per_materialization = if input.can_reuse_bridge_first_coordinate {
        input.full_okdm_coordinate_count.saturating_sub(1)
    } else {
        input.full_okdm_coordinate_count
    };
    let okdm_coordinate_count = materializations.saturating_mul(okdm_per_materialization);
    let bridge_storage = resident_bridge_states.saturating_mul(input.bridge_state_bytes);
    let materialized_storage = materializations.saturating_mul(input.full_state_bytes);
    let storage_bytes = bridge_storage.saturating_add(materialized_storage);
    let peak_storage_bytes = match input.policy {
        MaterializationPolicy::EagerFull => input.full_state_bytes,
        MaterializationPolicy::BridgeImmediate => input
            .bridge_state_bytes
            .saturating_add(input.full_state_bytes),
        MaterializationPolicy::BridgeDeferred => input
            .bridge_state_count
            .saturating_mul(input.bridge_state_bytes)
            .saturating_add(input.full_state_bytes),
    };
    let communication_bytes =
        materializations.saturating_mul(input.communication_bytes_per_materialization);

    Ok(BridgePolicyEstimate {
        policy: input.policy.clone(),
        bridge_states: resident_bridge_states,
        full_materializations: materializations,
        okdm_coordinate_count,
        storage_bytes,
        peak_storage_bytes,
        communication_bytes,
        saved_percent_vs_eager_storage: saved_percent(eager_storage, storage_bytes),
        saved_percent_vs_eager_communication: saved_percent(
            eager_communication,
            communication_bytes,
        ),
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PublicLinearOp {
    pub name: String,
    pub primitive: PrimitiveBound,
    pub output_key_id: Option<String>,
}

impl PublicLinearOp {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.name.is_empty() {
            return Err(OperatorError::InvalidParams(
                "HARMONY bridge public linear op name must not be empty",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduledOp {
    pub name: String,
    pub requires_full_ahss: bool,
    pub allows_deferred_materialization: bool,
    pub can_reuse_bridge_first_coordinate: bool,
    pub estimated_okdm_coordinates: usize,
    pub estimated_communication_bytes: usize,
}

impl ScheduledOp {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.name.is_empty() {
            return Err(OperatorError::InvalidParams(
                "HARMONY scheduled op name must not be empty",
            ));
        }
        if self.requires_full_ahss && self.estimated_okdm_coordinates == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY full materialization requires at least one OKDM coordinate",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationDecision {
    pub policy: MaterializationPolicy,
    pub materialize_now: bool,
    pub reuse_bridge_first_coordinate: bool,
    pub okdm_coordinate_count: usize,
    pub communication_bytes: usize,
    pub reason: String,
}

pub trait HeBridgeBackend<Ct, Pt> {
    fn encrypt_bridge(&self, x: &Pt, cert: CoeffCert) -> Result<BridgeState<Ct>, OperatorError>;

    fn public_linear(
        &self,
        state: &BridgeState<Ct>,
        op: &PublicLinearOp,
    ) -> Result<BridgeState<Ct>, OperatorError>;

    fn size_bytes(&self, ct: &Ct) -> usize;
}

pub trait BridgeMaterializer<Ct, Pt, Share> {
    fn materialize_bridge(
        &self,
        bridge: &BridgeState<Ct>,
    ) -> Result<AhssFullState<Ct, Share>, OperatorError>;
}

pub fn bridge_from_he<Ct>(ciphertext: Ct, cert: CoeffCert, key_id: String) -> BridgeState<Ct> {
    BridgeState {
        ciphertext,
        cert,
        compatible_with_bks_coordinate_1: true,
        he_key_id: key_id,
    }
}

pub fn bridge_public_linear<Ct, B, Pt>(
    backend: &B,
    bridge: &BridgeState<Ct>,
    op: &PublicLinearOp,
) -> Result<BridgeState<Ct>, OperatorError>
where
    B: HeBridgeBackend<Ct, Pt>,
{
    bridge.validate()?;
    op.validate()?;
    let output_bound = apply_primitive_bound(&[bridge.cert.coeff_bound], &op.primitive)?;
    let mut out = backend.public_linear(bridge, op)?;
    out.cert = bridge.cert.with_bound(output_bound, &op.name);
    if let Some(key_id) = &op.output_key_id {
        out.he_key_id = key_id.clone();
    }
    out.compatible_with_bks_coordinate_1 = bridge.compatible_with_bks_coordinate_1
        && out.he_key_id == bridge.he_key_id
        && matches!(
            op.primitive,
            PrimitiveBound::Add { .. }
                | PrimitiveBound::Rotate { .. }
                | PrimitiveBound::PlainMul { .. }
                | PrimitiveBound::DiagonalMatmul { .. }
        );
    out.validate()?;
    Ok(out)
}

pub fn bridge_size_bytes<Ct, B, Pt>(backend: &B, bridge: &BridgeState<Ct>) -> usize
where
    B: HeBridgeBackend<Ct, Pt>,
{
    backend.size_bytes(&bridge.ciphertext)
}

pub fn eager_full_materialize<Ct, Pt, Share, B>(
    backend: &B,
    bridge: &BridgeState<Ct>,
) -> Result<AhssFullState<Ct, Share>, OperatorError>
where
    B: BridgeMaterializer<Ct, Pt, Share>,
{
    bridge.validate()?;
    backend.materialize_bridge(bridge)
}

pub fn deferred_materialize<Ct>(
    bridge: &BridgeState<Ct>,
    next_op: &ScheduledOp,
) -> Result<MaterializationDecision, OperatorError> {
    bridge.validate()?;
    next_op.validate()?;
    if !next_op.requires_full_ahss && next_op.allows_deferred_materialization {
        return Ok(MaterializationDecision {
            policy: MaterializationPolicy::BridgeDeferred,
            materialize_now: false,
            reuse_bridge_first_coordinate: false,
            okdm_coordinate_count: 0,
            communication_bytes: 0,
            reason: format!(
                "{} can consume or bypass a bridge without restricted multiplication",
                next_op.name
            ),
        });
    }

    let reuse = bridge.compatible_with_bks_coordinate_1
        && next_op.can_reuse_bridge_first_coordinate
        && next_op.estimated_okdm_coordinates > 0;
    Ok(MaterializationDecision {
        policy: if next_op.requires_full_ahss {
            MaterializationPolicy::BridgeImmediate
        } else {
            MaterializationPolicy::BridgeDeferred
        },
        materialize_now: next_op.requires_full_ahss,
        reuse_bridge_first_coordinate: reuse,
        okdm_coordinate_count: if reuse {
            next_op.estimated_okdm_coordinates.saturating_sub(1)
        } else {
            next_op.estimated_okdm_coordinates
        },
        communication_bytes: next_op.estimated_communication_bytes,
        reason: if reuse {
            "materialize before restricted multiplication with first-coordinate reuse".to_string()
        } else {
            "materialize before restricted multiplication".to_string()
        },
    })
}

fn ceil_div_usize(lhs: usize, rhs: usize) -> usize {
    if lhs == 0 { 0 } else { ((lhs - 1) / rhs) + 1 }
}

fn saved_percent(eager: usize, actual: usize) -> f64 {
    if eager == 0 {
        0.0
    } else {
        eager.saturating_sub(actual) as f64 * 100.0 / eager as f64
    }
}
