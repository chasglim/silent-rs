use serde::{Deserialize, Serialize};

use crate::error::OperatorError;
use crate::transformer_pipeline::PipelineWorkloadPlan;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaselineKind {
    BksNaive,
    PureFhePacked,
    CheetahStyle,
    MpcBeaver,
    VoleStyle,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaselineOp {
    LinearBackbone,
    Attention,
    QkScore,
    Pv,
    Ffn,
    TransformerBlock,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BaselineCalibration {
    pub ahss_linear_ms: f64,
    pub restricted_mul_ms: f64,
    pub direct_reload_ms: f64,
    pub pbs_ms: f64,
    pub fhe_linear_ms: f64,
    pub mpc_mul_ms: f64,
    pub vole_mul_ms: f64,
    pub beaver_offline_ms: f64,
    pub ciphertext_bytes: usize,
    pub share_pair_bytes: usize,
    pub full_state_bytes: usize,
}

impl Default for BaselineCalibration {
    fn default() -> Self {
        Self {
            ahss_linear_ms: 0.20,
            restricted_mul_ms: 0.80,
            direct_reload_ms: 0.50,
            pbs_ms: 1.20,
            fhe_linear_ms: 2.50,
            mpc_mul_ms: 0.25,
            vole_mul_ms: 0.18,
            beaver_offline_ms: 0.35,
            ciphertext_bytes: 4096,
            share_pair_bytes: 2048,
            full_state_bytes: 8192,
        }
    }
}

impl BaselineCalibration {
    pub fn validate(&self) -> Result<(), OperatorError> {
        let latencies = [
            self.ahss_linear_ms,
            self.restricted_mul_ms,
            self.direct_reload_ms,
            self.pbs_ms,
            self.fhe_linear_ms,
            self.mpc_mul_ms,
            self.vole_mul_ms,
            self.beaver_offline_ms,
        ];
        if latencies
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(OperatorError::InvalidParams(
                "HARMONY baseline calibration latencies must be finite and non-negative",
            ));
        }
        if self.ciphertext_bytes == 0 || self.share_pair_bytes == 0 || self.full_state_bytes == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY baseline calibration byte sizes must be positive",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BaselineInput {
    pub baseline: BaselineKind,
    pub op: BaselineOp,
    pub workload: PipelineWorkloadPlan,
    pub calibration: BaselineCalibration,
    pub ring_n: usize,
    pub q_bits: usize,
}

impl BaselineInput {
    pub fn validate(&self) -> Result<(), OperatorError> {
        self.calibration.validate()?;
        if self.ring_n == 0 || self.q_bits == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY baseline ring degree and q bits must be positive",
            ));
        }
        if self.workload.full_total_ops() == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY baseline workload must contain at least one operation",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BaselineEstimate {
    pub baseline: BaselineKind,
    pub op: BaselineOp,
    pub analytical: bool,
    pub offline_latency_ms: f64,
    pub online_latency_ms: f64,
    pub total_latency_ms: f64,
    pub communication_bytes: usize,
    pub storage_bytes: usize,
    pub linear_ops: usize,
    pub restricted_mul_ops: usize,
    pub nonlinear_rows: usize,
    pub notes: String,
}

pub fn estimate_baseline(input: &BaselineInput) -> Result<BaselineEstimate, OperatorError> {
    input.validate()?;
    let counts = counts_for_op(&input.workload, &input.op);
    let cal = &input.calibration;
    let linear = counts.linear_ops as f64;
    let restricted = counts.restricted_mul_ops as f64;
    let nonlinear = counts.nonlinear_rows as f64;

    let (offline_latency_ms, online_latency_ms, communication_bytes, storage_bytes, notes) =
        match input.baseline {
            BaselineKind::BksNaive => {
                let online = linear * cal.ahss_linear_ms
                    + restricted * cal.restricted_mul_ms
                    + nonlinear * (cal.direct_reload_ms + 5.0 * cal.pbs_ms);
                let communication = counts
                    .restricted_mul_ops
                    .saturating_add(counts.nonlinear_rows)
                    .saturating_mul(cal.share_pair_bytes);
                let storage = counts
                    .linear_ops
                    .saturating_add(counts.restricted_mul_ops)
                    .saturating_add(counts.nonlinear_rows)
                    .saturating_mul(cal.full_state_bytes);
                (
                    0.0,
                    online,
                    communication,
                    storage,
                    "analytical_calibrated:bks_naive_no_bridge_no_deferred_reload".to_string(),
                )
            }
            BaselineKind::PureFhePacked => {
                let fhe_ops = counts.linear_ops.saturating_add(counts.restricted_mul_ops);
                let online = fhe_ops as f64 * cal.fhe_linear_ms + nonlinear * 6.0 * cal.pbs_ms;
                let communication = nonlinear as usize * cal.ciphertext_bytes;
                let storage = fhe_ops
                    .saturating_add(counts.nonlinear_rows)
                    .saturating_mul(cal.ciphertext_bytes);
                (
                    0.0,
                    online,
                    communication,
                    storage,
                    "analytical_calibrated:pure_fhe_packed_linear_and_pbs_nonlinear".to_string(),
                )
            }
            BaselineKind::CheetahStyle => {
                let secure_ops = counts
                    .linear_ops
                    .saturating_add(counts.restricted_mul_ops)
                    .saturating_add(counts.nonlinear_rows);
                let offline = secure_ops as f64 * cal.beaver_offline_ms;
                let online = secure_ops as f64 * cal.mpc_mul_ms + nonlinear * cal.direct_reload_ms;
                let communication = secure_ops.saturating_mul(cal.share_pair_bytes);
                let storage = secure_ops.saturating_mul(cal.share_pair_bytes);
                (
                    offline,
                    online,
                    communication,
                    storage,
                    "analytical_calibrated:cheetah_style_he_ss_with_preprocessed_products"
                        .to_string(),
                )
            }
            BaselineKind::MpcBeaver => {
                let secure_ops = counts
                    .linear_ops
                    .saturating_add(counts.restricted_mul_ops)
                    .saturating_add(counts.nonlinear_rows);
                let offline = secure_ops as f64 * cal.beaver_offline_ms;
                let online = secure_ops as f64 * cal.mpc_mul_ms;
                let communication = secure_ops.saturating_mul(cal.share_pair_bytes);
                let storage = secure_ops.saturating_mul(cal.share_pair_bytes);
                (
                    offline,
                    online,
                    communication,
                    storage,
                    "analytical_calibrated:ot_mpc_beaver_triple_baseline".to_string(),
                )
            }
            BaselineKind::VoleStyle => {
                let secure_ops = counts
                    .linear_ops
                    .saturating_add(counts.restricted_mul_ops)
                    .saturating_add(counts.nonlinear_rows);
                let offline = secure_ops as f64 * cal.beaver_offline_ms * 0.55;
                let online = secure_ops as f64 * cal.vole_mul_ms;
                let communication = secure_ops.saturating_mul(cal.share_pair_bytes / 2);
                let storage = secure_ops.saturating_mul(cal.share_pair_bytes);
                (
                    offline,
                    online,
                    communication,
                    storage,
                    "analytical_calibrated:vole_style_correlated_randomness_baseline".to_string(),
                )
            }
        };

    Ok(BaselineEstimate {
        baseline: input.baseline.clone(),
        op: input.op.clone(),
        analytical: true,
        offline_latency_ms,
        online_latency_ms,
        total_latency_ms: offline_latency_ms + online_latency_ms,
        communication_bytes,
        storage_bytes,
        linear_ops: counts.linear_ops,
        restricted_mul_ops: counts.restricted_mul_ops,
        nonlinear_rows: counts.nonlinear_rows,
        notes,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BaselineCounts {
    linear_ops: usize,
    restricted_mul_ops: usize,
    nonlinear_rows: usize,
}

fn counts_for_op(workload: &PipelineWorkloadPlan, op: &BaselineOp) -> BaselineCounts {
    match op {
        BaselineOp::LinearBackbone => BaselineCounts {
            linear_ops: workload
                .full_qkv_projection_ops
                .saturating_add(workload.full_residual_ops)
                .saturating_add(workload.full_ffn_first_ops)
                .saturating_add(workload.full_ffn_second_ops),
            restricted_mul_ops: workload
                .full_qk_score_ops
                .saturating_add(workload.full_pv_ops),
            nonlinear_rows: workload.full_layernorm_rows,
        },
        BaselineOp::Attention => BaselineCounts {
            linear_ops: workload.full_qkv_projection_ops,
            restricted_mul_ops: workload
                .full_qk_score_ops
                .saturating_add(workload.full_pv_ops),
            nonlinear_rows: workload.full_softmax_rows,
        },
        BaselineOp::QkScore => BaselineCounts {
            linear_ops: 0,
            restricted_mul_ops: workload.full_qk_score_ops,
            nonlinear_rows: 0,
        },
        BaselineOp::Pv => BaselineCounts {
            linear_ops: 0,
            restricted_mul_ops: workload.full_pv_ops,
            nonlinear_rows: 0,
        },
        BaselineOp::Ffn => BaselineCounts {
            linear_ops: workload
                .full_ffn_first_ops
                .saturating_add(workload.full_ffn_second_ops),
            restricted_mul_ops: workload.full_gelu_ops,
            nonlinear_rows: 0,
        },
        BaselineOp::TransformerBlock => BaselineCounts {
            linear_ops: workload
                .full_qkv_projection_ops
                .saturating_add(workload.full_residual_ops)
                .saturating_add(workload.full_ffn_first_ops)
                .saturating_add(workload.full_ffn_second_ops),
            restricted_mul_ops: workload
                .full_qk_score_ops
                .saturating_add(workload.full_pv_ops)
                .saturating_add(workload.full_gelu_ops),
            nonlinear_rows: workload
                .full_softmax_rows
                .saturating_add(workload.full_layernorm_rows),
        },
    }
}
