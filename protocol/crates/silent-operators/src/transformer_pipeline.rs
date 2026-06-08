use serde::{Deserialize, Serialize};

use crate::direct_reload::ReloadMode;
use crate::error::OperatorError;
use crate::he_bridge::MaterializationPolicy;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformerShape {
    pub batch: usize,
    pub seq: usize,
    pub hidden: usize,
    pub heads: usize,
    pub ffn: usize,
}

impl TransformerShape {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.batch == 0 || self.seq == 0 || self.hidden == 0 || self.heads == 0 || self.ffn == 0
        {
            return Err(OperatorError::InvalidParams(
                "HARMONY transformer shape dimensions must be positive",
            ));
        }
        if self.hidden % self.heads != 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY transformer hidden dimension must be divisible by heads",
            ));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    pub fn token_count(&self) -> usize {
        self.batch.saturating_mul(self.seq)
    }

    pub fn attention_row_count(&self) -> usize {
        self.batch
            .saturating_mul(self.heads)
            .saturating_mul(self.seq)
    }

    pub fn attention_score_count(&self) -> usize {
        self.attention_row_count().saturating_mul(self.seq)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoftmaxMode {
    TorusClipped,
    ArithmeticPolynomial,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeluMode {
    AhssPolynomial,
    TorusLut,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub shape: TransformerShape,
    pub softmax: SoftmaxMode,
    pub gelu: GeluMode,
    pub materialization_policy: MaterializationPolicy,
    pub reload_mode: ReloadMode,
    pub bitwidth: usize,
}

impl PipelineConfig {
    pub fn validate(&self) -> Result<(), OperatorError> {
        self.shape.validate()?;
        if self.bitwidth == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY pipeline bitwidth must be positive",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineSampling {
    pub batch: usize,
    pub seq: usize,
    pub heads: usize,
    pub head_dim_tiles: usize,
    pub hidden_tiles: usize,
    pub ffn_tiles: usize,
}

impl PipelineSampling {
    pub fn validate(&self, shape: &TransformerShape) -> Result<(), OperatorError> {
        shape.validate()?;
        if self.batch == 0
            || self.seq == 0
            || self.heads == 0
            || self.head_dim_tiles == 0
            || self.hidden_tiles == 0
            || self.ffn_tiles == 0
        {
            return Err(OperatorError::InvalidParams(
                "HARMONY pipeline sampling dimensions must be positive",
            ));
        }
        if self.batch > shape.batch || self.seq > shape.seq || self.heads > shape.heads {
            return Err(OperatorError::InvalidParams(
                "HARMONY pipeline sampling cannot exceed the full transformer shape",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineWorkloadPlan {
    pub full_batch: usize,
    pub full_seq: usize,
    pub full_hidden: usize,
    pub full_heads: usize,
    pub full_head_dim: usize,
    pub full_ffn: usize,
    pub measured_batch: usize,
    pub measured_seq: usize,
    pub measured_heads: usize,
    pub full_head_dim_tiles: usize,
    pub full_hidden_tiles: usize,
    pub full_ffn_tiles: usize,
    pub measured_head_dim_tiles: usize,
    pub measured_hidden_tiles: usize,
    pub measured_ffn_tiles: usize,
    pub full_attention_rows: usize,
    pub measured_attention_rows: usize,
    pub full_qkv_projection_ops: usize,
    pub measured_qkv_projection_ops: usize,
    pub full_qk_score_ops: usize,
    pub measured_qk_score_ops: usize,
    pub full_softmax_rows: usize,
    pub measured_softmax_rows: usize,
    pub full_pv_ops: usize,
    pub measured_pv_ops: usize,
    pub full_residual_ops: usize,
    pub measured_residual_ops: usize,
    pub full_layernorm_rows: usize,
    pub measured_layernorm_rows: usize,
    pub full_ffn_first_ops: usize,
    pub measured_ffn_first_ops: usize,
    pub full_gelu_ops: usize,
    pub measured_gelu_ops: usize,
    pub full_ffn_second_ops: usize,
    pub measured_ffn_second_ops: usize,
}

impl PipelineWorkloadPlan {
    pub fn measured_total_ops(&self) -> usize {
        self.measured_qkv_projection_ops
            .saturating_add(self.measured_qk_score_ops)
            .saturating_add(self.measured_softmax_rows)
            .saturating_add(self.measured_pv_ops)
            .saturating_add(self.measured_residual_ops)
            .saturating_add(self.measured_layernorm_rows)
            .saturating_add(self.measured_ffn_first_ops)
            .saturating_add(self.measured_gelu_ops)
            .saturating_add(self.measured_ffn_second_ops)
    }

    pub fn full_total_ops(&self) -> usize {
        self.full_qkv_projection_ops
            .saturating_add(self.full_qk_score_ops)
            .saturating_add(self.full_softmax_rows)
            .saturating_add(self.full_pv_ops)
            .saturating_add(self.full_residual_ops)
            .saturating_add(self.full_layernorm_rows)
            .saturating_add(self.full_ffn_first_ops)
            .saturating_add(self.full_gelu_ops)
            .saturating_add(self.full_ffn_second_ops)
    }

    pub fn scale_factor(&self) -> f64 {
        let measured = self.measured_total_ops();
        if measured == 0 {
            0.0
        } else {
            self.full_total_ops() as f64 / measured as f64
        }
    }
}

pub fn plan_transformer_block(
    shape: &TransformerShape,
    sampling: &PipelineSampling,
    ring_n: usize,
) -> Result<PipelineWorkloadPlan, OperatorError> {
    shape.validate()?;
    sampling.validate(shape)?;
    if ring_n == 0 {
        return Err(OperatorError::InvalidParams(
            "HARMONY pipeline ring degree must be positive",
        ));
    }

    let head_dim = shape.head_dim();
    let full_head_dim_tiles = ceil_div_usize(head_dim, ring_n);
    let full_hidden_tiles = ceil_div_usize(shape.hidden, ring_n);
    let full_ffn_tiles = ceil_div_usize(shape.ffn, ring_n);
    let measured_head_dim_tiles = sampling.head_dim_tiles.min(full_head_dim_tiles).max(1);
    let measured_hidden_tiles = sampling.hidden_tiles.min(full_hidden_tiles).max(1);
    let measured_ffn_tiles = sampling.ffn_tiles.min(full_ffn_tiles).max(1);

    let full_tokens = shape.token_count();
    let measured_tokens = sampling.batch.saturating_mul(sampling.seq);
    let full_attention_rows = shape.attention_row_count();
    let measured_attention_rows = sampling
        .batch
        .saturating_mul(sampling.heads)
        .saturating_mul(sampling.seq);
    let full_layernorm_rows = full_tokens.saturating_mul(2);
    let measured_layernorm_rows = measured_tokens.saturating_mul(2);

    Ok(PipelineWorkloadPlan {
        full_batch: shape.batch,
        full_seq: shape.seq,
        full_hidden: shape.hidden,
        full_heads: shape.heads,
        full_head_dim: head_dim,
        full_ffn: shape.ffn,
        measured_batch: sampling.batch,
        measured_seq: sampling.seq,
        measured_heads: sampling.heads,
        full_head_dim_tiles,
        full_hidden_tiles,
        full_ffn_tiles,
        measured_head_dim_tiles,
        measured_hidden_tiles,
        measured_ffn_tiles,
        full_attention_rows,
        measured_attention_rows,
        full_qkv_projection_ops: full_tokens
            .saturating_mul(3)
            .saturating_mul(full_hidden_tiles),
        measured_qkv_projection_ops: measured_tokens
            .saturating_mul(3)
            .saturating_mul(measured_hidden_tiles),
        full_qk_score_ops: shape
            .attention_score_count()
            .saturating_mul(full_head_dim_tiles),
        measured_qk_score_ops: measured_attention_rows
            .saturating_mul(sampling.seq)
            .saturating_mul(measured_head_dim_tiles),
        full_softmax_rows: full_attention_rows,
        measured_softmax_rows: measured_attention_rows,
        full_pv_ops: full_attention_rows.saturating_mul(full_head_dim_tiles),
        measured_pv_ops: measured_attention_rows.saturating_mul(measured_head_dim_tiles),
        full_residual_ops: full_tokens
            .saturating_mul(2)
            .saturating_mul(full_hidden_tiles),
        measured_residual_ops: measured_tokens
            .saturating_mul(2)
            .saturating_mul(measured_hidden_tiles),
        full_layernorm_rows,
        measured_layernorm_rows,
        full_ffn_first_ops: full_tokens.saturating_mul(full_ffn_tiles),
        measured_ffn_first_ops: measured_tokens.saturating_mul(measured_ffn_tiles),
        full_gelu_ops: full_tokens.saturating_mul(full_ffn_tiles),
        measured_gelu_ops: measured_tokens.saturating_mul(measured_ffn_tiles),
        full_ffn_second_ops: full_tokens.saturating_mul(full_hidden_tiles),
        measured_ffn_second_ops: measured_tokens.saturating_mul(measured_hidden_tiles),
    })
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PipelineStats {
    pub total_latency_ms: f64,
    pub offline_latency_ms: f64,
    pub online_latency_ms: f64,
    pub ahss_time_ms: f64,
    pub torus_time_ms: f64,
    pub transition_time_ms: f64,
    pub direct_reload_time_ms: f64,
    pub communication_bytes: usize,
    pub peak_memory_bytes: usize,
    pub transitions: usize,
    pub full_materializations: usize,
    pub bmax: u128,
    pub required_p_bits: usize,
    pub required_q_bits: usize,
    pub max_abs_error: f64,
}

impl PipelineStats {
    pub fn recompute_totals(&mut self) {
        self.online_latency_ms = self.ahss_time_ms + self.torus_time_ms + self.transition_time_ms;
        self.total_latency_ms = self.offline_latency_ms + self.online_latency_ms;
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PipelineAccumulator {
    stats: PipelineStats,
}

impl PipelineAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_ahss_ms(&mut self, latency_ms: f64) {
        self.stats.ahss_time_ms += latency_ms;
        self.stats.recompute_totals();
    }

    pub fn add_torus_ms(&mut self, latency_ms: f64) {
        self.stats.torus_time_ms += latency_ms;
        self.stats.recompute_totals();
    }

    pub fn add_transition_ms(&mut self, latency_ms: f64) {
        self.stats.transition_time_ms += latency_ms;
        self.stats.transitions += 1;
        self.stats.recompute_totals();
    }

    pub fn add_reload_ms(&mut self, latency_ms: f64) {
        self.stats.direct_reload_time_ms += latency_ms;
        self.stats.transition_time_ms += latency_ms;
        self.stats.full_materializations += 1;
        self.stats.transitions += 1;
        self.stats.recompute_totals();
    }

    pub fn add_communication_bytes(&mut self, bytes: usize) {
        self.stats.communication_bytes += bytes;
    }

    pub fn observe_memory_bytes(&mut self, bytes: usize) {
        self.stats.peak_memory_bytes = self.stats.peak_memory_bytes.max(bytes);
    }

    pub fn observe_bmax(&mut self, bmax: u128) {
        self.stats.bmax = self.stats.bmax.max(bmax);
    }

    pub fn observe_required_bits(&mut self, p_bits: usize, q_bits: usize) {
        self.stats.required_p_bits = self.stats.required_p_bits.max(p_bits);
        self.stats.required_q_bits = self.stats.required_q_bits.max(q_bits);
    }

    pub fn observe_error(&mut self, max_abs_error: f64) {
        self.stats.max_abs_error = self.stats.max_abs_error.max(max_abs_error);
    }

    pub fn finish(mut self) -> PipelineStats {
        self.stats.recompute_totals();
        self.stats
    }
}

fn ceil_div_usize(lhs: usize, rhs: usize) -> usize {
    if lhs == 0 { 0 } else { ((lhs - 1) / rhs) + 1 }
}
