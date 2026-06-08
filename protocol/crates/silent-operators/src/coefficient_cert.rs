use serde::{Deserialize, Serialize};

use crate::error::OperatorError;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShape {
    pub batch: usize,
    pub seq: usize,
    pub hidden: usize,
    pub heads: usize,
    pub ffn: Option<usize>,
}

impl TensorShape {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.batch == 0 || self.seq == 0 || self.hidden == 0 || self.heads == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY tensor shape dimensions must be positive",
            ));
        }
        if self.hidden % self.heads != 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY tensor hidden dimension must be divisible by heads",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayoutKind {
    Coefficient,
    CrtSlots,
    HeadWiseBlock,
    Diagonal,
    RowPacked,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layout {
    pub id: String,
    pub kind: LayoutKind,
    pub ring_n: usize,
    pub slot_count: usize,
    pub head_partition: Option<usize>,
    pub signed_centering: bool,
    pub basis: EncodingBasis,
}

impl Layout {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.id.is_empty() {
            return Err(OperatorError::InvalidParams(
                "HARMONY layout id must not be empty",
            ));
        }
        if !self.ring_n.is_power_of_two() || self.ring_n == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY ring degree must be a non-zero power of two",
            ));
        }
        if self.slot_count == 0 || self.slot_count > self.ring_n {
            return Err(OperatorError::InvalidParams(
                "HARMONY slot count must be in 1..=ring_n",
            ));
        }
        if matches!(self.kind, LayoutKind::HeadWiseBlock) && self.head_partition.unwrap_or(0) == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY head-wise layout requires a positive head partition",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingBasis {
    Coefficient,
    Crt,
    Ntt,
    Torus,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScaleInfo {
    pub input_scale: i128,
    pub output_scale: i128,
    pub rounding: RoundingMode,
    pub accumulated_rounding_error: f64,
}

impl ScaleInfo {
    pub fn unit() -> Self {
        Self {
            input_scale: 1,
            output_scale: 1,
            rounding: RoundingMode::None,
            accumulated_rounding_error: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundingMode {
    None,
    Nearest,
    Floor,
    Stochastic,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainCert {
    pub min: i128,
    pub max: i128,
    pub bitwidth: usize,
    pub clipping_range: Option<(i128, i128)>,
}

impl DomainCert {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.min > self.max {
            return Err(OperatorError::InvalidParams(
                "HARMONY domain minimum must not exceed maximum",
            ));
        }
        if self.bitwidth == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY domain bitwidth must be positive",
            ));
        }
        if let Some((min, max)) = self.clipping_range {
            if min > max {
                return Err(OperatorError::InvalidParams(
                    "HARMONY clipping range minimum must not exceed maximum",
                ));
            }
        }
        Ok(())
    }

    pub fn max_abs(&self) -> u128 {
        self.min.unsigned_abs().max(self.max.unsigned_abs())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoeffCert {
    pub layout: Layout,
    pub scale: ScaleInfo,
    pub semantic_range: DomainCert,
    pub coeff_bound: u128,
    pub secret_product_bound: Option<u128>,
    pub noise_bound: f64,
    pub modulus_p: u128,
    pub modulus_q_bits: usize,
    pub exact: bool,
    pub compatible_next_ops: Vec<String>,
}

impl CoeffCert {
    pub fn validate(&self) -> Result<(), OperatorError> {
        self.layout.validate()?;
        self.semantic_range.validate()?;
        if self.modulus_p < 2 {
            return Err(OperatorError::InvalidParams(
                "HARMONY coefficient certificate requires p >= 2",
            ));
        }
        if self.modulus_q_bits == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY coefficient certificate requires q bits > 0",
            ));
        }
        if !self.noise_bound.is_finite() || self.noise_bound < 0.0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY coefficient certificate noise bound must be finite and non-negative",
            ));
        }
        Ok(())
    }

    pub fn with_bound(&self, coeff_bound: u128, primitive_name: &str) -> Self {
        let mut out = self.clone();
        out.coeff_bound = coeff_bound;
        out.compatible_next_ops = vec![primitive_name.to_string()];
        out
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PrimitiveBound {
    Encode {
        range: DomainCert,
    },
    Add {
        fanin: usize,
    },
    Rotate {
        amount: isize,
    },
    PlainMul {
        l1_coeff_norm: u128,
        sparsity: usize,
    },
    RestrictedMul {
        rhs_bound: u128,
    },
    DiagonalMatmul {
        terms: usize,
        max_l1: u128,
    },
    ReductionTree {
        fanin: usize,
        levels: usize,
        rescale_each_level: bool,
    },
    Rescale {
        divisor: u128,
        rounding_error: f64,
    },
    Repack {
        operator_norm: u128,
    },
}

impl PrimitiveBound {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Encode { .. } => "encode",
            Self::Add { .. } => "add",
            Self::Rotate { .. } => "rotate",
            Self::PlainMul { .. } => "plain_mul",
            Self::RestrictedMul { .. } => "restricted_mul",
            Self::DiagonalMatmul { .. } => "diagonal_matmul",
            Self::ReductionTree { .. } => "reduction_tree",
            Self::Rescale { .. } => "rescale",
            Self::Repack { .. } => "repack",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoundTraceStep {
    pub primitive: PrimitiveBound,
    pub input_bounds: Vec<u128>,
    pub output_bound: u128,
    pub empirical_bound: Option<u128>,
    pub notes: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoundTrace {
    pub operator: String,
    pub layout_id: String,
    pub steps: Vec<BoundTraceStep>,
    pub final_cert: CoeffCert,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BksParams {
    pub security_lambda: usize,
    pub ring_n: usize,
    pub p: u128,
    pub q_bits: usize,
    pub secret_bound: u128,
    pub noise_bound: f64,
    pub max_add_fanin: usize,
}

impl BksParams {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if !self.ring_n.is_power_of_two() || self.ring_n == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY BKS ring degree must be a non-zero power of two",
            ));
        }
        if self.p < 2 || self.q_bits == 0 || self.secret_bound == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY BKS parameters require p >= 2, q bits > 0, secret bound > 0",
            ));
        }
        if self.max_add_fanin == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY BKS max add fan-in must be positive",
            ));
        }
        if !self.noise_bound.is_finite() || self.noise_bound < 0.0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY BKS noise bound must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissibilityReport {
    pub pass: bool,
    pub required_p_bits: usize,
    pub required_q_bits: usize,
    pub p_margin_bits: i64,
    pub q_margin_bits: i64,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CertComparisonReport {
    pub empirical_bound: u128,
    pub certified_bound: u128,
    pub naive_bound: u128,
    pub ratio_naive_certified: f64,
    pub ratio_certified_empirical: f64,
    pub certified_above_empirical: bool,
    pub structured_not_worse_than_naive: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicMatrixCert {
    pub layout_id: String,
    pub terms: usize,
    pub max_l1_coeff_norm: u128,
    pub sparsity: usize,
    pub empirical_bound: Option<u128>,
    pub naive_bound: u128,
}

impl PublicMatrixCert {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.layout_id.is_empty() {
            return Err(OperatorError::InvalidParams(
                "HARMONY public matrix certificate layout id must not be empty",
            ));
        }
        if self.terms == 0 || self.max_l1_coeff_norm == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY public matrix certificate requires positive terms and l1 norm",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatmulPlan {
    pub operator: String,
    pub layout: Layout,
    pub reduction: ReductionPlan,
}

impl MatmulPlan {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.operator.is_empty() {
            return Err(OperatorError::InvalidParams(
                "HARMONY matmul plan operator must not be empty",
            ));
        }
        self.layout.validate()?;
        self.reduction.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionPlan {
    pub operator: String,
    pub layout: Layout,
    pub shape: TensorShape,
    pub head_dim: usize,
    pub reduction: ReductionPlan,
}

impl AttentionPlan {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.operator.is_empty() {
            return Err(OperatorError::InvalidParams(
                "HARMONY attention plan operator must not be empty",
            ));
        }
        self.layout.validate()?;
        self.shape.validate()?;
        self.reduction.validate()?;
        if self.head_dim == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY attention plan requires a positive head dimension",
            ));
        }
        let expected = self.shape.hidden / self.shape.heads;
        if self.head_dim != expected {
            return Err(OperatorError::InvalidParams(
                "HARMONY attention head dimension must match hidden / heads",
            ));
        }
        ensure_reduction_capacity(self.head_dim, &self.reduction)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReductionPlan {
    pub fanin: usize,
    pub levels: usize,
    pub rescale_each_level: bool,
    pub rescale_divisor: u128,
}

impl ReductionPlan {
    pub fn flat(fanin: usize) -> Self {
        Self {
            fanin,
            levels: 1,
            rescale_each_level: false,
            rescale_divisor: 1,
        }
    }

    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.fanin == 0 || self.levels == 0 || self.rescale_divisor == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY reduction plan requires positive fan-in, levels, and rescale divisor",
            ));
        }
        Ok(())
    }
}

pub trait BoundTransformer {
    fn apply(&self, inputs: &[CoeffCert]) -> Result<(CoeffCert, BoundTraceStep), OperatorError>;
}

pub trait EmpiricalCoeffNorm {
    fn empirical_coeff_norm(&self) -> u128;
}

pub trait CertifiableOperator {
    type Plan;

    fn certify(&self, plan: &Self::Plan, inputs: &[CoeffCert])
    -> Result<BoundTrace, OperatorError>;
}

pub fn certify_matmul(
    lhs: &CoeffCert,
    rhs: &PublicMatrixCert,
    plan: &MatmulPlan,
    bks: &BksParams,
) -> Result<(CoeffCert, BoundTrace, AdmissibilityReport), OperatorError> {
    lhs.validate()?;
    rhs.validate()?;
    plan.validate()?;
    bks.validate()?;
    if lhs.layout.ring_n != bks.ring_n || plan.layout.ring_n != bks.ring_n {
        return Err(OperatorError::InvalidParams(
            "HARMONY matmul certificate ring degree mismatch",
        ));
    }

    let primitive = PrimitiveBound::DiagonalMatmul {
        terms: rhs.terms,
        max_l1: rhs.max_l1_coeff_norm,
    };
    let mut output_bound = apply_primitive_bound(&[lhs.coeff_bound], &primitive)?;
    let mut steps = vec![BoundTraceStep {
        primitive,
        input_bounds: vec![lhs.coeff_bound],
        output_bound,
        empirical_bound: rhs.empirical_bound,
        notes: "layout-aware public diagonal/plaintext matrix multiply".to_string(),
    }];

    if plan.reduction.rescale_each_level {
        let rescale = PrimitiveBound::Rescale {
            divisor: plan.reduction.rescale_divisor,
            rounding_error: 1.0,
        };
        let rescaled = apply_primitive_bound(&[output_bound], &rescale)?;
        steps.push(BoundTraceStep {
            primitive: rescale,
            input_bounds: vec![output_bound],
            output_bound: rescaled,
            empirical_bound: None,
            notes: "public rescale point recorded by matmul plan".to_string(),
        });
        output_bound = rescaled;
    }

    let mut out = lhs.with_bound(output_bound, &plan.operator);
    out.layout = plan.layout.clone();
    out.secret_product_bound = Some(saturating_mul(output_bound, bks.secret_bound));
    out.modulus_p = bks.p;
    out.modulus_q_bits = bks.q_bits;
    out.noise_bound += bks.noise_bound;

    let trace = BoundTrace {
        operator: plan.operator.clone(),
        layout_id: plan.layout.id.clone(),
        steps,
        final_cert: out.clone(),
    };
    let report = check_bks_admissibility(&out, bks);
    Ok((out, trace, report))
}

pub fn certify_attention_score(
    q: &CoeffCert,
    k: &CoeffCert,
    plan: &AttentionPlan,
    bks: &BksParams,
) -> Result<(CoeffCert, BoundTrace, AdmissibilityReport), OperatorError> {
    q.validate()?;
    k.validate()?;
    plan.validate()?;
    bks.validate()?;
    if q.layout.ring_n != bks.ring_n || k.layout.ring_n != bks.ring_n {
        return Err(OperatorError::InvalidParams(
            "HARMONY attention certificate ring degree mismatch",
        ));
    }
    if q.layout.id != k.layout.id {
        return Err(OperatorError::InvalidParams(
            "HARMONY attention score requires matching Q/K layout ids",
        ));
    }
    if q.scale.output_scale == 0 || k.scale.output_scale == 0 {
        return Err(OperatorError::InvalidParams(
            "HARMONY attention score scales must be non-zero",
        ));
    }

    let primitive = PrimitiveBound::RestrictedMul {
        rhs_bound: k.coeff_bound,
    };
    let product_bound = apply_primitive_bound(&[q.coeff_bound], &primitive)?;
    let mut steps = vec![BoundTraceStep {
        primitive,
        input_bounds: vec![q.coeff_bound, k.coeff_bound],
        output_bound: product_bound,
        empirical_bound: None,
        notes: "Q/K coefficient product bound before head-dimension reduction".to_string(),
    }];
    let (output_bound, mut reduction_steps) =
        reduction_steps_from_uniform_bound(product_bound, plan.head_dim, &plan.reduction)?;
    steps.append(&mut reduction_steps);

    let mut out = q.with_bound(output_bound, &plan.operator);
    out.layout = plan.layout.clone();
    out.scale.output_scale = q.scale.output_scale.saturating_mul(k.scale.output_scale);
    out.scale.accumulated_rounding_error += if plan.reduction.rescale_each_level {
        plan.reduction.levels as f64
    } else {
        0.0
    };
    out.secret_product_bound = Some(saturating_mul(output_bound, bks.secret_bound));
    out.modulus_p = bks.p;
    out.modulus_q_bits = bks.q_bits;
    out.noise_bound = q.noise_bound + k.noise_bound + bks.noise_bound;
    out.exact = q.exact && k.exact && !plan.reduction.rescale_each_level;
    out.compatible_next_ops = vec!["scale_mask".to_string(), "torus_softmax".to_string()];

    let trace = BoundTrace {
        operator: plan.operator.clone(),
        layout_id: plan.layout.id.clone(),
        steps,
        final_cert: out.clone(),
    };
    let report = check_bks_admissibility(&out, bks);
    Ok((out, trace, report))
}

pub fn certify_reduction_tree(
    inputs: &[CoeffCert],
    plan: &ReductionPlan,
) -> Result<(CoeffCert, BoundTrace), OperatorError> {
    plan.validate()?;
    if inputs.is_empty() {
        return Err(OperatorError::InvalidParams(
            "HARMONY reduction certification requires at least one input",
        ));
    }
    for input in inputs {
        input.validate()?;
    }
    let first = &inputs[0];
    if inputs.iter().any(|cert| cert.layout.id != first.layout.id) {
        return Err(OperatorError::InvalidParams(
            "HARMONY reduction inputs must use the same layout id",
        ));
    }
    ensure_reduction_capacity(inputs.len(), plan)?;

    let input_bounds = inputs
        .iter()
        .map(|cert| cert.coeff_bound)
        .collect::<Vec<_>>();
    let (output_bound, steps) = reduction_steps_from_bounds(&input_bounds, plan)?;
    let mut out = first.with_bound(output_bound, "reduction_tree");
    out.noise_bound = inputs
        .iter()
        .map(|cert| cert.noise_bound)
        .fold(0.0, f64::max);
    out.exact = inputs.iter().all(|cert| cert.exact) && !plan.rescale_each_level;

    let trace = BoundTrace {
        operator: "reduction_tree".to_string(),
        layout_id: first.layout.id.clone(),
        steps,
        final_cert: out.clone(),
    };
    Ok((out, trace))
}

pub fn check_bks_admissibility(cert: &CoeffCert, params: &BksParams) -> AdmissibilityReport {
    if let Err(err) = cert.validate().and_then(|_| params.validate()) {
        return AdmissibilityReport {
            pass: false,
            required_p_bits: 0,
            required_q_bits: 0,
            p_margin_bits: i64::MIN,
            q_margin_bits: i64::MIN,
            reason: err.to_string(),
        };
    }
    if cert.layout.ring_n != params.ring_n {
        return AdmissibilityReport {
            pass: false,
            required_p_bits: 0,
            required_q_bits: 0,
            p_margin_bits: i64::MIN,
            q_margin_bits: i64::MIN,
            reason: "ring degree mismatch".to_string(),
        };
    }

    let semantic_bound = cert.semantic_range.max_abs();
    let coeff_bound = cert.coeff_bound.max(semantic_bound).max(1);
    let secret_product = cert
        .secret_product_bound
        .unwrap_or_else(|| saturating_mul(coeff_bound, params.secret_bound))
        .max(1);
    let noise = ceil_f64_to_u128(cert.noise_bound.max(params.noise_bound)).max(1);

    let p_slack = ceil_f64_to_u128(cert.noise_bound.max(params.noise_bound) * 2.0).max(64);
    let required_p_value =
        saturating_add(saturating_add(saturating_mul(coeff_bound, 4), p_slack), 1);
    let required_p_bits = bit_length_u128(required_p_value);
    let q_margin = (params.security_lambda / 8).clamp(8, 32);
    let required_q_bits = bit_length_u128(params.p)
        .saturating_add(bit_length_u128(secret_product.max(noise)))
        .saturating_add(q_margin);

    let actual_p_bits = bit_length_u128(params.p);
    let p_margin_bits = actual_p_bits as i64 - required_p_bits as i64;
    let q_margin_bits = params.q_bits as i64 - required_q_bits as i64;
    let p_value_pass = params.p >= required_p_value;
    let q_bits_pass = q_margin_bits >= 0;
    let pass = p_value_pass && q_bits_pass;
    let reason = if pass {
        "BKS coefficient/no-wrap margins satisfied".to_string()
    } else if !p_value_pass && !q_bits_pass {
        "plaintext and ciphertext moduli are too small for certified coefficient bound".to_string()
    } else if !p_value_pass {
        "plaintext modulus is too small for certified coefficient bound".to_string()
    } else {
        "ciphertext modulus is too small for certified secret-product/noise bound".to_string()
    };

    AdmissibilityReport {
        pass,
        required_p_bits,
        required_q_bits,
        p_margin_bits,
        q_margin_bits,
        reason,
    }
}

pub fn compare_naive_structured_empirical(
    trace: &BoundTrace,
    empirical: u128,
    naive: u128,
) -> CertComparisonReport {
    let certified = trace.final_cert.coeff_bound;
    CertComparisonReport {
        empirical_bound: empirical,
        certified_bound: certified,
        naive_bound: naive,
        ratio_naive_certified: ratio(naive, certified),
        ratio_certified_empirical: ratio(certified, empirical),
        certified_above_empirical: certified >= empirical,
        structured_not_worse_than_naive: certified <= naive,
    }
}

pub fn apply_primitive_bound(
    input_bounds: &[u128],
    primitive: &PrimitiveBound,
) -> Result<u128, OperatorError> {
    match primitive {
        PrimitiveBound::Encode { range } => {
            range.validate()?;
            Ok(range.max_abs())
        }
        PrimitiveBound::Add { fanin } => {
            if *fanin == 0 {
                return Err(OperatorError::InvalidParams(
                    "HARMONY add primitive requires positive fan-in",
                ));
            }
            if input_bounds.is_empty() {
                return Ok(0);
            }
            Ok(input_bounds
                .iter()
                .take(*fanin)
                .fold(0u128, |acc, &bound| saturating_add(acc, bound)))
        }
        PrimitiveBound::Rotate { .. } => {
            input_bounds
                .first()
                .copied()
                .ok_or(OperatorError::InvalidParams(
                    "HARMONY rotate primitive requires one input bound",
                ))
        }
        PrimitiveBound::PlainMul { l1_coeff_norm, .. } => {
            let input = input_bounds
                .first()
                .copied()
                .ok_or(OperatorError::InvalidParams(
                    "HARMONY plaintext multiplication primitive requires one input bound",
                ))?;
            Ok(saturating_mul(input, *l1_coeff_norm))
        }
        PrimitiveBound::RestrictedMul { rhs_bound } => {
            let input = input_bounds
                .first()
                .copied()
                .ok_or(OperatorError::InvalidParams(
                    "HARMONY restricted multiplication primitive requires one input bound",
                ))?;
            Ok(saturating_mul(input, *rhs_bound))
        }
        PrimitiveBound::DiagonalMatmul { terms, max_l1 } => {
            if *terms == 0 {
                return Err(OperatorError::InvalidParams(
                    "HARMONY diagonal matmul primitive requires positive term count",
                ));
            }
            let input = input_bounds
                .first()
                .copied()
                .ok_or(OperatorError::InvalidParams(
                    "HARMONY diagonal matmul primitive requires one input bound",
                ))?;
            Ok(saturating_mul(
                saturating_mul(input, *max_l1),
                *terms as u128,
            ))
        }
        PrimitiveBound::ReductionTree {
            fanin,
            levels,
            rescale_each_level,
        } => {
            if *fanin == 0 || *levels == 0 {
                return Err(OperatorError::InvalidParams(
                    "HARMONY reduction tree primitive requires positive fan-in and levels",
                ));
            }
            let mut bound = if input_bounds.is_empty() {
                0
            } else {
                input_bounds
                    .iter()
                    .fold(0u128, |acc, &value| saturating_add(acc, value))
            };
            if input_bounds.is_empty() {
                bound = (*fanin as u128).saturating_pow(*levels as u32);
            }
            if *rescale_each_level {
                bound = bound.saturating_add(*levels as u128);
            }
            Ok(bound)
        }
        PrimitiveBound::Rescale {
            divisor,
            rounding_error,
        } => {
            if *divisor == 0 {
                return Err(OperatorError::InvalidParams(
                    "HARMONY rescale primitive requires non-zero divisor",
                ));
            }
            if !rounding_error.is_finite() || *rounding_error < 0.0 {
                return Err(OperatorError::InvalidParams(
                    "HARMONY rescale primitive requires finite non-negative rounding error",
                ));
            }
            let input = input_bounds
                .first()
                .copied()
                .ok_or(OperatorError::InvalidParams(
                    "HARMONY rescale primitive requires one input bound",
                ))?;
            Ok(ceil_div(input, *divisor).saturating_add(ceil_f64_to_u128(*rounding_error)))
        }
        PrimitiveBound::Repack { operator_norm } => {
            let input = input_bounds
                .first()
                .copied()
                .ok_or(OperatorError::InvalidParams(
                    "HARMONY repack primitive requires one input bound",
                ))?;
            Ok(saturating_mul(input, *operator_norm))
        }
    }
}

fn ensure_reduction_capacity(
    input_count: usize,
    plan: &ReductionPlan,
) -> Result<(), OperatorError> {
    if input_count == 0 {
        return Err(OperatorError::InvalidParams(
            "HARMONY reduction requires at least one input",
        ));
    }
    let capacity = (0..plan.levels)
        .try_fold(1usize, |acc, _| acc.checked_mul(plan.fanin))
        .unwrap_or(usize::MAX);
    if capacity < input_count {
        return Err(OperatorError::InvalidParams(
            "HARMONY reduction plan fan-in/levels cannot cover all inputs",
        ));
    }
    Ok(())
}

fn reduction_steps_from_uniform_bound(
    bound: u128,
    count: usize,
    plan: &ReductionPlan,
) -> Result<(u128, Vec<BoundTraceStep>), OperatorError> {
    reduction_steps_from_bounds(&vec![bound; count], plan)
}

fn reduction_steps_from_bounds(
    input_bounds: &[u128],
    plan: &ReductionPlan,
) -> Result<(u128, Vec<BoundTraceStep>), OperatorError> {
    ensure_reduction_capacity(input_bounds.len(), plan)?;
    let mut current = input_bounds.to_vec();
    let mut steps = Vec::new();

    for level in 0..plan.levels {
        if current.len() == 1 {
            break;
        }
        let mut next = Vec::with_capacity(ceil_div_usize(current.len(), plan.fanin));
        let mut level_output = 0u128;
        for chunk in current.chunks(plan.fanin) {
            let sum = chunk
                .iter()
                .fold(0u128, |acc, &bound| saturating_add(acc, bound));
            level_output = level_output.max(sum);
            next.push(sum);
        }
        steps.push(BoundTraceStep {
            primitive: PrimitiveBound::ReductionTree {
                fanin: plan.fanin.min(current.len()),
                levels: 1,
                rescale_each_level: false,
            },
            input_bounds: current.clone(),
            output_bound: level_output,
            empirical_bound: None,
            notes: format!(
                "reduction tree level {} combines {} groups",
                level + 1,
                next.len()
            ),
        });

        if plan.rescale_each_level {
            let mut rescaled = Vec::with_capacity(next.len());
            let mut rescaled_output = 0u128;
            for bound in next {
                let value = apply_primitive_bound(
                    &[bound],
                    &PrimitiveBound::Rescale {
                        divisor: plan.reduction_divisor_checked()?,
                        rounding_error: 1.0,
                    },
                )?;
                rescaled_output = rescaled_output.max(value);
                rescaled.push(value);
            }
            steps.push(BoundTraceStep {
                primitive: PrimitiveBound::Rescale {
                    divisor: plan.rescale_divisor,
                    rounding_error: 1.0,
                },
                input_bounds: vec![level_output],
                output_bound: rescaled_output,
                empirical_bound: None,
                notes: format!("public rescale after reduction level {}", level + 1),
            });
            current = rescaled;
        } else {
            current = next;
        }
    }

    let final_bound = current.first().copied().unwrap_or(0);
    Ok((final_bound, steps))
}

trait ReductionPlanChecked {
    fn reduction_divisor_checked(&self) -> Result<u128, OperatorError>;
}

impl ReductionPlanChecked for ReductionPlan {
    fn reduction_divisor_checked(&self) -> Result<u128, OperatorError> {
        if self.rescale_divisor == 0 {
            Err(OperatorError::InvalidParams(
                "HARMONY reduction rescale divisor must be non-zero",
            ))
        } else {
            Ok(self.rescale_divisor)
        }
    }
}

pub fn centered_abs_u64(value: u64, modulus: u64) -> u128 {
    if modulus == 0 {
        return value as u128;
    }
    let value = value % modulus;
    let neg = modulus - value;
    value.min(neg) as u128
}

pub fn empirical_coeff_norm_centered_u64(coeffs: &[u64], modulus: u64) -> u128 {
    coeffs
        .iter()
        .map(|&coeff| centered_abs_u64(coeff, modulus))
        .max()
        .unwrap_or(0)
}

pub fn rotate_coefficients_negacyclic(coeffs: &[i128], amount: isize) -> Vec<i128> {
    if coeffs.is_empty() {
        return Vec::new();
    }
    let n = coeffs.len() as isize;
    let shift = amount.rem_euclid(n);
    let mut out = vec![0i128; coeffs.len()];
    for (idx, &coeff) in coeffs.iter().enumerate() {
        let raw = idx as isize + shift;
        let wraps = raw.div_euclid(n);
        let target = raw.rem_euclid(n) as usize;
        out[target] = if wraps % 2 == 0 { coeff } else { -coeff };
    }
    out
}

fn bit_length_u128(value: u128) -> usize {
    if value == 0 {
        0
    } else {
        (u128::BITS - value.leading_zeros()) as usize
    }
}

fn ceil_f64_to_u128(value: f64) -> u128 {
    if value <= 0.0 {
        0
    } else if value >= u128::MAX as f64 {
        u128::MAX
    } else {
        value.ceil() as u128
    }
}

fn ceil_div(lhs: u128, rhs: u128) -> u128 {
    if lhs == 0 { 0 } else { ((lhs - 1) / rhs) + 1 }
}

fn ceil_div_usize(lhs: usize, rhs: usize) -> usize {
    if lhs == 0 { 0 } else { ((lhs - 1) / rhs) + 1 }
}

fn ratio(lhs: u128, rhs: u128) -> f64 {
    if rhs == 0 {
        f64::INFINITY
    } else {
        lhs as f64 / rhs as f64
    }
}

fn saturating_add(lhs: u128, rhs: u128) -> u128 {
    lhs.saturating_add(rhs)
}

fn saturating_mul(lhs: u128, rhs: u128) -> u128 {
    lhs.saturating_mul(rhs)
}
