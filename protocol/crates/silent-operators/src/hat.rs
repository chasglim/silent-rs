use rand::{RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use silent_hss::{HssCiphertext, HssEvaluator, HssShare};
use silent_math::arith::{add_mod, mul_mod_u64 as mul_mod, sub_mod};
use silent_utils::rng::SecureRng;
use std::collections::HashMap;
use std::time::Instant;

use crate::ahss::{AhssFullState, AhssMemoryShare};
use crate::bks_ahss::BksAhssEngine;
use crate::coefficient_cert::{
    CoeffCert, DomainCert, EncodingBasis, Layout, LayoutKind, ScaleInfo,
};
use crate::error::OperatorError;
use crate::shares::AdditiveShares;

/// Public HAT parameter bundle for the primitive experiments.
///
/// The current HAT executor is an instruction-level experiment backend: it
/// executes the shifted RMS program with local arithmetic while preserving the
/// same instruction vocabulary that the BKS backend must implement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HatParams {
    pub bitwidth: u32,
    pub scale: u32,
    pub modulus: u64,
}

impl HatParams {
    pub fn new(bitwidth: u32, scale: u32) -> Result<Self, OperatorError> {
        if bitwidth == 0 || bitwidth >= 63 {
            return Err(OperatorError::InvalidParams(
                "HAT bitwidth must be in 1..63",
            ));
        }
        if scale > bitwidth {
            return Err(OperatorError::InvalidParams(
                "HAT fixed-point scale must not exceed bitwidth",
            ));
        }
        Ok(Self {
            bitwidth,
            scale,
            modulus: 1u64 << bitwidth,
        })
    }

    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.bitwidth == 0 || self.bitwidth >= 63 {
            return Err(OperatorError::InvalidParams(
                "HAT bitwidth must be in 1..63",
            ));
        }
        if self.modulus != (1u64 << self.bitwidth) {
            return Err(OperatorError::InvalidParams(
                "HAT modulus must equal 2^bitwidth",
            ));
        }
        if self.scale > self.bitwidth {
            return Err(OperatorError::InvalidParams(
                "HAT fixed-point scale must not exceed bitwidth",
            ));
        }
        Ok(())
    }

    pub fn element_bytes(self) -> usize {
        self.bitwidth.div_ceil(8) as usize
    }

    pub fn encode_signed(self, value: i128) -> u64 {
        reduce_i128_mod(value, self.modulus)
    }

    pub fn decode_signed(self, value: u64) -> i128 {
        decode_twos_complement(value % self.modulus, self.bitwidth)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HatOp {
    Cmp { threshold: i128 },
    Relu,
    MaxPool,
    Trunc { shift: u32, signed: bool },
    Poly { coeffs: Vec<i128> },
}

impl HatOp {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cmp { .. } => "cmp",
            Self::Relu => "relu",
            Self::MaxPool => "maxpool",
            Self::Trunc { .. } => "trunc",
            Self::Poly { .. } => "poly",
        }
    }

    pub fn requires_r_wire(&self) -> bool {
        matches!(self, Self::Relu | Self::MaxPool | Self::Poly { .. })
    }

    pub fn polynomial_degree(&self) -> usize {
        match self {
            Self::Poly { coeffs } => coeffs.len().saturating_sub(1),
            _ => 0,
        }
    }
}

/// A logical BKS input encoding slice used by the HAT token accounting.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BksInputEncoding {
    pub label: String,
    pub elements: usize,
    pub bytes: usize,
}

/// Per-token evaluation-key accounting. A real BKS backend can replace this
/// logical byte count with serialized key slices.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BksEvalKeySlice {
    pub label: String,
    pub bytes: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HatTokenAccounting {
    pub r_share_bytes: usize,
    pub hss_r_encoding_bytes: usize,
    pub hss_bits_encoding_bytes: usize,
    pub hss_low_limb_encoding_bytes: usize,
    pub hss_high_limb_encoding_bytes: usize,
    pub eval_key_slice_bytes: usize,
    pub metadata_bytes: usize,
}

impl HatTokenAccounting {
    pub fn total_bytes(&self) -> usize {
        self.r_share_bytes
            .saturating_add(self.hss_r_encoding_bytes)
            .saturating_add(self.hss_bits_encoding_bytes)
            .saturating_add(self.hss_low_limb_encoding_bytes)
            .saturating_add(self.hss_high_limb_encoding_bytes)
            .saturating_add(self.eval_key_slice_bytes)
            .saturating_add(self.metadata_bytes)
    }
}

#[derive(Clone, Debug)]
pub struct HatToken {
    pub id: u64,
    pub op: HatOp,
    pub params: HatParams,
    pub r_share: AdditiveShares,
    pub hss_r: Option<BksInputEncoding>,
    pub hss_bits: Option<Vec<BksInputEncoding>>,
    pub hss_low_limb: Option<Vec<BksInputEncoding>>,
    pub hss_high_limb: Option<BksInputEncoding>,
    pub eval_key: BksEvalKeySlice,
    pub consumed: bool,
    pub offline_ms: f64,
    pub accounting: HatTokenAccounting,
    mask_values: Vec<u64>,
}

impl HatToken {
    pub fn batch_size(&self) -> usize {
        self.mask_values.len()
    }

    pub fn token_bytes(&self) -> usize {
        self.accounting.total_bytes()
    }

    pub fn mask_values_for_testing(&self) -> &[u64] {
        &self.mask_values
    }

    fn ensure_fresh(&self) -> Result<(), OperatorError> {
        if self.consumed {
            return Err(OperatorError::Protocol("HAT token reused"));
        }
        Ok(())
    }

    fn consume(&mut self) {
        self.consumed = true;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RmsInputWire {
    R,
    Bit(usize),
    LowBit(usize),
    HighLimb,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RmsInstruction {
    Const(i128),
    Load(RmsInputWire),
    Lin {
        constant: i128,
        terms: Vec<(i128, usize)>,
    },
    MulIn {
        input: RmsInputWire,
        memory: usize,
    },
    /// This instruction is deliberately outside the HAT-native subset. It is
    /// present so audits can detect accidental memory-memory multiplication.
    MulMem {
        lhs: usize,
        rhs: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmsProgram {
    pub operator: String,
    pub batch_size: usize,
    pub bitwidth: u32,
    pub modulus: u64,
    pub instructions: Vec<RmsInstruction>,
    pub output: usize,
}

impl RmsProgram {
    pub fn new(
        operator: impl Into<String>,
        batch_size: usize,
        bitwidth: u32,
        modulus: u64,
    ) -> Self {
        Self {
            operator: operator.into(),
            batch_size,
            bitwidth,
            modulus,
            instructions: Vec::new(),
            output: 0,
        }
    }

    pub fn push(&mut self, instruction: RmsInstruction) -> usize {
        let id = self.instructions.len();
        self.instructions.push(instruction);
        self.output = id;
        id
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmsAudit {
    pub operator: String,
    pub batch_size: usize,
    pub bitwidth: u32,
    pub num_rms_const: usize,
    pub num_rms_load: usize,
    pub num_rms_linear: usize,
    pub num_rms_mulin: usize,
    pub num_memory_memory_mul: usize,
    pub num_online_he_ops: usize,
    pub num_online_beaver_mul: usize,
    pub num_online_flights: usize,
    pub rms_program_size: usize,
    pub hat_native: bool,
}

impl RmsAudit {
    pub fn validate_hat_native(&self) -> Result<(), OperatorError> {
        if !self.hat_native {
            return Err(OperatorError::Protocol("RMS program is not HAT-native"));
        }
        if self.num_memory_memory_mul != 0 {
            return Err(OperatorError::Protocol(
                "HAT RMS audit found memory-memory multiplication",
            ));
        }
        if self.num_online_he_ops != 0 || self.num_online_beaver_mul != 0 {
            return Err(OperatorError::Protocol(
                "HAT online audit found HE or Beaver work",
            ));
        }
        if self.num_online_flights != 1 {
            return Err(OperatorError::Protocol(
                "HAT online audit must have exactly one masked-opening flight",
            ));
        }
        Ok(())
    }
}

pub fn audit_rms_program(program: &RmsProgram) -> RmsAudit {
    let batch = program.batch_size.max(1);
    let mut audit = RmsAudit {
        operator: program.operator.clone(),
        batch_size: program.batch_size,
        bitwidth: program.bitwidth,
        num_online_flights: 1,
        ..RmsAudit::default()
    };
    for instruction in &program.instructions {
        match instruction {
            RmsInstruction::Const(_) => audit.num_rms_const += batch,
            RmsInstruction::Load(_) => audit.num_rms_load += batch,
            RmsInstruction::Lin { .. } => audit.num_rms_linear += batch,
            RmsInstruction::MulIn { .. } => audit.num_rms_mulin += batch,
            RmsInstruction::MulMem { .. } => audit.num_memory_memory_mul += batch,
        }
    }
    audit.rms_program_size = program.instructions.len().saturating_mul(batch);
    audit.hat_native = audit.num_memory_memory_mul == 0
        && audit.num_online_he_ops == 0
        && audit.num_online_beaver_mul == 0
        && audit.num_online_flights == 1;
    audit
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HatEvalReport {
    pub offline_prep_ms: f64,
    pub token_bytes: usize,
    pub online_opening_bytes: usize,
    pub online_eval_ms: f64,
    pub online_total_ms: f64,
    pub online_flights: usize,
    pub rms_mulin_count: usize,
    pub rms_program_size: usize,
    pub rms_audit: RmsAudit,
}

#[derive(Clone, Debug)]
pub struct HatEvalOutput {
    pub shares: AdditiveShares,
    pub report: HatEvalReport,
}

#[derive(Clone, Debug)]
pub struct HatRuntime {
    next_token_id: u64,
}

impl Default for HatRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl HatRuntime {
    pub fn new() -> Self {
        Self { next_token_id: 1 }
    }

    pub fn prep_token<R: RngCore + ?Sized>(
        &mut self,
        op: HatOp,
        batch_size: usize,
        params: HatParams,
        rng: &mut R,
    ) -> Result<HatToken, OperatorError> {
        params.validate()?;
        if batch_size == 0 {
            return Err(OperatorError::InvalidParams(
                "HAT token batch size must be positive",
            ));
        }
        if let HatOp::Trunc { shift, .. } = &op {
            if *shift == 0 || *shift > params.bitwidth {
                return Err(OperatorError::InvalidParams(
                    "HAT truncation shift must be in 1..=bitwidth",
                ));
            }
        }
        if let HatOp::Poly { coeffs } = &op {
            if coeffs.is_empty() {
                return Err(OperatorError::InvalidParams(
                    "HAT polynomial needs at least one coefficient",
                ));
            }
        }

        let start = Instant::now();
        let id = self.next_token_id;
        self.next_token_id = self.next_token_id.saturating_add(1);

        let mask_values = (0..batch_size)
            .map(|_| rng.next_u64() % params.modulus)
            .collect::<Vec<_>>();
        let r_share = AdditiveShares::share_with_rng(&mask_values, params.modulus, rng)?;
        let accounting = token_accounting(&op, params, batch_size);
        let hss_r = op.requires_r_wire().then(|| BksInputEncoding {
            label: "r".to_string(),
            elements: batch_size,
            bytes: accounting.hss_r_encoding_bytes,
        });
        let hss_bits = hss_bit_encodings(&op, params, batch_size);
        let hss_low_limb = hss_low_limb_encodings(&op, params, batch_size);
        let hss_high_limb = matches!(op, HatOp::Trunc { .. }).then(|| BksInputEncoding {
            label: "r_high_limb".to_string(),
            elements: batch_size,
            bytes: accounting.hss_high_limb_encoding_bytes,
        });

        Ok(HatToken {
            id,
            op,
            params,
            r_share,
            hss_r,
            hss_bits,
            hss_low_limb,
            hss_high_limb,
            eval_key: BksEvalKeySlice {
                label: "global_bks_eval_key_slice".to_string(),
                bytes: accounting.eval_key_slice_bytes,
            },
            consumed: false,
            offline_ms: elapsed_ms(start),
            accounting,
            mask_values,
        })
    }
}

#[derive(Clone, Debug)]
pub struct BksHatToken {
    pub base: HatToken,
    pub input_states: HashMap<RmsInputWire, AhssFullState<HssCiphertext, HssShare>>,
    pub protocol_offline_ms: f64,
}

#[derive(Clone, Debug)]
pub struct BksHatRuntime {
    inner: HatRuntime,
}

impl Default for BksHatRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl BksHatRuntime {
    pub fn new() -> Self {
        Self {
            inner: HatRuntime::new(),
        }
    }

    /// Prepares a real BKS-AHSS token for protocol smoke/calibration runs.
    ///
    /// This first implementation intentionally supports one packed element per
    /// token. The large CSV sweeps use the local RMS semantics executor; this
    /// protocol path verifies that every emitted nonlinear instruction can be
    /// realized as BKS `Load` plus `RestrMul`.
    pub fn prep_token<R: RngCore + ?Sized>(
        &mut self,
        engine: &BksAhssEngine,
        op: HatOp,
        params: HatParams,
        rng: &mut R,
    ) -> Result<BksHatToken, OperatorError> {
        params.validate()?;
        if engine.context().plain_modulus() != params.modulus {
            return Err(OperatorError::InvalidParams(
                "BKS-HAT engine plaintext modulus must equal HAT modulus",
            ));
        }
        let start = Instant::now();
        let mut base = self.inner.prep_token(op, 1, params, rng)?;
        let mut input_states = HashMap::new();
        for wire in required_input_wires(&base.op, params)? {
            let value = token_wire_value(&base.op, params, base.mask_values[0], wire)?;
            let cert = hat_coeff_cert(
                engine.context().degree(),
                params,
                1,
                coeff_bound(value, params),
            );
            let material = engine.input_material_from_coeffs(&[value])?;
            let state = engine.load_full(material, cert)?;
            input_states.insert(wire, state);
        }
        base.offline_ms = elapsed_ms(start);
        Ok(BksHatToken {
            base,
            input_states,
            protocol_offline_ms: elapsed_ms(start),
        })
    }
}

pub fn eval_bks_hat_cmp<R: RngCore + ?Sized>(
    engine: &BksAhssEngine,
    token: &mut BksHatToken,
    input: &AdditiveShares,
    rng: &mut R,
) -> Result<HatEvalOutput, OperatorError> {
    let threshold = match token.base.op {
        HatOp::Cmp { threshold } => threshold,
        _ => return Err(OperatorError::InvalidParams("BKS-HAT token is not for cmp")),
    };
    eval_bks_hat_unary(engine, token, input, rng, |params, delta| {
        compile_cmp_program(params, delta, threshold, 1)
    })
}

pub fn eval_bks_hat_relu<R: RngCore + ?Sized>(
    engine: &BksAhssEngine,
    token: &mut BksHatToken,
    input: &AdditiveShares,
    rng: &mut R,
) -> Result<HatEvalOutput, OperatorError> {
    if token.base.op != HatOp::Relu {
        return Err(OperatorError::InvalidParams(
            "BKS-HAT token is not for relu",
        ));
    }
    eval_bks_hat_unary(engine, token, input, rng, |params, delta| {
        compile_relu_program(params, delta, 1)
    })
}

pub fn eval_bks_hat_trunc<R: RngCore + ?Sized>(
    engine: &BksAhssEngine,
    token: &mut BksHatToken,
    input: &AdditiveShares,
    rng: &mut R,
) -> Result<HatEvalOutput, OperatorError> {
    let (shift, signed) = match token.base.op {
        HatOp::Trunc { shift, signed } => (shift, signed),
        _ => {
            return Err(OperatorError::InvalidParams(
                "BKS-HAT token is not for trunc",
            ));
        }
    };
    eval_bks_hat_unary(engine, token, input, rng, |params, delta| {
        compile_trunc_program(params, delta, shift, signed, 1)
    })
}

pub fn eval_bks_hat_poly<R: RngCore + ?Sized>(
    engine: &BksAhssEngine,
    token: &mut BksHatToken,
    input: &AdditiveShares,
    rng: &mut R,
) -> Result<HatEvalOutput, OperatorError> {
    let coeffs = match &token.base.op {
        HatOp::Poly { coeffs } => coeffs.clone(),
        _ => {
            return Err(OperatorError::InvalidParams(
                "BKS-HAT token is not for poly",
            ));
        }
    };
    eval_bks_hat_unary(engine, token, input, rng, move |params, delta| {
        compile_poly_program(params, delta, &coeffs, 1)
    })
}

pub fn eval_bks_hat_maxpool<R: RngCore + ?Sized>(
    engine: &BksAhssEngine,
    token: &mut BksHatToken,
    lhs: &AdditiveShares,
    rhs: &AdditiveShares,
    rng: &mut R,
) -> Result<HatEvalOutput, OperatorError> {
    if token.base.op != HatOp::MaxPool {
        return Err(OperatorError::InvalidParams(
            "BKS-HAT token is not for maxpool",
        ));
    }
    token.base.ensure_fresh()?;
    check_input_compatible(&token.base, lhs)?;
    check_input_compatible(&token.base, rhs)?;
    let online_start = Instant::now();
    let diff = lhs.sub(rhs)?;
    let delta = diff.sub(&token.base.r_share)?.reconstruct();
    let program = compile_relu_program(token.base.params, delta[0], 1)?;
    let shares = exec_bks_rms_program(engine, token, &program, rng)?;
    let selected_diff = bks_memory_pair_to_additive(engine, &shares, 1)?;
    let out = rhs.add(&selected_diff)?;
    let report = report_from_template(
        &token.base,
        &compile_relu_program(token.base.params, 0, 1)?,
        opening_bytes(token.base.params, 1),
        elapsed_ms(online_start),
    )?;
    token.base.consume();
    Ok(HatEvalOutput {
        shares: out,
        report,
    })
}

pub fn eval_hat_cmp<R: RngCore + ?Sized>(
    token: &mut HatToken,
    input: &AdditiveShares,
    rng: &mut R,
) -> Result<HatEvalOutput, OperatorError> {
    let threshold = match token.op {
        HatOp::Cmp { threshold } => threshold,
        _ => return Err(OperatorError::InvalidParams("HAT token is not for cmp")),
    };
    eval_hat_unary(token, input, rng, |params, delta| {
        compile_cmp_program(params, delta, threshold, 1)
    })
}

pub fn eval_hat_relu<R: RngCore + ?Sized>(
    token: &mut HatToken,
    input: &AdditiveShares,
    rng: &mut R,
) -> Result<HatEvalOutput, OperatorError> {
    if token.op != HatOp::Relu {
        return Err(OperatorError::InvalidParams("HAT token is not for relu"));
    }
    eval_hat_unary(token, input, rng, |params, delta| {
        compile_relu_program(params, delta, 1)
    })
}

pub fn eval_hat_trunc<R: RngCore + ?Sized>(
    token: &mut HatToken,
    input: &AdditiveShares,
    rng: &mut R,
) -> Result<HatEvalOutput, OperatorError> {
    let (shift, signed) = match token.op {
        HatOp::Trunc { shift, signed } => (shift, signed),
        _ => return Err(OperatorError::InvalidParams("HAT token is not for trunc")),
    };
    eval_hat_unary(token, input, rng, |params, delta| {
        compile_trunc_program(params, delta, shift, signed, 1)
    })
}

pub fn eval_hat_poly<R: RngCore + ?Sized>(
    token: &mut HatToken,
    input: &AdditiveShares,
    rng: &mut R,
) -> Result<HatEvalOutput, OperatorError> {
    let coeffs = match &token.op {
        HatOp::Poly { coeffs } => coeffs.clone(),
        _ => return Err(OperatorError::InvalidParams("HAT token is not for poly")),
    };
    eval_hat_unary(token, input, rng, move |params, delta| {
        compile_poly_program(params, delta, &coeffs, 1)
    })
}

pub fn eval_hat_maxpool<R: RngCore + ?Sized>(
    token: &mut HatToken,
    lhs: &AdditiveShares,
    rhs: &AdditiveShares,
    rng: &mut R,
) -> Result<HatEvalOutput, OperatorError> {
    if token.op != HatOp::MaxPool {
        return Err(OperatorError::InvalidParams("HAT token is not for maxpool"));
    }
    token.ensure_fresh()?;
    check_input_compatible(token, lhs)?;
    check_input_compatible(token, rhs)?;
    if lhs.len() != rhs.len() {
        return Err(OperatorError::InvalidParams(
            "HAT MaxPool inputs must have equal length",
        ));
    }
    let online_start = Instant::now();
    let diff = lhs.sub(rhs)?;
    let delta = diff.sub(&token.r_share)?.reconstruct();
    let opening_bytes = opening_bytes(token.params, token.batch_size());
    let mut selected_diff = Vec::with_capacity(delta.len());
    for (&r, &d) in token.mask_values.iter().zip(delta.iter()) {
        let program = compile_relu_program(token.params, d, 1)?;
        selected_diff.push(exec_rms_program(
            &program,
            &RmsInputs::from_mask(token.params, r, token_shift(&token.op), d),
        )?);
    }
    let selected_diff_shares =
        AdditiveShares::share_with_rng(&selected_diff, token.params.modulus, rng)?;
    let shares = rhs.add(&selected_diff_shares)?;
    let template = compile_relu_program(token.params, 0, token.batch_size())?;
    let report = report_from_template(token, &template, opening_bytes, elapsed_ms(online_start))?;
    token.consume();
    Ok(HatEvalOutput { shares, report })
}

pub fn reference_cmp(values: &[u64], params: HatParams, threshold: i128) -> Vec<u64> {
    values
        .iter()
        .map(|&value| u64::from(params.decode_signed(value) >= threshold))
        .collect()
}

pub fn reference_relu(values: &[u64], params: HatParams) -> Vec<u64> {
    values
        .iter()
        .map(|&value| params.encode_signed(params.decode_signed(value).max(0)))
        .collect()
}

pub fn reference_maxpool(lhs: &[u64], rhs: &[u64], params: HatParams) -> Vec<u64> {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&a, &b)| {
            let a_signed = params.decode_signed(a);
            let b_signed = params.decode_signed(b);
            params.encode_signed(a_signed.max(b_signed))
        })
        .collect()
}

pub fn reference_trunc(values: &[u64], params: HatParams, shift: u32, signed: bool) -> Vec<u64> {
    values
        .iter()
        .map(|&value| {
            if signed {
                params.encode_signed(floor_div_pow2(params.decode_signed(value), shift))
            } else {
                (value % params.modulus) >> shift
            }
        })
        .collect()
}

pub fn reference_poly(values: &[u64], params: HatParams, coeffs: &[i128]) -> Vec<u64> {
    values
        .iter()
        .map(|&value| eval_poly_mod(value % params.modulus, coeffs, params.modulus))
        .collect()
}

pub fn compile_cmp_program(
    params: HatParams,
    delta: u64,
    threshold: i128,
    batch_size: usize,
) -> Result<RmsProgram, OperatorError> {
    params.validate()?;
    let gamma = reduce_i128_mod(delta as i128 - threshold, params.modulus);
    let mut program = RmsProgram::new("cmp", batch_size, params.bitwidth, params.modulus);
    let bits = emit_bit_add_const(
        &mut program,
        params.bitwidth as usize,
        gamma,
        RmsInputWire::Bit,
    );
    let msb = bits.sum_bits[params.bitwidth as usize - 1];
    let beta = program.push(RmsInstruction::Lin {
        constant: 1,
        terms: vec![(-1, msb)],
    });
    program.output = beta;
    Ok(program)
}

pub fn compile_relu_program(
    params: HatParams,
    delta: u64,
    batch_size: usize,
) -> Result<RmsProgram, OperatorError> {
    params.validate()?;
    let mut program = compile_cmp_program(params, delta, 0, batch_size)?;
    program.operator = "relu".to_string();
    let beta = program.output;
    let rho = program.push(RmsInstruction::MulIn {
        input: RmsInputWire::R,
        memory: beta,
    });
    let out = program.push(RmsInstruction::Lin {
        constant: 0,
        terms: vec![(1, rho), (delta as i128, beta)],
    });
    program.output = out;
    Ok(program)
}

pub fn compile_trunc_program(
    params: HatParams,
    delta: u64,
    shift: u32,
    signed: bool,
    batch_size: usize,
) -> Result<RmsProgram, OperatorError> {
    params.validate()?;
    if shift == 0 || shift > params.bitwidth {
        return Err(OperatorError::InvalidParams(
            "HAT truncation shift must be in 1..=bitwidth",
        ));
    }
    let mut program = RmsProgram::new("trunc", batch_size, params.bitwidth, params.modulus);
    let add_bits = if signed {
        emit_bit_add_const(
            &mut program,
            params.bitwidth as usize,
            delta,
            RmsInputWire::Bit,
        )
    } else {
        let low_mask = (1u64 << shift) - 1;
        emit_bit_add_const(
            &mut program,
            shift as usize,
            delta & low_mask,
            RmsInputWire::LowBit,
        )
    };
    let carry_low = add_bits.carries[shift as usize];
    let high_limb = program.push(RmsInstruction::Load(RmsInputWire::HighLimb));
    let delta_hi = (delta >> shift) as i128;
    let mut terms = vec![(1, high_limb), (1, carry_low)];
    if signed {
        let sign = add_bits.sum_bits[params.bitwidth as usize - 1];
        let carry_out = add_bits.carries[params.bitwidth as usize];
        let high_range = 1u64 << (params.bitwidth - shift);
        let sign_ext = params.modulus - high_range;
        terms.push((sign_ext as i128, sign));
        terms.push((sign_ext as i128, carry_out));
    }
    let out = program.push(RmsInstruction::Lin {
        constant: delta_hi,
        terms,
    });
    program.output = out;
    Ok(program)
}

pub fn compile_poly_program(
    params: HatParams,
    delta: u64,
    coeffs: &[i128],
    batch_size: usize,
) -> Result<RmsProgram, OperatorError> {
    params.validate()?;
    if coeffs.is_empty() {
        return Err(OperatorError::InvalidParams(
            "HAT polynomial needs at least one coefficient",
        ));
    }
    let shifted = shifted_coefficients(coeffs, delta, params.modulus)?;
    let mut program = RmsProgram::new("poly", batch_size, params.bitwidth, params.modulus);
    let mut acc = program.push(RmsInstruction::Const(shifted[shifted.len() - 1] as i128));
    for &coeff in shifted[..shifted.len() - 1].iter().rev() {
        let prod = program.push(RmsInstruction::MulIn {
            input: RmsInputWire::R,
            memory: acc,
        });
        acc = program.push(RmsInstruction::Lin {
            constant: coeff as i128,
            terms: vec![(1, prod)],
        });
    }
    program.output = acc;
    Ok(program)
}

#[derive(Clone, Debug)]
struct BitAddResult {
    sum_bits: Vec<usize>,
    carries: Vec<usize>,
}

fn emit_bit_add_const<F>(
    program: &mut RmsProgram,
    bits: usize,
    gamma: u64,
    wire_for_bit: F,
) -> BitAddResult
where
    F: Fn(usize) -> RmsInputWire,
{
    let mut sum_bits = Vec::with_capacity(bits);
    let mut carries = Vec::with_capacity(bits + 1);
    let mut carry = program.push(RmsInstruction::Const(0));
    carries.push(carry);
    for i in 0..bits {
        let wire = wire_for_bit(i);
        let bit = program.push(RmsInstruction::Load(wire));
        let prod = program.push(RmsInstruction::MulIn {
            input: wire,
            memory: carry,
        });
        let gamma_i = (gamma >> i) & 1;
        let next_carry = if gamma_i == 0 {
            program.push(RmsInstruction::Lin {
                constant: 0,
                terms: vec![(1, prod)],
            })
        } else {
            program.push(RmsInstruction::Lin {
                constant: 0,
                terms: vec![(1, bit), (1, carry), (-1, prod)],
            })
        };
        let sum = if gamma_i == 0 {
            program.push(RmsInstruction::Lin {
                constant: 0,
                terms: vec![(1, bit), (1, carry), (-2, prod)],
            })
        } else {
            program.push(RmsInstruction::Lin {
                constant: 1,
                terms: vec![(-1, bit), (-1, carry), (2, prod)],
            })
        };
        carry = next_carry;
        carries.push(carry);
        sum_bits.push(sum);
    }
    BitAddResult { sum_bits, carries }
}

#[derive(Clone, Debug)]
struct RmsInputs {
    r: u64,
    bits: Vec<u64>,
    low_bits: Vec<u64>,
    high_limb: u64,
}

impl RmsInputs {
    fn from_mask(params: HatParams, r: u64, trunc_shift: Option<u32>, _delta: u64) -> Self {
        let bits = (0..params.bitwidth)
            .map(|i| (r >> i) & 1)
            .collect::<Vec<_>>();
        let shift = trunc_shift.unwrap_or(params.scale).min(params.bitwidth);
        let low_bits = (0..shift).map(|i| (r >> i) & 1).collect::<Vec<_>>();
        let high_limb = if shift >= params.bitwidth {
            0
        } else {
            r >> shift
        };
        Self {
            r,
            bits,
            low_bits,
            high_limb,
        }
    }

    fn value(&self, wire: RmsInputWire) -> Result<u64, OperatorError> {
        match wire {
            RmsInputWire::R => Ok(self.r),
            RmsInputWire::Bit(i) => self
                .bits
                .get(i)
                .copied()
                .ok_or(OperatorError::Protocol("missing HAT bit input wire")),
            RmsInputWire::LowBit(i) => self
                .low_bits
                .get(i)
                .copied()
                .ok_or(OperatorError::Protocol("missing HAT low-limb input wire")),
            RmsInputWire::HighLimb => Ok(self.high_limb),
        }
    }
}

fn exec_rms_program(program: &RmsProgram, inputs: &RmsInputs) -> Result<u64, OperatorError> {
    let mut memory = Vec::with_capacity(program.instructions.len());
    for instruction in &program.instructions {
        let value = match instruction {
            RmsInstruction::Const(value) => reduce_i128_mod(*value, program.modulus),
            RmsInstruction::Load(wire) => inputs.value(*wire)? % program.modulus,
            RmsInstruction::Lin { constant, terms } => {
                let mut acc = reduce_i128_mod(*constant, program.modulus);
                for &(coeff, id) in terms {
                    let term = mul_mod(
                        reduce_i128_mod(coeff, program.modulus),
                        *memory
                            .get(id)
                            .ok_or(OperatorError::Protocol("bad RMS memory reference"))?,
                        program.modulus,
                    );
                    acc = add_mod(acc, term, program.modulus);
                }
                acc
            }
            RmsInstruction::MulIn { input, memory: id } => {
                let lhs = inputs.value(*input)? % program.modulus;
                let rhs = *memory
                    .get(*id)
                    .ok_or(OperatorError::Protocol("bad RMS memory reference"))?;
                mul_mod(lhs, rhs, program.modulus)
            }
            RmsInstruction::MulMem { lhs, rhs } => {
                let lhs = *memory
                    .get(*lhs)
                    .ok_or(OperatorError::Protocol("bad RMS memory reference"))?;
                let rhs = *memory
                    .get(*rhs)
                    .ok_or(OperatorError::Protocol("bad RMS memory reference"))?;
                mul_mod(lhs, rhs, program.modulus)
            }
        };
        memory.push(value);
    }
    memory
        .get(program.output)
        .copied()
        .ok_or(OperatorError::Protocol("bad RMS output reference"))
}

fn eval_hat_unary<R, F>(
    token: &mut HatToken,
    input: &AdditiveShares,
    rng: &mut R,
    compile: F,
) -> Result<HatEvalOutput, OperatorError>
where
    R: RngCore + ?Sized,
    F: Fn(HatParams, u64) -> Result<RmsProgram, OperatorError>,
{
    token.ensure_fresh()?;
    check_input_compatible(token, input)?;
    let online_start = Instant::now();
    let delta = input.sub(&token.r_share)?.reconstruct();
    let opening_bytes = opening_bytes(token.params, token.batch_size());
    let mut values = Vec::with_capacity(delta.len());
    for (&r, &d) in token.mask_values.iter().zip(delta.iter()) {
        let program = compile(token.params, d)?;
        let inputs = RmsInputs::from_mask(token.params, r, token_shift(&token.op), d);
        values.push(exec_rms_program(&program, &inputs)?);
    }
    let shares = AdditiveShares::share_with_rng(&values, token.params.modulus, rng)?;
    let template = compile(token.params, 0)?.with_batch_size(token.batch_size());
    let report = report_from_template(token, &template, opening_bytes, elapsed_ms(online_start))?;
    token.consume();
    Ok(HatEvalOutput { shares, report })
}

fn eval_bks_hat_unary<R, F>(
    engine: &BksAhssEngine,
    token: &mut BksHatToken,
    input: &AdditiveShares,
    _rng: &mut R,
    compile: F,
) -> Result<HatEvalOutput, OperatorError>
where
    R: RngCore + ?Sized,
    F: Fn(HatParams, u64) -> Result<RmsProgram, OperatorError>,
{
    token.base.ensure_fresh()?;
    check_input_compatible(&token.base, input)?;
    let online_start = Instant::now();
    let delta = input.sub(&token.base.r_share)?.reconstruct();
    let program = compile(token.base.params, delta[0])?;
    let shares = exec_bks_rms_program(engine, token, &program, _rng)?;
    let out = bks_memory_pair_to_additive(engine, &shares, 1)?;
    let template = compile(token.base.params, 0)?.with_batch_size(1);
    let report = report_from_template(
        &token.base,
        &template,
        opening_bytes(token.base.params, 1),
        elapsed_ms(online_start),
    )?;
    token.base.consume();
    Ok(HatEvalOutput {
        shares: out,
        report,
    })
}

fn exec_bks_rms_program<R: RngCore + ?Sized>(
    engine: &BksAhssEngine,
    token: &BksHatToken,
    program: &RmsProgram,
    rng: &mut R,
) -> Result<[AhssMemoryShare<HssShare>; 2], OperatorError> {
    let mut memory: Vec<[AhssMemoryShare<HssShare>; 2]> =
        Vec::with_capacity(program.instructions.len());
    for instruction in &program.instructions {
        let value = match instruction {
            RmsInstruction::Const(value) => {
                bks_public_memory(engine, &[reduce_i128_mod(*value, program.modulus)], rng)?
            }
            RmsInstruction::Load(wire) => token
                .input_states
                .get(wire)
                .ok_or(OperatorError::Protocol("missing BKS-HAT input wire"))?
                .memory_shares
                .clone(),
            RmsInstruction::Lin { constant, terms } => {
                let mut acc = reduce_i128_mod(*constant, program.modulus);
                for &(coeff, id) in terms {
                    let value = engine
                        .reconstruct_coeffs(
                            memory
                                .get(id)
                                .ok_or(OperatorError::Protocol("bad RMS memory reference"))?,
                        )?
                        .first()
                        .copied()
                        .ok_or(OperatorError::Protocol("empty BKS-HAT memory value"))?;
                    let term = mul_mod(
                        reduce_i128_mod(coeff, program.modulus),
                        value % program.modulus,
                        program.modulus,
                    );
                    acc = add_mod(acc, term, program.modulus);
                }
                bks_public_memory(engine, &[acc], rng)?
            }
            RmsInstruction::MulIn { input, memory: id } => {
                let input_state = token
                    .input_states
                    .get(input)
                    .ok_or(OperatorError::Protocol("missing BKS-HAT input wire"))?;
                let memory_pair = memory
                    .get(*id)
                    .ok_or(OperatorError::Protocol("bad RMS memory reference"))?;
                engine
                    .restricted_mul_pair_exact(input_state, memory_pair)?
                    .0
            }
            RmsInstruction::MulMem { .. } => {
                return Err(OperatorError::Protocol(
                    "BKS-HAT executor rejects memory-memory multiplication",
                ));
            }
        };
        memory.push(value);
    }
    memory
        .get(program.output)
        .cloned()
        .ok_or(OperatorError::Protocol("bad RMS output reference"))
}

fn report_from_template(
    token: &HatToken,
    template: &RmsProgram,
    opening_bytes_value: usize,
    online_ms: f64,
) -> Result<HatEvalReport, OperatorError> {
    let audit = audit_rms_program(template);
    audit.validate_hat_native()?;
    Ok(HatEvalReport {
        offline_prep_ms: token.offline_ms,
        token_bytes: token.token_bytes(),
        online_opening_bytes: opening_bytes_value,
        online_eval_ms: online_ms,
        online_total_ms: online_ms,
        online_flights: audit.num_online_flights,
        rms_mulin_count: audit.num_rms_mulin,
        rms_program_size: audit.rms_program_size,
        rms_audit: audit,
    })
}

fn bks_public_memory<R: RngCore + ?Sized>(
    engine: &BksAhssEngine,
    coeffs: &[u64],
    rng: &mut R,
) -> Result<[AhssMemoryShare<HssShare>; 2], OperatorError> {
    let mut secure_rng = derive_secure_rng(rng);
    let (party0, party1) = HssEvaluator::share_plain_coeffs_exact(
        engine.context(),
        coeffs,
        &engine.eval_key(0)?.share,
        &engine.eval_key(1)?.share,
        &mut secure_rng,
    )?;
    Ok([
        AhssMemoryShare {
            party_id: 0,
            value_times_secret_share: party0,
        },
        AhssMemoryShare {
            party_id: 1,
            value_times_secret_share: party1,
        },
    ])
}

fn bks_memory_pair_to_additive(
    engine: &BksAhssEngine,
    shares: &[AhssMemoryShare<HssShare>; 2],
    len: usize,
) -> Result<AdditiveShares, OperatorError> {
    let values = engine
        .reconstruct_coeffs(shares)?
        .into_iter()
        .take(len)
        .collect::<Vec<_>>();
    AdditiveShares::share_public(&values, engine.context().plain_modulus())
}

fn check_input_compatible(token: &HatToken, input: &AdditiveShares) -> Result<(), OperatorError> {
    if input.modulus() != token.params.modulus {
        return Err(OperatorError::InvalidParams(
            "HAT input share modulus must match token modulus",
        ));
    }
    if input.len() != token.batch_size() {
        return Err(OperatorError::InvalidParams(
            "HAT input length must match token batch size",
        ));
    }
    Ok(())
}

fn token_shift(op: &HatOp) -> Option<u32> {
    match op {
        HatOp::Trunc { shift, .. } => Some(*shift),
        _ => None,
    }
}

fn required_input_wires(op: &HatOp, params: HatParams) -> Result<Vec<RmsInputWire>, OperatorError> {
    let mut wires = Vec::new();
    match op {
        HatOp::Cmp { .. } => {
            for i in 0..params.bitwidth as usize {
                wires.push(RmsInputWire::Bit(i));
            }
        }
        HatOp::Relu | HatOp::MaxPool => {
            wires.push(RmsInputWire::R);
            for i in 0..params.bitwidth as usize {
                wires.push(RmsInputWire::Bit(i));
            }
        }
        HatOp::Trunc { shift, signed } => {
            if *shift == 0 || *shift > params.bitwidth {
                return Err(OperatorError::InvalidParams(
                    "HAT truncation shift must be in 1..=bitwidth",
                ));
            }
            if *signed {
                for i in 0..params.bitwidth as usize {
                    wires.push(RmsInputWire::Bit(i));
                }
            } else {
                for i in 0..*shift as usize {
                    wires.push(RmsInputWire::LowBit(i));
                }
            }
            wires.push(RmsInputWire::HighLimb);
        }
        HatOp::Poly { .. } => wires.push(RmsInputWire::R),
    }
    Ok(wires)
}

fn token_wire_value(
    op: &HatOp,
    params: HatParams,
    r: u64,
    wire: RmsInputWire,
) -> Result<u64, OperatorError> {
    Ok(match wire {
        RmsInputWire::R => r % params.modulus,
        RmsInputWire::Bit(i) => {
            if i >= params.bitwidth as usize {
                return Err(OperatorError::Protocol("HAT bit wire out of range"));
            }
            (r >> i) & 1
        }
        RmsInputWire::LowBit(i) => {
            let shift = token_shift(op).unwrap_or(params.scale);
            if i >= shift as usize {
                return Err(OperatorError::Protocol("HAT low-limb wire out of range"));
            }
            (r >> i) & 1
        }
        RmsInputWire::HighLimb => {
            let shift = token_shift(op).unwrap_or(params.scale);
            if shift >= params.bitwidth {
                0
            } else {
                r >> shift
            }
        }
    })
}

fn coeff_bound(value: u64, params: HatParams) -> u128 {
    let value = value % params.modulus;
    value.min(params.modulus - value) as u128
}

fn hat_coeff_cert(ring_n: usize, params: HatParams, batch: usize, coeff_bound: u128) -> CoeffCert {
    CoeffCert {
        layout: Layout {
            id: "hat-bks-coeff".to_string(),
            kind: LayoutKind::Coefficient,
            ring_n,
            slot_count: batch.max(1).min(ring_n),
            head_partition: None,
            signed_centering: true,
            basis: EncodingBasis::Coefficient,
        },
        scale: ScaleInfo::unit(),
        semantic_range: DomainCert {
            min: -(1i128 << (params.bitwidth - 1)),
            max: (1i128 << (params.bitwidth - 1)) - 1,
            bitwidth: params.bitwidth as usize,
            clipping_range: None,
        },
        coeff_bound,
        secret_product_bound: None,
        noise_bound: 0.0,
        modulus_p: params.modulus as u128,
        modulus_q_bits: 128,
        exact: true,
        compatible_next_ops: vec!["restricted_mul".to_string()],
    }
}

fn token_accounting(op: &HatOp, params: HatParams, batch_size: usize) -> HatTokenAccounting {
    let elem = params.element_bytes();
    let mut accounting = HatTokenAccounting {
        r_share_bytes: 2usize.saturating_mul(batch_size).saturating_mul(elem),
        metadata_bytes: 32,
        ..HatTokenAccounting::default()
    };
    if op.requires_r_wire() {
        accounting.hss_r_encoding_bytes = batch_size.saturating_mul(elem);
    }
    match op {
        HatOp::Cmp { .. } | HatOp::Relu | HatOp::MaxPool => {
            accounting.hss_bits_encoding_bytes =
                batch_size.saturating_mul(params.bitwidth as usize);
        }
        HatOp::Trunc { shift, signed } => {
            if *signed {
                accounting.hss_bits_encoding_bytes =
                    batch_size.saturating_mul((params.bitwidth - *shift) as usize);
            }
            accounting.hss_low_limb_encoding_bytes = batch_size.saturating_mul(*shift as usize);
            accounting.hss_high_limb_encoding_bytes = batch_size.saturating_mul(elem);
        }
        HatOp::Poly { .. } => {}
    }
    accounting
}

fn hss_bit_encodings(
    op: &HatOp,
    params: HatParams,
    batch_size: usize,
) -> Option<Vec<BksInputEncoding>> {
    let count = match op {
        HatOp::Cmp { .. } | HatOp::Relu | HatOp::MaxPool => params.bitwidth as usize,
        HatOp::Trunc { shift, signed } if *signed => (params.bitwidth - *shift) as usize,
        _ => 0,
    };
    (count > 0).then(|| {
        (0..count)
            .map(|i| BksInputEncoding {
                label: format!("r_bit_{i}"),
                elements: batch_size,
                bytes: batch_size,
            })
            .collect()
    })
}

fn hss_low_limb_encodings(
    op: &HatOp,
    _params: HatParams,
    batch_size: usize,
) -> Option<Vec<BksInputEncoding>> {
    let count = match op {
        HatOp::Trunc { shift, .. } => *shift as usize,
        _ => 0,
    };
    (count > 0).then(|| {
        (0..count)
            .map(|i| BksInputEncoding {
                label: format!("r_low_limb_bit_{i}"),
                elements: batch_size,
                bytes: batch_size,
            })
            .collect()
    })
}

fn opening_bytes(params: HatParams, batch_size: usize) -> usize {
    2usize
        .saturating_mul(batch_size)
        .saturating_mul(params.element_bytes())
}

fn shifted_coefficients(
    coeffs: &[i128],
    delta: u64,
    modulus: u64,
) -> Result<Vec<u64>, OperatorError> {
    let degree = coeffs.len() - 1;
    let mut shifted = vec![0u64; coeffs.len()];
    for i in 0..=degree {
        let mut acc = 0u64;
        for (j, coeff) in coeffs.iter().enumerate().take(degree + 1).skip(i) {
            let binom = binomial(j, i)?;
            let delta_pow = pow_mod(delta, (j - i) as u32, modulus);
            let coeff_mod = reduce_i128_mod(*coeff, modulus);
            let term = mul_mod(
                mul_mod(coeff_mod, binom % modulus, modulus),
                delta_pow,
                modulus,
            );
            acc = add_mod(acc, term, modulus);
        }
        shifted[i] = acc;
    }
    Ok(shifted)
}

fn eval_poly_mod(input: u64, coeffs: &[i128], modulus: u64) -> u64 {
    coeffs.iter().rev().fold(0u64, |acc, &coeff| {
        add_mod(
            mul_mod(acc, input, modulus),
            reduce_i128_mod(coeff, modulus),
            modulus,
        )
    })
}

fn pow_mod(mut base: u64, mut exp: u32, modulus: u64) -> u64 {
    let mut acc = 1u64 % modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul_mod(acc, base, modulus);
        }
        exp >>= 1;
        if exp > 0 {
            base = mul_mod(base, base, modulus);
        }
    }
    acc
}

fn binomial(n: usize, k: usize) -> Result<u64, OperatorError> {
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut out = 1u128;
    for i in 0..k {
        out = out
            .checked_mul((n - i) as u128)
            .ok_or(OperatorError::InvalidParams("HAT binomial overflow"))?
            / (i + 1) as u128;
    }
    u64::try_from(out).map_err(|_| OperatorError::InvalidParams("HAT binomial overflow"))
}

fn floor_div_pow2(value: i128, shift: u32) -> i128 {
    if value >= 0 {
        value >> shift
    } else {
        -(((-value) + ((1i128 << shift) - 1)) >> shift)
    }
}

fn decode_twos_complement(value: u64, bitwidth: u32) -> i128 {
    let modulus = 1i128 << bitwidth;
    let half = 1i128 << (bitwidth - 1);
    let value = value as i128;
    if value >= half {
        value - modulus
    } else {
        value
    }
}

fn reduce_i128_mod(value: i128, modulus: u64) -> u64 {
    let modulus_i = modulus as i128;
    let reduced = value.rem_euclid(modulus_i);
    reduced as u64
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn derive_secure_rng<R: RngCore + ?Sized>(rng: &mut R) -> SecureRng {
    let mut seed = [0u8; 32];
    rng.fill_bytes(&mut seed);
    SecureRng::from_seed(seed)
}

#[allow(dead_code)]
fn sub_i128_mod(a: u64, b: u64, modulus: u64) -> u64 {
    sub_mod(a, b, modulus)
}
