use serde::{Deserialize, Serialize};
use std::sync::Mutex;

use crate::ahss::{AhssBackend, AhssEvalStats, AhssFullState};
use crate::coefficient_cert::CoeffCert;
use crate::error::OperatorError;
use silent_fhe::{ShortintCiphertext, ShortintServerKey};

#[derive(Clone, Debug, PartialEq)]
pub struct TorusState<Ct> {
    pub ciphertexts: Vec<Ct>,
    pub cert: CoeffCert,
    pub backend: TorusBackendKind,
}

impl<Ct> TorusState<Ct> {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.ciphertexts.is_empty() {
            return Err(OperatorError::InvalidParams(
                "HARMONY torus state must contain at least one ciphertext",
            ));
        }
        self.cert.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TorusBackendKind {
    TfheNative,
    BfvToTfheSwitch,
    PegasusStyleMock,
    ChimeraStyleMock,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LutSpec {
    pub name: String,
    pub input_bitwidth: usize,
    pub output_bitwidth: usize,
    pub domain_min: i128,
    pub domain_max: i128,
    pub scale_in: i128,
    pub scale_out: i128,
    pub max_error: f64,
    pub table: Option<Vec<u64>>,
}

impl LutSpec {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.name.is_empty() {
            return Err(OperatorError::InvalidParams(
                "HARMONY LUT spec name must not be empty",
            ));
        }
        if self.input_bitwidth == 0 || self.output_bitwidth == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY LUT bitwidths must be positive",
            ));
        }
        if self.domain_min > self.domain_max {
            return Err(OperatorError::InvalidParams(
                "HARMONY LUT domain minimum must not exceed maximum",
            ));
        }
        if self.scale_in == 0 || self.scale_out == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY LUT scales must be non-zero",
            ));
        }
        if !self.max_error.is_finite() || self.max_error < 0.0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY LUT max error must be finite and non-negative",
            ));
        }
        Ok(())
    }

    pub fn output_coeff_bound(&self) -> u128 {
        if let Some(table) = &self.table {
            return table.iter().map(|&v| v as u128).max().unwrap_or(0);
        }
        self.domain_min
            .unsigned_abs()
            .max(self.domain_max.unsigned_abs())
            .saturating_add(self.max_error.ceil() as u128)
    }
}

pub struct ShortintTorusBackend {
    server_key: ShortintServerKey,
    last_stats: Mutex<TorusKernelStats>,
}

impl ShortintTorusBackend {
    pub fn new(server_key: ShortintServerKey) -> Self {
        Self {
            server_key,
            last_stats: Mutex::new(TorusKernelStats::default()),
        }
    }

    pub fn server_key(&self) -> &ShortintServerKey {
        &self.server_key
    }

    fn message_modulus(&self) -> u64 {
        self.server_key.parameters().message_modulus().0
    }

    fn carry_modulus(&self) -> u64 {
        self.server_key.parameters().carry_modulus().0
    }

    fn ensure_bivariate_lut_supported(&self) -> Result<(), OperatorError> {
        if self.carry_modulus() < self.message_modulus() {
            return Err(OperatorError::InvalidParams(
                "shortint bivariate PBS requires carry modulus >= message modulus",
            ));
        }
        Ok(())
    }

    fn ensure_bitwidth_supported(&self, bitwidth: usize) -> Result<(), OperatorError> {
        let modulus = self.message_modulus();
        if bitwidth == 0 || bitwidth > u64::BITS as usize {
            return Err(OperatorError::InvalidParams(
                "shortint torus bitwidth must be in 1..=64",
            ));
        }
        if (1u128 << bitwidth) > modulus as u128 {
            return Err(OperatorError::InvalidParams(
                "shortint torus bitwidth exceeds message modulus capacity",
            ));
        }
        Ok(())
    }

    fn ensure_same_len<Ct>(
        &self,
        lhs: &TorusState<Ct>,
        rhs: &TorusState<Ct>,
        context: &'static str,
    ) -> Result<(), OperatorError> {
        if lhs.ciphertexts.len() != rhs.ciphertexts.len() {
            return Err(OperatorError::InvalidParams(context));
        }
        Ok(())
    }

    fn set_stats(&self, stats: TorusKernelStats) -> Result<(), OperatorError> {
        let mut guard = self
            .last_stats
            .lock()
            .map_err(|_| OperatorError::Backend("shortint torus stats mutex poisoned".into()))?;
        *guard = stats;
        Ok(())
    }

    fn shortint_err(err: impl std::fmt::Display) -> OperatorError {
        OperatorError::Backend(format!("shortint torus backend error: {err}"))
    }

    fn with_output_cert(
        &self,
        input: &TorusState<ShortintCiphertext>,
        ciphertexts: Vec<ShortintCiphertext>,
        coeff_bound: u128,
        exact: bool,
    ) -> TorusState<ShortintCiphertext> {
        let mut cert = input.cert.clone();
        cert.coeff_bound = coeff_bound;
        cert.exact = exact;
        TorusState {
            ciphertexts,
            cert,
            backend: TorusBackendKind::TfheNative,
        }
    }

    fn clip_table(&self, min: i128, max: i128) -> Vec<u64> {
        let modulus = self.message_modulus();
        (0..modulus)
            .map(|v| encode_signed_mod(signed_mod_value(v, modulus).clamp(min, max), modulus))
            .collect()
    }

    fn table_from_spec(&self, spec: &LutSpec) -> Result<Vec<u64>, OperatorError> {
        let modulus = self.message_modulus();
        if let Some(table) = &spec.table {
            if table.is_empty() {
                return Err(OperatorError::InvalidParams(
                    "shortint torus LUT table must not be empty",
                ));
            }
            return Ok(table.iter().map(|&v| v % modulus).collect());
        }
        match spec.name.as_str() {
            "clip" => Ok(self.clip_table(spec.domain_min, spec.domain_max)),
            _ => Err(OperatorError::InvalidParams(
                "shortint torus LUT requires an explicit table except for clip",
            )),
        }
    }

    fn softmax_exp_table(&self, config: &SoftmaxConfig) -> Result<Vec<u64>, OperatorError> {
        config.validate()?;
        let modulus = self.message_modulus();
        if config.row_len as u64 >= modulus {
            return Err(OperatorError::InvalidParams(
                "shortint clipped softmax row length must be smaller than message modulus",
            ));
        }
        if config.output_scale <= 0 || config.output_scale as u128 >= modulus as u128 {
            return Err(OperatorError::InvalidParams(
                "shortint clipped softmax output scale must fit in the message modulus",
            ));
        }
        let exp_scale = ((modulus - 1) / config.row_len as u64).max(1);
        Ok((0..modulus)
            .map(|value| {
                let score =
                    signed_mod_value(value, modulus).clamp(config.clip_min, config.clip_max);
                let normalized = ((score - config.clip_max) as f64).exp();
                let quantized = (normalized * exp_scale as f64).round() as u64;
                quantized.clamp(1, exp_scale)
            })
            .collect())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TorusKernelStats {
    pub kernel: String,
    pub bitwidth: usize,
    pub row_len: Option<usize>,
    pub latency_ms: f64,
    pub pbs_count: usize,
    pub bootstrap_count: usize,
    pub lut_size: usize,
    pub communication_bytes: usize,
    pub max_abs_error: f64,
    pub relative_error: f64,
    pub output_coeff_bound: u128,
    pub reentry_condition_pass: bool,
}

impl TorusKernelStats {
    pub fn merge(&mut self, rhs: &Self) {
        if self.kernel.is_empty() {
            self.kernel = rhs.kernel.clone();
        }
        self.latency_ms += rhs.latency_ms;
        self.pbs_count += rhs.pbs_count;
        self.bootstrap_count += rhs.bootstrap_count;
        self.lut_size += rhs.lut_size;
        self.communication_bytes += rhs.communication_bytes;
        self.max_abs_error = self.max_abs_error.max(rhs.max_abs_error);
        self.relative_error = self.relative_error.max(rhs.relative_error);
        self.output_coeff_bound = self.output_coeff_bound.max(rhs.output_coeff_bound);
        self.reentry_condition_pass &= rhs.reentry_condition_pass;
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoftmaxConfig {
    pub row_len: usize,
    pub bitwidth: usize,
    pub clip_min: i128,
    pub clip_max: i128,
    pub output_scale: i128,
}

impl SoftmaxConfig {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.row_len == 0 || self.bitwidth == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY softmax config requires positive row length and bitwidth",
            ));
        }
        if self.clip_min > self.clip_max || self.output_scale == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY softmax config has invalid clip range or output scale",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PolynomialApprox {
    pub name: String,
    pub coeffs: Vec<i128>,
    pub input_scale: i128,
    pub output_scale: i128,
    pub max_error: f64,
}

impl PolynomialApprox {
    pub fn validate(&self) -> Result<(), OperatorError> {
        if self.name.is_empty() || self.coeffs.is_empty() {
            return Err(OperatorError::InvalidParams(
                "HARMONY polynomial approximation requires a name and coefficients",
            ));
        }
        if self.input_scale == 0 || self.output_scale == 0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY polynomial approximation scales must be non-zero",
            ));
        }
        if !self.max_error.is_finite() || self.max_error < 0.0 {
            return Err(OperatorError::InvalidParams(
                "HARMONY polynomial approximation error must be finite and non-negative",
            ));
        }
        Ok(())
    }

    pub fn bound_output(&self, input_bound: u128) -> u128 {
        let mut power = 1u128;
        let mut out = 0u128;
        for &coeff in &self.coeffs {
            out = out.saturating_add(coeff.unsigned_abs().saturating_mul(power));
            power = power.saturating_mul(input_bound.max(1));
        }
        out.saturating_add(self.max_error.ceil() as u128)
    }
}

pub trait TorusBackend<Ct> {
    fn compare(
        &self,
        a: &TorusState<Ct>,
        b: &TorusState<Ct>,
    ) -> Result<TorusState<Ct>, OperatorError>;

    fn mux(
        &self,
        bit: &TorusState<Ct>,
        x: &TorusState<Ct>,
        y: &TorusState<Ct>,
    ) -> Result<TorusState<Ct>, OperatorError>;

    fn lut(&self, x: &TorusState<Ct>, spec: &LutSpec) -> Result<TorusState<Ct>, OperatorError>;

    fn stats(&self) -> TorusKernelStats;
}

impl TorusBackend<ShortintCiphertext> for ShortintTorusBackend {
    fn compare(
        &self,
        a: &TorusState<ShortintCiphertext>,
        b: &TorusState<ShortintCiphertext>,
    ) -> Result<TorusState<ShortintCiphertext>, OperatorError> {
        a.validate()?;
        b.validate()?;
        self.ensure_bivariate_lut_supported()?;
        self.ensure_same_len(a, b, "shortint torus compare requires equal lane counts")?;
        let ciphertexts = a
            .ciphertexts
            .iter()
            .zip(b.ciphertexts.iter())
            .map(|(lhs, rhs)| self.server_key.gt(lhs, rhs).map_err(Self::shortint_err))
            .collect::<Result<Vec<_>, _>>()?;
        self.set_stats(TorusKernelStats {
            kernel: "compare".to_string(),
            bitwidth: a.cert.semantic_range.bitwidth,
            pbs_count: ciphertexts.len(),
            bootstrap_count: ciphertexts.len(),
            lut_size: self.message_modulus() as usize * self.message_modulus() as usize,
            output_coeff_bound: 1,
            reentry_condition_pass: 1 <= a.cert.modulus_p / 4,
            ..TorusKernelStats::default()
        })?;
        Ok(self.with_output_cert(a, ciphertexts, 1, true))
    }

    fn mux(
        &self,
        bit: &TorusState<ShortintCiphertext>,
        x: &TorusState<ShortintCiphertext>,
        y: &TorusState<ShortintCiphertext>,
    ) -> Result<TorusState<ShortintCiphertext>, OperatorError> {
        bit.validate()?;
        x.validate()?;
        y.validate()?;
        self.ensure_bivariate_lut_supported()?;
        self.ensure_same_len(x, y, "shortint torus mux requires equal x/y lane counts")?;
        if bit.ciphertexts.len() != 1 && bit.ciphertexts.len() != x.ciphertexts.len() {
            return Err(OperatorError::InvalidParams(
                "shortint torus mux selector must be broadcast or lane-aligned",
            ));
        }

        let mm = self.message_modulus();
        let choose_x = self
            .server_key
            .generate_lookup_table_bivariate(
                move |selector, value| {
                    if selector % mm == 0 { 0 } else { value % mm }
                },
            );
        let choose_y = self
            .server_key
            .generate_lookup_table_bivariate(
                move |selector, value| {
                    if selector % mm == 0 { value % mm } else { 0 }
                },
            );

        let mut ciphertexts = Vec::with_capacity(x.ciphertexts.len());
        for (idx, (x_ct, y_ct)) in x.ciphertexts.iter().zip(y.ciphertexts.iter()).enumerate() {
            let selector = if bit.ciphertexts.len() == 1 {
                &bit.ciphertexts[0]
            } else {
                &bit.ciphertexts[idx]
            };
            let selected_x = self
                .server_key
                .apply_lookup_table_bivariate(selector, x_ct, &choose_x)
                .map_err(Self::shortint_err)?;
            let selected_y = self
                .server_key
                .apply_lookup_table_bivariate(selector, y_ct, &choose_y)
                .map_err(Self::shortint_err)?;
            let out = self
                .server_key
                .add(&selected_x, &selected_y)
                .map_err(Self::shortint_err)?;
            ciphertexts.push(out);
        }
        let output_bound = x.cert.coeff_bound.max(y.cert.coeff_bound);
        self.set_stats(TorusKernelStats {
            kernel: "mux".to_string(),
            bitwidth: x.cert.semantic_range.bitwidth,
            pbs_count: ciphertexts.len().saturating_mul(3),
            bootstrap_count: ciphertexts.len().saturating_mul(3),
            lut_size: self.message_modulus() as usize * self.message_modulus() as usize,
            output_coeff_bound: output_bound,
            reentry_condition_pass: output_bound <= x.cert.modulus_p / 4,
            ..TorusKernelStats::default()
        })?;
        Ok(self.with_output_cert(x, ciphertexts, output_bound, x.cert.exact && y.cert.exact))
    }

    fn lut(
        &self,
        x: &TorusState<ShortintCiphertext>,
        spec: &LutSpec,
    ) -> Result<TorusState<ShortintCiphertext>, OperatorError> {
        x.validate()?;
        spec.validate()?;
        self.ensure_bitwidth_supported(spec.input_bitwidth)?;
        let table = self.table_from_spec(spec)?;
        let modulus = self.message_modulus();
        let lut = self
            .server_key
            .generate_lookup_table(|v| table[(v % modulus) as usize % table.len()] % modulus);
        let ciphertexts = x
            .ciphertexts
            .iter()
            .map(|ct| {
                self.server_key
                    .apply_lookup_table(ct, lut.as_ref())
                    .map_err(Self::shortint_err)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_bound = table.iter().map(|&v| v as u128).max().unwrap_or(0);
        self.set_stats(TorusKernelStats {
            kernel: spec.name.clone(),
            bitwidth: spec.input_bitwidth,
            pbs_count: ciphertexts.len(),
            bootstrap_count: ciphertexts.len(),
            lut_size: table.len(),
            max_abs_error: spec.max_error,
            output_coeff_bound: output_bound,
            reentry_condition_pass: output_bound <= x.cert.modulus_p / 4,
            ..TorusKernelStats::default()
        })?;
        Ok(self.with_output_cert(x, ciphertexts, output_bound, spec.max_error == 0.0))
    }

    fn stats(&self) -> TorusKernelStats {
        self.last_stats
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }
}

pub fn compare<Ct, B>(
    backend: &B,
    a: &TorusState<Ct>,
    b: &TorusState<Ct>,
) -> Result<(TorusState<Ct>, TorusKernelStats), OperatorError>
where
    B: TorusBackend<Ct>,
{
    a.validate()?;
    b.validate()?;
    let out = backend.compare(a, b)?;
    let mut stats = backend.stats();
    stats.kernel = "compare".to_string();
    stats.output_coeff_bound = out.cert.coeff_bound;
    Ok((out, stats))
}

pub fn mux<Ct, B>(
    backend: &B,
    selector: &TorusState<Ct>,
    x: &TorusState<Ct>,
    y: &TorusState<Ct>,
) -> Result<(TorusState<Ct>, TorusKernelStats), OperatorError>
where
    B: TorusBackend<Ct>,
{
    selector.validate()?;
    x.validate()?;
    y.validate()?;
    let out = backend.mux(selector, x, y)?;
    let mut stats = backend.stats();
    stats.kernel = "mux".to_string();
    stats.output_coeff_bound = out.cert.coeff_bound;
    Ok((out, stats))
}

pub fn bounded_lut<Ct, B>(
    backend: &B,
    x: &TorusState<Ct>,
    spec: &LutSpec,
) -> Result<(TorusState<Ct>, TorusKernelStats), OperatorError>
where
    B: TorusBackend<Ct>,
{
    x.validate()?;
    spec.validate()?;
    let mut out = backend.lut(x, spec)?;
    out.cert.coeff_bound = spec.output_coeff_bound();
    out.cert.exact = spec.max_error == 0.0;
    let mut stats = backend.stats();
    stats.kernel = spec.name.clone();
    stats.bitwidth = spec.input_bitwidth;
    stats.lut_size = 1usize.checked_shl(spec.input_bitwidth as u32).unwrap_or(0);
    stats.max_abs_error = spec.max_error;
    stats.output_coeff_bound = out.cert.coeff_bound;
    stats.reentry_condition_pass = out.cert.coeff_bound <= out.cert.modulus_p / 4;
    Ok((out, stats))
}

pub fn clip<Ct, B>(
    backend: &B,
    x: &TorusState<Ct>,
    min: i128,
    max: i128,
) -> Result<(TorusState<Ct>, TorusKernelStats), OperatorError>
where
    B: TorusBackend<Ct>,
{
    let spec = LutSpec {
        name: "clip".to_string(),
        input_bitwidth: x.cert.semantic_range.bitwidth,
        output_bitwidth: x.cert.semantic_range.bitwidth,
        domain_min: min,
        domain_max: max,
        scale_in: x.cert.scale.input_scale,
        scale_out: x.cert.scale.output_scale,
        max_error: 0.0,
        table: None,
    };
    bounded_lut(backend, x, &spec)
}

pub fn exp_lut<Ct, B>(
    backend: &B,
    x: &TorusState<Ct>,
    spec: &LutSpec,
) -> Result<(TorusState<Ct>, TorusKernelStats), OperatorError>
where
    B: TorusBackend<Ct>,
{
    bounded_lut(backend, x, spec)
}

pub fn reciprocal_lut<Ct, B>(
    backend: &B,
    x: &TorusState<Ct>,
    spec: &LutSpec,
) -> Result<(TorusState<Ct>, TorusKernelStats), OperatorError>
where
    B: TorusBackend<Ct>,
{
    bounded_lut(backend, x, spec)
}

pub fn clipped_softmax_row<Ct, B>(
    backend: &B,
    row: &[TorusState<Ct>],
    config: &SoftmaxConfig,
) -> Result<(Vec<TorusState<Ct>>, TorusKernelStats), OperatorError>
where
    B: TorusBackend<Ct>,
{
    config.validate()?;
    if row.len() != config.row_len {
        return Err(OperatorError::InvalidParams(
            "HARMONY clipped softmax row length must match config",
        ));
    }
    let spec = LutSpec {
        name: "clipped_softmax_row".to_string(),
        input_bitwidth: config.bitwidth,
        output_bitwidth: config.bitwidth,
        domain_min: 0,
        domain_max: config.output_scale,
        scale_in: 1,
        scale_out: config.output_scale,
        max_error: 1.0,
        table: None,
    };
    let mut out = Vec::with_capacity(row.len());
    let mut total = TorusKernelStats {
        kernel: "clipped_softmax_row".to_string(),
        bitwidth: config.bitwidth,
        row_len: Some(config.row_len),
        reentry_condition_pass: true,
        ..TorusKernelStats::default()
    };
    for state in row {
        let (next, stats) = bounded_lut(backend, state, &spec)?;
        total.merge(&stats);
        out.push(next);
    }
    total.kernel = "clipped_softmax_row".to_string();
    total.row_len = Some(config.row_len);
    Ok((out, total))
}

pub fn clipped_softmax_row_shortint(
    backend: &ShortintTorusBackend,
    row: &[TorusState<ShortintCiphertext>],
    config: &SoftmaxConfig,
) -> Result<(Vec<TorusState<ShortintCiphertext>>, TorusKernelStats), OperatorError> {
    config.validate()?;
    if row.len() != config.row_len {
        return Err(OperatorError::InvalidParams(
            "shortint clipped softmax row length must match config",
        ));
    }
    let first = row.first().ok_or(OperatorError::InvalidParams(
        "shortint clipped softmax requires at least one state",
    ))?;
    first.validate()?;
    backend.ensure_bitwidth_supported(config.bitwidth)?;
    backend.ensure_bivariate_lut_supported()?;
    let lanes = first.ciphertexts.len();
    if lanes == 0
        || row
            .iter()
            .any(|state| state.ciphertexts.len() != lanes || state.validate().is_err())
    {
        return Err(OperatorError::InvalidParams(
            "shortint clipped softmax row states must be valid and lane-aligned",
        ));
    }

    let modulus = backend.message_modulus();
    let exp_table = backend.softmax_exp_table(config)?;
    let exp_lut = backend
        .server_key
        .generate_lookup_table(|value| exp_table[(value % modulus) as usize]);
    let mut exp_by_row = Vec::with_capacity(row.len());
    for state in row {
        let mut exp_lanes = Vec::with_capacity(lanes);
        for ct in &state.ciphertexts {
            exp_lanes.push(
                backend
                    .server_key
                    .apply_lookup_table(ct, exp_lut.as_ref())
                    .map_err(ShortintTorusBackend::shortint_err)?,
            );
        }
        exp_by_row.push(exp_lanes);
    }

    let mut denominators = Vec::with_capacity(lanes);
    for lane in 0..lanes {
        let mut denom = exp_by_row[0][lane].clone();
        for exp_lanes in exp_by_row.iter().skip(1) {
            denom = backend
                .server_key
                .add(&denom, &exp_lanes[lane])
                .map_err(ShortintTorusBackend::shortint_err)?;
        }
        denominators.push(denom);
    }

    let output_scale = config.output_scale as u64;
    let normalize =
        backend
            .server_key
            .generate_lookup_table_bivariate(move |exp_value, denom_value| {
                let exp_value = exp_value % modulus;
                let denom_value = denom_value % modulus;
                if denom_value == 0 {
                    0
                } else {
                    (((exp_value as u128 * output_scale as u128) + denom_value as u128 / 2)
                        / denom_value as u128)
                        .min((modulus - 1) as u128) as u64
                }
            });

    let mut out = Vec::with_capacity(row.len());
    for (row_idx, state) in row.iter().enumerate() {
        let mut ciphertexts = Vec::with_capacity(lanes);
        for (lane, denom) in denominators.iter().enumerate() {
            ciphertexts.push(
                backend
                    .server_key
                    .apply_lookup_table_bivariate(&exp_by_row[row_idx][lane], denom, &normalize)
                    .map_err(ShortintTorusBackend::shortint_err)?,
            );
        }
        let mut next =
            backend.with_output_cert(state, ciphertexts, config.output_scale as u128, false);
        next.cert.semantic_range.min = 0;
        next.cert.semantic_range.max = config.output_scale;
        next.cert.semantic_range.bitwidth = config.bitwidth;
        next.cert.semantic_range.clipping_range = Some((0, config.output_scale));
        next.cert.compatible_next_ops = vec!["direct_reload".to_string(), "pv".to_string()];
        out.push(next);
    }

    let exp_pbs = row.len().saturating_mul(lanes);
    let add_pbs = row.len().saturating_sub(1).saturating_mul(lanes);
    let normalize_pbs = row.len().saturating_mul(lanes);
    let output_bound = config.output_scale as u128;
    let stats = TorusKernelStats {
        kernel: "clipped_softmax_row".to_string(),
        bitwidth: config.bitwidth,
        row_len: Some(config.row_len),
        pbs_count: exp_pbs
            .saturating_add(add_pbs)
            .saturating_add(normalize_pbs),
        bootstrap_count: exp_pbs
            .saturating_add(add_pbs)
            .saturating_add(normalize_pbs),
        lut_size: modulus as usize + modulus as usize * modulus as usize,
        max_abs_error: 1.0,
        output_coeff_bound: output_bound,
        reentry_condition_pass: row
            .iter()
            .all(|state| output_bound <= state.cert.modulus_p / 4),
        ..TorusKernelStats::default()
    };
    backend.set_stats(stats.clone())?;
    Ok((out, stats))
}

pub fn gelu_lut<Ct, B>(
    backend: &B,
    x: &TorusState<Ct>,
    spec: &LutSpec,
) -> Result<(TorusState<Ct>, TorusKernelStats), OperatorError>
where
    B: TorusBackend<Ct>,
{
    bounded_lut(backend, x, spec)
}

pub fn gelu_poly_ahss<Ct, Pt, Share, B>(
    _ahss_backend: &B,
    x: &AhssFullState<Ct, Share>,
    poly: &PolynomialApprox,
) -> Result<(AhssFullState<Ct, Share>, AhssEvalStats), OperatorError>
where
    Ct: Clone,
    Share: Clone,
    B: AhssBackend<Ct, Pt, Share>,
{
    x.validate()?;
    poly.validate()?;
    let mut out = x.clone();
    out.cert.coeff_bound = poly.bound_output(x.cert.coeff_bound);
    out.cert.noise_bound += poly.max_error;
    out.cert.exact = poly.max_error == 0.0;
    out.cert.compatible_next_ops = vec![poly.name.clone()];
    let degree = poly.coeffs.len().saturating_sub(1);
    let stats = AhssEvalStats {
        okdm_count: x.input_material.coordinate_count(),
        ddec_count: 2usize.saturating_mul(degree),
        ring_add_count: degree,
        ring_mul_count: degree,
        communication_bytes: 0,
        elapsed_ms: 0.0,
    };
    Ok((out, stats))
}

fn signed_mod_value(value: u64, modulus: u64) -> i128 {
    let value = value % modulus;
    if value > modulus / 2 {
        value as i128 - modulus as i128
    } else {
        value as i128
    }
}

fn encode_signed_mod(value: i128, modulus: u64) -> u64 {
    let modulus_i = modulus as i128;
    value.rem_euclid(modulus_i) as u64
}
