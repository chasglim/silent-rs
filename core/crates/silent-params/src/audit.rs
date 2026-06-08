//! Parameter audit helpers for scheme-specific implementation constraints.
//!
//! These checks are deliberately narrow and reproducible. They complement
//! [`ParameterSet::validate`](crate::ParameterSet::validate) by returning a
//! report of findings that can be rendered in tooling, CI, or benchmark logs.

use crate::fhe::BgvParams;
use crate::ids::ParameterSet;
use crate::newtypes::ModulusBits;
use crate::registry::{RegisteredParameterSet, all_presets};
use crate::security::SecurityLevel;
use crate::standard::find_max_log_q;
use silent_math::modulus::Modulus;
use silent_math::numth;

/// Severity of a BGV parameter-audit finding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BgvAuditSeverity {
    /// The check passed.
    Pass,
    /// The check produced useful context, but does not indicate a problem.
    Info,
    /// The parameters are accepted, but the finding should be reviewed before production use.
    Warning,
    /// The parameters fail the requested audit target or current implementation constraints.
    Error,
}

impl BgvAuditSeverity {
    /// Stable lowercase label for machine-readable reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

/// Machine-readable BGV audit finding code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BgvAuditCode {
    /// Result of the canonical parameter validation path.
    ParameterValidation,
    /// Whether the declared security level satisfies the requested target.
    SecurityTarget,
    /// Whether ciphertext modulus budget fits the HE Standard v1.1 log-q table.
    HeStandardBudget,
    /// Whether plaintext modulus construction succeeds and has a usable size.
    PlaintextModulus,
    /// Whether plaintext modulus supports the current batch encoder NTT geometry.
    PlaintextNttCompatibility,
    /// Whether generated ciphertext moduli are coprime to the plaintext modulus.
    PlaintextCoprimeWithCiphertextModuli,
    /// Whether the modulus chain length is plausible for the requested multiplicative depth.
    ModulusChainDepth,
    /// Whether modulus generation succeeds for the declared bit-size chain.
    ModulusGeneration,
}

impl BgvAuditCode {
    /// Stable snake_case label for machine-readable reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ParameterValidation => "parameter_validation",
            Self::SecurityTarget => "security_target",
            Self::HeStandardBudget => "he_standard_budget",
            Self::PlaintextModulus => "plaintext_modulus",
            Self::PlaintextNttCompatibility => "plaintext_ntt_compatibility",
            Self::PlaintextCoprimeWithCiphertextModuli => {
                "plaintext_coprime_with_ciphertext_moduli"
            }
            Self::ModulusChainDepth => "modulus_chain_depth",
            Self::ModulusGeneration => "modulus_generation",
        }
    }
}

/// One BGV parameter-audit finding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvAuditFinding {
    /// Stable finding code.
    pub code: BgvAuditCode,
    /// Finding severity.
    pub severity: BgvAuditSeverity,
    /// Human-readable detail for logs or diagnostics.
    pub detail: String,
}

/// BGV audit result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvAuditReport {
    /// Parameter-set name.
    pub name: &'static str,
    /// Requested minimum security target for this audit.
    pub requested_security: SecurityLevel,
    /// Declared parameter security.
    pub declared_security: SecurityLevel,
    /// Ring dimension.
    pub ring_dim: usize,
    /// Sum of ciphertext modulus bit sizes.
    pub ciphertext_modulus_budget: ModulusBits,
    /// HE Standard v1.1 max log-q for the requested target, when available.
    pub he_standard_max_log_q: Option<ModulusBits>,
    /// Signed margin `max_log_q - budget`, when a standard row is available.
    pub he_standard_margin_bits: Option<i32>,
    /// Ordered audit findings.
    pub findings: Vec<BgvAuditFinding>,
}

/// Audit result for a BGV preset registered in [`crate::registry`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgvPresetAuditReport {
    /// Registered preset name.
    pub preset_name: &'static str,
    /// Audit report for the preset.
    pub report: BgvAuditReport,
}

impl BgvAuditReport {
    /// Return true if any finding has [`BgvAuditSeverity::Error`].
    pub fn has_errors(&self) -> bool {
        self.findings
            .iter()
            .any(|finding| finding.severity == BgvAuditSeverity::Error)
    }

    /// Return true if any finding has [`BgvAuditSeverity::Warning`].
    pub fn has_warnings(&self) -> bool {
        self.findings
            .iter()
            .any(|finding| finding.severity == BgvAuditSeverity::Warning)
    }

    /// Number of error findings.
    pub fn error_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|finding| finding.severity == BgvAuditSeverity::Error)
            .count()
    }

    /// Number of warning findings.
    pub fn warning_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|finding| finding.severity == BgvAuditSeverity::Warning)
            .count()
    }

    /// Return true when the report has no errors for the requested target.
    pub fn satisfies_requested_security(&self) -> bool {
        !self.has_errors()
    }
}

/// Audit BGV parameters against current SILENT implementation constraints.
///
/// This is not a lattice estimator. It checks canonical validation, the HE
/// Standard v1.1 log-q table already used by SILENT presets, generated
/// modulus compatibility, and current BGV batch-encoder requirements.
pub fn audit_bgv_params(params: &BgvParams, requested_security: SecurityLevel) -> BgvAuditReport {
    let budget = params.rlwe.ciphertext_modulus_budget_bits();
    let standard_max = if requested_security == SecurityLevel::Toy {
        None
    } else {
        find_max_log_q(
            params.rlwe.ring.distribution,
            requested_security,
            params.rlwe.ring.ring_dim,
        )
    };
    let standard_margin =
        standard_max.map(|max| i32::from(max.0).saturating_sub(i32::from(budget.0)));
    let mut report = BgvAuditReport {
        name: params.name,
        requested_security,
        declared_security: params.security_level(),
        ring_dim: params.rlwe.ring.ring_dim.0,
        ciphertext_modulus_budget: budget,
        he_standard_max_log_q: standard_max,
        he_standard_margin_bits: standard_margin,
        findings: Vec::new(),
    };

    match params.validate() {
        Ok(()) => report.pass(
            BgvAuditCode::ParameterValidation,
            "canonical BGV parameter validation passed",
        ),
        Err(error) => report.error(
            BgvAuditCode::ParameterValidation,
            format!("canonical BGV parameter validation failed: {error}"),
        ),
    }

    audit_security_target(params, requested_security, &mut report);
    audit_standard_budget(requested_security, budget, standard_max, &mut report);
    audit_plaintext_modulus(params, &mut report);
    audit_modulus_generation(params, &mut report);
    audit_chain_depth(params, &mut report);

    report
}

/// Audit every registered BGV preset.
///
/// Set `include_toy` to `false` for production reports that should ignore
/// deliberately small test-only presets.
pub fn audit_bgv_presets(
    requested_security: SecurityLevel,
    include_toy: bool,
) -> Vec<BgvPresetAuditReport> {
    all_presets()
        .into_iter()
        .filter_map(|preset| match preset {
            RegisteredParameterSet::Bgv(params) => {
                if !include_toy && params.security_level() == SecurityLevel::Toy {
                    return None;
                }
                let report = audit_bgv_params(&params, requested_security);
                Some(BgvPresetAuditReport {
                    preset_name: params.name,
                    report,
                })
            }
            _ => None,
        })
        .collect()
}

impl BgvAuditReport {
    fn finding(
        &mut self,
        code: BgvAuditCode,
        severity: BgvAuditSeverity,
        detail: impl Into<String>,
    ) {
        self.findings.push(BgvAuditFinding {
            code,
            severity,
            detail: detail.into(),
        });
    }

    fn pass(&mut self, code: BgvAuditCode, detail: impl Into<String>) {
        self.finding(code, BgvAuditSeverity::Pass, detail);
    }

    fn info(&mut self, code: BgvAuditCode, detail: impl Into<String>) {
        self.finding(code, BgvAuditSeverity::Info, detail);
    }

    fn warning(&mut self, code: BgvAuditCode, detail: impl Into<String>) {
        self.finding(code, BgvAuditSeverity::Warning, detail);
    }

    fn error(&mut self, code: BgvAuditCode, detail: impl Into<String>) {
        self.finding(code, BgvAuditSeverity::Error, detail);
    }
}

fn audit_security_target(
    params: &BgvParams,
    requested_security: SecurityLevel,
    report: &mut BgvAuditReport,
) {
    if requested_security == SecurityLevel::Toy {
        report.info(
            BgvAuditCode::SecurityTarget,
            "toy audit target accepts any declared security level",
        );
        return;
    }

    let declared = params.security_level();
    if security_satisfies(declared, requested_security) {
        report.pass(
            BgvAuditCode::SecurityTarget,
            format!("declared security {declared:?} satisfies requested {requested_security:?}"),
        );
    } else {
        report.error(
            BgvAuditCode::SecurityTarget,
            format!("declared security {declared:?} does not satisfy {requested_security:?}"),
        );
    }
}

fn audit_standard_budget(
    requested_security: SecurityLevel,
    budget: ModulusBits,
    standard_max: Option<ModulusBits>,
    report: &mut BgvAuditReport,
) {
    if requested_security == SecurityLevel::Toy {
        report.info(
            BgvAuditCode::HeStandardBudget,
            "HE Standard log-q budget check skipped for toy target",
        );
        return;
    }

    let Some(max) = standard_max else {
        report.error(
            BgvAuditCode::HeStandardBudget,
            format!("no HE Standard log-q row for requested security {requested_security:?}"),
        );
        return;
    };

    if budget.0 <= max.0 {
        report.pass(
            BgvAuditCode::HeStandardBudget,
            format!(
                "ciphertext modulus budget {} bits is within HE Standard max {} bits",
                budget.0, max.0
            ),
        );
    } else {
        report.error(
            BgvAuditCode::HeStandardBudget,
            format!(
                "ciphertext modulus budget {} bits exceeds HE Standard max {} bits",
                budget.0, max.0
            ),
        );
    }
}

fn audit_plaintext_modulus(params: &BgvParams, report: &mut BgvAuditReport) {
    let t = params.plaintext_modulus.0;
    match Modulus::new(t) {
        Ok(modulus) => {
            report.pass(
                BgvAuditCode::PlaintextModulus,
                format!("plaintext modulus {t} is constructible"),
            );

            let twice_degree = (params.rlwe.ring.ring_dim.0 as u64).saturating_mul(2);
            let ntt_friendly = twice_degree != 0 && t % twice_degree == 1;
            if modulus.is_prime() && ntt_friendly {
                report.pass(
                    BgvAuditCode::PlaintextNttCompatibility,
                    format!(
                        "plaintext modulus {t} is prime and congruent to 1 modulo {twice_degree}"
                    ),
                );
            } else {
                report.error(
                    BgvAuditCode::PlaintextNttCompatibility,
                    format!(
                        "current BGV batch encoder requires prime t with t ≡ 1 mod 2N; \
                         got t={t}, prime={}, t mod 2N={}",
                        modulus.is_prime(),
                        if twice_degree == 0 {
                            0
                        } else {
                            t % twice_degree
                        }
                    ),
                );
            }
        }
        Err(error) => report.error(
            BgvAuditCode::PlaintextModulus,
            format!("plaintext modulus {t} is invalid: {error:?}"),
        ),
    }
}

fn audit_modulus_generation(params: &BgvParams, report: &mut BgvAuditReport) {
    let generated = match params.rlwe.gen_moduli() {
        Ok(generated) => generated,
        Err(error) => {
            report.error(
                BgvAuditCode::ModulusGeneration,
                format!("failed to generate BGV RNS moduli: {error}"),
            );
            return;
        }
    };

    report.pass(
        BgvAuditCode::ModulusGeneration,
        format!(
            "generated {} ciphertext moduli, {} special moduli, key-switch modulus present={}",
            generated.ciphertext_moduli.len(),
            generated.special_moduli.len(),
            generated.key_switch_modulus.is_some()
        ),
    );

    let t = params.plaintext_modulus.0;
    if generated
        .ciphertext_moduli
        .iter()
        .all(|&q| numth::gcd(q, t) == 1)
    {
        report.pass(
            BgvAuditCode::PlaintextCoprimeWithCiphertextModuli,
            "all generated ciphertext moduli are coprime to the plaintext modulus",
        );
    } else {
        report.error(
            BgvAuditCode::PlaintextCoprimeWithCiphertextModuli,
            "at least one generated ciphertext modulus is not coprime to the plaintext modulus",
        );
    }
}

fn audit_chain_depth(params: &BgvParams, report: &mut BgvAuditReport) {
    let q_limbs = params.rlwe.ciphertext_modulus_bits.len();
    let requested_depth = params.multiplicative_depth.0 as usize;
    let conservative_required = requested_depth.saturating_add(1).max(1);
    if q_limbs >= conservative_required {
        report.pass(
            BgvAuditCode::ModulusChainDepth,
            format!(
                "{q_limbs} ciphertext limbs cover conservative depth requirement \
                 of {conservative_required} limbs"
            ),
        );
    } else {
        report.warning(
            BgvAuditCode::ModulusChainDepth,
            format!(
                "{q_limbs} ciphertext limbs may be too short for multiplicative depth \
                 {requested_depth}; conservative requirement is {conservative_required}"
            ),
        );
    }
}

fn security_satisfies(actual: SecurityLevel, minimum: SecurityLevel) -> bool {
    if minimum == SecurityLevel::Toy {
        return true;
    }
    if actual == SecurityLevel::Toy || actual == SecurityLevel::NotSet {
        return false;
    }
    if minimum.is_quantum() && !actual.is_quantum() {
        return false;
    }
    if minimum.is_classical() && !actual.is_classical() {
        return false;
    }
    actual.bits().unwrap_or(0) >= minimum.bits().unwrap_or(u16::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::newtypes::MultiplicativeDepth;
    use crate::presets;

    fn finding(report: &BgvAuditReport, code: BgvAuditCode) -> Option<&BgvAuditFinding> {
        report.findings.iter().find(|finding| finding.code == code)
    }

    #[test]
    fn bgv_audit_accepts_dev_bgv_at_classical128() {
        let params = presets::dev::dev_bgv_4096();
        let report = audit_bgv_params(&params, SecurityLevel::Classical128);

        assert!(!report.has_errors(), "{report:#?}");
        assert_eq!(report.he_standard_max_log_q, Some(ModulusBits(109)));
        assert_eq!(report.he_standard_margin_bits, Some(0));
        assert_eq!(
            finding(&report, BgvAuditCode::PlaintextNttCompatibility)
                .unwrap()
                .severity,
            BgvAuditSeverity::Pass
        );
        assert_eq!(
            BgvAuditCode::HeStandardBudget.as_str(),
            "he_standard_budget"
        );
        assert_eq!(BgvAuditSeverity::Pass.as_str(), "pass");
    }

    #[test]
    fn bgv_audit_flags_toy_preset_for_production_target() {
        let params = presets::toy::toy_bgv_1024();
        let report = audit_bgv_params(&params, SecurityLevel::Classical128);

        assert!(report.has_errors(), "{report:#?}");
        assert!(report.error_count() >= 2);
        assert_eq!(
            finding(&report, BgvAuditCode::SecurityTarget)
                .unwrap()
                .severity,
            BgvAuditSeverity::Error
        );
        assert_eq!(
            finding(&report, BgvAuditCode::PlaintextNttCompatibility)
                .unwrap()
                .severity,
            BgvAuditSeverity::Error
        );
    }

    #[test]
    fn bgv_audit_warns_on_chain_too_short_for_declared_depth() {
        let mut params = presets::dev::dev_bgv_4096();
        params.multiplicative_depth = MultiplicativeDepth(4);
        let report = audit_bgv_params(&params, SecurityLevel::Classical128);

        assert_eq!(
            finding(&report, BgvAuditCode::ModulusChainDepth)
                .unwrap()
                .severity,
            BgvAuditSeverity::Warning
        );
    }

    #[test]
    fn bgv_preset_registry_audit_reports_toy_risk_and_production_presets() {
        let reports = audit_bgv_presets(SecurityLevel::Classical128, true);
        assert_eq!(reports.len(), 3);

        let toy = reports
            .iter()
            .find(|report| report.preset_name == "toy-bgv-1024-v1")
            .expect("toy bgv audit");
        assert!(toy.report.has_errors(), "{toy:#?}");

        let dev = reports
            .iter()
            .find(|report| report.preset_name == "dev-bgv-4096-v1")
            .expect("dev bgv audit");
        assert!(!dev.report.has_errors(), "{dev:#?}");

        let production = audit_bgv_presets(SecurityLevel::Classical128, false);
        assert_eq!(
            production
                .iter()
                .map(|report| report.preset_name)
                .collect::<Vec<_>>(),
            vec!["dev-bgv-4096-v1", "current-bgv-v1"]
        );
        assert!(production.iter().all(|report| !report.report.has_errors()));
    }
}
