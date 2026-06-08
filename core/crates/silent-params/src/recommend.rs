//! Lightweight preset recommendation helpers.
//!
//! This module deliberately chooses from SILENT's audited preset registry
//! instead of running a heavy lattice estimator. For RLWE-family encryption
//! schemes, preset validation and ranking are tied to the Homomorphic
//! Encryption Standard v1.1 log-q tables exposed by [`crate::standard`]. HSS
//! recommendations additionally account for SILENT's BKS19-style
//! bound-aware budget check. PQC recommendations use the FIPS 203 ML-KEM and
//! FIPS 204 ML-DSA parameter-set mappings. The goal is ergonomic selection for
//! examples, tests, demos, and CI while keeping each preset's `validate()`
//! implementation as the final parameter authority.

use crate::audit::audit_bgv_params;
use crate::registry::{RegisteredParameterSet, all_presets};
use crate::scheme::SchemeFamily;
use crate::security::SecurityLevel;
use crate::standard::find_max_log_q;

/// Selection style for [`recommend`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecommendationProfile {
    /// Prefer the smallest valid preset, including toy presets when allowed.
    FastTest,
    /// Prefer the smallest non-toy preset that satisfies the requested security.
    Balanced,
    /// Prefer the named `current-*` preset for the scheme, falling back to
    /// [`Balanced`](Self::Balanced) when no current preset exists.
    Current,
}

/// Inputs for preset recommendation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecommendationRequest {
    /// Scheme family to select.
    pub scheme: SchemeFamily,
    /// Minimum security level required by the caller.
    pub min_security: SecurityLevel,
    /// Selection style.
    pub profile: RecommendationProfile,
    /// Whether toy presets may be returned.
    pub allow_toy: bool,
}

impl RecommendationRequest {
    /// Build a balanced request for the given scheme and minimum security.
    pub const fn balanced(scheme: SchemeFamily, min_security: SecurityLevel) -> Self {
        Self {
            scheme,
            min_security,
            profile: RecommendationProfile::Balanced,
            allow_toy: false,
        }
    }

    /// Build a fast-test request that can return toy presets.
    pub const fn fast_test(scheme: SchemeFamily) -> Self {
        Self {
            scheme,
            min_security: SecurityLevel::Toy,
            profile: RecommendationProfile::FastTest,
            allow_toy: true,
        }
    }

    /// Build a request that prefers the current named preset.
    pub const fn current(scheme: SchemeFamily, min_security: SecurityLevel) -> Self {
        Self {
            scheme,
            min_security,
            profile: RecommendationProfile::Current,
            allow_toy: false,
        }
    }
}

/// A selected preset plus a short machine-readable reason.
#[derive(Clone, Debug)]
pub struct RecommendedParameterSet {
    /// The selected preset.
    pub preset: RegisteredParameterSet,
    /// Why this preset was selected.
    pub reason: &'static str,
}

/// Return the best matching preset for a request.
pub fn recommend(request: RecommendationRequest) -> Option<RecommendedParameterSet> {
    let mut candidates = all_presets()
        .into_iter()
        .filter(|preset| preset.scheme_family() == request.scheme)
        .filter(|preset| request.allow_toy || preset.security_level() != SecurityLevel::Toy)
        .filter(|preset| security_satisfies(preset.security_level(), request.min_security))
        .filter(|preset| preset.validate().is_ok())
        .filter(|preset| preset_satisfies_scheme_audit(preset, request))
        .collect::<Vec<_>>();

    match request.profile {
        RecommendationProfile::Current => {
            if let Some(preset) = candidates
                .iter()
                .find(|preset| preset.name().starts_with("current-"))
                .cloned()
            {
                return Some(RecommendedParameterSet {
                    preset,
                    reason: "current preset",
                });
            }
            if let Some(selected) = fips_pqc_recommendation(&candidates, request) {
                return Some(selected);
            }
            candidates.sort_by_key(|preset| candidate_sort_key(preset, request.min_security));
            candidates
                .into_iter()
                .next()
                .map(|preset| RecommendedParameterSet {
                    reason: recommendation_reason(&preset, request.min_security),
                    preset,
                })
        }
        RecommendationProfile::FastTest => {
            candidates.sort_by_key(|preset| candidate_sort_key(preset, request.min_security));
            candidates
                .into_iter()
                .next()
                .map(|preset| RecommendedParameterSet {
                    preset,
                    reason: "smallest validated preset",
                })
        }
        RecommendationProfile::Balanced => {
            if let Some(selected) = fips_pqc_recommendation(&candidates, request) {
                return Some(selected);
            }
            candidates.sort_by_key(|preset| candidate_sort_key(preset, request.min_security));
            candidates
                .into_iter()
                .next()
                .map(|preset| RecommendedParameterSet {
                    reason: recommendation_reason(&preset, request.min_security),
                    preset,
                })
        }
    }
}

/// Convenience wrapper returning only the preset.
pub fn recommend_preset(
    scheme: SchemeFamily,
    min_security: SecurityLevel,
) -> Option<RegisteredParameterSet> {
    recommend(RecommendationRequest::balanced(scheme, min_security)).map(|selected| selected.preset)
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

fn candidate_sort_key(
    preset: &RegisteredParameterSet,
    minimum_security: SecurityLevel,
) -> (u16, u8, u64, u16, u8, &'static str) {
    let security = preset.security_level().bits().unwrap_or(u16::MAX);
    let class = match preset.security_level() {
        SecurityLevel::Toy => 0,
        level if level.is_classical() => 1,
        level if level.is_quantum() => 2,
        SecurityLevel::NotSet => 3,
        _ => 3,
    };
    let standard_slack = standard_margin_bits_for(preset, minimum_security).unwrap_or(0);
    let current_alias_penalty = u8::from(preset.name().starts_with("current-"));
    (
        security,
        class,
        size_hint(preset),
        standard_slack,
        current_alias_penalty,
        preset.name(),
    )
}

fn recommendation_reason(
    preset: &RegisteredParameterSet,
    minimum_security: SecurityLevel,
) -> &'static str {
    if hss_margin_bits_for(preset).is_some() {
        "smallest BKS19-bound-compatible HSS preset"
    } else if matches!(preset, RegisteredParameterSet::Bgv(_))
        && standard_margin_bits_for(preset, minimum_security).is_some()
    {
        "smallest BGV-audited HE-standard-v1.1-compliant non-toy preset"
    } else if standard_margin_bits_for(preset, minimum_security).is_some() {
        "smallest HE-standard-v1.1-compliant non-toy preset"
    } else {
        "smallest sufficient non-toy preset"
    }
}

fn preset_satisfies_scheme_audit(
    preset: &RegisteredParameterSet,
    request: RecommendationRequest,
) -> bool {
    match preset {
        RegisteredParameterSet::Bgv(params) => {
            audit_bgv_params(params, request.min_security).satisfies_requested_security()
        }
        _ => true,
    }
}

fn standard_margin_bits_for(
    preset: &RegisteredParameterSet,
    minimum_security: SecurityLevel,
) -> Option<u16> {
    if minimum_security == SecurityLevel::Toy {
        return None;
    }
    let params = rlwe_params_for(preset)?;
    let max = find_max_log_q(
        params.ring.distribution,
        minimum_security,
        params.ring.ring_dim,
    )?;
    max.0.checked_sub(params.ciphertext_modulus_budget_bits().0)
}

fn hss_margin_bits_for(preset: &RegisteredParameterSet) -> Option<u16> {
    let RegisteredParameterSet::Hss(params) = preset else {
        return None;
    };
    params
        .rlwe
        .ciphertext_modulus_budget_bits()
        .0
        .checked_sub(params.estimated_required_budget_bits())
}

fn rlwe_params_for(preset: &RegisteredParameterSet) -> Option<&crate::rlwe::RlweParams> {
    match preset {
        RegisteredParameterSet::Rlwe(params) => Some(params),
        RegisteredParameterSet::Bfv(params) => Some(&params.rlwe),
        RegisteredParameterSet::Bgv(params) => Some(&params.rlwe),
        RegisteredParameterSet::Ckks(params) => Some(&params.rlwe),
        RegisteredParameterSet::Hss(params) => Some(&params.rlwe),
        RegisteredParameterSet::Tfhe(_)
        | RegisteredParameterSet::PqcKem(_)
        | RegisteredParameterSet::PqcSig(_) => None,
    }
}

fn fips_pqc_recommendation(
    candidates: &[RegisteredParameterSet],
    request: RecommendationRequest,
) -> Option<RecommendedParameterSet> {
    let (name, reason) = match request.scheme {
        SchemeFamily::PqcKem => fips203_ml_kem_choice(request.min_security)?,
        SchemeFamily::PqcSig => fips204_ml_dsa_choice(request.min_security)?,
        _ => return None,
    };
    candidates
        .iter()
        .find(|preset| preset.name() == name)
        .cloned()
        .map(|preset| RecommendedParameterSet { preset, reason })
}

fn fips203_ml_kem_choice(minimum_security: SecurityLevel) -> Option<(&'static str, &'static str)> {
    match minimum_security {
        SecurityLevel::Toy | SecurityLevel::Quantum128 | SecurityLevel::Quantum192 => {
            Some(("mlkem768-v1", "FIPS 203 default ML-KEM parameter set"))
        }
        SecurityLevel::Quantum256 => Some(("mlkem1024-v1", "FIPS 203 category 5 parameter set")),
        SecurityLevel::Classical128
        | SecurityLevel::Classical192
        | SecurityLevel::Classical256
        | SecurityLevel::NotSet => None,
    }
}

fn fips204_ml_dsa_choice(minimum_security: SecurityLevel) -> Option<(&'static str, &'static str)> {
    match minimum_security {
        SecurityLevel::Toy | SecurityLevel::Quantum128 => {
            Some(("mldsa44-v1", "FIPS 204 category 2 parameter set"))
        }
        SecurityLevel::Quantum192 => Some(("mldsa65-v1", "FIPS 204 category 3 parameter set")),
        SecurityLevel::Quantum256 => Some(("mldsa87-v1", "FIPS 204 category 5 parameter set")),
        SecurityLevel::Classical128
        | SecurityLevel::Classical192
        | SecurityLevel::Classical256
        | SecurityLevel::NotSet => None,
    }
}

fn size_hint(preset: &RegisteredParameterSet) -> u64 {
    match preset {
        RegisteredParameterSet::Rlwe(params) => params.ring.ring_dim.0 as u64,
        RegisteredParameterSet::Bfv(params) => params.rlwe.ring.ring_dim.0 as u64,
        RegisteredParameterSet::Bgv(params) => params.rlwe.ring.ring_dim.0 as u64,
        RegisteredParameterSet::Ckks(params) => params.rlwe.ring.ring_dim.0 as u64,
        RegisteredParameterSet::Hss(params) => params.rlwe.ring.ring_dim.0 as u64,
        RegisteredParameterSet::Tfhe(params) => {
            params.lwe_dimension.0 as u64
                + (params.glwe_dimension.0 as u64 * params.polynomial_size.0 as u64)
        }
        RegisteredParameterSet::PqcKem(params) => {
            params.public_key_bytes as u64
                + params.secret_key_bytes as u64
                + params.ciphertext_bytes as u64
        }
        RegisteredParameterSet::PqcSig(params) => {
            params.public_key_bytes as u64
                + params.secret_key_bytes as u64
                + params.signature_bytes as u64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_test_can_select_toy_tfhe() {
        let selected = recommend(RecommendationRequest::fast_test(SchemeFamily::Tfhe))
            .expect("tfhe toy preset");
        assert_eq!(selected.preset.name(), "toy-tfhe-n512-v1");
        assert_eq!(selected.reason, "smallest validated preset");
    }

    #[test]
    fn balanced_bfv_skips_toy() {
        let selected = recommend(RecommendationRequest::balanced(
            SchemeFamily::Bfv,
            SecurityLevel::Classical128,
        ))
        .expect("bfv dev preset");
        assert_eq!(selected.preset.name(), "dev-bfv-4096-v1");
        assert_eq!(
            selected.reason,
            "smallest HE-standard-v1.1-compliant non-toy preset"
        );
    }

    #[test]
    fn fast_test_bgv_skips_encoder_incompatible_toy_preset() {
        let selected =
            recommend(RecommendationRequest::fast_test(SchemeFamily::Bgv)).expect("bgv preset");
        assert_eq!(selected.preset.name(), "dev-bgv-4096-v1");
        assert_eq!(selected.reason, "smallest validated preset");
    }

    #[test]
    fn balanced_bgv_uses_audited_preset_reason() {
        let selected = recommend(RecommendationRequest::balanced(
            SchemeFamily::Bgv,
            SecurityLevel::Classical128,
        ))
        .expect("bgv dev preset");
        assert_eq!(selected.preset.name(), "dev-bgv-4096-v1");
        assert_eq!(
            selected.reason,
            "smallest BGV-audited HE-standard-v1.1-compliant non-toy preset"
        );
    }

    #[test]
    fn standard_margin_uses_requested_he_security_table() {
        let preset = RegisteredParameterSet::Bfv(crate::presets::dev::dev_bfv_4096());
        assert_eq!(
            standard_margin_bits_for(&preset, SecurityLevel::Classical128),
            Some(0)
        );
        assert_eq!(
            standard_margin_bits_for(&preset, SecurityLevel::Classical192),
            None
        );
    }

    #[test]
    fn current_hss_prefers_current_name() {
        let selected = recommend(RecommendationRequest::current(
            SchemeFamily::Hss,
            SecurityLevel::Classical128,
        ))
        .expect("hss current preset");
        assert_eq!(selected.preset.name(), "current-hss-v1");
        assert_eq!(selected.reason, "current preset");
    }

    #[test]
    fn balanced_hss_reports_bks19_bound_reason() {
        let selected = recommend(RecommendationRequest::balanced(
            SchemeFamily::Hss,
            SecurityLevel::Classical128,
        ))
        .expect("hss dev preset");
        assert_eq!(selected.preset.name(), "dev-hss-4096-v1");
        assert_eq!(hss_margin_bits_for(&selected.preset), Some(7));
        assert_eq!(
            selected.reason,
            "smallest BKS19-bound-compatible HSS preset"
        );
    }

    #[test]
    fn balanced_pqc_kem_uses_fips203_default_for_128_and_192() {
        let selected = recommend_preset(SchemeFamily::PqcKem, SecurityLevel::Quantum128)
            .expect("ml-kem 768 default");
        assert_eq!(selected.name(), "mlkem768-v1");

        let selected = recommend(RecommendationRequest::balanced(
            SchemeFamily::PqcKem,
            SecurityLevel::Quantum192,
        ))
        .expect("ml-kem 768");
        assert_eq!(selected.preset.name(), "mlkem768-v1");
        assert_eq!(selected.reason, "FIPS 203 default ML-KEM parameter set");
    }

    #[test]
    fn balanced_pqc_kem_uses_fips203_category5_for_quantum256() {
        let selected = recommend(RecommendationRequest::balanced(
            SchemeFamily::PqcKem,
            SecurityLevel::Quantum256,
        ))
        .expect("ml-kem 1024");
        assert_eq!(selected.preset.name(), "mlkem1024-v1");
        assert_eq!(selected.reason, "FIPS 203 category 5 parameter set");
    }

    #[test]
    fn fast_test_pqc_kem_can_select_smallest_valid_fips203_set() {
        let selected =
            recommend(RecommendationRequest::fast_test(SchemeFamily::PqcKem)).expect("ml-kem 512");
        assert_eq!(selected.preset.name(), "mlkem512-v1");
        assert_eq!(selected.reason, "smallest validated preset");
    }

    #[test]
    fn balanced_pqc_sig_matches_fips204_categories() {
        let selected = recommend(RecommendationRequest::balanced(
            SchemeFamily::PqcSig,
            SecurityLevel::Quantum128,
        ))
        .expect("ml-dsa 44");
        assert_eq!(selected.preset.name(), "mldsa44-v1");
        assert_eq!(selected.reason, "FIPS 204 category 2 parameter set");

        let selected = recommend(RecommendationRequest::balanced(
            SchemeFamily::PqcSig,
            SecurityLevel::Quantum192,
        ))
        .expect("ml-dsa 65");
        assert_eq!(selected.preset.name(), "mldsa65-v1");
        assert_eq!(selected.reason, "FIPS 204 category 3 parameter set");

        let selected = recommend(RecommendationRequest::balanced(
            SchemeFamily::PqcSig,
            SecurityLevel::Quantum256,
        ))
        .expect("ml-dsa 87");
        assert_eq!(selected.preset.name(), "mldsa87-v1");
        assert_eq!(selected.reason, "FIPS 204 category 5 parameter set");
    }
}
