use serde::{Deserialize, Serialize};

use crate::ahss::AhssFullState;
use crate::coefficient_cert::{BksParams, CoeffCert, check_bks_admissibility};
use crate::error::OperatorError;
use crate::he_bridge::BridgeState;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReloadMode {
    ExactFastPath,
    ApproxFullRematerialization,
    ApproxRoundedBridgeUpdate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnsafeReloadBaseline {
    LoadRandomSharesSeparately,
    MixedMaterial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterializationAuthority {
    /// A key-owner or trusted materialization service synthesizes BKS material.
    KeyOwnerService,
    /// Setup exposes an OKDM materialization key; this needs KDM/circular assumptions.
    AuthorizedOkdmEvaluationKey,
    /// Test-only plaintext oracle used by toy/unit backends.
    TestOnlyPlaintextOracle,
}

impl MaterializationAuthority {
    pub fn as_str(self) -> &'static str {
        match self {
            MaterializationAuthority::KeyOwnerService => "key_owner_service",
            MaterializationAuthority::AuthorizedOkdmEvaluationKey => {
                "authorized_okdm_evaluation_key"
            }
            MaterializationAuthority::TestOnlyPlaintextOracle => "test_only_plaintext_oracle",
        }
    }

    pub fn requires_kdm_assumption(self) -> bool {
        matches!(self, MaterializationAuthority::AuthorizedOkdmEvaluationKey)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fh2aExactConfig {
    pub noise_flooding_bits: usize,
    pub mask_distribution: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fh2aApproxConfig {
    pub rounding_bits: usize,
    pub max_error: i128,
    pub mode: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReloadStats {
    pub mode: ReloadMode,
    pub materialization_authority: MaterializationAuthority,
    pub requires_kdm_assumption: bool,
    pub fh2a_latency_ms: f64,
    pub okdm_synthesis_ms: f64,
    pub material_add_ms: f64,
    pub load_latency_ms: f64,
    pub total_latency_ms: f64,
    pub communication_bytes: usize,
    pub max_coeff_norm_z: u128,
    pub max_coeff_norm_z0: u128,
    pub max_coeff_norm_z1: u128,
    pub bks_admissible_z: bool,
    pub bks_admissible_z0: bool,
    pub bks_admissible_z1: bool,
    pub correctness_pass: bool,
    pub max_abs_error: i128,
}

impl ReloadStats {
    pub fn recompute_total(&mut self) {
        self.total_latency_ms = self.fh2a_latency_ms
            + self.okdm_synthesis_ms
            + self.material_add_ms
            + self.load_latency_ms;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ApproxShareResult<Pt> {
    pub shares: [Pt; 2],
    pub reconstructed: Pt,
    pub max_abs_error: i128,
}

pub trait Fh2aExact<Ct, Pt> {
    fn share_exact(&self, ciphertext: &Ct) -> Result<[Pt; 2], OperatorError>;

    fn latency_ms(&self) -> f64 {
        0.0
    }
}

pub trait Fh2aApprox<Ct, Pt> {
    fn share_approx(&self, ciphertext: &Ct) -> Result<ApproxShareResult<Pt>, OperatorError>;

    fn latency_ms(&self) -> f64 {
        0.0
    }
}

pub trait BridgeRound<Ct> {
    fn round_bridge(&self, ciphertext: &Ct, rounding_bits: usize) -> Result<Ct, OperatorError>;

    fn max_abs_error(&self, _original: &Ct, _rounded: &Ct) -> Result<i128, OperatorError> {
        Ok(0)
    }

    fn latency_ms(&self) -> f64 {
        0.0
    }
}

pub trait ReloadMaterializer<Ct, Pt, Share> {
    fn combine_plain(&self, shares: &[Pt; 2]) -> Result<Pt, OperatorError>;

    fn materialization_authority(&self) -> MaterializationAuthority {
        MaterializationAuthority::KeyOwnerService
    }

    fn synthesize_material(
        &self,
        value: &Pt,
        bridge: &BridgeState<Ct>,
        mode: ReloadMode,
    ) -> Result<AhssFullState<Ct, Share>, OperatorError>;

    fn coeff_norm(&self, value: &Pt) -> u128;

    fn communication_bytes(&self, _value: &Pt, _mode: ReloadMode) -> usize {
        0
    }

    fn okdm_synthesis_ms(&self, _value: &Pt, _mode: ReloadMode) -> f64 {
        0.0
    }

    fn material_add_ms(&self, _mode: ReloadMode) -> f64 {
        0.0
    }

    fn load_latency_ms(&self, _mode: ReloadMode) -> f64 {
        0.0
    }
}

pub fn direct_reload_exact<Ct, Pt, Share, B, F>(
    ahss_backend: &B,
    fh2a: &F,
    bridge: &BridgeState<Ct>,
    bks_params: &BksParams,
) -> Result<(AhssFullState<Ct, Share>, ReloadStats), OperatorError>
where
    B: ReloadMaterializer<Ct, Pt, Share>,
    F: Fh2aExact<Ct, Pt>,
{
    bridge.validate()?;
    if !bridge.compatible_with_bks_coordinate_1 {
        return Err(OperatorError::Protocol(
            "exact DirectReload fast path requires bridge coordinate compatibility",
        ));
    }
    let shares = fh2a.share_exact(&bridge.ciphertext)?;
    let z = ahss_backend.combine_plain(&shares)?;
    let state = ahss_backend.synthesize_material(&z, bridge, ReloadMode::ExactFastPath)?;
    let stats = reload_stats_from_values(
        ahss_backend,
        &bridge.cert,
        bks_params,
        ReloadMode::ExactFastPath,
        &z,
        &shares[0],
        &shares[1],
        fh2a.latency_ms(),
        0,
        true,
    );
    Ok((state, stats))
}

pub fn direct_reload_approx_full_remat<Ct, Pt, Share, B, F>(
    ahss_backend: &B,
    fh2a: &F,
    bridge: &BridgeState<Ct>,
    bks_params: &BksParams,
) -> Result<(AhssFullState<Ct, Share>, ReloadStats), OperatorError>
where
    B: ReloadMaterializer<Ct, Pt, Share>,
    F: Fh2aApprox<Ct, Pt>,
{
    bridge.validate()?;
    let result = fh2a.share_approx(&bridge.ciphertext)?;
    let state = ahss_backend.synthesize_material(
        &result.reconstructed,
        bridge,
        ReloadMode::ApproxFullRematerialization,
    )?;
    let stats = reload_stats_from_values(
        ahss_backend,
        &bridge.cert,
        bks_params,
        ReloadMode::ApproxFullRematerialization,
        &result.reconstructed,
        &result.shares[0],
        &result.shares[1],
        fh2a.latency_ms(),
        result.max_abs_error,
        true,
    );
    Ok((state, stats))
}

pub fn direct_reload_approx_rounded<Ct, Pt, Share, B, F, R>(
    ahss_backend: &B,
    fh2a: &F,
    rounder: &R,
    bridge: &BridgeState<Ct>,
    bks_params: &BksParams,
    rounding_bits: usize,
) -> Result<(AhssFullState<Ct, Share>, ReloadStats), OperatorError>
where
    Ct: Clone,
    B: ReloadMaterializer<Ct, Pt, Share>,
    F: Fh2aApprox<Ct, Pt>,
    R: BridgeRound<Ct>,
{
    bridge.validate()?;
    let rounded = rounder.round_bridge(&bridge.ciphertext, rounding_bits)?;
    let rounding_error = rounder.max_abs_error(&bridge.ciphertext, &rounded)?;
    let rounded_bridge = BridgeState {
        ciphertext: rounded,
        cert: {
            let mut cert = bridge.cert.clone();
            cert.exact = false;
            cert
        },
        compatible_with_bks_coordinate_1: false,
        he_key_id: bridge.he_key_id.clone(),
    };
    let result = fh2a.share_approx(&rounded_bridge.ciphertext)?;
    let state = ahss_backend.synthesize_material(
        &result.reconstructed,
        &rounded_bridge,
        ReloadMode::ApproxRoundedBridgeUpdate,
    )?;
    let stats = reload_stats_from_values(
        ahss_backend,
        &rounded_bridge.cert,
        bks_params,
        ReloadMode::ApproxRoundedBridgeUpdate,
        &result.reconstructed,
        &result.shares[0],
        &result.shares[1],
        fh2a.latency_ms() + rounder.latency_ms(),
        result.max_abs_error.max(rounding_error),
        true,
    );
    Ok((state, stats))
}

pub fn unsafe_load_random_shares_baseline<Ct, Pt, Share, B, F>(
    ahss_backend: &B,
    fh2a: &F,
    bridge: &BridgeState<Ct>,
    bks_params: &BksParams,
) -> Result<ReloadStats, OperatorError>
where
    B: ReloadMaterializer<Ct, Pt, Share>,
    F: Fh2aExact<Ct, Pt>,
{
    bridge.validate()?;
    let shares = fh2a.share_exact(&bridge.ciphertext)?;
    let z = ahss_backend.combine_plain(&shares)?;
    let mut stats = reload_stats_from_values(
        ahss_backend,
        &bridge.cert,
        bks_params,
        ReloadMode::ExactFastPath,
        &z,
        &shares[0],
        &shares[1],
        fh2a.latency_ms(),
        0,
        false,
    );
    stats.correctness_pass = stats.bks_admissible_z0 && stats.bks_admissible_z1;
    Ok(stats)
}

pub fn unsafe_mixed_material_baseline<Ct, Pt, Share, B, F>(
    ahss_backend: &B,
    fh2a: &F,
    bridge: &BridgeState<Ct>,
    bks_params: &BksParams,
) -> Result<ReloadStats, OperatorError>
where
    B: ReloadMaterializer<Ct, Pt, Share>,
    F: Fh2aApprox<Ct, Pt>,
{
    bridge.validate()?;
    let result = fh2a.share_approx(&bridge.ciphertext)?;
    let mut stats = reload_stats_from_values(
        ahss_backend,
        &bridge.cert,
        bks_params,
        ReloadMode::ApproxRoundedBridgeUpdate,
        &result.reconstructed,
        &result.shares[0],
        &result.shares[1],
        fh2a.latency_ms(),
        result.max_abs_error,
        false,
    );
    stats.correctness_pass = false;
    Ok(stats)
}

fn reload_stats_from_values<Ct, Pt, Share, B>(
    backend: &B,
    cert: &CoeffCert,
    bks_params: &BksParams,
    mode: ReloadMode,
    z: &Pt,
    z0: &Pt,
    z1: &Pt,
    fh2a_latency_ms: f64,
    max_abs_error: i128,
    correctness_pass: bool,
) -> ReloadStats
where
    B: ReloadMaterializer<Ct, Pt, Share>,
{
    let max_coeff_norm_z = backend.coeff_norm(z);
    let max_coeff_norm_z0 = backend.coeff_norm(z0);
    let max_coeff_norm_z1 = backend.coeff_norm(z1);
    let bks_admissible_z = admissible_for_bound(cert, bks_params, max_coeff_norm_z);
    let bks_admissible_z0 = admissible_for_bound(cert, bks_params, max_coeff_norm_z0);
    let bks_admissible_z1 = admissible_for_bound(cert, bks_params, max_coeff_norm_z1);
    let materialization_authority = backend.materialization_authority();
    let mut stats = ReloadStats {
        mode,
        materialization_authority,
        requires_kdm_assumption: materialization_authority.requires_kdm_assumption(),
        fh2a_latency_ms,
        okdm_synthesis_ms: backend.okdm_synthesis_ms(z, mode),
        material_add_ms: backend.material_add_ms(mode),
        load_latency_ms: backend.load_latency_ms(mode),
        total_latency_ms: 0.0,
        communication_bytes: backend.communication_bytes(z, mode),
        max_coeff_norm_z,
        max_coeff_norm_z0,
        max_coeff_norm_z1,
        bks_admissible_z,
        bks_admissible_z0,
        bks_admissible_z1,
        correctness_pass,
        max_abs_error,
    };
    stats.recompute_total();
    stats
}

fn admissible_for_bound(cert: &CoeffCert, bks_params: &BksParams, coeff_bound: u128) -> bool {
    let mut bounded = cert.clone();
    bounded.coeff_bound = coeff_bound;
    bounded.secret_product_bound = Some(coeff_bound.saturating_mul(bks_params.secret_bound));
    check_bks_admissibility(&bounded, bks_params).pass
}
