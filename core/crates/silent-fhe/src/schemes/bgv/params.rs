use crate::core::context::HeContext;
use silent_io::HasParamsId;
use silent_math::modulus::Modulus;
use silent_math::numth;
use silent_math::rns::{RnsBase, RnsError};
use silent_math::rns_tool::RnsTool;
use silent_math::rns_tool::RnsToolConfig;
use silent_params::{BgvParams, ModulusBits, MultiplicativeDepth, ParamError, ParameterSet};
use silent_ring::RingContext;
use silent_rlwe::EncryptionParams;

use std::sync::Arc;
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct BgvParameters {
    canonical: Arc<BgvParams>,
    runtime: Arc<EncryptionParams>,
    mod_switch: Arc<BgvModSwitchTables>,
}

#[derive(Clone, Debug)]
pub(crate) struct BgvModSwitchStep {
    pub source_q_limbs: usize,
    pub target_q_limbs: usize,
    pub dropped_modulus: u64,
    pub t_inv_mod_dropped: u64,
    pub dropped_inv_mod_targets: Arc<[u64]>,
}

#[derive(Clone, Debug)]
struct BgvModSwitchTables {
    steps_by_source_limbs: Vec<BgvModSwitchStep>,
    factor_inv_by_target_limbs: Vec<u64>,
}

#[derive(Debug, Error)]
pub enum BgvLevelError {
    #[error("BGV ciphertext modulus chain cannot be empty")]
    EmptyCiphertextModulus,
    #[error("requested {requested} BGV ciphertext moduli but only {available} are available")]
    TooManyCiphertextModuli { requested: usize, available: usize },
    #[error("ciphertext degree {actual} does not match BGV context degree {expected}")]
    CiphertextDegreeMismatch { expected: usize, actual: usize },
    #[error("ciphertext limb count {actual} does not match BGV context limb count {expected}")]
    CiphertextLimbMismatch { expected: usize, actual: usize },
    #[error("ciphertext Q-limb level mismatch: left has {left}, right has {right}")]
    CiphertextLevelMismatch { left: usize, right: usize },
    #[error(
        "ciphertext plaintext modulus {actual} does not match context plaintext modulus {expected}"
    )]
    CiphertextPlaintextModulusMismatch { expected: u64, actual: u64 },
    #[error("ciphertext modulus chains do not match at the requested BGV level")]
    CiphertextModulusChainMismatch,
    #[error("evaluation key has {actual} gadget levels but {expected} are required")]
    EvaluationKeyLevelMismatch { expected: usize, actual: usize },
    #[error("BGV public key must contain exactly two RLWE components, got {actual}")]
    PublicKeySizeMismatch { actual: usize },
    #[error("BGV proxy re-encryption supports only binary ciphertexts, got {actual} components")]
    ProxyReencryptionCiphertextSize { actual: usize },
    #[error("BGV public recryption supports only binary ciphertexts, got {actual} components")]
    RecryptionCiphertextSize { actual: usize },
    #[error(
        "BGV bootstrap carry interpolation bound {bound} is too large for plaintext modulus {modulus}"
    )]
    BootstrapCarryBoundTooLarge { bound: u64, modulus: u64 },
    #[error(
        "BGV bootstrap carry interpolation denominator {denominator} is not invertible modulo plaintext modulus {modulus}"
    )]
    BootstrapCarryInterpolationDenominatorNotInvertible { denominator: u64, modulus: u64 },
    #[error(
        "BGV bootstrap CRT auxiliary prime search failed for {bits}-bit candidates with step {step}"
    )]
    BootstrapCrtPrimeSearchFailed { bits: u32, step: u64 },
    #[error("BGV bootstrap CRT expected {expected} coefficients but got {actual}")]
    BootstrapCrtCoefficientCountMismatch { expected: usize, actual: usize },
    #[error("BGV bootstrap CRT expected {expected} residues but got {actual}")]
    BootstrapCrtResidueCountMismatch { expected: usize, actual: usize },
    #[error(
        "BGV bootstrap CRT reconstruction for coefficient {coefficient} is outside all planned wrap intervals"
    )]
    BootstrapCrtReconstructionOutOfRange { coefficient: usize },
    #[error(
        "BGV bootstrap CRT modulus product has {product_bits} bits but {required_bits} bits are required for the planned phase bound"
    )]
    BootstrapCrtModulusProductTooSmall {
        product_bits: u64,
        required_bits: u64,
    },
    #[error(
        "BGV bootstrap single-modulus classifier plaintext modulus {modulus} is too small for phase coverage requiring more than {required} centered residues"
    )]
    BootstrapSingleModulusClassifierCoverageTooSmall { modulus: u64, required: u64 },
    #[error(
        "BGV bootstrap classifier lookup over plaintext modulus {modulus} exceeds the supported {max_points} interpolation points"
    )]
    BootstrapClassifierLookupTooLarge { modulus: u64, max_points: u64 },
    #[error(
        "BGV bootstrap same-field CRT classifier plaintext modulus {classifier_modulus} must exceed residue modulus {residue_modulus}"
    )]
    BootstrapSameFieldCrtResidueModulusTooLarge {
        classifier_modulus: u64,
        residue_modulus: u64,
    },
    #[error(
        "BGV bootstrap same-field decode plaintext modulus {fusion_modulus} must be at least source plaintext modulus {source_modulus}"
    )]
    BootstrapSameFieldDecodeSourceModulusTooLarge {
        fusion_modulus: u64,
        source_modulus: u64,
    },
    #[error(
        "BGV bootstrap same-field CRT classifier has {tuple_count} residue tuples but supports at most {max_tuples}"
    )]
    BootstrapSameFieldCrtTupleCountTooLarge { tuple_count: u64, max_tuples: u64 },
    #[error(
        "BGV bootstrap fusion-to-source bridge needs noise correction because fusion plaintext modulus {fusion_modulus} is not a multiple of source plaintext modulus {source_modulus}"
    )]
    BootstrapFusionToSourceBridgeRequiresNoiseCorrection {
        fusion_modulus: u64,
        source_modulus: u64,
    },
    #[error(
        "BGV bootstrap fusion noise scale lcm({fusion_modulus}, {source_modulus}) overflows u64"
    )]
    BootstrapFusionNoiseScaleOverflow {
        fusion_modulus: u64,
        source_modulus: u64,
    },
    #[error(
        "BGV fusion error scale {error_scalar} must be a multiple of plaintext modulus {plain_modulus}"
    )]
    BootstrapFusionErrorScaleNotPlaintextMultiple {
        error_scalar: u64,
        plain_modulus: u64,
    },
    #[error("missing BGV Galois key for substitution index {index}")]
    MissingGaloisKey { index: u32 },
    #[error("BGV multi-ciphertext multiplication requires at least one ciphertext")]
    EmptyMulManyInput,
    #[error("BGV plaintext-weighted linear sum requires at least one ciphertext")]
    EmptyLinearCombinationInput,
    #[error(
        "BGV plaintext-weighted linear sum length mismatch: {ciphertexts} ciphertexts and {plaintexts} plaintexts"
    )]
    LinearCombinationLengthMismatch {
        ciphertexts: usize,
        plaintexts: usize,
    },
    #[error("BGV polynomial evaluation requires at least one plaintext coefficient")]
    EmptyPolynomialEvaluation,
    #[error("BGV slot linear transform matrix cannot be empty")]
    EmptySlotLinearTransformMatrix,
    #[error("BGV slot linear transform matrix requires {expected} rows but got {actual}")]
    SlotLinearTransformRowCountMismatch { expected: usize, actual: usize },
    #[error(
        "BGV slot linear transform matrix row {row} requires {expected} columns but got {actual}"
    )]
    SlotLinearTransformColumnCountMismatch {
        row: usize,
        expected: usize,
        actual: usize,
    },
    #[error("BGV packed slot matrix dimensions {rows}x{columns} do not fit in {slots} slots")]
    InvalidSlotMatrixDimensions {
        rows: usize,
        columns: usize,
        slots: usize,
    },
    #[error("BGV slot merge requires at least one ciphertext")]
    EmptySlotMergeInput,
    #[error("BGV cyclic slot rotation {rotation} is not supported for {slots} slots")]
    SlotRotationUnsupported { rotation: i32, slots: usize },
    #[error("BGV slot index {slot} is out of range for {slots} slots")]
    SlotIndexOutOfRange { slot: usize, slots: usize },
    #[error("BGV active slot count {requested} is invalid for {available} available slots")]
    InvalidActiveSlotCount { requested: usize, available: usize },
    #[error(
        "BGV slot merge requested {requested} ciphertexts but only {available} slots are available"
    )]
    SlotMergeInputCountExceeded { requested: usize, available: usize },
    #[error("BGV diagonal linear transform requires at least one diagonal")]
    EmptyDiagonalTransform,
    #[error("plaintext degree {actual} does not match BGV context degree {expected}")]
    PlaintextDegreeMismatch { expected: usize, actual: usize },
    #[error("plaintext limb count {actual} does not match BGV plaintext limb count {expected}")]
    PlaintextLimbMismatch { expected: usize, actual: usize },
    #[error("BGV Galois element {element} is not invertible modulo cyclotomic order {modulus}")]
    InvalidGaloisElement { element: u32, modulus: u64 },
    #[error("plaintext factor {factor} is not invertible modulo plaintext modulus {modulus}")]
    PlaintextFactorNotInvertible { factor: u64, modulus: u64 },
    #[error("BGV parameter construction failed: {0}")]
    Param(#[from] ParamError),
    #[error("BGV RNS construction failed: {0:?}")]
    Rns(RnsError),
}

impl From<RnsError> for BgvLevelError {
    fn from(error: RnsError) -> Self {
        Self::Rns(error)
    }
}

impl BgvParameters {
    pub fn new(params: BgvParams) -> Result<Self, ParamError> {
        params.validate()?;
        let rlwe_params = Arc::new(EncryptionParams::from_bgv_parameter_set(&params)?);
        Ok(Self::from_validated_parts(params, rlwe_params))
    }

    pub fn from_runtime_parts(
        params: BgvParams,
        rlwe_params: Arc<EncryptionParams>,
    ) -> Result<Self, ParamError> {
        params.validate()?;
        rlwe_params.validate_against_rlwe_parameter_set(&params.rlwe, params.plaintext_modulus)?;
        Ok(Self::from_validated_parts(params, rlwe_params))
    }

    pub fn from_runtime(
        name: &'static str,
        distribution: silent_params::DistributionType,
        security_level: silent_params::SecurityLevel,
        multiplicative_depth: silent_params::MultiplicativeDepth,
        plain_modulus: silent_params::PlaintextModulus,
        rlwe_params: Arc<EncryptionParams>,
    ) -> Result<Self, ParamError> {
        let degree = rlwe_params.ring.degree();
        let log_n = silent_params::LogN(degree.trailing_zeros() as u8);
        let ciphertext_modulus_bits = rlwe_params
            .ring
            .rns()
            .moduli()
            .iter()
            .map(|modulus| modulus_bits(modulus.value()))
            .collect();
        let special_modulus_bits = rlwe_params
            .rns_tool
            .base_p()
            .map(|base| {
                base.moduli()
                    .iter()
                    .map(|modulus| modulus_bits(modulus.value()))
                    .collect()
            })
            .unwrap_or_default();
        let key_switch_modulus_bits = rlwe_params
            .key_switch_modulus
            .map(|modulus| modulus_bits(modulus.value()));
        let params = BgvParams {
            name,
            rlwe: silent_params::RlweParams::new(
                name,
                silent_params::RingParams::new(
                    log_n,
                    silent_params::RingDim(degree),
                    distribution,
                ),
                ciphertext_modulus_bits,
                special_modulus_bits,
                key_switch_modulus_bits,
                security_level,
            ),
            plaintext_modulus: plain_modulus,
            multiplicative_depth,
        };
        Self::from_runtime_parts(params, rlwe_params)
    }

    fn from_validated_parts(params: BgvParams, rlwe_params: Arc<EncryptionParams>) -> Self {
        let mod_switch = Arc::new(BgvModSwitchTables::new(
            &rlwe_params,
            params.plaintext_modulus.0,
        ));
        Self {
            canonical: Arc::new(params),
            runtime: rlwe_params,
            mod_switch,
        }
    }

    pub fn canonical(&self) -> &BgvParams {
        self.canonical.as_ref()
    }

    pub fn runtime_params(&self) -> &EncryptionParams {
        self.runtime.as_ref()
    }

    pub fn runtime_params_arc(&self) -> Arc<EncryptionParams> {
        Arc::clone(&self.runtime)
    }

    pub fn ring(&self) -> &RingContext {
        &self.runtime.ring
    }

    pub fn rns_tool(&self) -> &RnsTool {
        &self.runtime.rns_tool
    }

    pub fn degree(&self) -> usize {
        self.canonical.rlwe.ring.ring_dim.0
    }

    pub fn plain_modulus(&self) -> u64 {
        self.canonical.plaintext_modulus.0
    }

    pub fn security_level_bits(&self) -> u32 {
        self.canonical.security_level().bits().unwrap_or(0).into()
    }

    pub fn q_modulus_count(&self) -> usize {
        self.runtime.ring.rns().len()
    }

    pub(crate) fn mod_switch_step_for_source_limbs(
        &self,
        source_q_limbs: usize,
    ) -> Result<&BgvModSwitchStep, BgvLevelError> {
        if source_q_limbs < 2 {
            return Err(BgvLevelError::TooManyCiphertextModuli {
                requested: source_q_limbs.saturating_sub(1),
                available: source_q_limbs,
            });
        }
        if source_q_limbs > self.q_modulus_count() {
            return Err(BgvLevelError::TooManyCiphertextModuli {
                requested: source_q_limbs,
                available: self.q_modulus_count(),
            });
        }
        Ok(&self.mod_switch.steps_by_source_limbs[source_q_limbs - 2])
    }

    pub(crate) fn mod_switch_factor_inv_to_q_limbs(
        &self,
        target_q_limbs: usize,
    ) -> Result<u64, BgvLevelError> {
        if target_q_limbs == 0 {
            return Err(BgvLevelError::EmptyCiphertextModulus);
        }
        if target_q_limbs > self.q_modulus_count() {
            return Err(BgvLevelError::TooManyCiphertextModuli {
                requested: target_q_limbs,
                available: self.q_modulus_count(),
            });
        }
        Ok(self.mod_switch.factor_inv_by_target_limbs[target_q_limbs])
    }

    pub fn with_q_prefix(&self, q_limbs: usize) -> Result<Self, BgvLevelError> {
        if q_limbs == 0 {
            return Err(BgvLevelError::EmptyCiphertextModulus);
        }

        let q_moduli = self.runtime.ring.rns().moduli();
        if q_limbs > q_moduli.len() {
            return Err(BgvLevelError::TooManyCiphertextModuli {
                requested: q_limbs,
                available: q_moduli.len(),
            });
        }

        let base_q = RnsBase::new(q_moduli[..q_limbs].to_vec())?;
        let base_t = Modulus::new(self.plain_modulus()).map_err(ParamError::from)?;
        let mut config = RnsToolConfig::new(base_q, base_t);

        if let Some(base_p) = self.runtime.rns_tool.base_p() {
            config = config
                .with_base_p(base_p.clone())
                .enable_openfhe_switch(false);
        }
        if let Some(key_switch_modulus) = self.runtime.key_switch_modulus {
            config = config.with_key_switch_modulus(key_switch_modulus);
        }

        let runtime = Arc::new(EncryptionParams::from_rns_config(self.degree(), config)?);
        let q_bits = runtime
            .ring
            .rns()
            .moduli()
            .iter()
            .map(|modulus| ModulusBits(modulus_bits(modulus.value()).0))
            .collect();
        let special_bits = runtime
            .rns_tool
            .base_p()
            .map(|base| {
                base.moduli()
                    .iter()
                    .map(|modulus| ModulusBits(modulus_bits(modulus.value()).0))
                    .collect()
            })
            .unwrap_or_default();
        let key_switch_bits = runtime
            .key_switch_modulus
            .map(|modulus| ModulusBits(modulus_bits(modulus.value()).0));
        let remaining_depth = MultiplicativeDepth(
            self.canonical
                .multiplicative_depth
                .0
                .min(q_limbs.saturating_sub(1) as u16),
        );

        let params = BgvParams {
            name: self.canonical.name,
            rlwe: silent_params::RlweParams::new(
                self.canonical.name,
                self.canonical.rlwe.ring.clone(),
                q_bits,
                special_bits,
                key_switch_bits,
                self.canonical.rlwe.security_level,
            ),
            plaintext_modulus: self.canonical.plaintext_modulus,
            multiplicative_depth: remaining_depth,
        };

        Ok(Self::from_runtime_parts(params, runtime)?)
    }
}

impl BgvModSwitchTables {
    fn new(rlwe_params: &EncryptionParams, plain_modulus: u64) -> Self {
        let moduli = rlwe_params.ring.rns().moduli();
        let q_limbs = moduli.len();

        let steps_by_source_limbs = (2..=q_limbs)
            .map(|source_q_limbs| {
                let target_q_limbs = source_q_limbs - 1;
                let dropped_modulus = moduli[target_q_limbs].value();
                let t_inv_mod_dropped =
                    numth::mod_inverse(plain_modulus % dropped_modulus, dropped_modulus)
                        .expect("BGV plaintext modulus is coprime to every Q prime");
                let dropped_inv_mod_targets = moduli[..target_q_limbs]
                    .iter()
                    .map(|modulus| {
                        numth::mod_inverse(dropped_modulus % modulus.value(), modulus.value())
                            .expect("BGV Q primes are pairwise coprime")
                    })
                    .collect::<Vec<_>>()
                    .into();

                BgvModSwitchStep {
                    source_q_limbs,
                    target_q_limbs,
                    dropped_modulus,
                    t_inv_mod_dropped,
                    dropped_inv_mod_targets,
                }
            })
            .collect();

        let mut factor_inv_by_target_limbs = vec![1u64; q_limbs + 1];
        let mut dropped_product = 1u64;
        for target_q_limbs in (0..q_limbs).rev() {
            dropped_product = ((u128::from(dropped_product)
                * u128::from(moduli[target_q_limbs].value() % plain_modulus))
                % u128::from(plain_modulus)) as u64;
            factor_inv_by_target_limbs[target_q_limbs] =
                numth::mod_inverse(dropped_product, plain_modulus)
                    .expect("BGV Q primes are invertible modulo plaintext modulus");
        }

        Self {
            steps_by_source_limbs,
            factor_inv_by_target_limbs,
        }
    }
}

fn modulus_bits(value: u64) -> silent_params::ModulusBits {
    silent_params::ModulusBits((u64::BITS - value.leading_zeros()) as u16)
}

impl HeContext for BgvParameters {
    fn security_level(&self) -> u32 {
        self.security_level_bits()
    }

    fn poly_degree(&self) -> usize {
        self.degree()
    }
}

impl HasParamsId for BgvParameters {
    fn params_id(&self) -> silent_io::ParamsId {
        silent_io::ParamsId(ParameterSet::params_id(self.canonical.as_ref()).0)
    }
}
