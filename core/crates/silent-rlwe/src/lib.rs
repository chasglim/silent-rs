//! Unified RLWE types for SILENT.

pub mod bootstrap;
pub mod ggsw;
pub mod glwe;
pub mod io_impls;
pub mod lwe;

pub use bootstrap::LweBootstrapKey;
pub use ggsw::{GgswCiphertext, GgswCiphertextList, GgswLevelMatrix};
pub use glwe::{GlweCiphertext, GlweSecretKey};
pub use io_impls::{EvaluationKeyDecodeParams, GaloisKeyDecodeParams};
pub use lwe::{LweCiphertext, LweKeyswitchKey, LwePublicKey, LweSecretKey};

use silent_math::modulus::Modulus;
use silent_math::rns::RnsError;
use silent_math::rns_tool::{RnsTool, RnsToolConfig};
use silent_params::{
    BfvParams, BgvParams, HssParams, ParamError, ParameterSet, PlaintextModulus, RlweParams,
};
use silent_ring::{Poly, PolyShoup, RingContext};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct EncryptionParams {
    pub ring: Arc<RingContext>,
    pub rns_tool: Arc<RnsTool>,
    pub key_switch_modulus: Option<Modulus>,
}

impl EncryptionParams {
    pub fn new(ring: RingContext, rns_tool: RnsTool) -> Result<Self, RnsError> {
        if ring.rns().moduli() != rns_tool.base_q().moduli() {
            return Err(RnsError::InvalidBase);
        }
        Ok(Self {
            ring: Arc::new(ring),
            rns_tool: Arc::new(rns_tool),
            key_switch_modulus: None,
        })
    }

    pub fn from_rns_config(degree: usize, config: RnsToolConfig) -> Result<Self, RnsError> {
        let ring = RingContext::new(degree, config.base_q().clone());
        let key_switch_modulus = config.key_switch_modulus;
        let rns_tool = RnsTool::from_config(config)?;
        Ok(Self {
            ring: Arc::new(ring),
            rns_tool: Arc::new(rns_tool),
            key_switch_modulus,
        })
    }

    pub fn from_rlwe_parameter_set(
        params: &RlweParams,
        plaintext_modulus: PlaintextModulus,
    ) -> Result<Self, ParamError> {
        params.validate()?;
        let config = params.to_rns_tool_config(plaintext_modulus)?;
        Self::from_rns_config(params.ring.ring_dim.0, config).map_err(ParamError::from)
    }

    pub fn from_bfv_parameter_set(params: &BfvParams) -> Result<Self, ParamError> {
        params.validate()?;
        Self::from_rlwe_parameter_set(&params.rlwe, params.plaintext_modulus)
    }

    pub fn from_bgv_parameter_set(params: &BgvParams) -> Result<Self, ParamError> {
        params.validate()?;
        Self::from_rlwe_parameter_set(&params.rlwe, params.plaintext_modulus)
    }

    pub fn from_hss_parameter_set(params: &HssParams) -> Result<Self, ParamError> {
        params.validate()?;
        Self::from_rlwe_parameter_set(&params.rlwe, params.plaintext_modulus)
    }

    pub fn validate_against_rlwe_parameter_set(
        &self,
        params: &RlweParams,
        plaintext_modulus: PlaintextModulus,
    ) -> Result<(), ParamError> {
        params.validate()?;

        if self.ring.degree() != params.ring.ring_dim.0 {
            return Err(ParamError::RuntimeRingDimensionMismatch {
                expected: params.ring.ring_dim.0,
                actual: self.ring.degree(),
            });
        }

        let actual_plaintext_modulus = self.rns_tool.base_t().value();
        if actual_plaintext_modulus != plaintext_modulus.0 {
            return Err(ParamError::RuntimePlaintextModulusMismatch {
                expected: plaintext_modulus.0,
                actual: actual_plaintext_modulus,
            });
        }

        let expected_q_bits = params
            .ciphertext_modulus_bits
            .iter()
            .map(|bits| bits.0)
            .collect::<Vec<_>>();
        let actual_q_bits = modulus_bits_list(self.ring.rns().moduli());
        if actual_q_bits != expected_q_bits {
            return Err(ParamError::RuntimeModulusChainMismatch {
                label: "ciphertext",
                expected: expected_q_bits,
                actual: actual_q_bits,
            });
        }

        let expected_special_bits = params
            .special_modulus_bits
            .iter()
            .map(|bits| bits.0)
            .collect::<Vec<_>>();
        let actual_special_bits = self
            .rns_tool
            .base_p()
            .map(|base| modulus_bits_list(base.moduli()))
            .unwrap_or_default();
        if actual_special_bits != expected_special_bits {
            return Err(ParamError::RuntimeModulusChainMismatch {
                label: "special",
                expected: expected_special_bits,
                actual: actual_special_bits,
            });
        }

        let expected_key_switch_bits = params.key_switch_modulus_bits.map(|bits| bits.0);
        let actual_key_switch_bits = self
            .key_switch_modulus
            .map(|modulus| bit_width(modulus.value()));
        if actual_key_switch_bits != expected_key_switch_bits {
            return Err(ParamError::RuntimeKeySwitchModulusMismatch {
                expected: expected_key_switch_bits,
                actual: actual_key_switch_bits,
            });
        }

        Ok(())
    }
}

fn modulus_bits_list(moduli: &[Modulus]) -> Vec<u16> {
    moduli
        .iter()
        .map(|modulus| bit_width(modulus.value()))
        .collect()
}

fn bit_width(value: u64) -> u16 {
    if value == 0 {
        0
    } else {
        (u64::BITS - value.leading_zeros()) as u16
    }
}

// Re-export specific Poly type if needed, but for now we use generic Poly
// use silent_ring::Poly; // Already imported at top

#[derive(Clone, Debug)]
pub struct SecretKey {
    pub value: Poly,
}

#[derive(Clone, Debug)]
pub struct Plaintext {
    pub value: Poly,
}

#[derive(Clone, Debug)]
pub struct PublicKey {
    pub pk: Ciphertext,
}

#[derive(Clone, Debug)]
pub struct EvaluationKey {
    /// Keys for RNS relinearization (switching s^2 -> s).
    /// Typically stores encryptions of P*s^2.
    /// In Hybrid mode, this is a single Ciphertext in context Q U P.
    pub elements: Vec<Ciphertext>,
}

#[derive(Clone, Debug)]
pub struct GaloisKey {
    /// Maps Galois element k to the corresponding switching key.
    pub keys: HashMap<u32, EvaluationKey>,
}

#[derive(Clone, Debug)]
pub struct Ciphertext {
    pub data: Vec<Poly>,
    pub params: EncryptionParams,
    pub is_ntt: bool,
    pub seed: Option<[u8; 64]>,
    pub is_seeded_a: bool,
}

impl Ciphertext {
    pub fn new(data: Vec<Poly>, params: EncryptionParams, is_ntt: bool) -> Self {
        Self {
            data,
            params,
            is_ntt,
            seed: None,
            is_seeded_a: false,
        }
    }
}

use rand::{RngCore, SeedableRng};
use silent_utils::rng::{Blake2xbRng, SecureRng};
use std::cell::OnceCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Clone)]
struct QPCache {
    _ring_qp: RingContext,
    _rns_tool_qp: RnsTool,
    _params_qp: EncryptionParams,
    _base_qp: silent_math::rns::RnsBase,
}

#[derive(Clone)]
struct KSwitchCache {
    ring_qk: RingContext,
    params_qk: EncryptionParams,
    base_qk: silent_math::rns::RnsBase,
    qk: silent_math::modulus::Modulus,
}

#[derive(Clone)]
struct SkQpCache {
    sk_hash: u64,
    sk_len: usize,
    sk_qp: Arc<Poly>,
    sk_qp_shoup: Arc<PolyShoup>,
}

#[derive(Clone)]
struct SkQCache {
    sk_hash: u64,
    sk_len: usize,
    sk_q_shoup: Arc<PolyShoup>,
}

struct KeygenScratch {
    e_qk: Poly,
    cbd_bytes: Vec<u8>,
}

impl KeygenScratch {
    fn new(degree: usize, num_moduli: usize) -> Self {
        Self {
            e_qk: unsafe { Poly::new_uninit(degree, num_moduli) },
            cbd_bytes: vec![0u8; degree * 6],
        }
    }

    fn ensure(&mut self, degree: usize, num_moduli: usize) {
        if self.e_qk.degree() != degree || self.e_qk.num_moduli() != num_moduli {
            *self = Self::new(degree, num_moduli);
            return;
        }
        let needed = degree * 6;
        if self.cbd_bytes.len() != needed {
            self.cbd_bytes.resize(needed, 0);
        }
    }
}

fn hash_u64_slice(data: &[u64]) -> u64 {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

mod keygen_profile {
    use super::*;
    use std::cell::RefCell;
    use std::time::Duration;

    #[derive(Clone, Copy)]
    pub enum Kind {
        SkqkInvNtt,
        SkqkExtend,
        SkqkNtt,
        RelinS2,
        GaloisRotate,
        KswTotal,
        KswSampleA,
        KswCbd,
        KswENTT,
        KswMulAS,
        KswAddE,
        KswNegate,
        KswAddQk,
    }

    struct Totals {
        ns: AtomicU64,
        count: AtomicU64,
    }

    impl Totals {
        const fn new() -> Self {
            Self {
                ns: AtomicU64::new(0),
                count: AtomicU64::new(0),
            }
        }

        fn reset(&self) {
            self.ns.store(0, Ordering::Relaxed);
            self.count.store(0, Ordering::Relaxed);
        }
    }

    static ENABLED: OnceLock<bool> = OnceLock::new();
    static SKQK_INV_NTT: Totals = Totals::new();
    static SKQK_EXTEND: Totals = Totals::new();
    static SKQK_NTT: Totals = Totals::new();
    static RELIN_S2: Totals = Totals::new();
    static GALOIS_ROTATE: Totals = Totals::new();
    static KSW_TOTAL: Totals = Totals::new();
    static KSW_SAMPLE_A: Totals = Totals::new();
    static KSW_CBD: Totals = Totals::new();
    static KSW_E_NTT: Totals = Totals::new();
    static KSW_MUL_AS: Totals = Totals::new();
    static KSW_ADD_E: Totals = Totals::new();
    static KSW_NEGATE: Totals = Totals::new();
    static KSW_ADD_QK: Totals = Totals::new();
    static TRACE_ENABLED: OnceLock<bool> = OnceLock::new();

    thread_local! {
        static TRACE_BASE: RefCell<Option<Instant>> = RefCell::new(None);
    }

    fn enabled() -> bool {
        *ENABLED.get_or_init(|| std::env::var("SILENT_KEYGEN_PROFILE").is_ok())
    }

    fn trace_enabled() -> bool {
        *TRACE_ENABLED.get_or_init(|| std::env::var("SILENT_KEYGEN_TRACE").is_ok())
    }

    pub fn trace_on() -> bool {
        trace_enabled()
    }

    fn totals(kind: Kind) -> &'static Totals {
        match kind {
            Kind::SkqkInvNtt => &SKQK_INV_NTT,
            Kind::SkqkExtend => &SKQK_EXTEND,
            Kind::SkqkNtt => &SKQK_NTT,
            Kind::RelinS2 => &RELIN_S2,
            Kind::GaloisRotate => &GALOIS_ROTATE,
            Kind::KswTotal => &KSW_TOTAL,
            Kind::KswSampleA => &KSW_SAMPLE_A,
            Kind::KswCbd => &KSW_CBD,
            Kind::KswENTT => &KSW_E_NTT,
            Kind::KswMulAS => &KSW_MUL_AS,
            Kind::KswAddE => &KSW_ADD_E,
            Kind::KswNegate => &KSW_NEGATE,
            Kind::KswAddQk => &KSW_ADD_QK,
        }
    }

    fn kind_label(kind: Kind) -> &'static str {
        match kind {
            Kind::SkqkInvNtt => "skqk.inv_ntt",
            Kind::SkqkExtend => "skqk.extend",
            Kind::SkqkNtt => "skqk.ntt",
            Kind::RelinS2 => "relin.s2",
            Kind::GaloisRotate => "galois.rotate",
            Kind::KswTotal => "ksw.total",
            Kind::KswSampleA => "ksw.sample_a",
            Kind::KswCbd => "ksw.cbd",
            Kind::KswENTT => "ksw.e_ntt",
            Kind::KswMulAS => "ksw.mul_as",
            Kind::KswAddE => "ksw.add_e",
            Kind::KswNegate => "ksw.negate",
            Kind::KswAddQk => "ksw.add_qk",
        }
    }

    fn trace_emit(label: &str, start: Instant, elapsed: Duration) {
        if !trace_enabled() {
            return;
        }
        let base = TRACE_BASE.with(|cell| cell.borrow().unwrap_or(start));
        let t0 = start.duration_since(base);
        let t1 = t0 + elapsed;
        eprintln!(
            "[keygen][trace] {:<18} t0_ms={:.3} t1_ms={:.3} dt_us={:.3}",
            label,
            t0.as_secs_f64() * 1_000.0,
            t1.as_secs_f64() * 1_000.0,
            elapsed.as_secs_f64() * 1_000_000.0
        );
    }

    pub fn time_if_enabled<T, F: FnOnce() -> T>(kind: Kind, f: F) -> T {
        if enabled() || trace_enabled() {
            let start = Instant::now();
            let out = f();
            let elapsed = start.elapsed();
            if enabled() {
                let t = totals(kind);
                t.ns.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
                t.count.fetch_add(1, Ordering::Relaxed);
            }
            trace_emit(kind_label(kind), start, elapsed);
            out
        } else {
            f()
        }
    }

    pub fn start() -> Option<Instant> {
        if enabled() || trace_enabled() {
            Some(Instant::now())
        } else {
            None
        }
    }

    pub fn end(start: Option<Instant>, kind: Kind) {
        if let Some(start) = start {
            let elapsed = start.elapsed();
            if enabled() {
                let t = totals(kind);
                t.ns.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
                t.count.fetch_add(1, Ordering::Relaxed);
            }
            trace_emit(kind_label(kind), start, elapsed);
        }
    }

    pub struct TraceGuard {
        label: String,
        start: Instant,
    }

    impl Drop for TraceGuard {
        fn drop(&mut self) {
            if !trace_enabled() {
                return;
            }
            let elapsed = self.start.elapsed();
            trace_emit(&self.label, self.start, elapsed);
            TRACE_BASE.with(|cell| {
                *cell.borrow_mut() = None;
            });
        }
    }

    pub fn trace_begin(label: &str) -> Option<TraceGuard> {
        if !trace_enabled() {
            return None;
        }
        let start = Instant::now();
        TRACE_BASE.with(|cell| {
            *cell.borrow_mut() = Some(start);
        });
        eprintln!("[keygen][trace] {} start t0_ms=0.000", label);
        Some(TraceGuard {
            label: label.to_string(),
            start,
        })
    }

    pub fn trace_step<T, F: FnOnce() -> T>(label: &str, f: F) -> T {
        if !trace_enabled() {
            return f();
        }
        let start = Instant::now();
        let out = f();
        let elapsed = start.elapsed();
        trace_emit(label, start, elapsed);
        out
    }

    pub fn dump_and_reset(label: &str) {
        if !enabled() {
            return;
        }
        fn dump_line(name: &str, totals: &Totals) {
            let count = totals.count.load(Ordering::Relaxed);
            if count == 0 {
                return;
            }
            let ns = totals.ns.load(Ordering::Relaxed);
            let avg_ns = ns / count;
            eprintln!(
                "[keygen] {:<18} count={} total_ms={:.3} avg_us={:.3}",
                name,
                count,
                ns as f64 / 1_000_000.0,
                avg_ns as f64 / 1_000.0
            );
        }

        eprintln!("[keygen] profile: {}", label);
        dump_line("skqk.inv_ntt", &SKQK_INV_NTT);
        dump_line("skqk.extend", &SKQK_EXTEND);
        dump_line("skqk.ntt", &SKQK_NTT);
        dump_line("relin.s2", &RELIN_S2);
        dump_line("galois.rotate", &GALOIS_ROTATE);
        dump_line("ksw.total", &KSW_TOTAL);
        dump_line("ksw.sample_a", &KSW_SAMPLE_A);
        dump_line("ksw.cbd", &KSW_CBD);
        dump_line("ksw.e_ntt", &KSW_E_NTT);
        dump_line("ksw.mul_as", &KSW_MUL_AS);
        dump_line("ksw.add_e", &KSW_ADD_E);
        dump_line("ksw.negate", &KSW_NEGATE);
        dump_line("ksw.add_qk", &KSW_ADD_QK);

        SKQK_INV_NTT.reset();
        SKQK_EXTEND.reset();
        SKQK_NTT.reset();
        RELIN_S2.reset();
        GALOIS_ROTATE.reset();
        KSW_TOTAL.reset();
        KSW_SAMPLE_A.reset();
        KSW_CBD.reset();
        KSW_E_NTT.reset();
        KSW_MUL_AS.reset();
        KSW_ADD_E.reset();
        KSW_NEGATE.reset();
        KSW_ADD_QK.reset();
    }
}

fn sample_poly_uniform_seal_blake(
    rng: &mut Blake2xbRng,
    moduli: &[Modulus],
    degree: usize,
    dest: &mut [u64],
) {
    let trace = keygen_profile::trace_on();
    let byte_len = dest.len() * std::mem::size_of::<u64>();
    let bytes = unsafe { std::slice::from_raw_parts_mut(dest.as_mut_ptr() as *mut u8, byte_len) };
    keygen_profile::trace_step("sample_a.fill", || {
        rng.fill_bytes_direct(bytes);
    });

    let max_random = u64::MAX;
    let mut refill_count: u64 = 0;
    keygen_profile::trace_step("sample_a.reduce", || {
        for (limb_idx, modulus) in moduli.iter().enumerate() {
            let max_multiple = max_random - modulus.reduce_u64_fast(max_random) - 1;
            let start = limb_idx * degree;
            let limb = &mut dest[start..start + degree];
            for i in 0..degree {
                let mut rand = unsafe { *limb.get_unchecked(i) };
                while rand >= max_multiple {
                    refill_count += 1;
                    let mut rand_bytes = [0u8; 8];
                    rng.fill_bytes_direct(&mut rand_bytes);
                    rand = u64::from_le_bytes(rand_bytes);
                }
                unsafe {
                    *limb.get_unchecked_mut(i) = modulus.reduce_u64_fast(rand);
                }
            }
        }
    });
    if trace {
        let total = dest.len() as u64 + refill_count;
        eprintln!(
            "[keygen][trace] sample_a.refill count={} total_draws={}",
            refill_count, total
        );
    }
}

fn sample_poly_cbd_seal_blake(
    rng: &mut Blake2xbRng,
    moduli: &[Modulus],
    degree: usize,
    dest: &mut [u64],
) {
    let mut bytes = vec![0u8; degree * 6];
    sample_poly_cbd_seal_blake_with_buf(rng, moduli, degree, dest, &mut bytes);
}

fn sample_poly_cbd_seal_blake_with_buf(
    rng: &mut Blake2xbRng,
    moduli: &[Modulus],
    degree: usize,
    dest: &mut [u64],
    bytes: &mut [u8],
) {
    debug_assert_eq!(bytes.len(), degree * 6);
    keygen_profile::trace_step("sample_cbd.fill", || {
        rng.fill_bytes_direct(bytes);
    });
    keygen_profile::trace_step("sample_cbd.unpack", || {
        for i in 0..degree {
            let base = i * 6;
            let x0 = bytes[base] as u64;
            let x1 = bytes[base + 1] as u64;
            let x2 = (bytes[base + 2] & 0x1F) as u64;
            let x3 = bytes[base + 3] as u64;
            let x4 = bytes[base + 4] as u64;
            let x5 = (bytes[base + 5] & 0x1F) as u64;

            let pos_bits = x0 | (x1 << 8) | (x2 << 16);
            let neg_bits = x3 | (x4 << 8) | (x5 << 16);
            let pos = (pos_bits.count_ones()) as u32;
            let neg = (neg_bits.count_ones()) as u32;
            let noise = pos as i32 - neg as i32;
            let flag = if noise < 0 { u64::MAX } else { 0 };
            let noise_u = noise as u64;
            for (limb_idx, modulus) in moduli.iter().enumerate() {
                unsafe {
                    *dest.get_unchecked_mut(limb_idx * degree + i) =
                        noise_u.wrapping_add(flag & modulus.value());
                }
            }
        }
    });
}

fn fill_ternary_coeffs<R: RngCore>(rng: &mut R, coeffs: &mut [i64]) {
    let max_multiple = u32::MAX - (u32::MAX % 3);
    for coeff in coeffs.iter_mut() {
        loop {
            let v = rng.next_u32();
            if v < max_multiple {
                *coeff = (v % 3) as i64 - 1;
                break;
            }
        }
    }
}

fn negate_poly_inplace(poly: &mut Poly, ring: &RingContext) {
    let moduli = ring.rns().moduli();
    let degree = ring.degree();
    for (i, modulus) in moduli.iter().enumerate() {
        let qi = modulus.value();
        let limb = poly.limb_mut(i);
        for j in 0..degree {
            unsafe {
                let v = *limb.get_unchecked(j);
                *limb.get_unchecked_mut(j) = silent_math::arith::neg_mod(v, qi);
            }
        }
    }
}

fn scalar_mul_poly_inplace(poly: &mut Poly, ring: &RingContext, scalar: u64) {
    let moduli = ring.rns().moduli();
    let degree = ring.degree();
    for (i, modulus) in moduli.iter().enumerate() {
        let qi = modulus.value();
        let factor = scalar % qi;
        let limb = poly.limb_mut(i);
        for j in 0..degree {
            unsafe {
                let v = *limb.get_unchecked(j);
                *limb.get_unchecked_mut(j) = silent_math::arith::mul_mod_u64(v, factor, qi);
            }
        }
    }
}

pub struct KeyGenerator {
    params: EncryptionParams,
    _sk: SecretKey,
    seal_rng: Blake2xbRng,
    qp_cache: OnceCell<QPCache>,
    sk_q_cache: Option<SkQCache>,
    kswitch_cache: OnceCell<KSwitchCache>,
    sk_qk_cache: Option<SkQpCache>,
    keygen_scratch: Option<KeygenScratch>,
}

impl KeyGenerator {
    pub fn new(params: EncryptionParams, rng: SecureRng) -> Self {
        let mut seed = [0u8; 64];
        let mut seed_rng = rng;
        seed_rng.fill_bytes(&mut seed);
        let seal_rng = Blake2xbRng::from_seed_bytes(seed);
        Self {
            params,
            _sk: SecretKey {
                value: Poly::new(0, 0),
            }, // Placeholder, actual SK generated by secret_key()
            seal_rng,
            qp_cache: OnceCell::new(),
            sk_q_cache: None,
            kswitch_cache: OnceCell::new(),
            sk_qk_cache: None,
            keygen_scratch: None,
        }
    }

    pub fn secret_key(&mut self) -> SecretKey {
        let _trace = keygen_profile::trace_begin("secret_key");
        let degree = self.params.ring.degree();
        let rns_tool = &self.params.rns_tool;
        let base_q = rns_tool.base_q();
        let base_p_opt = rns_tool.base_p();

        // Ensure QP cache is initialized if P exists
        if let Some(base_p) = base_p_opt {
            if self.qp_cache.get().is_none() {
                let mut moduli_qp = base_q.moduli().to_vec();
                moduli_qp.extend_from_slice(base_p.moduli());
                let base_qp = silent_math::rns::RnsBase::new(moduli_qp).unwrap(); // Should be valid
                let ring_qp = RingContext::new(degree, base_qp.clone());
                let t = rns_tool.base_t();
                let rns_tool_qp = RnsTool::new(base_qp.clone(), t, None).unwrap();
                let params_qp =
                    EncryptionParams::new(ring_qp.clone(), rns_tool_qp.clone()).unwrap();

                let cache = QPCache {
                    _ring_qp: ring_qp,
                    _rns_tool_qp: rns_tool_qp,
                    _params_qp: params_qp,
                    _base_qp: base_qp,
                };
                let _ = self.qp_cache.set(cache);
            }
        }

        let moduli_q = base_q.moduli();
        let size_q = moduli_q.len();

        // 1. Generate Uniform Ternary Key in Coefficient Domain for Q
        let mut sk_poly = Poly::new(degree, size_q);

        let mut values = vec![0i64; degree];
        keygen_profile::trace_step("sk.ternary", || {
            fill_ternary_coeffs(&mut self.seal_rng, &mut values);
        });

        // Apply to all limbs
        for (mod_idx, modulus) in moduli_q.iter().enumerate() {
            let limb = sk_poly.limb_mut(mod_idx);
            let m = modulus.value();
            for i in 0..degree {
                let v = values[i];
                if v < 0 {
                    limb[i] = m.wrapping_sub(v.abs() as u64); // v is -1
                } else {
                    limb[i] = v as u64;
                }
            }
        }

        // 2. Transform to NTT domain
        // Q part
        keygen_profile::trace_step("sk.ntt", || {
            let ntt_tables_q = self.params.ring.ntt_tables();
            let data = sk_poly.data_mut();
            for i in 0..size_q {
                let limb = &mut data[i * degree..(i + 1) * degree];
                let table = &ntt_tables_q[i];
                silent_math::ntt::ntt_forward(limb, table);
            }
        });

        SecretKey { value: sk_poly }
    }

    pub fn public_key(&mut self, sk: &SecretKey) -> PublicKey {
        let _trace = keygen_profile::trace_begin("public_key");
        let mut seed = [0u8; 32];
        self.seal_rng.fill_bytes(&mut seed);
        let rng = SecureRng::from_seed(seed);
        let mut encryptor = Encryptor::new(self.params.clone(), rng);
        let pk_ct =
            keygen_profile::trace_step("pk.encrypt_zero", || encryptor.encrypt_zero_symmetric(sk));
        PublicKey { pk: pk_ct }
    }

    pub fn public_key_scaled_error(&mut self, sk: &SecretKey, error_scalar: u64) -> PublicKey {
        let _trace = keygen_profile::trace_begin("public_key_scaled_error");
        let mut seed = [0u8; 32];
        self.seal_rng.fill_bytes(&mut seed);
        let rng = SecureRng::from_seed(seed);
        let mut encryptor = Encryptor::new(self.params.clone(), rng);
        let pk_ct = keygen_profile::trace_step("pk.encrypt_zero_scaled", || {
            encryptor.encrypt_zero_symmetric_scaled_error(sk, error_scalar)
        });
        PublicKey { pk: pk_ct }
    }

    fn sk_qk_cached(
        &mut self,
        sk: &SecretKey,
        base_qk: &silent_math::rns::RnsBase,
        qk: &silent_math::modulus::Modulus,
        ring_qk: &RingContext,
    ) -> Arc<Poly> {
        if sk.value.num_moduli() == base_qk.len() {
            return Arc::new(sk.value.clone());
        }

        let sk_data = sk.value.data();
        let sk_hash = hash_u64_slice(sk_data);
        if let Some(cache) = &self.sk_qk_cache {
            if cache.sk_hash == sk_hash && cache.sk_len == sk_data.len() {
                return Arc::clone(&cache.sk_qp);
            }
        }

        let degree = ring_qk.degree();
        let mut s_coeff = sk.value.clone();
        keygen_profile::time_if_enabled(keygen_profile::Kind::SkqkInvNtt, || {
            s_coeff.ntt_inverse(&self.params.ring);
        });

        let mut s_extended = unsafe { silent_ring::Poly::new_uninit(degree, base_qk.len()) };
        let q_moduli = self.params.ring.rns().moduli();
        let size_q = q_moduli.len();

        keygen_profile::time_if_enabled(keygen_profile::Kind::SkqkExtend, || {
            for i in 0..size_q {
                s_extended.limb_mut(i).copy_from_slice(s_coeff.limb(i));
            }

            let q0 = q_moduli[0].value();
            let q0_half = q0 >> 1;
            let qk_val = qk.value();

            for i in 0..degree {
                let val_q = s_coeff.limb(0)[i];
                let (val_abs, is_neg) = if val_q > q0_half {
                    (q0 - val_q, true)
                } else {
                    (val_q, false)
                };

                let val_k = if is_neg { qk_val - val_abs } else { val_abs };
                s_extended.limb_mut(size_q)[i] = val_k;
            }
        });

        let ntt_tables = ring_qk.ntt_tables();
        keygen_profile::time_if_enabled(keygen_profile::Kind::SkqkNtt, || {
            for i in 0..s_extended.num_moduli() {
                let limb = s_extended.limb_mut(i);
                silent_math::ntt::ntt_forward(limb, &ntt_tables[i]);
            }
        });

        let s_extended = Arc::new(s_extended);
        let sk_qp_shoup = Arc::new(PolyShoup::from_poly(&s_extended, ring_qk.rns().moduli()));
        self.sk_qk_cache = Some(SkQpCache {
            sk_hash,
            sk_len: sk_data.len(),
            sk_qp: Arc::clone(&s_extended),
            sk_qp_shoup,
        });

        s_extended
    }

    fn sk_qk_shoup_cached(
        &mut self,
        sk: &SecretKey,
        sk_qk: &Arc<Poly>,
        ring_qk: &RingContext,
    ) -> Arc<PolyShoup> {
        if let Some(cache) = &self.sk_qk_cache {
            let hash = hash_u64_slice(sk.value.data());
            if cache.sk_hash == hash && cache.sk_len == sk.value.data().len() {
                return Arc::clone(&cache.sk_qp_shoup);
            }
        }
        let sk_qp_shoup = Arc::new(PolyShoup::from_poly(sk_qk, ring_qk.rns().moduli()));
        let sk_data = sk.value.data();
        let sk_hash = hash_u64_slice(sk_data);
        self.sk_qk_cache = Some(SkQpCache {
            sk_hash,
            sk_len: sk_data.len(),
            sk_qp: Arc::clone(sk_qk),
            sk_qp_shoup: Arc::clone(&sk_qp_shoup),
        });
        sk_qp_shoup
    }

    fn sk_q_shoup_cached(&mut self, sk: &SecretKey, ring_q: &RingContext) -> Arc<PolyShoup> {
        if let Some(cache) = &self.sk_q_cache {
            let hash = hash_u64_slice(sk.value.data());
            if cache.sk_hash == hash && cache.sk_len == sk.value.data().len() {
                return Arc::clone(&cache.sk_q_shoup);
            }
        }
        let sk_q_shoup = Arc::new(PolyShoup::from_poly(&sk.value, ring_q.rns().moduli()));
        let sk_data = sk.value.data();
        let sk_hash = hash_u64_slice(sk_data);
        self.sk_q_cache = Some(SkQCache {
            sk_hash,
            sk_len: sk_data.len(),
            sk_q_shoup: Arc::clone(&sk_q_shoup),
        });
        sk_q_shoup
    }

    fn compute_hps_garner_factors(base_q: &silent_math::rns::RnsBase) -> Vec<Vec<u64>> {
        let size_q = base_q.len();
        let moduli = base_q.moduli();
        let punctured = base_q.punctured_prod();
        let invs = base_q.inv_punctured_prod_mod_base();
        let mut factors = vec![vec![0u64; size_q]; size_q];
        for i in 0..size_q {
            let inv = invs[i];
            for (j, modulus) in moduli.iter().enumerate() {
                let qj = modulus.value();
                let qhat_mod_qj = punctured[i].mod_u64(qj);
                factors[i][j] = silent_math::arith::mul_mod_u64(qhat_mod_qj, inv, qj);
            }
        }
        factors
    }

    fn build_kswitch_key_into(
        &mut self,
        new_key: &Poly,
        index: usize,
        sk_qk_shoup: &PolyShoup,
        ring_qk: &RingContext,
        base_q: &silent_math::rns::RnsBase,
        qk: silent_math::modulus::Modulus,
        params_qk: &EncryptionParams,
        save_seed: bool,
        error_scalar: Option<u64>,
        out: &mut Ciphertext,
    ) {
        let ksw_total = keygen_profile::start();
        let degree = ring_qk.degree();
        let size_q = base_q.len();
        let size_qk = size_q + 1;

        if out.data.len() != 2
            || out.data[0].degree() != degree
            || out.data[0].num_moduli() != size_qk
        {
            out.data = vec![unsafe { Poly::new_uninit(degree, size_qk) }, unsafe {
                Poly::new_uninit(degree, size_qk)
            }];
        }
        keygen_profile::trace_step("ksw.params", || {
            if out.params.ring.degree() != degree || out.params.rns_tool.base_q().len() != size_qk {
                out.params = params_qk.clone();
            }
        });
        out.is_ntt = true;

        let (b_poly, a_poly) = out.data.split_at_mut(1);
        let b_poly = &mut b_poly[0];
        let a_poly = &mut a_poly[0];

        // Sample a uniformly in NTT domain (QK) into a_poly
        if save_seed {
            let seed_key = keygen_profile::trace_step("ksw.seed", || {
                let mut seed_key = [0u8; 64];
                self.seal_rng.fill_bytes_direct(&mut seed_key);
                seed_key
            });
            let mut a_rng = keygen_profile::trace_step("ksw.seed_rng", || {
                Blake2xbRng::from_seed_bytes(seed_key)
            });
            keygen_profile::time_if_enabled(keygen_profile::Kind::KswSampleA, || {
                sample_poly_uniform_seal_blake(
                    &mut a_rng,
                    ring_qk.rns().moduli(),
                    degree,
                    a_poly.data_mut(),
                );
            });
            out.seed = Some(seed_key);
            out.is_seeded_a = true;
        } else {
            keygen_profile::time_if_enabled(keygen_profile::Kind::KswSampleA, || {
                sample_poly_uniform_seal_blake(
                    &mut self.seal_rng,
                    ring_qk.rns().moduli(),
                    degree,
                    a_poly.data_mut(),
                );
            });
            out.seed = None;
            out.is_seeded_a = false;
        }

        // Scratch for error poly
        let scratch = self
            .keygen_scratch
            .get_or_insert_with(|| KeygenScratch::new(degree, size_qk));
        keygen_profile::trace_step("ksw.scratch", || {
            scratch.ensure(degree, size_qk);
        });

        // Sample error in coeff domain (SEAL-style CBD), then NTT
        keygen_profile::time_if_enabled(keygen_profile::Kind::KswCbd, || {
            sample_poly_cbd_seal_blake_with_buf(
                &mut self.seal_rng,
                ring_qk.rns().moduli(),
                degree,
                scratch.e_qk.data_mut(),
                &mut scratch.cbd_bytes,
            );
        });
        if let Some(scalar) = error_scalar {
            keygen_profile::trace_step("ksw.error_scalar", || {
                scalar_mul_poly_inplace(&mut scratch.e_qk, ring_qk, scalar);
            });
        }
        keygen_profile::time_if_enabled(keygen_profile::Kind::KswENTT, || {
            scratch.e_qk.ntt_forward(ring_qk);
        });

        // b = -(a*s + e) (dyadic product in NTT domain)
        let moduli = ring_qk.rns().moduli();
        keygen_profile::time_if_enabled(keygen_profile::Kind::KswMulAS, || {
            for i in 0..size_qk {
                let modulus = moduli[i].value();
                let a_limb = a_poly.limb(i);
                let s_limb = sk_qk_shoup.limb(i);
                let b_limb = b_poly.limb_mut(i);
                for j in 0..degree {
                    let shoup = &s_limb[j];
                    b_limb[j] = silent_math::arith::mul_mod_shoup(
                        a_limb[j],
                        shoup.operand,
                        shoup.quotient,
                        modulus,
                    );
                }
            }
        });
        keygen_profile::time_if_enabled(keygen_profile::Kind::KswAddE, || {
            b_poly.add_assign(&scratch.e_qk, ring_qk);
        });
        keygen_profile::time_if_enabled(keygen_profile::Kind::KswNegate, || {
            negate_poly_inplace(b_poly, ring_qk);
        });

        // Add qk * new_key to the index-th limb (mod q_i)
        let modulus = &base_q.moduli()[index];
        let qi = modulus.value();
        let factor = modulus.reduce_u64(qk.value());
        let b_limb = b_poly.limb_mut(index);
        let new_limb = new_key.limb(index);
        keygen_profile::time_if_enabled(keygen_profile::Kind::KswAddQk, || {
            for j in 0..degree {
                let term = silent_math::arith::mul_mod_u64(new_limb[j], factor, qi);
                b_limb[j] = silent_math::arith::add_mod(b_limb[j], term, qi);
            }
        });

        keygen_profile::end(ksw_total, keygen_profile::Kind::KswTotal);
    }

    fn build_kswitch_key_hps_into(
        &mut self,
        new_key: &Poly,
        index: usize,
        sk_q_shoup: &PolyShoup,
        ring_q: &RingContext,
        base_q: &silent_math::rns::RnsBase,
        params_q: &EncryptionParams,
        save_seed: bool,
        error_scalar: Option<u64>,
        out: &mut Ciphertext,
        garner_factors: &[Vec<u64>],
    ) {
        let ksw_total = keygen_profile::start();
        let degree = ring_q.degree();
        let size_q = base_q.len();

        if out.data.len() != 2
            || out.data[0].degree() != degree
            || out.data[0].num_moduli() != size_q
        {
            out.data = vec![unsafe { Poly::new_uninit(degree, size_q) }, unsafe {
                Poly::new_uninit(degree, size_q)
            }];
        }
        keygen_profile::trace_step("ksw.params", || {
            if out.params.ring.degree() != degree || out.params.rns_tool.base_q().len() != size_q {
                out.params = params_q.clone();
            }
        });
        out.is_ntt = true;

        let (b_poly, a_poly) = out.data.split_at_mut(1);
        let b_poly = &mut b_poly[0];
        let a_poly = &mut a_poly[0];

        // Sample a uniformly in NTT domain (Q) into a_poly
        if save_seed {
            let seed_key = keygen_profile::trace_step("ksw.seed", || {
                let mut seed_key = [0u8; 64];
                self.seal_rng.fill_bytes_direct(&mut seed_key);
                seed_key
            });
            let mut a_rng = keygen_profile::trace_step("ksw.seed_rng", || {
                Blake2xbRng::from_seed_bytes(seed_key)
            });
            keygen_profile::time_if_enabled(keygen_profile::Kind::KswSampleA, || {
                sample_poly_uniform_seal_blake(
                    &mut a_rng,
                    ring_q.rns().moduli(),
                    degree,
                    a_poly.data_mut(),
                );
            });
            out.seed = Some(seed_key);
            out.is_seeded_a = true;
        } else {
            keygen_profile::time_if_enabled(keygen_profile::Kind::KswSampleA, || {
                sample_poly_uniform_seal_blake(
                    &mut self.seal_rng,
                    ring_q.rns().moduli(),
                    degree,
                    a_poly.data_mut(),
                );
            });
            out.seed = None;
            out.is_seeded_a = false;
        }

        // Scratch for error poly
        let scratch = self
            .keygen_scratch
            .get_or_insert_with(|| KeygenScratch::new(degree, size_q));
        keygen_profile::trace_step("ksw.scratch", || {
            scratch.ensure(degree, size_q);
        });

        // Sample error in coeff domain (SEAL-style CBD), then NTT
        keygen_profile::time_if_enabled(keygen_profile::Kind::KswCbd, || {
            sample_poly_cbd_seal_blake_with_buf(
                &mut self.seal_rng,
                ring_q.rns().moduli(),
                degree,
                scratch.e_qk.data_mut(),
                &mut scratch.cbd_bytes,
            );
        });
        if let Some(scalar) = error_scalar {
            keygen_profile::trace_step("ksw.error_scalar", || {
                scalar_mul_poly_inplace(&mut scratch.e_qk, ring_q, scalar);
            });
        }
        keygen_profile::time_if_enabled(keygen_profile::Kind::KswENTT, || {
            scratch.e_qk.ntt_forward(ring_q);
        });

        // b = -(a*s + e) (dyadic product in NTT domain)
        let moduli = ring_q.rns().moduli();
        keygen_profile::time_if_enabled(keygen_profile::Kind::KswMulAS, || {
            for i in 0..size_q {
                let modulus = moduli[i].value();
                let a_limb = a_poly.limb(i);
                let s_limb = sk_q_shoup.limb(i);
                let b_limb = b_poly.limb_mut(i);
                for j in 0..degree {
                    let shoup = &s_limb[j];
                    b_limb[j] = silent_math::arith::mul_mod_shoup(
                        a_limb[j],
                        shoup.operand,
                        shoup.quotient,
                        modulus,
                    );
                }
            }
        });
        keygen_profile::time_if_enabled(keygen_profile::Kind::KswAddE, || {
            b_poly.add_assign(&scratch.e_qk, ring_q);
        });
        keygen_profile::time_if_enabled(keygen_profile::Kind::KswNegate, || {
            negate_poly_inplace(b_poly, ring_q);
        });

        // Add g_i * new_key to all limbs (Garner CRT basis)
        let factors = &garner_factors[index];
        keygen_profile::time_if_enabled(keygen_profile::Kind::KswAddQk, || {
            for (j, modulus) in moduli.iter().enumerate() {
                let qj = modulus.value();
                let factor = factors[j];
                let b_limb = b_poly.limb_mut(j);
                let new_limb = new_key.limb(j);
                for k in 0..degree {
                    let term = silent_math::arith::mul_mod_u64(new_limb[k], factor, qj);
                    b_limb[k] = silent_math::arith::add_mod(b_limb[k], term, qj);
                }
            }
        });

        keygen_profile::end(ksw_total, keygen_profile::Kind::KswTotal);
    }

    pub fn relinearization_key(&mut self, sk: &SecretKey) -> Result<EvaluationKey, RnsError> {
        self.relinearization_key_with_error_scalar(sk, None, "relin_keys")
    }

    pub fn relinearization_key_scaled_error(
        &mut self,
        sk: &SecretKey,
        error_scalar: u64,
    ) -> Result<EvaluationKey, RnsError> {
        self.relinearization_key_with_error_scalar(
            sk,
            Some(error_scalar),
            "relin_keys_scaled_error",
        )
    }

    fn relinearization_key_with_error_scalar(
        &mut self,
        sk: &SecretKey,
        error_scalar: Option<u64>,
        profile_label: &'static str,
    ) -> Result<EvaluationKey, RnsError> {
        let _trace = keygen_profile::trace_begin(profile_label);
        let base_q = self.params.rns_tool.base_q().clone();
        let qk = if let Some(qk) = self.params.key_switch_modulus {
            qk
        } else {
            let base_p = self.params.rns_tool.base_p().ok_or(RnsError::InvalidBase)?;
            *base_p.moduli().last().ok_or(RnsError::InvalidBase)?
        };
        let size_q = base_q.len();

        if self.kswitch_cache.get().is_none() {
            let mut moduli_qk = base_q.moduli().to_vec();
            moduli_qk.push(qk);
            let base_qk = silent_math::rns::RnsBase::new(moduli_qk)?;
            let ring_qk = RingContext::new(self.params.ring.degree(), base_qk.clone());

            let t = self.params.rns_tool.base_t();
            let rns_tool_qk = RnsTool::new(base_qk.clone(), t, None)?;
            let params_qk = EncryptionParams::new(ring_qk.clone(), rns_tool_qk)?;

            let cache = KSwitchCache {
                ring_qk,
                params_qk,
                base_qk,
                qk,
            };
            let _ = self.kswitch_cache.set(cache);
        }

        let (ring_qk, base_qk, params_qk, qk) = {
            let cache = self.kswitch_cache.get().unwrap();
            (
                cache.ring_qk.clone(),
                cache.base_qk.clone(),
                cache.params_qk.clone(),
                cache.qk,
            )
        };

        let sk_qk = keygen_profile::trace_step("relin.sk_qk", || {
            self.sk_qk_cached(sk, &base_qk, &qk, &ring_qk)
        });
        let sk_qk_shoup = keygen_profile::trace_step("relin.sk_qk_shoup", || {
            self.sk_qk_shoup_cached(sk, &sk_qk, &ring_qk)
        });

        // s^2 in NTT (base Q)
        let mut s2 = sk.value.clone();
        keygen_profile::time_if_enabled(keygen_profile::Kind::RelinS2, || {
            s2.mul_assign(&sk.value, &self.params.ring);
        });

        let mut elements = Vec::with_capacity(size_q);
        for i in 0..size_q {
            let mut ct = Ciphertext::new(
                vec![
                    unsafe { Poly::new_uninit(ring_qk.degree(), base_q.len() + 1) },
                    unsafe { Poly::new_uninit(ring_qk.degree(), base_q.len() + 1) },
                ],
                params_qk.clone(),
                true,
            );
            keygen_profile::trace_step(&format!("relin.ksw[{}]", i), || {
                self.build_kswitch_key_into(
                    &s2,
                    i,
                    sk_qk_shoup.as_ref(),
                    &ring_qk,
                    &base_q,
                    qk,
                    &params_qk,
                    false,
                    error_scalar,
                    &mut ct,
                );
            });
            elements.push(ct);
        }

        let out = EvaluationKey { elements };
        keygen_profile::dump_and_reset(profile_label);
        Ok(out)
    }

    pub fn relinearization_key_hps(&mut self, sk: &SecretKey) -> Result<EvaluationKey, RnsError> {
        self.relinearization_key_hps_with_error_scalar(sk, None, "relin_keys_hps")
    }

    pub fn relinearization_key_hps_scaled_error(
        &mut self,
        sk: &SecretKey,
        error_scalar: u64,
    ) -> Result<EvaluationKey, RnsError> {
        self.relinearization_key_hps_with_error_scalar(
            sk,
            Some(error_scalar),
            "relin_keys_hps_scaled_error",
        )
    }

    fn relinearization_key_hps_with_error_scalar(
        &mut self,
        sk: &SecretKey,
        error_scalar: Option<u64>,
        profile_label: &'static str,
    ) -> Result<EvaluationKey, RnsError> {
        let _trace = keygen_profile::trace_begin(profile_label);
        let base_q = self.params.rns_tool.base_q().clone();
        let ring_q = self.params.ring.clone();
        let params_q = self.params.clone();
        let size_q = base_q.len();

        let sk_q_shoup = self.sk_q_shoup_cached(sk, &ring_q);
        let sk_q_shoup = Arc::clone(&sk_q_shoup);

        let mut s2 = sk.value.clone();
        keygen_profile::time_if_enabled(keygen_profile::Kind::RelinS2, || {
            s2.mul_assign(&sk.value, &self.params.ring);
        });

        let garner_factors = Self::compute_hps_garner_factors(&base_q);

        let mut elements = Vec::with_capacity(size_q);
        for i in 0..size_q {
            let mut ct = Ciphertext::new(
                vec![
                    unsafe { Poly::new_uninit(ring_q.degree(), base_q.len()) },
                    unsafe { Poly::new_uninit(ring_q.degree(), base_q.len()) },
                ],
                params_q.clone(),
                true,
            );
            keygen_profile::trace_step(&format!("relin_hps.ksw[{}]", i), || {
                self.build_kswitch_key_hps_into(
                    &s2,
                    i,
                    sk_q_shoup.as_ref(),
                    &ring_q,
                    &base_q,
                    &params_q,
                    false,
                    error_scalar,
                    &mut ct,
                    &garner_factors,
                );
            });
            elements.push(ct);
        }

        let out = EvaluationKey { elements };
        keygen_profile::dump_and_reset(profile_label);
        Ok(out)
    }

    pub fn galois_keys(
        &mut self,
        sk: &SecretKey,
        substitution_indices: &[u32],
    ) -> Result<GaloisKey, RnsError> {
        self.galois_keys_with_error_scalar(sk, substitution_indices, None, "galois_keys")
    }

    pub fn galois_keys_scaled_error(
        &mut self,
        sk: &SecretKey,
        substitution_indices: &[u32],
        error_scalar: u64,
    ) -> Result<GaloisKey, RnsError> {
        self.galois_keys_with_error_scalar(
            sk,
            substitution_indices,
            Some(error_scalar),
            "galois_keys_scaled_error",
        )
    }

    fn galois_keys_with_error_scalar(
        &mut self,
        sk: &SecretKey,
        substitution_indices: &[u32],
        error_scalar: Option<u64>,
        profile_label: &'static str,
    ) -> Result<GaloisKey, RnsError> {
        let _trace = keygen_profile::trace_begin(profile_label);
        let base_q = self.params.rns_tool.base_q().clone();
        let qk = if let Some(qk) = self.params.key_switch_modulus {
            qk
        } else {
            let base_p = self.params.rns_tool.base_p().ok_or(RnsError::InvalidBase)?;
            *base_p.moduli().last().ok_or(RnsError::InvalidBase)?
        };
        let size_q = base_q.len();

        if self.kswitch_cache.get().is_none() {
            let mut moduli_qk = base_q.moduli().to_vec();
            moduli_qk.push(qk);
            let base_qk = silent_math::rns::RnsBase::new(moduli_qk)?;
            let ring_qk = RingContext::new(self.params.ring.degree(), base_qk.clone());

            let t = self.params.rns_tool.base_t();
            let rns_tool_qk = RnsTool::new(base_qk.clone(), t, None)?;
            let params_qk = EncryptionParams::new(ring_qk.clone(), rns_tool_qk)?;

            let cache = KSwitchCache {
                ring_qk,
                params_qk,
                base_qk,
                qk,
            };
            let _ = self.kswitch_cache.set(cache);
        }

        let (ring_qk, base_qk, params_qk, qk) = {
            let cache = self.kswitch_cache.get().unwrap();
            (
                cache.ring_qk.clone(),
                cache.base_qk.clone(),
                cache.params_qk.clone(),
                cache.qk,
            )
        };

        let sk_qk = keygen_profile::trace_step("galois.sk_qk", || {
            self.sk_qk_cached(sk, &base_qk, &qk, &ring_qk)
        });
        let sk_qk_shoup = keygen_profile::trace_step("galois.sk_qk_shoup", || {
            self.sk_qk_shoup_cached(sk, &sk_qk, &ring_qk)
        });
        let ring_q = self.params.ring.clone();

        let mut keys: HashMap<u32, EvaluationKey> = HashMap::new();
        for &k in substitution_indices {
            let map = ring_q.automorphism_map(k as u64);
            let mut sk_rot = sk.value.clone();
            keygen_profile::time_if_enabled(keygen_profile::Kind::GaloisRotate, || {
                sk_rot.apply_automorphism(&map);
            });

            let mut elements = Vec::with_capacity(size_q);
            for i in 0..size_q {
                let mut ct = Ciphertext::new(
                    vec![
                        unsafe { Poly::new_uninit(ring_qk.degree(), base_q.len() + 1) },
                        unsafe { Poly::new_uninit(ring_qk.degree(), base_q.len() + 1) },
                    ],
                    params_qk.clone(),
                    true,
                );
                keygen_profile::trace_step(&format!("galois.ksw[{}]", i), || {
                    self.build_kswitch_key_into(
                        &sk_rot,
                        i,
                        sk_qk_shoup.as_ref(),
                        &ring_qk,
                        &base_q,
                        qk,
                        &params_qk,
                        false,
                        error_scalar,
                        &mut ct,
                    );
                });
                elements.push(ct);
            }
            keys.insert(k, EvaluationKey { elements });
        }

        let out = GaloisKey { keys };
        keygen_profile::dump_and_reset(profile_label);
        Ok(out)
    }

    pub fn galois_keys_hps(
        &mut self,
        sk: &SecretKey,
        substitution_indices: &[u32],
    ) -> Result<GaloisKey, RnsError> {
        self.galois_keys_hps_with_error_scalar(sk, substitution_indices, None, "galois_keys_hps")
    }

    pub fn galois_keys_hps_scaled_error(
        &mut self,
        sk: &SecretKey,
        substitution_indices: &[u32],
        error_scalar: u64,
    ) -> Result<GaloisKey, RnsError> {
        self.galois_keys_hps_with_error_scalar(
            sk,
            substitution_indices,
            Some(error_scalar),
            "galois_keys_hps_scaled_error",
        )
    }

    fn galois_keys_hps_with_error_scalar(
        &mut self,
        sk: &SecretKey,
        substitution_indices: &[u32],
        error_scalar: Option<u64>,
        profile_label: &'static str,
    ) -> Result<GaloisKey, RnsError> {
        let _trace = keygen_profile::trace_begin(profile_label);
        let base_q = self.params.rns_tool.base_q().clone();
        let ring_q = self.params.ring.clone();
        let params_q = self.params.clone();
        let size_q = base_q.len();

        let sk_q_shoup = self.sk_q_shoup_cached(sk, &ring_q);
        let sk_q_shoup = Arc::clone(&sk_q_shoup);
        let garner_factors = Self::compute_hps_garner_factors(&base_q);

        let mut keys: HashMap<u32, EvaluationKey> = HashMap::new();
        for &k in substitution_indices {
            let map = ring_q.automorphism_map(k as u64);
            let mut sk_rot = sk.value.clone();
            keygen_profile::time_if_enabled(keygen_profile::Kind::GaloisRotate, || {
                sk_rot.apply_automorphism(&map);
            });

            let mut elements = Vec::with_capacity(size_q);
            for i in 0..size_q {
                let mut ct = Ciphertext::new(
                    vec![
                        unsafe { Poly::new_uninit(ring_q.degree(), base_q.len()) },
                        unsafe { Poly::new_uninit(ring_q.degree(), base_q.len()) },
                    ],
                    params_q.clone(),
                    true,
                );
                keygen_profile::trace_step(&format!("galois_hps.ksw[{}]", i), || {
                    self.build_kswitch_key_hps_into(
                        &sk_rot,
                        i,
                        sk_q_shoup.as_ref(),
                        &ring_q,
                        &base_q,
                        &params_q,
                        false,
                        error_scalar,
                        &mut ct,
                        &garner_factors,
                    );
                });
                elements.push(ct);
            }
            keys.insert(k, EvaluationKey { elements });
        }

        let out = GaloisKey { keys };
        keygen_profile::dump_and_reset(profile_label);
        Ok(out)
    }
}

pub struct Encryptor {
    params: EncryptionParams,
    seal_rng: Blake2xbRng,
}

impl Encryptor {
    pub fn new(params: EncryptionParams, rng: SecureRng) -> Self {
        let mut seed = [0u8; 64];
        let mut seed_rng = rng;
        seed_rng.fill_bytes(&mut seed);
        let seal_rng = Blake2xbRng::from_seed_bytes(seed);
        Self { params, seal_rng }
    }

    /// Encrypts zero: (b, a) = (-a*s + e, a)
    pub fn encrypt_zero_symmetric(&mut self, sk: &SecretKey) -> Ciphertext {
        self.encrypt_zero_symmetric_with_seed(sk, false)
    }

    pub fn encrypt_zero_symmetric_scaled_error(
        &mut self,
        sk: &SecretKey,
        error_scalar: u64,
    ) -> Ciphertext {
        let mut ct = self.encrypt_zero_symmetric(sk);
        for poly in ct.data.iter_mut() {
            scalar_mul_poly_inplace(poly, &self.params.ring, error_scalar);
        }
        ct
    }

    /// Encrypts zero under a public key using fresh public-encryption randomness.
    pub fn encrypt_zero_public(&mut self, pk: &PublicKey) -> Ciphertext {
        self.encrypt_zero_public_scaled_error(pk, 1)
    }

    /// Encrypts zero under a public key and scales the fresh public-encryption
    /// error by `error_scalar`.
    pub fn encrypt_zero_public_scaled_error(
        &mut self,
        pk: &PublicKey,
        error_scalar: u64,
    ) -> Ciphertext {
        assert_eq!(
            pk.pk.data.len(),
            2,
            "public key must have two RLWE components"
        );

        let degree = self.params.ring.degree();
        let moduli = self.params.ring.rns().moduli();
        let num_moduli = moduli.len();

        for component in &pk.pk.data {
            assert_eq!(component.degree(), degree, "public key degree mismatch");
            assert_eq!(
                component.num_moduli(),
                num_moduli,
                "public key modulus count mismatch"
            );
        }

        let mut pk0 = pk.pk.data[0].clone();
        let mut pk1 = pk.pk.data[1].clone();
        if !pk.pk.is_ntt {
            pk0.ntt_forward(&self.params.ring);
            pk1.ntt_forward(&self.params.ring);
        }

        let mut u_poly = unsafe { Poly::new_uninit(degree, num_moduli) };
        sample_poly_cbd_seal_blake(&mut self.seal_rng, moduli, degree, u_poly.data_mut());
        u_poly.ntt_forward(&self.params.ring);

        let mut e0_poly = unsafe { Poly::new_uninit(degree, num_moduli) };
        let mut e1_poly = unsafe { Poly::new_uninit(degree, num_moduli) };
        sample_poly_cbd_seal_blake(&mut self.seal_rng, moduli, degree, e0_poly.data_mut());
        sample_poly_cbd_seal_blake(&mut self.seal_rng, moduli, degree, e1_poly.data_mut());
        e0_poly.ntt_forward(&self.params.ring);
        e1_poly.ntt_forward(&self.params.ring);

        if error_scalar != 1 {
            scalar_mul_poly_inplace(&mut e0_poly, &self.params.ring, error_scalar);
            scalar_mul_poly_inplace(&mut e1_poly, &self.params.ring, error_scalar);
        }

        let mut c0 = u_poly.clone();
        c0.mul_assign(&pk0, &self.params.ring);
        c0.add_assign(&e0_poly, &self.params.ring);

        let mut c1 = u_poly;
        c1.mul_assign(&pk1, &self.params.ring);
        c1.add_assign(&e1_poly, &self.params.ring);

        Ciphertext::new(vec![c0, c1], self.params.clone(), true)
    }

    /// Encrypts zero with optional seeded-a (save_seed=true matches SEAL's serializable mode).
    pub fn encrypt_zero_symmetric_with_seed(
        &mut self,
        sk: &SecretKey,
        save_seed: bool,
    ) -> Ciphertext {
        let degree = self.params.ring.degree();
        let moduli = self.params.ring.rns().moduli();
        let num_moduli = moduli.len();

        // 1. Sample 'a' uniformly in NTT domain
        let mut a_poly = unsafe { Poly::new_uninit(degree, num_moduli) };
        let (seed, is_seeded) = if save_seed {
            let mut seed_key = [0u8; 64];
            self.seal_rng.fill_bytes(&mut seed_key);
            let mut a_rng = Blake2xbRng::from_seed_bytes(seed_key);
            sample_poly_uniform_seal_blake(&mut a_rng, moduli, degree, a_poly.data_mut());
            (Some(seed_key), true)
        } else {
            sample_poly_uniform_seal_blake(&mut self.seal_rng, moduli, degree, a_poly.data_mut());
            (None, false)
        };

        // 2. Sample 'e' (Error) in Coefficient Domain and transform to NTT
        let mut e_poly = unsafe { Poly::new_uninit(degree, num_moduli) };
        sample_poly_cbd_seal_blake(&mut self.seal_rng, moduli, degree, e_poly.data_mut());
        e_poly.ntt_forward(&self.params.ring);

        // 3. Compute b = -(a * s + e) (all in NTT)
        let mut b_poly = unsafe { Poly::new_uninit(degree, num_moduli) };
        // Use only Q part of SK if it's larger (QP)
        let sk_data = sk.value.data();
        let sk_data_q = &sk_data[..degree * num_moduli];
        // Clone Q part into a Poly for mul_assign
        let mut sk_poly_q = unsafe { Poly::new_uninit(degree, num_moduli) };
        sk_poly_q.data_mut().copy_from_slice(sk_data_q);

        {
            let moduli = self.params.ring.rns().moduli();
            for i in 0..num_moduli {
                let modulus = &moduli[i];
                let a_limb = a_poly.limb(i);
                let s_limb = sk_poly_q.limb(i);
                let b_limb = b_poly.limb_mut(i);
                for j in 0..degree {
                    b_limb[j] = silent_math::arith::mul_mod(a_limb[j], s_limb[j], modulus);
                }
            }
        }
        b_poly.add_assign(&e_poly, &self.params.ring); // a * s + e
        negate_poly_inplace(&mut b_poly, &self.params.ring);

        // Return (b, a)
        let mut ct = Ciphertext::new(vec![b_poly, a_poly], self.params.clone(), true);
        ct.seed = seed;
        ct.is_seeded_a = is_seeded;
        ct
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use silent_math::modulus::Modulus;
    use silent_math::rns::RnsBase;

    #[test]
    fn encryption_params_from_config_builds_rns_tool() {
        let base_q = RnsBase::from_values(vec![17, 97]).expect("base q");
        let base_p = RnsBase::from_values(vec![73]).expect("base p");
        let base_t = Modulus::new(5).expect("base t");
        let config = RnsToolConfig::new(base_q.clone(), base_t)
            .with_base_p(base_p)
            .enable_openfhe_approx_scale();

        let params = EncryptionParams::from_rns_config(8, config).expect("params");

        assert_eq!(params.ring.degree(), 8);
        assert_eq!(params.ring.rns().moduli(), base_q.moduli());
        assert_eq!(params.rns_tool.base_q().moduli(), base_q.moduli());
    }

    #[test]
    fn encryption_params_rejects_mismatched_base() {
        let base_q = RnsBase::from_values(vec![17, 97]).expect("base q");
        let base_q_alt = RnsBase::from_values(vec![19, 89]).expect("base q alt");
        let base_t = Modulus::new(5).expect("base t");

        let ring = RingContext::new(8, base_q);
        let rns_tool = RnsTool::new(base_q_alt, base_t, None).expect("tool");
        let err = EncryptionParams::new(ring, rns_tool).expect_err("mismatch");

        assert_eq!(err, RnsError::InvalidBase);
    }

    #[test]
    fn keygen_and_encryption_works() {
        let base_q = RnsBase::from_values(vec![17, 97]).expect("base q");
        // base_p can be empty for simple test if not used in KeyGen
        let base_p = RnsBase::from_values(vec![73]).expect("base p");
        let base_t = Modulus::new(5).expect("base t");
        let config = RnsToolConfig::new(base_q.clone(), base_t)
            .with_base_p(base_p)
            .enable_openfhe_approx_scale();

        let params = EncryptionParams::from_rns_config(8, config).expect("params");
        let rng = SecureRng::from_seed([42u8; 32]);

        let mut keygen = KeyGenerator::new(params.clone(), rng.clone());
        let sk = keygen.secret_key();

        // Check SK is not empty/zero (probabilistic)
        // SK in NTT domain should have non-zero coeffs
        let sk_data = sk.value.data();
        assert!(
            sk_data.iter().any(|&x| x != 0),
            "Secret key should not be all zeros"
        );

        let mut encryptor = Encryptor::new(params.clone(), rng);
        let ct = encryptor.encrypt_zero_symmetric(&sk);

        // Verify Ciphertext structure
        assert_eq!(ct.data.len(), 2);
        assert_eq!(ct.is_ntt, true);

        // Verify Decryption logic manually ( Homomorphic Decryption )
        // Dec(ct) = b + a*s
        // ct = (b, a)
        // result = b + a*s = (-a*s + e) + a*s = e
        let mut result = ct.data[0].clone(); // b
        let mut a_s = ct.data[1].clone(); // a
        a_s.mul_assign(&sk.value, &params.ring); // a * s
        result.add_assign(&a_s, &params.ring); // b + a*s

        // Transform back to Coeff domain to check noise size
        result.ntt_inverse(&params.ring);

        // Check noise is small
        // With q=17, noise should be small (e.g. 0, 1, -1 -> 16).
        // Standard dev 3.2. Max noise approx 10-20.
        // q=17 is too small for standard noise! We might wrap around.
        // Let's check against q=97.
        // For q=17, noise could be anything mod 17 basically.
        // For q=97, noise should be small.
        let modulus = base_q.moduli()[1]; // q=97
        let coeffs_q97 = result.limb(1);
        for &val in coeffs_q97 {
            // Centered noise: if val > q/2, val - q is negative noise.
            let noise = if val > modulus.value() / 2 {
                (modulus.value() - val) as i64
            } else {
                val as i64
            };
            assert!(noise.abs() < 20, "Noise too large: {}", noise);
        }
    }

    #[test]
    fn public_key_scaled_encryption_of_zero_decrypts_mod_plaintext() {
        let base_q = RnsBase::from_values(vec![257, 769]).expect("base q");
        let base_t = Modulus::new(17).expect("base t");
        let config = RnsToolConfig::new(base_q.clone(), base_t);
        let params = EncryptionParams::from_rns_config(8, config).expect("params");

        let mut keygen = KeyGenerator::new(params.clone(), SecureRng::from_seed([43u8; 32]));
        let sk = keygen.secret_key();
        let pk = keygen.public_key_scaled_error(&sk, 17);

        let mut encryptor = Encryptor::new(params.clone(), SecureRng::from_seed([44u8; 32]));
        let ct = encryptor.encrypt_zero_public_scaled_error(&pk, 17);

        assert_eq!(ct.data.len(), 2);
        assert!(ct.is_ntt);

        let mut phase = ct.data[0].clone();
        let mut c1s = ct.data[1].clone();
        c1s.mul_assign(&sk.value, &params.ring);
        phase.add_assign(&c1s, &params.ring);
        phase.ntt_inverse(&params.ring);

        let q = base_q.moduli()[1].value();
        let q_half = q >> 1;
        let q_mod_t = q % 17;
        for &coeff in phase.limb(1) {
            let reduced = if coeff > q_half {
                silent_math::arith::sub_mod(coeff % 17, q_mod_t, 17)
            } else {
                coeff % 17
            };
            assert_eq!(reduced, 0);
        }
    }
}
