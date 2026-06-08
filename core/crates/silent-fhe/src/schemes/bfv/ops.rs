use crate::core::evaluator::HeEvaluator;
use crate::schemes::bfv::params::{BfvMulMethod, BfvParameters};
use silent_math::arith;
use silent_math::rns_tool::RnsTool;
use silent_ring::{Poly, PolyShoup};
use silent_rlwe::{Ciphertext, GaloisKey};

use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DyadicMode {
    Standard,
    Split,
    Karatsuba,
}

fn dyadic_mode() -> DyadicMode {
    static MODE: OnceLock<DyadicMode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("SILENT_BFV_DYADIC") {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            if value == "karatsuba" || value == "k" {
                DyadicMode::Karatsuba
            } else if value == "split" || value == "seal" || value == "s" {
                DyadicMode::Split
            } else {
                DyadicMode::Standard
            }
        }
        Err(_) => DyadicMode::Standard,
    })
}

#[inline(always)]
unsafe fn dyadic_product_barrett(
    mut a_ptr: *const u64,
    mut b_ptr: *const u64,
    mut r_ptr: *mut u64,
    degree: usize,
    modulus_value: u64,
    ratio0: u64,
    ratio1: u64,
) {
    for _ in 0..degree {
        unsafe {
            *r_ptr = silent_math::arith::mul_mod_barrett_u64(
                *a_ptr,
                *b_ptr,
                modulus_value,
                ratio0,
                ratio1,
            );
            a_ptr = a_ptr.add(1);
            b_ptr = b_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
    }
}

#[inline(always)]
unsafe fn dyadic_product_add_barrett(
    mut a_ptr: *const u64,
    mut b_ptr: *const u64,
    mut r_ptr: *mut u64,
    degree: usize,
    modulus_value: u64,
    ratio0: u64,
    ratio1: u64,
) {
    for _ in 0..degree {
        unsafe {
            let t = silent_math::arith::mul_mod_barrett_u64(
                *a_ptr,
                *b_ptr,
                modulus_value,
                ratio0,
                ratio1,
            );
            *r_ptr = silent_math::arith::add_mod(*r_ptr, t, modulus_value);
            a_ptr = a_ptr.add(1);
            b_ptr = b_ptr.add(1);
            r_ptr = r_ptr.add(1);
        }
    }
}

mod profile {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    pub enum Kind {
        RelinModUp,
        RelinModUpIntt,
        RelinModUpSwitch,
        RelinModUpNtt,
        RelinDot,
        RelinModDown,
        RotateModUp,
        RotateModUpIntt,
        RotateModUpSwitch,
        RotateModUpNtt,
        RotateDot,
        RotateModDown,
        MulExpand,
        MulExpandCv1Intt,
        MulExpandCv1Q2P,
        MulExpandCv1NttP,
        MulExpandCv2Intt,
        MulExpandCv2Q2P,
        MulExpandCv2P2Q,
        MulExpandCv2NttQ,
        MulExpandCv2NttP,
        MulMult,
        MulMultR0,
        MulMultR1,
        MulMultR2,
        MulScale,
        MulScaleInttQp,
        MulScaleQ2P,
        MulScaleKP,
        MulScaleP2Q,
        MulScaleRound,
        MulScaleCombine,
        MulScaleNttQ,
        MulBehzExpand,
        MulBehzNttQ,
        MulBehzQToBskMtilde,
        MulBehzMrq,
        MulBehzNttBsk,
        MulBehzDyadic,
        MulBehzDyadicQ,
        MulBehzDyadicBsk,
        MulBehzInttQ,
        MulBehzInttBsk,
        MulBehzMulT,
        MulBehzFloor,
        MulBehzBaseConvSk,
        MulBehzNttQFinal,
        KeySwitchC2Intt,
        KeySwitchC2Ntt,
        KeySwitchPrecompute,
        KeySwitchDot,
        KeySwitchModDown,
        KeySwitchModDownInvNtt,
        KeySwitchModDownCore,
        KeySwitchQpBaseConv,
        KeySwitchQpNttP,
        KeySwitchQpMul,
        KeySwitchQpModDown,
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
    }

    static ENABLED: OnceLock<bool> = OnceLock::new();
    static RELIN_MOD_UP: Totals = Totals::new();
    static RELIN_MOD_UP_INTT: Totals = Totals::new();
    static RELIN_MOD_UP_SWITCH: Totals = Totals::new();
    static RELIN_MOD_UP_NTT: Totals = Totals::new();
    static RELIN_DOT: Totals = Totals::new();
    static RELIN_MOD_DOWN: Totals = Totals::new();
    static ROTATE_MOD_UP: Totals = Totals::new();
    static ROTATE_MOD_UP_INTT: Totals = Totals::new();
    static ROTATE_MOD_UP_SWITCH: Totals = Totals::new();
    static ROTATE_MOD_UP_NTT: Totals = Totals::new();
    static ROTATE_DOT: Totals = Totals::new();
    static ROTATE_MOD_DOWN: Totals = Totals::new();
    static MUL_EXPAND: Totals = Totals::new();
    static MUL_EXPAND_CV1_INTT: Totals = Totals::new();
    static MUL_EXPAND_CV1_Q2P: Totals = Totals::new();
    static MUL_EXPAND_CV1_NTT_P: Totals = Totals::new();
    static MUL_EXPAND_CV2_INTT: Totals = Totals::new();
    static MUL_EXPAND_CV2_Q2P: Totals = Totals::new();
    static MUL_EXPAND_CV2_P2Q: Totals = Totals::new();
    static MUL_EXPAND_CV2_NTT_Q: Totals = Totals::new();
    static MUL_EXPAND_CV2_NTT_P: Totals = Totals::new();
    static MUL_MULT: Totals = Totals::new();
    static MUL_MULT_R0: Totals = Totals::new();
    static MUL_MULT_R1: Totals = Totals::new();
    static MUL_MULT_R2: Totals = Totals::new();
    static MUL_SCALE: Totals = Totals::new();
    static MUL_SCALE_INTT_QP: Totals = Totals::new();
    static MUL_SCALE_Q2P: Totals = Totals::new();
    static MUL_SCALE_KP: Totals = Totals::new();
    static MUL_SCALE_P2Q: Totals = Totals::new();
    static MUL_SCALE_ROUND: Totals = Totals::new();
    static MUL_SCALE_COMBINE: Totals = Totals::new();
    static MUL_SCALE_NTT_Q: Totals = Totals::new();
    static MUL_BEHZ_EXPAND: Totals = Totals::new();
    static MUL_BEHZ_NTT_Q: Totals = Totals::new();
    static MUL_BEHZ_Q_TO_BSK_MTILDE: Totals = Totals::new();
    static MUL_BEHZ_MRQ: Totals = Totals::new();
    static MUL_BEHZ_NTT_BSK: Totals = Totals::new();
    static MUL_BEHZ_DYADIC: Totals = Totals::new();
    static MUL_BEHZ_DYADIC_Q: Totals = Totals::new();
    static MUL_BEHZ_DYADIC_BSK: Totals = Totals::new();
    static MUL_BEHZ_INTT_Q: Totals = Totals::new();
    static MUL_BEHZ_INTT_BSK: Totals = Totals::new();
    static MUL_BEHZ_MUL_T: Totals = Totals::new();
    static MUL_BEHZ_FLOOR: Totals = Totals::new();
    static MUL_BEHZ_BASE_CONV_SK: Totals = Totals::new();
    static MUL_BEHZ_NTT_Q_FINAL: Totals = Totals::new();
    static KEYSWITCH_C2_INTT: Totals = Totals::new();
    static KEYSWITCH_C2_NTT: Totals = Totals::new();
    static KEYSWITCH_PRECOMPUTE: Totals = Totals::new();
    static KEYSWITCH_DOT: Totals = Totals::new();
    static KEYSWITCH_MOD_DOWN: Totals = Totals::new();
    static KEYSWITCH_MOD_DOWN_INV_NTT: Totals = Totals::new();
    static KEYSWITCH_MOD_DOWN_CORE: Totals = Totals::new();
    static KEYSWITCH_QP_BASECONV: Totals = Totals::new();
    static KEYSWITCH_QP_NTT_P: Totals = Totals::new();
    static KEYSWITCH_QP_MUL: Totals = Totals::new();
    static KEYSWITCH_QP_MOD_DOWN: Totals = Totals::new();

    fn enabled() -> bool {
        *ENABLED.get_or_init(|| std::env::var("SILENT_PROFILE").is_ok())
    }

    fn totals(kind: Kind) -> &'static Totals {
        match kind {
            Kind::RelinModUp => &RELIN_MOD_UP,
            Kind::RelinModUpIntt => &RELIN_MOD_UP_INTT,
            Kind::RelinModUpSwitch => &RELIN_MOD_UP_SWITCH,
            Kind::RelinModUpNtt => &RELIN_MOD_UP_NTT,
            Kind::RelinDot => &RELIN_DOT,
            Kind::RelinModDown => &RELIN_MOD_DOWN,
            Kind::RotateModUp => &ROTATE_MOD_UP,
            Kind::RotateModUpIntt => &ROTATE_MOD_UP_INTT,
            Kind::RotateModUpSwitch => &ROTATE_MOD_UP_SWITCH,
            Kind::RotateModUpNtt => &ROTATE_MOD_UP_NTT,
            Kind::RotateDot => &ROTATE_DOT,
            Kind::RotateModDown => &ROTATE_MOD_DOWN,
            Kind::MulExpand => &MUL_EXPAND,
            Kind::MulExpandCv1Intt => &MUL_EXPAND_CV1_INTT,
            Kind::MulExpandCv1Q2P => &MUL_EXPAND_CV1_Q2P,
            Kind::MulExpandCv1NttP => &MUL_EXPAND_CV1_NTT_P,
            Kind::MulExpandCv2Intt => &MUL_EXPAND_CV2_INTT,
            Kind::MulExpandCv2Q2P => &MUL_EXPAND_CV2_Q2P,
            Kind::MulExpandCv2P2Q => &MUL_EXPAND_CV2_P2Q,
            Kind::MulExpandCv2NttQ => &MUL_EXPAND_CV2_NTT_Q,
            Kind::MulExpandCv2NttP => &MUL_EXPAND_CV2_NTT_P,
            Kind::MulMult => &MUL_MULT,
            Kind::MulMultR0 => &MUL_MULT_R0,
            Kind::MulMultR1 => &MUL_MULT_R1,
            Kind::MulMultR2 => &MUL_MULT_R2,
            Kind::MulScale => &MUL_SCALE,
            Kind::MulScaleInttQp => &MUL_SCALE_INTT_QP,
            Kind::MulScaleQ2P => &MUL_SCALE_Q2P,
            Kind::MulScaleKP => &MUL_SCALE_KP,
            Kind::MulScaleP2Q => &MUL_SCALE_P2Q,
            Kind::MulScaleRound => &MUL_SCALE_ROUND,
            Kind::MulScaleCombine => &MUL_SCALE_COMBINE,
            Kind::MulScaleNttQ => &MUL_SCALE_NTT_Q,
            Kind::MulBehzExpand => &MUL_BEHZ_EXPAND,
            Kind::MulBehzNttQ => &MUL_BEHZ_NTT_Q,
            Kind::MulBehzQToBskMtilde => &MUL_BEHZ_Q_TO_BSK_MTILDE,
            Kind::MulBehzMrq => &MUL_BEHZ_MRQ,
            Kind::MulBehzNttBsk => &MUL_BEHZ_NTT_BSK,
            Kind::MulBehzDyadic => &MUL_BEHZ_DYADIC,
            Kind::MulBehzDyadicQ => &MUL_BEHZ_DYADIC_Q,
            Kind::MulBehzDyadicBsk => &MUL_BEHZ_DYADIC_BSK,
            Kind::MulBehzInttQ => &MUL_BEHZ_INTT_Q,
            Kind::MulBehzInttBsk => &MUL_BEHZ_INTT_BSK,
            Kind::MulBehzMulT => &MUL_BEHZ_MUL_T,
            Kind::MulBehzFloor => &MUL_BEHZ_FLOOR,
            Kind::MulBehzBaseConvSk => &MUL_BEHZ_BASE_CONV_SK,
            Kind::MulBehzNttQFinal => &MUL_BEHZ_NTT_Q_FINAL,
            Kind::KeySwitchC2Intt => &KEYSWITCH_C2_INTT,
            Kind::KeySwitchC2Ntt => &KEYSWITCH_C2_NTT,
            Kind::KeySwitchPrecompute => &KEYSWITCH_PRECOMPUTE,
            Kind::KeySwitchDot => &KEYSWITCH_DOT,
            Kind::KeySwitchModDown => &KEYSWITCH_MOD_DOWN,
            Kind::KeySwitchModDownInvNtt => &KEYSWITCH_MOD_DOWN_INV_NTT,
            Kind::KeySwitchModDownCore => &KEYSWITCH_MOD_DOWN_CORE,
            Kind::KeySwitchQpBaseConv => &KEYSWITCH_QP_BASECONV,
            Kind::KeySwitchQpNttP => &KEYSWITCH_QP_NTT_P,
            Kind::KeySwitchQpMul => &KEYSWITCH_QP_MUL,
            Kind::KeySwitchQpModDown => &KEYSWITCH_QP_MOD_DOWN,
        }
    }

    pub struct Scope {
        kind: Kind,
        start: Option<Instant>,
    }

    impl Scope {
        pub fn new(kind: Kind) -> Self {
            if enabled() {
                Self {
                    kind,
                    start: Some(Instant::now()),
                }
            } else {
                Self { kind, start: None }
            }
        }
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            if let Some(start) = self.start.take() {
                let elapsed = start.elapsed();
                let totals = totals(self.kind);
                totals
                    .ns
                    .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
                totals.count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn dump_line(name: &str, totals: &Totals) {
        let count = totals.count.load(Ordering::Relaxed);
        if count == 0 {
            return;
        }
        let ns = totals.ns.load(Ordering::Relaxed);
        let avg_ns = ns / count;
        eprintln!(
            "[profile] {:<16} count={} total_ms={:.3} avg_us={:.3}",
            name,
            count,
            ns as f64 / 1_000_000.0,
            avg_ns as f64 / 1_000.0
        );
    }

    pub fn dump() {
        if !enabled() {
            return;
        }
        eprintln!("[profile] BFV relinearize/rotate breakdown");
        dump_line("relin.mod_up", &RELIN_MOD_UP);
        dump_line("relin.mod_up_intt", &RELIN_MOD_UP_INTT);
        dump_line("relin.mod_up_switch", &RELIN_MOD_UP_SWITCH);
        dump_line("relin.mod_up_ntt", &RELIN_MOD_UP_NTT);
        dump_line("relin.dot", &RELIN_DOT);
        dump_line("relin.mod_down", &RELIN_MOD_DOWN);
        dump_line("rotate.mod_up", &ROTATE_MOD_UP);
        dump_line("rotate.mod_up_intt", &ROTATE_MOD_UP_INTT);
        dump_line("rotate.mod_up_switch", &ROTATE_MOD_UP_SWITCH);
        dump_line("rotate.mod_up_ntt", &ROTATE_MOD_UP_NTT);
        dump_line("rotate.dot", &ROTATE_DOT);
        dump_line("rotate.mod_down", &ROTATE_MOD_DOWN);
        dump_line("mul.expand", &MUL_EXPAND);
        dump_line("mul.expand.cv1_intt", &MUL_EXPAND_CV1_INTT);
        dump_line("mul.expand.cv1_q2p", &MUL_EXPAND_CV1_Q2P);
        dump_line("mul.expand.cv1_ntt_p", &MUL_EXPAND_CV1_NTT_P);
        dump_line("mul.expand.cv2_intt", &MUL_EXPAND_CV2_INTT);
        dump_line("mul.expand.cv2_q2p", &MUL_EXPAND_CV2_Q2P);
        dump_line("mul.expand.cv2_p2q", &MUL_EXPAND_CV2_P2Q);
        dump_line("mul.expand.cv2_ntt_q", &MUL_EXPAND_CV2_NTT_Q);
        dump_line("mul.expand.cv2_ntt_p", &MUL_EXPAND_CV2_NTT_P);
        dump_line("mul.mult", &MUL_MULT);
        dump_line("mul.mult.r0", &MUL_MULT_R0);
        dump_line("mul.mult.r1", &MUL_MULT_R1);
        dump_line("mul.mult.r2", &MUL_MULT_R2);
        dump_line("mul.scale", &MUL_SCALE);
        dump_line("mul.scale.intt_qp", &MUL_SCALE_INTT_QP);
        dump_line("mul.scale.q2p", &MUL_SCALE_Q2P);
        dump_line("mul.scale.kp", &MUL_SCALE_KP);
        dump_line("mul.scale.p2q", &MUL_SCALE_P2Q);
        dump_line("mul.scale.round", &MUL_SCALE_ROUND);
        dump_line("mul.scale.combine", &MUL_SCALE_COMBINE);
        dump_line("mul.scale.ntt_q", &MUL_SCALE_NTT_Q);
        dump_line("behz.expand", &MUL_BEHZ_EXPAND);
        dump_line("behz.ntt_q_in", &MUL_BEHZ_NTT_Q);
        dump_line("behz.q_to_bsk_mtilde", &MUL_BEHZ_Q_TO_BSK_MTILDE);
        dump_line("behz.mrq", &MUL_BEHZ_MRQ);
        dump_line("behz.ntt_bsk", &MUL_BEHZ_NTT_BSK);
        dump_line("behz.dyadic", &MUL_BEHZ_DYADIC);
        dump_line("behz.dyadic_q", &MUL_BEHZ_DYADIC_Q);
        dump_line("behz.dyadic_bsk", &MUL_BEHZ_DYADIC_BSK);
        dump_line("behz.intt_q", &MUL_BEHZ_INTT_Q);
        dump_line("behz.intt_bsk", &MUL_BEHZ_INTT_BSK);
        dump_line("behz.mul_t", &MUL_BEHZ_MUL_T);
        dump_line("behz.floor", &MUL_BEHZ_FLOOR);
        dump_line("behz.baseconv_sk", &MUL_BEHZ_BASE_CONV_SK);
        dump_line("behz.ntt_q_final", &MUL_BEHZ_NTT_Q_FINAL);
        eprintln!("[profile] BFV key switch breakdown");
        dump_line("ks.c2_intt", &KEYSWITCH_C2_INTT);
        dump_line("ks.c2_ntt", &KEYSWITCH_C2_NTT);
        dump_line("ks.precompute", &KEYSWITCH_PRECOMPUTE);
        dump_line("ks.dot", &KEYSWITCH_DOT);
        dump_line("ks.mod_down", &KEYSWITCH_MOD_DOWN);
        dump_line("ks.mod_down_invntt", &KEYSWITCH_MOD_DOWN_INV_NTT);
        dump_line("ks.mod_down_core", &KEYSWITCH_MOD_DOWN_CORE);
        dump_line("ks.qp.baseconv", &KEYSWITCH_QP_BASECONV);
        dump_line("ks.qp.ntt_p", &KEYSWITCH_QP_NTT_P);
        dump_line("ks.qp.mul", &KEYSWITCH_QP_MUL);
        dump_line("ks.qp.mod_down", &KEYSWITCH_QP_MOD_DOWN);
    }
}

/// Pre-allocated scratch buffers for relinearize/rotate operations
#[derive(Debug)]
struct RelinScratch {
    // Temp poly in QP (for base conversion/NTT)
    t_qp: Poly,    // size_qp * degree
    prod_qp: Poly, // size_qp * degree (multiplication result)

    // Temp poly in Q (for c2_coeff)
    temp_q: Poly, // size_q * degree

    // Scratch for RNS operations
    mod_scratch: Vec<u64>,
    digit_conv_scratch: Vec<u64>,
}

#[derive(Debug)]
struct KeySwitchScratch {
    c2_coeff: Poly,
    c2_ntt: Poly,
    prod0: Poly,
    prod1: Poly,
    t_ntt_all: Vec<u64>,
    tmp: Vec<u64>,
    tmp2: Vec<u64>,
    acc0: Vec<u128>,
    acc1: Vec<u128>,
}

#[derive(Debug)]
struct KeySwitchModDownCache {
    inv_qk_shoup: Vec<silent_math::arith::MultiplyUIntModOperand>,
    qk_half_mod_q: Vec<u64>,
}

impl KeySwitchScratch {
    fn new(degree: usize, size_q: usize) -> Self {
        let size_qk = size_q + 1;
        Self {
            c2_coeff: Poly::new(degree, size_q),
            c2_ntt: Poly::new(degree, size_q),
            prod0: Poly::new(degree, size_qk),
            prod1: Poly::new(degree, size_qk),
            t_ntt_all: vec![0u64; degree * size_q * size_qk],
            tmp: vec![0u64; degree],
            tmp2: vec![0u64; degree],
            acc0: vec![0u128; degree],
            acc1: vec![0u128; degree],
        }
    }
}

impl RelinScratch {
    fn new(degree: usize, size_q: usize, size_p: usize) -> Self {
        let size_qp = size_q + size_p;
        // Scratch size estimation
        let mod_scratch_size = 2 * (size_p * degree + size_p * 128);

        Self {
            t_qp: Poly::new(degree, size_qp),
            prod_qp: Poly::new(degree, size_qp),
            temp_q: Poly::new(degree, size_q),
            mod_scratch: vec![0u64; mod_scratch_size],
            digit_conv_scratch: vec![0u64; size_q * 128], // Block size
        }
    }
}

#[derive(Debug)]
struct MulScratch {
    // Primary workspace for BFV Multiplication
    // Layout:
    // [0..2*len]: cv1_qp (2 polys)
    // [2*len..4*len]: cv2_qp (2 polys)
    // [4*len..7*len]: result_qp (3 polys)
    // where len = degree * size_qp
    workspace: Vec<u64>,

    // Auxiliary scratch for RnsTool and Scaling ops
    // Unions multiple scratch needs:
    // - conv_scratch
    // - scale_round_scratch
    // - k_p, k_q, x_q_p buffers
    misc_scratch: Vec<u64>,

    // Precomputed HPS tables (cached constants)
    // Corresponds to "P" base constants
    q_inv_mod_p: Vec<u64>,
}

#[derive(Debug)]
struct MulBehzScratch {
    in_q_ntt: Vec<u64>,     // 4 * size_q * degree (c1[0], c1[1], c2[0], c2[1])
    in_bsk_ntt: Vec<u64>,   // 4 * size_bsk * degree (c1[0], c1[1], c2[0], c2[1])
    out_q_ntt: Vec<u64>,    // 3 * size_q * degree
    out_bsk_ntt: Vec<u64>,  // 3 * size_bsk * degree
    tmp_q: Vec<u64>,        // size_q * degree
    floor_in: Vec<u64>,     // (size_q + size_bsk) * degree
    bsk_mtilde: Vec<u64>,   // (size_bsk + 1) * degree
    work_scratch: Vec<u64>, // shared scratch for BEHZ base conversions
    t_mod_q: Vec<arith::MultiplyUIntModOperand>,
    t_mod_bsk: Vec<arith::MultiplyUIntModOperand>,
    t_cached: u64,
    size_q_cached: usize,
    size_bsk_cached: usize,
}

impl MulBehzScratch {
    fn new(degree: usize, size_q: usize, size_bsk: usize, size_b: usize) -> Self {
        let poly_q_len = degree * size_q;
        let poly_bsk_len = degree * size_bsk;
        let conv_needed = size_q * degree + degree;
        let floor_needed = size_q * degree;
        let sk_needed = 2 * degree + size_b;
        let scratch_len = conv_needed.max(floor_needed).max(sk_needed);
        Self {
            in_q_ntt: vec![0u64; 4 * poly_q_len],
            in_bsk_ntt: vec![0u64; 4 * poly_bsk_len],
            out_q_ntt: vec![0u64; 3 * poly_q_len],
            out_bsk_ntt: vec![0u64; 3 * poly_bsk_len],
            tmp_q: vec![0u64; poly_q_len],
            floor_in: vec![0u64; (size_q + size_bsk) * degree],
            bsk_mtilde: vec![0u64; (size_bsk + 1) * degree],
            work_scratch: vec![0u64; scratch_len],
            t_mod_q: Vec::new(),
            t_mod_bsk: Vec::new(),
            t_cached: 0,
            size_q_cached: 0,
            size_bsk_cached: 0,
        }
    }

    fn ensure(&mut self, degree: usize, size_q: usize, size_bsk: usize, size_b: usize) {
        let poly_q_len = degree * size_q;
        let poly_bsk_len = degree * size_bsk;
        let conv_needed = size_q * degree + degree;
        let floor_needed = size_q * degree;
        let sk_needed = 2 * degree + size_b;
        let scratch_len = conv_needed.max(floor_needed).max(sk_needed);
        if self.in_q_ntt.len() != 4 * poly_q_len
            || self.in_bsk_ntt.len() != 4 * poly_bsk_len
            || self.out_q_ntt.len() != 3 * poly_q_len
            || self.out_bsk_ntt.len() != 3 * poly_bsk_len
            || self.tmp_q.len() != poly_q_len
            || self.floor_in.len() != (size_q + size_bsk) * degree
            || self.bsk_mtilde.len() != (size_bsk + 1) * degree
            || self.work_scratch.len() != scratch_len
        {
            *self = Self::new(degree, size_q, size_bsk, size_b);
        }
    }

    fn ensure_t(
        &mut self,
        t: u64,
        q_moduli: &[silent_math::modulus::Modulus],
        bsk_moduli: &[silent_math::modulus::Modulus],
    ) {
        if self.t_cached == t
            && self.size_q_cached == q_moduli.len()
            && self.size_bsk_cached == bsk_moduli.len()
        {
            return;
        }
        self.t_cached = t;
        self.size_q_cached = q_moduli.len();
        self.size_bsk_cached = bsk_moduli.len();
        self.t_mod_q = q_moduli
            .iter()
            .map(|m| arith::MultiplyUIntModOperand::new(t % m.value(), m.value()))
            .collect();
        self.t_mod_bsk = bsk_moduli
            .iter()
            .map(|m| arith::MultiplyUIntModOperand::new(t % m.value(), m.value()))
            .collect();
    }
}

impl MulScratch {
    fn new(
        degree: usize,
        size_q: usize,
        size_p: usize,
        base_q: &silent_math::rns::RnsBase,
        base_p: &silent_math::rns::RnsBase,
        _t: u64,
    ) -> Self {
        let size_qp = size_q + size_p;
        // Workspace for cv1 (2), cv2 (2), result (3) in QP basis
        let workspace_size = 7 * degree * size_qp;

        // Misc scratch logic:
        // We split misc_scratch into 3 chunks for parallel execution (line 747).
        // Each chunk must satisfy the split requirements in lines 756-762:
        // - term2_buf: degree
        // - scale_round_scratch: degree * size_q
        // - x_q_p: degree * size_p
        // - k_p: degree * size_p
        // - k_q: degree * size_q
        // - scratch_small_q: size_q
        // - scratch_small_p: size_p
        // Total per thread = degree * (1 + 2*size_q + 2*size_p) + size_q + size_p

        let per_thread_size = degree * (1 + 2 * size_q + 2 * size_p) + size_q + size_p;

        // We also need to support the Expand phase (lines 550-662).
        // Expand splits misc_scratch into 2 chunks.
        // Chunk 1 (Expand cv1) needs:
        // - misc_convert: degree * size_q
        // - scratch_small_q: size_q
        // Chunk 2 (Expand cv2) needs:
        // - misc_convert: degree * size_q
        // - conv_scratch: max(degree * size_p, size_q * BaseConverter::BLOCK_SIZE)
        //   (approx_switch_crt_basis_into uses BaseConverter scratch sized by base_q)
        // - scratch_small_p: size_p
        // Expand total ~ degree * (2*size_q + size_p) + size_q + size_p.
        //
        // Ensure misc_scratch can satisfy both scaling and expand requirements.
        let conv_scratch_len = std::cmp::max(
            degree * size_p,
            size_q * silent_math::rns::BaseConverter::BLOCK_SIZE,
        );
        let expand_need = degree * size_q + conv_scratch_len + size_p;
        // 3 * per_thread_size for scaling is usually larger than Expand requirements,
        // but small degrees with larger size_q can violate that. Guard with max.
        let misc_size = std::cmp::max(3 * per_thread_size, 2 * expand_need);

        // Precompute constants for P base (HpsPOverQ)
        let q_big = base_q.base_prod_u128().unwrap();
        // Q^-1 mod p_i
        let q_inv_mod_p: Vec<u64> = base_p
            .moduli()
            .iter()
            .map(|m| {
                let mv = m.value();
                let q_mod = (q_big % (mv as u128)) as u64;
                silent_math::numth::mod_inverse(q_mod, mv).expect("Q inv P")
            })
            .collect();

        Self {
            workspace: vec![0u64; workspace_size],
            misc_scratch: vec![0u64; misc_size],
            q_inv_mod_p,
        }
    }
}

pub struct BfvEvaluator {
    params: BfvParameters,
    // Cached QP ring context for relinearize/rotate (lazy initialized)
    ring_qp: OnceCell<silent_ring::RingContext>,
    // Cached QR ring context for HPS mul (lazy initialized)
    ring_qr: OnceCell<silent_ring::RingContext>,
    // Cached QK ring context for SEAL-style keyswitching (lazy initialized)
    ring_qk: OnceCell<silent_ring::RingContext>,
    // Cached Bsk ring context for BEHZ mul (lazy initialized)
    ring_bsk: OnceCell<silent_ring::RingContext>,
    // Cached scratch buffers
    relin_scratch: RefCell<Option<RelinScratch>>,
    mul_scratch: RefCell<Option<MulScratch>>,
    mul_behz_scratch: RefCell<Option<MulBehzScratch>>,
    rotate_scratch: RefCell<Vec<u64>>,
    keyswitch_shoup_cache: RefCell<HashMap<usize, (PolyShoup, PolyShoup)>>,
    kswitch_scratch: RefCell<Option<KeySwitchScratch>>,
    kswitch_key_shoup_cache: RefCell<HashMap<usize, Vec<(PolyShoup, PolyShoup)>>>,
    kswitch_key_shoup_cache_q: RefCell<HashMap<usize, Vec<(PolyShoup, PolyShoup)>>>,
    kswitch_moddown_cache: OnceCell<KeySwitchModDownCache>,
}

impl HeEvaluator for BfvEvaluator {
    type Context = BfvParameters;

    fn new(context: BfvParameters) -> Self {
        Self {
            params: context,
            ring_qp: OnceCell::new(),
            ring_qr: OnceCell::new(),
            ring_qk: OnceCell::new(),
            ring_bsk: OnceCell::new(),
            relin_scratch: RefCell::new(None),
            mul_scratch: RefCell::new(None),
            mul_behz_scratch: RefCell::new(None),
            rotate_scratch: RefCell::new(Vec::new()),
            keyswitch_shoup_cache: RefCell::new(HashMap::new()),
            kswitch_scratch: RefCell::new(None),
            kswitch_key_shoup_cache: RefCell::new(HashMap::new()),
            kswitch_key_shoup_cache_q: RefCell::new(HashMap::new()),
            kswitch_moddown_cache: OnceCell::new(),
        }
    }

    fn add(&self, c1: &Ciphertext, c2: &Ciphertext) -> Ciphertext {
        // Optimized: clone c1 once then modify in-place
        let mut result = c1.clone();
        let ring = self.params.ring();
        if c1.is_ntt != c2.is_ntt {
            // Align to coefficient domain
            if result.is_ntt {
                for poly in result.data.iter_mut() {
                    poly.ntt_inverse(ring);
                }
            }
            let mut c2_coeff = c2.clone();
            if c2_coeff.is_ntt {
                for poly in c2_coeff.data.iter_mut() {
                    poly.ntt_inverse(ring);
                }
            }
            for (p1, p2) in result.data.iter_mut().zip(c2_coeff.data.iter()) {
                p1.add_assign(p2, ring);
            }
            result.is_ntt = false;
            return result;
        }

        for (p1, p2) in result.data.iter_mut().zip(c2.data.iter()) {
            p1.add_assign(p2, ring);
        }
        if result.is_ntt {
            for poly in result.data.iter_mut() {
                poly.ntt_inverse(ring);
            }
            result.is_ntt = false;
        }
        result
    }

    fn sub(&self, c1: &Ciphertext, c2: &Ciphertext) -> Ciphertext {
        let mut result = c1.clone();
        let ring = self.params.ring();
        if c1.is_ntt != c2.is_ntt {
            if result.is_ntt {
                for poly in result.data.iter_mut() {
                    poly.ntt_inverse(ring);
                }
            }
            let mut c2_coeff = c2.clone();
            if c2_coeff.is_ntt {
                for poly in c2_coeff.data.iter_mut() {
                    poly.ntt_inverse(ring);
                }
            }
            for (p1, p2) in result.data.iter_mut().zip(c2_coeff.data.iter()) {
                p1.sub_assign(p2, ring);
            }
            result.is_ntt = false;
            return result;
        }

        for (p1, p2) in result.data.iter_mut().zip(c2.data.iter()) {
            p1.sub_assign(p2, ring);
        }
        if result.is_ntt {
            for poly in result.data.iter_mut() {
                poly.ntt_inverse(ring);
            }
            result.is_ntt = false;
        }
        result
    }

    fn mul(&self, c1: &Ciphertext, c2: &Ciphertext) -> Ciphertext {
        let ring_q = self.params.ring();
        let rns_tool = self.params.rns_tool();

        let mul_method = self.current_mul_method();

        match mul_method {
            BfvMulMethod::Hps => {
                return self.mul_hps(c1, c2, ring_q, rns_tool);
            }
            BfvMulMethod::Behz => {
                return self.mul_behz(c1, c2, ring_q, rns_tool);
            }
            BfvMulMethod::HpsPoverq | BfvMulMethod::HpsPoverqLeveled => {}
        }

        // BFV Multiplication using HPSPOVERQ (exactly like OpenFHE)
        // c_new = Round(t/P * c1 * c2) where P is auxiliary base
        // Uses expand_crt_basis (Q→Q*P) and approx_mod_down (Q*P→Q) with precomputed alpha
        let use_float_scale = matches!(mul_method, BfvMulMethod::HpsPoverqLeveled);

        // Check if P base is available (HPSPOVERQ path)
        if let Some(base_p) = rns_tool.base_p() {
            return self.mul_hpspoverq(c1, c2, ring_q, rns_tool, base_p, use_float_scale);
        }

        // Fallback to B base (legacy HPS path)
        let base_b = rns_tool.base_b();
        if base_b.is_none() {
            // No auxiliary basis - simple tensor product
            let mut p1_0 = c1.data[0].clone();
            let mut p1_1 = c1.data[1].clone();
            let mut p2_0 = c2.data[0].clone();
            let mut p2_1 = c2.data[1].clone();

            if !c1.is_ntt {
                p1_0.ntt_forward(ring_q);
                p1_1.ntt_forward(ring_q);
            }
            if !c2.is_ntt {
                p2_0.ntt_forward(ring_q);
                p2_1.ntt_forward(ring_q);
            }

            let mut c0 = p1_0.clone();
            c0.mul_assign(&p2_0, ring_q);
            let mut c2_poly = p1_1.clone();
            c2_poly.mul_assign(&p2_1, ring_q);
            let mut term1 = p1_0;
            term1.mul_assign(&p2_1, ring_q);
            let mut term2 = p1_1;
            term2.mul_assign(&p2_0, ring_q);
            term1.add_assign(&term2, ring_q);

            // Return coeff-domain ciphertext
            c0.ntt_inverse(ring_q);
            term1.ntt_inverse(ring_q);
            c2_poly.ntt_inverse(ring_q);

            return Ciphertext::new(
                vec![c0, term1, c2_poly],
                self.params.runtime_params().clone(),
                false,
            );
        }

        // If we are here, base_p is None but base_b is Some.
        // We do not support the legacy Base B path anymore as MulScratch is optimized for QP.
        panic!(
            "BfvEvaluator::mul: Legacy Base B path is removed. Please configure RnsTool with base_p for HPSPOVERQ."
        );
    }
}
impl BfvEvaluator {
    fn current_mul_method(&self) -> BfvMulMethod {
        match std::env::var("SILENT_BFV_MUL_METHOD") {
            Ok(value) => match value.to_ascii_lowercase().as_str() {
                "behz" => BfvMulMethod::Behz,
                "hps" => BfvMulMethod::Hps,
                "hpspoverq" | "hps_poverq" => BfvMulMethod::HpsPoverq,
                "hpspoverqleveled" | "hps_poverq_leveled" | "hpspoverq_leveled" => {
                    BfvMulMethod::HpsPoverqLeveled
                }
                other => {
                    eprintln!(
                        "[bfv] unknown SILENT_BFV_MUL_METHOD='{}', defaulting to {:?}",
                        other,
                        self.params.mul_method()
                    );
                    self.params.mul_method()
                }
            },
            Err(_) => self.params.mul_method(),
        }
    }

    pub fn dump_profile() {
        profile::dump();
        silent_math::rns_tool::RnsTool::dump_profile();
    }

    /// HPS Multiplication (baseline, correct)
    /// Performs QR-NTT multiplication, ScaleAndRound in R, then SwitchCRTBasis back to Q.
    fn mul_hps(
        &self,
        c1: &Ciphertext,
        c2: &Ciphertext,
        ring_q: &silent_ring::RingContext,
        rns_tool: &silent_math::rns_tool::RnsTool,
    ) -> Ciphertext {
        let degree = ring_q.degree();
        let size_q = ring_q.rns().len();
        let base_r = rns_tool
            .base_r()
            .expect("HPS requires base_r in RnsToolConfig");
        let size_r = base_r.len();
        let size_qr = size_q + size_r;

        let ring_qr = self.ring_qr.get_or_init(|| {
            let mut moduli_qr = ring_q.rns().moduli().to_vec();
            moduli_qr.extend_from_slice(base_r.moduli());
            let base_qr = silent_math::rns::RnsBase::new(moduli_qr).expect("QR base");
            silent_ring::RingContext::new(degree, base_qr)
        });

        let poly_len = degree * size_qr;
        let mut cv1_qr = vec![0u64; 2 * poly_len];
        let mut cv2_qr = vec![0u64; 2 * poly_len];
        let mut res_qr = vec![0u64; 3 * poly_len];

        let mut temp_q = vec![0u64; degree * size_q];
        let mut scratch_q = vec![0u64; size_q];

        let c1_is_ntt = c1.is_ntt;
        let c2_is_ntt = c2.is_ntt;

        for i in 0..2 {
            let start = i * poly_len;
            let poly_data = &mut cv1_qr[start..start + poly_len];
            let (q_part, r_part) = poly_data.split_at_mut(degree * size_q);

            let mut coeff_poly = c1.data[i].clone();
            if c1_is_ntt {
                coeff_poly.ntt_inverse(ring_q);
            }
            temp_q.copy_from_slice(coeff_poly.data());

            if c1_is_ntt {
                q_part.copy_from_slice(c1.data[i].data());
            } else {
                q_part.copy_from_slice(&temp_q);
                for limb in 0..size_q {
                    let slice = &mut q_part[limb * degree..(limb + 1) * degree];
                    let tables = &ring_q.ntt_tables()[limb];
                    silent_math::ntt::ntt_forward(slice, tables);
                }
            }

            rns_tool
                .switch_crt_basis_r_into(&temp_q, degree, r_part, &mut scratch_q)
                .expect("switch q->r");

            for limb in 0..size_r {
                let slice = &mut r_part[limb * degree..(limb + 1) * degree];
                let tables = &ring_qr.ntt_tables()[size_q + limb];
                silent_math::ntt::ntt_forward(slice, tables);
            }
        }

        for i in 0..2 {
            let start = i * poly_len;
            let poly_data = &mut cv2_qr[start..start + poly_len];
            let (q_part, r_part) = poly_data.split_at_mut(degree * size_q);

            let mut coeff_poly = c2.data[i].clone();
            if c2_is_ntt {
                coeff_poly.ntt_inverse(ring_q);
            }
            temp_q.copy_from_slice(coeff_poly.data());

            if c2_is_ntt {
                q_part.copy_from_slice(c2.data[i].data());
            } else {
                q_part.copy_from_slice(&temp_q);
                for limb in 0..size_q {
                    let slice = &mut q_part[limb * degree..(limb + 1) * degree];
                    let tables = &ring_q.ntt_tables()[limb];
                    silent_math::ntt::ntt_forward(slice, tables);
                }
            }

            rns_tool
                .switch_crt_basis_r_into(&temp_q, degree, r_part, &mut scratch_q)
                .expect("switch q->r");

            for limb in 0..size_r {
                let slice = &mut r_part[limb * degree..(limb + 1) * degree];
                let tables = &ring_qr.ntt_tables()[size_q + limb];
                silent_math::ntt::ntt_forward(slice, tables);
            }
        }

        let qr_moduli = ring_qr.rns().moduli();
        {
            // r0 = cv1[0] * cv2[0]
            let a = &cv1_qr[0..poly_len];
            let b = &cv2_qr[0..poly_len];
            let r0 = &mut res_qr[0..poly_len];
            for limb in 0..size_qr {
                let modulus = &qr_moduli[limb];
                let ratio = modulus.const_ratio();
                let ratio0 = ratio[0];
                let ratio1 = ratio[1];
                let modulus_value = modulus.value();
                let start = limb * degree;
                for j in 0..degree {
                    r0[start + j] = silent_math::arith::mul_mod_barrett_u64(
                        a[start + j],
                        b[start + j],
                        modulus_value,
                        ratio0,
                        ratio1,
                    );
                }
            }
        }
        {
            // r2 = cv1[1] * cv2[1]
            let a = &cv1_qr[poly_len..2 * poly_len];
            let b = &cv2_qr[poly_len..2 * poly_len];
            let r2 = &mut res_qr[2 * poly_len..3 * poly_len];
            for limb in 0..size_qr {
                let modulus = &qr_moduli[limb];
                let ratio = modulus.const_ratio();
                let ratio0 = ratio[0];
                let ratio1 = ratio[1];
                let modulus_value = modulus.value();
                let start = limb * degree;
                for j in 0..degree {
                    r2[start + j] = silent_math::arith::mul_mod_barrett_u64(
                        a[start + j],
                        b[start + j],
                        modulus_value,
                        ratio0,
                        ratio1,
                    );
                }
            }
        }
        {
            // r1 = cv1[0] * cv2[1] + cv1[1] * cv2[0]
            let a1 = &cv1_qr[0..poly_len];
            let b1 = &cv2_qr[poly_len..2 * poly_len];
            let a2 = &cv1_qr[poly_len..2 * poly_len];
            let b2 = &cv2_qr[0..poly_len];
            let r1 = &mut res_qr[poly_len..2 * poly_len];
            for limb in 0..size_qr {
                let modulus = &qr_moduli[limb];
                let ratio = modulus.const_ratio();
                let ratio0 = ratio[0];
                let ratio1 = ratio[1];
                let modulus_val = modulus.value();
                let start = limb * degree;
                for j in 0..degree {
                    let t1 = silent_math::arith::mul_mod_barrett_u64(
                        a1[start + j],
                        b1[start + j],
                        modulus_val,
                        ratio0,
                        ratio1,
                    );
                    let t2 = silent_math::arith::mul_mod_barrett_u64(
                        a2[start + j],
                        b2[start + j],
                        modulus_val,
                        ratio0,
                        ratio1,
                    );
                    r1[start + j] = silent_math::arith::add_mod(t1, t2, modulus_val);
                }
            }
        }

        // INTT on QR results
        for poly_idx in 0..3 {
            let start = poly_idx * poly_len;
            let poly = &mut res_qr[start..start + poly_len];
            for limb in 0..size_qr {
                let slice = &mut poly[limb * degree..(limb + 1) * degree];
                let tables = &ring_qr.ntt_tables()[limb];
                silent_math::ntt::ntt_inverse(slice, tables);
            }
        }

        let mut result_polys = vec![Poly::new(degree, size_q); 3];
        let mut r_buf = vec![0u64; degree * size_r];
        let mut scratch_r = vec![0u64; size_r];

        for idx in 0..3 {
            let src = &res_qr[idx * poly_len..(idx + 1) * poly_len];
            rns_tool
                .hps_scale_and_round_qr_to_r_into(src, degree, &mut r_buf)
                .expect("hps scale");
            let out_q = result_polys[idx].data_mut();
            rns_tool
                .switch_crt_basis_r_to_q_into(&r_buf, degree, out_q, &mut scratch_r)
                .expect("r->q");
        }

        Ciphertext::new(result_polys, self.params.runtime_params().clone(), false)
    }

    /// HPSPOVERQ Multiplication - exactly like OpenFHE
    /// Uses P base (from SwitchTables) with precomputed alpha tables

    fn mul_hpspoverq(
        &self,
        c1: &Ciphertext,
        c2: &Ciphertext,
        ring_q: &silent_ring::RingContext,
        rns_tool: &silent_math::rns_tool::RnsTool,
        base_p: &silent_math::rns::RnsBase,
        use_float_scale: bool,
    ) -> Ciphertext {
        let degree = ring_q.degree();
        let t = self.params.plain_modulus();
        let c1_is_ntt = c1.is_ntt;
        let c2_is_ntt = c2.is_ntt;

        let size_q = ring_q.rns().len();
        let size_p = base_p.len();
        let size_qp = size_q + size_p;

        // Create Q*P ring context (cached in evaluator ideally)
        let ring_qp = self.ring_qp.get_or_init(|| {
            let mut moduli_qp = ring_q.rns().moduli().to_vec();
            moduli_qp.extend_from_slice(base_p.moduli());
            let base_qp = silent_math::rns::RnsBase::new(moduli_qp).expect("QP base");
            silent_ring::RingContext::new(degree, base_qp)
        });

        // Initialize scratch if needed
        let mut borrowed = self.mul_scratch.borrow_mut();
        if borrowed.is_none() {
            *borrowed = Some(MulScratch::new(
                degree,
                size_q,
                size_p,
                ring_q.rns(),
                base_p,
                t,
            ));
        }
        let scratch = borrowed.as_mut().unwrap();

        // Slices from workspace (Manual logic: [cv1 (2)][cv2 (2)][res (3)])
        // total size = 7 * degree * size_qp
        let poly_len = degree * size_qp;
        let (cv1_slice, rest1) = scratch.workspace.split_at_mut(2 * poly_len);
        let (cv2_slice, res_slice) = rest1.split_at_mut(2 * poly_len);

        // Misc scratch subdivision - Moved to local scopes for parallelism
        // let (misc_convert, misc_rest) = scratch.misc_scratch.split_at_mut(2 * degree * size_qp);
        // ...

        // Helper to access slices by polygon index - Using macros or inline logic to avoid closure capture issues
        // We will just inline the indexing.

        // 1. Expand cv1 & cv2 (Parallel)
        {
            let _scope = profile::Scope::new(profile::Kind::MulExpand);

            // Sequential expansion
            // Expand cv1 logic
            // Need: temp_q (deg*Q), scratch_small_q (Q), scratch_small_p (P)
            // Reuse scratch normally. We can just use the first chunk of scratch if sequential.
            let scratch_len = scratch.misc_scratch.len();
            let (scratch_cv1, _) = scratch.misc_scratch.split_at_mut(scratch_len / 2);
            // Wait, we need enough scratch. Original parallel split scratch in half. Sequential can reuse FULL scratch?
            // Actually, `misc_scratch` allocation in `MulScratch::new` assumes 3 parallel threads for scaling.
            // For Expand, it checked Requirements.
            // Let's just use the first half as before to be safe/simple, or the whole thing.
            let (misc_convert, misc_rest) = scratch_cv1.split_at_mut(degree * size_q);
            let (scratch_small_q, _) = misc_rest.split_at_mut(size_q);

            for i in 0..2 {
                let start = i * poly_len;
                let poly_data = unsafe {
                    std::slice::from_raw_parts_mut(cv1_slice.as_mut_ptr().add(start), poly_len)
                };
                let (q_part, p_part) = poly_data.split_at_mut(degree * size_q);

                // Compute coeff-domain temp_q and Q-part in NTT
                let temp_q = &mut misc_convert[..degree * size_q];
                temp_q.copy_from_slice(c1.data[i].data());
                if c1_is_ntt {
                    // temp_q = INTT(c1)
                    let _scope = profile::Scope::new(profile::Kind::MulExpandCv1Intt);
                    for limb in 0..size_q {
                        let slice = &mut temp_q[limb * degree..(limb + 1) * degree];
                        let tables = &ring_q.ntt_tables()[limb];
                        silent_math::ntt::ntt_inverse(slice, tables);
                    }
                    // Q-part is already NTT
                    q_part.copy_from_slice(c1.data[i].data());
                } else {
                    // Q-part = NTT(coeff)
                    q_part.copy_from_slice(temp_q);
                    for limb in 0..size_q {
                        let slice = &mut q_part[limb * degree..(limb + 1) * degree];
                        let tables = &ring_q.ntt_tables()[limb];
                        silent_math::ntt::ntt_forward(slice, tables);
                    }
                }

                {
                    let _scope = profile::Scope::new(profile::Kind::MulExpandCv1Q2P);
                    rns_tool
                        .switch_crt_basis_into(temp_q, degree, p_part, scratch_small_q)
                        .expect("switch cv1");
                }

                {
                    let _scope = profile::Scope::new(profile::Kind::MulExpandCv1NttP);
                    for limb in 0..size_p {
                        let slice = &mut p_part[limb * degree..(limb + 1) * degree];
                        let tables = &ring_qp.ntt_tables()[size_q + limb];
                        silent_math::ntt::ntt_forward_lazy(slice, tables);
                    }
                }
            }

            // Expand cv2 logic
            // Reuse scratch_cv1 (since sequential now) or stick to cv2 split?
            // Reusing is better for cache but logic is already written for cv2 split.
            // Note: `scratch_cv2` was `scratch.misc_scratch` second half.
            // Let's just run it sequentially on same scratch if safely reusable, or distinct.
            // Using same scratch variable `scratch_cv1` is fine since we are done with cv1.
            let (misc_convert, misc_rest) = scratch_cv1.split_at_mut(degree * size_q);
            let conv_scratch_len = std::cmp::max(
                degree * size_p,
                size_q * silent_math::rns::BaseConverter::BLOCK_SIZE,
            );
            let (conv_scratch, rest2) = misc_rest.split_at_mut(conv_scratch_len);
            let (scratch_small_p, _) = rest2.split_at_mut(size_p);

            for i in 0..2 {
                let start = i * poly_len;
                let poly_data = unsafe {
                    std::slice::from_raw_parts_mut(cv2_slice.as_mut_ptr().add(start), poly_len)
                };
                let (q_part, p_part) = poly_data.split_at_mut(degree * size_q);

                let temp_q = &mut misc_convert[..degree * size_q];
                temp_q.copy_from_slice(c2.data[i].data());
                if c2_is_ntt {
                    let _scope = profile::Scope::new(profile::Kind::MulExpandCv2Intt);
                    for limb in 0..size_q {
                        let slice = &mut temp_q[limb * degree..(limb + 1) * degree];
                        let tables = &ring_q.ntt_tables()[limb];
                        silent_math::ntt::ntt_inverse(slice, tables);
                    }
                }

                {
                    let _scope = profile::Scope::new(profile::Kind::MulExpandCv2Q2P);
                    rns_tool
                        .approx_switch_crt_basis_into(temp_q, degree, p_part, conv_scratch)
                        .expect("fast expand q->p");
                }

                {
                    let _scope = profile::Scope::new(profile::Kind::MulExpandCv2P2Q);
                    rns_tool
                        .switch_crt_basis_p_to_q_into(
                            p_part,
                            degree,
                            q_part,
                            scratch_small_p,
                            false,
                        )
                        .expect("fast expand p->q");
                }

                {
                    let _scope = profile::Scope::new(profile::Kind::MulExpandCv2NttQ);
                    for limb in 0..size_q {
                        let slice = &mut q_part[limb * degree..(limb + 1) * degree];
                        let tables = &ring_q.ntt_tables()[limb];
                        silent_math::ntt::ntt_forward(slice, tables);
                    }
                }

                {
                    let _scope = profile::Scope::new(profile::Kind::MulExpandCv2NttP);
                    for limb in 0..size_p {
                        let slice = &mut p_part[limb * degree..(limb + 1) * degree];
                        let tables = &ring_qp.ntt_tables()[size_q + limb];
                        silent_math::ntt::ntt_forward_lazy(slice, tables);
                    }
                }
            }
        }

        // 3. Multiply in Q*P (tensor product)
        let qp_moduli = ring_qp.rns().moduli();

        {
            let _scope = profile::Scope::new(profile::Kind::MulMult);
            // r0 = cv1[0] * cv2[0]
            {
                let _scope = profile::Scope::new(profile::Kind::MulMultR0);
                let a = &cv1_slice[0..poly_len];
                let b = &cv2_slice[0..poly_len];
                let r0 = &mut res_slice[0..poly_len];
                for limb in 0..size_qp {
                    let modulus = &qp_moduli[limb];
                    let ratio = modulus.const_ratio();
                    let ratio0 = ratio[0];
                    let ratio1 = ratio[1];
                    let modulus_value = modulus.value();
                    let start = limb * degree;
                    for j in 0..degree {
                        r0[start + j] = silent_math::arith::mul_mod_barrett_u64(
                            a[start + j],
                            b[start + j],
                            modulus_value,
                            ratio0,
                            ratio1,
                        );
                    }
                }
            }

            // r2 = cv1[1] * cv2[1]
            {
                let _scope = profile::Scope::new(profile::Kind::MulMultR2);
                let a = &cv1_slice[poly_len..2 * poly_len];
                let b = &cv2_slice[poly_len..2 * poly_len];
                let r2 = &mut res_slice[2 * poly_len..3 * poly_len];
                for limb in 0..size_qp {
                    let modulus = &qp_moduli[limb];
                    let ratio = modulus.const_ratio();
                    let ratio0 = ratio[0];
                    let ratio1 = ratio[1];
                    let modulus_value = modulus.value();
                    let start = limb * degree;
                    for j in 0..degree {
                        r2[start + j] = silent_math::arith::mul_mod_barrett_u64(
                            a[start + j],
                            b[start + j],
                            modulus_value,
                            ratio0,
                            ratio1,
                        );
                    }
                }
            }

            // r1 = cv1[0] * cv2[1] + cv1[1] * cv2[0]
            {
                let _scope = profile::Scope::new(profile::Kind::MulMultR1);
                let a1 = &cv1_slice[0..poly_len];
                let b1 = &cv2_slice[poly_len..2 * poly_len];
                let a2 = &cv1_slice[poly_len..2 * poly_len];
                let b2 = &cv2_slice[0..poly_len];
                let r1 = &mut res_slice[poly_len..2 * poly_len];

                for limb in 0..size_qp {
                    let modulus = &qp_moduli[limb];
                    let ratio = modulus.const_ratio();
                    let ratio0 = ratio[0];
                    let ratio1 = ratio[1];
                    let modulus_val = modulus.value();
                    let start = limb * degree;
                    for j in 0..degree {
                        let t1 = silent_math::arith::mul_mod_barrett_u64(
                            a1[start + j],
                            b1[start + j],
                            modulus_val,
                            ratio0,
                            ratio1,
                        );
                        let t2 = silent_math::arith::mul_mod_barrett_u64(
                            a2[start + j],
                            b2[start + j],
                            modulus_val,
                            ratio0,
                            ratio1,
                        );
                        r1[start + j] = silent_math::arith::add_mod(t1, t2, modulus_val);
                    }
                }
            }
        }

        // Prepare result polys
        let mut result_polys = vec![Poly::new(degree, size_q); 3];

        // 4. Scale & Round
        {
            let _scope = profile::Scope::new(profile::Kind::MulScale);

            // 4. INTT result in Q*P (Sequential)
            {
                let _scope = profile::Scope::new(profile::Kind::MulScaleInttQp);
                res_slice.chunks_mut(poly_len).for_each(|poly| {
                    for limb in 0..size_qp {
                        let slice = &mut poly[limb * degree..(limb + 1) * degree];
                        let tables = &ring_qp.ntt_tables()[limb];
                        silent_math::ntt::ntt_inverse(slice, tables);
                    }
                });
            }

            // Extract constants
            let q_inv_mod_p = scratch.q_inv_mod_p.clone();

            // Prepare result polys
            // let mut result_polys = vec![Poly::new(degree, size_q); 3]; // Moved outside

            // Scaling (Sequential)
            // Need to zip result_polys with res_slice chunks
            // Use single chunk of scratch if sequential (safe?)
            // Or reuse chunking logic.
            let chunk_size = scratch.misc_scratch.len() / 3;
            let misc_chunks = scratch.misc_scratch.chunks_mut(chunk_size); // Changed from par_chunks_mut

            result_polys
                .iter_mut() // Changed from par_iter_mut
                .zip(res_slice.chunks_mut(poly_len)) // Changed from par_chunks_mut
                .zip(misc_chunks)
                .for_each(|((dest_poly, poly_data), thread_scratch)| {
                    // Parse thread scratch
                    // Need: term2_buf (deg), scale_round_scratch (deg*Q), x_q_p (deg*P), k_p (deg*P), k_q (deg*Q), scratch_smalls
                    let (term2_buf, rest1) = thread_scratch.split_at_mut(degree);
                    let (scale_round_scratch, rest2) = rest1.split_at_mut(degree * size_q);
                    let (x_q_p, rest3) = rest2.split_at_mut(degree * size_p);
                    let (k_p, rest4) = rest3.split_at_mut(degree * size_p);
                    let (k_q, rest5) = rest4.split_at_mut(degree * size_q);
                    let (scratch_small_q, rest6) = rest5.split_at_mut(size_q);
                    let (scratch_small_p, _) = rest6.split_at_mut(size_p);

                    let (x_q_data, x_p_data) = poly_data.split_at(degree * size_q);

                    // 1. Convert X_q to P basis -> X_q_p
                    {
                        let _scope = profile::Scope::new(profile::Kind::MulScaleQ2P);
                        rns_tool
                            .switch_crt_basis_into(x_q_data, degree, x_q_p, scratch_small_q)
                            .expect("switch Q->P");
                    }

                    // 2. Compute k_p = (X_p - X_q_p) * Q^-1 mod P
                    let p_moduli = base_p.moduli();
                    {
                        let _scope = profile::Scope::new(profile::Kind::MulScaleKP);
                        for limb in 0..size_p {
                            let modulus = &p_moduli[limb];
                            let modulus_val = modulus.value();
                            let q_inv = q_inv_mod_p[limb];
                            let start = limb * degree;
                            for j in 0..degree {
                                let diff = arith::sub_mod(
                                    x_p_data[start + j],
                                    x_q_p[start + j],
                                    modulus_val,
                                );
                                k_p[start + j] = arith::mul_mod(diff, q_inv, modulus);
                            }
                        }
                    }

                    // 3. Convert k_p to Q basis -> k_q
                    {
                        let _scope = profile::Scope::new(profile::Kind::MulScaleP2Q);
                        rns_tool
                            .switch_crt_basis_p_to_q_into(k_p, degree, k_q, scratch_small_p, false)
                            .expect("switch P->Q");
                    }

                    // 4-6. Result = k_q * t + ScaleAndRound(X_q)
                    {
                        let _scope = profile::Scope::new(profile::Kind::MulScaleRound);
                        if use_float_scale {
                            rns_tool
                                .scale_and_round_float_into(x_q_data, degree, term2_buf)
                                .expect("scale_and_round_float (enable_openfhe_scale)");
                        } else {
                            rns_tool
                                .scale_and_round_into(
                                    x_q_data,
                                    degree,
                                    term2_buf,
                                    scale_round_scratch,
                                )
                                .expect("scale_and_round");
                        }
                    }

                    let q_moduli = ring_q.rns().moduli();
                    let dest = dest_poly.data_mut();

                    {
                        let _scope = profile::Scope::new(profile::Kind::MulScaleCombine);
                        for limb in 0..size_q {
                            let modulus = &q_moduli[limb];
                            let modulus_val = modulus.value();
                            let t_mod = t % modulus_val;
                            let start = limb * degree;
                            for j in 0..degree {
                                let term1 = arith::mul_mod(k_q[start + j], t_mod, modulus);
                                dest[start + j] = arith::add_mod(term1, term2_buf[j], modulus_val);
                            }
                        }
                    }
                });
        }

        Ciphertext::new(result_polys, self.params.runtime_params().clone(), false)
    }

    /// BEHZ Multiplication (SEAL/OpenFHE integer path)
    /// Uses base Bsk with Montgomery reduction tables.
    fn mul_behz(
        &self,
        c1: &Ciphertext,
        c2: &Ciphertext,
        ring_q: &silent_ring::RingContext,
        rns_tool: &silent_math::rns_tool::RnsTool,
    ) -> Ciphertext {
        let base_bsk = rns_tool.base_bsk().unwrap_or_else(|| {
            panic!(
                "BFV BEHZ requires OpenFHE Bsk tables. Configure RnsToolConfig with \
                 with_openfhe_bsk(m_tilde, m_sk) and base_b."
            )
        });

        let degree = ring_q.degree();
        let size_q = ring_q.rns().len();
        let size_bsk = base_bsk.len();
        let size_b = rns_tool
            .base_b()
            .map(|base_b| base_b.len())
            .unwrap_or_else(|| panic!("BFV BEHZ requires base_b in RnsToolConfig"));
        let t = self.params.plain_modulus();
        let poly_q_len = degree * size_q;
        let poly_bsk_len = degree * size_bsk;

        let ring_bsk = self
            .ring_bsk
            .get_or_init(|| silent_ring::RingContext::new(degree, base_bsk.clone()));

        // Scratch
        let mut borrowed = self.mul_behz_scratch.borrow_mut();
        if borrowed.is_none() {
            *borrowed = Some(MulBehzScratch::new(degree, size_q, size_bsk, size_b));
        }
        let scratch = borrowed.as_mut().unwrap();
        scratch.ensure(degree, size_q, size_bsk, size_b);

        // 1) Convert inputs to base Bsk in coeff domain, then NTT (lazy) in Bsk.
        // Also prepare NTT-form inputs in base Q for dyadic multiplication.
        let inputs = [
            (&c1.data[0], c1.is_ntt),
            (&c1.data[1], c1.is_ntt),
            (&c2.data[0], c2.is_ntt),
            (&c2.data[1], c2.is_ntt),
        ];
        {
            let _scope = profile::Scope::new(profile::Kind::MulBehzExpand);
            for (idx, (poly, is_ntt)) in inputs.iter().enumerate() {
                // temp_q = coeff-domain input
                scratch.tmp_q.copy_from_slice(poly.data());
                if *is_ntt {
                    for limb in 0..size_q {
                        let slice = &mut scratch.tmp_q[limb * degree..(limb + 1) * degree];
                        let tables = &ring_q.ntt_tables()[limb];
                        silent_math::ntt::ntt_inverse(slice, tables);
                    }
                    // Q NTT input already available in poly
                    let out_q = &mut scratch.in_q_ntt[idx * poly_q_len..(idx + 1) * poly_q_len];
                    out_q.copy_from_slice(poly.data());
                } else {
                    // Build Q-NTT input from coeff-domain data
                    let out_q = &mut scratch.in_q_ntt[idx * poly_q_len..(idx + 1) * poly_q_len];
                    let _scope = profile::Scope::new(profile::Kind::MulBehzNttQ);
                    out_q.copy_from_slice(&scratch.tmp_q);
                    for limb in 0..size_q {
                        let slice = &mut out_q[limb * degree..(limb + 1) * degree];
                        let tables = &ring_q.ntt_tables()[limb];
                        silent_math::ntt::ntt_forward_lazy(slice, tables);
                    }
                }
                let out = &mut scratch.in_bsk_ntt[idx * poly_bsk_len..(idx + 1) * poly_bsk_len];
                {
                    let _scope = profile::Scope::new(profile::Kind::MulBehzQToBskMtilde);
                    rns_tool
                        .behz_convert_q_to_bsk_mtilde_into(
                            &scratch.tmp_q,
                            degree,
                            &mut scratch.bsk_mtilde,
                            &mut scratch.work_scratch,
                        )
                        .expect("BEHZ Q->Bsk+mt");
                }
                {
                    let _scope = profile::Scope::new(profile::Kind::MulBehzMrq);
                    rns_tool
                        .behz_montgomery_reduce_into(
                            &scratch.bsk_mtilde,
                            degree,
                            out,
                            &mut scratch.work_scratch,
                        )
                        .expect("BEHZ MRQ");
                }

                {
                    let _scope = profile::Scope::new(profile::Kind::MulBehzNttBsk);
                    for limb in 0..size_bsk {
                        let slice = &mut out[limb * degree..(limb + 1) * degree];
                        let tables = &ring_bsk.ntt_tables()[limb];
                        silent_math::ntt::ntt_forward_lazy(slice, tables);
                    }
                }
            }
        }

        // 2) Multiply in NTT domain for Q and Bsk bases
        let q_moduli = ring_q.rns().moduli();
        let bsk_moduli = ring_bsk.rns().moduli();

        let c1_q0 = &scratch.in_q_ntt[0..poly_q_len];
        let c1_q1 = &scratch.in_q_ntt[poly_q_len..2 * poly_q_len];
        let c2_q0 = &scratch.in_q_ntt[2 * poly_q_len..3 * poly_q_len];
        let c2_q1 = &scratch.in_q_ntt[3 * poly_q_len..4 * poly_q_len];

        let c1_bsk0 = &scratch.in_bsk_ntt[0..poly_bsk_len];
        let c1_bsk1 = &scratch.in_bsk_ntt[poly_bsk_len..2 * poly_bsk_len];
        let c2_bsk0 = &scratch.in_bsk_ntt[2 * poly_bsk_len..3 * poly_bsk_len];
        let c2_bsk1 = &scratch.in_bsk_ntt[3 * poly_bsk_len..4 * poly_bsk_len];

        let (r0_q, rest_q) = scratch.out_q_ntt.split_at_mut(poly_q_len);
        let (r1_q, r2_q) = rest_q.split_at_mut(poly_q_len);
        let (r0_bsk, rest_bsk) = scratch.out_bsk_ntt.split_at_mut(poly_bsk_len);
        let (r1_bsk, r2_bsk) = rest_bsk.split_at_mut(poly_bsk_len);

        {
            let _scope = profile::Scope::new(profile::Kind::MulBehzDyadic);
            match dyadic_mode() {
                DyadicMode::Standard => {
                    // Q base
                    {
                        let _scope = profile::Scope::new(profile::Kind::MulBehzDyadicQ);
                        for limb in 0..size_q {
                            let modulus = &q_moduli[limb];
                            let ratio = modulus.const_ratio();
                            let ratio0 = ratio[0];
                            let ratio1 = ratio[1];
                            let modulus_value = modulus.value();
                            let start = limb * degree;
                            unsafe {
                                let mut a0_ptr = c1_q0.as_ptr().add(start);
                                let mut a1_ptr = c1_q1.as_ptr().add(start);
                                let mut b0_ptr = c2_q0.as_ptr().add(start);
                                let mut b1_ptr = c2_q1.as_ptr().add(start);
                                let mut r0_ptr = r0_q.as_mut_ptr().add(start);
                                let mut r1_ptr = r1_q.as_mut_ptr().add(start);
                                let mut r2_ptr = r2_q.as_mut_ptr().add(start);
                                for _ in 0..degree {
                                    let a0 = *a0_ptr;
                                    let a1 = *a1_ptr;
                                    let b0 = *b0_ptr;
                                    let b1 = *b1_ptr;
                                    *r0_ptr = silent_math::arith::mul_mod_barrett_u64(
                                        a0,
                                        b0,
                                        modulus_value,
                                        ratio0,
                                        ratio1,
                                    );
                                    *r2_ptr = silent_math::arith::mul_mod_barrett_u64(
                                        a1,
                                        b1,
                                        modulus_value,
                                        ratio0,
                                        ratio1,
                                    );
                                    let t1 = silent_math::arith::mul_mod_barrett_u64(
                                        a0,
                                        b1,
                                        modulus_value,
                                        ratio0,
                                        ratio1,
                                    );
                                    let t2 = silent_math::arith::mul_mod_barrett_u64(
                                        a1,
                                        b0,
                                        modulus_value,
                                        ratio0,
                                        ratio1,
                                    );
                                    *r1_ptr = silent_math::arith::add_mod(t1, t2, modulus_value);
                                    a0_ptr = a0_ptr.add(1);
                                    a1_ptr = a1_ptr.add(1);
                                    b0_ptr = b0_ptr.add(1);
                                    b1_ptr = b1_ptr.add(1);
                                    r0_ptr = r0_ptr.add(1);
                                    r1_ptr = r1_ptr.add(1);
                                    r2_ptr = r2_ptr.add(1);
                                }
                            }
                        }
                    }

                    // Bsk base
                    {
                        let _scope = profile::Scope::new(profile::Kind::MulBehzDyadicBsk);
                        for limb in 0..size_bsk {
                            let modulus = &bsk_moduli[limb];
                            let ratio = modulus.const_ratio();
                            let ratio0 = ratio[0];
                            let ratio1 = ratio[1];
                            let modulus_value = modulus.value();
                            let start = limb * degree;
                            unsafe {
                                let mut a0_ptr = c1_bsk0.as_ptr().add(start);
                                let mut a1_ptr = c1_bsk1.as_ptr().add(start);
                                let mut b0_ptr = c2_bsk0.as_ptr().add(start);
                                let mut b1_ptr = c2_bsk1.as_ptr().add(start);
                                let mut r0_ptr = r0_bsk.as_mut_ptr().add(start);
                                let mut r1_ptr = r1_bsk.as_mut_ptr().add(start);
                                let mut r2_ptr = r2_bsk.as_mut_ptr().add(start);
                                for _ in 0..degree {
                                    let a0 = *a0_ptr;
                                    let a1 = *a1_ptr;
                                    let b0 = *b0_ptr;
                                    let b1 = *b1_ptr;
                                    *r0_ptr = silent_math::arith::mul_mod_barrett_u64(
                                        a0,
                                        b0,
                                        modulus_value,
                                        ratio0,
                                        ratio1,
                                    );
                                    *r2_ptr = silent_math::arith::mul_mod_barrett_u64(
                                        a1,
                                        b1,
                                        modulus_value,
                                        ratio0,
                                        ratio1,
                                    );
                                    let t1 = silent_math::arith::mul_mod_barrett_u64(
                                        a0,
                                        b1,
                                        modulus_value,
                                        ratio0,
                                        ratio1,
                                    );
                                    let t2 = silent_math::arith::mul_mod_barrett_u64(
                                        a1,
                                        b0,
                                        modulus_value,
                                        ratio0,
                                        ratio1,
                                    );
                                    *r1_ptr = silent_math::arith::add_mod(t1, t2, modulus_value);
                                    a0_ptr = a0_ptr.add(1);
                                    a1_ptr = a1_ptr.add(1);
                                    b0_ptr = b0_ptr.add(1);
                                    b1_ptr = b1_ptr.add(1);
                                    r0_ptr = r0_ptr.add(1);
                                    r1_ptr = r1_ptr.add(1);
                                    r2_ptr = r2_ptr.add(1);
                                }
                            }
                        }
                    }
                }
                DyadicMode::Split => {
                    // Q base (SEAL-style separate passes)
                    {
                        let _scope = profile::Scope::new(profile::Kind::MulBehzDyadicQ);
                        for limb in 0..size_q {
                            let modulus = &q_moduli[limb];
                            let ratio = modulus.const_ratio();
                            let ratio0 = ratio[0];
                            let ratio1 = ratio[1];
                            let modulus_value = modulus.value();
                            let start = limb * degree;
                            unsafe {
                                let a0_ptr = c1_q0.as_ptr().add(start);
                                let a1_ptr = c1_q1.as_ptr().add(start);
                                let b0_ptr = c2_q0.as_ptr().add(start);
                                let b1_ptr = c2_q1.as_ptr().add(start);
                                let r0_ptr = r0_q.as_mut_ptr().add(start);
                                let r1_ptr = r1_q.as_mut_ptr().add(start);
                                let r2_ptr = r2_q.as_mut_ptr().add(start);
                                dyadic_product_barrett(
                                    a0_ptr,
                                    b0_ptr,
                                    r0_ptr,
                                    degree,
                                    modulus_value,
                                    ratio0,
                                    ratio1,
                                );
                                dyadic_product_barrett(
                                    a1_ptr,
                                    b1_ptr,
                                    r2_ptr,
                                    degree,
                                    modulus_value,
                                    ratio0,
                                    ratio1,
                                );
                                dyadic_product_barrett(
                                    a0_ptr,
                                    b1_ptr,
                                    r1_ptr,
                                    degree,
                                    modulus_value,
                                    ratio0,
                                    ratio1,
                                );
                                dyadic_product_add_barrett(
                                    a1_ptr,
                                    b0_ptr,
                                    r1_ptr,
                                    degree,
                                    modulus_value,
                                    ratio0,
                                    ratio1,
                                );
                            }
                        }
                    }

                    // Bsk base (SEAL-style separate passes)
                    {
                        let _scope = profile::Scope::new(profile::Kind::MulBehzDyadicBsk);
                        for limb in 0..size_bsk {
                            let modulus = &bsk_moduli[limb];
                            let ratio = modulus.const_ratio();
                            let ratio0 = ratio[0];
                            let ratio1 = ratio[1];
                            let modulus_value = modulus.value();
                            let start = limb * degree;
                            unsafe {
                                let a0_ptr = c1_bsk0.as_ptr().add(start);
                                let a1_ptr = c1_bsk1.as_ptr().add(start);
                                let b0_ptr = c2_bsk0.as_ptr().add(start);
                                let b1_ptr = c2_bsk1.as_ptr().add(start);
                                let r0_ptr = r0_bsk.as_mut_ptr().add(start);
                                let r1_ptr = r1_bsk.as_mut_ptr().add(start);
                                let r2_ptr = r2_bsk.as_mut_ptr().add(start);
                                dyadic_product_barrett(
                                    a0_ptr,
                                    b0_ptr,
                                    r0_ptr,
                                    degree,
                                    modulus_value,
                                    ratio0,
                                    ratio1,
                                );
                                dyadic_product_barrett(
                                    a1_ptr,
                                    b1_ptr,
                                    r2_ptr,
                                    degree,
                                    modulus_value,
                                    ratio0,
                                    ratio1,
                                );
                                dyadic_product_barrett(
                                    a0_ptr,
                                    b1_ptr,
                                    r1_ptr,
                                    degree,
                                    modulus_value,
                                    ratio0,
                                    ratio1,
                                );
                                dyadic_product_add_barrett(
                                    a1_ptr,
                                    b0_ptr,
                                    r1_ptr,
                                    degree,
                                    modulus_value,
                                    ratio0,
                                    ratio1,
                                );
                            }
                        }
                    }
                }
                DyadicMode::Karatsuba => {
                    // Q base
                    {
                        let _scope = profile::Scope::new(profile::Kind::MulBehzDyadicQ);
                        for limb in 0..size_q {
                            let modulus = &q_moduli[limb];
                            let ratio = modulus.const_ratio();
                            let ratio0 = ratio[0];
                            let ratio1 = ratio[1];
                            let modulus_value = modulus.value();
                            let two_mod = modulus_value << 1;
                            let start = limb * degree;
                            unsafe {
                                let mut a0_ptr = c1_q0.as_ptr().add(start);
                                let mut a1_ptr = c1_q1.as_ptr().add(start);
                                let mut b0_ptr = c2_q0.as_ptr().add(start);
                                let mut b1_ptr = c2_q1.as_ptr().add(start);
                                let mut r0_ptr = r0_q.as_mut_ptr().add(start);
                                let mut r1_ptr = r1_q.as_mut_ptr().add(start);
                                let mut r2_ptr = r2_q.as_mut_ptr().add(start);
                                for _ in 0..degree {
                                    let a0 = *a0_ptr;
                                    let a1 = *a1_ptr;
                                    let b0 = *b0_ptr;
                                    let b1 = *b1_ptr;

                                    let mut s0 = a0.wrapping_add(a1);
                                    let s0_mask = ((s0 >= two_mod) as u64).wrapping_neg();
                                    s0 = s0.wrapping_sub(two_mod & s0_mask);
                                    let mut s1 = b0.wrapping_add(b1);
                                    let s1_mask = ((s1 >= two_mod) as u64).wrapping_neg();
                                    s1 = s1.wrapping_sub(two_mod & s1_mask);

                                    let r0 = silent_math::arith::mul_mod_barrett_u64(
                                        a0,
                                        b0,
                                        modulus_value,
                                        ratio0,
                                        ratio1,
                                    );
                                    let r2 = silent_math::arith::mul_mod_barrett_u64(
                                        a1,
                                        b1,
                                        modulus_value,
                                        ratio0,
                                        ratio1,
                                    );
                                    let m = silent_math::arith::mul_mod_barrett_u64(
                                        s0,
                                        s1,
                                        modulus_value,
                                        ratio0,
                                        ratio1,
                                    );
                                    let r1 = silent_math::arith::sub_mod(
                                        silent_math::arith::sub_mod(m, r0, modulus_value),
                                        r2,
                                        modulus_value,
                                    );

                                    *r0_ptr = r0;
                                    *r1_ptr = r1;
                                    *r2_ptr = r2;
                                    a0_ptr = a0_ptr.add(1);
                                    a1_ptr = a1_ptr.add(1);
                                    b0_ptr = b0_ptr.add(1);
                                    b1_ptr = b1_ptr.add(1);
                                    r0_ptr = r0_ptr.add(1);
                                    r1_ptr = r1_ptr.add(1);
                                    r2_ptr = r2_ptr.add(1);
                                }
                            }
                        }
                    }

                    // Bsk base
                    {
                        let _scope = profile::Scope::new(profile::Kind::MulBehzDyadicBsk);
                        for limb in 0..size_bsk {
                            let modulus = &bsk_moduli[limb];
                            let ratio = modulus.const_ratio();
                            let ratio0 = ratio[0];
                            let ratio1 = ratio[1];
                            let modulus_value = modulus.value();
                            let two_mod = modulus_value << 1;
                            let start = limb * degree;
                            unsafe {
                                let mut a0_ptr = c1_bsk0.as_ptr().add(start);
                                let mut a1_ptr = c1_bsk1.as_ptr().add(start);
                                let mut b0_ptr = c2_bsk0.as_ptr().add(start);
                                let mut b1_ptr = c2_bsk1.as_ptr().add(start);
                                let mut r0_ptr = r0_bsk.as_mut_ptr().add(start);
                                let mut r1_ptr = r1_bsk.as_mut_ptr().add(start);
                                let mut r2_ptr = r2_bsk.as_mut_ptr().add(start);
                                for _ in 0..degree {
                                    let a0 = *a0_ptr;
                                    let a1 = *a1_ptr;
                                    let b0 = *b0_ptr;
                                    let b1 = *b1_ptr;

                                    let mut s0 = a0.wrapping_add(a1);
                                    let s0_mask = ((s0 >= two_mod) as u64).wrapping_neg();
                                    s0 = s0.wrapping_sub(two_mod & s0_mask);
                                    let mut s1 = b0.wrapping_add(b1);
                                    let s1_mask = ((s1 >= two_mod) as u64).wrapping_neg();
                                    s1 = s1.wrapping_sub(two_mod & s1_mask);

                                    let r0 = silent_math::arith::mul_mod_barrett_u64(
                                        a0,
                                        b0,
                                        modulus_value,
                                        ratio0,
                                        ratio1,
                                    );
                                    let r2 = silent_math::arith::mul_mod_barrett_u64(
                                        a1,
                                        b1,
                                        modulus_value,
                                        ratio0,
                                        ratio1,
                                    );
                                    let m = silent_math::arith::mul_mod_barrett_u64(
                                        s0,
                                        s1,
                                        modulus_value,
                                        ratio0,
                                        ratio1,
                                    );
                                    let r1 = silent_math::arith::sub_mod(
                                        silent_math::arith::sub_mod(m, r0, modulus_value),
                                        r2,
                                        modulus_value,
                                    );

                                    *r0_ptr = r0;
                                    *r1_ptr = r1;
                                    *r2_ptr = r2;
                                    a0_ptr = a0_ptr.add(1);
                                    a1_ptr = a1_ptr.add(1);
                                    b0_ptr = b0_ptr.add(1);
                                    b1_ptr = b1_ptr.add(1);
                                    r0_ptr = r0_ptr.add(1);
                                    r1_ptr = r1_ptr.add(1);
                                    r2_ptr = r2_ptr.add(1);
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3) INTT outputs (Q and Bsk)
        {
            let _scope = profile::Scope::new(profile::Kind::MulBehzInttQ);
            for poly in scratch.out_q_ntt.chunks_mut(poly_q_len) {
                for limb in 0..size_q {
                    let slice = &mut poly[limb * degree..(limb + 1) * degree];
                    let tables = &ring_q.ntt_tables()[limb];
                    silent_math::ntt::ntt_inverse_lazy(slice, tables);
                }
            }
        }
        {
            let _scope = profile::Scope::new(profile::Kind::MulBehzInttBsk);
            for poly in scratch.out_bsk_ntt.chunks_mut(poly_bsk_len) {
                for limb in 0..size_bsk {
                    let slice = &mut poly[limb * degree..(limb + 1) * degree];
                    let tables = &ring_bsk.ntt_tables()[limb];
                    silent_math::ntt::ntt_inverse_lazy(slice, tables);
                }
            }
        }

        // 4) Multiply by t in Q and Bsk
        let q_moduli = ring_q.rns().moduli();
        let bsk_moduli = ring_bsk.rns().moduli();
        scratch.ensure_t(t, q_moduli, bsk_moduli);

        {
            let _scope = profile::Scope::new(profile::Kind::MulBehzMulT);
            for poly in scratch.out_q_ntt.chunks_mut(poly_q_len) {
                for limb in 0..size_q {
                    let modulus = &q_moduli[limb];
                    let t_mod = &scratch.t_mod_q[limb];
                    let start = limb * degree;
                    unsafe {
                        let poly_ptr = poly.as_mut_ptr().add(start);
                        for j in 0..degree {
                            let val = *poly_ptr.add(j);
                            *poly_ptr.add(j) = arith::mul_mod_shoup(
                                val,
                                t_mod.operand,
                                t_mod.quotient,
                                modulus.value(),
                            );
                        }
                    }
                }
            }
            for poly in scratch.out_bsk_ntt.chunks_mut(poly_bsk_len) {
                for limb in 0..size_bsk {
                    let modulus = &bsk_moduli[limb];
                    let t_mod = &scratch.t_mod_bsk[limb];
                    let start = limb * degree;
                    unsafe {
                        let poly_ptr = poly.as_mut_ptr().add(start);
                        for j in 0..degree {
                            let val = *poly_ptr.add(j);
                            *poly_ptr.add(j) = arith::mul_mod_shoup(
                                val,
                                t_mod.operand,
                                t_mod.quotient,
                                modulus.value(),
                            );
                        }
                    }
                }
            }
        }

        // 5) Scale & round (t/q) and convert back to base Q
        let mut result_polys = vec![Poly::new(degree, size_q); 3];
        for poly_idx in 0..3 {
            let q_start = poly_idx * poly_q_len;
            let q_end = q_start + poly_q_len;
            let bsk_start = poly_idx * poly_bsk_len;
            let bsk_end = bsk_start + poly_bsk_len;

            scratch.floor_in[..poly_q_len].copy_from_slice(&scratch.out_q_ntt[q_start..q_end]);
            scratch.floor_in[poly_q_len..]
                .copy_from_slice(&scratch.out_bsk_ntt[bsk_start..bsk_end]);

            {
                let _scope = profile::Scope::new(profile::Kind::MulBehzFloor);
                let bsk_out = &mut scratch.out_bsk_ntt[bsk_start..bsk_end];
                rns_tool
                    .behz_floor_into(
                        &scratch.floor_in,
                        degree,
                        bsk_out,
                        &mut scratch.work_scratch,
                    )
                    .expect("BEHZ rns_floor");
            }

            {
                let _scope = profile::Scope::new(profile::Kind::MulBehzBaseConvSk);
                let q_out = &mut scratch.out_q_ntt[q_start..q_end];
                rns_tool
                    .behz_baseconv_sk_into(
                        &scratch.out_bsk_ntt[bsk_start..bsk_end],
                        degree,
                        q_out,
                        &mut scratch.work_scratch,
                    )
                    .expect("BEHZ base_conv_sk");
            }

            let dest = result_polys[poly_idx].data_mut();
            dest.copy_from_slice(&scratch.out_q_ntt[q_start..q_end]);
        }

        Ciphertext::new(result_polys, self.params.runtime_params().clone(), false)
    }

    pub fn relinearize(&self, ct: &Ciphertext, evk: &silent_rlwe::EvaluationKey) -> Ciphertext {
        // Implementation of SEAL's relinearize_internal logic (simplified for size 3->2)
        if ct.data.len() != 3 {
            // Only support relin from size 3 to 2 for now, as is common in Mul
            return ct.clone();
        }

        let mut encrypted = ct.clone();
        let input_is_ntt = ct.is_ntt;
        let mul_method = self.current_mul_method();

        // Target: encrypted.data[2] (the last component)
        // Remove c2 from encrypted, as we will add the result to c0, c1
        let c2 = encrypted.data.pop().expect("Has 3 elements");

        // Prepare accumulators (c0, c1 are already in encrypted)
        // But we need mutable references to them.
        let (c0, c1) = encrypted.data.split_at_mut(1);
        let c0_acc = &mut c0[0];
        let c1_acc = &mut c1[0];

        // Perform KeySwitching: c2 * evk -> (sw0, sw1) added to (c0, c1)
        let ring_q = self.params.ring();
        let size_q = ring_q.rns().len();
        if matches!(mul_method, BfvMulMethod::Hps) {
            if input_is_ntt {
                self.switch_key_accumulate_hps(&c2, evk, c0_acc, c1_acc, true, true);
                encrypted.is_ntt = true;
                return encrypted;
            }

            let mut sw0 = Poly::new(ring_q.degree(), size_q);
            let mut sw1 = Poly::new(ring_q.degree(), size_q);
            self.switch_key_accumulate_hps(&c2, evk, &mut sw0, &mut sw1, false, false);
            c0_acc.add_assign(&sw0, ring_q);
            c1_acc.add_assign(&sw1, ring_q);
            encrypted.is_ntt = false;
            return encrypted;
        }

        if input_is_ntt {
            if evk.elements.len() == size_q && evk.elements[0].data[0].num_moduli() == size_q + 1 {
                self.switch_key_accumulate_rns(&c2, evk, c0_acc, c1_acc, true, true);
            } else {
                self.switch_key_accumulate(&c2, &evk.elements[0], c0_acc, c1_acc, true);
            }
            encrypted.is_ntt = true;
            return encrypted;
        }

        // Coeff-domain: compute keyswitch result in NTT, then add in coeff domain.
        let mut sw0_ntt = Poly::new(ring_q.degree(), size_q);
        let mut sw1_ntt = Poly::new(ring_q.degree(), size_q);

        if evk.elements.len() == size_q && evk.elements[0].data[0].num_moduli() == size_q + 1 {
            self.switch_key_accumulate_rns(&c2, evk, &mut sw0_ntt, &mut sw1_ntt, false, false);
        } else {
            self.switch_key_accumulate(&c2, &evk.elements[0], &mut sw0_ntt, &mut sw1_ntt, false);
        }

        c0_acc.add_assign(&sw0_ntt, ring_q);
        c1_acc.add_assign(&sw1_ntt, ring_q);
        encrypted.is_ntt = false;
        encrypted
    }

    // Optimized switch_key that accumulates result into destination polys
    // Uses RelinScratch to avoid intermediate allocations
    fn switch_key_accumulate(
        &self,
        c2: &Poly,
        k: &Ciphertext,
        acc0: &mut Poly,
        acc1: &mut Poly,
        c2_is_ntt: bool,
    ) {
        let ring_q = self.params.ring();
        let degree = ring_q.degree();
        let rns_tool = self.params.rns_tool();
        let base_q = rns_tool.base_q();
        let base_p = rns_tool.base_p().expect("Base P missing");
        let size_q = base_q.len();
        let size_p = base_p.len();
        let size_qp = size_q + size_p;
        let verify = std::env::var("SILENT_VERIFY_KEYSWITCH").is_ok();

        let mut acc0_ref = None;
        let mut acc1_ref = None;
        if verify {
            acc0_ref = Some(acc0.clone());
            acc1_ref = Some(acc1.clone());
        }

        // Ensure RingQP initialized
        let ring_qp = self.ring_qp.get_or_init(|| {
            let mut moduli_qp = base_q.moduli().to_vec();
            moduli_qp.extend_from_slice(base_p.moduli());
            let base_qp =
                silent_math::rns::RnsBase::new(moduli_qp).expect("Failed to build base QP");
            silent_ring::RingContext::new(ring_q.degree(), base_qp)
        });

        // Initialize scratch if needed
        {
            let mut borrowed = self.relin_scratch.borrow_mut();
            if borrowed.is_none() {
                *borrowed = Some(RelinScratch::new(degree, size_q, size_p));
            }
        }

        let mut scratch_ref = self.relin_scratch.borrow_mut();
        let scratch = scratch_ref.as_mut().unwrap();

        {
            let _scope = profile::Scope::new(profile::Kind::KeySwitchQpBaseConv);
            // 1. temp_q in coeff domain for base conversion
            {
                let src = c2.data();
                let dst = scratch.temp_q.data_mut();
                dst.copy_from_slice(src);
            }
            if c2_is_ntt {
                scratch.temp_q.ntt_inverse(ring_q);
            }

            // 2. Expand Q -> QP (t_qp)
            // Q part comes directly from c2 in NTT (no extra NTT needed)
            {
                let dst = scratch.t_qp.data_mut();
                if c2_is_ntt {
                    dst[..size_q * degree].copy_from_slice(c2.data());
                } else {
                    dst[..size_q * degree].copy_from_slice(scratch.temp_q.data());
                    for i in 0..size_q {
                        let limb = &mut dst[i * degree..(i + 1) * degree];
                        let tables = &ring_q.ntt_tables()[i];
                        silent_math::ntt::ntt_forward(limb, tables);
                    }
                }

                // Expand to P using coeff-domain temp_q
                let src_coeff = scratch.temp_q.data();
                let p_out = &mut dst[size_q * degree..];
                rns_tool
                    .base_q_to_p()
                    .expect("Q->P")
                    .fast_convert_array_into(
                        src_coeff,
                        degree,
                        p_out,
                        &mut scratch.digit_conv_scratch,
                    )
                    .unwrap();
            }
        }

        // 3. NTT(t_qp) for P limbs only (Q limbs already in NTT)
        {
            let _scope = profile::Scope::new(profile::Kind::KeySwitchQpNttP);
            let tables_qp = ring_qp.ntt_tables();
            for i in size_q..size_qp {
                let limb = scratch.t_qp.limb_mut(i);
                silent_math::ntt::ntt_forward_lazy(limb, &tables_qp[i]);
            }
        }

        // Precompute or fetch Shoup operands for key polynomials
        let key_id = k as *const _ as usize;
        {
            let mut cache = self.keyswitch_shoup_cache.borrow_mut();
            if !cache.contains_key(&key_id) {
                let moduli_qp = ring_qp.rns().moduli();
                let k0_shoup = PolyShoup::from_poly(&k.data[0], moduli_qp);
                let k1_shoup = PolyShoup::from_poly(&k.data[1], moduli_qp);
                cache.insert(key_id, (k0_shoup, k1_shoup));
            }
        }
        let cache = self.keyswitch_shoup_cache.borrow();
        let (k0_shoup, k1_shoup) = cache.get(&key_id).expect("shoup cache");

        // 4. Multiply and Accumulate
        // We reuse temp_q for ModDown result
        // We reuse prod_qp for product
        // k0
        {
            let _scope = profile::Scope::new(profile::Kind::KeySwitchQpMul);
            // prod_qp = t_qp * k0
            // Manual element-wise mul
            // Since t_qp is in NTT, k0 is in NTT (in QP).
            // But wait, k is STORED in QP form? Or Q form?
            // EvaluationKey stores Ciphertext. `Ciphertext` params usually Q.
            // But Galois/Relin keys are stored in specialized form?
            // SEAL stores them in Double CRT form (QP).
            // My `KeyGenerator` stored `ct_qp` (size 2 Poly in QP).
            // `BfvKeyGenerator::galois_keys` -> `ring_qp`.
            // Yes, they are in QP.

            // BUT `Ciphertext` struct likely expects Params to be Q.
            // If `k.context` says Q parameters, but data has P limbs?
            // Let's assume k has correct number of limbs (size_qp).
            // Checking `switch_key` logic: `k0 = &k.data[0]`.

            // Multiply
            // scratch.prod_qp.copy_from(scratch.t_qp)
            // prod_qp.mul_assign(k0, ring_qp)
            // To avoid copy, we can just mul inputs.
            // Poly mul logic:
            let degree = degree;
            let moduli_qp = ring_qp.rns().moduli();

            let t_data = scratch.t_qp.data();
            let prod_data = scratch.prod_qp.data_mut();

            for i in 0..size_qp {
                let mod_val = &moduli_qp[i];
                let modulus = mod_val.value();
                let offset = i * degree;
                let k0_shoup_limb = k0_shoup.limb(i);
                for j in 0..degree {
                    let shoup = &k0_shoup_limb[j];
                    prod_data[offset + j] = arith::mul_mod_shoup(
                        t_data[offset + j],
                        shoup.operand,
                        shoup.quotient,
                        modulus,
                    );
                }
            }

            // Mod Down to temp_q
            {
                let _scope = profile::Scope::new(profile::Kind::KeySwitchQpModDown);
                self.mod_down_qp_to_q(
                    &mut scratch.prod_qp,
                    &mut scratch.temp_q,
                    ring_qp,
                    rns_tool,
                    &mut scratch.mod_scratch,
                );
            }

            // Add to acc0
            acc0.add_assign(&scratch.temp_q, ring_q);
        }

        // k1
        {
            let _scope = profile::Scope::new(profile::Kind::KeySwitchQpMul);
            let t_data = scratch.t_qp.data();
            let prod_data = scratch.prod_qp.data_mut();

            let moduli_qp = ring_qp.rns().moduli();

            for i in 0..size_qp {
                let mod_val = &moduli_qp[i];
                let modulus = mod_val.value();
                let offset = i * degree;
                let k1_shoup_limb = k1_shoup.limb(i);
                for j in 0..degree {
                    let shoup = &k1_shoup_limb[j];
                    prod_data[offset + j] = arith::mul_mod_shoup(
                        t_data[offset + j],
                        shoup.operand,
                        shoup.quotient,
                        modulus,
                    );
                }
            }

            // Mod Down to temp_q
            {
                let _scope = profile::Scope::new(profile::Kind::KeySwitchQpModDown);
                self.mod_down_qp_to_q(
                    &mut scratch.prod_qp,
                    &mut scratch.temp_q,
                    ring_qp,
                    rns_tool,
                    &mut scratch.mod_scratch,
                );
            }

            // Add to acc1
            acc1.add_assign(&scratch.temp_q, ring_q);
        }

        if verify {
            let mut acc0_ref = acc0_ref.take().expect("acc0 ref");
            let mut acc1_ref = acc1_ref.take().expect("acc1 ref");

            // Reference path: full Q->QP NTT then mul/moddown using generic mul_mod
            let mut temp_q = c2.clone();
            temp_q.ntt_inverse(ring_q);

            let mut t_qp = Poly::new(degree, size_qp);
            {
                let dst = t_qp.data_mut();
                dst[..size_q * degree].copy_from_slice(temp_q.data());

                let p_out = &mut dst[size_q * degree..];
                let mut conv_scratch = vec![0u64; size_q * 128];
                rns_tool
                    .base_q_to_p()
                    .expect("Q->P")
                    .fast_convert_array_into(temp_q.data(), degree, p_out, &mut conv_scratch)
                    .unwrap();
            }

            // NTT all limbs (Q and P)
            {
                let tables_qp = ring_qp.ntt_tables();
                for i in 0..size_qp {
                    let limb = t_qp.limb_mut(i);
                    silent_math::ntt::ntt_forward_lazy(limb, &tables_qp[i]);
                }
            }

            let moduli_qp = ring_qp.rns().moduli();
            let mut prod_qp = Poly::new(degree, size_qp);
            let mut mod_scratch = vec![0u64; 2 * (size_p * degree + size_p * 128)];
            let mut temp_out = Poly::new(degree, size_q);

            // k0
            {
                let t_data = t_qp.data();
                let k0_data = k.data[0].data();
                let prod_data = prod_qp.data_mut();

                for i in 0..size_qp {
                    let mod_val = &moduli_qp[i];
                    let offset = i * degree;
                    for j in 0..degree {
                        prod_data[offset + j] = silent_math::arith::mul_mod(
                            t_data[offset + j],
                            k0_data[offset + j],
                            mod_val,
                        );
                    }
                }

                self.mod_down_qp_to_q(
                    &mut prod_qp,
                    &mut temp_out,
                    ring_qp,
                    rns_tool,
                    &mut mod_scratch,
                );
                acc0_ref.add_assign(&temp_out, ring_q);
            }

            // k1
            {
                let t_data = t_qp.data();
                let k1_data = k.data[1].data();
                let prod_data = prod_qp.data_mut();

                for i in 0..size_qp {
                    let mod_val = &moduli_qp[i];
                    let offset = i * degree;
                    for j in 0..degree {
                        prod_data[offset + j] = silent_math::arith::mul_mod(
                            t_data[offset + j],
                            k1_data[offset + j],
                            mod_val,
                        );
                    }
                }

                self.mod_down_qp_to_q(
                    &mut prod_qp,
                    &mut temp_out,
                    ring_qp,
                    rns_tool,
                    &mut mod_scratch,
                );
                acc1_ref.add_assign(&temp_out, ring_q);
            }

            if acc0_ref.data() != acc0.data() || acc1_ref.data() != acc1.data() {
                panic!("KeySwitch verification failed: optimized path mismatch");
            }
        }
    }

    fn mod_down_qp_to_q(
        &self,
        input_qp: &mut Poly,
        output_q: &mut Poly,
        ring_qp: &silent_ring::RingContext,
        rns_tool: &RnsTool,
        mod_scratch: &mut [u64],
    ) {
        let degree = ring_qp.degree();
        rns_tool
            .approx_mod_down_ntt_into(
                input_qp.data_mut(),
                degree,
                output_q.data_mut(),
                mod_scratch,
                ring_qp.ntt_tables(),
            )
            .expect("approx_mod_down");
    }

    fn mod_down_qk_add(
        &self,
        prod: &mut Poly,
        acc: &mut Poly,
        tmp: &mut [u64],
        tmp2: &mut [u64],
        ring_q: &silent_ring::RingContext,
        ring_qk: &silent_ring::RingContext,
        qk: silent_math::modulus::Modulus,
        cache: &KeySwitchModDownCache,
        output_ntt: bool,
    ) {
        let degree = ring_q.degree();
        let size_q = ring_q.rns().len();
        let qk_val = qk.value();
        let qk_half = qk_val >> 1;

        // t_last = inverse NTT of qk limb, rounded by qk/2
        {
            let _scope = profile::Scope::new(profile::Kind::KeySwitchModDownInvNtt);
            tmp2.copy_from_slice(prod.limb(size_q));
            silent_math::ntt::ntt_inverse_lazy(tmp2, &ring_qk.ntt_tables()[size_q]);
            for v in tmp2.iter_mut() {
                if *v >= qk_val {
                    *v -= qk_val;
                }
                *v = v.wrapping_add(qk_half);
                if *v >= qk_val {
                    *v -= qk_val;
                }
            }
        }

        {
            let _scope = profile::Scope::new(profile::Kind::KeySwitchModDownCore);
            for i in 0..size_q {
                let modulus = &ring_q.rns().moduli()[i];
                let qi = modulus.value();
                let inv_qk_shoup = &cache.inv_qk_shoup[i];
                let qk_half_mod_qi = cache.qk_half_mod_q[i];

                // Build t in coeff domain from qk limb: (t_last mod qi) + (qi - qk_half mod qi)
                unsafe {
                    let tmp_ptr = tmp.as_mut_ptr();
                    let tmp2_ptr = tmp2.as_ptr();
                    let fix = qi - qk_half_mod_qi;
                    for j in 0..degree {
                        let t_last_mod = modulus.reduce_u64_fast(*tmp2_ptr.add(j));
                        *tmp_ptr.add(j) = t_last_mod + fix;
                    }
                }

                let acc_limb = acc.limb_mut(i);
                if output_ntt {
                    // NTT forward t
                    silent_math::ntt::ntt_forward(tmp, &ring_q.ntt_tables()[i]);
                    let prod_limb = prod.limb(i);
                    unsafe {
                        let prod_ptr = prod_limb.as_ptr();
                        let tmp_ptr = tmp.as_ptr();
                        let acc_ptr = acc_limb.as_mut_ptr();
                        for j in 0..degree {
                            let diff = if *prod_ptr.add(j) >= *tmp_ptr.add(j) {
                                *prod_ptr.add(j) - *tmp_ptr.add(j)
                            } else {
                                *prod_ptr.add(j) + qi - *tmp_ptr.add(j)
                            };
                            let scaled = silent_math::arith::mul_mod_shoup_lazy(
                                diff,
                                inv_qk_shoup.operand,
                                inv_qk_shoup.quotient,
                                qi,
                            );
                            *acc_ptr.add(j) = (*acc_ptr.add(j)).wrapping_add(scaled);
                        }
                    }
                } else {
                    // BFV coeff-domain mod down: inverse NTT prod limb, then apply correction.
                    let prod_limb = prod.limb_mut(i);
                    silent_math::ntt::ntt_inverse_lazy(prod_limb, &ring_q.ntt_tables()[i]);
                    let qi_lazy = qi << 1;
                    unsafe {
                        let prod_ptr = prod_limb.as_ptr();
                        let tmp_ptr = tmp.as_ptr();
                        let acc_ptr = acc_limb.as_mut_ptr();
                        for j in 0..degree {
                            let mut diff = *prod_ptr.add(j) + qi_lazy - *tmp_ptr.add(j);
                            if diff >= qi_lazy {
                                diff -= qi_lazy;
                            }
                            let scaled = silent_math::arith::mul_mod_shoup_lazy(
                                diff,
                                inv_qk_shoup.operand,
                                inv_qk_shoup.quotient,
                                qi,
                            );
                            *acc_ptr.add(j) = (*acc_ptr.add(j)).wrapping_add(scaled);
                        }
                    }
                }

                // Normalize acc limb from [0, 3q) to [0, q)
                unsafe {
                    let acc_ptr = acc_limb.as_mut_ptr();
                    let two_q = qi << 1;
                    for j in 0..degree {
                        let mut v = *acc_ptr.add(j);
                        if v >= two_q {
                            v -= two_q;
                        }
                        if v >= qi {
                            v -= qi;
                        }
                        *acc_ptr.add(j) = v;
                    }
                }
            }
        }
    }

    fn apply_galois_coeff_inplace(
        &self,
        poly: &mut Poly,
        galois_elt: u32,
        ring_q: &silent_ring::RingContext,
        scratch: &mut [u64],
    ) {
        let degree = ring_q.degree();
        let map = ring_q.coeff_galois_map(galois_elt as u64);
        let moduli = ring_q.rns().moduli();
        for (mod_idx, modulus) in moduli.iter().enumerate() {
            let modulus_val = modulus.value();
            let limb = poly.limb_mut(mod_idx);
            for i in 0..degree {
                let index = map[i] as usize;
                let val = limb[i];
                if index < degree {
                    scratch[index] = val;
                } else {
                    let dest = index - degree;
                    scratch[dest] = if val == 0 { 0 } else { modulus_val - val };
                }
            }
            limb.copy_from_slice(&scratch[..degree]);
        }
    }

    fn switch_key_accumulate_rns(
        &self,
        c2: &Poly,
        evk: &silent_rlwe::EvaluationKey,
        acc0: &mut Poly,
        acc1: &mut Poly,
        c2_is_ntt: bool,
        output_ntt: bool,
    ) {
        let ring_q = self.params.ring();
        let rns_tool = self.params.rns_tool();
        let base_q = rns_tool.base_q();
        let size_q = base_q.len();
        let degree = ring_q.degree();

        let qk = if let Some(qk) = self.params.runtime_params().key_switch_modulus {
            qk
        } else {
            let base_p = rns_tool.base_p().expect("Base P missing");
            *base_p
                .moduli()
                .last()
                .expect("Base P must contain special prime")
        };

        let ring_qk = self.ring_qk.get_or_init(|| {
            let mut moduli_qk = base_q.moduli().to_vec();
            moduli_qk.push(qk);
            let base_qk =
                silent_math::rns::RnsBase::new(moduli_qk).expect("Failed to build base QK");
            silent_ring::RingContext::new(ring_q.degree(), base_qk)
        });

        let moddown_cache = self.kswitch_moddown_cache.get_or_init(|| {
            let qk_val = qk.value();
            let qk_half = qk_val >> 1;
            let mut inv_qk_shoup = Vec::with_capacity(size_q);
            let mut qk_half_mod_q = Vec::with_capacity(size_q);
            for modulus in ring_q.rns().moduli() {
                let qi = modulus.value();
                let inv_qk = silent_math::numth::mod_inverse(qk_val % qi, qi).expect("inv qk");
                inv_qk_shoup.push(silent_math::arith::MultiplyUIntModOperand::new(
                    inv_qk, qi,
                ));
                qk_half_mod_q.push(qk_half % qi);
            }
            KeySwitchModDownCache {
                inv_qk_shoup,
                qk_half_mod_q,
            }
        });
        let verify = std::env::var("SILENT_VERIFY_KEYSWITCH_RNS").is_ok();
        let strict = std::env::var("SILENT_KEYSWITCH_STRICT").is_ok();
        let seal_lazy = std::env::var("SILENT_KEYSWITCH_SEAL_LAZY")
            .map(|v| v != "0")
            .unwrap_or(true);
        let mut lazy_bound = usize::MAX;
        if seal_lazy {
            let max_bits = ring_q
                .rns()
                .moduli()
                .iter()
                .map(|m| m.bit_count() as usize)
                .max()
                .unwrap_or(0);
            if max_bits > 32 {
                let shift = 128usize.saturating_sub(max_bits << 1);
                lazy_bound = 1usize << shift;
            }
        }
        let mut acc0_ref = None;
        let mut acc1_ref = None;
        if verify {
            acc0_ref = Some(acc0.clone());
            acc1_ref = Some(acc1.clone());
        }

        // Init scratch
        {
            let mut borrowed = self.kswitch_scratch.borrow_mut();
            if borrowed.is_none() {
                *borrowed = Some(KeySwitchScratch::new(degree, size_q));
            }
        }
        let mut scratch_ref = self.kswitch_scratch.borrow_mut();
        let scratch = scratch_ref.as_mut().unwrap();
        let (c2_coeff, c2_ntt, prod0, prod1, _t_ntt_all, tmp, tmp2, acc0_buf, acc1_buf) = {
            let s = scratch;
            (
                &mut s.c2_coeff,
                &mut s.c2_ntt,
                &mut s.prod0,
                &mut s.prod1,
                &mut s.t_ntt_all,
                &mut s.tmp,
                &mut s.tmp2,
                &mut s.acc0,
                &mut s.acc1,
            )
        };

        if c2_is_ntt {
            let _scope = profile::Scope::new(profile::Kind::KeySwitchC2Intt);
            // c2_coeff = INTT(c2) (lazy to skip final reduction)
            c2_coeff.data_mut().copy_from_slice(c2.data());
            for i in 0..size_q {
                let limb = c2_coeff.limb_mut(i);
                silent_math::ntt::ntt_inverse_lazy(limb, &ring_q.ntt_tables()[i]);
            }
        } else {
            let _scope = profile::Scope::new(profile::Kind::KeySwitchC2Ntt);
            // c2 is already in coeff domain
            c2_coeff.data_mut().copy_from_slice(c2.data());
            c2_ntt.data_mut().copy_from_slice(c2.data());
            for i in 0..size_q {
                let limb = c2_ntt.limb_mut(i);
                silent_math::ntt::ntt_forward_lazy(limb, &ring_q.ntt_tables()[i]);
            }
        }

        // Zero prod buffers
        prod0.data_mut().fill(0);
        prod1.data_mut().fill(0);

        let moduli_q = ring_q.rns().moduli();
        let moduli_qk = ring_qk.rns().moduli();
        let size_qk = size_q + 1;

        let key_ntt_tables = ring_qk.ntt_tables();
        // Precompute shoup for keys if missing
        let evk_id = evk as *const _ as usize;
        {
            let mut cache = self.kswitch_key_shoup_cache.borrow_mut();
            if !cache.contains_key(&evk_id) {
                let moduli_qk = ring_qk.rns().moduli();
                let mut shoups = Vec::with_capacity(evk.elements.len());
                for key_ct in evk.elements.iter() {
                    let k0 = PolyShoup::from_poly(&key_ct.data[0], moduli_qk);
                    let k1 = PolyShoup::from_poly(&key_ct.data[1], moduli_qk);
                    shoups.push((k0, k1));
                }
                cache.insert(evk_id, shoups);
            }
        }
        let key_shoups = self.kswitch_key_shoup_cache.borrow();
        let key_shoups = key_shoups.get(&evk_id).expect("shoup cache");

        let key_shoups_ref = key_shoups;

        for key_index in 0..size_qk {
            let key_modulus = &moduli_qk[key_index];
            let key_modulus_val = key_modulus.value();
            let key_ntt = &key_ntt_tables[key_index];

            let prod0_limb = prod0.limb_mut(key_index);
            let prod1_limb = prod1.limb_mut(key_index);

            acc0_buf.fill(0);
            acc1_buf.fill(0);
            let mut lazy_counter = lazy_bound;

            for j in 0..size_q {
                let (k0_shoup, k1_shoup) = &key_shoups_ref[j];
                let k0 = k0_shoup.limb(key_index);
                let k1 = k1_shoup.limb(key_index);

                let t_operand: &[u64] = if !strict && key_index < size_q && key_index == j {
                    if c2_is_ntt {
                        c2.limb(j)
                    } else {
                        c2_ntt.limb(j)
                    }
                } else {
                    let coeff_limb = c2_coeff.limb(j);
                    {
                        let _scope = profile::Scope::new(profile::Kind::KeySwitchPrecompute);
                        if moduli_q[j].value() <= key_modulus_val {
                            tmp.copy_from_slice(coeff_limb);
                        } else {
                            unsafe {
                                let dst_ptr = tmp.as_mut_ptr();
                                let src_ptr = coeff_limb.as_ptr();
                                for k in 0..degree {
                                    *dst_ptr.add(k) = key_modulus.reduce_u64_fast(*src_ptr.add(k));
                                }
                            }
                        }
                        silent_math::ntt::ntt_forward_lazy(tmp, key_ntt);
                    }
                    &*tmp
                };

                let _scope = profile::Scope::new(profile::Kind::KeySwitchDot);
                unsafe {
                    let t_ptr = t_operand.as_ptr();
                    let k0_ptr = k0.as_ptr();
                    let k1_ptr = k1.as_ptr();
                    let acc0_ptr = acc0_buf.as_mut_ptr();
                    let acc1_ptr = acc1_buf.as_mut_ptr();
                    for k in 0..degree {
                        let t = *t_ptr.add(k);
                        let k0v = *k0_ptr.add(k);
                        let k1v = *k1_ptr.add(k);
                        let term0 = silent_math::arith::mul_mod_shoup_lazy(
                            t,
                            k0v.operand,
                            k0v.quotient,
                            key_modulus_val,
                        );
                        let term1 = silent_math::arith::mul_mod_shoup_lazy(
                            t,
                            k1v.operand,
                            k1v.quotient,
                            key_modulus_val,
                        );
                        *acc0_ptr.add(k) += term0 as u128;
                        *acc1_ptr.add(k) += term1 as u128;
                    }
                }

                if seal_lazy && lazy_counter == 0 {
                    unsafe {
                        let acc0_ptr = acc0_buf.as_mut_ptr();
                        let acc1_ptr = acc1_buf.as_mut_ptr();
                        for k in 0..degree {
                            *acc0_ptr.add(k) = key_modulus.reduce_u128(*acc0_ptr.add(k)) as u128;
                            *acc1_ptr.add(k) = key_modulus.reduce_u128(*acc1_ptr.add(k)) as u128;
                        }
                    }
                    lazy_counter = lazy_bound;
                }
                if seal_lazy && lazy_counter != usize::MAX {
                    lazy_counter = lazy_counter.saturating_sub(1);
                }
            }

            unsafe {
                let acc0_ptr = acc0_buf.as_ptr();
                let acc1_ptr = acc1_buf.as_ptr();
                let prod0_ptr = prod0_limb.as_mut_ptr();
                let prod1_ptr = prod1_limb.as_mut_ptr();
                for k in 0..degree {
                    *prod0_ptr.add(k) = key_modulus.reduce_u128(*acc0_ptr.add(k));
                    *prod1_ptr.add(k) = key_modulus.reduce_u128(*acc1_ptr.add(k));
                }
            }
        }

        // Mod down from QK to Q and accumulate
        {
            let _scope = profile::Scope::new(profile::Kind::KeySwitchModDown);
            self.mod_down_qk_add(
                prod0,
                acc0,
                tmp,
                tmp2,
                ring_q,
                ring_qk,
                qk,
                moddown_cache,
                output_ntt,
            );
            self.mod_down_qk_add(
                prod1,
                acc1,
                tmp,
                tmp2,
                ring_q,
                ring_qk,
                qk,
                moddown_cache,
                output_ntt,
            );
        }

        if verify {
            let mut acc0_ref = acc0_ref.take().expect("acc0 ref");
            let mut acc1_ref = acc1_ref.take().expect("acc1 ref");

            // Reference (slow) path to validate keyswitch logic.
            let mut c2_coeff_ref = c2.clone();
            if c2_is_ntt {
                c2_coeff_ref.ntt_inverse(ring_q);
            }

            let mut prod0_ref = Poly::new(degree, size_qk);
            let mut prod1_ref = Poly::new(degree, size_qk);
            let mut t_ntt = vec![0u64; degree];
            let mut acc0_buf_ref = vec![0u128; degree];
            let mut acc1_buf_ref = vec![0u128; degree];

            for key_index in 0..size_qk {
                let key_modulus = &moduli_qk[key_index];
                let key_modulus_val = key_modulus.value();
                let key_ntt = &key_ntt_tables[key_index];

                acc0_buf_ref.fill(0);
                acc1_buf_ref.fill(0);

                for j in 0..size_q {
                    let (k0_shoup, k1_shoup) = &key_shoups_ref[j];
                    let k0 = k0_shoup.limb(key_index);
                    let k1 = k1_shoup.limb(key_index);

                    if key_index < size_q && key_index == j && c2_is_ntt {
                        t_ntt.copy_from_slice(c2.limb(j));
                    } else {
                        let coeff_limb = c2_coeff_ref.limb(j);
                        if moduli_q[j].value() <= key_modulus_val {
                            t_ntt.copy_from_slice(coeff_limb);
                        } else {
                            for k in 0..degree {
                                t_ntt[k] = key_modulus.reduce_u64(coeff_limb[k]);
                            }
                        }
                        silent_math::ntt::ntt_forward(&mut t_ntt, key_ntt);
                    }

                    for k in 0..degree {
                        let t = t_ntt[k] as u128;
                        acc0_buf_ref[k] += t * (k0[k].operand as u128);
                        acc1_buf_ref[k] += t * (k1[k].operand as u128);
                    }
                }

                for k in 0..degree {
                    prod0_ref.limb_mut(key_index)[k] = key_modulus.reduce_u128(acc0_buf_ref[k]);
                    prod1_ref.limb_mut(key_index)[k] = key_modulus.reduce_u128(acc1_buf_ref[k]);
                }
            }

            // Mod down (reference, strict reductions)
            let mut t_last = vec![0u64; degree];
            let mut t_tmp = vec![0u64; degree];

            for (prod_ref, acc_ref) in [
                (&mut prod0_ref, &mut acc0_ref),
                (&mut prod1_ref, &mut acc1_ref),
            ] {
                t_last.copy_from_slice(prod_ref.limb(size_q));
                silent_math::ntt::ntt_inverse(&mut t_last, &ring_qk.ntt_tables()[size_q]);
                let qk_val = qk.value();
                let qk_half = qk_val >> 1;
                for v in t_last.iter_mut() {
                    let mut tmpv = *v;
                    if tmpv >= qk_val {
                        tmpv -= qk_val;
                    }
                    tmpv = tmpv.wrapping_add(qk_half);
                    if tmpv >= qk_val {
                        tmpv -= qk_val;
                    }
                    *v = tmpv;
                }

                for i in 0..size_q {
                    let modulus = &moduli_q[i];
                    let qi = modulus.value();
                    let inv_qk = moddown_cache.inv_qk_shoup[i].operand;
                    let fix = qi - moddown_cache.qk_half_mod_q[i];

                    for j in 0..degree {
                        let t_mod = modulus.reduce_u64(t_last[j]);
                        let mut val = t_mod + fix;
                        if val >= qi {
                            val -= qi;
                        }
                        t_tmp[j] = val;
                    }

                    if output_ntt {
                        silent_math::ntt::ntt_forward(&mut t_tmp, &ring_q.ntt_tables()[i]);
                    } else {
                        let prod_limb = prod_ref.limb_mut(i);
                        silent_math::ntt::ntt_inverse(prod_limb, &ring_q.ntt_tables()[i]);
                    }

                    let prod_limb = prod_ref.limb(i);
                    let acc_limb = acc_ref.limb_mut(i);
                    for j in 0..degree {
                        let mut diff = if prod_limb[j] >= t_tmp[j] {
                            prod_limb[j] - t_tmp[j]
                        } else {
                            prod_limb[j] + qi - t_tmp[j]
                        };
                        diff = silent_math::arith::mul_mod(diff, inv_qk, modulus);
                        acc_limb[j] = silent_math::arith::add_mod(acc_limb[j], diff, qi);
                    }
                }
            }

            if acc0_ref.data() != acc0.data() || acc1_ref.data() != acc1.data() {
                panic!("KeySwitch RNS verification failed: optimized path mismatch");
            }
        }
    }

    fn switch_key_accumulate_hps(
        &self,
        c2: &Poly,
        evk: &silent_rlwe::EvaluationKey,
        acc0: &mut Poly,
        acc1: &mut Poly,
        c2_is_ntt: bool,
        output_ntt: bool,
    ) {
        let ring_q = self.params.ring();
        let size_q = ring_q.rns().len();
        let degree = ring_q.degree();

        if evk.elements.len() != size_q || evk.elements[0].data[0].num_moduli() != size_q {
            panic!("HPS keyswitch expects Q-only keys (size_q moduli)");
        }

        // Init scratch
        {
            let mut borrowed = self.kswitch_scratch.borrow_mut();
            if borrowed.is_none() {
                *borrowed = Some(KeySwitchScratch::new(degree, size_q));
            }
        }
        let mut scratch_ref = self.kswitch_scratch.borrow_mut();
        let scratch = scratch_ref.as_mut().unwrap();
        let c2_coeff = &mut scratch.c2_coeff;
        let tmp = &mut scratch.tmp;
        let acc0_buf = &mut scratch.acc0;
        let acc1_buf = &mut scratch.acc1;

        if c2_is_ntt {
            c2_coeff.data_mut().copy_from_slice(c2.data());
            for i in 0..size_q {
                let limb = c2_coeff.limb_mut(i);
                silent_math::ntt::ntt_inverse_lazy(limb, &ring_q.ntt_tables()[i]);
            }
        } else {
            c2_coeff.data_mut().copy_from_slice(c2.data());
        }

        // Precompute shoup for keys if missing (Q-only)
        let evk_id = evk as *const _ as usize;
        {
            let mut cache = self.kswitch_key_shoup_cache_q.borrow_mut();
            if !cache.contains_key(&evk_id) {
                let moduli_q = ring_q.rns().moduli();
                let mut shoups = Vec::with_capacity(evk.elements.len());
                for key_ct in evk.elements.iter() {
                    let k0 = PolyShoup::from_poly(&key_ct.data[0], moduli_q);
                    let k1 = PolyShoup::from_poly(&key_ct.data[1], moduli_q);
                    shoups.push((k0, k1));
                }
                cache.insert(evk_id, shoups);
            }
        }
        let key_shoups = self.kswitch_key_shoup_cache_q.borrow();
        let key_shoups = key_shoups.get(&evk_id).expect("hps shoup cache");

        let moduli = ring_q.rns().moduli();

        for j in 0..size_q {
            let modulus = &moduli[j];
            let qi = modulus.value();
            let limb_tables = &ring_q.ntt_tables()[j];

            acc0_buf.fill(0);
            acc1_buf.fill(0);

            for (i, (k0_shoup, k1_shoup)) in key_shoups.iter().enumerate() {
                let t_ntt: &[u64] = if c2_is_ntt && i == j {
                    c2.limb(i)
                } else {
                    let coeff_limb = c2_coeff.limb(i);
                    let qi_src = moduli[i].value();
                    if !c2_is_ntt && qi_src <= qi {
                        tmp.copy_from_slice(coeff_limb);
                    } else {
                        for k in 0..degree {
                            tmp[k] = modulus.reduce_u64_fast(coeff_limb[k]);
                        }
                    }
                    silent_math::ntt::ntt_forward_lazy(tmp, limb_tables);
                    &*tmp
                };

                let k0_limb = k0_shoup.limb(j);
                let k1_limb = k1_shoup.limb(j);
                for k in 0..degree {
                    let t = t_ntt[k];
                    let term0 = silent_math::arith::mul_mod_shoup_lazy(
                        t,
                        k0_limb[k].operand,
                        k0_limb[k].quotient,
                        qi,
                    );
                    let term1 = silent_math::arith::mul_mod_shoup_lazy(
                        t,
                        k1_limb[k].operand,
                        k1_limb[k].quotient,
                        qi,
                    );
                    acc0_buf[k] += term0 as u128;
                    acc1_buf[k] += term1 as u128;
                }
            }

            let acc0_limb = acc0.limb_mut(j);
            let acc1_limb = acc1.limb_mut(j);
            for k in 0..degree {
                let term0 = modulus.reduce_u128(acc0_buf[k]);
                let term1 = modulus.reduce_u128(acc1_buf[k]);
                acc0_limb[k] = silent_math::arith::add_mod(acc0_limb[k], term0, qi);
                acc1_limb[k] = silent_math::arith::add_mod(acc1_limb[k], term1, qi);
            }
        }

        if !output_ntt {
            acc0.ntt_inverse(ring_q);
            acc1.ntt_inverse(ring_q);
        }
    }

    pub fn rotate(&self, ct: &Ciphertext, k: u32, gk: &GaloisKey) -> Ciphertext {
        if ct.data.len() != 2 {
            // Only support size 2 for now
            return ct.clone();
        }
        let switch_key_k = match gk.keys.get(&k) {
            Some(key) => key,
            None => {
                if k == 0 {
                    return ct.clone();
                }
                panic!("Rotation key for k={} not found", k);
            }
        };

        // Contexts
        let ring_q = self.params.ring();
        let degree = ring_q.degree();
        let size_q = ring_q.rns().len();

        let mut c0_rot = ct.data[0].clone();
        let mut c1_rot = ct.data[1].clone();

        // Ensure coefficient-domain representation for BFV rotation (SEAL-style).
        if ct.is_ntt {
            c0_rot.ntt_inverse(ring_q);
            c1_rot.ntt_inverse(ring_q);
        }

        // 1. Apply Galois Automorphism in coefficient domain to c0, c1
        let mut scratch = self.rotate_scratch.borrow_mut();
        if scratch.len() != degree {
            scratch.resize(degree, 0);
        }
        let scratch = &mut scratch[..];

        self.apply_galois_coeff_inplace(&mut c0_rot, k, ring_q, scratch);
        self.apply_galois_coeff_inplace(&mut c1_rot, k, ring_q, scratch);

        // 2. Key Switch c1_rot
        // We need result = (c0_rot + sw0, sw1)
        // We can initialize result with (c0_rot, 0)
        // Then accumulate (sw0, sw1) into it.

        // Key switching in NTT domain, then add to coeff-domain c0.
        let mut sw0_ntt = Poly::new(degree, size_q);
        let mut sw1_ntt = Poly::new(degree, size_q);
        let mul_method = self.current_mul_method();
        if matches!(mul_method, BfvMulMethod::Hps) {
            self.switch_key_accumulate_hps(
                &c1_rot,
                switch_key_k,
                &mut sw0_ntt,
                &mut sw1_ntt,
                false,
                false,
            );
        } else if switch_key_k.elements.len() == size_q
            && switch_key_k.elements[0].data[0].num_moduli() == size_q + 1
        {
            self.switch_key_accumulate_rns(
                &c1_rot,
                switch_key_k,
                &mut sw0_ntt,
                &mut sw1_ntt,
                false,
                false,
            );
        } else {
            self.switch_key_accumulate(
                &c1_rot,
                &switch_key_k.elements[0],
                &mut sw0_ntt,
                &mut sw1_ntt,
                false,
            );
        }

        c0_rot.add_assign(&sw0_ntt, ring_q);
        let zero_c1 = sw1_ntt;

        Ciphertext::new(
            vec![c0_rot, zero_c1],
            self.params.runtime_params().clone(),
            false,
        )
    }
}
#[cfg(test)]
mod tests {
    // use super::*;
    use silent_math::arith;
    use silent_math::rns::RnsBase;

    #[test]
    fn test_base_b_to_q_correction() {
        // Setup B (larger) and Q (smaller)
        // B > Q so we can represent values up to B-1
        // We pick a value X such that Q < X < B mod B
        // And convert to Q.

        let bases_q = vec![10007u64]; // Small primes for easy debug
        let bases_b = vec![20011u64, 20021u64];

        let base_q = RnsBase::from_values(bases_q).unwrap();
        let base_b = RnsBase::from_values(bases_b).unwrap();

        // Let X = 15000.
        // X > Q (10007).
        // X < B (approx 4e8).
        // X mod Q = 15000 % 10007 = 4993.

        let x = 15000u64;
        let x_b: Vec<u64> = base_b.moduli().iter().map(|m| x % m.value()).collect();

        // Use fastbconv
        let converter =
            silent_math::rns::BaseConverter::new(base_b.clone(), base_q.clone()).unwrap();
        let res_fast = converter.fast_convert(&x_b).unwrap();

        // Check finding
        let res_q = res_fast[0];
        println!(
            "X={}, Q={}, X%Q={}",
            x,
            base_q.moduli()[0].value(),
            x % base_q.moduli()[0].value()
        );
        println!("FastConv Result: {}", res_q);

        // It likely does NOT match 4993 immediately if alpha > 0.
        // alpha approx sum(x_i / b_i).
        // x_0 = 15000, b_0 = 20011. ratio ~ 0.75
        // x_1 = 15000, b_1 = 20021. ratio ~ 0.75
        // sum ~ 1.5. alpha = round(1.5) ? could be 1 or 2.
        // Actually alpha = sum (x_i * b_i*) / B approximately?
        // Let's implement the alpha correction logic here and verify it recovers 4993.

        // Correction Implement
        let mut alpha_sum = 0.5f64;
        let _inv_b: Vec<f64> = base_b
            .moduli()
            .iter()
            .zip(base_b.inv_punctured_prod_mod_base().iter())
            .map(|(m, &inv)| (inv as f64) / (m.value() as f64))
            .collect();

        // We need residues * b_i* mod b_i.
        // fastbconv 'fast_convert' does NOT give us the intermediate w_i without exposing internals.
        // But we can compute w_i = x_i * inv_punct_i mod b_i.

        for i in 0..base_b.len() {
            let val = x_b[i];
            let inv = base_b.inv_punctured_prod_mod_base()[i];
            let modulus = &base_b.moduli()[i];
            let w = arith::mul_mod(val, inv, modulus);

            // Check consistency: w * (B/b_i) = X mod b_i
            // It essentially reconstructs the CRT components.

            // The alpha term comes from sum(w_i * (B/b_i)) = X + alpha * B.
            // alpha = floor( sum( w_i / b_i ) )

            alpha_sum += (w as f64) / (modulus.value() as f64);
        }

        let alpha = alpha_sum as u64;
        println!("Computed Alpha: {}", alpha);

        let b_mod_q = base_b.base_prod().mod_u64(base_q.moduli()[0].value());
        let correction = arith::mul_mod_u64(alpha, b_mod_q, base_q.moduli()[0].value());

        let final_res = arith::sub_mod(res_q, correction, base_q.moduli()[0].value());
        println!("Corrected Result: {}", final_res);

        assert_eq!(final_res, x % base_q.moduli()[0].value());
    }
}
