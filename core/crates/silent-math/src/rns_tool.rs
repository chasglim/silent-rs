//! RNS tool helpers (fast base conversion and scaling/rounding).

use crate::arith;
use crate::bigint::BigUint;
use crate::modulus::Modulus;
use crate::ntt::{NttTables, ntt_forward, ntt_inverse};
use crate::numth;
use crate::rns::{BaseConverter, RnsBase, RnsError, RnsRounder};
// use silent_utils::arena::Arena;

mod profile {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    #[derive(Clone, Copy)]
    pub enum Kind {
        ModDownNttInttP,
        ModDownNttScaleP,
        ModDownNttSwitch,
        ModDownNttQntt,
        ModDownNttCombine,
        ModDownCoeffSwitch,
        ModDownCoeffCombine,
        BehzQMulMtilde,
        BehzQToBsk,
        BehzQToMtilde,
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
    static MOD_DOWN_NTT_INTT_P: Totals = Totals::new();
    static MOD_DOWN_NTT_SCALE_P: Totals = Totals::new();
    static MOD_DOWN_NTT_SWITCH: Totals = Totals::new();
    static MOD_DOWN_NTT_Q_NTT: Totals = Totals::new();
    static MOD_DOWN_NTT_COMBINE: Totals = Totals::new();
    static MOD_DOWN_COEFF_SWITCH: Totals = Totals::new();
    static MOD_DOWN_COEFF_COMBINE: Totals = Totals::new();
    static BEHZ_Q_MUL_MTILDE: Totals = Totals::new();
    static BEHZ_Q_TO_BSK: Totals = Totals::new();
    static BEHZ_Q_TO_MTILDE: Totals = Totals::new();

    fn enabled() -> bool {
        *ENABLED.get_or_init(|| std::env::var("SILENT_PROFILE").is_ok())
    }

    fn totals(kind: Kind) -> &'static Totals {
        match kind {
            Kind::ModDownNttInttP => &MOD_DOWN_NTT_INTT_P,
            Kind::ModDownNttScaleP => &MOD_DOWN_NTT_SCALE_P,
            Kind::ModDownNttSwitch => &MOD_DOWN_NTT_SWITCH,
            Kind::ModDownNttQntt => &MOD_DOWN_NTT_Q_NTT,
            Kind::ModDownNttCombine => &MOD_DOWN_NTT_COMBINE,
            Kind::ModDownCoeffSwitch => &MOD_DOWN_COEFF_SWITCH,
            Kind::ModDownCoeffCombine => &MOD_DOWN_COEFF_COMBINE,
            Kind::BehzQMulMtilde => &BEHZ_Q_MUL_MTILDE,
            Kind::BehzQToBsk => &BEHZ_Q_TO_BSK,
            Kind::BehzQToMtilde => &BEHZ_Q_TO_MTILDE,
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
            "[profile] {:<18} count={} total_ms={:.3} avg_us={:.3}",
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
        eprintln!("[profile] RNS mod_down breakdown");
        dump_line("mod_down.intt_p", &MOD_DOWN_NTT_INTT_P);
        dump_line("mod_down.scale_p", &MOD_DOWN_NTT_SCALE_P);
        dump_line("mod_down.switch", &MOD_DOWN_NTT_SWITCH);
        dump_line("mod_down.q_ntt", &MOD_DOWN_NTT_Q_NTT);
        dump_line("mod_down.combine", &MOD_DOWN_NTT_COMBINE);
        dump_line("mod_downc.switch", &MOD_DOWN_COEFF_SWITCH);
        dump_line("mod_downc.combine", &MOD_DOWN_COEFF_COMBINE);
        dump_line("behz.mul_mtilde", &BEHZ_Q_MUL_MTILDE);
        dump_line("behz.q_to_bsk", &BEHZ_Q_TO_BSK);
        dump_line("behz.q_to_mtilde", &BEHZ_Q_TO_MTILDE);
    }
}

#[derive(Clone, Debug)]
struct OpenFheTables {
    base_b: RnsBase,
    base_bsk: RnsBase,
    m_tilde: u64,
    m_sk: Modulus,
    q_hat_mod_bsk: Vec<Vec<u64>>,
    q_hat_mod_m_tilde: Vec<u64>,
    m_tilde_q_hat_inv_mod_q: Vec<u64>,
    q_mod_bsk: Vec<u64>,
    neg_q_inv_mod_m_tilde: u64,
    m_tilde_inv_mod_bsk: Vec<u64>,
    q_inv_mod_bsk: Vec<Vec<u64>>,
    t_q_hat_inv_mod_q: Vec<u64>,
    t_q_inv_mod_bsk: Vec<u64>,
    b_hat_inv_mod_b: Vec<u64>,
    b_hat_mod_m_sk: Vec<u64>,
    b_inv_mod_m_sk: u64,
    b_hat_mod_q: Vec<Vec<u64>>,
    b_mod_q: Vec<u64>,
}

#[derive(Clone, Debug)]
struct BehzTables {
    base_b: RnsBase,
    base_bsk: RnsBase,
    _base_bsk_mtilde: RnsBase,
    base_q_to_bsk: BaseConverter,
    base_q_to_mtilde: BaseConverter,
    base_b_to_q: BaseConverter,
    base_b_to_msk: BaseConverter,
    inv_prod_q_mod_bsk: Vec<arith::MultiplyUIntModOperand>,
    inv_m_tilde_mod_bsk: Vec<arith::MultiplyUIntModOperand>,
    _prod_q_mod_bsk: Vec<u64>,
    prod_q_mod_bsk_shoup: Vec<arith::MultiplyUIntModOperand>,
    neg_inv_prod_q_mod_mtilde: arith::MultiplyUIntModOperand,
    inv_prod_b_mod_msk: arith::MultiplyUIntModOperand,
    _prod_b_mod_q: Vec<u64>,
    prod_b_mod_q_shoup: Vec<arith::MultiplyUIntModOperand>,
    neg_prod_b_mod_q_shoup: Vec<arith::MultiplyUIntModOperand>,
    m_tilde_mod_q_shoup: Vec<arith::MultiplyUIntModOperand>,
    m_tilde: u64,
    m_sk: Modulus,
}

impl BehzTables {
    fn new(
        base_q: &RnsBase,
        base_b: &RnsBase,
        m_tilde: u64,
        m_sk: Modulus,
    ) -> Result<Self, RnsError> {
        if m_tilde == 0 || !m_tilde.is_power_of_two() {
            return Err(RnsError::InvalidMtilde);
        }

        let m_tilde_mod = Modulus::new(m_tilde)?;
        let mut bsk_moduli = base_b.moduli().to_vec();
        bsk_moduli.push(m_sk);
        let base_bsk = RnsBase::new(bsk_moduli)?;
        let mut bsk_mtilde_moduli = base_bsk.moduli().to_vec();
        bsk_mtilde_moduli.push(m_tilde_mod);
        let base_bsk_mtilde = RnsBase::new(bsk_mtilde_moduli)?;

        let base_q_to_bsk = BaseConverter::new(base_q.clone(), base_bsk.clone())?;
        let base_q_to_mtilde =
            BaseConverter::new(base_q.clone(), RnsBase::new(vec![m_tilde_mod])?)?;
        let base_b_to_q = BaseConverter::new(base_b.clone(), base_q.clone())?;
        let base_b_to_msk = BaseConverter::new(base_b.clone(), RnsBase::new(vec![m_sk])?)?;

        let mut prod_q_mod_bsk = vec![0u64; base_bsk.len()];
        let mut prod_q_mod_bsk_shoup = Vec::with_capacity(base_bsk.len());
        let mut inv_prod_q_mod_bsk = Vec::with_capacity(base_bsk.len());
        let mut inv_m_tilde_mod_bsk = Vec::with_capacity(base_bsk.len());

        for (idx, modulus_bsk) in base_bsk.moduli().iter().enumerate() {
            let bsk_value = modulus_bsk.value();
            let prod_q_mod = base_q.base_prod().mod_u64(bsk_value);
            prod_q_mod_bsk[idx] = prod_q_mod;
            prod_q_mod_bsk_shoup.push(arith::MultiplyUIntModOperand::new(prod_q_mod, bsk_value));

            let inv_prod_q = numth::mod_inverse(prod_q_mod % bsk_value, bsk_value)
                .ok_or(RnsError::MissingInverse)?;
            inv_prod_q_mod_bsk.push(arith::MultiplyUIntModOperand::new(inv_prod_q, bsk_value));

            let inv_m_tilde = numth::mod_inverse(m_tilde % bsk_value, bsk_value)
                .ok_or(RnsError::MissingInverse)?;
            inv_m_tilde_mod_bsk.push(arith::MultiplyUIntModOperand::new(inv_m_tilde, bsk_value));
        }

        let prod_q_mod_mtilde = base_q.base_prod().mod_u64(m_tilde);
        let inv_prod_q_mod_mtilde = numth::mod_inverse(prod_q_mod_mtilde % m_tilde, m_tilde)
            .ok_or(RnsError::MissingInverse)?;
        let neg_inv_prod_q_mod_mtilde = if inv_prod_q_mod_mtilde == 0 {
            0
        } else {
            m_tilde - inv_prod_q_mod_mtilde
        };
        let neg_inv_prod_q_mod_mtilde =
            arith::MultiplyUIntModOperand::new(neg_inv_prod_q_mod_mtilde, m_tilde);

        let prod_b_mod_msk = base_b.base_prod().mod_u64(m_sk.value());
        let inv_prod_b_mod_msk = numth::mod_inverse(prod_b_mod_msk % m_sk.value(), m_sk.value())
            .ok_or(RnsError::MissingInverse)?;
        let inv_prod_b_mod_msk =
            arith::MultiplyUIntModOperand::new(inv_prod_b_mod_msk, m_sk.value());

        let mut prod_b_mod_q = vec![0u64; base_q.len()];
        let mut prod_b_mod_q_shoup = Vec::with_capacity(base_q.len());
        let mut neg_prod_b_mod_q_shoup = Vec::with_capacity(base_q.len());
        for (idx, modulus_q) in base_q.moduli().iter().enumerate() {
            let q_value = modulus_q.value();
            let prod_b_mod = base_b.base_prod().mod_u64(q_value);
            prod_b_mod_q[idx] = prod_b_mod;
            prod_b_mod_q_shoup.push(arith::MultiplyUIntModOperand::new(prod_b_mod, q_value));
            let neg_prod = if prod_b_mod == 0 {
                0
            } else {
                q_value - prod_b_mod
            };
            neg_prod_b_mod_q_shoup.push(arith::MultiplyUIntModOperand::new(neg_prod, q_value));
        }

        let mut m_tilde_mod_q_shoup = Vec::with_capacity(base_q.len());
        for modulus_q in base_q.moduli().iter() {
            let q_value = modulus_q.value();
            let m_tilde_mod_q = m_tilde % q_value;
            m_tilde_mod_q_shoup.push(arith::MultiplyUIntModOperand::new(m_tilde_mod_q, q_value));
        }

        Ok(Self {
            base_b: base_b.clone(),
            base_bsk,
            _base_bsk_mtilde: base_bsk_mtilde,
            base_q_to_bsk,
            base_q_to_mtilde,
            base_b_to_q,
            base_b_to_msk,
            inv_prod_q_mod_bsk,
            inv_m_tilde_mod_bsk,
            _prod_q_mod_bsk: prod_q_mod_bsk,
            prod_q_mod_bsk_shoup,
            neg_inv_prod_q_mod_mtilde,
            inv_prod_b_mod_msk,
            _prod_b_mod_q: prod_b_mod_q,
            prod_b_mod_q_shoup,
            neg_prod_b_mod_q_shoup,
            m_tilde_mod_q_shoup,
            m_tilde,
            m_sk,
        })
    }
}

impl OpenFheTables {
    fn new(
        base_q: &RnsBase,
        base_b: &RnsBase,
        base_t: Modulus,
        m_tilde: u64,
        m_sk: Modulus,
    ) -> Result<Self, RnsError> {
        if m_tilde == 0 || !m_tilde.is_power_of_two() {
            return Err(RnsError::InvalidMtilde);
        }

        let mut bsk_moduli = base_b.moduli().to_vec();
        bsk_moduli.push(m_sk);
        let base_bsk = RnsBase::new(bsk_moduli)?;

        for modulus in base_bsk.moduli() {
            if m_tilde >= modulus.value() {
                return Err(RnsError::InvalidMtilde);
            }
        }

        let num_q = base_q.len();
        let num_b = base_b.len();
        let num_bsk = base_bsk.len();

        let mut q_hat_mod_bsk = vec![vec![0u64; num_bsk]; num_q];
        let mut q_hat_mod_m_tilde = vec![0u64; num_q];
        let mut m_tilde_q_hat_inv_mod_q = vec![0u64; num_q];
        let mut t_q_hat_inv_mod_q = vec![0u64; num_q];

        for (i, modulus_q) in base_q.moduli().iter().enumerate() {
            let q_hat = &base_q.punctured_prod()[i];
            q_hat_mod_m_tilde[i] = q_hat.mod_u64(m_tilde);
            let inv = base_q.inv_punctured_prod_mod_base()[i];
            let q_value = modulus_q.value();
            m_tilde_q_hat_inv_mod_q[i] = arith::mul_mod_u64(m_tilde % q_value, inv, q_value);
            t_q_hat_inv_mod_q[i] = arith::mul_mod_u64(base_t.value() % q_value, inv, q_value);
            for (j, modulus_bsk) in base_bsk.moduli().iter().enumerate() {
                q_hat_mod_bsk[i][j] = q_hat.mod_u64(modulus_bsk.value());
            }
        }

        let mut q_mod_bsk = vec![0u64; num_bsk];
        let mut t_q_inv_mod_bsk = vec![0u64; num_bsk];
        let mut m_tilde_inv_mod_bsk = vec![0u64; num_bsk];
        for (j, modulus_bsk) in base_bsk.moduli().iter().enumerate() {
            let bsk_value = modulus_bsk.value();
            let q_mod = base_q.base_prod().mod_u64(bsk_value);
            q_mod_bsk[j] = q_mod;
            let q_inv =
                numth::mod_inverse(q_mod % bsk_value, bsk_value).ok_or(RnsError::MissingInverse)?;
            t_q_inv_mod_bsk[j] = arith::mul_mod_u64(base_t.value() % bsk_value, q_inv, bsk_value);
            let m_tilde_inv = numth::mod_inverse(m_tilde % bsk_value, bsk_value)
                .ok_or(RnsError::MissingInverse)?;
            m_tilde_inv_mod_bsk[j] = m_tilde_inv;
        }

        let q_mod_m_tilde = base_q.base_prod().mod_u64(m_tilde);
        let q_inv_mod_m_tilde =
            numth::mod_inverse(q_mod_m_tilde % m_tilde, m_tilde).ok_or(RnsError::MissingInverse)?;
        let neg_q_inv_mod_m_tilde = if q_inv_mod_m_tilde == 0 {
            0
        } else {
            m_tilde - q_inv_mod_m_tilde
        };

        let mut q_inv_mod_bsk = vec![vec![0u64; num_bsk]; num_q];
        for (i, modulus_q) in base_q.moduli().iter().enumerate() {
            let q_value = modulus_q.value();
            for (j, modulus_bsk) in base_bsk.moduli().iter().enumerate() {
                let bsk_value = modulus_bsk.value();
                let inv = numth::mod_inverse(q_value % bsk_value, bsk_value)
                    .ok_or(RnsError::MissingInverse)?;
                q_inv_mod_bsk[i][j] = inv;
            }
        }

        let b_hat_inv_mod_b = base_b.inv_punctured_prod_mod_base().to_vec();
        let mut b_hat_mod_m_sk = vec![0u64; num_b];
        let mut b_hat_mod_q = vec![vec![0u64; num_q]; num_b];
        for (i, b_hat) in base_b.punctured_prod().iter().enumerate() {
            b_hat_mod_m_sk[i] = b_hat.mod_u64(m_sk.value());
            for (j, modulus_q) in base_q.moduli().iter().enumerate() {
                b_hat_mod_q[i][j] = b_hat.mod_u64(modulus_q.value());
            }
        }

        let mut b_mod_q = vec![0u64; num_q];
        for (j, modulus_q) in base_q.moduli().iter().enumerate() {
            b_mod_q[j] = base_b.base_prod().mod_u64(modulus_q.value());
        }
        let b_mod_m_sk = base_b.base_prod().mod_u64(m_sk.value());
        let b_inv_mod_m_sk = numth::mod_inverse(b_mod_m_sk % m_sk.value(), m_sk.value())
            .ok_or(RnsError::MissingInverse)?;

        Ok(Self {
            base_b: base_b.clone(),
            base_bsk,
            m_tilde,
            m_sk,
            q_hat_mod_bsk,
            q_hat_mod_m_tilde,
            m_tilde_q_hat_inv_mod_q,
            q_mod_bsk,
            neg_q_inv_mod_m_tilde,
            m_tilde_inv_mod_bsk,
            q_inv_mod_bsk,
            t_q_hat_inv_mod_q,
            t_q_inv_mod_bsk,
            b_hat_inv_mod_b,
            b_hat_mod_m_sk,
            b_inv_mod_m_sk,
            b_hat_mod_q,
            b_mod_q,
        })
    }
}

#[derive(Clone, Debug)]
struct SwitchTables {
    base_p: RnsBase,
    base_q_to_p: BaseConverter,
    base_p_to_q: BaseConverter,
    q_hat_inv_mod_q: Vec<arith::MultiplyUIntModOperand>,
    q_hat_mod_p: Vec<Vec<arith::MultiplyUIntModOperand>>,
    alpha_q_mod_p: Vec<Vec<arith::MultiplyUIntModOperand>>,
    q_inv: Vec<f64>,
    p_inv_mod_q: Vec<arith::MultiplyUIntModOperand>,
    p_hat_inv_mod_p: Vec<arith::MultiplyUIntModOperand>,
    p_hat_mod_q: Vec<Vec<arith::MultiplyUIntModOperand>>,
    t_inv_mod_p: Option<Vec<arith::MultiplyUIntModOperand>>,
    t_p_inv_mod_q: Option<Vec<arith::MultiplyUIntModOperand>>,
    p_inv: Vec<f64>,
    alpha_p_mod_q: Vec<Vec<arith::MultiplyUIntModOperand>>,
}

impl SwitchTables {
    fn new(base_q: &RnsBase, base_p: RnsBase, t: Option<Modulus>) -> Result<Self, RnsError> {
        if base_p.is_empty() {
            return Err(RnsError::EmptyBase);
        }
        for modulus_q in base_q.moduli() {
            for modulus_p in base_p.moduli() {
                if numth::gcd(modulus_q.value(), modulus_p.value()) != 1 {
                    return Err(RnsError::NonCoprime);
                }
            }
        }

        let size_q = base_q.len();
        let size_p = base_p.len();
        let base_q_to_p = BaseConverter::new(base_q.clone(), base_p.clone())?;
        let base_p_to_q = BaseConverter::new(base_p.clone(), base_q.clone())?;

        let mut q_hat_inv_mod_q = Vec::with_capacity(size_q);
        let mut q_hat_mod_p = vec![Vec::with_capacity(size_p); size_q];

        for (i, modulus_q) in base_q.moduli().iter().enumerate() {
            let q_val = modulus_q.value();
            q_hat_inv_mod_q.push(arith::MultiplyUIntModOperand::new(
                base_q.inv_punctured_prod_mod_base()[i],
                q_val,
            ));

            let q_hat = &base_q.punctured_prod()[i];
            for (_j, modulus_p) in base_p.moduli().iter().enumerate() {
                q_hat_mod_p[i].push(arith::MultiplyUIntModOperand::new(
                    q_hat.mod_u64(modulus_p.value()),
                    modulus_p.value(),
                ));
            }
        }

        let mut alpha_q_mod_p = vec![Vec::with_capacity(size_p); size_q + 1];
        for (_j, modulus_p) in base_p.moduli().iter().enumerate() {
            let p_val = modulus_p.value();
            let q_mod_p = base_q.base_prod().mod_u64(p_val);
            for alpha in 0..=size_q {
                let val = arith::mul_mod_u64(alpha as u64 % p_val, q_mod_p, p_val);
                alpha_q_mod_p[alpha].push(arith::MultiplyUIntModOperand::new(val, p_val));
            }
        }

        let q_inv: Vec<f64> = base_q
            .moduli()
            .iter()
            .map(|modulus| 1.0f64 / modulus.value() as f64)
            .collect();

        let mut p_inv_mod_q = Vec::with_capacity(size_q);
        for (_i, modulus_q) in base_q.moduli().iter().enumerate() {
            let q_value = modulus_q.value();
            let p_mod_q = base_p.base_prod().mod_u64(q_value);
            let inv = numth::mod_inverse(p_mod_q, q_value).ok_or(RnsError::MissingInverse)?;
            p_inv_mod_q.push(arith::MultiplyUIntModOperand::new(inv, q_value));
        }

        let mut p_hat_inv_mod_p = Vec::with_capacity(size_p);
        for (_i, modulus_p) in base_p.moduli().iter().enumerate() {
            p_hat_inv_mod_p.push(arith::MultiplyUIntModOperand::new(
                base_p.inv_punctured_prod_mod_base()[_i],
                modulus_p.value(),
            ));
        }

        let mut p_hat_mod_q = vec![Vec::with_capacity(size_q); size_p];
        for (i, p_hat) in base_p.punctured_prod().iter().enumerate() {
            for (_j, modulus_q) in base_q.moduli().iter().enumerate() {
                p_hat_mod_q[i].push(arith::MultiplyUIntModOperand::new(
                    p_hat.mod_u64(modulus_q.value()),
                    modulus_q.value(),
                ));
            }
        }

        let (t_inv_mod_p, t_p_inv_mod_q) = if let Some(t_modulus) = t {
            let t_value = t_modulus.value();
            let mut t_inv_p = Vec::with_capacity(size_p);
            for (_j, modulus_p) in base_p.moduli().iter().enumerate() {
                let p_value = modulus_p.value();
                let inv = numth::mod_inverse(t_value % p_value, p_value)
                    .ok_or(RnsError::MissingInverse)?;
                t_inv_p.push(arith::MultiplyUIntModOperand::new(inv, p_value));
            }

            let mut combined = Vec::with_capacity(size_q);
            for (i, modulus_q) in base_q.moduli().iter().enumerate() {
                let q_value = modulus_q.value();
                let t_mod_q = t_value % q_value;
                let p_inv = p_inv_mod_q[i].operand;
                let combined_val = arith::mul_mod_u64(t_mod_q, p_inv, q_value);
                combined.push(arith::MultiplyUIntModOperand::new(combined_val, q_value));
            }
            (Some(t_inv_p), Some(combined))
        } else {
            (None, None)
        };

        let p_inv = base_p
            .moduli()
            .iter()
            .map(|modulus| 1.0f64 / modulus.value() as f64)
            .collect();
        let mut alpha_p_mod_q = vec![Vec::with_capacity(size_q); size_p + 1];
        for (_i, modulus_q) in base_q.moduli().iter().enumerate() {
            let q_val = modulus_q.value();
            let p_mod_q = base_p.base_prod().mod_u64(q_val);
            for alpha in 0..=size_p {
                let val = arith::mul_mod_u64(alpha as u64 % q_val, p_mod_q, q_val);
                alpha_p_mod_q[alpha].push(arith::MultiplyUIntModOperand::new(val, q_val));
            }
        }

        Ok(Self {
            base_p,
            base_q_to_p,
            base_p_to_q,
            q_hat_inv_mod_q,
            q_hat_mod_p,
            alpha_q_mod_p,
            q_inv,
            p_inv_mod_q,
            p_hat_inv_mod_p,
            p_hat_mod_q,
            t_inv_mod_p,
            t_p_inv_mod_q,
            p_inv,
            alpha_p_mod_q,
        })
    }
}

#[derive(Clone, Debug)]
struct ScaleAndRoundTables {
    t: u64,
    t_is_power_of_two: bool,
    q_msb_half: Option<u32>,
    t_q_hat_inv_modq_divq_modt: Vec<u64>,
    t_q_hat_inv_modq_divq_frac: Vec<f64>,
    t_q_hat_inv_modq_b_divq_modt: Option<Vec<u64>>,
    t_q_hat_inv_modq_b_divq_frac: Option<Vec<f64>>,
    t_inv: f64,
}

impl ScaleAndRoundTables {
    fn new(base_q: &RnsBase, t: Modulus) -> Result<Self, RnsError> {
        let size_q = base_q.len();
        let t_value = t.value();
        let t_is_power_of_two = t_value.is_power_of_two();

        let q_msb = base_q
            .moduli()
            .iter()
            .map(|modulus| modulus.bit_count())
            .max()
            .unwrap_or(0);
        let size_q_msb = msb_u64(size_q as u64);
        let use_b_split = q_msb + size_q_msb >= 52;
        let q_msb_half = if use_b_split { Some(q_msb >> 1) } else { None };

        let mut t_q_hat_inv_modq_divq_modt = vec![0u64; size_q];
        let mut t_q_hat_inv_modq_divq_frac = vec![0f64; size_q];
        let mut t_q_hat_inv_modq_b_divq_modt = if use_b_split {
            Some(vec![0u64; size_q])
        } else {
            None
        };
        let mut t_q_hat_inv_modq_b_divq_frac = if use_b_split {
            Some(vec![0f64; size_q])
        } else {
            None
        };

        for (i, modulus_q) in base_q.moduli().iter().enumerate() {
            let qi = modulus_q.value();
            let inv = base_q.inv_punctured_prod_mod_base()[i];
            let t_q_hat_inv = BigUint::from_u64(inv).mul_u64(t_value);
            let (quotient, remainder) = t_q_hat_inv.div_mod_u64(qi);
            t_q_hat_inv_modq_divq_modt[i] = quotient.mod_u64(t_value);
            t_q_hat_inv_modq_divq_frac[i] = (remainder as f64) / (qi as f64);

            if let Some(shift) = q_msb_half {
                let scale = 1u64 << shift;
                let t_q_hat_inv_b = t_q_hat_inv.mul_u64(scale);
                let (quotient_b, remainder_b) = t_q_hat_inv_b.div_mod_u64(qi);
                if let Some(ref mut vals) = t_q_hat_inv_modq_b_divq_modt {
                    vals[i] = quotient_b.mod_u64(t_value);
                }
                if let Some(ref mut vals) = t_q_hat_inv_modq_b_divq_frac {
                    vals[i] = (remainder_b as f64) / (qi as f64);
                }
            }
        }

        Ok(Self {
            t: t_value,
            t_is_power_of_two,
            q_msb_half,
            t_q_hat_inv_modq_divq_modt,
            t_q_hat_inv_modq_divq_frac,
            t_q_hat_inv_modq_b_divq_modt,
            t_q_hat_inv_modq_b_divq_frac,
            t_inv: 1.0 / (t_value as f64),
        })
    }
}

#[derive(Clone, Debug)]
struct ApproxScaleTables {
    base_p: RnsBase,
    tps_hat_inv_mods_divs_modp: Vec<Vec<arith::MultiplyUIntModOperand>>,
}

#[derive(Clone, Debug)]
struct HpsScaleTables {
    t_rs_hat_inv_mods_divs_modr: Vec<Vec<arith::MultiplyUIntModOperand>>,
    t_rs_hat_inv_mods_divs_frac: Vec<f64>,
}

impl HpsScaleTables {
    fn new(base_q: &RnsBase, base_r: &RnsBase, t: Modulus) -> Result<Self, RnsError> {
        let size_q = base_q.len();
        let size_r = base_r.len();
        if size_q == 0 || size_r == 0 {
            return Err(RnsError::EmptyBase);
        }

        let t_value = t.value();
        let q_prod = base_q.base_prod();
        let r_prod = base_r.base_prod();
        let qr_prod = q_prod.mul_big(r_prod);

        let mut t_rs_hat_inv_mods_divs_frac = vec![0f64; size_q];
        for (i, modulus_q) in base_q.moduli().iter().enumerate() {
            let qi = modulus_q.value();
            let qr_div_qi = qr_prod.div_mod_u64(qi).0;
            let inv =
                numth::mod_inverse(qr_div_qi.mod_u64(qi), qi).ok_or(RnsError::MissingInverse)?;
            let r_mod_q = r_prod.mod_u64(qi);
            let tmp = (r_mod_q as u128)
                .wrapping_mul((t_value % qi) as u128)
                .wrapping_mul(inv as u128)
                % (qi as u128);
            t_rs_hat_inv_mods_divs_frac[i] = (tmp as f64) / (qi as f64);
        }

        let rt = r_prod.mul_u64(t_value);
        let mut t_rs_hat_inv_mods_divs_modr = Vec::with_capacity(size_r);
        for modulus_r in base_r.moduli().iter() {
            let rj = modulus_r.value();
            let mut row = Vec::with_capacity(size_q + 1);
            for modulus_q in base_q.moduli().iter() {
                let qi = modulus_q.value();
                let qr_div_qi = qr_prod.div_mod_u64(qi).0;
                let inv = numth::mod_inverse(qr_div_qi.mod_u64(qi), qi)
                    .ok_or(RnsError::MissingInverse)?;
                let t_rs_hat_inv = rt.mul_u64(inv);
                let (quotient, _) = t_rs_hat_inv.div_mod_u64(qi);
                let value = quotient.mod_u64(rj);
                row.push(arith::MultiplyUIntModOperand::new(value, rj));
            }

            let qr_div_rj = qr_prod.div_mod_u64(rj).0;
            let inv_rj =
                numth::mod_inverse(qr_div_rj.mod_u64(rj), rj).ok_or(RnsError::MissingInverse)?;
            let t_rs_hat_inv = rt.mul_u64(inv_rj);
            let (quotient, _) = t_rs_hat_inv.div_mod_u64(rj);
            let value = quotient.mod_u64(rj);
            row.push(arith::MultiplyUIntModOperand::new(value, rj));

            t_rs_hat_inv_mods_divs_modr.push(row);
        }

        Ok(Self {
            t_rs_hat_inv_mods_divs_modr,
            t_rs_hat_inv_mods_divs_frac,
        })
    }
}

#[derive(Clone, Debug)]
struct HpsTables {
    base_r: RnsBase,
    switch_r: SwitchTables,
    scale: HpsScaleTables,
}

impl ApproxScaleTables {
    fn new(base_q: &RnsBase, base_p: RnsBase, t: Modulus) -> Result<Self, RnsError> {
        let size_q = base_q.len();
        let size_p = base_p.len();
        if size_q == 0 || size_p == 0 {
            return Err(RnsError::EmptyBase);
        }

        let t_value = t.value();
        let p_prod = base_p.base_prod().clone();
        let q_prod = base_q.base_prod();

        let mut tps_hat_inv_mods_divs_modp = Vec::with_capacity(size_p);
        for (j, modulus_p) in base_p.moduli().iter().enumerate() {
            let p_value = modulus_p.value();
            let mut row = Vec::with_capacity(size_q + 1);
            for (i, modulus_q) in base_q.moduli().iter().enumerate() {
                let q_value = modulus_q.value();
                let q_hat_mod_q = base_q.punctured_prod()[i].mod_u64(q_value);
                let p_mod_q = p_prod.mod_u64(q_value);
                let s_mod = arith::mul_mod_u64(p_mod_q, q_hat_mod_q, q_value);
                let inv = numth::mod_inverse(s_mod, q_value).ok_or(RnsError::MissingInverse)?;

                let tps = p_prod.clone().mul_u64(t_value).mul_u64(inv);
                let (quotient, _) = tps.div_mod_u64(q_value);
                row.push(arith::MultiplyUIntModOperand::new(
                    quotient.mod_u64(p_value),
                    p_value,
                ));
            }

            let p_hat_mod_p = base_p.punctured_prod()[j].mod_u64(p_value);
            let q_mod_p = q_prod.mod_u64(p_value);
            let s_mod = arith::mul_mod_u64(q_mod_p, p_hat_mod_p, p_value);
            let inv = numth::mod_inverse(s_mod, p_value).ok_or(RnsError::MissingInverse)?;
            let tps = p_prod.clone().mul_u64(t_value).mul_u64(inv);
            let (quotient, _) = tps.div_mod_u64(p_value);
            row.push(arith::MultiplyUIntModOperand::new(
                quotient.mod_u64(p_value),
                p_value,
            ));

            tps_hat_inv_mods_divs_modp.push(row);
        }

        Ok(Self {
            base_p,
            tps_hat_inv_mods_divs_modp,
        })
    }
}

#[derive(Clone, Debug)]
pub struct RnsToolConfig {
    pub base_q: RnsBase,
    pub base_t: Modulus,
    pub base_b: Option<RnsBase>,
    pub base_p: Option<RnsBase>,
    pub base_r: Option<RnsBase>,
    pub key_switch_modulus: Option<Modulus>,
    pub m_tilde: Option<u64>,
    pub m_sk: Option<Modulus>,
    pub enable_openfhe_switch: bool,
    pub switch_with_t: bool,
    pub enable_openfhe_scale: bool,
    pub enable_openfhe_approx_scale: bool,
    pub enable_openfhe_hps: bool,
}

impl RnsToolConfig {
    pub fn new(base_q: RnsBase, base_t: Modulus) -> Self {
        Self {
            base_q,
            base_t,
            base_b: None,
            base_p: None,
            base_r: None,
            key_switch_modulus: None,
            m_tilde: None,
            m_sk: None,
            enable_openfhe_switch: false,
            switch_with_t: false,
            enable_openfhe_scale: false,
            enable_openfhe_approx_scale: false,
            enable_openfhe_hps: false,
        }
    }

    pub fn with_base_b(mut self, base_b: RnsBase) -> Self {
        self.base_b = Some(base_b);
        self
    }

    pub fn with_base_p(mut self, base_p: RnsBase) -> Self {
        self.base_p = Some(base_p);
        self
    }

    pub fn with_base_r(mut self, base_r: RnsBase) -> Self {
        self.base_r = Some(base_r);
        self
    }

    pub fn with_key_switch_modulus(mut self, modulus: Modulus) -> Self {
        self.key_switch_modulus = Some(modulus);
        self
    }

    pub fn with_openfhe_bsk(mut self, m_tilde: u64, m_sk: Modulus) -> Self {
        self.m_tilde = Some(m_tilde);
        self.m_sk = Some(m_sk);
        self
    }

    pub fn with_openfhe_behz_auto(mut self, degree: usize) -> Result<Self, RnsError> {
        let (base_b, m_tilde, m_sk) = openfhe_behz_params(&self.base_q, self.base_t, degree)?;
        self.base_b = Some(base_b);
        self.m_tilde = Some(m_tilde);
        self.m_sk = Some(m_sk);
        Ok(self)
    }

    pub fn enable_openfhe_switch(mut self, with_t: bool) -> Self {
        self.enable_openfhe_switch = true;
        self.switch_with_t = with_t;
        self
    }

    pub fn enable_openfhe_scale(mut self) -> Self {
        self.enable_openfhe_scale = true;
        self
    }

    pub fn enable_openfhe_approx_scale(mut self) -> Self {
        self.enable_openfhe_approx_scale = true;
        self
    }

    pub fn enable_openfhe_hps(mut self) -> Self {
        self.enable_openfhe_hps = true;
        self
    }

    pub fn base_q(&self) -> &RnsBase {
        &self.base_q
    }
}

fn openfhe_behz_params(
    base_q: &RnsBase,
    base_t: Modulus,
    degree: usize,
) -> Result<(RnsBase, u64, Modulus), RnsError> {
    if base_q.is_empty() {
        return Err(RnsError::EmptyBase);
    }
    let two_n = (degree as u64)
        .checked_mul(2)
        .ok_or(RnsError::InvalidBase)?;
    if two_n == 0 {
        return Err(RnsError::InvalidBase);
    }

    // OpenFHE BEHZ: B has |Q| moduli, each a previous NTT-friendly prime.
    let mut b_values = Vec::with_capacity(base_q.len());
    let mut last = base_q.moduli()[base_q.len() - 1].value();
    for _ in 0..base_q.len() {
        let prime = numth::prev_ntt_prime(last, two_n).ok_or(RnsError::PrimeNotFound)?;
        b_values.push(prime);
        last = prime;
    }
    let base_b = RnsBase::from_values(b_values)?;

    // OpenFHE BEHZ: m_tilde = 2^16
    let m_tilde = 1u64 << 16;

    // Choose m_sk to satisfy B * m_sk >= 2n * t * Q
    let mut m_sk_value = numth::prev_ntt_prime(last, two_n).ok_or(RnsError::PrimeNotFound)?;
    let max_conv = base_q.base_prod().mul_u64(two_n).mul_u64(base_t.value());
    let mut b_times_msk = base_b.base_prod().mul_u64(m_sk_value);
    if b_times_msk < max_conv {
        let mut bits = 64 - m_sk_value.leading_zeros();
        loop {
            bits += 1;
            if bits >= 63 {
                return Err(RnsError::PrimeNotFound);
            }
            let candidate =
                numth::first_ntt_prime_with_bits(bits, two_n).ok_or(RnsError::PrimeNotFound)?;
            m_sk_value = numth::next_ntt_prime(candidate, two_n).ok_or(RnsError::PrimeNotFound)?;
            b_times_msk = base_b.base_prod().mul_u64(m_sk_value);
            if b_times_msk >= max_conv {
                break;
            }
        }
    }
    let m_sk = Modulus::new(m_sk_value)?;
    Ok((base_b, m_tilde, m_sk))
}

use silent_utils::arena::PooledArena;
use silent_utils::memory::MemoryPoolHandle;

#[derive(Debug)]
pub struct RnsToolScratch {
    arena: PooledArena,
}

impl RnsToolScratch {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let pool = MemoryPoolHandle::new();
        // Determine a default capacity if 0? Or let PooledArena handle it.
        // Let's mimic Arena behavior but with pool.
        Self {
            arena: PooledArena::new(pool, capacity.max(128)),
        }
    }

    pub fn reset(&mut self) {
        self.arena.reset();
    }

    fn alloc(&mut self, len: usize) -> &mut [u64] {
        self.arena.alloc(len)
    }

    fn alloc_two(&mut self, first: usize, second: usize) -> (&mut [u64], &mut [u64]) {
        let total = first + second;
        let slice = self.arena.alloc(total);
        slice.split_at_mut(first)
    }
}

impl Default for RnsToolScratch {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct RnsTool {
    base_q: RnsBase,
    base_t: Modulus,
    base_b: Option<RnsBase>,
    base_q_to_t: BaseConverter,
    base_q_to_b: Option<BaseConverter>,
    rounder: RnsRounder,
    openfhe: Option<OpenFheTables>,
    behz: Option<BehzTables>,
    switch_tables: Option<SwitchTables>,
    scale_tables: Option<ScaleAndRoundTables>,
    approx_scale_tables: Option<ApproxScaleTables>,
    hps_tables: Option<HpsTables>,
}

impl RnsTool {
    pub fn new(
        base_q: RnsBase,
        base_t: Modulus,
        base_b: Option<RnsBase>,
    ) -> Result<Self, RnsError> {
        let base_t_base = RnsBase::new(vec![base_t])?;
        let base_q_to_t = BaseConverter::new(base_q.clone(), base_t_base)?;
        let base_q_to_b = match base_b.clone() {
            Some(base) => Some(BaseConverter::new(base_q.clone(), base)?),
            None => None,
        };
        let rounder = RnsRounder::new(base_q.clone(), base_t.value())?;

        Ok(Self {
            base_q,
            base_t,
            base_b,
            base_q_to_t,
            base_q_to_b,
            rounder,
            openfhe: None,
            behz: None,
            switch_tables: None,
            scale_tables: None,
            approx_scale_tables: None,
            hps_tables: None,
        })
    }

    pub fn from_config(config: RnsToolConfig) -> Result<Self, RnsError> {
        let mut tool = RnsTool::new(config.base_q, config.base_t, config.base_b)?;

        if let (Some(m_tilde), Some(m_sk)) = (config.m_tilde, config.m_sk) {
            tool.enable_openfhe(m_tilde, m_sk)?;
        }

        if config.enable_openfhe_switch {
            let base_p = config.base_p.clone().ok_or(RnsError::InvalidBase)?;
            let t_opt = if config.switch_with_t {
                Some(tool.base_t)
            } else {
                None
            };
            tool.enable_openfhe_switch(base_p, t_opt)?;
        }

        if config.enable_openfhe_scale {
            tool.enable_openfhe_scale_and_round()?;
        }

        if config.enable_openfhe_approx_scale {
            let base_p = config.base_p.clone().ok_or(RnsError::InvalidBase)?;
            tool.enable_openfhe_approx_scale_and_round(base_p)?;
        }

        if config.enable_openfhe_hps {
            let base_r = config.base_r.clone().ok_or(RnsError::InvalidBase)?;
            tool.enable_openfhe_hps(base_r)?;
        }

        Ok(tool)
    }

    pub fn base_q(&self) -> &RnsBase {
        &self.base_q
    }

    pub fn base_t(&self) -> Modulus {
        self.base_t
    }

    pub fn base_b(&self) -> Option<&RnsBase> {
        self.base_b.as_ref()
    }

    pub fn base_p(&self) -> Option<&RnsBase> {
        self.switch_tables.as_ref().map(|tables| &tables.base_p)
    }

    pub fn base_r(&self) -> Option<&RnsBase> {
        self.hps_tables.as_ref().map(|tables| &tables.base_r)
    }

    pub fn base_q_to_p(&self) -> Option<&BaseConverter> {
        self.switch_tables
            .as_ref()
            .map(|tables| &tables.base_q_to_p)
    }

    pub fn base_bsk(&self) -> Option<&RnsBase> {
        self.openfhe.as_ref().map(|tables| &tables.base_bsk)
    }

    pub fn enable_openfhe_switch(
        &mut self,
        base_p: RnsBase,
        t: Option<Modulus>,
    ) -> Result<(), RnsError> {
        let tables = SwitchTables::new(&self.base_q, base_p, t)?;
        self.switch_tables = Some(tables);
        Ok(())
    }

    pub fn with_openfhe_switch(
        mut self,
        base_p: RnsBase,
        t: Option<Modulus>,
    ) -> Result<Self, RnsError> {
        self.enable_openfhe_switch(base_p, t)?;
        Ok(self)
    }

    pub fn enable_openfhe_scale_and_round(&mut self) -> Result<(), RnsError> {
        let tables = ScaleAndRoundTables::new(&self.base_q, self.base_t)?;
        self.scale_tables = Some(tables);
        Ok(())
    }

    pub fn with_openfhe_scale_and_round(mut self) -> Result<Self, RnsError> {
        self.enable_openfhe_scale_and_round()?;
        Ok(self)
    }

    pub fn enable_openfhe_approx_scale_and_round(
        &mut self,
        base_p: RnsBase,
    ) -> Result<(), RnsError> {
        let tables = ApproxScaleTables::new(&self.base_q, base_p, self.base_t)?;
        self.approx_scale_tables = Some(tables);
        Ok(())
    }

    pub fn with_openfhe_approx_scale_and_round(
        mut self,
        base_p: RnsBase,
    ) -> Result<Self, RnsError> {
        self.enable_openfhe_approx_scale_and_round(base_p)?;
        Ok(self)
    }

    pub fn enable_openfhe(&mut self, m_tilde: u64, m_sk: Modulus) -> Result<(), RnsError> {
        let base_b = self.base_b.clone().ok_or(RnsError::InvalidBase)?;
        let tables = OpenFheTables::new(&self.base_q, &base_b, self.base_t, m_tilde, m_sk)?;
        self.openfhe = Some(tables);
        let behz = BehzTables::new(&self.base_q, &base_b, m_tilde, m_sk)?;
        self.behz = Some(behz);
        Ok(())
    }

    pub fn with_openfhe(mut self, m_tilde: u64, m_sk: Modulus) -> Result<Self, RnsError> {
        self.enable_openfhe(m_tilde, m_sk)?;
        Ok(self)
    }

    pub fn enable_openfhe_hps(&mut self, base_r: RnsBase) -> Result<(), RnsError> {
        let switch_r = SwitchTables::new(&self.base_q, base_r.clone(), None)?;
        let scale = HpsScaleTables::new(&self.base_q, &base_r, self.base_t)?;
        self.hps_tables = Some(HpsTables {
            base_r,
            switch_r,
            scale,
        });
        Ok(())
    }

    pub fn fastbconv_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let converter = self.base_q_to_b.as_ref().ok_or(RnsError::InvalidBase)?;
        converter.fast_convert_array_into(input, count, output, scratch)
    }

    pub fn fastbconv_with_scratch<'a>(
        &self,
        input: &[u64],
        count: usize,
        scratch: &'a mut RnsToolScratch,
    ) -> Result<&'a mut [u64], RnsError> {
        let converter = self.base_q_to_b.as_ref().ok_or(RnsError::InvalidBase)?;
        let (output, temp) = scratch.alloc_two(
            converter.obase().len() * count,
            converter.ibase().len() * BaseConverter::BLOCK_SIZE,
        );
        self.fastbconv_into(input, count, output, temp)?;
        Ok(output)
    }

    pub fn fastbconv(&self, input: &[u64], count: usize) -> Result<Vec<u64>, RnsError> {
        let converter = self.base_q_to_b.as_ref().ok_or(RnsError::InvalidBase)?;
        let mut output = vec![0u64; converter.obase().len() * count];
        let mut scratch = vec![0u64; converter.ibase().len() * BaseConverter::BLOCK_SIZE];
        self.fastbconv_into(input, count, &mut output, &mut scratch)?;
        Ok(output)
    }

    /// Approximate CRT basis switch from base Q to base P (OpenFHE style).
    pub fn approx_switch_crt_basis_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        tables
            .base_q_to_p
            .fast_convert_array_into(input, count, output, scratch)
    }

    /// Approximate CRT basis switch from base P to base Q (OpenFHE style).
    fn approx_switch_crt_basis_p_to_q_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        tables
            .base_p_to_q
            .fast_convert_array_into(input, count, output, scratch)
    }

    pub fn approx_switch_crt_basis_with_scratch<'a>(
        &self,
        input: &[u64],
        count: usize,
        scratch: &'a mut RnsToolScratch,
    ) -> Result<&'a mut [u64], RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_p = tables.base_p.len();
        let (output, temp) = scratch.alloc_two(
            size_p * count,
            self.base_q.len() * BaseConverter::BLOCK_SIZE,
        );
        self.approx_switch_crt_basis_into(input, count, output, temp)?;
        Ok(output)
    }

    pub fn approx_switch_crt_basis(
        &self,
        input: &[u64],
        count: usize,
    ) -> Result<Vec<u64>, RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_p = tables.base_p.len();
        let mut output = vec![0u64; size_p * count];
        let mut scratch = vec![0u64; self.base_q.len() * BaseConverter::BLOCK_SIZE];
        self.approx_switch_crt_basis_into(input, count, &mut output, &mut scratch)?;
        Ok(output)
    }

    fn switch_crt_basis_with_tables(
        &self,
        tables: &SwitchTables,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let size_q = self.base_q.len();
        let size_p = tables.base_p.len();
        if input.len() != size_q * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != size_p * count {
            return Err(RnsError::InvalidResidues);
        }
        if scratch.len() < size_q {
            return Err(RnsError::InvalidResidues);
        }

        let q_moduli = self.base_q.moduli();
        let p_moduli = &tables.base_p.moduli();

        let temp = &mut scratch[..size_q];

        for coeff in 0..count {
            let mut nu = 0.5f64;
            for i in 0..size_q {
                let modulus = q_moduli[i].value();
                let x = input[i * count + coeff];
                let shoup = tables.q_hat_inv_mod_q[i];
                let t = arith::mul_mod_shoup(x, shoup.operand, shoup.quotient, modulus);
                temp[i] = t;
                nu += (t as f64) * tables.q_inv[i];
            }
            let mut alpha = nu.floor() as usize;
            if alpha > size_q {
                alpha = size_q;
            }
            for j in 0..size_p {
                let modulus = &p_moduli[j];
                let mut acc_128: u128 = 0;
                for i in 0..size_q {
                    let q_hat = tables.q_hat_mod_p[i][j].operand;
                    acc_128 += (temp[i] as u128) * (q_hat as u128);
                }
                let acc = modulus.reduce_u128(acc_128);

                let alpha_term = tables.alpha_q_mod_p[alpha][j].operand;
                output[j * count + coeff] = arith::sub_mod(acc, alpha_term, modulus.value());
            }
        }
        Ok(())
    }

    /// CRT basis switch with centered correction from base Q to base P (OpenFHE style).
    pub fn switch_crt_basis_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        self.switch_crt_basis_with_tables(tables, input, count, output, scratch)
    }

    fn switch_crt_basis_p_to_q_with_tables(
        &self,
        tables: &SwitchTables,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let size_q = self.base_q.len();
        let size_p = tables.base_p.len();
        if input.len() != size_p * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != size_q * count {
            return Err(RnsError::InvalidResidues);
        }
        if scratch.len() < size_p {
            return Err(RnsError::InvalidResidues);
        }

        let temp = &mut scratch[..size_p];
        let q_moduli = self.base_q.moduli();
        let p_moduli = &tables.base_p.moduli();

        if size_p == 2 {
            let p_mod0 = p_moduli[0].value();
            let p_mod1 = p_moduli[1].value();
            let shoup0 = tables.p_hat_inv_mod_p[0];
            let shoup1 = tables.p_hat_inv_mod_p[1];
            let p_inv0 = tables.p_inv[0];
            let p_inv1 = tables.p_inv[1];
            let p_hat_mod_q0 = &tables.p_hat_mod_q[0];
            let p_hat_mod_q1 = &tables.p_hat_mod_q[1];

            for coeff in 0..count {
                let x0 = input[coeff];
                let x1 = input[count + coeff];
                let t0 = arith::mul_mod_shoup(x0, shoup0.operand, shoup0.quotient, p_mod0);
                let t1 = arith::mul_mod_shoup(x1, shoup1.operand, shoup1.quotient, p_mod1);

                let mut nu = 0.5f64;
                nu += (t0 as f64) * p_inv0;
                nu += (t1 as f64) * p_inv1;
                let mut alpha = nu.floor() as usize;
                if alpha > size_p {
                    alpha = size_p;
                }

                for i in 0..size_q {
                    let modulus = &q_moduli[i];
                    let modulus_val = modulus.value();
                    let term0 = arith::mul_mod_shoup_lazy(
                        t0,
                        p_hat_mod_q0[i].operand,
                        p_hat_mod_q0[i].quotient,
                        modulus_val,
                    );
                    let term1 = arith::mul_mod_shoup_lazy(
                        t1,
                        p_hat_mod_q1[i].operand,
                        p_hat_mod_q1[i].quotient,
                        modulus_val,
                    );
                    let acc = modulus.reduce_u128((term0 as u128) + (term1 as u128));
                    let alpha_term = tables.alpha_p_mod_q[alpha][i].operand;
                    output[i * count + coeff] = arith::sub_mod(acc, alpha_term, modulus_val);
                }
            }
            return Ok(());
        }

        for coeff in 0..count {
            let mut nu = 0.5f64;
            for j in 0..size_p {
                unsafe {
                    let modulus = p_moduli.get_unchecked(j).value();
                    let x = *input.get_unchecked(j * count + coeff);
                    let shoup = tables.p_hat_inv_mod_p.get_unchecked(j);
                    let t = arith::mul_mod_shoup(x, shoup.operand, shoup.quotient, modulus);
                    *temp.get_unchecked_mut(j) = t;
                    nu += (t as f64) * *tables.p_inv.get_unchecked(j);
                }
            }
            let mut alpha = nu.floor() as usize;
            if alpha > size_p {
                alpha = size_p;
            }
            for i in 0..size_q {
                unsafe {
                    let modulus = q_moduli.get_unchecked(i);
                    let modulus_val = modulus.value();
                    let mut acc_128: u128 = 0;
                    for j in 0..size_p {
                        let shoup = tables.p_hat_mod_q.get_unchecked(j).get_unchecked(i);
                        let term = arith::mul_mod_shoup_lazy(
                            *temp.get_unchecked(j),
                            shoup.operand,
                            shoup.quotient,
                            modulus_val,
                        );
                        acc_128 += term as u128;
                    }
                    let acc = modulus.reduce_u128(acc_128);

                    let alpha_shoup = tables.alpha_p_mod_q.get_unchecked(alpha).get_unchecked(i);
                    let alpha_term = alpha_shoup.operand;

                    *output.get_unchecked_mut(i * count + coeff) =
                        arith::sub_mod(acc, alpha_term, modulus_val);
                }
            }
        }
        Ok(())
    }

    pub fn switch_crt_basis_p_to_q_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
        _unused: bool,
    ) -> Result<(), RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        self.switch_crt_basis_p_to_q_with_tables(tables, input, count, output, scratch)
    }

    pub fn switch_crt_basis_with_scratch<'a>(
        &self,
        input: &[u64],
        count: usize,
        scratch: &'a mut RnsToolScratch,
    ) -> Result<&'a mut [u64], RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_p = tables.base_p.len();
        let (output, temp) = scratch.alloc_two(size_p * count, self.base_q.len());
        self.switch_crt_basis_into(input, count, output, temp)?;
        Ok(output)
    }

    pub fn switch_crt_basis(&self, input: &[u64], count: usize) -> Result<Vec<u64>, RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_p = tables.base_p.len();
        let mut output = vec![0u64; size_p * count];
        let mut scratch = vec![0u64; self.base_q.len()];
        self.switch_crt_basis_into(input, count, &mut output, &mut scratch)?;
        Ok(output)
    }

    pub fn switch_crt_basis_r_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let tables = self.hps_tables.as_ref().ok_or(RnsError::OpenFheDisabled)?;
        self.switch_crt_basis_with_tables(&tables.switch_r, input, count, output, scratch)
    }

    pub fn switch_crt_basis_r_to_q_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let tables = self.hps_tables.as_ref().ok_or(RnsError::OpenFheDisabled)?;
        self.switch_crt_basis_p_to_q_with_tables(&tables.switch_r, input, count, output, scratch)
    }

    /// Expand CRT basis: expand from base Q to base Q∪R using SwitchCRTBasis with centered correction.
    pub fn expand_crt_basis_r_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let tables = self.hps_tables.as_ref().ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_r = tables.base_r.len();
        if input.len() != size_q * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != (size_q + size_r) * count {
            return Err(RnsError::InvalidResidues);
        }

        let (out_q, out_r) = output.split_at_mut(size_q * count);
        out_q.copy_from_slice(input);
        self.switch_crt_basis_r_into(input, count, out_r, scratch)
    }

    /// HPS scale-and-round: from base Q∪R to base R.
    pub fn hps_scale_and_round_qr_to_r_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
    ) -> Result<(), RnsError> {
        let tables = self.hps_tables.as_ref().ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_r = tables.base_r.len();
        if input.len() != (size_q + size_r) * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != size_r * count {
            return Err(RnsError::InvalidResidues);
        }

        let moduli_r = tables.base_r.moduli();
        let frac = &tables.scale.t_rs_hat_inv_mods_divs_frac;
        let mods = &tables.scale.t_rs_hat_inv_mods_divs_modr;

        for coeff in 0..count {
            let mut nu = 0.5f64;
            for i in 0..size_q {
                let xi = input[i * count + coeff];
                nu += frac[i] * (xi as f64);
            }
            let alpha = nu.floor() as u64;

            for j in 0..size_r {
                let modulus = &moduli_r[j];
                let modulus_val = modulus.value();
                let mut acc_128: u128 = 0;
                for i in 0..size_q {
                    let xi = input[i * count + coeff];
                    let xi_mod = if xi >= modulus_val {
                        xi % modulus_val
                    } else {
                        xi
                    };
                    let shoup = mods[j][i];
                    let term = arith::mul_mod_shoup_lazy(
                        xi_mod,
                        shoup.operand,
                        shoup.quotient,
                        modulus_val,
                    );
                    acc_128 += term as u128;
                }
                let xr = input[(size_q + j) * count + coeff];
                let xr_mod = if xr >= modulus_val {
                    xr % modulus_val
                } else {
                    xr
                };
                let shoup_r = mods[j][size_q];
                let term_r = arith::mul_mod_shoup_lazy(
                    xr_mod,
                    shoup_r.operand,
                    shoup_r.quotient,
                    modulus_val,
                );
                acc_128 += term_r as u128;

                let acc = modulus.reduce_u128(acc_128);
                let alpha_mod = if alpha >= modulus_val {
                    alpha % modulus_val
                } else {
                    alpha
                };
                output[j * count + coeff] = arith::add_mod(acc, alpha_mod, modulus_val);
            }
        }
        Ok(())
    }

    /// Approximate modulus-up: expand from base Q to base Q∪P using ApproxSwitchCRTBasis.
    pub fn approx_mod_up_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_p = tables.base_p.len();
        if input.len() != size_q * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != (size_q + size_p) * count {
            return Err(RnsError::InvalidResidues);
        }

        let (out_q, out_p) = output.split_at_mut(size_q * count);
        out_q.copy_from_slice(input);
        self.approx_switch_crt_basis_into(input, count, out_p, scratch)
    }

    pub fn approx_mod_up_with_scratch<'a>(
        &self,
        input: &[u64],
        count: usize,
        scratch: &'a mut RnsToolScratch,
    ) -> Result<&'a mut [u64], RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_p = tables.base_p.len();
        let (output, temp) = scratch.alloc_two(
            (size_q + size_p) * count,
            size_q * BaseConverter::BLOCK_SIZE,
        );
        self.approx_mod_up_into(input, count, output, temp)?;
        Ok(output)
    }

    pub fn approx_mod_up(&self, input: &[u64], count: usize) -> Result<Vec<u64>, RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_p = tables.base_p.len();
        let mut output = vec![0u64; (size_q + size_p) * count];
        let mut scratch = vec![0u64; size_q * BaseConverter::BLOCK_SIZE];
        self.approx_mod_up_into(input, count, &mut output, &mut scratch)?;
        Ok(output)
    }

    /// Expand CRT basis: expand from base Q to base Q∪P using SwitchCRTBasis with centered correction.
    pub fn expand_crt_basis_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_p = tables.base_p.len();
        if input.len() != size_q * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != (size_q + size_p) * count {
            return Err(RnsError::InvalidResidues);
        }

        let (out_q, out_p) = output.split_at_mut(size_q * count);
        out_q.copy_from_slice(input);
        self.switch_crt_basis_into(input, count, out_p, scratch)
    }

    pub fn expand_crt_basis_with_scratch<'a>(
        &self,
        input: &[u64],
        count: usize,
        scratch: &'a mut RnsToolScratch,
    ) -> Result<&'a mut [u64], RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_p = tables.base_p.len();
        let (output, temp) = scratch.alloc_two((size_q + size_p) * count, size_q);
        self.expand_crt_basis_into(input, count, output, temp)?;
        Ok(output)
    }

    pub fn expand_crt_basis(&self, input: &[u64], count: usize) -> Result<Vec<u64>, RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_p = tables.base_p.len();
        let mut output = vec![0u64; (size_q + size_p) * count];
        let mut scratch = vec![0u64; size_q];
        self.expand_crt_basis_into(input, count, &mut output, &mut scratch)?;
        Ok(output)
    }

    /// Expand CRT basis with reverse ordering: output is P||Q (OpenFHE style).
    pub fn expand_crt_basis_reverse_order_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_p = tables.base_p.len();
        if input.len() != size_q * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != (size_q + size_p) * count {
            return Err(RnsError::InvalidResidues);
        }

        let (out_p, out_q) = output.split_at_mut(size_p * count);
        self.switch_crt_basis_into(input, count, out_p, scratch)?;
        out_q.copy_from_slice(input);
        Ok(())
    }

    pub fn expand_crt_basis_reverse_order_with_scratch<'a>(
        &self,
        input: &[u64],
        count: usize,
        scratch: &'a mut RnsToolScratch,
    ) -> Result<&'a mut [u64], RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_p = tables.base_p.len();
        let (output, temp) = scratch.alloc_two((size_q + size_p) * count, size_q);
        self.expand_crt_basis_reverse_order_into(input, count, output, temp)?;
        Ok(output)
    }

    pub fn expand_crt_basis_reverse_order(
        &self,
        input: &[u64],
        count: usize,
    ) -> Result<Vec<u64>, RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_p = tables.base_p.len();
        let mut output = vec![0u64; (size_q + size_p) * count];
        let mut scratch = vec![0u64; size_q];
        self.expand_crt_basis_reverse_order_into(input, count, &mut output, &mut scratch)?;
        Ok(output)
    }

    /// Approximate modulus down from base Q∪P to base Q (OpenFHE style).
    pub fn approx_mod_down_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
        _arg1: bool,
        _arg2: bool,
    ) -> Result<(), RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_p = tables.base_p.len();
        if input.len() != (size_q + size_p) * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != size_q * count {
            return Err(RnsError::InvalidResidues);
        }
        if scratch.len() < size_p + size_q {
            return Err(RnsError::InvalidResidues);
        }

        let (input_q, input_p) = input.split_at(size_q * count);
        let p_moduli = &tables.base_p.moduli();
        let q_moduli = self.base_q.moduli();

        {
            let _scope = profile::Scope::new(profile::Kind::ModDownCoeffSwitch);
            if let Some(t_inv_mod_p) = tables.t_inv_mod_p.as_ref() {
                let fast_scratch = size_p * count + size_p * BaseConverter::BLOCK_SIZE;
                if scratch.len() >= fast_scratch {
                    let (temp_p, conv_scratch) = scratch.split_at_mut(size_p * count);
                    for j in 0..size_p {
                        let modulus = p_moduli[j].value();
                        let shoup = &t_inv_mod_p[j];
                        let start = j * count;
                        for k in 0..count {
                            let value = input_p[start + k];
                            temp_p[start + k] =
                                arith::mul_mod_shoup(value, shoup.operand, shoup.quotient, modulus);
                        }
                    }
                    self.approx_switch_crt_basis_p_to_q_into(temp_p, count, output, conv_scratch)?;
                } else {
                    let mut local =
                        vec![0u64; size_p + size_p * BaseConverter::BLOCK_SIZE + size_q];
                    let (temp_p, rest) = local.split_at_mut(size_p);
                    let (conv_scratch, temp_q) =
                        rest.split_at_mut(size_p * BaseConverter::BLOCK_SIZE);
                    for coeff in 0..count {
                        for j in 0..size_p {
                            let modulus = p_moduli[j].value();
                            let shoup = &t_inv_mod_p[j];
                            let value = input_p[j * count + coeff];
                            temp_p[j] =
                                arith::mul_mod_shoup(value, shoup.operand, shoup.quotient, modulus);
                        }
                        self.approx_switch_crt_basis_p_to_q_into(temp_p, 1, temp_q, conv_scratch)?;
                        for i in 0..size_q {
                            output[i * count + coeff] = temp_q[i];
                        }
                    }
                }
            } else {
                self.approx_switch_crt_basis_p_to_q_into(input_p, count, output, scratch)?;
            }
        }

        {
            let _scope = profile::Scope::new(profile::Kind::ModDownCoeffCombine);
            if let Some(t_p_inv) = tables.t_p_inv_mod_q.as_ref() {
                for i in 0..size_q {
                    let modulus = q_moduli[i].value();
                    let p_inv = &tables.p_inv_mod_q[i];
                    let shoup = &t_p_inv[i];
                    let start = i * count;
                    for k in 0..count {
                        let q_val = input_q[start + k];
                        let q_scaled =
                            arith::mul_mod_shoup(q_val, p_inv.operand, p_inv.quotient, modulus);
                        let tmp_scaled = arith::mul_mod_shoup(
                            output[start + k],
                            shoup.operand,
                            shoup.quotient,
                            modulus,
                        );
                        output[start + k] = arith::sub_mod(q_scaled, tmp_scaled, modulus);
                    }
                }
            } else {
                for i in 0..size_q {
                    let modulus = q_moduli[i].value();
                    let p_inv = &tables.p_inv_mod_q[i];
                    let start = i * count;
                    for k in 0..count {
                        let q_val = input_q[start + k];
                        let diff = arith::sub_mod(q_val, output[start + k], modulus);
                        output[start + k] =
                            arith::mul_mod_shoup(diff, p_inv.operand, p_inv.quotient, modulus);
                    }
                }
            }
        }

        Ok(())
    }

    /// Approximate modulus down from base Q∪P to base Q with input/output in NTT form.
    /// This mirrors OpenFHE's evaluation-domain ApproxModDown: only P limbs are inverse-NTT'd.
    pub fn approx_mod_down_ntt_into(
        &self,
        input: &mut [u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
        ntt_tables: &[NttTables],
    ) -> Result<(), RnsError> {
        let tables = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_p = tables.base_p.len();
        if input.len() != (size_q + size_p) * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != size_q * count {
            return Err(RnsError::InvalidResidues);
        }
        if scratch.len() < size_p {
            return Err(RnsError::InvalidResidues);
        }
        if ntt_tables.len() < size_q + size_p {
            return Err(RnsError::InvalidResidues);
        }

        let (input_q, input_p) = input.split_at_mut(size_q * count);
        let p_moduli = &tables.base_p.moduli();
        let q_moduli = self.base_q.moduli();

        {
            let _scope = profile::Scope::new(profile::Kind::ModDownNttInttP);
            // INTT only P limbs (OpenFHE-style).
            for limb in 0..size_p {
                let slice = &mut input_p[limb * count..(limb + 1) * count];
                ntt_inverse(slice, &ntt_tables[size_q + limb]);
            }
        }

        if let Some(t_inv_mod_p) = tables.t_inv_mod_p.as_ref() {
            let _scope = profile::Scope::new(profile::Kind::ModDownNttScaleP);
            for j in 0..size_p {
                let modulus = p_moduli[j].value();
                let shoup = &t_inv_mod_p[j];
                let start = j * count;
                for k in 0..count {
                    let value = input_p[start + k];
                    input_p[start + k] =
                        arith::mul_mod_shoup(value, shoup.operand, shoup.quotient, modulus);
                }
            }
        }

        {
            let _scope = profile::Scope::new(profile::Kind::ModDownNttSwitch);
            // Convert P -> Q in coefficient domain.
            self.approx_switch_crt_basis_p_to_q_into(input_p, count, output, scratch)?;
        }

        {
            let _scope = profile::Scope::new(profile::Kind::ModDownNttQntt);
            // NTT Q limbs of the switched result.
            for limb in 0..size_q {
                let slice = &mut output[limb * count..(limb + 1) * count];
                ntt_forward(slice, &ntt_tables[limb]);
            }
        }

        {
            let _scope = profile::Scope::new(profile::Kind::ModDownNttCombine);
            // Combine with Q part in NTT domain.
            if let Some(t_p_inv) = tables.t_p_inv_mod_q.as_ref() {
                for (i, (modulus, (p_inv, shoup))) in q_moduli
                    .iter()
                    .zip(tables.p_inv_mod_q.iter().zip(t_p_inv.iter()))
                    .enumerate()
                {
                    let modulus_val = modulus.value();
                    let start = i * count;
                    let out_slice = &mut output[start..start + count];
                    let in_slice = &input_q[start..start + count];

                    for (out_val, &minput) in out_slice.iter_mut().zip(in_slice.iter()) {
                        let q_scaled = arith::mul_mod_shoup(
                            minput,
                            p_inv.operand,
                            p_inv.quotient,
                            modulus_val,
                        );
                        let tmp_scaled = arith::mul_mod_shoup(
                            *out_val,
                            shoup.operand,
                            shoup.quotient,
                            modulus_val,
                        );
                        *out_val = arith::sub_mod(q_scaled, tmp_scaled, modulus_val);
                    }
                }
            } else {
                for (i, (modulus, p_inv)) in
                    q_moduli.iter().zip(tables.p_inv_mod_q.iter()).enumerate()
                {
                    let modulus_val = modulus.value();
                    let start = i * count;
                    let out_slice = &mut output[start..start + count];
                    let in_slice = &input_q[start..start + count];

                    for (out_val, &minput) in out_slice.iter_mut().zip(in_slice.iter()) {
                        let diff = arith::sub_mod(minput, *out_val, modulus_val);
                        *out_val =
                            arith::mul_mod_shoup(diff, p_inv.operand, p_inv.quotient, modulus_val);
                    }
                }
            }
        }

        Ok(())
    }

    pub fn approx_mod_down_with_scratch<'a>(
        &self,
        input: &[u64],
        count: usize,
        scratch: &'a mut RnsToolScratch,
    ) -> Result<&'a mut [u64], RnsError> {
        let size_q = self.base_q.len();
        let size_p = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?
            .base_p
            .len();
        let scratch_len = size_p * count + size_p * BaseConverter::BLOCK_SIZE;
        let (output, temp) = scratch.alloc_two(size_q * count, scratch_len);
        self.approx_mod_down_into(input, count, output, temp, false, false)?;
        Ok(output)
    }

    pub fn approx_mod_down(&self, input: &[u64], count: usize) -> Result<Vec<u64>, RnsError> {
        let size_q = self.base_q.len();
        let size_p = self
            .switch_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?
            .base_p
            .len();
        let mut output = vec![0u64; size_q * count];
        let scratch_len = size_p * count + size_p * BaseConverter::BLOCK_SIZE;
        let mut scratch = vec![0u64; scratch_len];
        self.approx_mod_down_into(input, count, &mut output, &mut scratch, false, false)?;
        Ok(output)
    }

    pub fn dump_profile() {
        profile::dump();
    }

    pub fn fast_base_conv_q_to_bsk_montgomery(
        &self,
        input: &[u64],
        count: usize,
    ) -> Result<Vec<u64>, RnsError> {
        let tables = self.openfhe.as_ref().ok_or(RnsError::OpenFheDisabled)?;
        let num_q = self.base_q.len();
        if input.len() != num_q * count {
            return Err(RnsError::InvalidResidues);
        }
        let num_bsk = tables.base_bsk.len();
        let mut output = vec![0u64; num_bsk * count];
        let scratch_len = num_q * count + count;
        let mut scratch = vec![0u64; scratch_len];
        self.fast_base_conv_q_to_bsk_montgomery_into(input, count, &mut output, &mut scratch)?;
        Ok(output)
    }

    pub fn fast_base_conv_q_to_bsk_montgomery_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let tables = self.openfhe.as_ref().ok_or(RnsError::OpenFheDisabled)?;
        let num_q = self.base_q.len();
        if input.len() != num_q * count {
            return Err(RnsError::InvalidResidues);
        }
        let num_bsk = tables.base_bsk.len();
        if output.len() != num_bsk * count {
            return Err(RnsError::InvalidResidues);
        }
        let needed = num_q * count + count;
        if scratch.len() < needed {
            return Err(RnsError::InvalidResidues);
        }

        let (xi_mtilde, rest) = scratch.split_at_mut(num_q * count);
        let r_mtilde = &mut rest[..count];
        r_mtilde.fill(0);

        let mask = tables.m_tilde - 1;

        for i in 0..num_q {
            let modulus = self.base_q.moduli()[i].value();
            let factor = tables.m_tilde_q_hat_inv_mod_q[i];
            let q_hat_mod_m_tilde = tables.q_hat_mod_m_tilde[i] as u128;
            for k in 0..count {
                let x = input[i * count + k];
                let tmp = arith::mul_mod_u64(x, factor, modulus);
                xi_mtilde[i * count + k] = tmp;
                let prod = (tmp as u128) * q_hat_mod_m_tilde;
                r_mtilde[k] = r_mtilde[k].wrapping_add(prod as u64);
            }
        }

        for k in 0..count {
            r_mtilde[k] &= mask;
            r_mtilde[k] = r_mtilde[k].wrapping_mul(tables.neg_q_inv_mod_m_tilde) & mask;
        }

        for j in 0..num_bsk {
            let modulus = tables.base_bsk.moduli()[j].value();
            let q_mod_bsk = tables.q_mod_bsk[j];
            let mtilde_inv = tables.m_tilde_inv_mod_bsk[j];
            for k in 0..count {
                let mut acc = 0u64;
                for i in 0..num_q {
                    let term = arith::mul_mod_u64(
                        xi_mtilde[i * count + k],
                        tables.q_hat_mod_bsk[i][j],
                        modulus,
                    );
                    acc = arith::add_mod(acc, term, modulus);
                }
                let mut r = r_mtilde[k];
                if r >= (tables.m_tilde >> 1) {
                    r = r.wrapping_add(modulus).wrapping_sub(tables.m_tilde);
                }
                let r_mod = r % modulus;
                let mut corrected = arith::mul_mod_u64(r_mod, q_mod_bsk, modulus);
                corrected = arith::add_mod(corrected, acc, modulus);
                output[j * count + k] = arith::mul_mod_u64(corrected, mtilde_inv, modulus);
            }
        }

        Ok(())
    }

    pub fn fast_rns_floor_q(&self, input: &[u64], count: usize) -> Result<Vec<u64>, RnsError> {
        let tables = self.openfhe.as_ref().ok_or(RnsError::OpenFheDisabled)?;
        let num_q = self.base_q.len();
        let num_bsk = tables.base_bsk.len();
        if input.len() != (num_q + num_bsk) * count {
            return Err(RnsError::InvalidResidues);
        }
        let mut output = vec![0u64; num_bsk * count];
        let mut scratch = vec![0u64; num_q * count];
        self.fast_rns_floor_q_into(input, count, &mut output, &mut scratch)?;
        Ok(output)
    }

    pub fn fast_rns_floor_q_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let tables = self.openfhe.as_ref().ok_or(RnsError::OpenFheDisabled)?;
        let num_q = self.base_q.len();
        let num_bsk = tables.base_bsk.len();
        if input.len() != (num_q + num_bsk) * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != num_bsk * count {
            return Err(RnsError::InvalidResidues);
        }
        if scratch.len() < num_q * count {
            return Err(RnsError::InvalidResidues);
        }

        let (input_q, input_bsk) = input.split_at(num_q * count);
        let twisted = &mut scratch[..num_q * count];
        for i in 0..num_q {
            let modulus = self.base_q.moduli()[i].value();
            let factor = tables.t_q_hat_inv_mod_q[i];
            for k in 0..count {
                let x = input_q[i * count + k];
                twisted[i * count + k] = arith::mul_mod_u64(x, factor, modulus);
            }
        }

        for j in 0..num_bsk {
            let modulus = tables.base_bsk.moduli()[j].value();
            let t_div_q = tables.t_q_inv_mod_bsk[j];
            for k in 0..count {
                let mut acc = 0u64;
                for i in 0..num_q {
                    let term = arith::mul_mod_u64(
                        twisted[i * count + k],
                        tables.q_inv_mod_bsk[i][j],
                        modulus,
                    );
                    acc = arith::add_mod(acc, term, modulus);
                }
                let mut value = arith::mul_mod_u64(input_bsk[j * count + k], t_div_q, modulus);
                value = arith::sub_mod(value, acc, modulus);
                output[j * count + k] = value;
            }
        }

        Ok(())
    }

    pub fn fast_base_conv_sk(&self, input: &[u64], count: usize) -> Result<Vec<u64>, RnsError> {
        let tables = self.openfhe.as_ref().ok_or(RnsError::OpenFheDisabled)?;
        let base_b = &tables.base_b;
        let num_b = base_b.len();
        let num_bsk = tables.base_bsk.len();
        if input.len() != num_bsk * count {
            return Err(RnsError::InvalidResidues);
        }
        let num_q = self.base_q.len();
        let mut output = vec![0u64; num_q * count];
        let mut scratch = vec![0u64; num_b * count + count];
        self.fast_base_conv_sk_into(input, count, &mut output, &mut scratch)?;
        Ok(output)
    }

    pub fn fast_base_conv_sk_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let tables = self.openfhe.as_ref().ok_or(RnsError::OpenFheDisabled)?;
        let base_b = &tables.base_b;
        let num_b = base_b.len();
        let num_bsk = tables.base_bsk.len();
        if input.len() != num_bsk * count {
            return Err(RnsError::InvalidResidues);
        }
        let num_q = self.base_q.len();
        if output.len() != num_q * count {
            return Err(RnsError::InvalidResidues);
        }
        let needed = num_b * count + count;
        if scratch.len() < needed {
            return Err(RnsError::InvalidResidues);
        }

        let (alpha, temp_b) = scratch.split_at_mut(count);
        alpha.fill(0);

        let m_sk_value = tables.m_sk.value();
        let m_sk_half = m_sk_value >> 1;

        for i in 0..num_b {
            let modulus = base_b.moduli()[i].value();
            let b_hat_inv = tables.b_hat_inv_mod_b[i];
            let b_hat_mod_m_sk = tables.b_hat_mod_m_sk[i];
            for k in 0..count {
                let x = input[i * count + k];
                let scaled = arith::mul_mod_u64(x, b_hat_inv, modulus);
                temp_b[i * count + k] = scaled;
                let term = arith::mul_mod_u64(scaled, b_hat_mod_m_sk, m_sk_value);
                alpha[k] = arith::add_mod(alpha[k], term, m_sk_value);
            }
        }

        let m_sk_offset = num_b * count;
        for k in 0..count {
            let residue = input[m_sk_offset + k] % m_sk_value;
            let mut value = arith::sub_mod(alpha[k], residue, m_sk_value);
            value = arith::mul_mod_u64(value, tables.b_inv_mod_m_sk, m_sk_value);
            alpha[k] = value;
        }

        for j in 0..num_q {
            let modulus = self.base_q.moduli()[j].value();
            let b_mod_q = tables.b_mod_q[j];
            let m_sk_mod_q = m_sk_value % modulus;
            for k in 0..count {
                let mut acc = 0u64;
                for i in 0..num_b {
                    let term = arith::mul_mod_u64(
                        temp_b[i * count + k],
                        tables.b_hat_mod_q[i][j],
                        modulus,
                    );
                    acc = arith::add_mod(acc, term, modulus);
                }
                let mut alpha_mod_q = alpha[k] % modulus;
                if alpha[k] > m_sk_half {
                    alpha_mod_q = arith::sub_mod(alpha_mod_q, m_sk_mod_q, modulus);
                }
                let alpha_b_mod_q = arith::mul_mod_u64(alpha_mod_q, b_mod_q, modulus);
                output[j * count + k] = arith::sub_mod(acc, alpha_b_mod_q, modulus);
            }
        }

        Ok(())
    }

    pub fn behz_convert_q_to_bsk_mtilde_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let behz = self.behz.as_ref().ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_bsk = behz.base_bsk.len();
        if input.len() != size_q * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != (size_bsk + 1) * count {
            return Err(RnsError::InvalidResidues);
        }
        let needed = size_q * count + size_q;
        if scratch.len() < needed {
            return Err(RnsError::InvalidResidues);
        }

        let (temp_q, scratch_conv) = scratch.split_at_mut(size_q * count);
        {
            let _scope = profile::Scope::new(profile::Kind::BehzQMulMtilde);
            for i in 0..size_q {
                let modulus = self.base_q.moduli()[i].value();
                let m_tilde_op = &behz.m_tilde_mod_q_shoup[i];
                let offset = i * count;
                for k in 0..count {
                    let x = input[offset + k];
                    temp_q[offset + k] =
                        arith::mul_mod_shoup(x, m_tilde_op.operand, m_tilde_op.quotient, modulus);
                }
            }
        }

        let (out_bsk, out_mtilde) = output.split_at_mut(size_bsk * count);
        {
            let _scope = profile::Scope::new(profile::Kind::BehzQToBsk);
            behz.base_q_to_bsk
                .fast_convert_array_into(temp_q, count, out_bsk, scratch_conv)?;
        }
        {
            let _scope = profile::Scope::new(profile::Kind::BehzQToMtilde);
            behz.base_q_to_mtilde.fast_convert_array_into(
                temp_q,
                count,
                out_mtilde,
                scratch_conv,
            )?;
        }
        Ok(())
    }

    pub fn behz_montgomery_reduce_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let behz = self.behz.as_ref().ok_or(RnsError::OpenFheDisabled)?;
        let size_bsk = behz.base_bsk.len();
        if input.len() != (size_bsk + 1) * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != size_bsk * count {
            return Err(RnsError::InvalidResidues);
        }
        if scratch.len() < count {
            return Err(RnsError::InvalidResidues);
        }

        let m_tilde = behz.m_tilde;
        let m_tilde_div2 = m_tilde >> 1;
        let input_mtilde = &input[size_bsk * count..];

        let r_mtilde = &mut scratch[..count];
        for k in 0..count {
            let x = input_mtilde[k];
            r_mtilde[k] = arith::mul_mod_shoup(
                x,
                behz.neg_inv_prod_q_mod_mtilde.operand,
                behz.neg_inv_prod_q_mod_mtilde.quotient,
                m_tilde,
            );
        }

        for j in 0..size_bsk {
            let modulus = behz.base_bsk.moduli()[j].value();
            let prod_q_op = &behz.prod_q_mod_bsk_shoup[j];
            let inv_mtilde_op = &behz.inv_m_tilde_mod_bsk[j];
            let offset = j * count;
            for k in 0..count {
                let mut temp = r_mtilde[k];
                if temp >= m_tilde_div2 {
                    temp = temp.wrapping_add(modulus).wrapping_sub(m_tilde);
                }
                let term =
                    arith::mul_mod_shoup(temp, prod_q_op.operand, prod_q_op.quotient, modulus);
                let mut value = arith::add_mod(input[offset + k], term, modulus);
                value = arith::mul_mod_shoup(
                    value,
                    inv_mtilde_op.operand,
                    inv_mtilde_op.quotient,
                    modulus,
                );
                output[offset + k] = value;
            }
        }
        Ok(())
    }

    pub fn behz_floor_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let behz = self.behz.as_ref().ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_bsk = behz.base_bsk.len();
        if input.len() != (size_q + size_bsk) * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != size_bsk * count {
            return Err(RnsError::InvalidResidues);
        }
        if scratch.len() < size_q {
            return Err(RnsError::InvalidResidues);
        }

        let (input_q, input_bsk) = input.split_at(size_q * count);
        behz.base_q_to_bsk
            .fast_convert_array_into(input_q, count, output, scratch)?;

        for j in 0..size_bsk {
            let modulus = behz.base_bsk.moduli()[j].value();
            let inv_prod_q = &behz.inv_prod_q_mod_bsk[j];
            let offset = j * count;
            for k in 0..count {
                let diff = arith::sub_mod(input_bsk[offset + k], output[offset + k], modulus);
                output[offset + k] =
                    arith::mul_mod_shoup(diff, inv_prod_q.operand, inv_prod_q.quotient, modulus);
            }
        }
        Ok(())
    }

    pub fn behz_baseconv_sk_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        let behz = self.behz.as_ref().ok_or(RnsError::OpenFheDisabled)?;
        let size_b = behz.base_b.len();
        let size_bsk = behz.base_bsk.len();
        let size_q = self.base_q.len();
        if input.len() != size_bsk * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != size_q * count {
            return Err(RnsError::InvalidResidues);
        }
        let needed = 2 * count + size_b;
        if scratch.len() < needed {
            return Err(RnsError::InvalidResidues);
        }

        let (alpha, rest) = scratch.split_at_mut(count);
        let (temp_msk, scratch_conv) = rest.split_at_mut(count);
        alpha.fill(0);

        let input_b = &input[..size_b * count];
        behz.base_b_to_q
            .fast_convert_array_into(input_b, count, output, scratch_conv)?;
        behz.base_b_to_msk
            .fast_convert_array_into(input_b, count, temp_msk, scratch_conv)?;

        let m_sk_value = behz.m_sk.value();
        for k in 0..count {
            let input_sk = input[size_b * count + k] % m_sk_value;
            let mut value = arith::sub_mod(temp_msk[k], input_sk, m_sk_value);
            value = arith::mul_mod_shoup(
                value,
                behz.inv_prod_b_mod_msk.operand,
                behz.inv_prod_b_mod_msk.quotient,
                m_sk_value,
            );
            alpha[k] = value;
        }

        let m_sk_div2 = m_sk_value >> 1;
        for i in 0..size_q {
            let modulus = self.base_q.moduli()[i].value();
            let prod_op = &behz.prod_b_mod_q_shoup[i];
            let neg_prod_op = &behz.neg_prod_b_mod_q_shoup[i];
            let offset = i * count;
            for k in 0..count {
                let mut acc = output[offset + k];
                let a = alpha[k];
                if a > m_sk_div2 {
                    let neg = m_sk_value - a;
                    let term =
                        arith::mul_mod_shoup(neg, prod_op.operand, prod_op.quotient, modulus);
                    acc = arith::add_mod(acc, term, modulus);
                } else {
                    let term =
                        arith::mul_mod_shoup(a, neg_prod_op.operand, neg_prod_op.quotient, modulus);
                    acc = arith::add_mod(acc, term, modulus);
                }
                output[offset + k] = acc;
            }
        }

        Ok(())
    }

    /// OpenFHE approximate scale-and-round: {X}_{Q,P} -> {approx(t/Q * X)}_{P}.
    pub fn approx_scale_and_round_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
    ) -> Result<(), RnsError> {
        let tables = self
            .approx_scale_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        let size_p = tables.base_p.len();
        if input.len() != (size_q + size_p) * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != size_p * count {
            return Err(RnsError::InvalidResidues);
        }

        let (input_q, input_p) = input.split_at(size_q * count);
        for coeff in 0..count {
            for j in 0..size_p {
                let modulus = tables.base_p.moduli()[j].value();
                let row = &tables.tps_hat_inv_mods_divs_modp[j];
                let mut acc_128: u128 = 0;
                for i in 0..size_q {
                    let x = input_q[i * count + coeff];
                    let shoup = row[i];
                    let term = arith::mul_mod_shoup_lazy(x, shoup.operand, shoup.quotient, modulus);
                    acc_128 += term as u128;
                }
                let x_p = input_p[j * count + coeff];
                let shoup = row[size_q]; // The last element
                let term = arith::mul_mod_shoup_lazy(x_p, shoup.operand, shoup.quotient, modulus);
                acc_128 += term as u128;

                output[j * count + coeff] = (acc_128 % (modulus as u128)) as u64;
            }
        }

        Ok(())
    }

    pub fn approx_scale_and_round_with_scratch<'a>(
        &self,
        input: &[u64],
        count: usize,
        scratch: &'a mut RnsToolScratch,
    ) -> Result<&'a mut [u64], RnsError> {
        let size_p = self
            .approx_scale_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?
            .base_p
            .len();
        let output = scratch.alloc(size_p * count);
        self.approx_scale_and_round_into(input, count, output)?;
        Ok(output)
    }

    pub fn approx_scale_and_round(
        &self,
        input: &[u64],
        count: usize,
    ) -> Result<Vec<u64>, RnsError> {
        let size_p = self
            .approx_scale_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?
            .base_p
            .len();
        let mut output = vec![0u64; size_p * count];
        self.approx_scale_and_round_into(input, count, &mut output)?;
        Ok(output)
    }

    pub fn exact_round_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        self.base_q_to_t
            .exact_convert_array_into(input, count, output, scratch)
    }

    pub fn exact_round_with_scratch<'a>(
        &self,
        input: &[u64],
        count: usize,
        scratch: &'a mut RnsToolScratch,
    ) -> Result<&'a mut [u64], RnsError> {
        let (output, temp) = scratch.alloc_two(count, self.base_q.len());
        self.exact_round_into(input, count, output, temp)?;
        Ok(output)
    }

    pub fn exact_round(&self, input: &[u64], count: usize) -> Result<Vec<u64>, RnsError> {
        let mut output = vec![0u64; count];
        let mut scratch = vec![0u64; self.base_q.len()];
        self.exact_round_into(input, count, &mut output, &mut scratch)?;
        Ok(output)
    }

    pub fn scale_and_round_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
        scratch: &mut [u64],
    ) -> Result<(), RnsError> {
        self.rounder.round_array_into(input, count, output, scratch)
    }

    pub fn scale_and_round_with_scratch<'a>(
        &self,
        input: &[u64],
        count: usize,
        scratch: &'a mut RnsToolScratch,
    ) -> Result<&'a mut [u64], RnsError> {
        let (output, temp) = scratch.alloc_two(count, self.base_q.len());
        self.scale_and_round_into(input, count, output, temp)?;
        Ok(output)
    }

    pub fn scale_and_round(&self, input: &[u64], count: usize) -> Result<Vec<u64>, RnsError> {
        let mut output = vec![0u64; count];
        let mut scratch = vec![0u64; self.base_q.len()];
        self.scale_and_round_into(input, count, &mut output, &mut scratch)?;
        Ok(output)
    }

    /// OpenFHE float-assisted scale-and-round (approximate, fast).
    pub fn scale_and_round_float_into(
        &self,
        input: &[u64],
        count: usize,
        output: &mut [u64],
    ) -> Result<(), RnsError> {
        let tables = self
            .scale_tables
            .as_ref()
            .ok_or(RnsError::OpenFheDisabled)?;
        let size_q = self.base_q.len();
        if input.len() != size_q * count {
            return Err(RnsError::InvalidResidues);
        }
        if output.len() != count {
            return Err(RnsError::InvalidResidues);
        }

        let t = tables.t;
        let t_mod = t as u128;
        let t_mask = t.wrapping_sub(1);
        let use_b_split = tables.t_q_hat_inv_modq_b_divq_modt.is_some();

        for coeff in 0..count {
            let mut int_sum: u128 = 0;
            let mut float_sum: f64 = if tables.t_is_power_of_two { 0.5 } else { 0.0 };

            if use_b_split {
                let shift = tables.q_msb_half.unwrap_or(0);
                let mask = if shift == 0 { 0 } else { (1u64 << shift) - 1 };
                let divq_modt_b = tables.t_q_hat_inv_modq_b_divq_modt.as_ref().unwrap();
                let divq_frac_b = tables.t_q_hat_inv_modq_b_divq_frac.as_ref().unwrap();
                for i in 0..size_q {
                    let x = input[i * count + coeff];
                    let lo = if shift == 0 { 0 } else { x & mask };
                    let hi = if shift == 0 { x } else { x >> shift };
                    float_sum += (lo as f64) * tables.t_q_hat_inv_modq_divq_frac[i];
                    float_sum += (hi as f64) * divq_frac_b[i];

                    int_sum = (int_sum
                        + (lo as u128) * (tables.t_q_hat_inv_modq_divq_modt[i] as u128))
                        % t_mod;
                    int_sum = (int_sum + (hi as u128) * (divq_modt_b[i] as u128)) % t_mod;
                }
            } else {
                for i in 0..size_q {
                    let x = input[i * count + coeff];
                    float_sum += (x as f64) * tables.t_q_hat_inv_modq_divq_frac[i];
                    int_sum = (int_sum
                        + (x as u128) * (tables.t_q_hat_inv_modq_divq_modt[i] as u128))
                        % t_mod;
                }
            }

            if tables.t_is_power_of_two {
                let rounded = int_sum + float_sum.floor() as u128;
                output[coeff] = (rounded as u64) & t_mask;
            } else {
                float_sum += int_sum as f64;
                let quot = (float_sum * tables.t_inv).floor();
                float_sum -= (t as f64) * quot;
                let rounded = (float_sum + 0.5).floor() as u64;
                output[coeff] = rounded % t;
            }
        }

        Ok(())
    }

    pub fn scale_and_round_float_with_scratch<'a>(
        &self,
        input: &[u64],
        count: usize,
        scratch: &'a mut RnsToolScratch,
    ) -> Result<&'a mut [u64], RnsError> {
        let output = scratch.alloc(count);
        self.scale_and_round_float_into(input, count, output)?;
        Ok(output)
    }

    pub fn scale_and_round_float(&self, input: &[u64], count: usize) -> Result<Vec<u64>, RnsError> {
        let mut output = vec![0u64; count];
        self.scale_and_round_float_into(input, count, &mut output)?;
        Ok(output)
    }
}

fn msb_u64(value: u64) -> u32 {
    if value == 0 {
        0
    } else {
        63 - value.leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_scale_reference_u128(
        base_q: &RnsBase,
        base_p: &RnsBase,
        t: Modulus,
        input: &[u64],
        count: usize,
    ) -> Vec<u64> {
        let size_q = base_q.len();
        let size_p = base_p.len();
        let q_prod = base_q.base_prod_u128().expect("q prod");
        let p_prod = base_p.base_prod_u128().expect("p prod");
        let s_prod = q_prod * p_prod;

        let mut output = vec![0u64; size_p * count];
        for coeff in 0..count {
            for (j, modulus_p) in base_p.moduli().iter().enumerate() {
                let p_value = modulus_p.value() as u128;
                let mut acc = 0u64;

                for (i, modulus_q) in base_q.moduli().iter().enumerate() {
                    let q_value = modulus_q.value() as u128;
                    let s_hat = s_prod / q_value;
                    let inv =
                        numth::mod_inverse((s_hat % q_value) as u64, q_value as u64).expect("inv");
                    let alpha = (u128::from(t.value()) * p_prod * u128::from(inv)) / q_value;
                    let alpha_mod = (alpha % p_value) as u64;

                    let x = input[i * count + coeff];
                    let term = arith::mul_mod_u64(x, alpha_mod, modulus_p.value());
                    acc = arith::add_mod(acc, term, modulus_p.value());
                }

                let s_hat = s_prod / p_value;
                let inv =
                    numth::mod_inverse((s_hat % p_value) as u64, p_value as u64).expect("inv");
                let alpha = (u128::from(t.value()) * p_prod * u128::from(inv)) / p_value;
                let alpha_mod = (alpha % p_value) as u64;
                let x_p = input[size_q * count + j * count + coeff];
                let term = arith::mul_mod_u64(x_p, alpha_mod, modulus_p.value());
                acc = arith::add_mod(acc, term, modulus_p.value());

                output[j * count + coeff] = acc;
            }
        }

        output
    }

    #[test]
    fn fastbconv_matches_base_converter() {
        let base_q = RnsBase::from_values(vec![3, 5, 7]).expect("base_q");
        let base_b = RnsBase::from_values(vec![11, 13]).expect("base_b");
        let tool = RnsTool::new(
            base_q.clone(),
            Modulus::new(17).expect("t"),
            Some(base_b.clone()),
        )
        .expect("tool");

        let values = [0u128, 10, 22, 37, 115];
        let count = values.len();
        let mut input = vec![0u64; count * base_q.len()];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let output = tool.fastbconv(&input, count).expect("fastbconv");
        let converter = BaseConverter::new(base_q, base_b).expect("converter");
        let expected = converter
            .fast_convert_array(&input, count)
            .expect("convert");
        assert_eq!(output, expected);
    }

    #[test]
    fn exact_round_matches_naive() {
        let base_q = RnsBase::from_values(vec![3, 5, 7]).expect("base_q");
        let base_t = Modulus::new(11).expect("t");
        let tool = RnsTool::new(base_q.clone(), base_t, None).expect("tool");

        let values = [0u128, 1, 42, 77, 104, 128];
        let count = values.len();
        let q = base_q.base_prod_u128().unwrap();
        let t = u128::from(base_t.value());
        let mut input = vec![0u64; count * base_q.len()];
        for (idx, value) in values.iter().enumerate() {
            let value = value % q;
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let exact = tool.exact_round(&input, count).expect("exact");
        for (idx, value) in values.iter().enumerate() {
            let value = value % q;
            let expected = ((value * t + q / 2) / q) % t;
            assert_eq!(exact[idx], expected as u64);
        }
    }

    #[test]
    fn scale_and_round_matches_exact_round() {
        let base_q = RnsBase::from_values(vec![3, 5, 7]).expect("base_q");
        let tool = RnsTool::new(base_q.clone(), Modulus::new(11).expect("t"), None).expect("tool");

        let values = [0u128, 1, 42, 77, 104];
        let count = values.len();
        let mut input = vec![0u64; count * base_q.len()];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let exact = tool.exact_round(&input, count).expect("exact");
        let scaled = tool.scale_and_round(&input, count).expect("scale");
        assert_eq!(exact, scaled);
    }

    #[test]
    fn fastbconv_without_base_b_errors() {
        let base_q = RnsBase::from_values(vec![3, 5, 7]).expect("base_q");
        let tool = RnsTool::new(base_q.clone(), Modulus::new(11).expect("t"), None).expect("tool");
        let input = vec![0u64; base_q.len()];
        let err = tool.fastbconv(&input, 1).expect_err("missing base_b");
        assert!(matches!(err, RnsError::InvalidBase));
    }

    #[test]
    fn fastbconv_with_scratch_matches_heap() {
        let base_q = RnsBase::from_values(vec![17, 19]).expect("base_q");
        let base_b = RnsBase::from_values(vec![23]).expect("base_b");
        let tool =
            RnsTool::new(base_q.clone(), Modulus::new(7).expect("t"), Some(base_b)).expect("tool");

        let count = 4;
        let mut input = vec![0u64; count * base_q.len()];
        for (i, modulus) in base_q.moduli().iter().enumerate() {
            for k in 0..count {
                input[i * count + k] = ((i + 1 + k) as u64) % modulus.value();
            }
        }

        let expected = tool.fastbconv(&input, count).expect("heap");
        let mut scratch = RnsToolScratch::new();
        let output = tool
            .fastbconv_with_scratch(&input, count, &mut scratch)
            .expect("scratch");
        assert_eq!(output, expected.as_slice());
    }

    #[test]
    fn approx_mod_up_with_scratch_matches_heap() {
        let tool = openfhe_switch_tool();
        let base_q = tool.base_q().clone();
        let count = 5;
        let mut input = vec![0u64; count * base_q.len()];
        for (i, modulus) in base_q.moduli().iter().enumerate() {
            for k in 0..count {
                input[i * count + k] = ((k + 3 + i) as u64) % modulus.value();
            }
        }

        let expected = tool.approx_mod_up(&input, count).expect("heap");
        let mut scratch = RnsToolScratch::new();
        let output = tool
            .approx_mod_up_with_scratch(&input, count, &mut scratch)
            .expect("scratch");
        assert_eq!(output, expected.as_slice());
    }

    fn openfhe_tool() -> RnsTool {
        let base_q = RnsBase::from_values(vec![17, 19]).expect("base_q");
        let base_b = RnsBase::from_values(vec![23]).expect("base_b");
        let base_t = Modulus::new(7).expect("t");
        let mut tool = RnsTool::new(base_q, base_t, Some(base_b)).expect("tool");
        tool.enable_openfhe(8, Modulus::new(29).expect("m_sk"))
            .expect("openfhe");
        tool
    }

    fn openfhe_switch_tool() -> RnsTool {
        let base_q = RnsBase::from_values(vec![17, 19]).expect("base_q");
        let base_t = Modulus::new(7).expect("t");
        let mut tool = RnsTool::new(base_q, base_t, None).expect("tool");
        let base_p = RnsBase::from_values(vec![31]).expect("base_p");
        tool.enable_openfhe_switch(base_p, None).expect("switch");
        tool
    }

    fn openfhe_switch_tool_with_t() -> RnsTool {
        let base_q = RnsBase::from_values(vec![17, 19]).expect("base_q");
        let base_t = Modulus::new(7).expect("t");
        let mut tool = RnsTool::new(base_q, base_t, None).expect("tool");
        let base_p = RnsBase::from_values(vec![31]).expect("base_p");
        tool.enable_openfhe_switch(base_p, Some(base_t))
            .expect("switch");
        tool
    }

    fn openfhe_scale_tool() -> RnsTool {
        let base_q = RnsBase::from_values(vec![17, 19]).expect("base_q");
        let base_t = Modulus::new(7).expect("t");
        let mut tool = RnsTool::new(base_q, base_t, None).expect("tool");
        tool.enable_openfhe_scale_and_round().expect("scale");
        tool
    }

    fn openfhe_approx_scale_tool() -> RnsTool {
        let base_q = RnsBase::from_values(vec![17, 19]).expect("base_q");
        let base_t = Modulus::new(7).expect("t");
        let base_p = RnsBase::from_values(vec![23]).expect("base_p");
        let mut tool = RnsTool::new(base_q, base_t, None).expect("tool");
        tool.enable_openfhe_approx_scale_and_round(base_p)
            .expect("scale");
        tool
    }

    #[test]
    fn openfhe_fast_base_conv_q_to_bsk_matches_naive() {
        let tool = openfhe_tool();
        let base_q = tool.base_q().clone();
        let base_bsk = tool.base_bsk().expect("bsk").clone();
        let q = base_q.base_prod_u128().unwrap();
        let q_half = q / 2;

        let values = [0u128, 1, 2, 3, 4, 5, 7, 11, 23, 46, 100, 200, 322];
        let count = values.len();
        let mut input = vec![0u64; count * base_q.len()];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let output = tool
            .fast_base_conv_q_to_bsk_montgomery(&input, count)
            .expect("fast base conv");
        for (idx, value) in values.iter().enumerate() {
            let centered = if *value > q_half {
                *value as i128 - q as i128
            } else {
                *value as i128
            };
            for (j, modulus) in base_bsk.moduli().iter().enumerate() {
                let m = modulus.value() as i128;
                let mut expected = centered % m;
                if expected < 0 {
                    expected += m;
                }
                assert_eq!(output[j * count + idx], expected as u64);
            }
        }
    }

    #[test]
    fn openfhe_fast_base_conv_sk_matches_naive() {
        let tool = openfhe_tool();
        let base_q = tool.base_q().clone();
        let base_bsk = tool.base_bsk().expect("bsk").clone();

        let values = [0u128, 1, 2, 3, 4, 5, 7, 11, 23, 46, 100, 200, 322];
        let count = values.len();
        let mut input = vec![0u64; count * base_bsk.len()];
        for (idx, value) in values.iter().enumerate() {
            for (j, modulus) in base_bsk.moduli().iter().enumerate() {
                input[j * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let output = tool
            .fast_base_conv_sk(&input, count)
            .expect("fast base conv sk");
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                let expected = (value % u128::from(modulus.value())) as u64;
                assert_eq!(output[i * count + idx], expected);
            }
        }
    }

    #[test]
    fn openfhe_fast_rns_floor_q_matches_reference() {
        let tool = openfhe_tool();
        let base_q = tool.base_q().clone();
        let base_bsk = tool.base_bsk().expect("bsk").clone();
        let tables = tool.openfhe.as_ref().expect("tables");

        let values = [0u128, 1, 2, 3, 4, 5, 7, 11, 23, 46, 100, 200, 322];
        let count = values.len();
        let mut input = vec![0u64; (base_q.len() + base_bsk.len()) * count];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
            let offset = base_q.len() * count;
            for (j, modulus) in base_bsk.moduli().iter().enumerate() {
                input[offset + j * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let output = tool.fast_rns_floor_q(&input, count).expect("fast floor");
        for (idx, _) in values.iter().enumerate() {
            let mut coeff_q = vec![0u64; base_q.len()];
            let mut coeff_bsk = vec![0u64; base_bsk.len()];
            for i in 0..base_q.len() {
                coeff_q[i] = input[i * count + idx];
            }
            let offset = base_q.len() * count;
            for j in 0..base_bsk.len() {
                coeff_bsk[j] = input[offset + j * count + idx];
            }

            let mut expected = vec![0u64; base_bsk.len()];
            for (i, modulus_q) in base_q.moduli().iter().enumerate() {
                let q_value = modulus_q.value();
                let twisted = arith::mul_mod_u64(coeff_q[i], tables.t_q_hat_inv_mod_q[i], q_value);
                coeff_q[i] = twisted;
            }
            for (j, modulus_bsk) in base_bsk.moduli().iter().enumerate() {
                let bsk_value = modulus_bsk.value();
                let mut acc = 0u64;
                for i in 0..base_q.len() {
                    let term =
                        arith::mul_mod_u64(coeff_q[i], tables.q_inv_mod_bsk[i][j], bsk_value);
                    acc = arith::add_mod(acc, term, bsk_value);
                }
                let mut value =
                    arith::mul_mod_u64(coeff_bsk[j], tables.t_q_inv_mod_bsk[j], bsk_value);
                value = arith::sub_mod(value, acc, bsk_value);
                expected[j] = value;
            }

            for j in 0..base_bsk.len() {
                assert_eq!(output[j * count + idx], expected[j]);
            }
        }
    }

    #[test]
    fn openfhe_approx_switch_crt_basis_matches_naive() {
        let tool = openfhe_switch_tool();
        let base_q = tool.base_q().clone();
        let base_p = tool.base_p().expect("base_p").clone();

        let values = [0u128, 1, 2, 7, 11, 23, 46, 100, 161, 162, 200, 322];
        let count = values.len();
        let mut input = vec![0u64; count * base_q.len()];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let output = tool
            .approx_switch_crt_basis(&input, count)
            .expect("approx switch");
        let converter = BaseConverter::new(base_q, base_p).expect("converter");
        let expected = converter
            .fast_convert_array(&input, count)
            .expect("convert");
        assert_eq!(output, expected);
    }

    #[test]
    fn openfhe_switch_crt_basis_matches_centered() {
        let tool = openfhe_switch_tool();
        let base_q = tool.base_q().clone();
        let base_p = tool.base_p().expect("base_p").clone();
        let q = base_q.base_prod_u128().unwrap();
        let q_half = q / 2;

        let values = [0u128, 1, 2, 7, 11, 23, 46, 100, 161, 162, 200, 322];
        let count = values.len();
        let mut input = vec![0u64; count * base_q.len()];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let output = tool.switch_crt_basis(&input, count).expect("switch");
        for (idx, value) in values.iter().enumerate() {
            let centered = if *value > q_half {
                *value as i128 - q as i128
            } else {
                *value as i128
            };
            for (j, modulus) in base_p.moduli().iter().enumerate() {
                let m = modulus.value() as i128;
                let mut expected = centered % m;
                if expected < 0 {
                    expected += m;
                }
                assert_eq!(output[j * count + idx], expected as u64);
            }
        }
    }

    #[test]
    fn openfhe_approx_mod_up_expands_basis() {
        let tool = openfhe_switch_tool();
        let base_q = tool.base_q().clone();
        let base_p = tool.base_p().expect("base_p").clone();
        let base_q_len = base_q.len();

        let values = [0u128, 1, 2, 7, 11, 23, 46, 100, 161, 162, 200, 322];
        let count = values.len();
        let mut input = vec![0u64; count * base_q.len()];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let output = tool.approx_mod_up(&input, count).expect("mod up");
        assert_eq!(&output[..base_q_len * count], &input[..]);

        let converter = BaseConverter::new(base_q, base_p).expect("converter");
        let expected_p = converter
            .fast_convert_array(&input, count)
            .expect("convert");
        assert_eq!(&output[base_q_len * count..], expected_p.as_slice());
    }

    #[test]
    fn openfhe_expand_crt_basis_matches_centered() {
        let tool = openfhe_switch_tool();
        let base_q = tool.base_q().clone();
        let base_p = tool.base_p().expect("base_p").clone();
        let q = base_q.base_prod_u128().unwrap();
        let q_half = q / 2;

        let values = [0u128, 1, 2, 7, 11, 23, 46, 100, 161, 162, 200, 322];
        let count = values.len();
        let mut input = vec![0u64; count * base_q.len()];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let output = tool.expand_crt_basis(&input, count).expect("expand");
        assert_eq!(&output[..base_q.len() * count], &input[..]);

        for (idx, value) in values.iter().enumerate() {
            let centered = if *value > q_half {
                *value as i128 - q as i128
            } else {
                *value as i128
            };
            for (j, modulus) in base_p.moduli().iter().enumerate() {
                let m = modulus.value() as i128;
                let mut expected = centered % m;
                if expected < 0 {
                    expected += m;
                }
                let actual = output[base_q.len() * count + j * count + idx];
                assert_eq!(actual, expected as u64);
            }
        }
    }

    #[test]
    fn openfhe_expand_crt_basis_reverse_order() {
        let tool = openfhe_switch_tool();
        let base_q = tool.base_q().clone();
        let base_p = tool.base_p().expect("base_p").clone();
        let q = base_q.base_prod_u128().unwrap();
        let q_half = q / 2;

        let values = [0u128, 1, 2, 7, 11, 23, 46, 100, 161, 162, 200, 322];
        let count = values.len();
        let mut input = vec![0u64; count * base_q.len()];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let output = tool
            .expand_crt_basis_reverse_order(&input, count)
            .expect("reverse order");

        for (idx, value) in values.iter().enumerate() {
            let centered = if *value > q_half {
                *value as i128 - q as i128
            } else {
                *value as i128
            };
            for (j, modulus) in base_p.moduli().iter().enumerate() {
                let m = modulus.value() as i128;
                let mut expected = centered % m;
                if expected < 0 {
                    expected += m;
                }
                assert_eq!(output[j * count + idx], expected as u64);
            }
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                let expected = (value % u128::from(modulus.value())) as u64;
                let offset = base_p.len() * count;
                assert_eq!(output[offset + i * count + idx], expected);
            }
        }
    }

    #[test]
    fn openfhe_approx_mod_down_matches_floor() {
        let tool = openfhe_switch_tool();
        let base_q = tool.base_q().clone();
        let base_p = tool.base_p().expect("base_p").clone();
        let p = base_p.base_prod_u128().unwrap();

        let values = [0u128, 1, 2, 23, 46, 100, 322, 323, 324, 999, 10012];
        let count = values.len();
        let mut input = vec![0u64; (base_q.len() + base_p.len()) * count];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
            let offset = base_q.len() * count;
            for (j, modulus) in base_p.moduli().iter().enumerate() {
                input[offset + j * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let output = tool.approx_mod_down(&input, count).expect("mod down");
        for (idx, value) in values.iter().enumerate() {
            let expected = (value / p) as u64;
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                let expected_residue = expected % modulus.value();
                assert_eq!(output[i * count + idx], expected_residue);
            }
        }
    }

    #[test]
    fn openfhe_approx_mod_down_large_scratch_matches_heap() {
        let tool = openfhe_switch_tool();
        let base_q = tool.base_q().clone();
        let base_p = tool.base_p().expect("base_p").clone();

        let values = [0u128, 1, 2, 23, 46, 100, 322, 323, 324, 999, 10012];
        let count = values.len();
        let mut input = vec![0u64; (base_q.len() + base_p.len()) * count];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
            let offset = base_q.len() * count;
            for (j, modulus) in base_p.moduli().iter().enumerate() {
                input[offset + j * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let mut output = vec![0u64; base_q.len() * count];
        let mut scratch =
            vec![0u64; base_p.len() * count + base_p.len() * BaseConverter::BLOCK_SIZE];
        tool.approx_mod_down_into(&input, count, &mut output, &mut scratch, false, false)
            .expect("mod down");

        let expected = tool.approx_mod_down(&input, count).expect("heap");
        assert_eq!(output, expected);
    }

    #[test]
    fn openfhe_approx_mod_down_ntt_matches_coeff() {
        let base_q =
            RnsBase::from_values(vec![36028797018652673u64, 36028797017571329u64]).expect("base_q");
        let base_p = RnsBase::from_values(vec![1152921504606830593u64, 1152921504606748673u64])
            .expect("base_p");
        let base_t = Modulus::new(65537).expect("t");
        let mut tool = RnsTool::new(base_q.clone(), base_t, None).expect("tool");
        tool.enable_openfhe_switch(base_p.clone(), Some(base_t))
            .expect("switch");

        let degree = 4096usize;
        let count = degree;
        let size_q = base_q.len();
        let size_p = base_p.len();

        let mut input = vec![0u64; (size_q + size_p) * count];
        for (i, modulus) in base_q.moduli().iter().enumerate() {
            let q = modulus.value();
            for k in 0..count {
                input[i * count + k] = ((k as u64).wrapping_mul(13).wrapping_add(i as u64)) % q;
            }
        }
        let offset = size_q * count;
        for (j, modulus) in base_p.moduli().iter().enumerate() {
            let p = modulus.value();
            for k in 0..count {
                input[offset + j * count + k] =
                    ((k as u64).wrapping_mul(13).wrapping_add(j as u64)) % p;
            }
        }

        let mut ntt_tables = Vec::with_capacity(size_q + size_p);
        for modulus in base_q.moduli().iter().chain(base_p.moduli().iter()) {
            let table = NttTables::new_negacyclic(degree, *modulus).expect("ntt tables");
            ntt_tables.push(table);
        }

        let mut input_ntt = input.clone();
        for limb in 0..(size_q + size_p) {
            let slice = &mut input_ntt[limb * count..(limb + 1) * count];
            ntt_forward(slice, &ntt_tables[limb]);
        }

        let mut output_coeff = vec![0u64; size_q * count];
        let mut scratch_coeff = vec![0u64; size_p * count + size_p * BaseConverter::BLOCK_SIZE];
        tool.approx_mod_down_into(
            &input,
            count,
            &mut output_coeff,
            &mut scratch_coeff,
            false,
            false,
        )
        .expect("mod down");

        let mut output_ntt_expected = output_coeff.clone();
        for limb in 0..size_q {
            let slice = &mut output_ntt_expected[limb * count..(limb + 1) * count];
            ntt_forward(slice, &ntt_tables[limb]);
        }

        let mut output_ntt_actual = vec![0u64; size_q * count];
        let mut scratch_ntt = vec![0u64; size_p * count + size_p * BaseConverter::BLOCK_SIZE];
        tool.approx_mod_down_ntt_into(
            &mut input_ntt,
            count,
            &mut output_ntt_actual,
            &mut scratch_ntt,
            &ntt_tables,
        )
        .expect("mod down ntt");

        assert_eq!(output_ntt_actual, output_ntt_expected);
    }

    #[test]
    fn openfhe_approx_mod_down_with_t_large_scratch_matches_heap() {
        let tool = openfhe_switch_tool_with_t();
        let base_q = tool.base_q().clone();
        let base_p = tool.base_p().expect("base_p").clone();

        let values = [0u128, 1, 2, 23, 46, 100, 322, 323, 324, 999, 10012];
        let count = values.len();
        let mut input = vec![0u64; (base_q.len() + base_p.len()) * count];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
            let offset = base_q.len() * count;
            for (j, modulus) in base_p.moduli().iter().enumerate() {
                input[offset + j * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let mut output = vec![0u64; base_q.len() * count];
        let mut scratch =
            vec![0u64; base_p.len() * count + base_p.len() * BaseConverter::BLOCK_SIZE];
        tool.approx_mod_down_into(&input, count, &mut output, &mut scratch, false, false)
            .expect("mod down");

        let expected = tool.approx_mod_down(&input, count).expect("heap");
        assert_eq!(output, expected);
    }

    #[test]
    fn openfhe_scale_and_round_float_matches_exact() {
        let tool = openfhe_scale_tool();
        let base_q = tool.base_q().clone();

        let values = [0u128, 1, 2, 7, 11, 23, 46, 100, 161, 162, 200, 322];
        let count = values.len();
        let mut input = vec![0u64; count * base_q.len()];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let exact = tool.scale_and_round(&input, count).expect("exact");
        let float = tool.scale_and_round_float(&input, count).expect("float");
        assert_eq!(float, exact);
    }

    #[test]
    fn openfhe_approx_scale_and_round_matches_reference() {
        let tool = openfhe_approx_scale_tool();
        let base_q = tool.base_q().clone();
        let base_p = tool
            .approx_scale_tables
            .as_ref()
            .expect("tables")
            .base_p
            .clone();
        let t = tool.base_t();

        let values = [0u128, 1, 2, 7, 11, 23, 46, 100, 161, 162, 200, 322];
        let count = values.len();
        let mut input = vec![0u64; (base_q.len() + base_p.len()) * count];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
            let offset = base_q.len() * count;
            for (j, modulus) in base_p.moduli().iter().enumerate() {
                input[offset + j * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let output = tool
            .approx_scale_and_round(&input, count)
            .expect("approx scale");
        let expected = approx_scale_reference_u128(&base_q, &base_p, t, &input, count);
        assert_eq!(output, expected);
    }

    #[test]
    fn rns_tool_config_auto_precompute() {
        let base_q = RnsBase::from_values(vec![17, 19]).expect("base_q");
        let base_b = RnsBase::from_values(vec![23]).expect("base_b");
        let base_p = RnsBase::from_values(vec![29]).expect("base_p");
        let base_t = Modulus::new(7).expect("t");

        let config = RnsToolConfig::new(base_q.clone(), base_t)
            .with_base_b(base_b)
            .with_base_p(base_p.clone())
            .with_openfhe_bsk(8, Modulus::new(31).expect("m_sk"))
            .enable_openfhe_switch(false)
            .enable_openfhe_scale()
            .enable_openfhe_approx_scale();

        let tool = RnsTool::from_config(config).expect("tool");
        assert!(tool.base_bsk().is_some());

        let values = [0u128, 1, 2, 7, 11, 23, 46, 100, 161, 162, 200, 322];
        let count = values.len();
        let mut input = vec![0u64; (base_q.len() + base_p.len()) * count];
        for (idx, value) in values.iter().enumerate() {
            for (i, modulus) in base_q.moduli().iter().enumerate() {
                input[i * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
            let offset = base_q.len() * count;
            for (j, modulus) in base_p.moduli().iter().enumerate() {
                input[offset + j * count + idx] = (value % u128::from(modulus.value())) as u64;
            }
        }

        let output = tool
            .approx_scale_and_round(&input, count)
            .expect("approx scale");
        assert_eq!(output.len(), base_p.len() * count);
    }
}
