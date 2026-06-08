macro_rules! simple_newtype {
    ($name:ident, $inner:ty) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub $inner);

        impl $name {
            pub const fn get(self) -> $inner {
                self.0
            }
        }
    };
}

simple_newtype!(LogN, u8);
simple_newtype!(RingDim, usize);
simple_newtype!(ModulusBits, u16);
simple_newtype!(PlaintextModulus, u64);
simple_newtype!(ScaleBits, u8);
simple_newtype!(MultiplicativeDepth, u16);
simple_newtype!(MaxRmultDepth, u16);
simple_newtype!(ShareModulusBits, u16);
simple_newtype!(MaxLinearTerms, u32);
simple_newtype!(ReconstructionBoundBits, u16);
simple_newtype!(CorrectnessMarginBits, u16);

// ---------------------------------------------------------------------------
// TFHE-specific newtypes
//
// These wrap the integers used by CGGI-style TFHE (LWE / GLWE dimensions,
// gadget decomposition base/level, message/carry partition of the small
// plaintext space, and the native power-of-two ciphertext modulus exponent).
// ---------------------------------------------------------------------------

simple_newtype!(LweDimension, usize);
simple_newtype!(GlweDimension, usize);
simple_newtype!(PolynomialSize, usize);
simple_newtype!(DecompositionBaseLog, u8);
simple_newtype!(DecompositionLevelCount, u8);
simple_newtype!(MessageModulus, u64);
simple_newtype!(CarryModulus, u64);

// CiphertextModulusLog: base-2 logarithm of a native power-of-two ciphertext
// modulus.  TFHE works over `Z_q` with `q = 2^k`, where `k` is encoded by this
// newtype.  Only `32` and `64` are supported by the MVP backend.
simple_newtype!(CiphertextModulusLog, u8);

impl PolynomialSize {
    pub const fn is_power_of_two(self) -> bool {
        self.0.is_power_of_two()
    }

    pub fn log2(self) -> Option<u32> {
        if self.0 == 0 || !self.0.is_power_of_two() {
            None
        } else {
            Some(self.0.trailing_zeros())
        }
    }
}

impl LweDimension {
    pub const fn lwe_size(self) -> usize {
        self.0 + 1
    }
}

impl GlweDimension {
    pub const fn glwe_size(self) -> usize {
        self.0 + 1
    }

    pub const fn equivalent_lwe_dimension(self, poly_size: PolynomialSize) -> LweDimension {
        LweDimension(self.0 * poly_size.0)
    }
}

/// Selector between TFHE's `Big` (GLWE-side) and `Small` (LWE-side) keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EncryptionKeyChoice {
    Big = 0,
    Small = 1,
}

/// Noise distribution descriptor for TFHE (and other RLWE-based) schemes.
///
/// `Gaussian { stddev }` matches OpenFHE's `HEStd_error`; `TUniform { bound_log2 }`
/// matches tfhe-rs's `t-uniform` distribution as used in the Classic PBS preset
/// `PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128` (via `bound_log2 = 45/17`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoiseDistribution {
    Gaussian { stddev: f64 },
    TUniform { bound_log2: u8 },
}

/// Wrapper around a base-2 logarithm of an estimated PBS failure probability.
///
/// Stored as the negation, i.e. `Log2PFail(128)` means `p_fail <= 2^-128`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Log2PFail(pub f64);

impl LogN {
    pub fn to_ring_dim(self) -> Option<RingDim> {
        1usize.checked_shl(u32::from(self.0)).map(RingDim)
    }
}

impl RingDim {
    pub const fn is_power_of_two(self) -> bool {
        self.0.is_power_of_two()
    }

    pub fn to_log_n(self) -> Option<LogN> {
        if self.0 == 0 || !self.0.is_power_of_two() {
            return None;
        }
        Some(LogN(self.0.trailing_zeros() as u8))
    }
}

pub fn bits_required_u64(value: u64) -> u16 {
    if value == 0 {
        0
    } else {
        (u64::BITS - value.leading_zeros()) as u16
    }
}

pub fn ceil_log2_u32(value: u32) -> u16 {
    if value <= 1 {
        0
    } else {
        (u32::BITS - (value - 1).leading_zeros()) as u16
    }
}
