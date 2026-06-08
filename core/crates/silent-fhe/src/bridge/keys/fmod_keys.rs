/// BFV Fmod key material.
#[derive(Clone, Debug, Default)]
pub struct BfvFmodKeys {
    /// Center-only interpolation coefficients for audit/debug and future
    /// interval Fmod backends. The current center backend is linear and uses
    /// `center_decode_scalar` directly.
    pub coeffs: Vec<u64>,
    /// Scalar `Δ_T^{-1} mod t` for exact center phases.
    pub center_decode_scalar: Option<u64>,
    /// Reserved for a future audited periodic Fmod backend. The bridge does
    /// not populate or consume this field in public conversions.
    pub periodic_center_coeffs: Vec<u64>,
}
