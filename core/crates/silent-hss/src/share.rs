use silent_ring::Poly;

#[derive(Clone, Debug)]
pub struct HssShare {
    /// Share of the secret key vector.
    /// In LHSS, this corresponds to `SecretKey` which has 2 components (b, a).
    /// Typically corresponds to shares of (1, s).
    pub elements: [Poly; 2],
}

impl HssShare {
    pub fn new(b: Poly, a: Poly) -> Self {
        Self { elements: [b, a] }
    }
}

/// Paper-style evaluation key for one server in BKS19 Fig.2.
/// Contains one additive share of secret key vector `(1, s)` and shared PRF key `K`.
#[derive(Clone, Debug)]
pub struct HssEvalKey {
    pub share: HssShare,
    pub party_index: u8, // must be 0 or 1
    pub prf_key: [u8; 32],
}

impl HssEvalKey {
    pub fn new(share: HssShare, party_index: u8, prf_key: [u8; 32]) -> Self {
        Self {
            share,
            party_index,
            prf_key,
        }
    }

    #[inline]
    pub fn is_party_zero(&self) -> bool {
        self.party_index == 0
    }
}
