//! Canonical serialization for HSS types.

use crate::ciphertext::HssCiphertext;
use crate::share::{HssEvalKey, HssShare};

use silent_io::error::IoError;
use silent_io::object_type::ObjectType;
use silent_io::primitives::*;
use silent_io::traits::{CanonicalEncode, DecodeWithParams, ObjectKind};
use silent_ring::{Poly, RingContext};
use std::io::{Read, Write};

// ── HssShare ───────────────────────────────────────────────────────────────

impl ObjectKind for HssShare {
    const OBJECT_TYPE: ObjectType = ObjectType::HSS_SHARE;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for HssShare {
    fn encoded_len(&self) -> usize {
        self.elements[0].encoded_len() + self.elements[1].encoded_len()
    }
    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut n = 0;
        n += self.elements[0].encode_to(w)?;
        n += self.elements[1].encode_to(w)?;
        Ok(n)
    }
}

impl DecodeWithParams<RingContext> for HssShare {
    fn decode_with_params<R: Read>(ctx: &RingContext, r: &mut R) -> Result<Self, IoError> {
        let b = Poly::decode_with_params(ctx, r)?;
        let a = Poly::decode_with_params(ctx, r)?;
        Ok(HssShare::new(b, a))
    }
}

// ── HssEvalKey ─────────────────────────────────────────────────────────────

impl ObjectKind for HssEvalKey {
    const OBJECT_TYPE: ObjectType = ObjectType::HSS_EVAL_KEY;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for HssEvalKey {
    fn encoded_len(&self) -> usize {
        self.share.encoded_len() + 1 + 32
    }
    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut n = 0;
        n += self.share.encode_to(w)?;
        n += write_u8(w, self.party_index)?;
        n += write_bytes(w, &self.prf_key)?;
        Ok(n)
    }
}

impl DecodeWithParams<RingContext> for HssEvalKey {
    fn decode_with_params<R: Read>(ctx: &RingContext, r: &mut R) -> Result<Self, IoError> {
        let share = HssShare::decode_with_params(ctx, r)?;
        let party_index = read_u8(r)?;
        if party_index > 1 {
            return Err(IoError::InvalidEncoding {
                detail: format!("party_index must be 0 or 1, got {}", party_index),
            });
        }
        let mut prf_key = [0u8; 32];
        read_exact(r, &mut prf_key)?;
        Ok(HssEvalKey::new(share, party_index, prf_key))
    }
}

// ── HssCiphertext ──────────────────────────────────────────────────────────
// HssCiphertext contains two silent_rlwe::Ciphertext which itself contains
// EncryptionParams (ring + rns_tool). Serializing full RLWE Ciphertext is
// complex and deferred — HssCiphertext is typically not persisted independently.
// We provide ObjectKind for type registration.

impl ObjectKind for HssCiphertext {
    const OBJECT_TYPE: ObjectType = ObjectType::HSS_CIPHERTEXT;
    const SERIALIZED_VERSION: u16 = 1;
}

#[cfg(test)]
mod tests {
    use super::*;
    use silent_math::rns::RnsBase;
    use std::io::Cursor;

    fn get_context() -> RingContext {
        let rns = RnsBase::from_values(vec![17, 97]).unwrap();
        RingContext::new(8, rns)
    }

    #[test]
    fn hss_share_roundtrip() {
        let ctx = get_context();
        let b = Poly::new(8, 2);
        let a = Poly::new(8, 2);
        let share = HssShare::new(b, a);
        let mut buf = Vec::new();
        let len = share.encode_to(&mut buf).unwrap();
        assert_eq!(len, share.encoded_len());
        let dec = HssShare::decode_with_params(&ctx, &mut Cursor::new(&buf)).unwrap();
        assert_eq!(share.elements, dec.elements);
    }

    #[test]
    fn hss_eval_key_roundtrip() {
        let ctx = get_context();
        let share = HssShare::new(Poly::new(8, 2), Poly::new(8, 2));
        let prf_key = [0xABu8; 32];
        let ek = HssEvalKey::new(share, 1, prf_key);
        let mut buf = Vec::new();
        let len = ek.encode_to(&mut buf).unwrap();
        assert_eq!(len, ek.encoded_len());
        let dec = HssEvalKey::decode_with_params(&ctx, &mut Cursor::new(&buf)).unwrap();
        assert_eq!(dec.party_index, 1);
        assert_eq!(dec.prf_key, prf_key);
    }
}
