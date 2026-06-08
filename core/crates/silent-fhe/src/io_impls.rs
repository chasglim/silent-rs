//! Canonical serialization for silent-fhe scheme wrapper types.

use crate::schemes::bgv::ciphertext::{BgvCiphertext, BgvPlaintextFactor};
use crate::schemes::bgv::params::BgvParameters;
use crate::{Degree, NoiseLevel, ShortintCiphertext, TfheParameters};

use silent_io::error::IoError;
use silent_io::object_type::ObjectType;
use silent_io::primitives::*;
use silent_io::traits::{CanonicalEncode, DecodeWithParams, ObjectKind};
use silent_rlwe::{Ciphertext, LweCiphertext};
use std::io::{Read, Write};

impl ObjectKind for BgvCiphertext {
    const OBJECT_TYPE: ObjectType = ObjectType::BGV_CIPHERTEXT;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for BgvCiphertext {
    fn encoded_len(&self) -> usize {
        4 + 8 + 8 + self.raw().encoded_len()
    }

    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let q_limbs: u32 =
            self.q_modulus_count()
                .try_into()
                .map_err(|_| IoError::InvalidEncoding {
                    detail: format!(
                        "BGV ciphertext Q-limb count {} exceeds u32::MAX",
                        self.q_modulus_count()
                    ),
                })?;
        let mut n = 0;
        n += write_u32(w, q_limbs)?;
        n += write_u64(w, self.int_factor().value())?;
        n += write_u64(w, self.noise_scale_degree() as u64)?;
        n += self.raw().encode_to(w)?;
        Ok(n)
    }
}

impl DecodeWithParams<BgvParameters> for BgvCiphertext {
    fn decode_with_params<R: Read>(params: &BgvParameters, r: &mut R) -> Result<Self, IoError> {
        let q_limbs = read_u32(r)? as usize;
        let int_factor = read_u64(r)?;
        let noise_scale_degree: usize =
            read_u64(r)?
                .try_into()
                .map_err(|_| IoError::InvalidEncoding {
                    detail: "BGV noise-scale degree does not fit usize".into(),
                })?;

        let level_params =
            params
                .with_q_prefix(q_limbs)
                .map_err(|error| IoError::InvalidEncoding {
                    detail: format!("invalid BGV ciphertext level: {error}"),
                })?;
        let factor =
            BgvPlaintextFactor::new(int_factor, level_params.plain_modulus()).map_err(|error| {
                IoError::InvalidEncoding {
                    detail: format!("invalid BGV plaintext factor: {error}"),
                }
            })?;
        let raw = Ciphertext::decode_with_params(level_params.runtime_params(), r)?;

        BgvCiphertext::with_metadata(level_params, raw, factor, noise_scale_degree).map_err(
            |error| IoError::InvalidEncoding {
                detail: format!("invalid BGV ciphertext metadata: {error}"),
            },
        )
    }
}

impl ObjectKind for ShortintCiphertext {
    const OBJECT_TYPE: ObjectType = ObjectType::SHORTINT_CIPHERTEXT;
    const SERIALIZED_VERSION: u16 = 1;
}

impl CanonicalEncode for ShortintCiphertext {
    fn encoded_len(&self) -> usize {
        self.lwe_ciphertext().encoded_len() + 8 + 8 + 8 + 8
    }
    fn encode_to<W: Write>(&self, w: &mut W) -> Result<usize, IoError> {
        let mut n = 0;
        n += self.lwe_ciphertext().encode_to(w)?;
        n += write_u64(w, self.degree().get())?;
        n += write_u64(w, self.noise_level().get())?;
        n += write_u64(w, self.message_modulus())?;
        n += write_u64(w, self.carry_modulus())?;
        Ok(n)
    }
}

impl DecodeWithParams<TfheParameters> for ShortintCiphertext {
    fn decode_with_params<R: Read>(params: &TfheParameters, r: &mut R) -> Result<Self, IoError> {
        let lwe_params = (params.lwe_dimension(), params.ciphertext_modulus_log());
        let inner = LweCiphertext::decode_with_params(&lwe_params, r)?;
        let degree = Degree::new(read_u64(r)?);
        let noise = NoiseLevel::new(read_u64(r)?);
        let mm = read_u64(r)?;
        let cm = read_u64(r)?;
        if mm != params.message_modulus().0 || cm != params.carry_modulus().0 {
            return Err(IoError::InvalidEncoding {
                detail: format!(
                    "ShortintCiphertext mm/cm ({}/{}) != expected ({}/{})",
                    mm,
                    cm,
                    params.message_modulus().0,
                    params.carry_modulus().0
                ),
            });
        }
        Ok(ShortintCiphertext::from_parts(inner, degree, noise, mm, cm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::encoder::HeEncoder;
    use crate::core::evaluator::HeEvaluator;
    use crate::core::keys::KeyGenerator;
    use crate::schemes::bgv::crypto::{BgvDecryptor, BgvEncryptor};
    use crate::schemes::bgv::encoding::BgvBatchEncoder;
    use crate::schemes::bgv::keys::BgvKeyGenerator;
    use crate::schemes::bgv::ops::BgvEvaluator;
    use rand_core::SeedableRng;
    use silent_math::modulus::Modulus;
    use silent_math::rns::RnsBase;
    use silent_math::rns_tool::RnsToolConfig;
    use silent_params::CiphertextModulusLog;
    use silent_params::presets::toy::toy_tfhe_n512;
    use silent_rlwe::{
        EncryptionParams, EvaluationKey, EvaluationKeyDecodeParams, GaloisKey,
        GaloisKeyDecodeParams, LweCiphertext,
    };
    use silent_utils::rng::SecureRng;
    use std::io::Cursor;
    use std::sync::Arc;

    fn cm(log: u8) -> CiphertextModulusLog {
        CiphertextModulusLog(log)
    }

    fn create_bgv_io_context(degree: usize) -> BgvParameters {
        let base_q = RnsBase::from_values(vec![
            1152921504606830593u64,
            1152921504606748673u64,
            1152921504606683137u64,
            1152921504606601217u64,
        ])
        .unwrap();
        let base_t = Modulus::new(65_537).unwrap();
        let config = RnsToolConfig::new(base_q, base_t);
        let params = EncryptionParams::from_rns_config(degree, config).unwrap();

        BgvParameters::from_runtime(
            "test-bgv-io-v1",
            silent_params::DistributionType::Ternary,
            silent_params::SecurityLevel::NotSet,
            silent_params::MultiplicativeDepth(1),
            silent_params::PlaintextModulus(65_537),
            Arc::new(params),
        )
        .expect("BGV IO params")
    }

    #[test]
    fn shortint_ciphertext_roundtrip() {
        let p = toy_tfhe_n512();
        let params = TfheParameters::new(p).unwrap();

        let lwe = LweCiphertext::zeros(silent_params::LweDimension(512), cm(64));
        let ct = ShortintCiphertext::from_parts(lwe, Degree::new(1), NoiseLevel::NOMINAL, 4, 4);

        let mut buf = Vec::new();
        let len = ct.encode_to(&mut buf).unwrap();
        assert_eq!(len, ct.encoded_len());

        let dec = ShortintCiphertext::decode_with_params(&params, &mut Cursor::new(&buf)).unwrap();
        assert_eq!(ct, dec);
    }

    #[test]
    fn bgv_ciphertext_roundtrip_preserves_level_factor_and_raw_data() {
        let degree = 8;
        let context = create_bgv_io_context(degree);
        let mut keygen = BgvKeyGenerator::new(context.clone());
        let sk = keygen.generate_secret_key();

        let encoder = BgvBatchEncoder::new(context.clone());
        let msg = vec![4, 8, 15, 16, 23, 42, 7, 11];
        let rng = SecureRng::from_seed([67u8; 32]);
        let mut encryptor = BgvEncryptor::new(context.clone(), rng);
        let ct = encryptor.encrypt_symmetric_bgv(&sk, &encoder.encode(&msg));

        let evaluator = BgvEvaluator::new(context.clone());
        let stored = evaluator
            .compress_bgv(&ct, 2)
            .and_then(|ct| evaluator.scale_plaintext_factor_bgv(&ct, 7))
            .expect("build stored BGV ciphertext");

        let mut buf = Vec::new();
        let len = stored.encode_to(&mut buf).unwrap();
        assert_eq!(len, stored.encoded_len());
        assert_eq!(BgvCiphertext::OBJECT_TYPE, ObjectType::BGV_CIPHERTEXT);

        let decoded = BgvCiphertext::decode_with_params(&context, &mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded.q_modulus_count(), stored.q_modulus_count());
        assert_eq!(decoded.int_factor(), stored.int_factor());
        assert_eq!(decoded.noise_scale_degree(), stored.noise_scale_degree());
        assert_eq!(decoded.raw().data.len(), stored.raw().data.len());
        assert_eq!(decoded.raw().is_ntt, stored.raw().is_ntt);
        assert_eq!(decoded.raw().seed, stored.raw().seed);
        assert_eq!(decoded.raw().is_seeded_a, stored.raw().is_seeded_a);
        for (left, right) in decoded.raw().data.iter().zip(stored.raw().data.iter()) {
            assert_eq!(left, right);
        }

        let decryptor = BgvDecryptor::new(context.clone(), sk);
        let level_encoder = BgvBatchEncoder::new(decoded.params().clone());
        assert_eq!(level_encoder.decode(&decryptor.decrypt_bgv(&decoded)), msg);

        let mut bad_factor = buf.clone();
        bad_factor[4..12].copy_from_slice(&0u64.to_le_bytes());
        assert!(matches!(
            BgvCiphertext::decode_with_params(&context, &mut Cursor::new(&bad_factor)),
            Err(IoError::InvalidEncoding { .. })
        ));
    }

    #[test]
    fn bgv_key_switching_keys_roundtrip_remain_usable() {
        let degree = 8;
        let context = create_bgv_io_context(degree);
        let mut keygen = BgvKeyGenerator::new(context.clone());
        let sk = keygen.generate_secret_key();
        let evk = keygen.relinearization_key(&sk).unwrap();
        let gk = keygen.galois_keys(&sk, &[3]).unwrap();

        let mut evk_buf = Vec::new();
        let evk_len = evk.encode_to(&mut evk_buf).unwrap();
        assert_eq!(evk_len, evk.encoded_len());

        let evk_decode_params = EvaluationKeyDecodeParams {
            ciphertext_params: context.runtime_params().clone(),
            expected_element_count: Some(evk.elements.len()),
        };
        let decoded_evk =
            EvaluationKey::decode_with_params(&evk_decode_params, &mut Cursor::new(&evk_buf))
                .unwrap();
        assert_eq!(decoded_evk.elements.len(), evk.elements.len());

        let mut gk_buf = Vec::new();
        let gk_len = gk.encode_to(&mut gk_buf).unwrap();
        assert_eq!(gk_len, gk.encoded_len());
        let gk_decode_params = GaloisKeyDecodeParams {
            ciphertext_params: context.runtime_params().clone(),
            expected_galois_elements: Some(vec![3]),
            expected_elements_per_key: Some(gk.keys.get(&3).unwrap().elements.len()),
        };
        let decoded_gk =
            GaloisKey::decode_with_params(&gk_decode_params, &mut Cursor::new(&gk_buf)).unwrap();
        assert!(decoded_gk.keys.contains_key(&3));

        let encoder = BgvBatchEncoder::new(context.clone());
        let msg_a = vec![2, 3, 5, 7, 11, 13, 17, 19];
        let msg_b = vec![4, 6, 8, 10, 12, 14, 16, 18];
        let rng = SecureRng::from_seed([71u8; 32]);
        let mut encryptor = BgvEncryptor::new(context.clone(), rng);
        let ct_a = encryptor.encrypt_symmetric_bgv(&sk, &encoder.encode(&msg_a));
        let ct_b = encryptor.encrypt_symmetric_bgv(&sk, &encoder.encode(&msg_b));

        let evaluator = BgvEvaluator::new(context.clone());
        let product = evaluator
            .mul_relinearized_bgv(&ct_a, &ct_b, &decoded_evk)
            .unwrap();
        let rotated = evaluator.rotate_bgv(&ct_a, 3, &decoded_gk).unwrap();

        let decryptor = BgvDecryptor::new(context.clone(), sk);
        let plain_modulus = context.plain_modulus();
        let expected_product = msg_a
            .iter()
            .zip(msg_b.iter())
            .map(|(&a, &b)| (a * b) % plain_modulus)
            .collect::<Vec<_>>();
        assert_eq!(
            encoder.decode(&decryptor.decrypt_bgv(&product)),
            expected_product
        );

        let map = context.ring().gen_automorphism_map(3);
        let expected_rotation = (0..degree).map(|i| msg_a[map[i]]).collect::<Vec<_>>();
        assert_eq!(
            encoder.decode(&decryptor.decrypt_bgv(&rotated)),
            expected_rotation
        );
    }
}
