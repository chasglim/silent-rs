use crate::code::{LinearCode, ReedSolomonCode};
use crate::error::LpnError;
use crate::field::{add, ensure_modulus, hamming_weight, mul, neg, sub};
use crate::linalg::{add_vectors, dot, scale_add_assign, sub_vectors};
use crate::lpn::{LpnParams, LpnPublic};
use crate::sampler::{SparseVector, sample_bernoulli_error, sample_uniform_vector};
use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct LpnBksHssParams {
    pub modulus: u64,
    pub input_len: usize,
    pub output_len: usize,
    pub code_len: usize,
    pub decode_radius: usize,
    pub kappa: usize,
    pub mask_len: usize,
    pub client_weight: usize,
    pub noise_rate_x: f64,
    pub noise_rate_u: f64,
}

impl LpnBksHssParams {
    pub fn validate(&self) -> Result<(), LpnError> {
        ensure_modulus(self.modulus)?;
        if self.input_len == 0 || self.output_len == 0 {
            return Err(LpnError::InvalidParams(
                "LPN-HSS params require positive input and output lengths",
            ));
        }
        if self.code_len <= self.output_len {
            return Err(LpnError::InvalidParams(
                "LPN-HSS code_len must be larger than output_len",
            ));
        }
        if self.decode_radius > (self.code_len - self.output_len) / 2 {
            return Err(LpnError::InvalidParams(
                "LPN-HSS decode radius exceeds Reed-Solomon radius",
            ));
        }
        if !(0.0..=1.0).contains(&self.noise_rate_x) || !(0.0..=1.0).contains(&self.noise_rate_u) {
            return Err(LpnError::InvalidParams(
                "LPN-HSS noise rates must be in [0,1]",
            ));
        }
        LpnParams {
            modulus: self.modulus,
            input_len: self.input_len,
            kappa: self.kappa,
            mask_len: self.mask_len,
            client_weight: self.client_weight,
        }
        .validate()
    }

    pub fn online_comm_field_elements(&self, client_output: bool) -> usize {
        self.kappa
            + self.code_len * (self.input_len + self.mask_len)
            + (self.code_len - self.output_len) * self.kappa
            + if client_output { self.output_len } else { 0 }
    }
}

#[derive(Clone, Debug)]
pub struct LpnBksHssPublic {
    pub params: LpnBksHssParams,
    pub lpn: LpnPublic,
    pub code: ReedSolomonCode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoisyBilinearOutput {
    pub digest: Vec<u64>,
    pub client_mask: SparseVector,
    pub client_codeword_share: Vec<u64>,
    pub server_codeword_share: Vec<u64>,
    pub target: Vec<u64>,
    pub clean_codeword: Vec<u64>,
    pub error: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LpnBksHssShareOutput {
    pub client_share: Vec<u64>,
    pub server_share: Vec<u64>,
    pub noisy: NoisyBilinearOutput,
    pub decoded_error_weight: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LpnBksClientDigest {
    pub digest: Vec<u64>,
    pub client_mask: SparseVector,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LpnBksServerEncoding {
    pub e: Vec<u64>,
    pub e_prime: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LpnBksServerPreprocessing {
    pub encodings: Vec<LpnBksServerEncoding>,
    pub server_masks: Vec<Vec<u64>>,
    pub syndrome_key: Vec<Vec<u64>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LpnBksServerEncodingOutput {
    pub encodings: Vec<LpnBksServerEncoding>,
    pub server_codeword_share: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LpnBksPlainCrscServerOutput {
    pub server_share: Vec<u64>,
    pub decoded_error: Vec<u64>,
}

impl LpnBksHssPublic {
    pub fn setup<R: Rng + ?Sized>(params: LpnBksHssParams, rng: &mut R) -> Result<Self, LpnError> {
        params.validate()?;
        let lpn = LpnPublic::setup(
            LpnParams {
                modulus: params.modulus,
                input_len: params.input_len,
                kappa: params.kappa,
                mask_len: params.mask_len,
                client_weight: params.client_weight,
            },
            rng,
        )?;
        let code = ReedSolomonCode::new(
            params.modulus,
            params.code_len,
            params.output_len,
            params.decode_radius,
        )?;
        Ok(Self { params, lpn, code })
    }

    pub fn client_digest<R: Rng + ?Sized>(
        &self,
        client_x: &[u64],
        rng: &mut R,
    ) -> Result<LpnBksClientDigest, LpnError> {
        if client_x.len() != self.params.input_len {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.params.input_len,
                rhs: client_x.len(),
                context: "LpnBksHssPublic::client_digest",
            });
        }
        let client_mask = self.lpn.sample_client_mask(rng)?;
        let digest = self.lpn.digest(client_x, &client_mask)?;
        Ok(LpnBksClientDigest {
            digest,
            client_mask,
        })
    }

    pub fn server_encode_from_digest<R: Rng + ?Sized>(
        &self,
        digest: &[u64],
        server_vectors: &[Vec<u64>],
        rng: &mut R,
    ) -> Result<LpnBksServerEncodingOutput, LpnError> {
        let preprocessing = self.server_preprocess_encodings(server_vectors, rng)?;
        let server_codeword_share =
            self.server_codeword_share_from_preprocessing(digest, &preprocessing)?;
        Ok(LpnBksServerEncodingOutput {
            encodings: preprocessing.encodings,
            server_codeword_share,
        })
    }

    pub fn server_preprocess_encodings<R: Rng + ?Sized>(
        &self,
        server_vectors: &[Vec<u64>],
        rng: &mut R,
    ) -> Result<LpnBksServerPreprocessing, LpnError> {
        self.check_server_vectors(server_vectors)?;
        let q = self.params.modulus;
        let mut encodings = Vec::with_capacity(self.params.code_len);
        let mut server_masks = Vec::with_capacity(self.params.code_len);

        for j in 0..self.params.code_len {
            let mut bar_y = vec![0; self.params.input_len];
            for (ell, y) in server_vectors.iter().enumerate() {
                let coeff = self.code.generator().get(j, ell)?;
                scale_add_assign(&mut bar_y, coeff, y, q)?;
            }

            let s_j = sample_uniform_vector(self.params.kappa, q, rng)?;
            let xi_j =
                sample_bernoulli_error(self.params.input_len, self.params.noise_rate_x, q, rng)?;
            let xi_prime_j =
                sample_bernoulli_error(self.params.mask_len, self.params.noise_rate_u, q, rng)?;

            let a_t_s = self.lpn.a.transpose_mul_vec(&s_j)?;
            let b_t_s = self.lpn.b.transpose_mul_vec(&s_j)?;

            let mut e = add_vectors(&a_t_s, &bar_y, q)?;
            e = add_vectors(&e, &xi_j, q)?;
            let e_prime = add_vectors(&b_t_s, &xi_prime_j, q)?;

            encodings.push(LpnBksServerEncoding { e, e_prime });
            server_masks.push(s_j);
        }
        let syndrome_key = self.syndrome_key_from_masks(&server_masks)?;

        Ok(LpnBksServerPreprocessing {
            encodings,
            server_masks,
            syndrome_key,
        })
    }

    pub fn server_codeword_share_from_preprocessing(
        &self,
        digest: &[u64],
        preprocessing: &LpnBksServerPreprocessing,
    ) -> Result<Vec<u64>, LpnError> {
        if digest.len() != self.params.kappa {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.params.kappa,
                rhs: digest.len(),
                context: "LpnBksHssPublic::server_codeword_share_from_preprocessing digest",
            });
        }
        if preprocessing.encodings.len() != self.params.code_len
            || preprocessing.server_masks.len() != self.params.code_len
        {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.params.code_len,
                rhs: preprocessing
                    .encodings
                    .len()
                    .min(preprocessing.server_masks.len()),
                context: "LpnBksHssPublic::server_codeword_share_from_preprocessing preprocessing",
            });
        }
        let q = self.params.modulus;
        let mut out = Vec::with_capacity(self.params.code_len);
        for s_j in &preprocessing.server_masks {
            if s_j.len() != self.params.kappa {
                return Err(LpnError::VectorLengthMismatch {
                    lhs: self.params.kappa,
                    rhs: s_j.len(),
                    context: "LpnBksHssPublic::server_codeword_share_from_preprocessing mask",
                });
            }
            out.push(neg(dot(s_j, digest, q)?, q));
        }
        Ok(out)
    }

    pub fn client_codeword_share_from_encodings(
        &self,
        client_x: &[u64],
        client_mask: &SparseVector,
        encodings: &[LpnBksServerEncoding],
    ) -> Result<Vec<u64>, LpnError> {
        if client_x.len() != self.params.input_len {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.params.input_len,
                rhs: client_x.len(),
                context: "LpnBksHssPublic::client_codeword_share_from_encodings client_x",
            });
        }
        if client_mask.len() != self.params.mask_len || client_mask.modulus() != self.params.modulus
        {
            return Err(LpnError::InvalidParams(
                "LPN-HSS client mask length or modulus mismatch",
            ));
        }
        if encodings.len() != self.params.code_len {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.params.code_len,
                rhs: encodings.len(),
                context: "LpnBksHssPublic::client_codeword_share_from_encodings encodings",
            });
        }
        let q = self.params.modulus;
        let mask_dense = client_mask.to_dense();
        let mut out = Vec::with_capacity(self.params.code_len);
        for encoding in encodings {
            if encoding.e.len() != self.params.input_len
                || encoding.e_prime.len() != self.params.mask_len
            {
                return Err(LpnError::InvalidParams(
                    "LPN-HSS server encoding length mismatch",
                ));
            }
            out.push(add(
                dot(&encoding.e, client_x, q)?,
                dot(&encoding.e_prime, &mask_dense, q)?,
                q,
            ));
        }
        Ok(out)
    }

    pub fn syndrome_key_from_masks(
        &self,
        server_masks: &[Vec<u64>],
    ) -> Result<Vec<Vec<u64>>, LpnError> {
        if server_masks.len() != self.params.code_len {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.params.code_len,
                rhs: server_masks.len(),
                context: "LpnBksHssPublic::syndrome_key_from_masks",
            });
        }
        let rows = self.params.code_len - self.params.output_len;
        let q = self.params.modulus;
        let mut key = vec![vec![0; self.params.kappa]; rows];
        for (j, mask) in server_masks.iter().enumerate() {
            if mask.len() != self.params.kappa {
                return Err(LpnError::VectorLengthMismatch {
                    lhs: self.params.kappa,
                    rhs: mask.len(),
                    context: "LpnBksHssPublic::syndrome_key_from_masks mask",
                });
            }
            for row in 0..rows {
                let h = self.code.parity_check().get(row, j)?;
                if h == 0 {
                    continue;
                }
                for (col, &mask_value) in mask.iter().enumerate() {
                    key[row][col] = add(key[row][col], mul(h, mask_value, q), q);
                }
            }
        }
        Ok(key)
    }

    pub fn syndrome_from_key(
        &self,
        client_codeword_share: &[u64],
        digest: &[u64],
        syndrome_key: &[Vec<u64>],
    ) -> Result<Vec<u64>, LpnError> {
        if digest.len() != self.params.kappa {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.params.kappa,
                rhs: digest.len(),
                context: "LpnBksHssPublic::syndrome_from_key digest",
            });
        }
        let rows = self.params.code_len - self.params.output_len;
        if syndrome_key.len() != rows {
            return Err(LpnError::VectorLengthMismatch {
                lhs: rows,
                rhs: syndrome_key.len(),
                context: "LpnBksHssPublic::syndrome_from_key key rows",
            });
        }
        let q = self.params.modulus;
        let mut syndrome = self.code.syndrome(client_codeword_share)?;
        for (row, key_row) in syndrome_key.iter().enumerate() {
            if key_row.len() != self.params.kappa {
                return Err(LpnError::VectorLengthMismatch {
                    lhs: self.params.kappa,
                    rhs: key_row.len(),
                    context: "LpnBksHssPublic::syndrome_from_key key row",
                });
            }
            syndrome[row] = sub(syndrome[row], dot(key_row, digest, q)?, q);
        }
        Ok(syndrome)
    }

    pub fn syndrome_key_crsc_convert(
        &self,
        client_codeword_share: &[u64],
        server_codeword_share: &[u64],
        digest: &[u64],
        syndrome_key: &[Vec<u64>],
    ) -> Result<crate::code::PlainCrscOutput, LpnError> {
        let syndrome = self.syndrome_from_key(client_codeword_share, digest, syndrome_key)?;
        let error = self.code.decode_error_from_syndrome(&syndrome)?;
        let client_clean = sub_vectors(client_codeword_share, &error, self.params.modulus)?;
        let client_share = self.code.recover(&client_clean)?;
        let server_share = self.code.recover(server_codeword_share)?;
        Ok(crate::code::PlainCrscOutput {
            client_share,
            server_share,
            error,
        })
    }

    pub fn noisy_bilinear<R: Rng + ?Sized>(
        &self,
        client_x: &[u64],
        server_vectors: &[Vec<u64>],
        rng: &mut R,
    ) -> Result<NoisyBilinearOutput, LpnError> {
        self.check_inputs(client_x, server_vectors)?;
        let client = self.client_digest(client_x, rng)?;
        let server = self.server_encode_from_digest(&client.digest, server_vectors, rng)?;
        let client_codeword_share = self.client_codeword_share_from_encodings(
            client_x,
            &client.client_mask,
            &server.encodings,
        )?;
        let q = self.params.modulus;
        let mut target = Vec::with_capacity(self.params.output_len);
        for y in server_vectors {
            target.push(dot(client_x, y, q)?);
        }
        let clean_codeword = self.code.encode(&target)?;
        let opened = add_vectors(
            &client_codeword_share,
            &server.server_codeword_share,
            self.params.modulus,
        )?;
        let error = crate::linalg::sub_vectors(&opened, &clean_codeword, self.params.modulus)?;

        Ok(NoisyBilinearOutput {
            digest: client.digest,
            client_mask: client.client_mask,
            client_codeword_share,
            server_codeword_share: server.server_codeword_share,
            target,
            clean_codeword,
            error,
        })
    }

    pub fn evaluate<R: Rng + ?Sized>(
        &self,
        client_x: &[u64],
        server_vectors: &[Vec<u64>],
        rng: &mut R,
    ) -> Result<LpnBksHssShareOutput, LpnError> {
        let preprocessing = self.server_preprocess_encodings(server_vectors, rng)?;
        let client = self.client_digest(client_x, rng)?;
        let client_codeword_share = self.client_codeword_share_from_encodings(
            client_x,
            &client.client_mask,
            &preprocessing.encodings,
        )?;
        let server_codeword_share =
            self.server_codeword_share_from_preprocessing(&client.digest, &preprocessing)?;
        let q = self.params.modulus;
        let mut target = Vec::with_capacity(self.params.output_len);
        for y in server_vectors {
            target.push(dot(client_x, y, q)?);
        }
        let clean_codeword = self.code.encode(&target)?;
        let opened = add_vectors(&client_codeword_share, &server_codeword_share, q)?;
        let error = sub_vectors(&opened, &clean_codeword, q)?;
        let noisy = NoisyBilinearOutput {
            digest: client.digest.clone(),
            client_mask: client.client_mask,
            client_codeword_share: client_codeword_share.clone(),
            server_codeword_share: server_codeword_share.clone(),
            target,
            clean_codeword,
            error,
        };
        let crsc = self.syndrome_key_crsc_convert(
            &client_codeword_share,
            &server_codeword_share,
            &client.digest,
            &preprocessing.syndrome_key,
        )?;
        let decoded_error_weight = hamming_weight(&crsc.error);
        Ok(LpnBksHssShareOutput {
            client_share: crsc.client_share,
            server_share: crsc.server_share,
            noisy,
            decoded_error_weight,
        })
    }

    pub fn check_noisy_invariant(&self, noisy: &NoisyBilinearOutput) -> Result<(), LpnError> {
        let opened = add_vectors(
            &noisy.client_codeword_share,
            &noisy.server_codeword_share,
            self.params.modulus,
        )?;
        let expected = add_vectors(&noisy.clean_codeword, &noisy.error, self.params.modulus)?;
        if opened != expected {
            return Err(LpnError::DecodeFailure(
                "noisy bilinear invariant y_C + y_S = C(m) + e failed",
            ));
        }
        Ok(())
    }

    pub fn client_plain_crsc_share(
        &self,
        client_codeword_share: &[u64],
    ) -> Result<Vec<u64>, LpnError> {
        self.code.recover(client_codeword_share)
    }

    pub fn client_syndrome_share(
        &self,
        client_codeword_share: &[u64],
    ) -> Result<Vec<u64>, LpnError> {
        self.code.syndrome(client_codeword_share)
    }

    pub fn server_plain_crsc_share(
        &self,
        server_codeword_share: &[u64],
        client_syndrome_share: &[u64],
    ) -> Result<LpnBksPlainCrscServerOutput, LpnError> {
        let server_syndrome = self.code.syndrome(server_codeword_share)?;
        let syndrome = add_vectors(client_syndrome_share, &server_syndrome, self.params.modulus)?;
        let decoded_error = self.code.decode_error_from_syndrome(&syndrome)?;
        let server_clean =
            crate::linalg::sub_vectors(server_codeword_share, &decoded_error, self.params.modulus)?;
        let server_share = self.code.recover(&server_clean)?;
        Ok(LpnBksPlainCrscServerOutput {
            server_share,
            decoded_error,
        })
    }

    fn check_inputs(&self, client_x: &[u64], server_vectors: &[Vec<u64>]) -> Result<(), LpnError> {
        if client_x.len() != self.params.input_len {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.params.input_len,
                rhs: client_x.len(),
                context: "LpnBksHssPublic::check_inputs client_x",
            });
        }
        self.check_server_vectors(server_vectors)
    }

    fn check_server_vectors(&self, server_vectors: &[Vec<u64>]) -> Result<(), LpnError> {
        if server_vectors.len() != self.params.output_len {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.params.output_len,
                rhs: server_vectors.len(),
                context: "LpnBksHssPublic::check_inputs server_vectors",
            });
        }
        for y in server_vectors {
            if y.len() != self.params.input_len {
                return Err(LpnError::VectorLengthMismatch {
                    lhs: self.params.input_len,
                    rhs: y.len(),
                    context: "LpnBksHssPublic::check_inputs server vector",
                });
            }
        }
        Ok(())
    }
}

pub fn rho_q_error_rate(weight: usize, tau: f64, field_size: u64) -> Result<f64, LpnError> {
    ensure_modulus(field_size)?;
    if !(0.0..=1.0).contains(&tau) {
        return Err(LpnError::InvalidParams(
            "rho_q_error_rate requires tau in [0,1]",
        ));
    }
    let q = field_size as f64;
    Ok((q - 1.0) / q * (1.0 - (1.0 - q * tau / (q - 1.0)).powi(weight as i32)))
}

pub fn choose_block_width(
    max_width: usize,
    tau_x: f64,
    tau_u: f64,
    client_weight: usize,
    field_size: u64,
    decode_delta: f64,
    safety_margin: f64,
) -> Result<usize, LpnError> {
    if max_width == 0 {
        return Err(LpnError::InvalidParams(
            "choose_block_width requires max_width > 0",
        ));
    }
    let target = decode_delta - safety_margin;
    for width in (1..=max_width).rev() {
        let estimate = rho_q_error_rate(width, tau_x, field_size)?
            + rho_q_error_rate(client_weight, tau_u, field_size)?;
        if estimate <= target {
            return Ok(width);
        }
    }
    Ok(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::add_vectors;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn params() -> LpnBksHssParams {
        LpnBksHssParams {
            modulus: 97,
            input_len: 4,
            output_len: 3,
            code_len: 7,
            decode_radius: 2,
            kappa: 5,
            mask_len: 6,
            client_weight: 2,
            noise_rate_x: 0.0,
            noise_rate_u: 0.0,
        }
    }

    #[test]
    fn noisy_bilinear_invariant_holds() {
        let mut rng = StdRng::seed_from_u64(123);
        let hss = LpnBksHssPublic::setup(params(), &mut rng).unwrap();
        let x = vec![1, 2, 3, 4];
        let server = vec![vec![5, 6, 7, 8], vec![1, 0, 2, 0], vec![3, 3, 3, 3]];
        let noisy = hss.noisy_bilinear(&x, &server, &mut rng).unwrap();
        hss.check_noisy_invariant(&noisy).unwrap();
    }

    #[test]
    fn lpn_bks_hss_outputs_additive_ip_shares() {
        let mut rng = StdRng::seed_from_u64(321);
        let hss = LpnBksHssPublic::setup(params(), &mut rng).unwrap();
        let x = vec![1, 2, 3, 4];
        let server = vec![vec![5, 6, 7, 8], vec![1, 0, 2, 0], vec![3, 3, 3, 3]];
        let out = hss.evaluate(&x, &server, &mut rng).unwrap();
        hss.check_noisy_invariant(&out.noisy).unwrap();
        let opened = add_vectors(&out.client_share, &out.server_share, 97).unwrap();
        assert_eq!(opened, out.noisy.target);
    }

    #[test]
    fn split_noisy_bilinear_and_syndrome_key_crsc_outputs_additive_shares() {
        let mut setup_rng = StdRng::seed_from_u64(400);
        let mut client_rng = StdRng::seed_from_u64(401);
        let mut server_rng = StdRng::seed_from_u64(402);
        let hss = LpnBksHssPublic::setup(params(), &mut setup_rng).unwrap();
        let x = vec![1, 2, 3, 4];
        let server = vec![vec![5, 6, 7, 8], vec![1, 0, 2, 0], vec![3, 3, 3, 3]];

        let client_digest = hss.client_digest(&x, &mut client_rng).unwrap();
        let preprocessing = hss
            .server_preprocess_encodings(&server, &mut server_rng)
            .unwrap();
        let server_codeword_share = hss
            .server_codeword_share_from_preprocessing(&client_digest.digest, &preprocessing)
            .unwrap();
        let y_c = hss
            .client_codeword_share_from_encodings(
                &x,
                &client_digest.client_mask,
                &preprocessing.encodings,
            )
            .unwrap();
        let out = hss
            .syndrome_key_crsc_convert(
                &y_c,
                &server_codeword_share,
                &client_digest.digest,
                &preprocessing.syndrome_key,
            )
            .unwrap();

        let opened = add_vectors(&out.client_share, &out.server_share, 97).unwrap();
        assert_eq!(opened, vec![70, 7, 30]);
    }

    #[test]
    fn server_preprocessing_matches_digest_dependent_encoding() {
        let mut setup_rng = StdRng::seed_from_u64(410);
        let mut client_rng = StdRng::seed_from_u64(411);
        let mut split_server_rng = StdRng::seed_from_u64(412);
        let mut legacy_server_rng = StdRng::seed_from_u64(412);
        let hss = LpnBksHssPublic::setup(params(), &mut setup_rng).unwrap();
        let x = vec![1, 2, 3, 4];
        let server = vec![vec![5, 6, 7, 8], vec![1, 0, 2, 0], vec![3, 3, 3, 3]];

        let client_digest = hss.client_digest(&x, &mut client_rng).unwrap();
        let preprocessing = hss
            .server_preprocess_encodings(&server, &mut split_server_rng)
            .unwrap();
        let split_share = hss
            .server_codeword_share_from_preprocessing(&client_digest.digest, &preprocessing)
            .unwrap();
        let legacy = hss
            .server_encode_from_digest(&client_digest.digest, &server, &mut legacy_server_rng)
            .unwrap();

        assert_eq!(preprocessing.encodings, legacy.encodings);
        assert_eq!(split_share, legacy.server_codeword_share);
    }

    #[test]
    fn rho_q_matches_small_noise_linear_term() {
        let rho = rho_q_error_rate(4, 0.001, 97).unwrap();
        assert!((rho - 0.004).abs() < 0.0001);
    }
}
