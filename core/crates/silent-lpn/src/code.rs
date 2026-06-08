use crate::error::LpnError;
use crate::field::{hamming_weight, mul, neg, pow, sub};
use crate::linalg::{
    FieldMatrix, add_vectors, invert_square, nullspace_basis_rows, solve_linear_system, sub_vectors,
};

pub trait LinearCode {
    fn modulus(&self) -> u64;
    fn n(&self) -> usize;
    fn r(&self) -> usize;
    fn t(&self) -> usize;
    fn encode(&self, msg: &[u64]) -> Result<Vec<u64>, LpnError>;
    fn syndrome(&self, word: &[u64]) -> Result<Vec<u64>, LpnError>;
    fn decode_error_from_syndrome(&self, syndrome: &[u64]) -> Result<Vec<u64>, LpnError>;
    fn recover(&self, clean_codeword: &[u64]) -> Result<Vec<u64>, LpnError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReedSolomonCode {
    modulus: u64,
    n: usize,
    r: usize,
    t: usize,
    points: Vec<u64>,
    generator: FieldMatrix,
    parity_check: FieldMatrix,
    recovery: FieldMatrix,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlainCrscOutput {
    pub client_share: Vec<u64>,
    pub server_share: Vec<u64>,
    pub error: Vec<u64>,
}

impl ReedSolomonCode {
    pub fn new(modulus: u64, n: usize, r: usize, t: usize) -> Result<Self, LpnError> {
        if r == 0 || n <= r {
            return Err(LpnError::InvalidParams(
                "ReedSolomonCode requires 0 < r < n",
            ));
        }
        if n as u64 >= modulus {
            return Err(LpnError::InvalidParams(
                "ReedSolomonCode requires n < modulus for distinct evaluation points",
            ));
        }
        if t > (n - r) / 2 {
            return Err(LpnError::InvalidParams(
                "ReedSolomonCode radius exceeds bounded-distance decoding radius",
            ));
        }
        let points = (1..=n).map(|i| i as u64).collect::<Vec<_>>();
        let mut generator = FieldMatrix::zeros(n, r, modulus)?;
        for row in 0..n {
            let alpha = points[row];
            let mut power = 1;
            for col in 0..r {
                generator.set(row, col, power)?;
                power = mul(power, alpha, modulus);
            }
        }

        let generator_t = generator.transpose()?;
        let parity_check = nullspace_basis_rows(&generator_t)?;
        if parity_check.rows() != n - r {
            return Err(LpnError::InvalidParams(
                "ReedSolomonCode failed to derive full-rank parity check",
            ));
        }

        let mut top_rows = Vec::with_capacity(r);
        for row in 0..r {
            top_rows.push(generator.row(row)?);
        }
        let top = FieldMatrix::from_rows(&top_rows, modulus)?;
        let top_inv = invert_square(&top)?;
        let mut recovery = FieldMatrix::zeros(r, n, modulus)?;
        for row in 0..r {
            for col in 0..r {
                recovery.set(row, col, top_inv.get(row, col)?)?;
            }
        }

        Ok(Self {
            modulus,
            n,
            r,
            t,
            points,
            generator,
            parity_check,
            recovery,
        })
    }

    pub fn generator(&self) -> &FieldMatrix {
        &self.generator
    }

    pub fn parity_check(&self) -> &FieldMatrix {
        &self.parity_check
    }

    pub fn recovery(&self) -> &FieldMatrix {
        &self.recovery
    }

    pub fn points(&self) -> &[u64] {
        &self.points
    }

    pub fn decode_word(&self, word: &[u64]) -> Result<Vec<u64>, LpnError> {
        self.check_word_len(word, "ReedSolomonCode::decode_word")?;
        if self.t == 0 {
            return self.recover(word);
        }
        if let Ok(msg) = self.berlekamp_welch_decode(word) {
            return Ok(msg);
        }
        let error = self.decode_error_from_syndrome(&self.syndrome(word)?)?;
        let clean = sub_vectors(word, &error, self.modulus)?;
        self.recover(&clean)
    }

    pub fn decode_error_from_word(&self, word: &[u64]) -> Result<Vec<u64>, LpnError> {
        let msg = self.decode_word(word)?;
        let codeword = self.encode(&msg)?;
        let error = sub_vectors(word, &codeword, self.modulus)?;
        if hamming_weight(&error) > self.t {
            return Err(LpnError::DecodeFailure(
                "decoded word is outside the configured radius",
            ));
        }
        Ok(error)
    }

    pub fn plain_crsc_convert(
        &self,
        client_share: &[u64],
        server_share: &[u64],
    ) -> Result<PlainCrscOutput, LpnError> {
        self.check_word_len(client_share, "ReedSolomonCode::plain_crsc_convert client")?;
        self.check_word_len(server_share, "ReedSolomonCode::plain_crsc_convert server")?;
        let noisy_word = add_vectors(client_share, server_share, self.modulus)?;
        let error = self.decode_error_from_word(&noisy_word)?;
        let client_clean = sub_vectors(client_share, &error, self.modulus)?;
        let client_out = self.recover(&client_clean)?;
        let server_out = self.recover(server_share)?;
        Ok(PlainCrscOutput {
            client_share: client_out,
            server_share: server_out,
            error,
        })
    }

    fn check_msg_len(&self, msg: &[u64], context: &'static str) -> Result<(), LpnError> {
        if msg.len() != self.r {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.r,
                rhs: msg.len(),
                context,
            });
        }
        Ok(())
    }

    fn check_word_len(&self, word: &[u64], context: &'static str) -> Result<(), LpnError> {
        if word.len() != self.n {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.n,
                rhs: word.len(),
                context,
            });
        }
        Ok(())
    }

    fn berlekamp_welch_decode(&self, word: &[u64]) -> Result<Vec<u64>, LpnError> {
        let q = self.modulus;
        let t = self.t;
        let q_degree_terms = self.r + t;
        let unknowns = q_degree_terms + t;
        if unknowns > self.n {
            return Err(LpnError::DecodeFailure(
                "not enough equations for Berlekamp-Welch decode",
            ));
        }

        let mut matrix = Vec::with_capacity(self.n);
        let mut rhs = Vec::with_capacity(self.n);
        for (&alpha, &y) in self.points.iter().zip(word.iter()) {
            let mut row = vec![0; unknowns];
            let mut power = 1;
            for slot in row.iter_mut().take(q_degree_terms) {
                *slot = power;
                power = mul(power, alpha, q);
            }
            power = 1;
            for h in 0..t {
                row[q_degree_terms + h] = neg(mul(y, power, q), q);
                power = mul(power, alpha, q);
            }
            rhs.push(mul(y, pow(alpha, t as u64, q), q));
            matrix.push(row);
        }

        let solution = solve_linear_system(&matrix, &rhs, q)?.ok_or(LpnError::DecodeFailure(
            "Berlekamp-Welch system inconsistent",
        ))?;
        let q_poly = solution[..q_degree_terms].to_vec();
        let mut e_poly = solution[q_degree_terms..].to_vec();
        e_poly.push(1);
        let (message, remainder) = poly_div(&q_poly, &e_poly, q)?;
        if remainder.iter().any(|&v| v != 0) {
            return Err(LpnError::DecodeFailure(
                "Berlekamp-Welch division had nonzero remainder",
            ));
        }
        let mut msg = vec![0; self.r];
        for (dst, src) in msg.iter_mut().zip(message.iter()) {
            *dst = *src;
        }
        let codeword = self.encode(&msg)?;
        let errors = word
            .iter()
            .zip(codeword.iter())
            .filter(|(a, b)| (*a % q) != (*b % q))
            .count();
        if errors > self.t {
            return Err(LpnError::DecodeFailure(
                "Berlekamp-Welch candidate exceeds error radius",
            ));
        }
        Ok(msg)
    }
}

impl LinearCode for ReedSolomonCode {
    fn modulus(&self) -> u64 {
        self.modulus
    }

    fn n(&self) -> usize {
        self.n
    }

    fn r(&self) -> usize {
        self.r
    }

    fn t(&self) -> usize {
        self.t
    }

    fn encode(&self, msg: &[u64]) -> Result<Vec<u64>, LpnError> {
        self.check_msg_len(msg, "ReedSolomonCode::encode")?;
        self.generator.mul_vec(msg)
    }

    fn syndrome(&self, word: &[u64]) -> Result<Vec<u64>, LpnError> {
        self.check_word_len(word, "ReedSolomonCode::syndrome")?;
        self.parity_check.mul_vec(word)
    }

    fn decode_error_from_syndrome(&self, syndrome: &[u64]) -> Result<Vec<u64>, LpnError> {
        if syndrome.len() != self.n - self.r {
            return Err(LpnError::VectorLengthMismatch {
                lhs: self.n - self.r,
                rhs: syndrome.len(),
                context: "ReedSolomonCode::decode_error_from_syndrome",
            });
        }
        if syndrome.iter().all(|&v| v == 0) {
            return Ok(vec![0; self.n]);
        }
        for weight in 1..=self.t {
            let mut support = Vec::with_capacity(weight);
            if let Some(error) = self.search_support(0, weight, syndrome, &mut support)? {
                return Ok(error);
            }
        }
        Err(LpnError::DecodeFailure(
            "no syndrome coset leader within configured radius",
        ))
    }

    fn recover(&self, clean_codeword: &[u64]) -> Result<Vec<u64>, LpnError> {
        self.check_word_len(clean_codeword, "ReedSolomonCode::recover")?;
        self.recovery.mul_vec(clean_codeword)
    }
}

impl ReedSolomonCode {
    fn search_support(
        &self,
        start: usize,
        remaining: usize,
        syndrome: &[u64],
        support: &mut Vec<usize>,
    ) -> Result<Option<Vec<u64>>, LpnError> {
        if remaining == 0 {
            return self.solve_support(syndrome, support);
        }
        let max_start = self.n.saturating_sub(remaining);
        for idx in start..=max_start {
            support.push(idx);
            if let Some(error) = self.search_support(idx + 1, remaining - 1, syndrome, support)? {
                return Ok(Some(error));
            }
            support.pop();
        }
        Ok(None)
    }

    fn solve_support(
        &self,
        syndrome: &[u64],
        support: &[usize],
    ) -> Result<Option<Vec<u64>>, LpnError> {
        let equations = self.parity_check.rows();
        let mut matrix = vec![vec![0; support.len()]; equations];
        for row in 0..equations {
            for (col, &pos) in support.iter().enumerate() {
                matrix[row][col] = self.parity_check.get(row, pos)?;
            }
        }
        let Some(values) = solve_linear_system(&matrix, syndrome, self.modulus)? else {
            return Ok(None);
        };
        if values.iter().any(|&v| v == 0) {
            return Ok(None);
        }
        let mut error = vec![0; self.n];
        for (&pos, &value) in support.iter().zip(values.iter()) {
            error[pos] = value;
        }
        if self.syndrome(&error)? == syndrome {
            Ok(Some(error))
        } else {
            Ok(None)
        }
    }
}

fn poly_div(
    numerator: &[u64],
    denominator: &[u64],
    modulus: u64,
) -> Result<(Vec<u64>, Vec<u64>), LpnError> {
    if denominator.is_empty() || denominator.iter().all(|&v| v == 0) {
        return Err(LpnError::InvalidParams("poly_div denominator is zero"));
    }
    let mut rem = numerator.to_vec();
    trim_poly(&mut rem);
    let mut den = denominator.to_vec();
    trim_poly(&mut den);
    if rem.len() < den.len() {
        return Ok((vec![0], rem));
    }
    let mut quo = vec![0; rem.len() - den.len() + 1];
    let den_lead = *den.last().unwrap();
    while rem.len() >= den.len() && !rem.is_empty() {
        let degree_delta = rem.len() - den.len();
        let coeff = crate::field::div(*rem.last().unwrap(), den_lead, modulus).ok_or(
            LpnError::InvalidParams("poly_div non-invertible leading coeff"),
        )?;
        quo[degree_delta] = coeff;
        for (i, &den_coeff) in den.iter().enumerate() {
            let idx = degree_delta + i;
            rem[idx] = sub(rem[idx], mul(coeff, den_coeff, modulus), modulus);
        }
        trim_poly(&mut rem);
    }
    if quo.is_empty() {
        quo.push(0);
    }
    if rem.is_empty() {
        rem.push(0);
    }
    Ok((quo, rem))
}

fn trim_poly(poly: &mut Vec<u64>) {
    while poly.last().is_some_and(|&v| v == 0) {
        poly.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::add;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn rs_recovers_message_from_bounded_errors() {
        let code = ReedSolomonCode::new(97, 7, 3, 2).unwrap();
        let msg = vec![4, 5, 6];
        let mut word = code.encode(&msg).unwrap();
        word[1] = add(word[1], 11, 97);
        word[5] = add(word[5], 9, 97);
        let decoded = code.decode_word(&word).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn syndrome_decoder_finds_exact_error() {
        let code = ReedSolomonCode::new(97, 7, 3, 2).unwrap();
        let mut error = vec![0; 7];
        error[2] = 17;
        error[6] = 19;
        let syndrome = code.syndrome(&error).unwrap();
        let decoded = code.decode_error_from_syndrome(&syndrome).unwrap();
        assert_eq!(decoded, error);
    }

    #[test]
    fn plain_crsc_outputs_additive_message_shares() {
        let mut rng = StdRng::seed_from_u64(77);
        let code = ReedSolomonCode::new(97, 7, 3, 2).unwrap();
        let msg = vec![8, 9, 10];
        let codeword = code.encode(&msg).unwrap();
        let mut noisy = codeword.clone();
        noisy[0] = add(noisy[0], 3, 97);
        noisy[4] = add(noisy[4], 6, 97);
        let client = (0..7).map(|_| rng.gen_range(0..97)).collect::<Vec<_>>();
        let server = sub_vectors(&noisy, &client, 97).unwrap();
        let out = code.plain_crsc_convert(&client, &server).unwrap();
        let opened = add_vectors(&out.client_share, &out.server_share, 97).unwrap();
        assert_eq!(opened, msg);
    }
}
