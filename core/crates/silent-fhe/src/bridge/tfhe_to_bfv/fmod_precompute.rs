use crate::bridge::{BridgeError, BridgeParams};
use silent_math::{arith, numth};

pub const MAX_PERIODIC_CENTER_POINTS: usize = 128;

/// Precompute the center-only exact Fmod polynomial.
///
/// This interpolates only the TFHE encoding centers
/// `x_k = k * q_T / (2M)` for `k in 0..M`, where
/// `M = message_modulus * carry_modulus`. It is exact for center phases and
/// is not an interval decoder for noisy TFHE phases.
pub fn precompute_fmod_lagrange(params: &BridgeParams) -> Result<Vec<u64>, BridgeError> {
    let m = params.message_modulus();
    let t = params.bfv_params.plain_modulus();
    let log_q = params.tfhe_params.ciphertext_modulus_log().0 as u32;
    if log_q >= 64 {
        return Err(BridgeError::IncompatibleParameters(
            "center-only Fmod interpolation requires TFHE torus modulus to fit in u64; use CRT/lifting for q=2^64"
                .into(),
        ));
    }
    if !numth::is_prime(t) {
        return Err(BridgeError::IncompatibleParameters(format!(
            "center-only Fmod interpolation requires prime BFV plaintext modulus, got {t}"
        )));
    }

    let padded_m = params.padded_message_modulus();
    let q_t = 1u128 << log_q;
    let delta = (q_t / padded_m as u128) as u64;

    let mut x = Vec::with_capacity(m as usize);
    let mut y = Vec::with_capacity(m as usize);
    for k in 0..m {
        x.push(((k as u128 * delta as u128) % t as u128) as u64);
        y.push(k as u64 % t);
    }

    interpolate_lagrange(&x, &y, t)
}

/// Precompute a small periodic center decoder.
///
/// The interpolation domain is:
///
/// ```text
/// x_{k,m} = k*q_T + Δ_T*m  (mod t)
/// k = 0..=n_lwe, m = 0..M-1
/// ```
///
/// This removes the exact `k*q_T` offset introduced by lifted naive partial
/// decryption. It still assumes zero TFHE noise and exact encoding centers.
pub fn precompute_periodic_center_lagrange(params: &BridgeParams) -> Result<Vec<u64>, BridgeError> {
    ensure_periodic_center_preconditions(params)?;

    let m = params.message_modulus();
    let t = params.bfv_params.plain_modulus();
    let log_q = params.tfhe_params.ciphertext_modulus_log().0 as u32;
    let q_t = 1u64 << log_q;
    let delta = q_t / params.padded_message_modulus();
    let periods = params.tfhe_params.lwe_dimension().0 + 1;

    let mut x = Vec::with_capacity(periods * m as usize);
    let mut y = Vec::with_capacity(periods * m as usize);
    for k in 0..periods {
        for message in 0..m {
            let point = (k as u128 * q_t as u128 + message as u128 * delta as u128) % t as u128;
            x.push(point as u64);
            y.push(message % t);
        }
    }

    interpolate_lagrange(&x, &y, t)
}

pub fn ensure_periodic_center_preconditions(params: &BridgeParams) -> Result<(), BridgeError> {
    let log_q = params.tfhe_params.ciphertext_modulus_log().0 as u32;
    if log_q >= 64 {
        return Err(BridgeError::IncompatibleParameters(
            "periodic center Fmod requires TFHE torus modulus to fit in u64".into(),
        ));
    }
    if !numth::is_prime(params.bfv_params.plain_modulus()) {
        return Err(BridgeError::IncompatibleParameters(format!(
            "periodic center Fmod requires prime BFV plaintext modulus, got {}",
            params.bfv_params.plain_modulus()
        )));
    }

    let periods = params
        .tfhe_params
        .lwe_dimension()
        .0
        .checked_add(1)
        .ok_or_else(|| {
            BridgeError::IncompatibleParameters("TFHE LWE dimension overflows usize".into())
        })?;
    let point_count = periods
        .checked_mul(params.message_modulus() as usize)
        .ok_or_else(|| {
            BridgeError::IncompatibleParameters("periodic Fmod point count overflows usize".into())
        })?;
    if point_count > MAX_PERIODIC_CENTER_POINTS {
        return Err(BridgeError::IncompatibleParameters(format!(
            "periodic center Fmod has {point_count} interpolation points, limit is {MAX_PERIODIC_CENTER_POINTS}; use CRT/plaintext lifting with a reviewed scalable backend"
        )));
    }

    let q_t = 1u128 << log_q;
    let delta = q_t / params.padded_message_modulus() as u128;
    let max_point = periods.checked_sub(1).ok_or_else(|| {
        BridgeError::IncompatibleParameters("periodic Fmod period count underflows".into())
    })? as u128
        * q_t
        + (params.message_modulus() as u128 - 1) * delta;
    let required_plain = max_point.checked_add(1).ok_or_else(|| {
        BridgeError::IncompatibleParameters("periodic Fmod plaintext bound overflows".into())
    })?;
    let plain_modulus = params.bfv_params.plain_modulus() as u128;
    if plain_modulus < required_plain {
        return Err(BridgeError::IncompatibleParameters(format!(
            "periodic center Fmod requires BFV plaintext modulus at least {}, got {}",
            required_plain, plain_modulus
        )));
    }

    Ok(())
}

/// Simple Lagrange interpolation over Z_t.
/// Returns coefficients of polynomial a_0 + a_1*x + ... + a_{n-1}*x^{n-1}
fn interpolate_lagrange(x: &[u64], y: &[u64], t: u64) -> Result<Vec<u64>, BridgeError> {
    let n = x.len();
    let mut coeffs = vec![0u64; n];

    for i in 0..n {
        // Basis polynomial L_i(x) = product_{j != i} (x - x_j) / (x_i - x_j)
        let mut li = vec![0u64; n];
        li[0] = 1;

        let mut denominator = 1u64;
        let mut current_degree = 0;

        for j in 0..n {
            if i == j {
                continue;
            }

            // Multiply li by (x - x_j)
            let neg_xj = if x[j] == 0 { 0 } else { t - x[j] };
            let mut next_li = vec![0u64; n];
            for k in 0..=current_degree {
                // x term: li[k] * x -> next_li[k+1]
                next_li[k + 1] = arith::add_mod(next_li[k + 1], li[k], t);
                // const term: li[k] * (-x[j]) -> next_li[k]
                next_li[k] = arith::add_mod(next_li[k], arith::mul_mod_u64(li[k], neg_xj, t), t);
            }
            li = next_li;
            current_degree += 1;

            let diff = if x[i] >= x[j] {
                x[i] - x[j]
            } else {
                t - (x[j] - x[i])
            };
            denominator = arith::mul_mod_u64(denominator, diff, t);
        }

        let inv_den = numth::mod_inverse(denominator, t).ok_or_else(|| {
            BridgeError::IncompatibleParameters(format!(
                "Interpolation failed: denominator {} not invertible mod {}",
                denominator, t
            ))
        })?;

        let factor = arith::mul_mod_u64(y[i], inv_den, t);
        for k in 0..n {
            coeffs[k] = arith::add_mod(coeffs[k], arith::mul_mod_u64(li[k], factor, t), t);
        }
    }

    Ok(coeffs)
}
