//! Sampling helpers for TFHE.
//!
//! Reuses [`silent_utils::rng::SecureRng`] for cryptographic randomness and
//! converts the high-level [`NoiseDistribution`] variants into concrete
//! integer samples on the native power-of-two ciphertext modulus.
//!
//! Intentionally narrow: only the building blocks the TFHE scheme code needs
//! (binary key, uniform mask, Gaussian noise, T-uniform noise).

use rand::RngCore;
use silent_params::NoiseDistribution;
use silent_utils::rng::SecureRng;

/// Sample a `f64` uniformly distributed in `[0, 1)` from a CSPRNG.
///
/// Drawn manually rather than via `Rng::gen::<f64>()` to side-step Rust 2024's
/// `gen` reserved keyword and to avoid widening this file's dependency
/// surface.
#[inline]
fn sample_unit_interval(rng: &mut SecureRng) -> f64 {
    // Match `rand::distributions::Standard for f64`: take 53 bits of entropy
    // and map them to [0, 1) by dividing by 2^53.
    let bits = rng.next_u64() >> 11;
    (bits as f64) * f64::from_bits(0x3CA0_0000_0000_0000) // 2^-53
}

/// Sample a uniform binary value (0 or 1).
#[inline]
pub fn sample_binary(rng: &mut SecureRng) -> u64 {
    (rng.next_u32() & 1) as u64
}

/// Fill a slice with i.i.d. uniform binary values.  Used for the LWE / GLWE
/// secret keys (the default `EncryptionKeyChoice::Big` setting in tfhe-rs is
/// a binary GLWE key).
pub fn fill_binary(rng: &mut SecureRng, dst: &mut [u64]) {
    let mut buf = [0u8; 4];
    let mut bit_index = 32u8;
    let mut cache = 0u32;
    for slot in dst.iter_mut() {
        if bit_index >= 32 {
            rng.fill_bytes(&mut buf);
            cache = u32::from_le_bytes(buf);
            bit_index = 0;
        }
        *slot = ((cache >> bit_index) & 1) as u64;
        bit_index += 1;
    }
}

/// Fill a slice with uniformly random `u64` values, then mask down to the
/// active modulus.  Used for the `a` part of LWE / GLWE encryptions.
pub fn fill_uniform_mod(rng: &mut SecureRng, log_modulus: u8, dst: &mut [u64]) {
    debug_assert!(log_modulus == 32 || log_modulus == 64);
    for slot in dst.iter_mut() {
        *slot = rng.next_u64();
    }
    if log_modulus < 64 {
        let mask = (1u64 << log_modulus) - 1;
        for slot in dst.iter_mut() {
            *slot &= mask;
        }
    }
}

/// Sample an integer noise value matching the requested distribution and
/// scale it into the native power-of-two ciphertext modulus.
///
/// For `Gaussian { stddev }`: `stddev` is interpreted relative to `q = 2^k`,
/// matching tfhe-rs convention; we sample a real Gaussian via Box–Muller and
/// round.
///
/// For `TUniform { bound_log2 }`: the result is uniform in
/// `(-2^bound_log2, 2^bound_log2]` (closed at the high end so the type lines
/// up with tfhe-rs; the discretisation difference is negligible for a CSPRNG).
pub fn sample_noise(rng: &mut SecureRng, dist: NoiseDistribution, log_modulus: u8) -> u64 {
    match dist {
        NoiseDistribution::Gaussian { stddev } => sample_gaussian_int(rng, stddev, log_modulus),
        NoiseDistribution::TUniform { bound_log2 } => sample_tuniform(rng, bound_log2, log_modulus),
    }
}

/// Fill a slice with i.i.d. samples from `dist`, scaled into the native
/// modulus.
pub fn fill_noise(rng: &mut SecureRng, dist: NoiseDistribution, log_modulus: u8, dst: &mut [u64]) {
    for slot in dst.iter_mut() {
        *slot = sample_noise(rng, dist, log_modulus);
    }
}

fn sample_gaussian_int(rng: &mut SecureRng, stddev_unit: f64, log_modulus: u8) -> u64 {
    // Convert the unit-interval stddev (i.e. relative to q) into integer units.
    // We can safely cast 2^k as f64 because k <= 64 and f64 has 53-bit mantissa
    // (the conversion loses precision but stddev is only an approximation).
    let q_as_f64 = if log_modulus == 64 {
        2f64.powi(64)
    } else {
        (1u64 << log_modulus) as f64
    };
    let stddev = stddev_unit * q_as_f64;
    let sample = box_muller(rng) * stddev;
    discretise_signed(sample, log_modulus)
}

fn sample_tuniform(rng: &mut SecureRng, bound_log2: u8, log_modulus: u8) -> u64 {
    debug_assert!(bound_log2 < log_modulus);
    let bits = u32::from(bound_log2) + 1;
    let bound = 1u64 << bound_log2;
    // Draw a uniform integer in (-bound, bound].  The high bit is the sign
    // and the next `bound_log2` bits are the magnitude minus the negative
    // pole contribution; matches tfhe-rs's t-uniform sampling convention.
    let raw: u64 = rng.next_u64() & ((1u64 << bits) - 1);
    // Map raw ∈ [0, 2^(bits)) to result ∈ (-bound, bound]:
    //   raw ∈ [0, bound)    -> raw + 1   (positive samples 1..=bound)
    //   raw ∈ [bound, 2bound) -> raw - 2*bound + 1  (negative samples)
    let signed = if raw < bound {
        raw as i128 + 1
    } else {
        raw as i128 - (bound as i128) * 2 + 1
    };
    if signed >= 0 {
        signed as u64
    } else if log_modulus == 64 {
        signed as u64 // wrapping conversion gives the right two's-complement repr
    } else {
        let mask = (1u64 << log_modulus) - 1;
        ((signed) as u64) & mask
    }
}

/// Box–Muller standard normal sampler. Produces independent samples one at a
/// time, caching the second to avoid recomputation.
fn box_muller(rng: &mut SecureRng) -> f64 {
    // We deliberately do not cache here: TFHE samples noise in tight loops and
    // we'd rather pay the extra log/sin than maintain thread-local state.
    // u1 must be strictly positive to avoid log(0).
    let mut u1 = sample_unit_interval(rng);
    while u1 <= f64::EPSILON {
        u1 = sample_unit_interval(rng);
    }
    let u2 = sample_unit_interval(rng);
    let r = (-2.0f64 * u1.ln()).sqrt();
    let theta = 2.0f64 * std::f64::consts::PI * u2;
    r * theta.cos()
}

fn discretise_signed(value: f64, log_modulus: u8) -> u64 {
    let rounded = value.round();
    if log_modulus == 64 {
        rounded as i64 as u64
    } else {
        let mask = (1u64 << log_modulus) - 1;
        (rounded as i64 as u64) & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> SecureRng {
        SecureRng::from_seed([1u8; 32])
    }

    #[test]
    fn binary_keys_are_zero_or_one() {
        let mut r = rng();
        let mut data = vec![0u64; 4096];
        fill_binary(&mut r, &mut data);
        for v in &data {
            assert!(*v == 0 || *v == 1);
        }
    }

    #[test]
    fn uniform_mask_lives_in_modulus() {
        let mut r = rng();
        let mut data = vec![0u64; 1024];
        fill_uniform_mod(&mut r, 32, &mut data);
        let mask = (1u64 << 32) - 1;
        for v in &data {
            assert!(*v <= mask);
        }
    }

    #[test]
    fn tuniform_within_bounds() {
        let mut r = rng();
        let bound_log2 = 5;
        let bound = 1i64 << bound_log2;
        for _ in 0..2048 {
            let s = sample_tuniform(&mut r, bound_log2, 64);
            // Interpret as signed two's complement.
            let signed = s as i64;
            assert!(
                signed > -bound && signed <= bound,
                "out of range: {}",
                signed
            );
        }
    }

    #[test]
    fn gaussian_centred_around_zero() {
        let mut r = rng();
        let stddev = 1e-8;
        let n = 4096;
        let mut acc: i128 = 0;
        for _ in 0..n {
            let s = sample_gaussian_int(&mut r, stddev, 64) as i64;
            acc += s as i128;
        }
        // mean / sample_size should be small in magnitude relative to stddev * q.
        let mean = (acc as f64) / (n as f64);
        let scaled_stddev = stddev * 2f64.powi(64);
        // Allow up to 5 sigma / sqrt(N).
        let bound = 5.0 * scaled_stddev / (n as f64).sqrt();
        assert!(mean.abs() < bound.max(1.0), "mean = {}", mean);
    }
}
