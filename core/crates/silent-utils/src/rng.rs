//! CSPRNG implementation using ChaCha20.
//!
//! Provides a secure random number generator compliant with `rand::Rng`.

use rand::{RngCore, SeedableRng};
pub use rand_chacha::ChaCha20Rng;

use crate::blake2xb::{blake2xb as blake2xb_pure, blake2xb_bytes};

/// Direct wrapper for blake2xb for benchmarking.
pub fn blake2xb_fill(out: &mut [u8], seed: &[u8; 64], counter: u64) {
    if blake2xb_bytes(out, counter, seed).is_err() {
        panic!("blake2xb failed");
    }
}

/// The standard secure RNG type for SILENT (ChaCha20).
pub type SecureRng = ChaCha20Rng;

/// SEAL-aligned PRNG (Blake2xb with 4096-byte buffer).
pub struct Blake2xbRng {
    seed: [u64; 8],
    counter: u64,
    buffer: [u8; 4096],
    index: usize,
}

impl Blake2xbRng {
    pub fn from_seed_u64(seed: [u64; 8]) -> Self {
        let rng = Self {
            seed,
            counter: 0,
            buffer: [0u8; 4096],
            index: 4096,
        };
        rng
    }

    pub fn from_seed_bytes(seed: [u8; 64]) -> Self {
        let mut seed_u64 = [0u64; 8];
        for i in 0..8 {
            let start = i * 8;
            seed_u64[i] = u64::from_le_bytes(seed[start..start + 8].try_into().unwrap());
        }
        Self::from_seed_u64(seed_u64)
    }

    #[inline]
    fn seed_bytes_copy(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        for (i, limb) in self.seed.iter().enumerate() {
            out[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        out
    }

    #[inline]
    fn refill(&mut self) {
        let seed = self.seed_bytes_copy();
        if blake2xb_bytes(&mut self.buffer, self.counter, &seed).is_err() {
            panic!("blake2xb failed");
        }
        self.counter = self.counter.wrapping_add(1);
        self.index = 0;
    }

    pub fn fill_bytes_direct(&mut self, dest: &mut [u8]) {
        let mut offset = 0;
        while offset < dest.len() {
            let remaining = dest.len() - offset;
            if remaining >= self.buffer.len() {
                let seed = self.seed_bytes_copy();
                let out_slice = &mut dest[offset..offset + self.buffer.len()];
                if blake2xb_pure(out_slice, &self.counter.to_le_bytes(), &seed).is_err() {
                    panic!("blake2xb failed");
                }
                self.counter = self.counter.wrapping_add(1);
                offset += self.buffer.len();
            } else {
                self.fill_bytes(&mut dest[offset..]);
                break;
            }
        }
    }
}

impl RngCore for Blake2xbRng {
    fn next_u32(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.fill_bytes(&mut buf);
        u32::from_le_bytes(buf)
    }

    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.fill_bytes(&mut buf);
        u64::from_le_bytes(buf)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut offset = 0;
        while offset < dest.len() {
            if self.index == self.buffer.len() {
                self.refill();
            }
            let remaining = self.buffer.len() - self.index;
            let to_copy = core::cmp::min(remaining, dest.len() - offset);
            dest[offset..offset + to_copy]
                .copy_from_slice(&self.buffer[self.index..self.index + to_copy]);
            self.index += to_copy;
            offset += to_copy;
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

/// Create a new secure RNG seeded from the system entropy.
pub fn create_rng_from_entropy() -> SecureRng {
    SecureRng::from_entropy()
}

/// Create a new secure RNG from a specific 32-byte seed.
pub fn create_rng_from_seed(seed: [u8; 32]) -> SecureRng {
    SecureRng::from_seed(seed)
}

/// Utility to fill a buffer with random bytes.
pub fn fill_random_bytes(dest: &mut [u8]) {
    let mut rng = create_rng_from_entropy();
    rng.fill_bytes(dest);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn rng_is_deterministic_with_seed() {
        let seed = [42u8; 32];
        let mut rng1 = create_rng_from_seed(seed);
        let mut rng2 = create_rng_from_seed(seed);

        let v1: u64 = rng1.r#gen();
        let v2: u64 = rng2.r#gen();

        assert_eq!(v1, v2);
    }

    #[test]
    fn rng_generates_randomness() {
        let mut rng = create_rng_from_entropy();
        let v1: u64 = rng.r#gen();
        let v2: u64 = rng.r#gen();
        assert_ne!(v1, v2); // Extremely unlikely to collide
    }
}
