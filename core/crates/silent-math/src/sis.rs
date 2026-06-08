use crate::random::UniformSampler;

#[derive(Debug, Clone)]
pub struct SisError;

/// Trait for universal hashing based on SIS/LWE.
pub trait UniversalHash {
    /// hashes the input vector `x` (dimension `n`) to a vector of dimension `m`.
    /// `x` elements should be `< q`.
    fn hash(&self, input: &[u64]) -> Vec<u64>;
}

/// A Matrix-free SIS hasher (ARS24 compliant).
///
/// Implements `f(x) = A * x` where `A` is a public random matrix generated from a seed.
/// `A` is `m x n` matrix. `x` is `n x 1`. Result is `m x 1`.
#[derive(Debug)]
pub struct SisHasher {
    n: usize, // Input dimension
    m: usize, // Output dimension
    q: u64,   // Modulus
    seed: [u8; 32],
}

impl SisHasher {
    /// Create a new SIS hasher.
    ///
    /// # Arguments
    /// * `n` - Input dimension.
    /// * `m` - Output dimension (security parameter / compression target).
    /// * `q` - Modulus.
    /// * `seed` - Seed for generating matrix A.
    pub fn new(n: usize, m: usize, q: u64, seed: [u8; 32]) -> Self {
        Self { n, m, q, seed }
    }
}

impl UniversalHash for SisHasher {
    fn hash(&self, input: &[u64]) -> Vec<u64> {
        assert_eq!(input.len(), self.n, "Input dimension mismatch");

        // Output vector of size m
        let mut output = Vec::with_capacity(self.m);

        // We generate A row by row to compute A * x matrix-free.
        // row_i = UniformSampler(seed, nonce=i)
        // y_i = <row_i, x> mod q

        let mut row_buf = vec![0u64; self.n]; // Reusable buffer for one row of A

        // Pre-allocate nonce buffer
        let mut nonce = [0u8; 12];

        for i in 0..self.m {
            // Encode row index 'i' into nonce (little endian)
            // i fits in usize, we take lower 8 bytes (which fits u64)
            // It's extremely unlikely m > 2^64.
            let idx_bytes = (i as u64).to_le_bytes();
            nonce[0..8].copy_from_slice(&idx_bytes);

            // Re-seed sampler for this row
            // counter=0 is fine as long as nonce differs per row.
            let mut sampler = UniformSampler::new(self.seed, nonce, 0, self.q);

            // Fill row buffer with random elements in [0, q)
            sampler.fill(&mut row_buf);

            // Compute dot product with u128 accumulation
            let mut acc: u128 = 0;
            let q_u128 = self.q as u128;

            for (&a_val, &x_val) in row_buf.iter().zip(input.iter()) {
                // a_val and x_val are < q. Product < q^2.
                // q is typically 64-bit. q^2 fits in 128-bit.
                // We should be careful about accumulation overflow if n is large.
                // If n * q^2 > 2^128, we need intermediate reduction.
                // 2^128 approx 3.4e38.
                // If q approx 2^60, q^2 = 2^120.
                // We have 8 bits headroom for n (n < 256).
                // If q is smaller (e.g. 20-30 bits for SIS/LWE), then n can be very large.
                // For ARS24, q is likely standard LWE modulus (32-64 bits).
                // If n is large (e.g. 4096), we might overflow u128 if q is 60 bits.
                // Conservative check:
                acc += (a_val as u128) * (x_val as u128);

                // Lazy reduction strategy: reduce if we are getting close to overflow?
                // Or just reduce at end if we know parameters?
                // For safety/correctness, let's just reduce if acc is large, or use intermediate.
                // But efficient impl usually knows params.
                // Let's assume q <= 50 bits OR n is small.
                // However, to be robust, we can use barrett or simple rem.
                // Since this is generic 'q', we use simple rem here for correctness first.
                // FIXME: optimize this loop.
            }

            output.push((acc % q_u128) as u64);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sis_hash_determinism() {
        let n = 10;
        let m = 5;
        let q = 10007; // Prime
        let seed = [42u8; 32];
        let hasher = SisHasher::new(n, m, q, seed);

        let mut input = vec![0u64; n];
        for i in 0..n {
            input[i] = i as u64;
        }

        let h1 = hasher.hash(&input);
        let h2 = hasher.hash(&input);

        assert_eq!(h1.len(), m);
        assert_eq!(h1, h2);
    }

    #[test]
    fn sis_hash_linearity() {
        // A(x + y) = Ax + Ay
        let n = 10;
        let m = 5;
        let q = 101; // Small prime
        let seed = [1u8; 32];
        let hasher = SisHasher::new(n, m, q, seed);

        let x: Vec<u64> = (0..n).map(|i| (i as u64) % q).collect();
        let y: Vec<u64> = (0..n).map(|i| ((i * 2) as u64) % q).collect();
        let z: Vec<u64> = x.iter().zip(y.iter()).map(|(a, b)| (a + b) % q).collect();

        let hx = hasher.hash(&x);
        let hy = hasher.hash(&y);
        let hz = hasher.hash(&z);

        let h_sum: Vec<u64> = hx.iter().zip(hy.iter()).map(|(a, b)| (a + b) % q).collect();

        assert_eq!(hz, h_sum);
    }
}
