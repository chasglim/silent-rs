//! CSPRNG and sampling utilities.

pub struct ChaCha20Rng {
    state: [u32; 16],
    buffer: [u8; 64],
    index: usize,
}

impl ChaCha20Rng {
    pub fn new(key: [u8; 32], nonce: [u8; 12], counter: u32) -> Self {
        let mut state = [0u32; 16];
        state[0] = 0x6170_7865;
        state[1] = 0x3320_646e;
        state[2] = 0x7962_2d32;
        state[3] = 0x6b20_6574;

        for i in 0..8 {
            state[4 + i] =
                u32::from_le_bytes([key[i * 4], key[i * 4 + 1], key[i * 4 + 2], key[i * 4 + 3]]);
        }

        state[12] = counter;
        state[13] = u32::from_le_bytes([nonce[0], nonce[1], nonce[2], nonce[3]]);
        state[14] = u32::from_le_bytes([nonce[4], nonce[5], nonce[6], nonce[7]]);
        state[15] = u32::from_le_bytes([nonce[8], nonce[9], nonce[10], nonce[11]]);

        Self {
            state,
            buffer: [0u8; 64],
            index: 64,
        }
    }

    pub fn fill_bytes(&mut self, out: &mut [u8]) {
        let mut offset = 0;
        while offset < out.len() {
            if self.index >= 64 {
                self.refill();
            }
            let take = (64 - self.index).min(out.len() - offset);
            out[offset..offset + take].copy_from_slice(&self.buffer[self.index..self.index + take]);
            self.index += take;
            offset += take;
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.fill_bytes(&mut buf);
        u64::from_le_bytes(buf)
    }

    pub fn next_f64(&mut self) -> f64 {
        const SCALE_INV: f64 = 1.0 / ((1u64 << 53) as f64);
        let value = self.next_u64() >> 11;
        (value as f64) * SCALE_INV
    }

    fn refill(&mut self) {
        let block = chacha20_block(&self.state);
        self.state[12] = self.state[12].wrapping_add(1);
        for (i, word) in block.iter().enumerate() {
            let bytes = word.to_le_bytes();
            let start = i * 4;
            self.buffer[start..start + 4].copy_from_slice(&bytes);
        }
        self.index = 0;
    }
}

pub struct UniformSampler {
    rng: ChaCha20Rng,
    modulus: u64,
    bound: u64,
}

impl UniformSampler {
    pub fn new(key: [u8; 32], nonce: [u8; 12], counter: u32, modulus: u64) -> Self {
        assert!(modulus > 0);
        let bound = u64::MAX - (u64::MAX % modulus);
        Self {
            rng: ChaCha20Rng::new(key, nonce, counter),
            modulus,
            bound,
        }
    }

    pub fn sample(&mut self) -> u64 {
        loop {
            let value = self.rng.next_u64();
            if value < self.bound {
                return value % self.modulus;
            }
        }
    }

    pub fn fill(&mut self, out: &mut [u64]) {
        for value in out.iter_mut() {
            *value = self.sample();
        }
    }

    pub fn fill_parallel(
        key: [u8; 32],
        nonce: [u8; 12],
        counter: u32,
        modulus: u64,
        out: &mut [u64],
    ) {
        fill_parallel_inner(out, key, nonce, counter, |seed, chunk| {
            let mut sampler = UniformSampler::new(seed.key, seed.nonce, seed.counter, modulus);
            sampler.fill(chunk);
        });
    }
}

pub struct SeedExpander {
    rng: ChaCha20Rng,
}

impl SeedExpander {
    pub fn new(key: [u8; 32], nonce: [u8; 12], counter: u32) -> Self {
        Self {
            rng: ChaCha20Rng::new(key, nonce, counter),
        }
    }

    pub fn fill_bytes(&mut self, out: &mut [u8]) {
        self.rng.fill_bytes(out);
    }

    pub fn next_seed(&mut self) -> Seed {
        let mut buf = [0u8; 48];
        self.fill_bytes(&mut buf);
        let mut key = [0u8; 32];
        let mut nonce = [0u8; 12];
        key.copy_from_slice(&buf[..32]);
        nonce.copy_from_slice(&buf[32..44]);
        let counter = u32::from_le_bytes([buf[44], buf[45], buf[46], buf[47]]);
        Seed {
            key,
            nonce,
            counter,
        }
    }
}

pub struct Seed {
    key: [u8; 32],
    nonce: [u8; 12],
    counter: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct SeedParams {
    pub key: [u8; 32],
    pub nonce: [u8; 12],
    pub counter: u32,
}

pub struct CenteredBinomialSampler {
    rng: ChaCha20Rng,
    eta: usize,
}

impl CenteredBinomialSampler {
    pub fn new(key: [u8; 32], nonce: [u8; 12], counter: u32, eta: usize) -> Self {
        assert!(eta > 0);
        Self {
            rng: ChaCha20Rng::new(key, nonce, counter),
            eta,
        }
    }

    pub fn eta(&self) -> usize {
        self.eta
    }

    pub fn sample(&mut self) -> i64 {
        let a = self.sample_weight(self.eta);
        let b = self.sample_weight(self.eta);
        a as i64 - b as i64
    }

    pub fn fill(&mut self, out: &mut [i64]) {
        for value in out.iter_mut() {
            *value = self.sample();
        }
    }

    pub fn fill_parallel(params: SeedParams, eta: usize, out: &mut [i64]) {
        fill_parallel_inner(
            out,
            params.key,
            params.nonce,
            params.counter,
            |seed, chunk| {
                let mut sampler =
                    CenteredBinomialSampler::new(seed.key, seed.nonce, seed.counter, eta);
                sampler.fill(chunk);
            },
        );
    }

    fn sample_weight(&mut self, bits: usize) -> u32 {
        let mut remaining = bits;
        let mut weight = 0u32;
        while remaining > 0 {
            let take = remaining.min(64);
            let mask = if take == 64 {
                u64::MAX
            } else {
                (1u64 << take) - 1
            };
            let value = self.rng.next_u64() & mask;
            weight += value.count_ones();
            remaining -= take;
        }
        weight
    }
}

pub struct DiscreteGaussianSampler {
    rng: ChaCha20Rng,
    std_dev: f64,
    a: f64,
    cdf: Vec<f64>,
    rejection_ratios: Vec<u32>, // optimization for small std_dev
    max_abs: usize,
}

impl DiscreteGaussianSampler {
    pub fn new(key: [u8; 32], nonce: [u8; 12], counter: u32, std_dev: f64) -> Self {
        assert!(std_dev.is_finite());
        assert!(std_dev > 0.0);
        let (cdf, a, max_abs) = build_gaussian_cdf(std_dev);

        // Build rejection ratios if max_abs fits in reasonable range for 6-bit sampling
        let rejection_ratios = if max_abs <= 31 {
            build_rejection_ratios(std_dev, max_abs)
        } else {
            Vec::new()
        };

        Self {
            rng: ChaCha20Rng::new(key, nonce, counter),
            std_dev,
            a,
            cdf,
            rejection_ratios,
            max_abs,
        }
    }

    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }

    pub fn max_abs(&self) -> usize {
        self.max_abs
    }

    pub fn sample(&mut self) -> i64 {
        if !self.rejection_ratios.is_empty() {
            // Integer Rejection Sampling
            loop {
                let rand = self.rng.next_u64();
                // Use lower 6 bits for value x in [-32, 31]
                // x = (rand & 0x3F) - 32
                // 0..63 -> -32..31
                let r_idx = (rand & 0x3F) as i64;
                let x = r_idx - 32;
                let abs_x = x.abs() as usize;

                if abs_x <= self.max_abs {
                    // Check acceptance probability
                    // ratio = floor(exp(...) * 2^32)
                    // Use upper 32 bits of rand for comparison?
                    // Or bits 6..38?
                    // rand is u64. Let's use (rand >> 16) as u32 quality randomness
                    let check = (rand >> 16) as u32;
                    // SAFETY: abs_x checked against max_abs. rejection_ratios covers max_abs.
                    if check < unsafe { *self.rejection_ratios.get_unchecked(abs_x) } {
                        return x;
                    }
                }
                // Retry
            }
        }

        // Fallback to CDF
        let seed = self.rng.next_f64() - 0.5;
        let tmp = seed.abs() - self.a * 0.5;
        if tmp <= 0.0 {
            return 0;
        }
        let idx = find_in_cdf(&self.cdf, tmp) as i64;
        if seed >= 0.0 { idx } else { -idx }
    }

    pub fn sample_mod_q(&mut self, modulus: u64) -> u64 {
        let value = self.sample();
        // Since value is bounded by max_abs (very small, ~20) and modulus is large (~60 bits),
        // we can avoid expensive % operator.
        debug_assert!((value.abs() as u64) < modulus);

        if value < 0 {
            modulus.wrapping_sub((-value) as u64)
        } else {
            value as u64
        }
    }

    pub fn fill(&mut self, out: &mut [i64]) {
        for value in out.iter_mut() {
            *value = self.sample();
        }
    }

    pub fn fill_mod_q(&mut self, modulus: u64, out: &mut [u64]) {
        for value in out.iter_mut() {
            *value = self.sample_mod_q(modulus);
        }
    }

    pub fn fill_parallel(params: SeedParams, std_dev: f64, out: &mut [i64]) {
        fill_parallel_inner(
            out,
            params.key,
            params.nonce,
            params.counter,
            |seed, chunk| {
                let mut sampler =
                    DiscreteGaussianSampler::new(seed.key, seed.nonce, seed.counter, std_dev);
                sampler.fill(chunk);
            },
        );
    }
}

fn build_gaussian_cdf(std_dev: f64) -> (Vec<f64>, f64, usize) {
    let m = 12.006_105_535_382_85f64;
    let max_abs = (std_dev * m).ceil() as usize;
    let variance = 2.0 * std_dev * std_dev;

    let mut cdf = Vec::with_capacity(max_abs);
    let mut cusum = 0.0;
    for x in 1..=max_abs {
        let exponent = -((x * x) as f64) / variance;
        cusum += exponent.exp();
        cdf.push(cusum);
    }
    let a = 1.0 / (2.0 * cusum + 1.0);
    for value in cdf.iter_mut() {
        *value *= a;
    }
    (cdf, a, max_abs)
}

fn build_rejection_ratios(std_dev: f64, max_abs: usize) -> Vec<u32> {
    let variance = 0.5 / (std_dev * std_dev); // 1 / (2s^2)
    let mut ratios = Vec::with_capacity(max_abs + 1);
    // Prob(x) / Prob(0) = exp(-x^2 / 2s^2)
    for x in 0..=max_abs {
        let x_f = x as f64;
        let p = (-x_f * x_f * variance).exp(); // 0..1
        let scaled = (p * (u32::MAX as f64)) as u32;
        ratios.push(scaled);
    }
    ratios
}

fn find_in_cdf(cdf: &[f64], search: f64) -> usize {
    match cdf.binary_search_by(|probe| probe.partial_cmp(&search).unwrap()) {
        Ok(index) => index + 1,
        Err(index) => index + 1,
    }
}

fn fill_parallel_inner<T, F>(
    out: &mut [T],
    key: [u8; 32],
    nonce: [u8; 12],
    counter: u32,
    mut fill_chunk: F,
) where
    T: Send,
    F: FnMut(Seed, &mut [T]) + Copy + Send,
{
    const MIN_PARALLEL_CHUNK: usize = 65536;
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if workers <= 1 || out.len() < MIN_PARALLEL_CHUNK {
        fill_chunk(
            Seed {
                key,
                nonce,
                counter,
            },
            out,
        );
        return;
    }

    let chunk_size = (out.len() + workers - 1) / workers;
    let mut expander = SeedExpander::new(key, nonce, counter);
    let chunk_count = (out.len() + chunk_size - 1) / chunk_size;
    let mut seeds = Vec::with_capacity(chunk_count);
    for _ in 0..chunk_count {
        seeds.push(expander.next_seed());
    }

    std::thread::scope(|scope| {
        for (chunk, seed) in out.chunks_mut(chunk_size).zip(seeds.into_iter()) {
            scope.spawn(move || {
                fill_chunk(
                    Seed {
                        key: seed.key,
                        nonce: seed.nonce,
                        counter: seed.counter,
                    },
                    chunk,
                );
            });
        }
    });
}

fn chacha20_block(state: &[u32; 16]) -> [u32; 16] {
    let mut working = *state;
    for _ in 0..10 {
        quarter_round(&mut working, 0, 4, 8, 12);
        quarter_round(&mut working, 1, 5, 9, 13);
        quarter_round(&mut working, 2, 6, 10, 14);
        quarter_round(&mut working, 3, 7, 11, 15);

        quarter_round(&mut working, 0, 5, 10, 15);
        quarter_round(&mut working, 1, 6, 11, 12);
        quarter_round(&mut working, 2, 7, 8, 13);
        quarter_round(&mut working, 3, 4, 9, 14);
    }

    for i in 0..16 {
        working[i] = working[i].wrapping_add(state[i]);
    }

    working
}

fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(16);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(12);

    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(8);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(7);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chacha20_block_matches_rfc8439() {
        let mut key = [0u8; 32];
        for (i, byte) in key.iter_mut().enumerate() {
            *byte = i as u8;
        }
        let nonce = [
            0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00,
        ];
        let mut rng = ChaCha20Rng::new(key, nonce, 1);
        let mut out = [0u8; 64];
        rng.fill_bytes(&mut out);

        let expected = [
            0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15, 0x50, 0x0f, 0xdd, 0x1f, 0xa3, 0x20,
            0x71, 0xc4, 0xc7, 0xd1, 0xf4, 0xc7, 0x33, 0xc0, 0x68, 0x03, 0x04, 0x22, 0xaa, 0x9a,
            0xc3, 0xd4, 0x6c, 0x4e, 0xd2, 0x82, 0x64, 0x46, 0x07, 0x9f, 0xaa, 0x09, 0x14, 0xc2,
            0xd7, 0x05, 0xd9, 0x8b, 0x02, 0xa2, 0xb5, 0x12, 0x9c, 0xd1, 0xde, 0x16, 0x4e, 0xb9,
            0xcb, 0xd0, 0x83, 0xe8, 0xa2, 0x50, 0x3c, 0x4e,
        ];

        assert_eq!(out, expected);
    }

    #[test]
    fn uniform_sampler_is_deterministic() {
        let key = [7u8; 32];
        let nonce = [9u8; 12];
        let mut sampler = UniformSampler::new(key, nonce, 0, 97);
        let mut sampler2 = UniformSampler::new(key, nonce, 0, 97);
        let mut out = [0u64; 16];
        let mut out2 = [0u64; 16];
        sampler.fill(&mut out);
        sampler2.fill(&mut out2);
        assert_eq!(out, out2);
        assert!(out.iter().all(|&v| v < 97));
    }

    #[test]
    fn uniform_sampler_parallel_is_deterministic() {
        let key = [5u8; 32];
        let nonce = [6u8; 12];
        let mut out = vec![0u64; 4096];
        let mut out2 = vec![0u64; 4096];
        UniformSampler::fill_parallel(key, nonce, 0, 97, &mut out);
        UniformSampler::fill_parallel(key, nonce, 0, 97, &mut out2);
        assert_eq!(out, out2);
    }

    #[test]
    fn seed_expander_is_deterministic() {
        let key = [42u8; 32];
        let nonce = [11u8; 12];
        let mut expander = SeedExpander::new(key, nonce, 7);
        let mut expander2 = SeedExpander::new(key, nonce, 7);

        let seed1 = expander.next_seed();
        let seed2 = expander.next_seed();
        let seed1b = expander2.next_seed();
        let seed2b = expander2.next_seed();

        assert_eq!(seed1.key, seed1b.key);
        assert_eq!(seed1.nonce, seed1b.nonce);
        assert_eq!(seed1.counter, seed1b.counter);
        assert_eq!(seed2.key, seed2b.key);
        assert_eq!(seed2.nonce, seed2b.nonce);
        assert_eq!(seed2.counter, seed2b.counter);
    }

    #[test]
    fn centered_binomial_sampler_range() {
        let key = [1u8; 32];
        let nonce = [2u8; 12];
        let mut sampler = CenteredBinomialSampler::new(key, nonce, 0, 6);
        let mut out = [0i64; 256];
        sampler.fill(&mut out);
        for value in out.iter() {
            assert!(*value >= -6);
            assert!(*value <= 6);
        }
    }

    #[test]
    fn discrete_gaussian_sampler_bounds() {
        let key = [3u8; 32];
        let nonce = [4u8; 12];
        let mut sampler = DiscreteGaussianSampler::new(key, nonce, 0, 3.2);
        let mut out = [0i64; 512];
        sampler.fill(&mut out);
        let max_abs = sampler.max_abs as i64;
        assert!(out.iter().all(|&v| v.abs() <= max_abs));
    }
}
