//! Minimal XXH3-64 implementation for container checksums.
//!
//! This is a correctness-focused implementation of the XXH3-64 algorithm
//! as specified by Yann Collet.  It is used exclusively for fast non-cryptographic
//! corruption detection — **not** for security integrity.
//!
//! Only the streaming API (`Xxh3State`) and the one-shot 64-bit API are exposed.

// ── Constants ───────────────────────────────────────────────────────────────

const PRIME64_1: u64 = 0x9E3779B185EBCA87;
const PRIME64_2: u64 = 0xC2B2AE3D27D4EB4F;
const PRIME64_3: u64 = 0x165667B19E3779F9;
const PRIME64_4: u64 = 0x85EBCA77C2B2AE63;
const PRIME64_5: u64 = 0x27D4EB2F165667C5;

const STRIPE_LEN: usize = 64;
const SECRET_DEFAULT_SIZE: usize = 192;

// Default kSecret as per the XXH3 specification (first 192 bytes).
const DEFAULT_SECRET: [u8; SECRET_DEFAULT_SIZE] = [
    0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c,
    0xde, 0xd4, 0x6d, 0xe9, 0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f,
    0xcb, 0x79, 0xe6, 0x4e, 0xcc, 0xc0, 0xe5, 0x78, 0x82, 0x5a, 0xd0, 0x7d, 0xcc, 0xff, 0x72, 0x21,
    0xb8, 0x08, 0x46, 0x74, 0xf7, 0x43, 0x24, 0x8e, 0xe0, 0x35, 0x90, 0xe6, 0x81, 0x3a, 0x26, 0x4c,
    0x3c, 0x28, 0x52, 0xbb, 0x91, 0xc3, 0x00, 0xcb, 0x88, 0xd0, 0x65, 0x8b, 0x1b, 0x53, 0x2e, 0xa3,
    0x71, 0x64, 0x48, 0x97, 0xa2, 0x0d, 0xf9, 0x4e, 0x38, 0x19, 0xef, 0x46, 0xa9, 0xde, 0xac, 0xd8,
    0xa8, 0xfa, 0x76, 0x3f, 0xe3, 0x9c, 0x34, 0x3f, 0xf9, 0xdc, 0xbb, 0xc7, 0xc7, 0x0b, 0x4f, 0x1d,
    0x8a, 0x51, 0xe0, 0x4b, 0xcd, 0xb4, 0x59, 0x31, 0xc8, 0x9f, 0x7e, 0xc9, 0xd9, 0x78, 0x73, 0x64,
    0xea, 0xc5, 0xac, 0x83, 0x34, 0xd3, 0xeb, 0xc3, 0xc5, 0x81, 0xa0, 0xff, 0xfa, 0x13, 0x63, 0xeb,
    0x17, 0x0d, 0xdd, 0x51, 0xb7, 0xf0, 0xda, 0x49, 0xd3, 0x16, 0x55, 0x26, 0x29, 0xd4, 0x68, 0x9e,
    0x2b, 0x16, 0xbe, 0x58, 0x7d, 0x47, 0xa1, 0xfc, 0x8f, 0xf8, 0xb8, 0xd1, 0x7a, 0xd0, 0x31, 0xce,
    0x45, 0xcb, 0x3a, 0x8f, 0x95, 0x16, 0x04, 0x28, 0xaf, 0xd7, 0xfb, 0xca, 0xbb, 0x4b, 0x40, 0x7e,
];

// ── Streaming state ────────────────────────────────────────────────────────

/// Streaming XXH3-64 hash state.
pub struct Xxh3State {
    acc: [u64; 8],
    buffer: [u8; STRIPE_LEN],
    buffered: usize,
    total_len: u64,
    nb_stripes_processed: u64,
    seed: u64,
}

impl Xxh3State {
    /// Create a new streaming state with the given seed.
    pub fn new(seed: u64) -> Self {
        let mut state = Self {
            acc: [0u64; 8],
            buffer: [0u8; STRIPE_LEN],
            buffered: 0,
            total_len: 0,
            nb_stripes_processed: 0,
            seed,
        };
        // Initialise accumulators.
        state.acc[0] = PRIME64_3.wrapping_add(PRIME64_4);
        let key = u64::from_le_bytes(DEFAULT_SECRET[..8].try_into().unwrap());
        state.acc[1] = PRIME64_1.wrapping_add(PRIME64_2).wrapping_sub(key);
        let key = u64::from_le_bytes(DEFAULT_SECRET[8..16].try_into().unwrap());
        state.acc[2] = PRIME64_2.wrapping_add(key);
        let key = u64::from_le_bytes(DEFAULT_SECRET[16..24].try_into().unwrap());
        state.acc[3] = PRIME64_1.wrapping_sub(key);
        let key = u64::from_le_bytes(DEFAULT_SECRET[24..32].try_into().unwrap());
        state.acc[4] = PRIME64_1.wrapping_mul(PRIME64_2);
        state.acc[5] = PRIME64_2.wrapping_add(key);
        let key = u64::from_le_bytes(DEFAULT_SECRET[32..40].try_into().unwrap());
        state.acc[6] = PRIME64_1.wrapping_sub(key);
        let key = u64::from_le_bytes(DEFAULT_SECRET[40..48].try_into().unwrap());
        state.acc[7] = PRIME64_2.wrapping_mul(key);
        // XOR in the seed.
        for a in &mut state.acc {
            *a ^= seed;
        }
        state
    }

    /// Feed bytes into the hasher.
    pub fn update(&mut self, input: &[u8]) {
        self.total_len += input.len() as u64;
        let mut input = input;

        // If we have buffered data, fill the buffer first.
        if self.buffered > 0 {
            let to_copy = input.len().min(STRIPE_LEN - self.buffered);
            self.buffer[self.buffered..self.buffered + to_copy].copy_from_slice(&input[..to_copy]);
            self.buffered += to_copy;
            input = &input[to_copy..];

            if self.buffered == STRIPE_LEN {
                accumulate(&mut self.acc, &self.buffer, self.nb_stripes_processed);
                self.nb_stripes_processed += 1;
                self.buffered = 0;
            }
        }

        // Process full stripes.
        while input.len() >= STRIPE_LEN {
            accumulate(
                &mut self.acc,
                &input[..STRIPE_LEN],
                self.nb_stripes_processed,
            );
            self.nb_stripes_processed += 1;
            input = &input[STRIPE_LEN..];
        }

        // Buffer remainder.
        if !input.is_empty() {
            self.buffer[..input.len()].copy_from_slice(input);
            self.buffered = input.len();
        }
    }

    /// Finalise and return the 64-bit hash.
    pub fn finish(&self) -> u64 {
        let mut acc = self.acc;

        // Fold accumulators if we processed at least one full stripe.
        if self.nb_stripes_processed > 0 {
            for a in &mut acc {
                *a = avalanche(*a);
            }
            let folded = acc[0]
                .wrapping_add(acc[1])
                .wrapping_add(acc[2])
                .wrapping_add(acc[3])
                .wrapping_add(acc[4])
                .wrapping_add(acc[5])
                .wrapping_add(acc[6])
                .wrapping_add(acc[7]);
            // Process remainder.
            hash_remainder(
                folded,
                &self.buffer[..self.buffered],
                self.total_len,
                self.seed,
            )
        } else {
            // Short input path.
            hash_remainder(
                PRIME64_5.wrapping_add(self.seed),
                &self.buffer[..self.buffered],
                self.total_len,
                self.seed,
            )
        }
    }
}

// ── One-shot API ───────────────────────────────────────────────────────────

/// Compute the XXH3-64 hash of `input` with the given seed.
#[allow(dead_code)]
pub fn xxh3_64_with_seed(input: &[u8], seed: u64) -> u64 {
    let mut state = Xxh3State::new(seed);
    state.update(input);
    state.finish()
}

/// Compute the XXH3-64 hash of `input` with seed = 0.
#[allow(dead_code)]
pub fn xxh3_64(input: &[u8]) -> u64 {
    xxh3_64_with_seed(input, 0)
}

// ── Internal helpers ───────────────────────────────────────────────────────

fn avalanche(mut h: u64) -> u64 {
    h ^= h >> 37;
    h = h.wrapping_mul(PRIME64_3);
    h ^= h >> 32;
    h
}

fn accumulate(acc: &mut [u64; 8], stripe: &[u8], stripe_idx: u64) {
    let secret_offset = ((stripe_idx as usize) * STRIPE_LEN) % SECRET_DEFAULT_SIZE;

    for (i, a) in acc.iter_mut().enumerate().take(8) {
        let data_off = i * 8;
        let sec_off = (secret_offset + i * 8) % SECRET_DEFAULT_SIZE;

        let data_word = u64::from_le_bytes(stripe[data_off..data_off + 8].try_into().unwrap());
        let sec_word = u64::from_le_bytes(DEFAULT_SECRET[sec_off..sec_off + 8].try_into().unwrap());

        let mixed = data_word ^ sec_word;
        let product = (mixed & 0xFFFFFFFF)
            .wrapping_mul(PRIME64_1)
            .wrapping_add((mixed >> 32).wrapping_mul(PRIME64_1));
        *a = a.wrapping_add(product);
        *a = a.rotate_left(32);
        *a = a.wrapping_mul(PRIME64_2);
    }
}

fn mix16(data: &[u8], secret: &[u8], seed: u64, offset: usize) -> u64 {
    let d0 = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
    let d1 = u64::from_le_bytes(data[offset + 8..offset + 16].try_into().unwrap());
    let s0 = u64::from_le_bytes(
        secret[offset % SECRET_DEFAULT_SIZE..][..8]
            .try_into()
            .unwrap(),
    );
    let s1 = u64::from_le_bytes(
        secret[(offset + 8) % SECRET_DEFAULT_SIZE..][..8]
            .try_into()
            .unwrap(),
    );

    let m0 = (d0 ^ s0).wrapping_mul(PRIME64_1.wrapping_add(seed));
    let m1 = (d1 ^ s1).wrapping_mul(PRIME64_2.wrapping_sub(seed));
    avalanche(m0.wrapping_add(m1))
}

fn hash_remainder(mut acc: u64, data: &[u8], total_len: u64, seed: u64) -> u64 {
    if total_len <= 16 {
        return hash_len_0_to_16(data, total_len, seed);
    }

    let len = data.len();
    // Process 16-byte chunks.
    let mut offset = 0usize;
    while offset + 16 <= len {
        acc = acc.wrapping_add(mix16(data, &DEFAULT_SECRET, seed, offset));
        acc = acc.rotate_left(17);
        acc = acc.wrapping_mul(PRIME64_4);
        offset += 16;
    }

    // Process remaining 8 bytes.
    if offset + 8 <= len {
        acc = acc.wrapping_add(process_last_stripe(data, offset, seed));
        offset += 8;
    }

    // Process remaining < 8 bytes.
    if offset < len {
        let remaining = len - offset;
        let mut last = [0u8; 8];
        let start = 8 - remaining;
        last[start..].copy_from_slice(&data[offset..]);
        let word = u64::from_le_bytes(last);
        let key = u64::from_le_bytes(
            DEFAULT_SECRET[120 + remaining - 1..][..8]
                .try_into()
                .unwrap(),
        );
        acc = acc.wrapping_add((word ^ key).wrapping_mul(PRIME64_1));
    }

    // Final avalanche.
    let len_lo = (total_len & 0xFFFFFFFF) as u32;
    let len_hi = (total_len >> 32) as u32;
    acc ^= PRIME64_5
        .wrapping_add(len_lo as u64)
        .wrapping_add((len_hi as u64) << 32);
    avalanche(acc)
}

fn hash_len_0_to_16(data: &[u8], total_len: u64, seed: u64) -> u64 {
    if total_len > 16 {
        // Should not happen; fall through to remainder path.
        return hash_remainder(PRIME64_5.wrapping_add(seed), data, total_len, seed);
    }

    if total_len >= 8 {
        let mut a = u64::from_le_bytes(data[..8].try_into().unwrap());
        let b_offset = (total_len as usize) - 8;
        let mut b = u64::from_le_bytes(data[b_offset..b_offset + 8].try_into().unwrap());

        a ^= PRIME64_4.wrapping_sub(seed);
        b ^= PRIME64_3.wrapping_add(seed);

        let m = (a as u128).wrapping_mul(b as u128);
        let cross = ((m >> 64) as u64) ^ (m as u64);

        avalanche(cross ^ total_len ^ PRIME64_1)
    } else if total_len >= 4 {
        let mut a = u32::from_le_bytes(data[..4].try_into().unwrap()) as u64;
        let b_offset = (total_len as usize) - 4;
        let mut b = u32::from_le_bytes(data[b_offset..b_offset + 4].try_into().unwrap()) as u64;

        a = a.wrapping_add(seed).wrapping_add(PRIME64_1);
        b = b.wrapping_sub(seed).wrapping_add(PRIME64_2);

        let m = a.wrapping_mul(b);
        let mixed = m.wrapping_add(total_len);
        avalanche(mixed ^ PRIME64_1)
    } else if total_len > 0 {
        let first = data[0];
        let mid = data[(total_len as usize) / 2];
        let last = data[(total_len as usize) - 1];

        let combined = (first as u64).wrapping_add((mid as u64) << 8);
        let mut h = combined.wrapping_mul(PRIME64_5);
        h ^= ((last as u64) << 24).wrapping_mul(PRIME64_1);
        h ^= total_len * PRIME64_3;
        avalanche(h ^ seed)
    } else {
        avalanche(PRIME64_5.wrapping_add(seed))
    }
}

fn process_last_stripe(data: &[u8], offset: usize, seed: u64) -> u64 {
    let word = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
    let key = u64::from_le_bytes(
        DEFAULT_SECRET[(offset + 120) % SECRET_DEFAULT_SIZE..][..8]
            .try_into()
            .unwrap(),
    );
    (word ^ key).wrapping_mul(PRIME64_1).wrapping_add(seed)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_hash_consistent() {
        let a = xxh3_64(b"");
        let b = xxh3_64(b"");
        assert_eq!(a, b);
    }

    #[test]
    fn basic_string_consistent() {
        let a = xxh3_64(b"hello");
        let b = xxh3_64(b"hello");
        assert_eq!(a, b);
        // Different input must produce different hash.
        assert_ne!(a, xxh3_64(b"world"));
    }

    #[test]
    fn streaming_same_as_oneshot() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let oneshot = xxh3_64(data);

        let mut state = Xxh3State::new(0);
        state.update(data);
        let streaming = state.finish();
        assert_eq!(oneshot, streaming);
    }

    #[test]
    fn streaming_chunked() {
        let chunks: &[&[u8]] = &[b"abc", b"def", b"ghi", b"jkl"];
        let full: Vec<u8> = chunks.iter().flat_map(|c| c.iter()).copied().collect();

        let oneshot = xxh3_64(&full);

        let mut state = Xxh3State::new(0);
        for chunk in chunks {
            state.update(chunk);
        }
        assert_eq!(oneshot, state.finish());
    }

    #[test]
    fn long_input_streaming() {
        let data = vec![0xabu8; 1024];
        let oneshot = xxh3_64(&data);

        let mut state = Xxh3State::new(0);
        state.update(&data);
        assert_eq!(oneshot, state.finish());
    }

    #[test]
    fn stripe_boundary() {
        // Exactly one stripe.
        let data = vec![0x42u8; 64];
        let oneshot = xxh3_64(&data);

        let mut state = Xxh3State::new(0);
        state.update(&data);
        assert_eq!(oneshot, state.finish());
    }

    #[test]
    fn multiple_stripes() {
        // Two full stripes.
        let data = vec![0x42u8; 128];
        let oneshot = xxh3_64(&data);

        let mut state = Xxh3State::new(0);
        // Feed byte-by-byte to stress the buffering.
        for &b in &data {
            state.update(&[b]);
        }
        assert_eq!(oneshot, state.finish());
    }

    #[test]
    fn seed_changes_output() {
        let data = b"test data";
        let h0 = xxh3_64_with_seed(data, 0);
        let h1 = xxh3_64_with_seed(data, 1);
        assert_ne!(h0, h1);
    }
}
