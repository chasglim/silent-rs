// Pure Rust Blake2b/Blake2xb implementation
// This is used to keep PRNG fully self-contained

const BLAKE2B_BLOCKBYTES: usize = 128;
const BLAKE2B_OUTBYTES: usize = 64;
const BLAKE2B_KEYBYTES: usize = 64;

const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    feature = "blake2b-neon"
))]
const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

#[inline(always)]
fn load64(input: &[u8]) -> u64 {
    u64::from_le_bytes(input[0..8].try_into().unwrap())
}

#[inline(always)]
unsafe fn load64_unchecked(ptr: *const u8) -> u64 {
    unsafe { (ptr as *const u64).read_unaligned().to_le() }
}

#[inline(always)]
fn store64(out: &mut [u8], value: u64) {
    unsafe {
        (out.as_mut_ptr() as *mut u64).write_unaligned(value.to_le());
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Blake2bParam {
    digest_length: u8,
    key_length: u8,
    fanout: u8,
    depth: u8,
    leaf_length: u32,
    node_offset: u32,
    xof_length: u32,
    node_depth: u8,
    inner_length: u8,
    reserved: [u8; 14],
    salt: [u8; 16],
    personal: [u8; 16],
}

impl Blake2bParam {
    fn as_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self as *const Blake2bParam as *const u8,
                out.as_mut_ptr(),
                out.len(),
            );
        }
        out
    }
}

#[derive(Clone)]
struct Blake2bState {
    h: [u64; 8],
    t: [u64; 2],
    f: [u64; 2],
    buf: [u8; BLAKE2B_BLOCKBYTES],
    buflen: usize,
    outlen: usize,
    last_node: bool,
}

impl Blake2bState {
    fn new() -> Self {
        Self {
            h: IV,
            t: [0u64; 2],
            f: [0u64; 2],
            buf: [0u8; BLAKE2B_BLOCKBYTES],
            buflen: 0,
            outlen: 0,
            last_node: false,
        }
    }
}

#[inline(always)]
fn blake2b_set_lastnode(state: &mut Blake2bState) {
    state.f[1] = u64::MAX;
}

#[inline(always)]
fn blake2b_is_lastblock(state: &Blake2bState) -> bool {
    state.f[0] != 0
}

#[inline(always)]
fn blake2b_set_lastblock(state: &mut Blake2bState) {
    if state.last_node {
        blake2b_set_lastnode(state);
    }
    state.f[0] = u64::MAX;
}

#[inline(always)]
fn blake2b_increment_counter(state: &mut Blake2bState, inc: u64) {
    state.t[0] = state.t[0].wrapping_add(inc);
    if state.t[0] < inc {
        state.t[1] = state.t[1].wrapping_add(1);
    }
}

fn blake2b_init_param(state: &mut Blake2bState, param: &Blake2bParam) {
    *state = Blake2bState::new();
    let bytes = param.as_bytes();
    for i in 0..8 {
        state.h[i] ^= load64(&bytes[i * 8..]);
    }
    state.outlen = param.digest_length as usize;
}

#[inline(always)]
fn blake2b_compress_scalar(state: &mut Blake2bState, block: &[u8]) {
    debug_assert_eq!(block.len(), BLAKE2B_BLOCKBYTES);
    let base = block.as_ptr();
    let m0 = unsafe { load64_unchecked(base.add(0)) };
    let m1 = unsafe { load64_unchecked(base.add(8)) };
    let m2 = unsafe { load64_unchecked(base.add(16)) };
    let m3 = unsafe { load64_unchecked(base.add(24)) };
    let m4 = unsafe { load64_unchecked(base.add(32)) };
    let m5 = unsafe { load64_unchecked(base.add(40)) };
    let m6 = unsafe { load64_unchecked(base.add(48)) };
    let m7 = unsafe { load64_unchecked(base.add(56)) };
    let m8 = unsafe { load64_unchecked(base.add(64)) };
    let m9 = unsafe { load64_unchecked(base.add(72)) };
    let m10 = unsafe { load64_unchecked(base.add(80)) };
    let m11 = unsafe { load64_unchecked(base.add(88)) };
    let m12 = unsafe { load64_unchecked(base.add(96)) };
    let m13 = unsafe { load64_unchecked(base.add(104)) };
    let m14 = unsafe { load64_unchecked(base.add(112)) };
    let m15 = unsafe { load64_unchecked(base.add(120)) };

    let mut v0 = state.h[0];
    let mut v1 = state.h[1];
    let mut v2 = state.h[2];
    let mut v3 = state.h[3];
    let mut v4 = state.h[4];
    let mut v5 = state.h[5];
    let mut v6 = state.h[6];
    let mut v7 = state.h[7];
    let mut v8 = IV[0];
    let mut v9 = IV[1];
    let mut v10 = IV[2];
    let mut v11 = IV[3];
    let mut v12 = IV[4] ^ state.t[0];
    let mut v13 = IV[5] ^ state.t[1];
    let mut v14 = IV[6] ^ state.f[0];
    let mut v15 = IV[7] ^ state.f[1];

    macro_rules! g_local {
        ($a:ident, $b:ident, $c:ident, $d:ident, $x:expr, $y:expr) => {{
            $a = $a.wrapping_add($b).wrapping_add($x);
            $d = ($d ^ $a).rotate_right(32);
            $c = $c.wrapping_add($d);
            $b = ($b ^ $c).rotate_right(24);
            $a = $a.wrapping_add($b).wrapping_add($y);
            $d = ($d ^ $a).rotate_right(16);
            $c = $c.wrapping_add($d);
            $b = ($b ^ $c).rotate_right(63);
        }};
    }

    macro_rules! round {
        ($s0:expr, $s1:expr, $s2:expr, $s3:expr, $s4:expr, $s5:expr, $s6:expr, $s7:expr,
         $s8:expr, $s9:expr, $s10:expr, $s11:expr, $s12:expr, $s13:expr, $s14:expr, $s15:expr) => {{
            g_local!(v0, v4, v8, v12, $s0, $s1);
            g_local!(v1, v5, v9, v13, $s2, $s3);
            g_local!(v2, v6, v10, v14, $s4, $s5);
            g_local!(v3, v7, v11, v15, $s6, $s7);
            g_local!(v0, v5, v10, v15, $s8, $s9);
            g_local!(v1, v6, v11, v12, $s10, $s11);
            g_local!(v2, v7, v8, v13, $s12, $s13);
            g_local!(v3, v4, v9, v14, $s14, $s15);
        }};
    }

    round!(
        m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
    );
    round!(
        m14, m10, m4, m8, m9, m15, m13, m6, m1, m12, m0, m2, m11, m7, m5, m3
    );
    round!(
        m11, m8, m12, m0, m5, m2, m15, m13, m10, m14, m3, m6, m7, m1, m9, m4
    );
    round!(
        m7, m9, m3, m1, m13, m12, m11, m14, m2, m6, m5, m10, m4, m0, m15, m8
    );
    round!(
        m9, m0, m5, m7, m2, m4, m10, m15, m14, m1, m11, m12, m6, m8, m3, m13
    );
    round!(
        m2, m12, m6, m10, m0, m11, m8, m3, m4, m13, m7, m5, m15, m14, m1, m9
    );
    round!(
        m12, m5, m1, m15, m14, m13, m4, m10, m0, m7, m6, m3, m9, m2, m8, m11
    );
    round!(
        m13, m11, m7, m14, m12, m1, m3, m9, m5, m0, m15, m4, m8, m6, m2, m10
    );
    round!(
        m6, m15, m14, m9, m11, m3, m0, m8, m12, m2, m13, m7, m1, m4, m10, m5
    );
    round!(
        m10, m2, m8, m4, m7, m6, m1, m5, m15, m11, m9, m14, m3, m12, m13, m0
    );
    round!(
        m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
    );
    round!(
        m14, m10, m4, m8, m9, m15, m13, m6, m1, m12, m0, m2, m11, m7, m5, m3
    );

    state.h[0] ^= v0 ^ v8;
    state.h[1] ^= v1 ^ v9;
    state.h[2] ^= v2 ^ v10;
    state.h[3] ^= v3 ^ v11;
    state.h[4] ^= v4 ^ v12;
    state.h[5] ^= v5 ^ v13;
    state.h[6] ^= v6 ^ v14;
    state.h[7] ^= v7 ^ v15;
}

#[inline(always)]
fn blake2b_compress(state: &mut Blake2bState, block: &[u8]) {
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        feature = "blake2b-neon"
    ))]
    {
        neon::compress(state, block);
    }

    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        feature = "blake2b-neon"
    )))]
    {
        blake2b_compress_scalar(state, block);
    }
}

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    feature = "blake2b-neon"
))]
mod neon {
    use super::{
        BLAKE2B_BLOCKBYTES, Blake2bState, IV, SIGMA, blake2b_compress_scalar, load64_unchecked,
    };
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    struct U64x4 {
        lo: uint64x2_t,
        hi: uint64x2_t,
    }

    impl U64x4 {
        #[inline(always)]
        fn from_array(v: [u64; 4]) -> Self {
            unsafe {
                Self {
                    lo: vld1q_u64(v.as_ptr()),
                    hi: vld1q_u64(v.as_ptr().add(2)),
                }
            }
        }

        #[inline(always)]
        unsafe fn load(ptr: *const u64) -> Self {
            unsafe {
                Self {
                    lo: vld1q_u64(ptr),
                    hi: vld1q_u64(ptr.add(2)),
                }
            }
        }

        #[inline(always)]
        unsafe fn store(self, ptr: *mut u64) {
            unsafe {
                vst1q_u64(ptr, self.lo);
                vst1q_u64(ptr.add(2), self.hi);
            }
        }

        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            unsafe {
                Self {
                    lo: vaddq_u64(self.lo, rhs.lo),
                    hi: vaddq_u64(self.hi, rhs.hi),
                }
            }
        }

        #[inline(always)]
        fn xor(self, rhs: Self) -> Self {
            unsafe {
                Self {
                    lo: veorq_u64(self.lo, rhs.lo),
                    hi: veorq_u64(self.hi, rhs.hi),
                }
            }
        }

        #[inline(always)]
        fn rot32(self) -> Self {
            unsafe {
                let lo = vorrq_u64(vshrq_n_u64(self.lo, 32), vshlq_n_u64(self.lo, 32));
                let hi = vorrq_u64(vshrq_n_u64(self.hi, 32), vshlq_n_u64(self.hi, 32));
                Self { lo, hi }
            }
        }

        #[inline(always)]
        fn rot24(self) -> Self {
            unsafe {
                let lo = vorrq_u64(vshrq_n_u64(self.lo, 24), vshlq_n_u64(self.lo, 40));
                let hi = vorrq_u64(vshrq_n_u64(self.hi, 24), vshlq_n_u64(self.hi, 40));
                Self { lo, hi }
            }
        }

        #[inline(always)]
        fn rot16(self) -> Self {
            unsafe {
                let lo = vorrq_u64(vshrq_n_u64(self.lo, 16), vshlq_n_u64(self.lo, 48));
                let hi = vorrq_u64(vshrq_n_u64(self.hi, 16), vshlq_n_u64(self.hi, 48));
                Self { lo, hi }
            }
        }

        #[inline(always)]
        fn rot63(self) -> Self {
            unsafe {
                let lo = vorrq_u64(vshrq_n_u64(self.lo, 63), vshlq_n_u64(self.lo, 1));
                let hi = vorrq_u64(vshrq_n_u64(self.hi, 63), vshlq_n_u64(self.hi, 1));
                Self { lo, hi }
            }
        }

        #[inline(always)]
        fn rot1(self) -> Self {
            unsafe {
                let lo = vextq_u64(self.lo, self.hi, 1);
                let hi = vextq_u64(self.hi, self.lo, 1);
                Self { lo, hi }
            }
        }

        #[inline(always)]
        fn rot2(self) -> Self {
            Self {
                lo: self.hi,
                hi: self.lo,
            }
        }

        #[inline(always)]
        fn rot3(self) -> Self {
            unsafe {
                let lo = vextq_u64(self.hi, self.lo, 1);
                let hi = vextq_u64(self.lo, self.hi, 1);
                Self { lo, hi }
            }
        }
    }

    #[inline(always)]
    unsafe fn gather(m: *const u64, a: usize, b: usize, c: usize, d: usize) -> U64x4 {
        unsafe {
            let lo = vsetq_lane_u64(*m.add(a), vdupq_n_u64(0), 0);
            let lo = vsetq_lane_u64(*m.add(b), lo, 1);
            let hi = vsetq_lane_u64(*m.add(c), vdupq_n_u64(0), 0);
            let hi = vsetq_lane_u64(*m.add(d), hi, 1);
            U64x4 { lo, hi }
        }
    }

    #[inline(always)]
    fn g(
        a: U64x4,
        b: U64x4,
        c: U64x4,
        d: U64x4,
        x: U64x4,
        y: U64x4,
    ) -> (U64x4, U64x4, U64x4, U64x4) {
        let a = a.add(b).add(x);
        let d = d.xor(a).rot32();
        let c = c.add(d);
        let b = b.xor(c).rot24();
        let a = a.add(b).add(y);
        let d = d.xor(a).rot16();
        let c = c.add(d);
        let b = b.xor(c).rot63();
        (a, b, c, d)
    }

    #[inline(always)]
    pub fn compress(state: &mut Blake2bState, block: &[u8]) {
        if block.len() != BLAKE2B_BLOCKBYTES {
            blake2b_compress_scalar(state, block);
            return;
        }

        let mut m = [0u64; 16];
        let mut i = 0usize;
        let base = block.as_ptr();
        while i < 16 {
            let off = i * 8;
            m[i] = unsafe { load64_unchecked(base.add(off)) };
            i += 1;
        }

        let mut v0 = U64x4::from_array([state.h[0], state.h[1], state.h[2], state.h[3]]);
        let mut v1 = U64x4::from_array([state.h[4], state.h[5], state.h[6], state.h[7]]);
        let mut v2 = U64x4::from_array([IV[0], IV[1], IV[2], IV[3]]);
        let mut v3 = U64x4::from_array([
            IV[4] ^ state.t[0],
            IV[5] ^ state.t[1],
            IV[6] ^ state.f[0],
            IV[7] ^ state.f[1],
        ]);

        let m_ptr = m.as_ptr();
        for r in 0..12 {
            let s = unsafe { SIGMA.get_unchecked(r) };
            let x0 = unsafe { gather(m_ptr, s[0], s[2], s[4], s[6]) };
            let y0 = unsafe { gather(m_ptr, s[1], s[3], s[5], s[7]) };
            let (a, b, c, d) = g(v0, v1, v2, v3, x0, y0);
            v0 = a;
            v1 = b;
            v2 = c;
            v3 = d;

            v1 = v1.rot1();
            v2 = v2.rot2();
            v3 = v3.rot3();

            let x1 = unsafe { gather(m_ptr, s[8], s[10], s[12], s[14]) };
            let y1 = unsafe { gather(m_ptr, s[9], s[11], s[13], s[15]) };
            let (a, b, c, d) = g(v0, v1, v2, v3, x1, y1);
            v0 = a;
            v1 = b;
            v2 = c;
            v3 = d;

            v1 = v1.rot3();
            v2 = v2.rot2();
            v3 = v3.rot1();
        }

        unsafe {
            let h0 = v0.xor(v2);
            let h1 = v1.xor(v3);

            let mut s0 = U64x4::load(state.h.as_ptr());
            let mut s1 = U64x4::load(state.h.as_ptr().add(4));

            s0 = s0.xor(h0);
            s1 = s1.xor(h1);

            s0.store(state.h.as_mut_ptr());
            s1.store(state.h.as_mut_ptr().add(4));
        }
    }
}

fn blake2b_update(state: &mut Blake2bState, input: &[u8]) {
    if input.is_empty() {
        return;
    }
    let mut in_offset = 0;
    let mut left = state.buflen;
    let fill = BLAKE2B_BLOCKBYTES - left;

    if input.len() > fill {
        state.buflen = 0;
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr(), state.buf.as_mut_ptr().add(left), fill);
        }
        blake2b_increment_counter(state, BLAKE2B_BLOCKBYTES as u64);
        let block = unsafe { std::slice::from_raw_parts(state.buf.as_ptr(), BLAKE2B_BLOCKBYTES) };
        blake2b_compress(state, block);
        in_offset += fill;

        while input.len() - in_offset > BLAKE2B_BLOCKBYTES {
            blake2b_increment_counter(state, BLAKE2B_BLOCKBYTES as u64);
            let block = &input[in_offset..in_offset + BLAKE2B_BLOCKBYTES];
            blake2b_compress(state, block);
            in_offset += BLAKE2B_BLOCKBYTES;
        }

        left = 0;
    }

    let remaining = input.len() - in_offset;
    unsafe {
        std::ptr::copy_nonoverlapping(
            input.as_ptr().add(in_offset),
            state.buf.as_mut_ptr().add(left),
            remaining,
        );
    }
    state.buflen = left + remaining;
}

fn blake2b_final(state: &mut Blake2bState, out: &mut [u8]) -> Result<(), ()> {
    if out.len() < state.outlen || blake2b_is_lastblock(state) {
        return Err(());
    }

    blake2b_increment_counter(state, state.buflen as u64);
    blake2b_set_lastblock(state);
    for b in state.buf[state.buflen..].iter_mut() {
        *b = 0;
    }
    let block = unsafe { std::slice::from_raw_parts(state.buf.as_ptr(), BLAKE2B_BLOCKBYTES) };
    blake2b_compress(state, block);

    if state.outlen == BLAKE2B_OUTBYTES {
        for i in 0..8 {
            store64(&mut out[i * 8..], state.h[i]);
        }
    } else {
        let mut buffer = [0u8; BLAKE2B_OUTBYTES];
        for i in 0..8 {
            store64(&mut buffer[i * 8..], state.h[i]);
        }
        out[..state.outlen].copy_from_slice(&buffer[..state.outlen]);
    }
    Ok(())
}

pub fn blake2xb(out: &mut [u8], input: &[u8], key: &[u8]) -> Result<(), ()> {
    if out.is_empty() || out.len() > 0xFFFF_FFFF {
        return Err(());
    }
    if !key.is_empty() && key.len() > BLAKE2B_KEYBYTES {
        return Err(());
    }

    let mut param = Blake2bParam {
        digest_length: BLAKE2B_OUTBYTES as u8,
        key_length: key.len() as u8,
        fanout: 1,
        depth: 1,
        leaf_length: 0,
        node_offset: 0,
        xof_length: out.len() as u32,
        node_depth: 0,
        inner_length: 0,
        reserved: [0u8; 14],
        salt: [0u8; 16],
        personal: [0u8; 16],
    };

    let mut root_state = Blake2bState::new();
    blake2b_init_param(&mut root_state, &param);
    if !key.is_empty() {
        let mut block = [0u8; BLAKE2B_BLOCKBYTES];
        block[..key.len()].copy_from_slice(key);
        blake2b_update(&mut root_state, &block);
    }
    blake2b_update(&mut root_state, input);
    let mut root = [0u8; BLAKE2B_OUTBYTES];
    blake2b_final(&mut root_state, &mut root)?;

    // Set common block values for output nodes
    param.key_length = 0;
    param.fanout = 0;
    param.depth = 0;
    param.leaf_length = BLAKE2B_OUTBYTES as u32;
    param.inner_length = BLAKE2B_OUTBYTES as u8;
    param.node_depth = 0;

    let mut offset = 0usize;
    let mut counter = 0u32;
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        feature = "blake2b-neon"
    ))]
    {
        let full_blocks = out.len() / BLAKE2B_OUTBYTES;
        if full_blocks >= 2 {
            let bytes = neon_x2::blake2xb_outblocks_x2(
                &mut out[..full_blocks * BLAKE2B_OUTBYTES],
                &root,
                &param,
                counter,
            );
            offset = bytes;
            counter = counter.wrapping_add((bytes / BLAKE2B_OUTBYTES) as u32);
        }
    }
    while offset < out.len() {
        let block_size = (out.len() - offset).min(BLAKE2B_OUTBYTES);
        param.digest_length = block_size as u8;
        param.node_offset = counter;

        let mut out_state = Blake2bState::new();
        blake2b_init_param(&mut out_state, &param);
        blake2b_update(&mut out_state, &root);
        blake2b_final(&mut out_state, &mut out[offset..offset + block_size])?;

        offset += block_size;
        counter = counter.wrapping_add(1);
    }
    Ok(())
}

pub fn blake2xb_bytes(out: &mut [u8], counter: u64, key: &[u8; 64]) -> Result<(), ()> {
    let input = counter.to_le_bytes();
    blake2xb(out, &input, key)
}

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    feature = "blake2b-neon"
))]
mod neon_x2 {
    use super::{
        BLAKE2B_OUTBYTES, Blake2bParam, Blake2bState, IV, blake2b_init_param, load64_unchecked,
    };
    use core::arch::aarch64::*;

    #[inline(always)]
    fn rot32(v: uint64x2_t) -> uint64x2_t {
        unsafe { vorrq_u64(vshrq_n_u64(v, 32), vshlq_n_u64(v, 32)) }
    }
    #[inline(always)]
    fn rot24(v: uint64x2_t) -> uint64x2_t {
        unsafe { vorrq_u64(vshrq_n_u64(v, 24), vshlq_n_u64(v, 40)) }
    }
    #[inline(always)]
    fn rot16(v: uint64x2_t) -> uint64x2_t {
        unsafe { vorrq_u64(vshrq_n_u64(v, 16), vshlq_n_u64(v, 48)) }
    }
    #[inline(always)]
    fn rot63(v: uint64x2_t) -> uint64x2_t {
        unsafe { vorrq_u64(vshrq_n_u64(v, 63), vshlq_n_u64(v, 1)) }
    }

    #[inline(always)]
    fn g(
        mut a: uint64x2_t,
        mut b: uint64x2_t,
        mut c: uint64x2_t,
        mut d: uint64x2_t,
        x: uint64x2_t,
        y: uint64x2_t,
    ) -> (uint64x2_t, uint64x2_t, uint64x2_t, uint64x2_t) {
        unsafe {
            a = vaddq_u64(vaddq_u64(a, b), x);
            d = rot32(veorq_u64(d, a));
            c = vaddq_u64(c, d);
            b = rot24(veorq_u64(b, c));
            a = vaddq_u64(vaddq_u64(a, b), y);
            d = rot16(veorq_u64(d, a));
            c = vaddq_u64(c, d);
            b = rot63(veorq_u64(b, c));
        }
        (a, b, c, d)
    }

    #[inline(always)]
    unsafe fn vdup(x: u64) -> uint64x2_t {
        unsafe { vdupq_n_u64(x) }
    }

    pub fn blake2xb_outblocks_x2(
        out: &mut [u8],
        root: &[u8; 64],
        param_template: &Blake2bParam,
        start_counter: u32,
    ) -> usize {
        let total_blocks = out.len() / BLAKE2B_OUTBYTES;
        let pair_blocks = total_blocks & !1;
        if pair_blocks == 0 {
            return 0;
        }

        // Prepare message words (root padded with zeros to 128 bytes).
        let mut block = [0u8; 128];
        block[..64].copy_from_slice(root);
        let base = block.as_ptr();
        let m0 = unsafe { load64_unchecked(base.add(0)) };
        let m1 = unsafe { load64_unchecked(base.add(8)) };
        let m2 = unsafe { load64_unchecked(base.add(16)) };
        let m3 = unsafe { load64_unchecked(base.add(24)) };
        let m4 = unsafe { load64_unchecked(base.add(32)) };
        let m5 = unsafe { load64_unchecked(base.add(40)) };
        let m6 = unsafe { load64_unchecked(base.add(48)) };
        let m7 = unsafe { load64_unchecked(base.add(56)) };
        let m8 = unsafe { load64_unchecked(base.add(64)) };
        let m9 = unsafe { load64_unchecked(base.add(72)) };
        let m10 = unsafe { load64_unchecked(base.add(80)) };
        let m11 = unsafe { load64_unchecked(base.add(88)) };
        let m12 = unsafe { load64_unchecked(base.add(96)) };
        let m13 = unsafe { load64_unchecked(base.add(104)) };
        let m14 = unsafe { load64_unchecked(base.add(112)) };
        let m15 = unsafe { load64_unchecked(base.add(120)) };

        let vm0 = unsafe { vdup(m0) };
        let vm1 = unsafe { vdup(m1) };
        let vm2 = unsafe { vdup(m2) };
        let vm3 = unsafe { vdup(m3) };
        let vm4 = unsafe { vdup(m4) };
        let vm5 = unsafe { vdup(m5) };
        let vm6 = unsafe { vdup(m6) };
        let vm7 = unsafe { vdup(m7) };
        let vm8 = unsafe { vdup(m8) };
        let vm9 = unsafe { vdup(m9) };
        let vm10 = unsafe { vdup(m10) };
        let vm11 = unsafe { vdup(m11) };
        let vm12 = unsafe { vdup(m12) };
        let vm13 = unsafe { vdup(m13) };
        let vm14 = unsafe { vdup(m14) };
        let vm15 = unsafe { vdup(m15) };

        let v_iv0 = unsafe { vdup(IV[0]) };
        let v_iv1 = unsafe { vdup(IV[1]) };
        let v_iv2 = unsafe { vdup(IV[2]) };
        let v_iv3 = unsafe { vdup(IV[3]) };
        let v_iv4 = unsafe { vdup(IV[4]) };
        let v_iv5 = unsafe { vdup(IV[5]) };
        let v_iv6 = unsafe { vdup(IV[6]) };
        let v_iv7 = unsafe { vdup(IV[7]) };

        let v_t0 = unsafe { vdup(64) };
        let v_t1 = unsafe { vdup(0) };
        let v_f0 = unsafe { vdup(u64::MAX) };
        let v_f1 = unsafe { vdup(0) };

        let mut block_index = 0usize;
        while block_index < pair_blocks {
            let mut p0 = *param_template;
            let mut p1 = *param_template;
            p0.node_offset = start_counter.wrapping_add(block_index as u32);
            p1.node_offset = start_counter.wrapping_add((block_index + 1) as u32);

            let mut s0 = Blake2bState::new();
            let mut s1 = Blake2bState::new();
            blake2b_init_param(&mut s0, &p0);
            blake2b_init_param(&mut s1, &p1);

            // Pack initial state into vectors
            let mut v0 = unsafe { vdupq_n_u64(s0.h[0]) };
            let mut v1 = unsafe { vdupq_n_u64(s0.h[1]) };
            let mut v2 = unsafe { vdupq_n_u64(s0.h[2]) };
            let mut v3 = unsafe { vdupq_n_u64(s0.h[3]) };
            let mut v4 = unsafe { vdupq_n_u64(s0.h[4]) };
            let mut v5 = unsafe { vdupq_n_u64(s0.h[5]) };
            let mut v6 = unsafe { vdupq_n_u64(s0.h[6]) };
            let mut v7 = unsafe { vdupq_n_u64(s0.h[7]) };
            unsafe {
                v0 = vsetq_lane_u64(s1.h[0], v0, 1);
                v1 = vsetq_lane_u64(s1.h[1], v1, 1);
                v2 = vsetq_lane_u64(s1.h[2], v2, 1);
                v3 = vsetq_lane_u64(s1.h[3], v3, 1);
                v4 = vsetq_lane_u64(s1.h[4], v4, 1);
                v5 = vsetq_lane_u64(s1.h[5], v5, 1);
                v6 = vsetq_lane_u64(s1.h[6], v6, 1);
                v7 = vsetq_lane_u64(s1.h[7], v7, 1);
            }

            let h0_init = v0;
            let h1_init = v1;
            let h2_init = v2;
            let h3_init = v3;
            let h4_init = v4;
            let h5_init = v5;
            let h6_init = v6;
            let h7_init = v7;

            let mut v8 = v_iv0;
            let mut v9 = v_iv1;
            let mut v10 = v_iv2;
            let mut v11 = v_iv3;
            let mut v12 = unsafe { veorq_u64(v_iv4, v_t0) };
            let mut v13 = unsafe { veorq_u64(v_iv5, v_t1) };
            let mut v14 = unsafe { veorq_u64(v_iv6, v_f0) };
            let mut v15 = unsafe { veorq_u64(v_iv7, v_f1) };

            macro_rules! round {
                ($s0:expr, $s1:expr, $s2:expr, $s3:expr, $s4:expr, $s5:expr, $s6:expr, $s7:expr,
                 $s8:expr, $s9:expr, $s10:expr, $s11:expr, $s12:expr, $s13:expr, $s14:expr, $s15:expr) => {{
                    let r = g(v0, v4, v8, v12, $s0, $s1);
                    v0 = r.0;
                    v4 = r.1;
                    v8 = r.2;
                    v12 = r.3;
                    let r = g(v1, v5, v9, v13, $s2, $s3);
                    v1 = r.0;
                    v5 = r.1;
                    v9 = r.2;
                    v13 = r.3;
                    let r = g(v2, v6, v10, v14, $s4, $s5);
                    v2 = r.0;
                    v6 = r.1;
                    v10 = r.2;
                    v14 = r.3;
                    let r = g(v3, v7, v11, v15, $s6, $s7);
                    v3 = r.0;
                    v7 = r.1;
                    v11 = r.2;
                    v15 = r.3;
                    let r = g(v0, v5, v10, v15, $s8, $s9);
                    v0 = r.0;
                    v5 = r.1;
                    v10 = r.2;
                    v15 = r.3;
                    let r = g(v1, v6, v11, v12, $s10, $s11);
                    v1 = r.0;
                    v6 = r.1;
                    v11 = r.2;
                    v12 = r.3;
                    let r = g(v2, v7, v8, v13, $s12, $s13);
                    v2 = r.0;
                    v7 = r.1;
                    v8 = r.2;
                    v13 = r.3;
                    let r = g(v3, v4, v9, v14, $s14, $s15);
                    v3 = r.0;
                    v4 = r.1;
                    v9 = r.2;
                    v14 = r.3;
                }};
            }

            round!(
                vm0, vm1, vm2, vm3, vm4, vm5, vm6, vm7, vm8, vm9, vm10, vm11, vm12, vm13, vm14,
                vm15
            );
            round!(
                vm14, vm10, vm4, vm8, vm9, vm15, vm13, vm6, vm1, vm12, vm0, vm2, vm11, vm7, vm5,
                vm3
            );
            round!(
                vm11, vm8, vm12, vm0, vm5, vm2, vm15, vm13, vm10, vm14, vm3, vm6, vm7, vm1, vm9,
                vm4
            );
            round!(
                vm7, vm9, vm3, vm1, vm13, vm12, vm11, vm14, vm2, vm6, vm5, vm10, vm4, vm0, vm15,
                vm8
            );
            round!(
                vm9, vm0, vm5, vm7, vm2, vm4, vm10, vm15, vm14, vm1, vm11, vm12, vm6, vm8, vm3,
                vm13
            );
            round!(
                vm2, vm12, vm6, vm10, vm0, vm11, vm8, vm3, vm4, vm13, vm7, vm5, vm15, vm14, vm1,
                vm9
            );
            round!(
                vm12, vm5, vm1, vm15, vm14, vm13, vm4, vm10, vm0, vm7, vm6, vm3, vm9, vm2, vm8,
                vm11
            );
            round!(
                vm13, vm11, vm7, vm14, vm12, vm1, vm3, vm9, vm5, vm0, vm15, vm4, vm8, vm6, vm2,
                vm10
            );
            round!(
                vm6, vm15, vm14, vm9, vm11, vm3, vm0, vm8, vm12, vm2, vm13, vm7, vm1, vm4, vm10,
                vm5
            );
            round!(
                vm10, vm2, vm8, vm4, vm7, vm6, vm1, vm5, vm15, vm11, vm9, vm14, vm3, vm12, vm13,
                vm0
            );
            round!(
                vm0, vm1, vm2, vm3, vm4, vm5, vm6, vm7, vm8, vm9, vm10, vm11, vm12, vm13, vm14,
                vm15
            );
            round!(
                vm14, vm10, vm4, vm8, vm9, vm15, vm13, vm6, vm1, vm12, vm0, vm2, vm11, vm7, vm5,
                vm3
            );

            let h0 = unsafe { veorq_u64(h0_init, veorq_u64(v0, v8)) };
            let h1 = unsafe { veorq_u64(h1_init, veorq_u64(v1, v9)) };
            let h2 = unsafe { veorq_u64(h2_init, veorq_u64(v2, v10)) };
            let h3 = unsafe { veorq_u64(h3_init, veorq_u64(v3, v11)) };
            let h4 = unsafe { veorq_u64(h4_init, veorq_u64(v4, v12)) };
            let h5 = unsafe { veorq_u64(h5_init, veorq_u64(v5, v13)) };
            let h6 = unsafe { veorq_u64(h6_init, veorq_u64(v6, v14)) };
            let h7 = unsafe { veorq_u64(h7_init, veorq_u64(v7, v15)) };

            unsafe {
                let mut tmp = [0u64; 2];
                let out0 = out.as_mut_ptr().add(block_index * BLAKE2B_OUTBYTES);
                let out1 = out.as_mut_ptr().add((block_index + 1) * BLAKE2B_OUTBYTES);

                vst1q_u64(tmp.as_mut_ptr(), h0);
                (out0 as *mut u64).write_unaligned(tmp[0].to_le());
                (out1 as *mut u64).write_unaligned(tmp[1].to_le());
                vst1q_u64(tmp.as_mut_ptr(), h1);
                (out0.add(8) as *mut u64).write_unaligned(tmp[0].to_le());
                (out1.add(8) as *mut u64).write_unaligned(tmp[1].to_le());
                vst1q_u64(tmp.as_mut_ptr(), h2);
                (out0.add(16) as *mut u64).write_unaligned(tmp[0].to_le());
                (out1.add(16) as *mut u64).write_unaligned(tmp[1].to_le());
                vst1q_u64(tmp.as_mut_ptr(), h3);
                (out0.add(24) as *mut u64).write_unaligned(tmp[0].to_le());
                (out1.add(24) as *mut u64).write_unaligned(tmp[1].to_le());
                vst1q_u64(tmp.as_mut_ptr(), h4);
                (out0.add(32) as *mut u64).write_unaligned(tmp[0].to_le());
                (out1.add(32) as *mut u64).write_unaligned(tmp[1].to_le());
                vst1q_u64(tmp.as_mut_ptr(), h5);
                (out0.add(40) as *mut u64).write_unaligned(tmp[0].to_le());
                (out1.add(40) as *mut u64).write_unaligned(tmp[1].to_le());
                vst1q_u64(tmp.as_mut_ptr(), h6);
                (out0.add(48) as *mut u64).write_unaligned(tmp[0].to_le());
                (out1.add(48) as *mut u64).write_unaligned(tmp[1].to_le());
                vst1q_u64(tmp.as_mut_ptr(), h7);
                (out0.add(56) as *mut u64).write_unaligned(tmp[0].to_le());
                (out1.add(56) as *mut u64).write_unaligned(tmp[1].to_le());
            }

            block_index += 2;
        }

        pair_blocks * BLAKE2B_OUTBYTES
    }
}
