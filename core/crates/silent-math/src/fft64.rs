//! Negacyclic polynomial arithmetic in `Z_q[X] / (X^N + 1)` for native
//! power-of-two moduli `q = 2^k`.
//!
//! TFHE relies heavily on multiplications of polynomials with native-modulus
//! coefficients (typically `q = 2^32` or `q = 2^64`).  Because the modulus is
//! a power of two, *no* NTT-friendly prime exists and the BFV-style
//! [`crate::ntt`] kernels cannot be reused.  A real production backend would
//! ship a 64-bit Cooley–Tukey FFT (similar to `concrete-fft` / `tfhe-fft`).
//!
//! The MVP exposes the abstract [`NegacyclicMul`] trait so that the
//! TFHE scheme code is decoupled from the underlying multiplier, and provides
//! three implementations:
//!
//!  - [`SchoolbookMul`] — `O(N²)` reference, used as a correctness oracle.
//!  - [`KaratsubaMul`]  — `O(N^{log₂ 3}) ≈ O(N^{1.585})`, ~5–6× over
//!    schoolbook at production sizes.
//!  - [`FftMul`]        — `O(N log N)` complex-`f64` split-precision
//!    backend, ~10–12× over schoolbook at N = 2048.
//!
//! All three produce bit-exact results modulo `2⁶⁴`; the FFT one is
//! cross-validated against `SchoolbookMul`/`KaratsubaMul` in the unit
//! tests.  The FFT keeps reusable scratch buffers in thread-local storage
//! so that hot loops (PBS / blind rotation) pay zero per-call allocation
//! overhead.

use rustfft::{FftPlanner, num_complex::Complex64};
use std::sync::Arc;

#[cfg(target_os = "macos")]
use std::{ffi::c_void, ptr::NonNull};

#[cfg(target_os = "macos")]
const VDSP_DFT_FORWARD: i32 = 1;
#[cfg(target_os = "macos")]
const VDSP_DFT_INVERSE: i32 = -1;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vDSP_DFT_zop_CreateSetupD(
        previous: *mut c_void,
        length: usize,
        direction: i32,
    ) -> *mut c_void;
    fn vDSP_DFT_DestroySetupD(setup: *mut c_void);
    fn vDSP_DFT_ExecuteD(
        setup: *mut c_void,
        input_real: *const f64,
        input_imag: *const f64,
        output_real: *mut f64,
        output_imag: *mut f64,
    );
}

/// Polynomial-multiplier interface for `Z_q[X] / (X^N + 1)` with native
/// power-of-two `q`.
///
/// Implementations are expected to *wrap* on overflow — Rust's wrapping
/// semantics on `u32`/`u64` already model the modular reduction.
pub trait NegacyclicMul {
    /// Compute `result := lhs * rhs` mod `(X^N + 1, 2^64)`.  All slices must
    /// have the same length, which must be a power of two.
    fn mul_assign_u64(&self, lhs: &[u64], rhs: &[u64], result: &mut [u64]);

    /// Compute `result := result + lhs * rhs` mod `(X^N + 1, 2^64)`.  All
    /// slices must have the same length, which must be a power of two.
    fn fma_assign_u64(&self, lhs: &[u64], rhs: &[u64], result: &mut [u64]);
}

/// `O(N^2)` reference implementation of negacyclic multiplication.
///
/// Used as the MVP backend and as a correctness oracle in tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct SchoolbookMul;

impl NegacyclicMul for SchoolbookMul {
    fn mul_assign_u64(&self, lhs: &[u64], rhs: &[u64], result: &mut [u64]) {
        debug_assert_eq!(lhs.len(), rhs.len());
        debug_assert_eq!(lhs.len(), result.len());
        debug_assert!(lhs.len().is_power_of_two() && !lhs.is_empty());

        for slot in result.iter_mut() {
            *slot = 0;
        }
        negacyclic_fma_u64(lhs, rhs, result);
    }

    fn fma_assign_u64(&self, lhs: &[u64], rhs: &[u64], result: &mut [u64]) {
        debug_assert_eq!(lhs.len(), rhs.len());
        debug_assert_eq!(lhs.len(), result.len());
        debug_assert!(lhs.len().is_power_of_two() && !lhs.is_empty());

        negacyclic_fma_u64(lhs, rhs, result);
    }
}

/// Karatsuba negacyclic multiplier — `O(N^{log₂ 3}) ≈ O(N^{1.585})`.
///
/// Recursively splits the input polynomials into halves and replaces each
/// "outer" sub-multiplication with three half-size multiplications:
///
/// ```text
///   a = a_lo + a_hi · X^{N/2}
///   b = b_lo + b_hi · X^{N/2}
///
///   z0 = a_lo · b_lo
///   z2 = a_hi · b_hi
///   z1 = (a_lo + a_hi)(b_lo + b_hi) − z0 − z2
///
///   a · b   ==   z0 + z1 · X^{N/2} + z2 · X^N
/// ```
///
/// We then apply the negacyclic reduction `X^N := -1` to the (2N − 1)-length
/// intermediate to bring the result back into `Z_q[X] / (X^N + 1)`.  Because
/// `q` is power-of-two, every operation is just `wrapping_*` on `u64`.
///
/// Recursion bottoms out at `N <= KARATSUBA_BASE` where the schoolbook
/// kernel is faster than the recursion overhead.  The threshold defaults to
/// 32 — small enough to amortise the overhead even for `N = 64` test
/// polynomials, large enough not to thrash for `N = 2048` production sizes.
#[derive(Clone, Copy, Debug, Default)]
pub struct KaratsubaMul;

const KARATSUBA_BASE: usize = 32;

impl NegacyclicMul for KaratsubaMul {
    fn mul_assign_u64(&self, lhs: &[u64], rhs: &[u64], result: &mut [u64]) {
        debug_assert_eq!(lhs.len(), rhs.len());
        debug_assert_eq!(lhs.len(), result.len());
        debug_assert!(lhs.len().is_power_of_two() && !lhs.is_empty());
        karatsuba_negacyclic_mul(lhs, rhs, result);
    }

    fn fma_assign_u64(&self, lhs: &[u64], rhs: &[u64], result: &mut [u64]) {
        debug_assert_eq!(lhs.len(), rhs.len());
        debug_assert_eq!(lhs.len(), result.len());
        debug_assert!(lhs.len().is_power_of_two() && !lhs.is_empty());

        // Allocate a scratch buffer once per call.  The hot path inside PBS
        // already pays one allocation per multiplication for SchoolbookMul
        // via the `LweCiphertext::clone` upstream, so this does not regress
        // the baseline.
        let mut tmp = vec![0u64; lhs.len()];
        karatsuba_negacyclic_mul(lhs, rhs, &mut tmp);
        for (r, t) in result.iter_mut().zip(tmp.iter()) {
            *r = r.wrapping_add(*t);
        }
    }
}

/// Top-level Karatsuba: produce the negacyclic product of two N-coefficient
/// polynomials by computing the regular polynomial product into a 2N scratch
/// buffer, then folding the upper half back with the negacyclic identity
/// `X^N := -1`.
fn karatsuba_negacyclic_mul(lhs: &[u64], rhs: &[u64], out: &mut [u64]) {
    let n = lhs.len();
    if n <= KARATSUBA_BASE {
        for slot in out.iter_mut() {
            *slot = 0;
        }
        negacyclic_fma_u64(lhs, rhs, out);
        return;
    }

    let mut full = vec![0u64; 2 * n];
    karatsuba_full_product(lhs, rhs, &mut full);

    // Fold the high half back: result[k] = full[k] - full[k + n].
    for k in 0..n {
        out[k] = full[k].wrapping_sub(full[k + n]);
    }
}

/// Compute the regular (non-negacyclic) product of two equal-length
/// polynomials into a buffer of length `2 * n` (length-`(2n - 1)` data + a
/// trailing zero slot for symmetry with the recursive splits).
fn karatsuba_full_product(lhs: &[u64], rhs: &[u64], out: &mut [u64]) {
    let n = lhs.len();
    debug_assert_eq!(rhs.len(), n);
    debug_assert_eq!(out.len(), 2 * n);

    if n <= KARATSUBA_BASE {
        for slot in out.iter_mut() {
            *slot = 0;
        }
        for i in 0..n {
            let a = lhs[i];
            if a == 0 {
                continue;
            }
            for j in 0..n {
                let b = rhs[j];
                if b == 0 {
                    continue;
                }
                out[i + j] = out[i + j].wrapping_add(a.wrapping_mul(b));
            }
        }
        return;
    }

    let half = n / 2;
    let (a_lo, a_hi) = lhs.split_at(half);
    let (b_lo, b_hi) = rhs.split_at(half);

    // z0 = a_lo · b_lo  (length 2 * half = n)
    let mut z0 = vec![0u64; n];
    karatsuba_full_product(a_lo, b_lo, &mut z0);

    // z2 = a_hi · b_hi  (length n)
    let mut z2 = vec![0u64; n];
    karatsuba_full_product(a_hi, b_hi, &mut z2);

    // z1 = (a_lo + a_hi)(b_lo + b_hi) − z0 − z2
    let mut a_sum = vec![0u64; half];
    let mut b_sum = vec![0u64; half];
    for k in 0..half {
        a_sum[k] = a_lo[k].wrapping_add(a_hi[k]);
        b_sum[k] = b_lo[k].wrapping_add(b_hi[k]);
    }
    let mut z1 = vec![0u64; n];
    karatsuba_full_product(&a_sum, &b_sum, &mut z1);
    for k in 0..n {
        z1[k] = z1[k].wrapping_sub(z0[k]).wrapping_sub(z2[k]);
    }

    // Compose: out = z0 + z1 · X^half + z2 · X^n.
    for slot in out.iter_mut() {
        *slot = 0;
    }
    for k in 0..n {
        out[k] = out[k].wrapping_add(z0[k]);
    }
    for k in 0..n {
        out[k + half] = out[k + half].wrapping_add(z1[k]);
    }
    for k in 0..n {
        out[k + n] = out[k + n].wrapping_add(z2[k]);
    }
}

// ---------------------------------------------------------------------------
// f64 complex Cooley–Tukey FFT for negacyclic polynomial multiplication
// ---------------------------------------------------------------------------
//
// Algorithm (split-precision negacyclic FFT):
//
// 1. Each `u64` coefficient is split into 4 unsigned 16-bit chunks
//    `a = a₀ + a₁·2¹⁶ + a₂·2³² + a₃·2⁴⁸`.  This bounds every per-chunk
//    coefficient to `[0, 2¹⁶)`, which is the key to keeping intermediate
//    f64 results inside the 53-bit mantissa.
// 2. Each chunk polynomial is *ψ-twisted* by `aᵢ ← aᵢ · ψⁱ` with
//    `ψ = exp(iπ/N)`.  After twisting the negacyclic product on
//    `Z[X]/(X^N + 1)` becomes a regular cyclic product on
//    `Z[X]/(X^N − 1)` of the same length, so a length-N complex FFT
//    suffices (no zero-padding to 2N).
// 3. Forward FFT every twisted chunk (4 forward FFTs per input).
// 4. For each `(k, l)` with `k + l ≤ 3` (chunks with `k + l ≥ 4` carry
//    weight `≥ 2⁶⁴` and vanish mod `2⁶⁴`), accumulate the pointwise
//    product `âₖ · b̂ₗ` into the bucket `s = k + l`.
// 5. Inverse FFT each bucket and untwist by `ψ⁻ⁱ`.
// 6. Round each bucket to nearest signed integer (its real part is, by
//    construction, within `(s+1)·N·2³² < 2⁴⁵`, well inside i64 range).
// 7. Recombine via wrapping `c[i] = Σ cₛ[i] · 2¹⁶ˢ  mod 2⁶⁴`.
//
// Precision budget per bucket-`s`:
//
//     |cₛ[i]|  ≤  (s + 1) · N · (2¹⁶ − 1)²
//             <  4 · N · 2³²
//
// For N = 2048 that is ≤ 2·N·2³² · 2 ≤ 2⁴⁵, leaving 8 bits of mantissa
// headroom.  For N = 65536 the bound is 2⁴⁸ — still within f64 precision.
//
// This is a faithful (but non-vectorised) port of the algorithm
// `concrete-fft` ships under the hood, minus the SIMD/AVX kernels.

/// Complex number used by the FFT backend.
type Cplx = Complex64;

#[inline(always)]
fn cplx_mul(a: Cplx, b: Cplx) -> Cplx {
    a * b
}

#[cfg(target_os = "macos")]
struct AccelerateDftPlan {
    forward: NonNull<c_void>,
    inverse: NonNull<c_void>,
    n: usize,
}

#[cfg(target_os = "macos")]
impl AccelerateDftPlan {
    fn new(n: usize) -> Option<Arc<Self>> {
        unsafe {
            let forward = NonNull::new(vDSP_DFT_zop_CreateSetupD(
                std::ptr::null_mut(),
                n,
                VDSP_DFT_FORWARD,
            ))?;
            let inverse = NonNull::new(vDSP_DFT_zop_CreateSetupD(
                std::ptr::null_mut(),
                n,
                VDSP_DFT_INVERSE,
            ));
            match inverse {
                Some(inverse) => Some(Arc::new(Self {
                    forward,
                    inverse,
                    n,
                })),
                None => {
                    vDSP_DFT_DestroySetupD(forward.as_ptr());
                    None
                }
            }
        }
    }

    fn forward(&self, data: &mut [Cplx]) {
        self.execute(self.forward, data);
    }

    fn inverse(&self, data: &mut [Cplx]) {
        self.execute(self.inverse, data);
    }

    fn execute(&self, setup: NonNull<c_void>, data: &mut [Cplx]) {
        debug_assert_eq!(data.len(), self.n);
        ACCELERATE_FFT_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_size(self.n);
            for (i, z) in data.iter().enumerate() {
                scratch.real[i] = z.re;
                scratch.imag[i] = z.im;
            }
            let real = scratch.real.as_mut_ptr();
            let imag = scratch.imag.as_mut_ptr();
            unsafe {
                vDSP_DFT_ExecuteD(
                    setup.as_ptr(),
                    real.cast_const(),
                    imag.cast_const(),
                    real,
                    imag,
                );
            }
            for (i, z) in data.iter_mut().enumerate() {
                *z = Cplx::new(scratch.real[i], scratch.imag[i]);
            }
        });
    }
}

#[cfg(target_os = "macos")]
impl Drop for AccelerateDftPlan {
    fn drop(&mut self) {
        unsafe {
            vDSP_DFT_DestroySetupD(self.inverse.as_ptr());
            vDSP_DFT_DestroySetupD(self.forward.as_ptr());
        }
    }
}

#[cfg(target_os = "macos")]
unsafe impl Send for AccelerateDftPlan {}
#[cfg(target_os = "macos")]
unsafe impl Sync for AccelerateDftPlan {}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, Default)]
struct AccelerateFftScratch {
    real: Vec<f64>,
    imag: Vec<f64>,
}

#[cfg(target_os = "macos")]
impl AccelerateFftScratch {
    fn ensure_size(&mut self, n: usize) {
        if self.real.len() != n {
            self.real.resize(n, 0.0);
        }
        if self.imag.len() != n {
            self.imag.resize(n, 0.0);
        }
    }
}

#[cfg(target_os = "macos")]
thread_local! {
    static ACCELERATE_FFT_SCRATCH: std::cell::RefCell<AccelerateFftScratch> =
        std::cell::RefCell::new(AccelerateFftScratch::default());
}

/// Pre-computed FFT handles + ψ powers for one fixed FFT length `N`.
///
/// Construct once per polynomial size and reuse for every multiplication.
/// The plan owns the forward / inverse kernels while thread-local scratch
/// buffers amortise all per-call workspace.
#[derive(Clone)]
pub struct FftPlan {
    n: usize,
    forward_fft: Arc<dyn rustfft::Fft<f64>>,
    inverse_fft: Arc<dyn rustfft::Fft<f64>>,
    #[cfg(target_os = "macos")]
    accelerate: Option<Arc<AccelerateDftPlan>>,
    forward_scratch_len: usize,
    inverse_scratch_len: usize,
    /// `psiⁱ = exp(iπ · i / N)` for `i = 0..N`.  Used by the negacyclic
    /// twist before the forward transform.
    psi_pow: Vec<Cplx>,
    /// `psi⁻ⁱ / N` for hot inverse paths where normalisation and untwist can
    /// be fused into one pass.
    psi_inv_scaled_pow: Vec<Cplx>,
}

impl std::fmt::Debug for FftPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan").field("n", &self.n).finish()
    }
}

impl FftPlan {
    /// Build a fresh FFT plan for size `n` (must be a power of two
    /// ≥ 2).  `n = 1` is allowed but degenerate.
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "FFT size must be a power of two");
        assert!(n >= 2, "FFT size must be ≥ 2");

        let mut planner = FftPlanner::<f64>::new();
        let forward_fft = planner.plan_fft_forward(n);
        let inverse_fft = planner.plan_fft_inverse(n);
        #[cfg(target_os = "macos")]
        let accelerate = if std::env::var_os("SILENT_ENABLE_ACCELERATE").is_some() {
            AccelerateDftPlan::new(n)
        } else {
            None
        };
        let forward_scratch_len = forward_fft.get_inplace_scratch_len();
        let inverse_scratch_len = inverse_fft.get_inplace_scratch_len();

        let pi_over_n = std::f64::consts::PI / (n as f64);
        let inv_n = 1.0 / (n as f64);
        let mut psi_pow = Vec::with_capacity(n);
        let mut psi_inv_scaled_pow = Vec::with_capacity(n);
        for i in 0..n {
            let theta = pi_over_n * (i as f64);
            psi_pow.push(Cplx::new(theta.cos(), theta.sin()));
            psi_inv_scaled_pow.push(Cplx::new(theta.cos() * inv_n, -theta.sin() * inv_n));
        }

        Self {
            n,
            forward_fft,
            inverse_fft,
            #[cfg(target_os = "macos")]
            accelerate,
            forward_scratch_len,
            inverse_scratch_len,
            psi_pow,
            psi_inv_scaled_pow,
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    fn ensure_fft_scratch(scratch: &mut Vec<Cplx>, required_len: usize) {
        if scratch.len() != required_len {
            scratch.resize(required_len, Cplx::default());
        }
    }

    /// In-place forward FFT using the cached backend plan.
    fn fft(&self, data: &mut [Cplx], scratch: &mut Vec<Cplx>) {
        debug_assert_eq!(data.len(), self.n);
        #[cfg(target_os = "macos")]
        if let Some(accelerate) = &self.accelerate {
            accelerate.forward(data);
            return;
        }

        Self::ensure_fft_scratch(scratch, self.forward_scratch_len);
        if self.forward_scratch_len == 0 {
            self.forward_fft.process(data);
        } else {
            self.forward_fft
                .process_with_scratch(data, &mut scratch[..self.forward_scratch_len]);
        }
    }

    /// In-place inverse FFT.  The backend leaves the transform
    /// unnormalised.
    fn ifft_unscaled(&self, data: &mut [Cplx], scratch: &mut Vec<Cplx>) {
        debug_assert_eq!(data.len(), self.n);
        #[cfg(target_os = "macos")]
        if let Some(accelerate) = &self.accelerate {
            accelerate.inverse(data);
            return;
        }

        Self::ensure_fft_scratch(scratch, self.inverse_scratch_len);
        if self.inverse_scratch_len == 0 {
            self.inverse_fft.process(data);
        } else {
            self.inverse_fft
                .process_with_scratch(data, &mut scratch[..self.inverse_scratch_len]);
        }
    }

    /// In-place inverse FFT.  The backend leaves the transform
    /// unnormalised, so we divide by `N` afterwards.
    #[cfg(test)]
    fn ifft(&self, data: &mut [Cplx], scratch: &mut Vec<Cplx>) {
        self.ifft_unscaled(data, scratch);
        let inv_n = 1.0 / (self.n as f64);
        for c in data.iter_mut() {
            c.re *= inv_n;
            c.im *= inv_n;
        }
    }
}

/// Reusable per-thread scratch buffers for [`FftMul`].
///
/// Holds the four split-precision forward-FFT chunks of each operand plus
/// the four accumulator buckets.  Lives in a thread-local so that every
/// inner-loop polynomial multiplication during PBS reuses the same
/// allocations — at N = 2048 this saves roughly 12 × N × 16 ≈ 393 KB of
/// malloc churn per polynomial mul (≈ 1.4 GB per full PBS).
#[derive(Clone, Debug, Default)]
struct FftScratch {
    a_hat: [Vec<Cplx>; 4],
    b_hat: [Vec<Cplx>; 4],
    bucket: [Vec<Cplx>; 4],
    packed: Vec<Cplx>,
    fft_scratch: Vec<Cplx>,
}

impl FftScratch {
    /// Resize every internal buffer to exactly `n` slots.  No-op once the
    /// scratch has converged on the working size.
    fn ensure_size(&mut self, n: usize, fft_scratch_len: usize) {
        for v in self.a_hat.iter_mut() {
            if v.len() != n {
                v.resize(n, Cplx::default());
            }
        }
        for v in self.b_hat.iter_mut() {
            if v.len() != n {
                v.resize(n, Cplx::default());
            }
        }
        for v in self.bucket.iter_mut() {
            if v.len() != n {
                v.resize(n, Cplx::default());
            }
        }
        if self.packed.len() != n {
            self.packed.resize(n, Cplx::default());
        }
        if self.fft_scratch.len() != fft_scratch_len {
            self.fft_scratch.resize(fft_scratch_len, Cplx::default());
        }
    }
}

thread_local! {
    /// One scratch arena per OS thread.  Resized lazily on first use of
    /// each polynomial size; PBS workloads converge to a fixed `N`
    /// after the first call so the resize is amortised away.
    static FFT_SCRATCH: std::cell::RefCell<FftScratch> =
        std::cell::RefCell::new(FftScratch::default());
}

/// Split-precision FFT-based negacyclic multiplier.  `O(N log N)` per
/// polynomial multiplication on the `f64` complex transform, bit-exact
/// modulo `2⁶⁴` for inputs of any magnitude.
///
/// One [`FftPlan`] per polynomial size — construct it once and reuse it
/// across every PBS in a session.  The transform itself is allocation-free
/// after the first call: scratch buffers live in thread-local storage and
/// are resized on demand.
///
/// `FftMul` is `Send + Sync`: instances may be shared by reference across
/// threads.  Each thread that actually invokes the multiplier owns an
/// independent scratch arena.
#[derive(Clone, Debug)]
pub struct FftMul {
    plan: FftPlan,
}

impl FftMul {
    /// Build an FFT-based multiplier for size `n` (must be a power of
    /// two ≥ 2).
    pub fn new(n: usize) -> Self {
        Self {
            plan: FftPlan::new(n),
        }
    }

    pub fn n(&self) -> usize {
        self.plan.n
    }

    /// Decompose a `u64` polynomial into 4 chunks of 16-bit unsigned
    /// values, ψ-twist each, and forward-FFT each.  Writes into
    /// caller-provided scratch (zero allocation).
    fn forward_split_chunks_into(
        &self,
        input: &[u64],
        out: &mut [Vec<Cplx>; 4],
        packed_scratch: &mut Vec<Cplx>,
        fft_scratch: &mut Vec<Cplx>,
    ) {
        let n = self.plan.n;
        debug_assert_eq!(input.len(), n);
        debug_assert_eq!(packed_scratch.len(), n);

        self.forward_chunk_pair_into(input, 0, 16, packed_scratch, out, 0, fft_scratch);
        self.forward_chunk_pair_into(input, 32, 48, packed_scratch, out, 2, fft_scratch);
    }

    fn forward_chunk_pair_into(
        &self,
        input: &[u64],
        low_shift: u32,
        high_shift: u32,
        packed: &mut [Cplx],
        out: &mut [Vec<Cplx>; 4],
        out_pair: usize,
        fft_scratch: &mut Vec<Cplx>,
    ) {
        let n = self.plan.n;
        for i in 0..n {
            let v = input[i];
            let low = ((v >> low_shift) & 0xFFFF) as f64;
            let high = ((v >> high_shift) & 0xFFFF) as f64;
            let psi = self.plan.psi_pow[i];
            packed[i] = Cplx::new(low * psi.re - high * psi.im, low * psi.im + high * psi.re);
        }

        self.plan.fft(packed, fft_scratch);
        let (low_chunks, high_chunks) = out.split_at_mut(out_pair + 1);
        unpack_paired_forward(packed, &mut low_chunks[out_pair], &mut high_chunks[0]);
    }

    fn forward_chunk_pair_half_into(
        &self,
        input: &[u64],
        low_shift: u32,
        high_shift: u32,
        packed: &mut [Cplx],
        out: &mut [Vec<Cplx>; 4],
        out_pair: usize,
        fft_scratch: &mut Vec<Cplx>,
    ) {
        let n = self.plan.n;
        for i in 0..n {
            let v = input[i];
            let low = ((v >> low_shift) & 0xFFFF) as f64;
            let high = ((v >> high_shift) & 0xFFFF) as f64;
            let psi = self.plan.psi_pow[i];
            packed[i] = Cplx::new(low * psi.re - high * psi.im, low * psi.im + high * psi.re);
        }

        self.plan.fft(packed, fft_scratch);
        let (low_chunks, high_chunks) = out.split_at_mut(out_pair + 1);
        unpack_paired_forward_half(packed, &mut low_chunks[out_pair], &mut high_chunks[0]);
    }

    fn inverse_bucket_pair_add_into(
        &self,
        low: &[Cplx],
        high: &[Cplx],
        packed: &mut [Cplx],
        fft_scratch: &mut Vec<Cplx>,
        low_shift: u32,
        high_shift: u32,
        out: &mut [u64],
    ) {
        let n = self.plan.n;
        for i in 0..n {
            packed[i] = low[i] + Cplx::new(-high[i].im, high[i].re);
        }

        self.plan.ifft_unscaled(packed, fft_scratch);
        for i in 0..n {
            let z = packed[i];
            let psi = self.plan.psi_inv_scaled_pow[i];
            let low = (z.re * psi.re - z.im * psi.im).round() as i64 as u64;
            let high = (z.re * psi.im + z.im * psi.re).round() as i64 as u64;
            out[i] = out[i]
                .wrapping_add(low.wrapping_shl(low_shift))
                .wrapping_add(high.wrapping_shl(high_shift));
        }
    }

    fn inverse_bucket_pair_write_into(
        &self,
        low: &[Cplx],
        high: &[Cplx],
        packed: &mut [Cplx],
        fft_scratch: &mut Vec<Cplx>,
        low_shift: u32,
        high_shift: u32,
        out: &mut [u64],
    ) {
        let n = self.plan.n;
        for i in 0..n {
            packed[i] = low[i] + Cplx::new(-high[i].im, high[i].re);
        }

        self.plan.ifft_unscaled(packed, fft_scratch);
        for i in 0..n {
            let z = packed[i];
            let psi = self.plan.psi_inv_scaled_pow[i];
            let low = (z.re * psi.re - z.im * psi.im).round() as i64 as u64;
            let high = (z.re * psi.im + z.im * psi.re).round() as i64 as u64;
            out[i] = low
                .wrapping_shl(low_shift)
                .wrapping_add(high.wrapping_shl(high_shift));
        }
    }

    fn inverse_bucket_pair_write_with_addend_into(
        &self,
        low: &[Cplx],
        high: &[Cplx],
        packed: &mut [Cplx],
        fft_scratch: &mut Vec<Cplx>,
        low_shift: u32,
        high_shift: u32,
        addend: &[u64],
        out: &mut [u64],
    ) {
        let n = self.plan.n;
        debug_assert_eq!(addend.len(), n);
        for i in 0..n {
            packed[i] = low[i] + Cplx::new(-high[i].im, high[i].re);
        }

        self.plan.ifft_unscaled(packed, fft_scratch);
        for i in 0..n {
            let z = packed[i];
            let psi = self.plan.psi_inv_scaled_pow[i];
            let low = (z.re * psi.re - z.im * psi.im).round() as i64 as u64;
            let high = (z.re * psi.im + z.im * psi.re).round() as i64 as u64;
            out[i] = addend[i]
                .wrapping_add(low.wrapping_shl(low_shift))
                .wrapping_add(high.wrapping_shl(high_shift));
        }
    }

    fn inverse_half_bucket_pair_add_into(
        &self,
        low: &[Cplx],
        high: &[Cplx],
        packed: &mut [Cplx],
        fft_scratch: &mut Vec<Cplx>,
        low_shift: u32,
        high_shift: u32,
        out: &mut [u64],
    ) {
        let n = self.plan.n;
        let half = n >> 1;
        debug_assert_eq!(low.len(), half);
        debug_assert_eq!(high.len(), half);
        debug_assert_eq!(packed.len(), n);

        pack_half_bucket_pair(low[0], high[0], packed, 0, 1);
        for t in 1..half {
            let k = t + 1;
            let sym = n + 1 - k;
            pack_half_bucket_pair(low[t], high[t], packed, k, sym);
        }

        self.plan.ifft_unscaled(packed, fft_scratch);
        for i in 0..n {
            let z = packed[i];
            let psi = self.plan.psi_inv_scaled_pow[i];
            let low = (z.re * psi.re - z.im * psi.im).round() as i64 as u64;
            let high = (z.re * psi.im + z.im * psi.re).round() as i64 as u64;
            out[i] = out[i]
                .wrapping_add(low.wrapping_shl(low_shift))
                .wrapping_add(high.wrapping_shl(high_shift));
        }
    }

    fn inverse_half_bucket_pair_write_into(
        &self,
        low: &[Cplx],
        high: &[Cplx],
        packed: &mut [Cplx],
        fft_scratch: &mut Vec<Cplx>,
        low_shift: u32,
        high_shift: u32,
        out: &mut [u64],
    ) {
        let n = self.plan.n;
        let half = n >> 1;
        debug_assert_eq!(low.len(), half);
        debug_assert_eq!(high.len(), half);
        debug_assert_eq!(packed.len(), n);

        pack_half_bucket_pair(low[0], high[0], packed, 0, 1);
        for t in 1..half {
            let k = t + 1;
            let sym = n + 1 - k;
            pack_half_bucket_pair(low[t], high[t], packed, k, sym);
        }

        self.plan.ifft_unscaled(packed, fft_scratch);
        for i in 0..n {
            let z = packed[i];
            let psi = self.plan.psi_inv_scaled_pow[i];
            let low = (z.re * psi.re - z.im * psi.im).round() as i64 as u64;
            let high = (z.re * psi.im + z.im * psi.re).round() as i64 as u64;
            out[i] = low
                .wrapping_shl(low_shift)
                .wrapping_add(high.wrapping_shl(high_shift));
        }
    }

    fn inverse_half_bucket_pair_write_with_addend_into(
        &self,
        low: &[Cplx],
        high: &[Cplx],
        packed: &mut [Cplx],
        fft_scratch: &mut Vec<Cplx>,
        low_shift: u32,
        high_shift: u32,
        addend: &[u64],
        out: &mut [u64],
    ) {
        let n = self.plan.n;
        let half = n >> 1;
        debug_assert_eq!(low.len(), half);
        debug_assert_eq!(high.len(), half);
        debug_assert_eq!(packed.len(), n);
        debug_assert_eq!(addend.len(), n);

        pack_half_bucket_pair(low[0], high[0], packed, 0, 1);
        for t in 1..half {
            let k = t + 1;
            let sym = n + 1 - k;
            pack_half_bucket_pair(low[t], high[t], packed, k, sym);
        }

        self.plan.ifft_unscaled(packed, fft_scratch);
        for i in 0..n {
            let z = packed[i];
            let psi = self.plan.psi_inv_scaled_pow[i];
            let low = (z.re * psi.re - z.im * psi.im).round() as i64 as u64;
            let high = (z.re * psi.im + z.im * psi.re).round() as i64 as u64;
            out[i] = addend[i]
                .wrapping_add(low.wrapping_shl(low_shift))
                .wrapping_add(high.wrapping_shl(high_shift));
        }
    }

    /// Core kernel: forward-transform both operands, pointwise-multiply
    /// into the bucket lattice, inverse-transform and untwist, then
    /// either *write* (`accumulate = false`) or *add* (`accumulate =
    /// true`) the recombined u64 limbs into `result`.
    ///
    /// Runs entirely on thread-local scratch — no heap allocation is
    /// performed once the scratch has been sized for `N`.
    fn compute_into(&self, lhs: &[u64], rhs: &[u64], result: &mut [u64], accumulate: bool) {
        let n = self.plan.n;
        debug_assert_eq!(lhs.len(), n);
        debug_assert_eq!(rhs.len(), n);
        debug_assert_eq!(result.len(), n);

        FFT_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_size(
                n,
                self.plan
                    .forward_scratch_len
                    .max(self.plan.inverse_scratch_len),
            );
            let FftScratch {
                a_hat,
                b_hat,
                bucket,
                packed,
                fft_scratch,
            } = &mut *scratch;

            self.forward_split_chunks_into(lhs, a_hat, packed, fft_scratch);
            self.forward_split_chunks_into(rhs, b_hat, packed, fft_scratch);

            let [a0, a1, a2, a3] = &*a_hat;
            let [b0, b1, b2, b3] = &*b_hat;
            let [dst0, dst1, dst2, dst3] = bucket;
            for k in 0..n {
                let sym = conjugate_symmetry_index(n, k);
                if k > sym {
                    continue;
                }

                let a0i = a0[k];
                let a1i = a1[k];
                let a2i = a2[k];
                let a3i = a3[k];
                let b0i = b0[k];
                let b1i = b1[k];
                let b2i = b2[k];
                let b3i = b3[k];
                let v0 = cplx_mul(a0i, b0i);
                let v1 = cplx_mul(a0i, b1i) + cplx_mul(a1i, b0i);
                let v2 = cplx_mul(a0i, b2i) + cplx_mul(a1i, b1i) + cplx_mul(a2i, b0i);
                let v3 = cplx_mul(a0i, b3i)
                    + cplx_mul(a1i, b2i)
                    + cplx_mul(a2i, b1i)
                    + cplx_mul(a3i, b0i);

                dst0[k] = v0;
                dst1[k] = v1;
                dst2[k] = v2;
                dst3[k] = v3;
                if sym != k {
                    dst0[sym] = v0.conj();
                    dst1[sym] = v1.conj();
                    dst2[sym] = v2.conj();
                    dst3[sym] = v3.conj();
                }
            }

            if !accumulate {
                result.fill(0);
            }

            let [bucket0, bucket1, bucket2, bucket3] = &*bucket;
            self.inverse_bucket_pair_add_into(bucket0, bucket1, packed, fft_scratch, 0, 16, result);
            self.inverse_bucket_pair_add_into(
                bucket2,
                bucket3,
                packed,
                fft_scratch,
                32,
                48,
                result,
            );
        });
    }
}

fn unpack_paired_forward(packed: &[Cplx], low: &mut [Cplx], high: &mut [Cplx]) {
    let n = packed.len();
    debug_assert_eq!(low.len(), n);
    debug_assert_eq!(high.len(), n);

    for k in 0..n {
        let sym = conjugate_symmetry_index(n, k);
        if k > sym {
            continue;
        }

        let z = packed[k];
        let z_sym_conj = packed[sym].conj();
        let low_k = (z + z_sym_conj) * 0.5;
        let diff = z - z_sym_conj;
        let high_k = Cplx::new(diff.im * 0.5, -diff.re * 0.5);

        low[k] = low_k;
        high[k] = high_k;
        low[sym] = low_k.conj();
        high[sym] = high_k.conj();
    }
}

fn unpack_paired_forward_half(packed: &[Cplx], low: &mut [Cplx], high: &mut [Cplx]) {
    let n = packed.len();
    let half = n >> 1;
    debug_assert_eq!(low.len(), half);
    debug_assert_eq!(high.len(), half);

    let z = packed[0];
    let z_sym_conj = packed[1].conj();
    let low_k = (z + z_sym_conj) * 0.5;
    let diff = z - z_sym_conj;
    low[0] = low_k;
    high[0] = Cplx::new(diff.im * 0.5, -diff.re * 0.5);

    for t in 1..half {
        let k = t + 1;
        let sym = n + 1 - k;
        let z = packed[k];
        let z_sym_conj = packed[sym].conj();
        let low_k = (z + z_sym_conj) * 0.5;
        let diff = z - z_sym_conj;
        low[t] = low_k;
        high[t] = Cplx::new(diff.im * 0.5, -diff.re * 0.5);
    }
}

#[inline(always)]
fn conjugate_symmetry_index(n: usize, k: usize) -> usize {
    (n + 1 - k) & (n - 1)
}

#[inline(always)]
fn add_signed_freq_pair(a: Cplx, b: Cplx, dst: &mut [Cplx], k: usize, sym: usize) {
    let re = a.re * b.re - a.im * b.im;
    let im = a.re * b.im + a.im * b.re;
    dst[k].re += re;
    dst[k].im += im;
    dst[sym].re += re;
    dst[sym].im -= im;
}

#[inline(always)]
fn add_signed_freq_value(a: Cplx, b: Cplx, dst: &mut [Cplx], idx: usize) {
    dst[idx].re += a.re * b.re - a.im * b.im;
    dst[idx].im += a.re * b.im + a.im * b.re;
}

#[inline(always)]
fn pack_half_bucket_pair(low: Cplx, high: Cplx, packed: &mut [Cplx], k: usize, sym: usize) {
    packed[k] = low + Cplx::new(-high.im, high.re);
    packed[sym] = low.conj() + Cplx::new(high.im, high.re);
}

impl NegacyclicMul for FftMul {
    fn mul_assign_u64(&self, lhs: &[u64], rhs: &[u64], result: &mut [u64]) {
        self.compute_into(lhs, rhs, result, false);
    }

    fn fma_assign_u64(&self, lhs: &[u64], rhs: &[u64], result: &mut [u64]) {
        self.compute_into(lhs, rhs, result, true);
    }
}

thread_local! {
    /// One [`FftMul`] plan per polynomial size `N`, per OS thread.  TFHE PBS
    /// calls reuse it so twiddle tables are not rebuilt on every bootstrap.
    static CACHED_FFT_MUL: std::cell::RefCell<std::collections::HashMap<usize, FftMul>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Run `f` with a thread-local cached [`FftMul`] for polynomial length `n`.
///
/// The first call for each distinct `n` on this thread allocates twiddle /
/// ψ tables (`O(n)`); subsequent calls reuse the same [`FftMul`].
///
/// # Panics
///
/// If `n` is not a power of two or is `< 2` ([`FftPlan::new`] requirements).
pub fn with_fft_mul<R>(n: usize, f: impl FnOnce(&FftMul) -> R) -> R {
    assert!(
        n.is_power_of_two() && n >= 2,
        "FFT negacyclic multiplier requires power-of-two polynomial size ≥ 2, got {n}"
    );
    CACHED_FFT_MUL.with(|cell| {
        let mut map = cell.borrow_mut();
        let fft = map.entry(n).or_insert_with(|| FftMul::new(n));
        f(fft)
    })
}

// ---------------------------------------------------------------------------
// Torus Fourier backend for TFHE-style PBS
// ---------------------------------------------------------------------------

/// A compact negacyclic Fourier transform for torus polynomials.
///
/// Unlike [`FftMul`], this backend does not try to perform a bit-exact
/// `Z_2^64` polynomial product.  It follows the TFHE PBS convention: bootstrap
/// key coefficients are represented as `f64` torus values and gadget digits as
/// signed integers.  The transform stores only `N / 2` complex coefficients for
/// a real polynomial of size `N`, which is the natural negacyclic Fourier size.
#[derive(Clone, Debug)]
pub struct TorusFft {
    plan: TorusFftPlan,
}

#[derive(Clone, Debug)]
struct TorusFftPlan {
    poly_n: usize,
    fft: FftPlan,
    twist: Vec<Cplx>,
    twist_inv_scaled: Vec<Cplx>,
}

/// One compact Fourier polynomial for [`TorusFft`].
#[derive(Clone, Debug)]
pub struct TorusFftPoly {
    data: Vec<Cplx>,
}

impl TorusFftPoly {
    /// Allocate a zero-filled compact Fourier polynomial for a real polynomial
    /// of size `poly_n`.
    pub fn zeros(poly_n: usize) -> Self {
        debug_assert!(poly_n.is_power_of_two() && poly_n >= 2);
        debug_assert_eq!(poly_n % 2, 0);
        Self {
            data: vec![Cplx::default(); poly_n >> 1],
        }
    }

    /// Reset all Fourier coefficients to zero.
    pub fn clear(&mut self) {
        for c in &mut self.data {
            *c = Cplx::default();
        }
    }
}

impl TorusFft {
    /// Build a compact negacyclic Fourier backend for real polynomial size
    /// `poly_n`.
    pub fn new(poly_n: usize) -> Self {
        assert!(
            poly_n.is_power_of_two() && poly_n >= 2 && poly_n % 2 == 0,
            "torus FFT requires an even power-of-two polynomial size ≥ 2, got {poly_n}"
        );
        let half = poly_n >> 1;
        let fft = FftPlan::new(half);
        let unit = std::f64::consts::PI / (poly_n as f64);
        let inv_half = 1.0 / (half as f64);
        let mut twist = Vec::with_capacity(half);
        let mut twist_inv_scaled = Vec::with_capacity(half);
        for i in 0..half {
            let theta = unit * (i as f64);
            let (sin, cos) = theta.sin_cos();
            twist.push(Cplx::new(cos, sin));
            twist_inv_scaled.push(Cplx::new(cos * inv_half, -sin * inv_half));
        }
        Self {
            plan: TorusFftPlan {
                poly_n,
                fft,
                twist,
                twist_inv_scaled,
            },
        }
    }

    /// Real polynomial size handled by this backend.
    pub fn n(&self) -> usize {
        self.plan.poly_n
    }

    /// Allocate an empty compact Fourier polynomial for this backend.
    pub fn empty_poly(&self) -> TorusFftPoly {
        TorusFftPoly::zeros(self.plan.poly_n)
    }

    /// Forward-transform a `u64` torus polynomial into compact Fourier form.
    ///
    /// Coefficients are interpreted as centered torus values in `[-1/2, 1/2)`.
    pub fn forward_torus_poly_into(&self, input: &[u64], out: &mut TorusFftPoly) {
        const TORUS_SCALE: f64 = 1.0 / 18_446_744_073_709_551_616.0;
        self.forward_with_scale_into(input, TORUS_SCALE, out);
    }

    /// Forward-transform a signed gadget-digit polynomial into compact Fourier
    /// form.  `input` stores two's-complement signed digits in `u64` slots.
    pub fn forward_signed_integer_poly_into(&self, input: &[u64], out: &mut TorusFftPoly) {
        self.forward_with_scale_into(input, 1.0, out);
    }

    fn forward_with_scale_into(&self, input: &[u64], scale: f64, out: &mut TorusFftPoly) {
        let n = self.plan.poly_n;
        let half = n >> 1;
        debug_assert_eq!(input.len(), n);
        if out.data.len() != half {
            out.data.resize(half, Cplx::default());
        }

        FFT_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_size(
                half,
                self.plan
                    .fft
                    .forward_scratch_len
                    .max(self.plan.fft.inverse_scratch_len),
            );
            for i in 0..half {
                let re = (input[i] as i64 as f64) * scale;
                let im = (input[i + half] as i64 as f64) * scale;
                let psi = self.plan.twist[i];
                scratch.packed[i] = Cplx::new(re * psi.re - im * psi.im, re * psi.im + im * psi.re);
            }
            let FftScratch {
                packed,
                fft_scratch,
                ..
            } = &mut *scratch;
            self.plan.fft.fft(&mut packed[..half], fft_scratch);
            out.data.copy_from_slice(&packed[..half]);
        });
    }

    /// Accumulate `acc += lhs * rhs` pointwise in compact Fourier form.
    pub fn fma_freq_into(&self, lhs: &TorusFftPoly, rhs: &TorusFftPoly, acc: &mut TorusFftPoly) {
        let half = self.plan.poly_n >> 1;
        debug_assert_eq!(lhs.data.len(), half);
        debug_assert_eq!(rhs.data.len(), half);
        if acc.data.len() != half {
            acc.data.resize(half, Cplx::default());
        }
        for ((dst, a), b) in acc.data.iter_mut().zip(&lhs.data).zip(&rhs.data) {
            *dst += cplx_mul(*a, *b);
        }
    }

    /// Inverse-transform `freq`, convert back to `u64` torus coefficients, add
    /// `addend`, and write into `out`.
    ///
    /// The transform consumes `freq` in place; callers should clear or rebuild it
    /// before reuse.
    pub fn inverse_torus_write_with_addend_into(
        &self,
        freq: &mut TorusFftPoly,
        addend: &[u64],
        out: &mut [u64],
    ) {
        let n = self.plan.poly_n;
        let half = n >> 1;
        debug_assert_eq!(freq.data.len(), half);
        debug_assert_eq!(addend.len(), n);
        debug_assert_eq!(out.len(), n);

        FFT_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_size(
                half,
                self.plan
                    .fft
                    .forward_scratch_len
                    .max(self.plan.fft.inverse_scratch_len),
            );
            self.plan
                .fft
                .ifft_unscaled(&mut freq.data, &mut scratch.fft_scratch);
        });

        for i in 0..half {
            let z = freq.data[i];
            let psi = self.plan.twist_inv_scaled[i];
            let re = z.re * psi.re - z.im * psi.im;
            let im = z.re * psi.im + z.im * psi.re;
            out[i] = addend[i].wrapping_add(f64_to_torus_u64(re));
            out[i + half] = addend[i + half].wrapping_add(f64_to_torus_u64(im));
        }
    }
}

#[inline]
fn f64_to_torus_u64(value: f64) -> u64 {
    const TWO64: f64 = 18_446_744_073_709_551_616.0;
    const TWO63: f64 = 9_223_372_036_854_775_808.0;
    let mut scaled = (value - value.round()) * TWO64;
    scaled = scaled.round();
    if scaled >= TWO63 || scaled < -TWO63 {
        i64::MIN as u64
    } else {
        scaled as i64 as u64
    }
}

thread_local! {
    /// One compact torus FFT per polynomial size `N`, per OS thread.
    static CACHED_TORUS_FFT: std::cell::RefCell<std::collections::HashMap<usize, TorusFft>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Run `f` with a thread-local cached compact torus FFT for real polynomial
/// length `n`.
pub fn with_torus_fft<R>(n: usize, f: impl FnOnce(&TorusFft) -> R) -> R {
    assert!(
        n.is_power_of_two() && n >= 2 && n % 2 == 0,
        "compact torus FFT requires an even power-of-two polynomial size ≥ 2, got {n}"
    );
    CACHED_TORUS_FFT.with(|cell| {
        let mut map = cell.borrow_mut();
        let fft = map.entry(n).or_insert_with(|| TorusFft::new(n));
        f(fft)
    })
}

// ---------------------------------------------------------------------------
// Long-lived frequency-domain types: FftPoly + FftBuckets
// ---------------------------------------------------------------------------
//
// The plain `NegacyclicMul` API treats every multiplication as a black box
// that re-FFTs both operands.  For TFHE PBS that is wasteful: the bootstrap
// key never changes, so its frequency-domain image can be computed once at
// keygen and then *read* during every blind rotation.
//
// `FftPoly` stores one polynomial in split-precision frequency domain and
// `FftBuckets` stores a partial pointwise accumulator.  Together they let
// callers (see `tfhe::bootstrap_fft`) drive an external product at roughly
// half the FFT count of the naïve path:
//
//   * forward FFTs:  3 × ((k+1)·level + (k+1))  per cmux  →  caches go away
//                    when both operands need to be FFT'd every call.
//   * inverse FFTs:  4 × (k+1)                  per cmux  (one chain of
//                    bucket IFFTs per output polynomial, instead of one
//                    chain per `(j, r, i)` triple).
//
// On `dev_tfhe_n2048` (lwe_dim=918, level=1, k+1=2) this drops the per-PBS
// FFT count from 44 064 to ~14 688, i.e. a ~3× reduction — within striking
// range of `concrete-fft`'s production numbers.

/// A polynomial transformed into split-precision frequency form.
///
/// Holds the four forward-FFT'd chunks of the input polynomial: each
/// chunk is one of the 16-bit slices of the original `u64`
/// coefficients, ψ-twisted (negacyclic) and then forward-FFT'd.  The
/// representation is bit-exactly reversible by [`FftMul::inverse_combine_into`].
#[derive(Clone, Debug)]
pub struct FftPoly {
    chunks: [Vec<Cplx>; 4],
}

impl FftPoly {
    /// Allocate a zero-initialised `FftPoly` sized for `n` slots per
    /// chunk.  Useful as scratch when calling
    /// [`FftMul::forward_poly_into`] in hot loops.
    pub fn zeros(n: usize) -> Self {
        Self {
            chunks: std::array::from_fn(|_| vec![Cplx::default(); n]),
        }
    }

    /// Polynomial size — the number of coefficients in the original
    /// `u64` polynomial (and the per-chunk vector length).
    pub fn n(&self) -> usize {
        self.chunks[0].len()
    }
}

/// Half-spectrum split-precision frequency form.
///
/// This stores the same conjugate-pair representatives used by
/// [`FftHalfBuckets`].  It is useful for bootstrap-key caches on signed-digit
/// external-product paths, where the missing half of the spectrum is never
/// read in the hot loop.
#[derive(Clone, Debug)]
pub struct FftHalfPoly {
    chunks: [Vec<Cplx>; 4],
}

impl FftHalfPoly {
    /// Allocate a zero-initialised half-spectrum `FftHalfPoly`.
    pub fn zeros(n: usize) -> Self {
        debug_assert!(n.is_power_of_two() && n >= 2);
        Self {
            chunks: std::array::from_fn(|_| vec![Cplx::default(); n >> 1]),
        }
    }
}

/// A gadget-decomposition digit polynomial in frequency domain.
///
/// TFHE external products multiply a small balanced signed digit polynomial by
/// a full-width bootstrap-key polynomial.  The digit coefficients are far
/// smaller than `u64`, so they can be transformed as one signed chunk instead
/// of being split into four 16-bit chunks.
#[derive(Clone, Debug)]
pub struct FftSignedPoly {
    chunk: Vec<Cplx>,
}

impl FftSignedPoly {
    /// Allocate a zero-initialised signed-digit transform buffer.
    pub fn zeros(n: usize) -> Self {
        Self {
            chunk: vec![Cplx::default(); n],
        }
    }

    /// Polynomial size — the number of coefficients in the original digit
    /// polynomial.
    pub fn n(&self) -> usize {
        self.chunk.len()
    }
}

/// Half-spectrum form of a signed digit polynomial.
#[derive(Clone, Debug)]
pub struct FftSignedHalfPoly {
    chunk: Vec<Cplx>,
}

impl FftSignedHalfPoly {
    /// Allocate a zero-initialised half-spectrum signed-digit buffer.
    pub fn zeros(n: usize) -> Self {
        debug_assert!(n.is_power_of_two() && n >= 2);
        Self {
            chunk: vec![Cplx::default(); n >> 1],
        }
    }
}

/// Frequency-domain accumulator for split-precision pointwise products.
///
/// Holds four buckets indexed by `s = k + l` where `(k, l)` are the
/// chunk indices of the operands.  Buckets with `s ≥ 4` would carry a
/// `2⁶⁴` weight and contribute nothing modulo `2⁶⁴`, so they are not
/// stored.
#[derive(Clone, Debug)]
pub struct FftBuckets {
    buckets: [Vec<Cplx>; 4],
}

impl FftBuckets {
    /// Zero-initialised buckets sized for `n` slots each.
    pub fn zeros(n: usize) -> Self {
        Self {
            buckets: std::array::from_fn(|_| vec![Cplx::default(); n]),
        }
    }

    /// Reset every bucket to zero.  No allocation.
    pub fn clear(&mut self) {
        for v in self.buckets.iter_mut() {
            for c in v.iter_mut() {
                *c = Cplx { re: 0.0, im: 0.0 };
            }
        }
    }

    /// Per-bucket length — equal to the polynomial size of the parent
    /// FFT plan.
    pub fn n(&self) -> usize {
        self.buckets[0].len()
    }
}

/// Half-spectrum accumulator for signed-digit external products.
///
/// Each bucket stores exactly one representative from every conjugate pair of
/// the negacyclic FFT spectrum.  It is expanded back to the full spectrum only
/// once, immediately before the inverse FFTs.
#[derive(Clone, Debug)]
pub struct FftHalfBuckets {
    buckets: [Vec<Cplx>; 4],
}

impl FftHalfBuckets {
    /// Zero-initialised half-spectrum buckets sized for `n / 2` slots each.
    pub fn zeros(n: usize) -> Self {
        debug_assert!(n.is_power_of_two() && n >= 2);
        Self {
            buckets: std::array::from_fn(|_| vec![Cplx::default(); n >> 1]),
        }
    }

    /// Reset every half-spectrum bucket to zero.
    pub fn clear(&mut self) {
        for v in self.buckets.iter_mut() {
            for c in v.iter_mut() {
                *c = Cplx { re: 0.0, im: 0.0 };
            }
        }
    }
}

impl FftMul {
    /// Allocate an empty (zero) `FftPoly` sized for this plan.  Useful
    /// as scratch with [`Self::forward_poly_into`].
    pub fn empty_poly(&self) -> FftPoly {
        FftPoly::zeros(self.plan.n)
    }

    /// Allocate an empty signed-digit frequency polynomial.
    pub fn empty_signed_poly(&self) -> FftSignedPoly {
        FftSignedPoly::zeros(self.plan.n)
    }

    /// Allocate an empty half-spectrum signed-digit frequency polynomial.
    pub fn empty_signed_half_poly(&self) -> FftSignedHalfPoly {
        FftSignedHalfPoly::zeros(self.plan.n)
    }

    /// Forward-transform `input` into split-precision frequency form.
    ///
    /// Allocates a fresh `FftPoly`.  In hot loops, prefer
    /// [`Self::forward_poly_into`].
    pub fn forward_poly(&self, input: &[u64]) -> FftPoly {
        let mut out = FftPoly::zeros(self.plan.n);
        self.forward_poly_into(input, &mut out);
        out
    }

    /// Forward-transform `input` and keep only the half-spectrum
    /// representatives.
    pub fn forward_poly_half(&self, input: &[u64]) -> FftHalfPoly {
        let mut out = FftHalfPoly::zeros(self.plan.n);
        self.forward_poly_half_into(input, &mut out);
        out
    }

    /// Like [`Self::forward_poly`] but writes into pre-allocated
    /// scratch.  Use this in hot loops to avoid the per-call `Vec`
    /// allocation.
    pub fn forward_poly_into(&self, input: &[u64], out: &mut FftPoly) {
        let n = self.plan.n;
        debug_assert_eq!(input.len(), n);
        for chunk in out.chunks.iter_mut() {
            if chunk.len() != n {
                chunk.resize(n, Cplx::default());
            }
        }
        FFT_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_size(
                n,
                self.plan
                    .forward_scratch_len
                    .max(self.plan.inverse_scratch_len),
            );
            let FftScratch {
                packed,
                fft_scratch,
                ..
            } = &mut *scratch;
            self.forward_chunk_pair_into(input, 0, 16, packed, &mut out.chunks, 0, fft_scratch);
            self.forward_chunk_pair_into(input, 32, 48, packed, &mut out.chunks, 2, fft_scratch);
        });
    }

    /// Like [`Self::forward_poly_half`] but writes into pre-allocated scratch.
    pub fn forward_poly_half_into(&self, input: &[u64], out: &mut FftHalfPoly) {
        let n = self.plan.n;
        let half = n >> 1;
        debug_assert_eq!(input.len(), n);
        for chunk in out.chunks.iter_mut() {
            if chunk.len() != half {
                chunk.resize(half, Cplx::default());
            }
        }
        FFT_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_size(
                n,
                self.plan
                    .forward_scratch_len
                    .max(self.plan.inverse_scratch_len),
            );
            let FftScratch {
                packed,
                fft_scratch,
                ..
            } = &mut *scratch;
            self.forward_chunk_pair_half_into(
                input,
                0,
                16,
                packed,
                &mut out.chunks,
                0,
                fft_scratch,
            );
            self.forward_chunk_pair_half_into(
                input,
                32,
                48,
                packed,
                &mut out.chunks,
                2,
                fft_scratch,
            );
        });
    }

    /// Forward-transform a balanced signed gadget digit polynomial.
    ///
    /// `input` stores two's-complement signed digits in `u64` slots.  This
    /// path is exact for the shortint PBS parameters where the digit magnitude
    /// is small enough to stay well inside the `f64` mantissa after
    /// convolution with one 16-bit bootstrap-key chunk.
    pub fn forward_signed_poly_into(&self, input: &[u64], out: &mut FftSignedPoly) {
        let n = self.plan.n;
        debug_assert_eq!(input.len(), n);
        if out.chunk.len() != n {
            out.chunk.resize(n, Cplx::default());
        }
        for i in 0..n {
            let digit = input[i] as i64 as f64;
            let psi = self.plan.psi_pow[i];
            out.chunk[i] = Cplx::new(digit * psi.re, digit * psi.im);
        }
        FFT_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_size(
                n,
                self.plan
                    .forward_scratch_len
                    .max(self.plan.inverse_scratch_len),
            );
            self.plan.fft(&mut out.chunk, &mut scratch.fft_scratch);
        });
    }

    /// Forward-transform two balanced signed digit polynomials using one
    /// packed complex FFT.
    pub fn forward_signed_poly_pair_into(
        &self,
        input0: &[u64],
        input1: &[u64],
        out0: &mut FftSignedPoly,
        out1: &mut FftSignedPoly,
    ) {
        let n = self.plan.n;
        debug_assert_eq!(input0.len(), n);
        debug_assert_eq!(input1.len(), n);
        if out0.chunk.len() != n {
            out0.chunk.resize(n, Cplx::default());
        }
        if out1.chunk.len() != n {
            out1.chunk.resize(n, Cplx::default());
        }

        FFT_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_size(
                n,
                self.plan
                    .forward_scratch_len
                    .max(self.plan.inverse_scratch_len),
            );
            for i in 0..n {
                let low = input0[i] as i64 as f64;
                let high = input1[i] as i64 as f64;
                let psi = self.plan.psi_pow[i];
                scratch.packed[i] =
                    Cplx::new(low * psi.re - high * psi.im, low * psi.im + high * psi.re);
            }
            let FftScratch {
                packed,
                fft_scratch,
                ..
            } = &mut *scratch;
            self.plan.fft(packed, fft_scratch);
            unpack_paired_forward(packed, &mut out0.chunk, &mut out1.chunk);
        });
    }

    /// Forward-transform two balanced signed digit polynomials, returning only
    /// the half-spectrum representatives used by signed external products.
    pub fn forward_signed_poly_pair_half_into(
        &self,
        input0: &[u64],
        input1: &[u64],
        out0: &mut FftSignedHalfPoly,
        out1: &mut FftSignedHalfPoly,
    ) {
        let n = self.plan.n;
        let half = n >> 1;
        debug_assert_eq!(input0.len(), n);
        debug_assert_eq!(input1.len(), n);
        if out0.chunk.len() != half {
            out0.chunk.resize(half, Cplx::default());
        }
        if out1.chunk.len() != half {
            out1.chunk.resize(half, Cplx::default());
        }

        FFT_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_size(
                n,
                self.plan
                    .forward_scratch_len
                    .max(self.plan.inverse_scratch_len),
            );
            for i in 0..n {
                let low = input0[i] as i64 as f64;
                let high = input1[i] as i64 as f64;
                let psi = self.plan.psi_pow[i];
                scratch.packed[i] =
                    Cplx::new(low * psi.re - high * psi.im, low * psi.im + high * psi.re);
            }
            let FftScratch {
                packed,
                fft_scratch,
                ..
            } = &mut *scratch;
            self.plan.fft(packed, fft_scratch);
            unpack_paired_forward_half(packed, &mut out0.chunk, &mut out1.chunk);
        });
    }

    /// Allocate a zero-initialised frequency-domain accumulator sized
    /// for this multiplier's plan.
    pub fn buckets_zero(&self) -> FftBuckets {
        FftBuckets::zeros(self.plan.n)
    }

    /// Allocate a zero-initialised half-spectrum accumulator.
    pub fn half_buckets_zero(&self) -> FftHalfBuckets {
        FftHalfBuckets::zeros(self.plan.n)
    }

    /// `acc += lhs * rhs` in frequency domain.  No IFFT, no
    /// recombine — pointwise multiplication is the *only* arithmetic.
    ///
    /// This is the inner kernel of the FFT-cached external product:
    /// the bootstrap key contributes the `rhs` (already in
    /// frequency domain), and the gadget-decomposed accumulator
    /// supplies `lhs`.  Many calls accumulate into the same `acc` and
    /// share a single [`Self::inverse_combine_into`] at the end.
    pub fn fma_freq_into(&self, lhs: &FftPoly, rhs: &FftPoly, acc: &mut FftBuckets) {
        let n = self.plan.n;
        debug_assert_eq!(lhs.chunks[0].len(), n);
        debug_assert_eq!(rhs.chunks[0].len(), n);
        debug_assert_eq!(acc.buckets[0].len(), n);
        let [a0, a1, a2, a3] = &lhs.chunks;
        let [b0, b1, b2, b3] = &rhs.chunks;
        let [dst0, dst1, dst2, dst3] = &mut acc.buckets;
        for k in 0..n {
            let sym = conjugate_symmetry_index(n, k);
            if k > sym {
                continue;
            }

            let a0i = a0[k];
            let a1i = a1[k];
            let a2i = a2[k];
            let a3i = a3[k];
            let b0i = b0[k];
            let b1i = b1[k];
            let b2i = b2[k];
            let b3i = b3[k];
            let v0 = cplx_mul(a0i, b0i);
            let v1 = cplx_mul(a0i, b1i) + cplx_mul(a1i, b0i);
            let v2 = cplx_mul(a0i, b2i) + cplx_mul(a1i, b1i) + cplx_mul(a2i, b0i);
            let v3 =
                cplx_mul(a0i, b3i) + cplx_mul(a1i, b2i) + cplx_mul(a2i, b1i) + cplx_mul(a3i, b0i);

            dst0[k] += v0;
            dst1[k] += v1;
            dst2[k] += v2;
            dst3[k] += v3;
            if sym != k {
                dst0[sym] += v0.conj();
                dst1[sym] += v1.conj();
                dst2[sym] += v2.conj();
                dst3[sym] += v3.conj();
            }
        }
    }

    /// `acc += signed_lhs * rhs` in frequency domain, where `signed_lhs` is a
    /// single balanced gadget digit chunk and `rhs` is a full 64-bit
    /// split-precision polynomial.
    pub fn fma_signed_lhs_freq_into(
        &self,
        lhs: &FftSignedPoly,
        rhs: &FftPoly,
        acc: &mut FftBuckets,
    ) {
        let n = self.plan.n;
        debug_assert_eq!(lhs.chunk.len(), n);
        debug_assert_eq!(rhs.chunks[0].len(), n);
        debug_assert_eq!(acc.buckets[0].len(), n);

        let a = &lhs.chunk;
        let [b0, b1, b2, b3] = &rhs.chunks;
        let [dst0, dst1, dst2, dst3] = &mut acc.buckets;
        add_signed_freq_pair(a[0], b0[0], dst0, 0, 1);
        add_signed_freq_pair(a[0], b1[0], dst1, 0, 1);
        add_signed_freq_pair(a[0], b2[0], dst2, 0, 1);
        add_signed_freq_pair(a[0], b3[0], dst3, 0, 1);

        for k in 2..=(n >> 1) {
            let sym = n + 1 - k;
            let ai = a[k];
            add_signed_freq_pair(ai, b0[k], dst0, k, sym);
            add_signed_freq_pair(ai, b1[k], dst1, k, sym);
            add_signed_freq_pair(ai, b2[k], dst2, k, sym);
            add_signed_freq_pair(ai, b3[k], dst3, k, sym);
        }
    }

    /// Half-spectrum variant of [`Self::fma_signed_lhs_freq_into`].
    pub fn fma_signed_lhs_freq_half_into(
        &self,
        lhs: &FftSignedPoly,
        rhs: &FftPoly,
        acc: &mut FftHalfBuckets,
    ) {
        let n = self.plan.n;
        let half = n >> 1;
        debug_assert_eq!(lhs.chunk.len(), n);
        debug_assert_eq!(rhs.chunks[0].len(), n);
        debug_assert_eq!(acc.buckets[0].len(), half);

        let a = &lhs.chunk;
        let [b0, b1, b2, b3] = &rhs.chunks;
        let [dst0, dst1, dst2, dst3] = &mut acc.buckets;
        add_signed_freq_value(a[0], b0[0], dst0, 0);
        add_signed_freq_value(a[0], b1[0], dst1, 0);
        add_signed_freq_value(a[0], b2[0], dst2, 0);
        add_signed_freq_value(a[0], b3[0], dst3, 0);

        for t in 1..half {
            let k = t + 1;
            let ai = a[k];
            add_signed_freq_value(ai, b0[k], dst0, t);
            add_signed_freq_value(ai, b1[k], dst1, t);
            add_signed_freq_value(ai, b2[k], dst2, t);
            add_signed_freq_value(ai, b3[k], dst3, t);
        }
    }

    /// Half-spectrum variant for signed digit inputs that were already stored
    /// in half-spectrum form.
    pub fn fma_signed_half_lhs_freq_half_into(
        &self,
        lhs: &FftSignedHalfPoly,
        rhs: &FftPoly,
        acc: &mut FftHalfBuckets,
    ) {
        let n = self.plan.n;
        let half = n >> 1;
        debug_assert_eq!(lhs.chunk.len(), half);
        debug_assert_eq!(rhs.chunks[0].len(), n);
        debug_assert_eq!(acc.buckets[0].len(), half);

        let a = &lhs.chunk;
        let [b0, b1, b2, b3] = &rhs.chunks;
        let [dst0, dst1, dst2, dst3] = &mut acc.buckets;
        add_signed_freq_value(a[0], b0[0], dst0, 0);
        add_signed_freq_value(a[0], b1[0], dst1, 0);
        add_signed_freq_value(a[0], b2[0], dst2, 0);
        add_signed_freq_value(a[0], b3[0], dst3, 0);

        for t in 1..half {
            let k = t + 1;
            let ai = a[t];
            add_signed_freq_value(ai, b0[k], dst0, t);
            add_signed_freq_value(ai, b1[k], dst1, t);
            add_signed_freq_value(ai, b2[k], dst2, t);
            add_signed_freq_value(ai, b3[k], dst3, t);
        }
    }

    /// Half-spectrum FMA where both the signed digit and the cached BSK
    /// polynomial are stored as representatives only.
    pub fn fma_signed_half_lhs_half_rhs_freq_half_into(
        &self,
        lhs: &FftSignedHalfPoly,
        rhs: &FftHalfPoly,
        acc: &mut FftHalfBuckets,
    ) {
        let half = self.plan.n >> 1;
        debug_assert_eq!(lhs.chunk.len(), half);
        debug_assert_eq!(rhs.chunks[0].len(), half);
        debug_assert_eq!(acc.buckets[0].len(), half);

        let a = &lhs.chunk;
        let [b0, b1, b2, b3] = &rhs.chunks;
        let [dst0, dst1, dst2, dst3] = &mut acc.buckets;
        for t in 0..half {
            let ai = a[t];
            add_signed_freq_value(ai, b0[t], dst0, t);
            add_signed_freq_value(ai, b1[t], dst1, t);
            add_signed_freq_value(ai, b2[t], dst2, t);
            add_signed_freq_value(ai, b3[t], dst3, t);
        }
    }

    /// IFFT each bucket of `acc`, untwist, round and recombine; *add*
    /// the resulting `u64` polynomial into `out`.
    ///
    /// `acc` is mutated in place (the IFFT consumes it), so callers
    /// that want to reuse the buckets for another accumulation must
    /// call [`FftBuckets::clear`] afterwards and rebuild via fresh
    /// [`Self::fma_freq_into`] calls.
    pub fn inverse_combine_into(&self, acc: &mut FftBuckets, out: &mut [u64]) {
        let n = self.plan.n;
        debug_assert_eq!(out.len(), n);
        debug_assert_eq!(acc.buckets[0].len(), n);
        FFT_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_size(
                n,
                self.plan
                    .forward_scratch_len
                    .max(self.plan.inverse_scratch_len),
            );
            let FftScratch {
                packed,
                fft_scratch,
                ..
            } = &mut *scratch;
            let [bucket0, bucket1, bucket2, bucket3] = &acc.buckets;
            self.inverse_bucket_pair_add_into(bucket0, bucket1, packed, fft_scratch, 0, 16, out);
            self.inverse_bucket_pair_add_into(bucket2, bucket3, packed, fft_scratch, 32, 48, out);
        });
    }

    /// IFFT each bucket of `acc`, untwist, round and recombine; overwrite
    /// `out` with the resulting `u64` polynomial.
    pub fn inverse_combine_write_into(&self, acc: &mut FftBuckets, out: &mut [u64]) {
        let n = self.plan.n;
        debug_assert_eq!(out.len(), n);
        debug_assert_eq!(acc.buckets[0].len(), n);
        FFT_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_size(
                n,
                self.plan
                    .forward_scratch_len
                    .max(self.plan.inverse_scratch_len),
            );
            let FftScratch {
                packed,
                fft_scratch,
                ..
            } = &mut *scratch;
            let [bucket0, bucket1, bucket2, bucket3] = &acc.buckets;
            self.inverse_bucket_pair_write_into(bucket0, bucket1, packed, fft_scratch, 0, 16, out);
            self.inverse_bucket_pair_add_into(bucket2, bucket3, packed, fft_scratch, 32, 48, out);
        });
    }

    /// Like [`Self::inverse_combine_write_into`], but adds `addend` while
    /// writing the first bucket pair.  This is useful for CMUX, whose output is
    /// `accumulator + external_product`.
    pub fn inverse_combine_write_with_addend_into(
        &self,
        acc: &mut FftBuckets,
        addend: &[u64],
        out: &mut [u64],
    ) {
        let n = self.plan.n;
        debug_assert_eq!(out.len(), n);
        debug_assert_eq!(addend.len(), n);
        debug_assert_eq!(acc.buckets[0].len(), n);
        FFT_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_size(
                n,
                self.plan
                    .forward_scratch_len
                    .max(self.plan.inverse_scratch_len),
            );
            let FftScratch {
                packed,
                fft_scratch,
                ..
            } = &mut *scratch;
            let [bucket0, bucket1, bucket2, bucket3] = &acc.buckets;
            self.inverse_bucket_pair_write_with_addend_into(
                bucket0,
                bucket1,
                packed,
                fft_scratch,
                0,
                16,
                addend,
                out,
            );
            self.inverse_bucket_pair_add_into(bucket2, bucket3, packed, fft_scratch, 32, 48, out);
        });
    }

    /// Expand a half-spectrum accumulator, run the inverse FFTs, and add the
    /// recombined polynomial into `out`.
    pub fn inverse_combine_half_into(&self, acc: &FftHalfBuckets, out: &mut [u64]) {
        let n = self.plan.n;
        debug_assert_eq!(out.len(), n);
        debug_assert_eq!(acc.buckets[0].len(), n >> 1);
        FFT_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_size(
                n,
                self.plan
                    .forward_scratch_len
                    .max(self.plan.inverse_scratch_len),
            );
            let FftScratch {
                packed,
                fft_scratch,
                ..
            } = &mut *scratch;
            let [bucket0, bucket1, bucket2, bucket3] = &acc.buckets;
            self.inverse_half_bucket_pair_add_into(
                bucket0,
                bucket1,
                packed,
                fft_scratch,
                0,
                16,
                out,
            );
            self.inverse_half_bucket_pair_add_into(
                bucket2,
                bucket3,
                packed,
                fft_scratch,
                32,
                48,
                out,
            );
        });
    }

    /// Expand a half-spectrum accumulator, run the inverse FFTs, and
    /// overwrite `out` with the recombined polynomial.
    pub fn inverse_combine_half_write_into(&self, acc: &FftHalfBuckets, out: &mut [u64]) {
        let n = self.plan.n;
        debug_assert_eq!(out.len(), n);
        debug_assert_eq!(acc.buckets[0].len(), n >> 1);
        FFT_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_size(
                n,
                self.plan
                    .forward_scratch_len
                    .max(self.plan.inverse_scratch_len),
            );
            let FftScratch {
                packed,
                fft_scratch,
                ..
            } = &mut *scratch;
            let [bucket0, bucket1, bucket2, bucket3] = &acc.buckets;
            self.inverse_half_bucket_pair_write_into(
                bucket0,
                bucket1,
                packed,
                fft_scratch,
                0,
                16,
                out,
            );
            self.inverse_half_bucket_pair_add_into(
                bucket2,
                bucket3,
                packed,
                fft_scratch,
                32,
                48,
                out,
            );
        });
    }

    /// Like [`Self::inverse_combine_half_write_into`], but adds `addend` while
    /// writing the first bucket pair.
    pub fn inverse_combine_half_write_with_addend_into(
        &self,
        acc: &FftHalfBuckets,
        addend: &[u64],
        out: &mut [u64],
    ) {
        let n = self.plan.n;
        debug_assert_eq!(out.len(), n);
        debug_assert_eq!(addend.len(), n);
        debug_assert_eq!(acc.buckets[0].len(), n >> 1);
        FFT_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure_size(
                n,
                self.plan
                    .forward_scratch_len
                    .max(self.plan.inverse_scratch_len),
            );
            let FftScratch {
                packed,
                fft_scratch,
                ..
            } = &mut *scratch;
            let [bucket0, bucket1, bucket2, bucket3] = &acc.buckets;
            self.inverse_half_bucket_pair_write_with_addend_into(
                bucket0,
                bucket1,
                packed,
                fft_scratch,
                0,
                16,
                addend,
                out,
            );
            self.inverse_half_bucket_pair_add_into(
                bucket2,
                bucket3,
                packed,
                fft_scratch,
                32,
                48,
                out,
            );
        });
    }
}

/// In-place `result += lhs * rhs` over `Z_2^64[X] / (X^N + 1)`.
///
/// Negacyclicity:  when the polynomial multiplication wraps past degree `N`,
/// the coefficients are *negated* (equivalent to `X^N = -1`).  We achieve
/// this by `wrapping_sub`-ing the wrapped term instead of `wrapping_add`.
pub fn negacyclic_fma_u64(lhs: &[u64], rhs: &[u64], result: &mut [u64]) {
    let n = lhs.len();
    debug_assert_eq!(n, rhs.len());
    debug_assert_eq!(n, result.len());

    for i in 0..n {
        let a = lhs[i];
        if a == 0 {
            continue;
        }
        for j in 0..n {
            let b = rhs[j];
            if b == 0 {
                continue;
            }
            let prod = a.wrapping_mul(b);
            let k = i + j;
            if k < n {
                result[k] = result[k].wrapping_add(prod);
            } else {
                let kk = k - n;
                result[kk] = result[kk].wrapping_sub(prod);
            }
        }
    }
}

/// Convenience wrapper: a fresh negacyclic product of two polynomials.
pub fn negacyclic_mul_u64(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
    let mut out = vec![0u64; lhs.len()];
    SchoolbookMul.mul_assign_u64(lhs, rhs, &mut out);
    out
}

/// Coefficient-wise `dst += src`.  Equivalent to addition in
/// `Z_2^64[X] / (X^N + 1)`.
#[inline]
pub fn add_assign_u64(dst: &mut [u64], src: &[u64]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = d.wrapping_add(*s);
    }
}

/// Coefficient-wise `dst -= src`.
#[inline]
pub fn sub_assign_u64(dst: &mut [u64], src: &[u64]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = d.wrapping_sub(*s);
    }
}

/// Coefficient-wise `dst = -dst`.
#[inline]
pub fn negate_assign_u64(dst: &mut [u64]) {
    for d in dst.iter_mut() {
        *d = d.wrapping_neg();
    }
}

/// Multiply every coefficient by a (wrapping) scalar.
#[inline]
pub fn scalar_mul_assign_u64(dst: &mut [u64], scalar: u64) {
    for d in dst.iter_mut() {
        *d = d.wrapping_mul(scalar);
    }
}

/// Negacyclic monomial multiplication: `out = poly * X^m mod (X^N + 1)`.
///
/// This is the building block of TFHE's blind rotation: rotating a GLWE
/// plaintext polynomial by `b - <a, s>` exponent positions, with a sign flip
/// every time the exponent wraps past `N`.
pub fn monomial_mul_u64(poly: &[u64], exponent: usize, out: &mut [u64]) {
    let n = poly.len();
    debug_assert_eq!(n, out.len());
    debug_assert!(n.is_power_of_two() && !poly.is_empty());

    let two_n = n << 1;
    let m = exponent & (two_n - 1);
    let (shift, negate) = if m < n { (m, false) } else { (m - n, true) };

    if !negate {
        // out[k] = poly[k - shift] for k in shift..n
        // out[k] = -poly[k - shift + n] for k in 0..shift
        for k in 0..n {
            let v = if k >= shift {
                poly[k - shift]
            } else {
                poly[k + n - shift].wrapping_neg()
            };
            out[k] = v;
        }
    } else {
        // X^(N + s) = -X^s mod (X^N + 1)
        for k in 0..n {
            let v = if k >= shift {
                poly[k - shift].wrapping_neg()
            } else {
                poly[k + n - shift]
            };
            out[k] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mul_naive(a: &[u64], b: &[u64]) -> Vec<u64> {
        let n = a.len();
        let mut out = vec![0u64; n];
        for i in 0..n {
            for j in 0..n {
                let prod = a[i].wrapping_mul(b[j]);
                let k = i + j;
                if k < n {
                    out[k] = out[k].wrapping_add(prod);
                } else {
                    out[k - n] = out[k - n].wrapping_sub(prod);
                }
            }
        }
        out
    }

    #[test]
    fn schoolbook_matches_reference_n4() {
        let a: Vec<u64> = (1..=4)
            .map(|i| (i as u64).wrapping_mul(0x1000_0000_0000_0001))
            .collect();
        let b: Vec<u64> = (5..=8)
            .map(|i| (i as u64).wrapping_mul(0x9000_0000_0000_0001))
            .collect();
        let expected = mul_naive(&a, &b);
        let mut out = vec![0u64; 4];
        SchoolbookMul.mul_assign_u64(&a, &b, &mut out);
        assert_eq!(out, expected);
    }

    #[test]
    fn fma_accumulates_correctly() {
        let n = 8;
        let a: Vec<u64> = (0..n).map(|i| i as u64 + 1).collect();
        let b: Vec<u64> = (0..n).map(|i| (i as u64 + 3).wrapping_mul(7)).collect();

        let mut acc = vec![123u64; n];
        let mut expected = acc.clone();
        let prod = mul_naive(&a, &b);
        for (e, p) in expected.iter_mut().zip(prod.iter()) {
            *e = e.wrapping_add(*p);
        }
        SchoolbookMul.fma_assign_u64(&a, &b, &mut acc);
        assert_eq!(acc, expected);
    }

    #[test]
    fn monomial_mul_wraps_with_sign_flip() {
        let n = 4usize;
        let p: Vec<u64> = vec![1, 2, 3, 4];
        let mut out = vec![0u64; n];

        // X^0 should be the identity.
        monomial_mul_u64(&p, 0, &mut out);
        assert_eq!(out, p);

        // X^1 should rotate and negate the wrapped coefficient.
        monomial_mul_u64(&p, 1, &mut out);
        // expected: p * X = -p[3] + p[0]X + p[1]X^2 + p[2]X^3
        assert_eq!(out, vec![(4u64).wrapping_neg(), 1, 2, 3]);

        // X^N should give -p (full negation).
        monomial_mul_u64(&p, n, &mut out);
        let expected: Vec<u64> = p.iter().map(|x| x.wrapping_neg()).collect();
        assert_eq!(out, expected);

        // X^(N+1) should rotate AND negate twice -> X * (-p).
        monomial_mul_u64(&p, n + 1, &mut out);
        let expected: Vec<u64> = vec![
            4,
            1u64.wrapping_neg(),
            2u64.wrapping_neg(),
            3u64.wrapping_neg(),
        ];
        assert_eq!(out, expected);
    }

    fn fill_random_u64s(state: &mut u64, dst: &mut [u64]) {
        for slot in dst.iter_mut() {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *slot = *state;
        }
    }

    #[test]
    fn fft_then_ifft_is_identity_on_real_signal() {
        let n = 64;
        let plan = FftPlan::new(n);
        let mut scratch = Vec::new();
        let mut data: Vec<Cplx> = (0..n)
            .map(|i| Cplx::new((i as f64) - (n as f64) / 2.0, 0.0))
            .collect();
        let original = data.clone();

        plan.fft(&mut data, &mut scratch);
        plan.ifft(&mut data, &mut scratch);

        for (got, expected) in data.iter().zip(original.iter()) {
            assert!(
                (got.re - expected.re).abs() < 1e-8,
                "FFT/IFFT round-trip drift: got {:?} expected {:?}",
                got,
                expected
            );
            assert!(
                got.im.abs() < 1e-8,
                "FFT/IFFT real input → imaginary output: {:?}",
                got
            );
        }
    }

    #[test]
    fn fft_of_unit_pulse_is_constant_one() {
        let n = 16;
        let plan = FftPlan::new(n);
        let mut scratch = Vec::new();
        let mut data: Vec<Cplx> = vec![Cplx::default(); n];
        data[0].re = 1.0;
        plan.fft(&mut data, &mut scratch);
        for c in data.iter() {
            assert!((c.re - 1.0).abs() < 1e-12);
            assert!(c.im.abs() < 1e-12);
        }
    }

    #[test]
    fn fft_mul_matches_schoolbook_n64_random() {
        let n = 64;
        let backend = FftMul::new(n);
        let mut state: u64 = 0xCAFEBABE_DEADBEEF;
        let mut a = vec![0u64; n];
        let mut b = vec![0u64; n];
        fill_random_u64s(&mut state, &mut a);
        fill_random_u64s(&mut state, &mut b);

        let mut sch = vec![0u64; n];
        let mut fft = vec![0u64; n];
        SchoolbookMul.mul_assign_u64(&a, &b, &mut sch);
        backend.mul_assign_u64(&a, &b, &mut fft);
        assert_eq!(sch, fft);
    }

    #[test]
    fn fft_mul_matches_schoolbook_n2048_random() {
        // Production-shape (dev_tfhe_n2048) negacyclic mul.  Schoolbook
        // is `O(N²)` ≈ 4M ops, fast enough for a single random test.
        let n = 2048;
        let backend = FftMul::new(n);
        let mut state: u64 = 1;
        let mut a = vec![0u64; n];
        let mut b = vec![0u64; n];
        fill_random_u64s(&mut state, &mut a);
        fill_random_u64s(&mut state, &mut b);

        let mut sch = vec![0u64; n];
        let mut fft = vec![0u64; n];
        SchoolbookMul.mul_assign_u64(&a, &b, &mut sch);
        backend.mul_assign_u64(&a, &b, &mut fft);
        assert_eq!(sch, fft, "FFT diverges from schoolbook at production scale");
    }

    #[test]
    fn fft_mul_matches_karatsuba_random() {
        // Cross-check the two production-grade backends against each
        // other on a non-trivial size that exercises Karatsuba's
        // recursion plus the FFT split-precision recombination.
        let n = 256;
        let backend = FftMul::new(n);
        let mut state: u64 = 0xDEADBEEF_C0FFEE;
        let mut a = vec![0u64; n];
        let mut b = vec![0u64; n];
        fill_random_u64s(&mut state, &mut a);
        fill_random_u64s(&mut state, &mut b);

        let mut kar = vec![0u64; n];
        let mut fft = vec![0u64; n];
        KaratsubaMul.mul_assign_u64(&a, &b, &mut kar);
        backend.mul_assign_u64(&a, &b, &mut fft);
        assert_eq!(kar, fft);
    }

    #[test]
    fn with_fft_mul_matches_standalone_new() {
        let n = 32;
        let a: Vec<u64> = (0..n).map(|i| i as u64 * 0x1_0000_0000_0003).collect();
        let b: Vec<u64> = (0..n)
            .map(|i| (n - i) as u64 * 0x9_F000_0000_0001)
            .collect();
        let standalone = FftMul::new(n);
        let mut expected = vec![0u64; n];
        let mut cached = vec![0u64; n];
        standalone.mul_assign_u64(&a, &b, &mut expected);
        with_fft_mul(n, |fft| fft.mul_assign_u64(&a, &b, &mut cached));
        assert_eq!(cached, expected);
    }

    #[test]
    fn fft_mul_handles_signed_negacyclic_wraparound() {
        // Pick coefficients that span all 4 chunks (each u64 has bits
        // set in [0,16), [16,32), [32,48), [48,64)) so the
        // recombination path exercises every shift bucket.
        let n = 32;
        let backend = FftMul::new(n);
        let a: Vec<u64> = (0..n)
            .map(|i| (i as u64).wrapping_mul(0xABCD_1234_5678_9101))
            .collect();
        let b: Vec<u64> = (0..n)
            .map(|i| (n as u64 - i as u64).wrapping_mul(0x9876_5432_FEDC_BA01))
            .collect();
        let mut sch = vec![0u64; n];
        let mut fft = vec![0u64; n];
        SchoolbookMul.mul_assign_u64(&a, &b, &mut sch);
        backend.mul_assign_u64(&a, &b, &mut fft);
        assert_eq!(sch, fft);
    }

    #[test]
    fn fft_fma_matches_reference() {
        let n = 128;
        let backend = FftMul::new(n);
        let mut state: u64 = 42;
        let mut a = vec![0u64; n];
        let mut b = vec![0u64; n];
        fill_random_u64s(&mut state, &mut a);
        fill_random_u64s(&mut state, &mut b);
        let mut acc_sch = vec![0xDEADBEEF_u64; n];
        let mut acc_fft = acc_sch.clone();
        SchoolbookMul.fma_assign_u64(&a, &b, &mut acc_sch);
        backend.fma_assign_u64(&a, &b, &mut acc_fft);
        assert_eq!(acc_sch, acc_fft);
    }

    #[test]
    fn karatsuba_matches_schoolbook_n4() {
        // Below the Karatsuba threshold — should fall through to schoolbook.
        let a: Vec<u64> = (1..=4)
            .map(|i| (i as u64).wrapping_mul(0x1000_0000_0000_0001))
            .collect();
        let b: Vec<u64> = (5..=8)
            .map(|i| (i as u64).wrapping_mul(0x9000_0000_0000_0001))
            .collect();
        let mut sch = vec![0u64; 4];
        let mut kar = vec![0u64; 4];
        SchoolbookMul.mul_assign_u64(&a, &b, &mut sch);
        KaratsubaMul.mul_assign_u64(&a, &b, &mut kar);
        assert_eq!(sch, kar);
    }

    #[test]
    fn karatsuba_matches_schoolbook_n_above_threshold() {
        // 64 > KARATSUBA_BASE = 32, so this exercises the recursive path.
        let n = 64usize;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut state: u64 = 0xCAFEBABE_DEADBEEF;
        for _ in 0..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            a.push(state);
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            b.push(state);
        }
        let mut sch = vec![0u64; n];
        let mut kar = vec![0u64; n];
        SchoolbookMul.mul_assign_u64(&a, &b, &mut sch);
        KaratsubaMul.mul_assign_u64(&a, &b, &mut kar);
        assert_eq!(sch, kar);
    }

    #[test]
    fn karatsuba_matches_schoolbook_n_2048() {
        // Production-shape polynomial size.  This test would be O(N^2) on
        // schoolbook (≈ 4M ops) so it stays cheap in debug.
        let n = 2048usize;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut state: u64 = 1;
        for _ in 0..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            a.push(state);
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            b.push(state);
        }
        let mut sch = vec![0u64; n];
        let mut kar = vec![0u64; n];
        SchoolbookMul.mul_assign_u64(&a, &b, &mut sch);
        KaratsubaMul.mul_assign_u64(&a, &b, &mut kar);
        assert_eq!(sch, kar);
    }

    #[test]
    fn karatsuba_fma_accumulates_correctly() {
        let n = 128usize;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut state: u64 = 42;
        for _ in 0..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            a.push(state);
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            b.push(state);
        }
        let mut acc_sch = vec![0xDEAD_BEEF_u64; n];
        let mut acc_kar = acc_sch.clone();
        SchoolbookMul.fma_assign_u64(&a, &b, &mut acc_sch);
        KaratsubaMul.fma_assign_u64(&a, &b, &mut acc_kar);
        assert_eq!(acc_sch, acc_kar);
    }

    #[test]
    fn add_sub_negate_scalar() {
        let mut a = vec![1u64, 2, 3, 4];
        let b = vec![10u64, 20, 30, 40];
        add_assign_u64(&mut a, &b);
        assert_eq!(a, vec![11, 22, 33, 44]);
        sub_assign_u64(&mut a, &b);
        assert_eq!(a, vec![1, 2, 3, 4]);
        scalar_mul_assign_u64(&mut a, 3);
        assert_eq!(a, vec![3, 6, 9, 12]);
        negate_assign_u64(&mut a);
        assert_eq!(
            a,
            vec![
                3u64.wrapping_neg(),
                6u64.wrapping_neg(),
                9u64.wrapping_neg(),
                12u64.wrapping_neg()
            ]
        );
    }
}
