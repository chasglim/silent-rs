# SILENT HSS

`hss` is a Rust implementation of Homomorphic Secret Sharing (HSS), designed for efficient secure computation. It leverages the Residual Number System (RNS) for high performance and compatibility with the `fhe` library.

This library enables distributed decryption and secure aggregation of encrypted data without revealing the full secret key to any single party.

## Features

-   **HSS Scheme**: Implements the BKS19-based Homomorphic Secret Sharing scheme, adapted for RNS arithmetic.
-   **SIMD Support**: Fully supports batching (packing multiple integers into a single ciphertext) using `HssBatchEncoder`, enabling parallel operations on vectors.
-   **Homomorphic Operations**:
    -   **Addition**: Add two HSS ciphertexts homomorphically (`Enc(a) + Enc(b) = Enc(a+b)`).
    -   **Scalar Multiplication**: Multiply an HSS ciphertext by a plaintext polynomial (`Enc(a) * b = Enc(a*b)`).
    -   **Decryption Share Arithmetic**: Perform weighted aggregation on decryption shares (`(C1 . s)*k1 + (C2 . s)*k2`).
-   **Distributed Decryption**: Supports splitting a secret key into shares (`s1`, `s2`) and computing partial decryption shares (`hss_dot_prod`).
-   **Correctness**: Verified against complex circuits and recursive multiplication scenarios.

## HSS Operations & Terminology (BKS19)

Following the BKS19 framework (`docs/bks19.md`), `hss` implements the core operations for **Restricted Multiplication Straight-line (RMS)** programs:

### 1. **Input Value ($x$)**
An input to the computation, encrypted as a ciphertext (`HssCiphertext`). In the HSS context, this ciphertext encrypts the input value scaled by the secret key components (`x · s`), enabling distributed evaluation.

### 2. **Memory Share ($\langle y \rangle$)**
A secret share of an intermediate computation value ($y$), maintained by each evaluating party (`HssShare`). A memory share effectively holds a noisy additive share of the value $y \cdot \mathbf{s}$.

### 3. **Paper-Exact RMS Interface (Fig.2)**
The following methods match BKS19 Figure 2 instruction semantics:

-   Load: `HssEvaluator::load(context, ek_b, C^x, id)`
-   Add memory: `HssEvaluator::add_mem(context, ek_b, t_b^x, t_b^{x'}, id)`
-   Add input: `HssEvaluator::add_input(context, C^x, C^{x'})`
-   Multiply memory by input: `HssEvaluator::mult_mem(context, ek_b, t_b^x, C^{x'}, id)`
-   Output share: `HssEvaluator::output_mem(context, t_b^x, r)`

At the primitive layer, raw runtime helpers such as `hss_mult(share, ciphertext, params, p)` are still available for low-level testing and algorithm work, but the recommended API is `HssContext`-first.

## Usage

Add the SILENT crates to your `Cargo.toml`:

```toml
[dependencies]
silent-hss = { path = "../silent-hss" }
silent-math = { path = "../silent-math" }
silent-rlwe = { path = "../silent-rlwe" }
silent-utils = { path = "../silent-utils" }
```

### 1. Setup & Key Generation

Initialize the context and generate keys.

```rust
use silent_hss::{HssBatchEncoder, HssContext, HssEncryptor, HssEvaluator, HssKeyGenerator};
use silent_math::modulus::Modulus;
use silent_math::rns::RnsBase;
use silent_math::rns_tool::RnsToolConfig;
use silent_params::{
    DistributionType, HssParams, LogN, MaxLinearTerms, MaxRmultDepth, ModulusBits,
    PlaintextModulus, RingDim, RingParams, RlweParams, SecurityLevel, ShareModulusBits,
};
use silent_rlwe::EncryptionParams;
use silent_utils::rng::SecureRng;
use rand::SeedableRng;

fn main() {
    // 1. Parameters
    let degree = 16;
    let plain_modulus = 97; // p
    // Q needs to be large enough for noise and Delta scaling (Delta = Q/p).
    // Use an NTT-friendly prime (e.g., 786433 for N=16)
    let q1 = 786433u64;

    let base_q = RnsBase::from_values(vec![q1]).expect("base_q");
    let base_t = Modulus::new(plain_modulus).expect("base_t");
    let config = RnsToolConfig::new(base_q.clone(), base_t);
    let runtime = EncryptionParams::from_rns_config(degree, config).expect("params");
    let params = HssParams {
        name: "readme-hss-dev-v1",
        rlwe: RlweParams::new(
            "readme-hss-rlwe-v1",
            RingParams::new(LogN(4), RingDim(degree), DistributionType::Ternary),
            vec![ModulusBits(20)],
            Vec::new(),
            None,
            SecurityLevel::NotSet,
        ),
        plaintext_modulus: PlaintextModulus(plain_modulus),
        share_modulus_bits: ShareModulusBits(12),
        max_linear_terms: MaxLinearTerms(16),
        max_rmult_depth: MaxRmultDepth(2),
        fixed_point_scale_bits: None,
        reconstruction_bound_bits: None,
        correctness_margin_bits: None,
    };
    let context = HssContext::from_runtime_parts(params, runtime).expect("context");
    let rng = SecureRng::from_seed([42u8; 32]);

    // 2. KeyGen
    let mut keygen = HssKeyGenerator::new(context.clone(), rng.clone());
    let sk = keygen.generate_secret_key();
    let pk = keygen.generate_public_key(&sk);
    // Split secret key into shares for 2-party computation
    let (share1_sk, share2_sk) = keygen.split_secret_key(&sk);
    
    // ...
}
```

### 2. SIMD Encoding & Encryption

Encode vectors of integers into polynomials and encrypt them.

```rust
    // 3. Encoder
    let encoder = HssBatchEncoder::new(degree, plain_modulus);

    // 4. Encrypt Vectors
    let vec1: Vec<u64> = vec![10; degree]; // [10, 10, ...]
    let vec2: Vec<u64> = vec![5; degree];  // [5, 5, ...]

    let poly1 = encoder.encode(&vec1);
    let poly2 = encoder.encode(&vec2);

    let mut encryptor = HssEncryptor::new(context.clone(), pk.clone(), rng.clone())
        .expect("encryptor");

    let ct1 = encryptor.encrypt_input_poly(&poly1).expect("e1");
    let ct2 = encryptor.encrypt_input_poly(&poly2).expect("e2");
```

### 3. Homomorphic Evaluation

Perform operations on ciphertexts.

```rust
    // Homomorphic Addition: ct_sum = Enc(v1 + v2)
    let ct_sum = HssEvaluator::add(&context, &ct1, &ct2);
    
    // Homomorphic Scalar Multiplication: ct_mul = Enc(v1 * scalar)
    // Note: The scalar must be lifted to R_Q (coefficient embedding + NTT)
    // This example assumes a helper or manual preparation for simplicity
    // ... (see tests/hss_ops_test.rs for implementation of lift_to_rq) ...
```

### 4. Distributed Decryption & Reconstruction

Each party computes a partial decryption share using their secret key share, then reconstructs the final result.

```rust
    // Party 1 computes share
    let share1 = HssEvaluator::dec_share(&context, &share1_sk, &ct_sum).expect("s1");
    
    // Party 2 computes share
    let share2 = HssEvaluator::dec_share(&context, &share2_sk, &ct_sum).expect("s2");

    // Reconstruct final result from shares
    // The reconstruction handles modulo reduction and rounding
    let plain_coeffs = HssEvaluator::reconstruct(&context, &[share1, share2]).expect("rec");

    // Decode back to vector
    let mut res_poly = unsafe { silent_ring::Poly::new_uninit(degree, 1) };
    res_poly.limb_mut(0).copy_from_slice(&plain_coeffs);
    let result_vec = encoder.decode(&res_poly);

    println!("Result: {:?}", result_vec); // Should be [15, 15, ...]
```

## Architecture

-   **`HssKeyGenerator`**: Manages secret key generation (`generate_secret_key`) and splitting for distributed setups (`split_secret_key`).
-   **`HssEncryptor`**: Encrypts messages (plaintext polynomials) into `HssCiphertext`.
-   **`HssContext`**: The canonical parameter boundary that binds validated `HssParams` to the RLWE runtime context.
-   **`HssEvaluator`**: The core engine for homomorphic operations (`add`, `mul_plain`) and partial decryption (`dec_share`, `reconstruct`).
-   **`HssBatchEncoder`**: Handles packing/unpacking of integer vectors into polynomials for SIMD operations, similar to BFV's BatchEncoder.

## Testing

Run the comprehensive test suite to verify correctness, including complex circuit evaluations and recursive multiplication:

```bash
cargo test -p hss
```
