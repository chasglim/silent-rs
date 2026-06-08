# `silent-io` — Canonical Serialization & Persistence Layer

## Motivation

`silent-io` is the **canonical persistence and wire encoding layer** for the SILENT cryptographic core. It addresses five concerns:

1. **Canonical encoding**: A cryptographic object (RLWE ciphertext, polynomial, secret key) must have exactly one valid byte representation. Ambiguous encodings break hashes, transcripts, test vectors, and cross-language interop.
2. **Parameter binding**: Ciphertexts and keys are meaningless without their parameters (modulus, degree, modulus chain). Deserialization must validate parameter consistency at decode time, not after.
3. **Large-object support**: Evaluation keys and bootstrap keys can reach hundreds of MB. The format must support streaming without loading everything into memory.
4. **Reproducible test vectors**: Fixed-seed object generation followed by canonical encoding must produce byte-for-byte identical output across runs and platforms.
5. **Security boundary**: Deserialization is a trust boundary. The crate must reject oversized input, non-canonical coefficients, unknown versions, and parameter mismatches—without panicking.

## Non-goals

- **RPC / network protocol**: Deferred to `silent-net`'s protobuf/tonic layer. `silent-io` provides framing primitives, not service discovery, authentication, or reconnection.
- **Parameter registry / object factory**: Belongs to the runtime/net layer. `silent-io` has no `ObjectRegistry`.
- **Security integrity (tamper resistance)**: The container checksum detects accidental corruption only. Authenticated integrity is deferred to a future sealed container (HMAC/AEAD).
- **JSON debug format**: Deferred to a standalone test-vectors crate (Phase 5), not a default dependency of `silent-io`.

## Dependency Graph

```
silent-io
├── thiserror
├── zeroize
└── xxhash-rust         (container checksum only)

silent-params → silent-io    (re-exports ParamsId)
silent-ring   → silent-io
silent-rlwe   → silent-io
silent-fhe    → silent-io
silent-hss    → silent-io
```

`silent-io` has **zero reverse dependencies** on SILENT business crates. `ParamsId` is defined here as a transparent byte wrapper; `silent-params` depends on `silent-io` and may re-export `ParamsId`. `scheme_id` is passed through as `u16`—`silent-io` never interprets its semantics.

## Module Structure

```
silent-io/src/
  lib.rs              # re-exports, version info
  error.rs            # IoError (length / version / type / param / encoding / I/O / checksum categories)
  traits.rs           # CanonicalEncode, CanonicalDecode, DecodeWithParams, ObjectKind
  primitives.rs       # read_u8..u64/i64, read_bytes, write_* — all little-endian fixint
  object_type.rs      # ObjectType enum (u32 discriminant)
  header.rs           # ContainerHeader, FrameHeader
  container.rs        # ContainerWriter / ContainerReader
  framed.rs           # FramedWriter / FramedReader
  secret.rs           # SecretBytes (zeroize-on-drop)
  validate.rs         # validate_coefficient_range, validate_size, validate_degree
```

## Core Traits

```rust
/// Encode: any object.
pub trait CanonicalEncode {
    fn encoded_len(&self) -> usize;
    fn encode_to<W: std::io::Write>(&self, writer: &mut W) -> Result<usize, IoError>;
    fn encode_to_vec(&self) -> Result<Vec<u8>, IoError> { /* default impl */ }
}

/// Decode without parameters.
/// Only for primitive types, parameter objects, and metadata.
/// Crypto objects (Poly, RLWE, HSS, etc.) MUST NOT implement this trait.
pub trait CanonicalDecode: Sized {
    fn decode_from<R: std::io::Read>(reader: &mut R) -> Result<Self, IoError>;
}

/// Decode with parameters.
/// All modulus/degree/chain-dependent types must use this.
pub trait DecodeWithParams<P>: Sized {
    fn decode_with_params<R: std::io::Read>(
        params: &P,
        reader: &mut R,
    ) -> Result<Self, IoError>;
}

/// Static metadata for every persistable type.
pub trait ObjectKind {
    const OBJECT_TYPE: ObjectType;
    const SERIALIZED_VERSION: u16;
}
```

**Hard rule**: `Poly`, `LweCiphertext`, `Ciphertext`, `EvaluationKey`, `HssShare`, and all other parameter-dependent crypto types **must implement decoding only through `DecodeWithParams`**, never `CanonicalDecode`. This prevents the unsafe "decode first, validate later" pattern.

## Encoding Conventions

### Primitive types — all little-endian fixint

| Type | Bytes | Notes |
|------|-------|-------|
| `u8` | 1 | |
| `u16` | 2 | LE |
| `u32` | 4 | LE |
| `u64` | 8 | LE |
| `i64` | 8 | LE |
| `bool` | 1 | 0x00 or 0x01 |
| `Vec<u8>` | 4 + N | `u32 LE len` then `len` bytes |
| `[T]` slice | 4 + N | `u32 LE count` then sequential encoding |
| `[u8; N]` | N | no length prefix |

Floating-point values are **not permitted in crypto object encoding**. They may only appear in test-vector metadata or non-crypto configuration layers.

### Poly

```
[num_moduli: u8] [degree: u32 LE] [data: u64 LE × (degree × num_moduli)]
```

Decoding validates:
- `degree` must equal the parameter's degree
- `num_moduli` must equal the parameter's modulus chain length
- Every coefficient must be `<` the corresponding modulus

### RLWE Ciphertext (compact seeded mode)

- Unseeded: `[num_polys: u16 LE] [Poly₀] ... [Polyₙ]`
- Seeded: `[num_polys: u16 LE] [seed: 64 bytes] [Poly₀]` — the a-polynomial is regenerated from the seed, saving ~50% space.

## ObjectType Enum

```rust
#[repr(u32)]
pub enum ObjectType {
    // ── ring ──
    Poly = 0x0001,
    NativePoly = 0x0002,
    GadgetDecomposition = 0x0003,

    // ── lattice / tfhe-compatible ──
    LweCiphertext = 0x0101,
    LweSecretKey = 0x0102,
    LwePublicKey = 0x0103,
    LweKeyswitchKey = 0x0104,
    GlweCiphertext = 0x0105,
    GlweSecretKey = 0x0106,
    GgswCiphertext = 0x0107,
    LweBootstrapKey = 0x0108,

    // ── rlwe / bfv-style ──
    RlweSecretKey = 0x0201,
    RlwePublicKey = 0x0202,
    RlweCiphertext = 0x0203,
    RlweEvaluationKey = 0x0204,
    RlweGaloisKey = 0x0205,

    // ── hss ──
    HssCiphertext = 0x0301,
    HssShare = 0x0302,
    HssEvalKey = 0x0303,

    // ── shortint ──
    ShortintCiphertext = 0x0401,
    ShortintClientKey = 0x0402,
    ShortintServerKey = 0x0403,

    // ── params ──
    RingParams = 0x1001,
    RlweParams = 0x1002,
    BfvParams = 0x1003,
    HssParams = 0x1004,
    TfheParams = 0x1005,
}
```

`PolyShoup` is deliberately excluded — Shoup precomputation values are reconstructed from `Poly + modulus` at runtime and are not a stable persistence target.

## Container Format

### File container

```
Offset  Size   Field
──────  ────   ─────
0       4      magic: b"SFIR"
4       2      format_version: u16 LE = 1
6       4      object_type: u32 LE
10      2      scheme_id: u16 LE
12      32     params_id: [u8; 32]          (all-zero = no parameter binding)
44      1      flags: u8
45      2      object_version: u16 LE
47      1      reserved: u8
48      8      payload_len: u64 LE
──────  ────   ─────  header: 56 bytes
56      N      payload: [u8; payload_len]
56+N    8      checksum: u64 LE (XXH3 of header bytes + payload bytes)
```

The checksum covers the 56 header bytes plus the payload. The checksum field itself is excluded from the hash.

### API

```rust
pub struct EncodeContext {
    pub scheme_id: u16,
    pub params_id: ParamsId,
    pub flags: u8,
}

impl ContainerWriter {
    pub fn write_object<T>(&mut self, object: &T, ctx: &EncodeContext) -> Result<(), IoError>
    where
        T: CanonicalEncode + ObjectKind;
}

impl ContainerReader {
    pub fn read_header(&mut self) -> Result<ContainerHeader, IoError>;
    pub fn read_payload<T: CanonicalDecode>(&mut self, header: &ContainerHeader) -> Result<T, IoError>;
    pub fn read_payload_with_params<T, P>(
        &mut self, header: &ContainerHeader, params: &P
    ) -> Result<T, IoError>
    where
        T: DecodeWithParams<P>;
}
```

### Stream frame

```
Offset  Size   Field
──────  ────   ─────
0       4      payload_len: u32 LE          (header bytes excluded)
4       4      object_type: u32 LE
8       1      flags: u8
9       3      reserved: [u8; 3]
──────  ────   ─────  frame header: 12 bytes
12      N      payload: [u8; payload_len]
```

`payload_len` is bounded by a configurable `max_frame_payload` to prevent memory amplification.

## Checksum Layering

| Purpose | Algorithm | Location |
|---------|-----------|----------|
| Accidental corruption detection | XXH3 (8 bytes) | `container.rs`, covers header + payload |
| Test vector canonical digest | SHA-256 | Phase 5 test-vectors crate |
| Security integrity (tamper resistance) | HMAC / AEAD | Future sealed container; not in current scope |

XXH3 is a fast non-cryptographic hash for corruption detection only. It does not provide cryptographic integrity guarantees.

## SecretBytes

```rust
pub struct SecretBytes(Vec<u8>);
// - zeroize on Drop
// - no Debug / Display / Clone
// - exposes as_ref() for encoding
```

Key material (`LweSecretKey`, `GlweSecretKey`, `RlweSecretKey`) is wrapped in `SecretBytes` during encoding to ensure memory clearing and prevent accidental log leakage.

## Net Integration (Phase 6)

```rust
/// Send an encoded object (frame header + payload).
fn send_object<T>(&mut self, object: &T) -> Result<(), IoError>
where
    T: CanonicalEncode + ObjectKind;

/// Receive a parameter-free object.
fn recv_object<T: CanonicalDecode>(&mut self) -> Result<T, IoError>;

/// Receive a parameter-dependent object (Poly, RLWE, HSS, etc.).
fn recv_object_with_params<T, P>(&mut self, params: &P) -> Result<T, IoError>
where
    T: DecodeWithParams<P>;

/// Send raw payload with explicit object type (for dynamic dispatch).
fn send_payload(
    &mut self,
    object_type: ObjectType,
    flags: u8,
    payload: &[u8],
) -> Result<(), IoError>;
```

Phase 6 implements only `LocalTransport` and `FramedTransport` (over `silent-io::framed`). protobuf/tonic integration is deferred.

## Implementation Phases

### Phase 1 — Core codec
`lib.rs`, `error.rs`, `traits.rs`, `primitives.rs`, `object_type.rs`

- `IoError` covering length, version, type, param-binding, non-canonical encoding, I/O, and checksum categories
- All primitive read/write in little-endian fixint with size limits
- Roundtrip tests for every primitive

### Phase 2 — Container + Framed + Security
`header.rs`, `container.rs`, `framed.rs`, `validate.rs`, `secret.rs`

- Container file I/O (magic + header + payload + XXH3)
- Length-delimited frame streaming
- `SecretBytes` zeroize wrapper
- Coefficient range / size / degree validation
- Malformed-data rejection tests

### Phase 3 — Params + Ring
`silent-params/src/io_impls.rs`, `silent-ring/src/io_impls.rs`

- All `*Params` types → `CanonicalEncode + CanonicalDecode`
- `Poly` / `NativePoly` → `CanonicalEncode + DecodeWithParams`
- Strict decode-time validation against parameters

### Phase 4 — RLWE / HSS / Shortint
`silent-rlwe/src/io_impls.rs`, `silent-hss/src/io_impls.rs`, `silent-fhe/src/io_impls.rs`

- RLWE ciphertext first (shared dependency of net, HSS, and FHE)
- Then: EvaluationKey, HssShare, TFHE types, Shortint types

### Phase 5 — Test Vectors
Standalone `silent-test-vectors` crate (or `test-vectors` feature of `silent-io`)

- JSON metadata + binary payload
- SHA-256 canonical digest
- Known-answer tests

### Phase 6 — silent-net MVP
- `send_object` / `recv_object` / `recv_object_with_params`
- `LocalTransport` + `FramedTransport`
- protobuf / tonic deferred
