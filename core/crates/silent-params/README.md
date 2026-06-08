# silent-params

`silent-params` is the canonical parameter layer for SILENT.

Chinese version: [`README-CN.md`](README-CN.md)

It is not the execution engine for BFV, HSS, RLWE, or future schemes. Instead,
it defines the parameter language, validation rules, stable identifiers, and
presets that the rest of the stack can rely on.

In short:

- `silent-params` describes what a parameter set means.
- runtime crates such as `silent-rlwe`, `silent-fhe`, and `silent-hss`
  decide how to execute it.

## Why This Crate Exists

Without a dedicated parameter layer, a crypto codebase usually drifts into a
fragile state:

- each scheme grows its own ad hoc parameter struct
- the same concept gets different names in different crates
- validation logic is duplicated or partially skipped
- security assumptions are implicit instead of encoded
- benchmarks and production presets use incompatible representations

SILENT is explicitly trying to avoid that outcome.

This crate turns parameters into first-class architecture. That matters for:

- correctness: invalid combinations are rejected before runtime
- auditability: parameter meaning is centralized and explicit
- reproducibility: canonical encodings lead to stable `ParamsId` values
- extensibility: new schemes can join a shared parameter vocabulary
- maintainability: runtime code can focus on execution, not parameter policy

## Design Goals

`silent-params` is designed around a few strong rules.

### 1. Canonical parameters come first

Every scheme should have a stable, validated, immutable parameter object that
captures its intended semantics.

Examples:

- `RingParams`
- `RlweParams`
- `BfvParams`
- `CkksParams`
- `HssParams`
- `PqcKemParams`
- `PqcSigParams`

These objects are the source of truth. Runtime-specific structs are derived
from them, not the other way around.

### 2. Strong types over bare integers

The crate uses newtypes such as:

- `LogN`
- `RingDim`
- `ModulusBits`
- `PlaintextModulus`
- `ScaleBits`
- `MultiplicativeDepth`
- `ShareModulusBits`

This follows the principle that cryptographic parameters should not be passed
around as unlabelled `usize` or `u64` values.

### 3. Validation is part of the model

Each parameter set implements `ParameterSet`, which includes `validate()`.

Validation is not an optional helper. It is part of the contract.

### 4. Stable identity matters

Every canonical parameter set can produce a `ParamsId([u8; 32])` through a
stable canonical encoding and hash. This supports:

- caching
- artifact tracking
- compatibility checks
- benchmark reproducibility
- future serialization boundaries

### 5. Presets must be versionable

This crate includes versionable presets such as:

- `presets/toy.rs`
- `presets/dev.rs`
- `presets/current.rs`

The intention is that SILENT can evolve defaults without losing the ability
to refer to older parameter sets precisely.

## Influences and SILENT's Position

SILENT does not directly clone OpenFHE, Lattigo, or TFHE-rs. It borrows the
best ideas and builds its own architecture.

### Borrowed ideas

- OpenFHE: explicit `SecurityLevel` and standard-table-driven validation
- Lattigo: literal parameter descriptions that become validated immutable params
- TFHE-rs: strong typing and a bias against naked integers

### SILENT-specific choices

- `ParamsId` for canonical identity
- a shared `ParameterSet` trait across scheme families
- versioned presets as part of the package structure
- a dedicated HSS model instead of pretending HSS is "just BFV with another
  wrapper"

## What Lives Here

The current crate is split into focused modules:

- `security.rs`: security taxonomy such as `Classical128` and `Quantum128`
- `distribution.rs`: secret/noise distribution families
- `scheme.rs`: scheme family identifiers
- `newtypes.rs`: strong parameter scalar wrappers
- `ids.rs`: `ParamsId`, canonical encoding, and `ParameterSet`
- `ring.rs`: ring-level invariants
- `rlwe.rs`: RLWE-level modulus-chain and generation logic
- `fhe.rs`: BFV and CKKS parameter sets
- `hss.rs`: HSS-specific bound-aware parameters
- `pqc.rs`: parameter shapes for future PQC modules
- `standard.rs`: HE Standard style lookup helpers
- `presets/`: curated parameter presets

## What This Layer Is Responsible For

`silent-params` is responsible for:

- representing parameter intent
- validating structural and security constraints
- generating or describing modulus-chain requirements
- exposing stable identities
- defining curated presets

It is not responsible for:

- ciphertext arithmetic
- key generation algorithms
- NTT execution
- RLWE sampling/runtime state
- serialization formats for network or disk transport

## Canonical Params vs Runtime Params

This distinction is central to the SILENT architecture.

Canonical params:

- are human-meaningful
- are stable
- can be validated early
- are suitable for presets and registry usage

Runtime params:

- are optimized for execution
- may cache tables, primes, or derived state
- may differ between implementations while representing the same canonical set

For example, BFV in `silent-fhe` can be driven from a validated `BfvParams`,
then lowered into runtime RLWE structures. That gives us a clean split between
"what the scheme means" and "how this implementation executes it".

## Why HSS Has Its Own Parameter Type

HSS is not modeled as a thin variation of BFV or plain RLWE.

`HssParams` includes HSS-specific concerns such as:

- `share_modulus_bits`
- `max_linear_terms`
- `max_rmult_depth`
- `fixed_point_scale_bits`
- `reconstruction_bound_bits`
- `correctness_margin_bits`

This is important because HSS correctness depends on bound management, not only
on ring dimension and ciphertext modulus budget.

If SILENT grows toward threshold analytics, private aggregation, or rule
evaluation, that extra structure is essential.

## Why This Matters for Future CKKS and TFHE

Yes, this crate is also the foundation for future scheme expansion.

### CKKS

CKKS naturally fits into this layer because it shares RLWE foundations with BFV
while adding its own semantics:

- default scale
- level/depth budgeting
- rescaling-related policy

`CkksParams` already exists as a first-class canonical type.

### TFHE

TFHE should not be forced into the BFV/RLWE worldview. But it should still join
the same parameter architecture.

A likely future direction is to add dedicated types such as:

- `LweParams`
- `GlweParams`
- `BootstrapParams`
- `KeyswitchParams`
- `TfheParams`

That keeps the shared SILENT design language while respecting TFHE's distinct
mathematics.

## Core Public Concepts

### `SecurityLevel`

Security levels intentionally include both classical and quantum labels:

- `Toy`
- `Classical128`
- `Classical192`
- `Classical256`
- `Quantum128`
- `Quantum192`
- `Quantum256`
- `NotSet`

This keeps the model future-facing instead of assuming only one security lens.

### `ParameterSet`

All canonical parameter sets implement:

```rust
pub trait ParameterSet {
    fn name(&self) -> &'static str;
    fn scheme_family(&self) -> SchemeFamily;
    fn security_level(&self) -> SecurityLevel;
    fn params_id(&self) -> ParamsId;
    fn validate(&self) -> Result<(), ParamError>;
}
```

This is the common contract that lets tooling, registries, and runtime bridges
work consistently across schemes.

### `ParamsId`

`ParamsId` is a stable identity for a canonical parameter set. It is derived
from a versioned canonical encoding rather than from debug output or runtime
cache state.

That is a deliberate architectural choice. It means identity belongs to
semantics, not to transient implementation details.

## Security Table Integration

`standard.rs` provides HE Standard style helpers such as:

- `find_max_log_q(...)`
- `find_min_ring_dim(...)`

The first version currently focuses on ternary secrets and classic security
levels, but the interface is intentionally broader than the initial table.

This lets SILENT start with a conservative foundation without freezing the API
to a single standards snapshot forever.

## Typical Flow

The intended flow is:

1. Choose or construct a canonical parameter set.
2. Validate it.
3. Derive runtime config from it.
4. Execute the scheme in the runtime crate.

For example:

```rust
use silent_params::{BfvParams, ParameterSet};

fn assert_valid(params: &BfvParams) {
    params.validate().expect("invalid BFV params");
    let _id = params.params_id();
}
```

The runtime crate is then free to lower those params into implementation-specific
structures.

## Architectural Meaning

If you only remember one thing, remember this:

`silent-params` makes parameter design part of the architecture, not an
implementation afterthought.

That decision is what allows SILENT to grow into a multi-scheme library
without turning into a pile of incompatible constructors and hidden assumptions.
