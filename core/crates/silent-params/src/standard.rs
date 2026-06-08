//! Homomorphic Encryption Standard v1.1 parameter tables.
//!
//! The v1.1 standard gives recommended `log q` upper bounds for power-of-two
//! cyclotomic RLWE rings with error standard deviation sigma ~= 3.2. Table 1
//! uses the classical BKZ.sieve cost model, and Table 2 uses the quantum
//! BKZ.qsieve cost model. Plaintext modulus and encoding are intentionally
//! outside these tables, so callers must still perform scheme-level correctness
//! checks.

use crate::distribution::DistributionType;
use crate::newtypes::{ModulusBits, RingDim};
use crate::security::SecurityLevel;

#[derive(Clone, Copy)]
struct StandardRow {
    ring_dim: usize,
    uniform: [u16; 6],
    gaussian_error: [u16; 6],
    ternary: [u16; 6],
}

const CLASSICAL_128: usize = 0;
const CLASSICAL_192: usize = 1;
const CLASSICAL_256: usize = 2;
const QUANTUM_128: usize = 3;
const QUANTUM_192: usize = 4;
const QUANTUM_256: usize = 5;

const STANDARD_V1_1_TABLE: &[StandardRow] = &[
    StandardRow {
        ring_dim: 1024,
        uniform: [29, 21, 16, 27, 19, 15],
        gaussian_error: [29, 21, 16, 27, 19, 15],
        ternary: [27, 19, 14, 25, 17, 13],
    },
    StandardRow {
        ring_dim: 2048,
        uniform: [56, 39, 31, 53, 37, 29],
        gaussian_error: [56, 39, 31, 53, 37, 29],
        ternary: [54, 37, 29, 51, 35, 27],
    },
    StandardRow {
        ring_dim: 4096,
        uniform: [111, 77, 60, 103, 72, 56],
        gaussian_error: [111, 77, 60, 103, 72, 56],
        ternary: [109, 75, 58, 101, 70, 54],
    },
    StandardRow {
        ring_dim: 8192,
        uniform: [220, 154, 120, 206, 143, 111],
        gaussian_error: [220, 154, 120, 206, 143, 111],
        ternary: [218, 152, 118, 202, 141, 109],
    },
    StandardRow {
        ring_dim: 16384,
        uniform: [440, 307, 239, 413, 286, 222],
        gaussian_error: [440, 307, 239, 413, 286, 222],
        ternary: [438, 305, 237, 411, 284, 220],
    },
    StandardRow {
        ring_dim: 32768,
        uniform: [880, 612, 478, 829, 573, 445],
        gaussian_error: [883, 613, 478, 829, 573, 445],
        ternary: [881, 611, 476, 827, 571, 443],
    },
];

/// Return the HE Standard v1.1 `log q` upper bound for a ring/distribution.
pub fn find_max_log_q(
    distribution: DistributionType,
    security: SecurityLevel,
    ring_dim: RingDim,
) -> Option<ModulusBits> {
    let index = security_index(security)?;
    let row = STANDARD_V1_1_TABLE
        .iter()
        .find(|row| row.ring_dim == ring_dim.0)?;
    Some(ModulusBits(distribution_values(row, distribution)[index]))
}

/// Return the first standard ring dimension that supports `required_log_q`.
pub fn find_min_ring_dim(
    distribution: DistributionType,
    security: SecurityLevel,
    required_log_q: ModulusBits,
) -> Option<RingDim> {
    let index = security_index(security)?;

    for row in STANDARD_V1_1_TABLE {
        let max = distribution_values(row, distribution)[index];
        if max >= required_log_q.0 {
            return Some(RingDim(row.ring_dim));
        }
    }
    None
}

fn security_index(security: SecurityLevel) -> Option<usize> {
    match security {
        SecurityLevel::Classical128 => Some(CLASSICAL_128),
        SecurityLevel::Classical192 => Some(CLASSICAL_192),
        SecurityLevel::Classical256 => Some(CLASSICAL_256),
        SecurityLevel::Quantum128 => Some(QUANTUM_128),
        SecurityLevel::Quantum192 => Some(QUANTUM_192),
        SecurityLevel::Quantum256 => Some(QUANTUM_256),
        SecurityLevel::Toy | SecurityLevel::NotSet => None,
    }
}

fn distribution_values(row: &StandardRow, distribution: DistributionType) -> &[u16; 6] {
    match distribution {
        DistributionType::Uniform => &row.uniform,
        DistributionType::GaussianError => &row.gaussian_error,
        DistributionType::Ternary => &row.ternary,
    }
}
