use crate::fhe::{BfvParams, BgvParams, CkksParams};
use crate::hss::HssParams;
use crate::pqc::{PqcKemParams, PqcSigParams};
use crate::security::SecurityLevel;

pub fn current_bfv() -> BfvParams {
    let mut params = crate::presets::dev::dev_bfv_4096();
    params.name = "current-bfv-v1";
    params.rlwe.name = "current-rlwe-4096-v1";
    params
}

pub fn current_bgv() -> BgvParams {
    let mut params = crate::presets::dev::dev_bgv_4096();
    params.name = "current-bgv-v1";
    params.rlwe.name = "current-rlwe-bgv-4096-v1";
    params
}

pub fn current_ckks() -> CkksParams {
    let mut params = crate::presets::dev::dev_ckks_8192();
    params.name = "current-ckks-v1";
    params.rlwe.name = "current-rlwe-8192-v1";
    params
}

pub fn current_hss() -> HssParams {
    let mut params = crate::presets::dev::dev_hss_4096();
    params.name = "current-hss-v1";
    params.rlwe.name = "current-rlwe-hss-4096-v1";
    params
}

pub fn mlkem512() -> PqcKemParams {
    PqcKemParams {
        name: "mlkem512-v1",
        algorithm: "ML-KEM-512",
        security_level: SecurityLevel::Quantum128,
        public_key_bytes: 800,
        secret_key_bytes: 1632,
        ciphertext_bytes: 768,
        shared_secret_bytes: 32,
    }
}

pub fn mlkem768() -> PqcKemParams {
    PqcKemParams {
        name: "mlkem768-v1",
        algorithm: "ML-KEM-768",
        security_level: SecurityLevel::Quantum192,
        public_key_bytes: 1184,
        secret_key_bytes: 2400,
        ciphertext_bytes: 1088,
        shared_secret_bytes: 32,
    }
}

pub fn mlkem1024() -> PqcKemParams {
    PqcKemParams {
        name: "mlkem1024-v1",
        algorithm: "ML-KEM-1024",
        security_level: SecurityLevel::Quantum256,
        public_key_bytes: 1568,
        secret_key_bytes: 3168,
        ciphertext_bytes: 1568,
        shared_secret_bytes: 32,
    }
}

pub fn mldsa44() -> PqcSigParams {
    PqcSigParams {
        name: "mldsa44-v1",
        algorithm: "ML-DSA-44",
        security_level: SecurityLevel::Quantum128,
        public_key_bytes: 1312,
        secret_key_bytes: 2560,
        signature_bytes: 2420,
    }
}

pub fn mldsa65() -> PqcSigParams {
    PqcSigParams {
        name: "mldsa65-v1",
        algorithm: "ML-DSA-65",
        security_level: SecurityLevel::Quantum192,
        public_key_bytes: 1952,
        secret_key_bytes: 4032,
        signature_bytes: 3309,
    }
}

pub fn mldsa87() -> PqcSigParams {
    PqcSigParams {
        name: "mldsa87-v1",
        algorithm: "ML-DSA-87",
        security_level: SecurityLevel::Quantum256,
        public_key_bytes: 2592,
        secret_key_bytes: 4896,
        signature_bytes: 4627,
    }
}
