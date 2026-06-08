use crate::error::ParamError;
use crate::ids::{
    CanonicalParamEncoding, ParameterSet, ParamsId, params_id_from_canonical, push_str, push_u8,
    push_usize,
};
use crate::scheme::SchemeFamily;
use crate::security::SecurityLevel;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PqcKemParams {
    pub name: &'static str,
    pub algorithm: &'static str,
    pub security_level: SecurityLevel,
    pub public_key_bytes: usize,
    pub secret_key_bytes: usize,
    pub ciphertext_bytes: usize,
    pub shared_secret_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PqcSigParams {
    pub name: &'static str,
    pub algorithm: &'static str,
    pub security_level: SecurityLevel,
    pub public_key_bytes: usize,
    pub secret_key_bytes: usize,
    pub signature_bytes: usize,
}

impl ParameterSet for PqcKemParams {
    fn name(&self) -> &'static str {
        self.name
    }

    fn scheme_family(&self) -> SchemeFamily {
        SchemeFamily::PqcKem
    }

    fn security_level(&self) -> SecurityLevel {
        self.security_level
    }

    fn params_id(&self) -> ParamsId {
        params_id_from_canonical(b"pqc-kem", self)
    }

    fn validate(&self) -> Result<(), ParamError> {
        if self.public_key_bytes == 0 {
            return Err(ParamError::InvalidPqcSize {
                field: "public_key_bytes",
            });
        }
        if self.secret_key_bytes == 0 {
            return Err(ParamError::InvalidPqcSize {
                field: "secret_key_bytes",
            });
        }
        if self.ciphertext_bytes == 0 {
            return Err(ParamError::InvalidPqcSize {
                field: "ciphertext_bytes",
            });
        }
        if self.shared_secret_bytes == 0 {
            return Err(ParamError::InvalidPqcSize {
                field: "shared_secret_bytes",
            });
        }
        Ok(())
    }
}

impl ParameterSet for PqcSigParams {
    fn name(&self) -> &'static str {
        self.name
    }

    fn scheme_family(&self) -> SchemeFamily {
        SchemeFamily::PqcSig
    }

    fn security_level(&self) -> SecurityLevel {
        self.security_level
    }

    fn params_id(&self) -> ParamsId {
        params_id_from_canonical(b"pqc-sig", self)
    }

    fn validate(&self) -> Result<(), ParamError> {
        if self.public_key_bytes == 0 {
            return Err(ParamError::InvalidPqcSize {
                field: "public_key_bytes",
            });
        }
        if self.secret_key_bytes == 0 {
            return Err(ParamError::InvalidPqcSize {
                field: "secret_key_bytes",
            });
        }
        if self.signature_bytes == 0 {
            return Err(ParamError::InvalidPqcSize {
                field: "signature_bytes",
            });
        }
        Ok(())
    }
}

impl CanonicalParamEncoding for PqcKemParams {
    fn encode_canonical(&self, out: &mut Vec<u8>) {
        push_str(out, self.name);
        push_str(out, self.algorithm);
        push_u8(out, self.security_level as u8);
        push_usize(out, self.public_key_bytes);
        push_usize(out, self.secret_key_bytes);
        push_usize(out, self.ciphertext_bytes);
        push_usize(out, self.shared_secret_bytes);
    }
}

impl CanonicalParamEncoding for PqcSigParams {
    fn encode_canonical(&self, out: &mut Vec<u8>) {
        push_str(out, self.name);
        push_str(out, self.algorithm);
        push_u8(out, self.security_level as u8);
        push_usize(out, self.public_key_bytes);
        push_usize(out, self.secret_key_bytes);
        push_usize(out, self.signature_bytes);
    }
}
