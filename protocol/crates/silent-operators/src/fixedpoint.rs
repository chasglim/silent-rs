use crate::error::OperatorError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedPointConfig {
    pub modulus: u64,
    pub scale: u64,
}

impl FixedPointConfig {
    pub fn new(modulus: u64, scale: u64) -> Result<Self, OperatorError> {
        if modulus < 3 {
            return Err(OperatorError::InvalidParams(
                "fixed-point modulus must be >= 3",
            ));
        }
        if scale == 0 {
            return Err(OperatorError::InvalidParams(
                "fixed-point scale must be positive",
            ));
        }
        Ok(Self { modulus, scale })
    }

    pub fn encode_f64(&self, value: f64) -> u64 {
        self.encode_scaled_int((value * self.scale as f64).round() as i128)
    }

    pub fn encode_scaled_int(&self, value: i128) -> u64 {
        reduce_i128_mod(value, self.modulus)
    }

    pub fn decode_f64(&self, value: u64) -> f64 {
        self.signed_centered(value) as f64 / self.scale as f64
    }

    pub fn signed_centered(&self, value: u64) -> i128 {
        let value = value % self.modulus;
        if value > self.modulus / 2 {
            value as i128 - self.modulus as i128
        } else {
            value as i128
        }
    }

    pub fn signed_threshold(&self, value: f64) -> i128 {
        (value * self.scale as f64).round() as i128
    }

    pub fn rescale_product_value(&self, value: u64) -> u64 {
        let centered = self.signed_centered(value);
        self.encode_scaled_int(div_round_nearest(centered, self.scale as i128))
    }
}

pub fn reduce_i128_mod(value: i128, modulus: u64) -> u64 {
    let modulus_i = modulus as i128;
    let mut reduced = value % modulus_i;
    if reduced < 0 {
        reduced += modulus_i;
    }
    reduced as u64
}

pub fn div_round_nearest(value: i128, divisor: i128) -> i128 {
    debug_assert!(divisor > 0);
    if value >= 0 {
        (value + divisor / 2) / divisor
    } else {
        -((-value + divisor / 2) / divisor)
    }
}
