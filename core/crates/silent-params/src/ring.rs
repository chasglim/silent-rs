use crate::distribution::DistributionType;
use crate::error::ParamError;
use crate::newtypes::{LogN, RingDim};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RingParams {
    pub log_n: LogN,
    pub ring_dim: RingDim,
    pub distribution: DistributionType,
}

impl RingParams {
    pub fn new(log_n: LogN, ring_dim: RingDim, distribution: DistributionType) -> Self {
        Self {
            log_n,
            ring_dim,
            distribution,
        }
    }

    pub fn from_ring_dim(
        ring_dim: RingDim,
        distribution: DistributionType,
    ) -> Result<Self, ParamError> {
        let log_n = ring_dim.to_log_n().ok_or_else(|| match ring_dim.0 {
            0 => ParamError::ZeroRingDim,
            _ => ParamError::RingDimNotPowerOfTwo(ring_dim.0),
        })?;
        Ok(Self::new(log_n, ring_dim, distribution))
    }

    pub fn validate(&self) -> Result<(), ParamError> {
        if self.ring_dim.0 == 0 {
            return Err(ParamError::ZeroRingDim);
        }
        if !self.ring_dim.0.is_power_of_two() {
            return Err(ParamError::RingDimNotPowerOfTwo(self.ring_dim.0));
        }
        let expected = self
            .log_n
            .to_ring_dim()
            .ok_or(ParamError::RingLogMismatch {
                log_n: self.log_n.0,
                ring_dim: self.ring_dim.0,
            })?;
        if expected != self.ring_dim {
            return Err(ParamError::RingLogMismatch {
                log_n: self.log_n.0,
                ring_dim: self.ring_dim.0,
            });
        }
        Ok(())
    }
}
