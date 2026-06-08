#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SecurityLevel {
    Toy = 0,
    Classical128 = 1,
    Classical192 = 2,
    Classical256 = 3,
    Quantum128 = 4,
    Quantum192 = 5,
    Quantum256 = 6,
    NotSet = 7,
}

impl SecurityLevel {
    pub const fn is_classical(self) -> bool {
        matches!(
            self,
            Self::Classical128 | Self::Classical192 | Self::Classical256
        )
    }

    pub const fn is_quantum(self) -> bool {
        matches!(self, Self::Quantum128 | Self::Quantum192 | Self::Quantum256)
    }

    pub const fn bits(self) -> Option<u16> {
        match self {
            Self::Toy => Some(0),
            Self::Classical128 | Self::Quantum128 => Some(128),
            Self::Classical192 | Self::Quantum192 => Some(192),
            Self::Classical256 | Self::Quantum256 => Some(256),
            Self::NotSet => None,
        }
    }
}
