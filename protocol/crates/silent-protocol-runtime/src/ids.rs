use silent_io::{EncodeContext, ParamsId};
use silent_net::frame::{DEFAULT_MAX_FRAME_PAYLOAD, MessageKind};
use silent_net::routing::RouteKey;

use crate::error::RuntimeError;

/// MPC or hybrid protocol family selected for a runtime session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProtocolKind {
    /// Single-process deterministic execution, useful for local operator tests.
    Local,
    /// Generic two-party semi-honest runtime.
    SemiHonest2pc,
    /// Two-party inference runtime using HE/HSS-style operators.
    Inference2pc,
    /// A named protocol supplied by an example or downstream crate.
    Named(String),
}

/// Arithmetic field/ring used by high-level values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldType {
    /// 64-bit ring arithmetic.
    Ring64,
    /// 128-bit ring arithmetic.
    Ring128,
    /// Prime-field arithmetic with the given modulus.
    Prime(u64),
}

impl FieldType {
    pub(crate) fn validate(self) -> Result<(), RuntimeError> {
        match self {
            Self::Prime(p) if p < 2 => Err(RuntimeError::InvalidConfig(
                "prime field modulus must be >= 2",
            )),
            _ => Ok(()),
        }
    }
}

/// Value visibility in the virtual runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    /// Public value known to every party.
    Public,
    /// Private value owned by one party.
    Private(PartyId),
    /// Secret-shared value.
    Secret,
}

/// Stable party identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PartyId(pub u16);

/// Validated set of parties for a protocol session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeTopology {
    parties: Vec<PartyId>,
}

impl RuntimeTopology {
    pub fn new(parties: Vec<PartyId>) -> Result<Self, RuntimeError> {
        if parties.is_empty() {
            return Err(RuntimeError::InvalidConfig(
                "runtime topology must contain at least one party",
            ));
        }
        let original_len = parties.len();
        let mut parties = parties;
        parties.sort_unstable();
        parties.dedup();
        if parties.len() != original_len {
            return Err(RuntimeError::InvalidConfig("party ids must be unique"));
        }
        Ok(Self { parties })
    }

    pub fn two_party() -> Self {
        Self {
            parties: vec![PartyId(0), PartyId(1)],
        }
    }

    pub fn parties(&self) -> &[PartyId] {
        &self.parties
    }

    pub fn contains(&self, party: PartyId) -> bool {
        self.parties.binary_search(&party).is_ok()
    }

    pub fn index_of(&self, party: PartyId) -> Option<usize> {
        self.parties.binary_search(&party).ok()
    }

    pub fn peers(&self, local_party: PartyId) -> Result<Vec<PartyId>, RuntimeError> {
        if !self.contains(local_party) {
            return Err(RuntimeError::InvalidConfig(
                "local party must be present in runtime topology",
            ));
        }
        Ok(self
            .parties
            .iter()
            .copied()
            .filter(|&party| party != local_party)
            .collect())
    }
}

/// Runtime session identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

/// Task identifier inside a session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

/// Gate/operator-message identifier inside a task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GateId(pub u64);

/// Runtime configuration for a SILENT protocol session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub protocol: ProtocolKind,
    pub field: FieldType,
    pub parties: Vec<PartyId>,
    pub fixed_point_fraction_bits: u32,
    pub max_concurrency: usize,
    pub max_frame_payload: u32,
    pub public_random_seed: u64,
    pub enable_action_trace: bool,
    pub enable_operator_trace: bool,
}

impl RuntimeConfig {
    pub fn two_party_inference() -> Self {
        Self {
            protocol: ProtocolKind::Inference2pc,
            field: FieldType::Ring64,
            parties: RuntimeTopology::two_party().parties,
            fixed_point_fraction_bits: 16,
            max_concurrency: 1,
            max_frame_payload: DEFAULT_MAX_FRAME_PAYLOAD,
            public_random_seed: 0,
            enable_action_trace: false,
            enable_operator_trace: false,
        }
    }

    pub fn validate(&self) -> Result<(), RuntimeError> {
        self.field.validate()?;
        RuntimeTopology::new(self.parties.clone())?;
        if self.max_concurrency == 0 {
            return Err(RuntimeError::InvalidConfig(
                "max_concurrency must be positive",
            ));
        }
        if self.max_frame_payload == 0 {
            return Err(RuntimeError::InvalidConfig(
                "max_frame_payload must be positive",
            ));
        }
        Ok(())
    }

    pub fn topology(&self) -> Result<RuntimeTopology, RuntimeError> {
        RuntimeTopology::new(self.parties.clone())
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::two_party_inference()
    }
}

/// Wire metadata shared by canonical I/O and network frames.
#[derive(Clone, Debug)]
pub struct WireContext {
    pub encode: EncodeContext,
    pub max_payload: u32,
}

impl WireContext {
    pub fn new(scheme_id: u16, params_id: ParamsId, max_payload: u32) -> Self {
        Self {
            encode: EncodeContext {
                scheme_id,
                params_id,
                flags: 0,
            },
            max_payload,
        }
    }
}

impl Default for WireContext {
    fn default() -> Self {
        Self::new(0, ParamsId::ZERO, DEFAULT_MAX_FRAME_PAYLOAD)
    }
}

/// Runtime route metadata compatible with `silent-net`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuntimeRoute {
    pub session: SessionId,
    pub task: TaskId,
    pub gate: GateId,
    pub kind: MessageKind,
}

impl RuntimeRoute {
    pub const fn new(session: SessionId, task: TaskId, gate: GateId, kind: MessageKind) -> Self {
        Self {
            session,
            task,
            gate,
            kind,
        }
    }

    pub fn route_key(self) -> RouteKey {
        RouteKey::new(self.session.0, self.task.0, self.gate.0, self.kind)
    }

    pub fn ensure_matches(self, actual: RuntimeRoute) -> Result<(), RuntimeError> {
        if self == actual {
            Ok(())
        } else {
            Err(RuntimeError::RouteMismatch {
                expected: self,
                actual,
            })
        }
    }
}

impl From<RuntimeRoute> for RouteKey {
    fn from(value: RuntimeRoute) -> Self {
        value.route_key()
    }
}

/// Monotonic gate allocator for one `(session, task)` stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateCursor {
    session: SessionId,
    task: TaskId,
    next_gate: u64,
}

impl GateCursor {
    pub const fn new(session: SessionId, task: TaskId) -> Self {
        Self {
            session,
            task,
            next_gate: 0,
        }
    }

    pub const fn with_next_gate(session: SessionId, task: TaskId, next_gate: u64) -> Self {
        Self {
            session,
            task,
            next_gate,
        }
    }

    pub fn next_gate(&self) -> GateId {
        GateId(self.next_gate)
    }

    pub fn next_route(&mut self, kind: MessageKind) -> Result<RuntimeRoute, RuntimeError> {
        let gate = self.next_gate;
        self.next_gate = self
            .next_gate
            .checked_add(1)
            .ok_or(RuntimeError::InvalidValue("gate cursor overflow"))?;
        Ok(RuntimeRoute::new(
            self.session,
            self.task,
            GateId(gate),
            kind,
        ))
    }
}
