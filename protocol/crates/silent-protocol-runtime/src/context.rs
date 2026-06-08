use silent_net::frame::MessageKind;

use crate::error::RuntimeError;
use crate::ids::{
    GateCursor, GateId, PartyId, RuntimeConfig, RuntimeRoute, RuntimeTopology, SessionId, TaskId,
    WireContext,
};

/// Per-session context passed to operators.
pub struct RuntimeContext {
    config: RuntimeConfig,
    local_party: PartyId,
    session: SessionId,
    wire: WireContext,
    trace: Vec<TraceEvent>,
}

impl RuntimeContext {
    pub fn new(
        config: RuntimeConfig,
        local_party: PartyId,
        session: SessionId,
        wire: WireContext,
    ) -> Result<Self, RuntimeError> {
        config.validate()?;
        if !config.parties.contains(&local_party) {
            return Err(RuntimeError::InvalidConfig(
                "local party must be present in runtime parties",
            ));
        }
        Ok(Self {
            config,
            local_party,
            session,
            wire,
            trace: Vec::new(),
        })
    }

    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    pub fn local_party(&self) -> PartyId {
        self.local_party
    }

    pub fn session(&self) -> SessionId {
        self.session
    }

    pub fn wire(&self) -> &WireContext {
        &self.wire
    }

    pub fn topology(&self) -> Result<RuntimeTopology, RuntimeError> {
        self.config.topology()
    }

    pub fn peers(&self) -> Result<Vec<PartyId>, RuntimeError> {
        self.topology()?.peers(self.local_party)
    }

    pub fn trace(&self) -> &[TraceEvent] {
        &self.trace
    }

    pub fn record(&mut self, event: TraceEvent) {
        if self.config.enable_action_trace || self.config.enable_operator_trace {
            self.trace.push(event);
        }
    }

    pub fn route(&self, task: TaskId, gate: GateId, kind: MessageKind) -> RuntimeRoute {
        RuntimeRoute::new(self.session, task, gate, kind)
    }

    pub fn gate_cursor(&self, task: TaskId) -> GateCursor {
        GateCursor::new(self.session, task)
    }
}

/// A runtime trace event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceEvent {
    pub op: String,
    pub inputs: usize,
    pub outputs: usize,
}
