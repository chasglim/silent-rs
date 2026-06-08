use std::collections::HashMap;

use silent_net::frame::MessageKind;

use crate::channel::{RuntimeMesh, RuntimeStats};
use crate::context::RuntimeContext;
use crate::error::RuntimeError;
use crate::ids::{GateCursor, PartyId, RuntimeRoute, TaskId};
use crate::program::{NamedRuntimeProgram, OperatorRegistry};
use crate::value::{SymbolTable, Value};

/// Per-party runtime session used by protocol implementations.
pub struct RuntimeSession {
    ctx: RuntimeContext,
    mesh: Option<RuntimeMesh>,
    gate_cursors: HashMap<TaskId, GateCursor>,
}

impl RuntimeSession {
    pub fn new(ctx: RuntimeContext) -> Self {
        Self {
            ctx,
            mesh: None,
            gate_cursors: HashMap::new(),
        }
    }

    pub fn with_mesh(ctx: RuntimeContext, mesh: RuntimeMesh) -> Result<Self, RuntimeError> {
        if mesh.local_party() != ctx.local_party() {
            return Err(RuntimeError::InvalidConfig(
                "runtime mesh local party must match context local party",
            ));
        }
        Ok(Self {
            ctx,
            mesh: Some(mesh),
            gate_cursors: HashMap::new(),
        })
    }

    pub fn context(&self) -> &RuntimeContext {
        &self.ctx
    }

    pub fn context_mut(&mut self) -> &mut RuntimeContext {
        &mut self.ctx
    }

    pub fn mesh(&self) -> Option<&RuntimeMesh> {
        self.mesh.as_ref()
    }

    pub fn mesh_mut(&mut self) -> Option<&mut RuntimeMesh> {
        self.mesh.as_mut()
    }

    pub fn attach_mesh(&mut self, mesh: RuntimeMesh) -> Result<(), RuntimeError> {
        if mesh.local_party() != self.ctx.local_party() {
            return Err(RuntimeError::InvalidConfig(
                "runtime mesh local party must match context local party",
            ));
        }
        self.mesh = Some(mesh);
        Ok(())
    }

    pub fn next_route(
        &mut self,
        task: TaskId,
        kind: MessageKind,
    ) -> Result<RuntimeRoute, RuntimeError> {
        self.gate_cursors
            .entry(task)
            .or_insert_with(|| GateCursor::new(self.ctx.session(), task))
            .next_route(kind)
    }

    pub fn execute_named(
        &mut self,
        registry: &OperatorRegistry,
        program: &NamedRuntimeProgram,
        symbols: &mut SymbolTable,
    ) -> Result<(), RuntimeError> {
        program.eval(registry, &mut self.ctx, symbols)
    }

    pub async fn send_value(
        &mut self,
        peer: PartyId,
        task: TaskId,
        kind: MessageKind,
        value: &Value,
    ) -> Result<RuntimeRoute, RuntimeError> {
        let route = self.next_route(task, kind)?;
        self.mesh_mut_required()?
            .send_value(peer, route, value)
            .await?;
        Ok(route)
    }

    pub async fn recv_expected_value(
        &mut self,
        peer: PartyId,
        expected: RuntimeRoute,
    ) -> Result<Value, RuntimeError> {
        self.mesh_mut_required()?
            .recv_expected_value(peer, expected)
            .await
    }

    pub async fn recv_value(
        &mut self,
        peer: PartyId,
    ) -> Result<(RuntimeRoute, Value), RuntimeError> {
        self.mesh_mut_required()?.recv_value(peer).await
    }

    pub async fn send_u64_vec(
        &mut self,
        peer: PartyId,
        task: TaskId,
        kind: MessageKind,
        values: &[u64],
    ) -> Result<RuntimeRoute, RuntimeError> {
        let route = self.next_route(task, kind)?;
        self.mesh_mut_required()?
            .send_u64_vec(peer, route, values)
            .await?;
        Ok(route)
    }

    pub async fn recv_expected_u64_vec(
        &mut self,
        peer: PartyId,
        expected: RuntimeRoute,
    ) -> Result<Vec<u64>, RuntimeError> {
        self.mesh_mut_required()?
            .recv_expected_u64_vec(peer, expected)
            .await
    }

    pub async fn recv_u64_vec(
        &mut self,
        peer: PartyId,
    ) -> Result<(RuntimeRoute, Vec<u64>), RuntimeError> {
        self.mesh_mut_required()?.recv_u64_vec(peer).await
    }

    pub async fn send_payload(
        &mut self,
        peer: PartyId,
        task: TaskId,
        kind: MessageKind,
        payload: Vec<u8>,
    ) -> Result<RuntimeRoute, RuntimeError> {
        let route = self.next_route(task, kind)?;
        self.mesh_mut_required()?
            .send_payload(peer, route, payload)
            .await?;
        Ok(route)
    }

    pub async fn recv_payload(
        &mut self,
        peer: PartyId,
    ) -> Result<(RuntimeRoute, Vec<u8>), RuntimeError> {
        self.mesh_mut_required()?.recv_payload(peer).await
    }

    pub fn stats(&self) -> RuntimeStats {
        self.mesh
            .as_ref()
            .map(RuntimeMesh::total_stats)
            .unwrap_or_default()
    }

    fn mesh_mut_required(&mut self) -> Result<&mut RuntimeMesh, RuntimeError> {
        self.mesh
            .as_mut()
            .ok_or(RuntimeError::InvalidConfig("runtime session has no mesh"))
    }
}
