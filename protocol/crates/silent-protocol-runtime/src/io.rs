use std::io::{Read, Write};

use silent_io::container::{ContainerReader, ContainerWriter};

use crate::channel::RUNTIME_PAYLOAD_OBJECT;
use crate::context::RuntimeContext;
use crate::error::RuntimeError;
use crate::ids::{FieldType, RuntimeConfig, Visibility, WireContext};
use crate::value::{SymbolTable, Value, ValueData, decode_value_payload, encode_value_payload};

const RUNTIME_VALUE_CONTAINER_VERSION: u16 = 1;
const DEFAULT_RUNTIME_CONTAINER_LIMIT: u64 = 64 * 1024 * 1024;

/// Canonical runtime infeed/outfeed helper.
///
/// This type deliberately does not manufacture secret shares. It only moves
/// already-defined runtime values across SILENT's canonical I/O boundary and
/// populates or reads the runtime symbol table. Protocol-specific sharing stays
/// in the protocol/operator layer where the security semantics are known.
#[derive(Clone, Debug)]
pub struct RuntimeIo {
    config: RuntimeConfig,
    wire: WireContext,
    max_container_payload: u64,
}

impl RuntimeIo {
    pub fn new(config: RuntimeConfig, wire: WireContext) -> Result<Self, RuntimeError> {
        config.validate()?;
        Ok(Self {
            config,
            wire,
            max_container_payload: DEFAULT_RUNTIME_CONTAINER_LIMIT,
        })
    }

    pub fn from_context(ctx: &RuntimeContext) -> Self {
        Self {
            config: ctx.config().clone(),
            wire: ctx.wire().clone(),
            max_container_payload: DEFAULT_RUNTIME_CONTAINER_LIMIT,
        }
    }

    pub fn with_max_container_payload(mut self, max_container_payload: u64) -> Self {
        self.max_container_payload = max_container_payload;
        self
    }

    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    pub fn wire(&self) -> &WireContext {
        &self.wire
    }

    pub fn public_u64(&self, values: Vec<u64>, shape: Vec<usize>) -> Result<Value, RuntimeError> {
        Value::public_u64(values, self.config.field, shape)
    }

    pub fn secret_bytes(&self, bytes: Vec<u8>, shape: Vec<usize>) -> Result<Value, RuntimeError> {
        Value::secret_bytes(bytes, self.config.field, shape)
    }

    pub fn handle(
        &self,
        handle: impl Into<String>,
        visibility: Visibility,
        shape: Vec<usize>,
    ) -> Result<Value, RuntimeError> {
        Value::handle(handle, visibility, self.config.field, shape)
    }

    pub fn set_public_u64(
        &self,
        symbols: &mut SymbolTable,
        name: impl Into<String>,
        values: Vec<u64>,
        shape: Vec<usize>,
    ) -> Result<(), RuntimeError> {
        symbols.set_var(name, self.public_u64(values, shape)?)
    }

    pub fn set_secret_bytes(
        &self,
        symbols: &mut SymbolTable,
        name: impl Into<String>,
        bytes: Vec<u8>,
        shape: Vec<usize>,
    ) -> Result<(), RuntimeError> {
        symbols.set_var(name, self.secret_bytes(bytes, shape)?)
    }

    pub fn set_handle(
        &self,
        symbols: &mut SymbolTable,
        name: impl Into<String>,
        handle: impl Into<String>,
        visibility: Visibility,
        shape: Vec<usize>,
    ) -> Result<(), RuntimeError> {
        symbols.set_var(name, self.handle(handle, visibility, shape)?)
    }

    pub fn get_public_u64(
        &self,
        symbols: &SymbolTable,
        name: &str,
    ) -> Result<Vec<u64>, RuntimeError> {
        let value = symbols.get_var(name)?;
        self.expect_field(value)?;
        match (&value.ty.visibility, &value.data) {
            (Visibility::Public, ValueData::PublicU64(values)) => Ok(values.clone()),
            _ => Err(RuntimeError::InvalidValue(
                "symbol is not a public u64 runtime value",
            )),
        }
    }

    pub fn write_value_container<W: Write>(
        &self,
        writer: &mut W,
        value: &Value,
    ) -> Result<usize, RuntimeError> {
        self.expect_field(value)?;
        let payload = encode_value_payload(value)?;
        let mut container = ContainerWriter::new(writer);
        container
            .write_raw(
                RUNTIME_PAYLOAD_OBJECT,
                RUNTIME_VALUE_CONTAINER_VERSION,
                &self.wire.encode,
                &payload,
            )
            .map_err(RuntimeError::from)
    }

    pub fn read_value_container<R: Read>(&self, reader: &mut R) -> Result<Value, RuntimeError> {
        let mut container =
            ContainerReader::new(reader).with_max_payload(self.max_container_payload);
        let header = container.read_header()?;
        if header.object_type != RUNTIME_PAYLOAD_OBJECT {
            return Err(RuntimeError::Codec(format!(
                "unexpected runtime value object type: {}",
                header.object_type
            )));
        }
        if header.object_version > RUNTIME_VALUE_CONTAINER_VERSION {
            return Err(RuntimeError::Codec(format!(
                "unsupported runtime value container version: {}",
                header.object_version
            )));
        }
        if header.scheme_id != self.wire.encode.scheme_id {
            return Err(RuntimeError::Codec(format!(
                "runtime value scheme id mismatch: expected {}, got {}",
                self.wire.encode.scheme_id, header.scheme_id
            )));
        }
        if header.params_id != self.wire.encode.params_id {
            return Err(RuntimeError::Codec(
                "runtime value params id mismatch".to_owned(),
            ));
        }
        if header.flags != self.wire.encode.flags {
            return Err(RuntimeError::Codec(format!(
                "runtime value flags mismatch: expected {}, got {}",
                self.wire.encode.flags, header.flags
            )));
        }
        let payload = container.read_payload(&header)?;
        container.verify_checksum(&header, &payload)?;
        let value = decode_value_payload(&payload)?;
        self.expect_field(&value)?;
        Ok(value)
    }

    fn expect_field(&self, value: &Value) -> Result<(), RuntimeError> {
        if !field_matches(self.config.field, value.ty.field) {
            return Err(RuntimeError::InvalidValue(
                "runtime value field does not match io config",
            ));
        }
        Ok(())
    }
}

fn field_matches(expected: FieldType, actual: FieldType) -> bool {
    match (expected, actual) {
        (FieldType::Prime(a), FieldType::Prime(b)) => a == b,
        (a, b) => a == b,
    }
}
