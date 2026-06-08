use std::collections::HashMap;
use std::io::Cursor;

use silent_io::primitives;

use crate::error::RuntimeError;
use crate::ids::{FieldType, PartyId, Visibility};

const RUNTIME_VALUE_PAYLOAD_VERSION: u16 = 1;

/// Typed tensor metadata carried by runtime values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueType {
    pub visibility: Visibility,
    pub field: FieldType,
    pub shape: Vec<usize>,
}

impl ValueType {
    pub fn scalar(visibility: Visibility, field: FieldType) -> Self {
        Self {
            visibility,
            field,
            shape: Vec::new(),
        }
    }

    pub fn element_count(&self) -> Result<usize, RuntimeError> {
        if self.shape.is_empty() {
            return Ok(1);
        }
        self.shape.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or(RuntimeError::InvalidValue("shape element count overflow"))
        })
    }
}

/// Runtime value payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueData {
    /// Public ring/field elements.
    PublicU64(Vec<u64>),
    /// Canonical bytes for a protocol-specific secret object.
    SecretBytes(Vec<u8>),
    /// Opaque handle to a value stored in an operator backend.
    Handle(String),
}

/// A value visible to the homogeneous runtime layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value {
    pub ty: ValueType,
    pub data: ValueData,
}

impl Value {
    pub fn public_u64(
        values: Vec<u64>,
        field: FieldType,
        shape: Vec<usize>,
    ) -> Result<Self, RuntimeError> {
        let ty = ValueType {
            visibility: Visibility::Public,
            field,
            shape,
        };
        if values.len() != ty.element_count()? {
            return Err(RuntimeError::InvalidValue(
                "public value length does not match shape",
            ));
        }
        Ok(Self {
            ty,
            data: ValueData::PublicU64(values),
        })
    }

    pub fn secret_bytes(
        bytes: Vec<u8>,
        field: FieldType,
        shape: Vec<usize>,
    ) -> Result<Self, RuntimeError> {
        let ty = ValueType {
            visibility: Visibility::Secret,
            field,
            shape,
        };
        ty.element_count()?;
        Ok(Self {
            ty,
            data: ValueData::SecretBytes(bytes),
        })
    }

    pub fn handle(
        handle: impl Into<String>,
        visibility: Visibility,
        field: FieldType,
        shape: Vec<usize>,
    ) -> Result<Self, RuntimeError> {
        let ty = ValueType {
            visibility,
            field,
            shape,
        };
        ty.element_count()?;
        let handle = handle.into();
        if handle.is_empty() {
            return Err(RuntimeError::InvalidValue(
                "runtime handle must be non-empty",
            ));
        }
        Ok(Self {
            ty,
            data: ValueData::Handle(handle),
        })
    }
}

/// Runtime variable store keyed by stable symbolic names.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SymbolTable {
    data: HashMap<String, Value>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn set_var(&mut self, name: impl Into<String>, value: Value) -> Result<(), RuntimeError> {
        let name = name.into();
        if name.is_empty() {
            return Err(RuntimeError::InvalidValue("symbol name must be non-empty"));
        }
        self.data.insert(name, value);
        Ok(())
    }

    pub fn get_var(&self, name: &str) -> Result<&Value, RuntimeError> {
        self.data
            .get(name)
            .ok_or_else(|| RuntimeError::UnknownSymbol(name.to_owned()))
    }

    pub fn get_var_cloned(&self, name: &str) -> Result<Value, RuntimeError> {
        self.get_var(name).cloned()
    }

    pub fn has_var(&self, name: &str) -> bool {
        self.data.contains_key(name)
    }

    pub fn del_var(&mut self, name: &str) -> Option<Value> {
        self.data.remove(name)
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn names(&self) -> Vec<&str> {
        let mut names = self.data.keys().map(String::as_str).collect::<Vec<_>>();
        names.sort_unstable();
        names
    }
}

pub fn encode_value_payload(value: &Value) -> Result<Vec<u8>, RuntimeError> {
    let mut payload = Vec::new();
    write_u16_rt(&mut payload, RUNTIME_VALUE_PAYLOAD_VERSION)?;
    encode_visibility(&mut payload, value.ty.visibility)?;
    encode_field(&mut payload, value.ty.field)?;
    write_len(&mut payload, value.ty.shape.len(), "value shape rank")?;
    for &dim in &value.ty.shape {
        write_u64_rt(&mut payload, dim as u64)?;
    }
    match &value.data {
        ValueData::PublicU64(values) => {
            write_u8_rt(&mut payload, 0)?;
            write_len(&mut payload, values.len(), "public value length")?;
            for &value in values {
                write_u64_rt(&mut payload, value)?;
            }
        }
        ValueData::SecretBytes(bytes) => {
            write_u8_rt(&mut payload, 1)?;
            write_bytes_with_len(&mut payload, bytes, "secret value payload")?;
        }
        ValueData::Handle(handle) => {
            write_u8_rt(&mut payload, 2)?;
            write_bytes_with_len(&mut payload, handle.as_bytes(), "runtime handle")?;
        }
    }
    Ok(payload)
}

pub fn decode_value_payload(payload: &[u8]) -> Result<Value, RuntimeError> {
    let mut cursor = Cursor::new(payload);
    let version = read_u16_rt(&mut cursor)?;
    if version > RUNTIME_VALUE_PAYLOAD_VERSION {
        return Err(RuntimeError::Codec(format!(
            "unsupported runtime value payload version: {version}"
        )));
    }
    let visibility = decode_visibility(&mut cursor)?;
    let field = decode_field(&mut cursor)?;
    let rank = read_len(&mut cursor, "value shape rank")?;
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        let dim = read_u64_rt(&mut cursor)?;
        let dim = usize::try_from(dim)
            .map_err(|_| RuntimeError::InvalidValue("shape dimension exceeds usize"))?;
        shape.push(dim);
    }
    let data_tag = read_u8_rt(&mut cursor)?;
    let value = match data_tag {
        0 => {
            if visibility != Visibility::Public {
                return Err(RuntimeError::InvalidValue(
                    "public value payload must have public visibility",
                ));
            }
            let len = read_len(&mut cursor, "public value length")?;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push(read_u64_rt(&mut cursor)?);
            }
            Value::public_u64(values, field, shape)?
        }
        1 => Value::secret_bytes(
            read_bytes_with_len(&mut cursor, "secret value payload")?,
            field,
            shape,
        )?,
        2 => {
            let bytes = read_bytes_with_len(&mut cursor, "runtime handle")?;
            let handle = String::from_utf8(bytes).map_err(|err| {
                RuntimeError::Codec(format!("runtime handle is not UTF-8: {err}"))
            })?;
            Value::handle(handle, visibility, field, shape)?
        }
        _ => {
            return Err(RuntimeError::Codec(format!(
                "unknown value data tag: {data_tag}"
            )));
        }
    };
    if cursor.position() != payload.len() as u64 {
        return Err(RuntimeError::Codec(
            "trailing bytes in runtime value payload".to_owned(),
        ));
    }
    Ok(value)
}

fn encode_visibility(out: &mut Vec<u8>, visibility: Visibility) -> Result<(), RuntimeError> {
    match visibility {
        Visibility::Public => write_u8_rt(out, 0),
        Visibility::Private(party) => {
            write_u8_rt(out, 1)?;
            write_u16_rt(out, party.0)
        }
        Visibility::Secret => write_u8_rt(out, 2),
    }
}

fn decode_visibility(cursor: &mut Cursor<&[u8]>) -> Result<Visibility, RuntimeError> {
    match read_u8_rt(cursor)? {
        0 => Ok(Visibility::Public),
        1 => Ok(Visibility::Private(PartyId(read_u16_rt(cursor)?))),
        2 => Ok(Visibility::Secret),
        tag => Err(RuntimeError::Codec(format!(
            "unknown visibility tag: {tag}"
        ))),
    }
}

fn encode_field(out: &mut Vec<u8>, field: FieldType) -> Result<(), RuntimeError> {
    match field {
        FieldType::Ring64 => write_u8_rt(out, 0),
        FieldType::Ring128 => write_u8_rt(out, 1),
        FieldType::Prime(modulus) => {
            write_u8_rt(out, 2)?;
            write_u64_rt(out, modulus)
        }
    }
}

fn decode_field(cursor: &mut Cursor<&[u8]>) -> Result<FieldType, RuntimeError> {
    let field = match read_u8_rt(cursor)? {
        0 => FieldType::Ring64,
        1 => FieldType::Ring128,
        2 => FieldType::Prime(read_u64_rt(cursor)?),
        tag => return Err(RuntimeError::Codec(format!("unknown field tag: {tag}"))),
    };
    field.validate()?;
    Ok(field)
}

fn write_bytes_with_len(
    out: &mut Vec<u8>,
    bytes: &[u8],
    what: &'static str,
) -> Result<(), RuntimeError> {
    write_len(out, bytes.len(), what)?;
    primitives::write_bytes(out, bytes).map_err(|err| RuntimeError::Codec(err.to_string()))?;
    Ok(())
}

fn read_bytes_with_len(
    cursor: &mut Cursor<&[u8]>,
    what: &'static str,
) -> Result<Vec<u8>, RuntimeError> {
    let len = read_len(cursor, what)?;
    primitives::read_vec(cursor, len).map_err(|err| RuntimeError::Codec(err.to_string()))
}

fn write_len(out: &mut Vec<u8>, len: usize, what: &'static str) -> Result<(), RuntimeError> {
    let len = u64::try_from(len)
        .map_err(|_| RuntimeError::InvalidValue("length does not fit into u64"))?;
    if len > u32::MAX as u64 {
        return Err(RuntimeError::InvalidValue(match what {
            "value shape rank" => "value shape rank exceeds u32::MAX",
            "public value length" => "public value length exceeds u32::MAX",
            "secret value payload" => "secret value payload exceeds u32::MAX",
            "runtime handle" => "runtime handle exceeds u32::MAX",
            _ => "runtime length exceeds u32::MAX",
        }));
    }
    write_u32_rt(out, len as u32)
}

fn read_len(cursor: &mut Cursor<&[u8]>, _what: &'static str) -> Result<usize, RuntimeError> {
    usize::try_from(read_u32_rt(cursor)?)
        .map_err(|_| RuntimeError::InvalidValue("runtime length exceeds usize"))
}

fn write_u8_rt(out: &mut Vec<u8>, value: u8) -> Result<(), RuntimeError> {
    primitives::write_u8(out, value)
        .map(|_| ())
        .map_err(|err| RuntimeError::Codec(err.to_string()))
}

fn write_u16_rt(out: &mut Vec<u8>, value: u16) -> Result<(), RuntimeError> {
    primitives::write_u16(out, value)
        .map(|_| ())
        .map_err(|err| RuntimeError::Codec(err.to_string()))
}

fn write_u32_rt(out: &mut Vec<u8>, value: u32) -> Result<(), RuntimeError> {
    primitives::write_u32(out, value)
        .map(|_| ())
        .map_err(|err| RuntimeError::Codec(err.to_string()))
}

fn write_u64_rt(out: &mut Vec<u8>, value: u64) -> Result<(), RuntimeError> {
    primitives::write_u64(out, value)
        .map(|_| ())
        .map_err(|err| RuntimeError::Codec(err.to_string()))
}

fn read_u8_rt(cursor: &mut Cursor<&[u8]>) -> Result<u8, RuntimeError> {
    primitives::read_u8(cursor).map_err(|err| RuntimeError::Codec(err.to_string()))
}

fn read_u16_rt(cursor: &mut Cursor<&[u8]>) -> Result<u16, RuntimeError> {
    primitives::read_u16(cursor).map_err(|err| RuntimeError::Codec(err.to_string()))
}

fn read_u32_rt(cursor: &mut Cursor<&[u8]>) -> Result<u32, RuntimeError> {
    primitives::read_u32(cursor).map_err(|err| RuntimeError::Codec(err.to_string()))
}

fn read_u64_rt(cursor: &mut Cursor<&[u8]>) -> Result<u64, RuntimeError> {
    primitives::read_u64(cursor).map_err(|err| RuntimeError::Codec(err.to_string()))
}
