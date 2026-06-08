use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::context::{RuntimeContext, TraceEvent};
use crate::error::RuntimeError;
use crate::value::{SymbolTable, Value};

/// A single operator invocation inside a lightweight runtime program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeStep {
    pub op: String,
    pub inputs: Vec<usize>,
}

impl RuntimeStep {
    pub fn new(op: impl Into<String>, inputs: Vec<usize>) -> Self {
        Self {
            op: op.into(),
            inputs,
        }
    }
}

/// A named operator invocation against a [`SymbolTable`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeInstruction {
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl RuntimeInstruction {
    pub fn new(
        op: impl Into<String>,
        inputs: impl IntoIterator<Item = impl Into<String>>,
        outputs: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, RuntimeError> {
        let op = op.into();
        if op.is_empty() {
            return Err(RuntimeError::InvalidConfig(
                "runtime instruction op must be non-empty",
            ));
        }
        let inputs = inputs.into_iter().map(Into::into).collect::<Vec<String>>();
        let outputs = outputs.into_iter().map(Into::into).collect::<Vec<String>>();
        validate_symbol_names(&inputs)?;
        validate_symbol_names(&outputs)?;
        let mut unique_outputs = HashSet::with_capacity(outputs.len());
        for output in &outputs {
            if !unique_outputs.insert(output.as_str()) {
                return Err(RuntimeError::InvalidConfig(
                    "runtime instruction output names must be unique",
                ));
            }
        }
        Ok(Self {
            op,
            inputs,
            outputs,
        })
    }
}

/// Sequential operator program used by examples and tests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeProgram {
    steps: Vec<RuntimeStep>,
}

impl RuntimeProgram {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn push_step(&mut self, step: RuntimeStep) {
        self.steps.push(step);
    }

    pub fn steps(&self) -> &[RuntimeStep] {
        &self.steps
    }

    pub fn eval(
        &self,
        registry: &OperatorRegistry,
        ctx: &mut RuntimeContext,
        mut values: Vec<Value>,
    ) -> Result<Vec<Value>, RuntimeError> {
        for step in &self.steps {
            let inputs = step
                .inputs
                .iter()
                .map(|&idx| {
                    values.get(idx).cloned().ok_or(RuntimeError::InvalidValue(
                        "runtime program input index out of bounds",
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut outputs = registry.eval(&step.op, ctx, &inputs)?;
            values.append(&mut outputs);
        }
        Ok(values)
    }
}

/// Sequential named program evaluated against a symbol table.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NamedRuntimeProgram {
    instructions: Vec<RuntimeInstruction>,
}

impl NamedRuntimeProgram {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    pub fn push_instruction(&mut self, instruction: RuntimeInstruction) {
        self.instructions.push(instruction);
    }

    pub fn instructions(&self) -> &[RuntimeInstruction] {
        &self.instructions
    }

    pub fn eval(
        &self,
        registry: &OperatorRegistry,
        ctx: &mut RuntimeContext,
        symbols: &mut SymbolTable,
    ) -> Result<(), RuntimeError> {
        for instruction in &self.instructions {
            let inputs = instruction
                .inputs
                .iter()
                .map(|name| symbols.get_var_cloned(name))
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = registry.eval(&instruction.op, ctx, &inputs)?;
            if outputs.len() != instruction.outputs.len() {
                return Err(RuntimeError::InvalidValue(
                    "operator output count does not match runtime instruction outputs",
                ));
            }
            for (name, value) in instruction.outputs.iter().zip(outputs.into_iter()) {
                symbols.set_var(name.clone(), value)?;
            }
        }
        Ok(())
    }
}

/// Operator callable from the runtime dispatch layer.
pub trait Operator: Send + Sync {
    fn name(&self) -> &'static str;
    fn eval(&self, ctx: &mut RuntimeContext, inputs: &[Value]) -> Result<Vec<Value>, RuntimeError>;
}

/// Registry for runtime operators.
#[derive(Default)]
pub struct OperatorRegistry {
    operators: HashMap<&'static str, Arc<dyn Operator>>,
}

impl OperatorRegistry {
    pub fn register<O: Operator + 'static>(&mut self, op: O) -> Result<(), RuntimeError> {
        let name = op.name();
        if self.operators.contains_key(name) {
            return Err(RuntimeError::DuplicateOperator(name));
        }
        self.operators.insert(name, Arc::new(op));
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Operator>> {
        self.operators.get(name).cloned()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.operators.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.operators.len()
    }

    pub fn is_empty(&self) -> bool {
        self.operators.is_empty()
    }

    pub fn names(&self) -> Vec<&'static str> {
        let mut names = self.operators.keys().copied().collect::<Vec<_>>();
        names.sort_unstable();
        names
    }

    pub fn eval(
        &self,
        name: &str,
        ctx: &mut RuntimeContext,
        inputs: &[Value],
    ) -> Result<Vec<Value>, RuntimeError> {
        let op = self
            .get(name)
            .ok_or_else(|| RuntimeError::UnknownOperator(name.to_owned()))?;
        let out = op.eval(ctx, inputs)?;
        ctx.record(TraceEvent {
            op: name.to_owned(),
            inputs: inputs.len(),
            outputs: out.len(),
        });
        Ok(out)
    }
}

fn validate_symbol_names(names: &[String]) -> Result<(), RuntimeError> {
    if names.iter().any(|name| name.is_empty()) {
        return Err(RuntimeError::InvalidConfig(
            "runtime symbol names must be non-empty",
        ));
    }
    Ok(())
}
