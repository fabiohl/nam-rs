// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Custom serde visitors for limit validation of `.nam` format (JSON).
//!
//! Contains caps for security against DoS, weight array visitor,
//! and metadata.training depth/size visitors.

use serde::Deserializer;

use super::error::JsonError;

/// Maximum number of floats in the `weights` array (MAX_MODEL_BYTES / 4).
const MAX_WEIGHTS: usize = (256 * 1024 * 1024 / 4) as usize; // 64 Mi floats

/// Maximum size of the `metadata.training` field in bytes.
const MAX_TRAINING_BYTES: usize = 1024 * 1024; // 1 MiB

/// Maximum depth of the JSON tree in `metadata.training`.
const MAX_TRAINING_DEPTH: usize = 16;

/// Custom visitor for `Vec<f32>` that aborts upon exceeding MAX_WEIGHTS floats.
struct WeightsVisitor;

impl<'de> serde::de::Visitor<'de> for WeightsVisitor {
    type Value = Vec<f32>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a sequence of f32 floats within the size limit")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Vec<f32>, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut weights = Vec::new();
        loop {
            match seq.next_element::<f32>() {
                Ok(Some(val)) => {
                    if weights.len() >= MAX_WEIGHTS {
                        return Err(serde::de::Error::custom(JsonError::WeightsExceedLimit {
                            got: weights.len() + 1,
                            max: MAX_WEIGHTS,
                        }));
                    }
                    weights.push(val);
                }
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(weights)
    }
}

/// Custom deserializer for `weights: Vec<f32>` with cap at MAX_WEIGHTS.
pub(crate) fn deserialize_weights<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_seq(WeightsVisitor)
}

/// JSON tree visitor for `metadata.training` with depth and size limits.
/// Uses `std::cell::Cell<usize>` so that child visitors share the size counter
/// with the parent, avoiding bypass of the 1 MiB aggregate limit.
struct LimitedValueVisitor {
    depth: usize,
    max_depth: usize,
    max_size: usize,
    current_size: std::cell::Cell<usize>,
}

impl LimitedValueVisitor {
    fn root(max_depth: usize, max_size: usize) -> Self {
        Self {
            depth: 0,
            max_depth,
            max_size,
            current_size: std::cell::Cell::new(0),
        }
    }

    fn child(&self) -> Self {
        Self {
            depth: self.depth + 1,
            max_depth: self.max_depth,
            max_size: self.max_size,
            current_size: self.current_size.clone(),
        }
    }

    fn add_size(&self, bytes: usize) -> Result<(), serde_json::Error> {
        let new = self.current_size.get() + bytes;
        self.current_size.set(new);
        if new > self.max_size {
            return Err(serde::de::Error::custom(JsonError::TrainingTooLarge {
                size: new,
                max_size: self.max_size,
            }));
        }
        Ok(())
    }

    fn check_depth(&self) -> Result<(), serde_json::Error> {
        if self.depth > self.max_depth {
            return Err(serde::de::Error::custom(JsonError::TrainingTooDeep {
                depth: self.depth,
                max_depth: self.max_depth,
            }));
        }
        Ok(())
    }
}

impl<'de> serde::de::Visitor<'de> for LimitedValueVisitor {
    type Value = serde_json::Value;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a JSON value within depth and size limits")
    }

    fn visit_bool<E>(self, v: bool) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.add_size(if v { 4 } else { 5 }).map_err(E::custom)?;
        Ok(serde_json::Value::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.add_size(16).map_err(E::custom)?;
        Ok(serde_json::Value::Number(serde_json::Number::from(v)))
    }

    fn visit_u64<E>(self, v: u64) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.add_size(16).map_err(E::custom)?;
        Ok(serde_json::Value::Number(serde_json::Number::from(v)))
    }

    fn visit_f64<E>(self, v: f64) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.add_size(16).map_err(E::custom)?;
        Ok(serde_json::Value::Number(
            serde_json::Number::from_f64(v).unwrap_or(serde_json::Number::from(0)),
        ))
    }

    fn visit_str<E>(self, v: &str) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.add_size(v.len() + 2).map_err(E::custom)?;
        Ok(serde_json::Value::String(v.to_string()))
    }

    fn visit_string<E>(self, v: String) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.add_size(v.len() + 2).map_err(E::custom)?;
        Ok(serde_json::Value::String(v))
    }

    fn visit_unit<E>(self) -> Result<serde_json::Value, E> {
        Ok(serde_json::Value::Null)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<serde_json::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        self.check_depth().map_err(serde::de::Error::custom)?;
        self.add_size(2).map_err(serde::de::Error::custom)?; // [ ]
        let mut arr = Vec::new();
        loop {
            match seq.next_element_seed(self.child()) {
                Ok(Some(val)) => {
                    self.add_size(1).map_err(serde::de::Error::custom)?; // comma
                    arr.push(val);
                }
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(serde_json::Value::Array(arr))
    }

    fn visit_map<A>(self, mut map: A) -> Result<serde_json::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        self.check_depth().map_err(serde::de::Error::custom)?;
        self.add_size(2).map_err(serde::de::Error::custom)?; // { }
        let mut obj = serde_json::Map::new();
        loop {
            match map.next_key::<String>() {
                Ok(Some(key)) => {
                    let key_len = key.len() + 4; // quotes and colon
                    self.add_size(key_len).map_err(serde::de::Error::custom)?;
                    let val: serde_json::Value = map.next_value_seed(self.child())?;
                    obj.insert(key, val);
                    self.add_size(1).map_err(serde::de::Error::custom)?; // comma
                }
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(serde_json::Value::Object(obj))
    }
}

impl<'de> serde::de::DeserializeSeed<'de> for LimitedValueVisitor {
    type Value = serde_json::Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(self)
    }
}

/// External visitor for `Option<serde_json::Value>`: returns `None` for null/absent,
/// and `Some(value)` with depth/size limits for present values.
struct TrainingOptionVisitor;

impl<'de> serde::de::Visitor<'de> for TrainingOptionVisitor {
    type Value = Option<serde_json::Value>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("an optional JSON value")
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(None)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(None)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let visitor = LimitedValueVisitor::root(MAX_TRAINING_DEPTH, MAX_TRAINING_BYTES);
        let value: serde_json::Value = deserializer.deserialize_any(visitor)?;
        Ok(Some(value))
    }
}

/// Custom deserializer for `metadata.training` with depth and size limits.
pub(crate) fn deserialize_training<'de, D>(
    deserializer: D,
) -> Result<Option<serde_json::Value>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_option(TrainingOptionVisitor)
}
