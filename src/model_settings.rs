use serde::{Deserialize, Serialize};

use crate::errors::{Result, TokenizerError};

/// Controls how much reasoning effort the model should apply.
///
/// This mirrors `ReasoningEffort` from `mistral-common` and is used by
/// [`ModelSettingsBuilder`] to constrain which values a model accepts.
///
/// # Variants
///
/// * `None` - No additional reasoning effort
/// * `High` - High reasoning effort for complex tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    High,
}

/// Validated model configuration settings for a request.
///
/// This mirrors `ModelSettings` from `mistral-common`. Instances are typically
/// produced by [`ModelSettingsBuilder::build_settings`], which resolves defaults
/// and rejects unauthorized values.
///
/// # Fields
///
/// * `reasoning_effort` - The reasoning effort to apply, if any
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSettings {
    /// The reasoning effort the model should apply, if any.
    pub reasoning_effort: Option<ReasoningEffort>,
}

/// The kind of validation a field builder performs.
///
/// Currently only enum validation is defined, matching `ValidatorType`
/// in `mistral-common`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ValidatorType {
    #[default]
    Enum,
}

/// Deserializes an `Option` while still requiring the key to be present
/// (`null` maps to `None`), matching `mistral-common` where `default` is a
/// required key on field builders.
fn required_option<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<ReasoningEffort>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    serde::Deserialize::deserialize(deserializer)
}

/// Builder for the `reasoning_effort` field of [`ModelSettings`].
///
/// This mirrors `EnumBuilder[ReasoningEffort]` from `mistral-common`: it lists
/// the values a model accepts for reasoning effort, whether an unset value is
/// allowed, and the default used when the value is unset.
///
/// # Fields
///
/// * `validator_type` - The kind of validation performed (serialized as `type`, always `enum`)
/// * `accepts_none` - Whether an unset value is allowed in requests
/// * `default` - Default used when the value is unset (only valid if `accepts_none`)
/// * `values` - The authorized values
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReasoningEffortBuilder {
    /// The kind of validation performed (always `enum`).
    #[serde(rename = "type", default)]
    pub validator_type: ValidatorType,
    /// Whether an unset value is allowed in requests.
    pub accepts_none: bool,
    /// Default used when the value is unset. Only valid if `accepts_none` is true.
    #[serde(deserialize_with = "required_option")]
    pub default: Option<ReasoningEffort>,
    /// The authorized values.
    pub values: Vec<ReasoningEffort>,
}

impl ReasoningEffortBuilder {
    /// Validates the structural invariants of this builder.
    ///
    /// Mirrors the pydantic model validators of `EnumBuilder` in `mistral-common`:
    /// a default is only allowed when `accepts_none` is set, values must be unique,
    /// the value list must be non-empty unless `accepts_none` is set, and the
    /// default (if set) must be among the authorized values.
    ///
    /// # Errors
    ///
    /// Returns `TokenizerError::InvalidConfig` if any invariant is violated.
    pub fn validate(&self) -> Result<()> {
        if !self.accepts_none && self.default.is_some() {
            return Err(TokenizerError::InvalidConfig(format!(
                "Default values can only be defined for accepts_none fields (accepts_none={}, default={:?})",
                self.accepts_none, self.default
            )));
        }

        let mut seen = std::collections::HashSet::new();
        for value in &self.values {
            if !seen.insert(value) {
                return Err(TokenizerError::InvalidConfig(format!(
                    "Duplicate values in {:?}",
                    self.values
                )));
            }
        }

        if self.values.is_empty() && !self.accepts_none {
            return Err(TokenizerError::InvalidConfig(
                "Empty list of values while not accepts_none".to_string(),
            ));
        }

        if let Some(default) = self.default
            && !self.values.contains(&default)
        {
            return Err(TokenizerError::InvalidConfig(format!(
                "Default value {:?} is not in {:?}",
                default, self.values
            )));
        }

        Ok(())
    }

    /// Resolves and validates a field value, returning the final built result.
    ///
    /// An unset value resolves to the default if `accepts_none` is set, and is
    /// rejected otherwise. A set value must be among the authorized values.
    ///
    /// # Errors
    ///
    /// Returns `TokenizerError::InvalidRequest` if the value is invalid or
    /// missing when required.
    pub fn build_value(&self, value: Option<ReasoningEffort>) -> Result<Option<ReasoningEffort>> {
        let value = match value {
            Some(value) => Some(value),
            None => {
                if !self.accepts_none {
                    return Err(TokenizerError::InvalidRequest(
                        "reasoning_effort should be set for this model".to_string(),
                    ));
                }
                self.default
            }
        };

        if let Some(value) = value {
            if self.values.is_empty() {
                return Err(TokenizerError::InvalidRequest(
                    "reasoning_effort not supported for this model".to_string(),
                ));
            }
            if !self.values.contains(&value) {
                return Err(TokenizerError::InvalidRequest(format!(
                    "reasoning_effort should be one of {:?}, got {:?}",
                    self.values, value
                )));
            }
        }

        Ok(value)
    }
}

/// Builder for [`ModelSettings`] ensuring only authorized values are used.
///
/// This mirrors `ModelSettingsBuilder` from `mistral-common`. It is loaded from
/// the `model_settings_builder` section of a `tekken.json` file, which is only
/// valid for tokenizer versions v15 and later.
///
/// # Fields
///
/// * `reasoning_effort` - Builder for the allowed reasoning effort values, or None if unsupported
///
/// # Examples
///
/// ```rust
/// use tekken::model_settings::{ModelSettingsBuilder, ReasoningEffort};
///
/// let builder: ModelSettingsBuilder = serde_json::from_str(
///     r#"{"reasoning_effort": {"type": "enum", "accepts_none": true, "default": "none", "values": ["none", "high"]}}"#,
/// )?;
/// builder.validate()?;
///
/// let settings = builder.build_settings(Some(ReasoningEffort::High))?;
/// assert_eq!(settings.reasoning_effort, Some(ReasoningEffort::High));
///
/// // Unset values resolve to the default
/// let settings = builder.build_settings(None)?;
/// assert_eq!(settings.reasoning_effort, Some(ReasoningEffort::None));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSettingsBuilder {
    /// Builder for the allowed reasoning effort values, or None if unsupported.
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffortBuilder>,
}

impl ModelSettingsBuilder {
    /// Returns a builder with no field builders configured.
    ///
    /// Mirrors `ModelSettingsBuilder.none()` from `mistral-common`.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// Validates the structural invariants of all configured field builders.
    ///
    /// # Errors
    ///
    /// Returns `TokenizerError::InvalidConfig` if any field builder is invalid.
    pub fn validate(&self) -> Result<()> {
        if let Some(builder) = &self.reasoning_effort {
            builder.validate()?;
        }
        Ok(())
    }

    /// Builds validated [`ModelSettings`] from raw request values.
    ///
    /// Fields without a configured builder are left unset; fields with a builder
    /// are resolved (applying defaults) and validated.
    ///
    /// # Errors
    ///
    /// Returns `TokenizerError::InvalidRequest` if any value is invalid or
    /// missing when required.
    pub fn build_settings(
        &self,
        reasoning_effort: Option<ReasoningEffort>,
    ) -> Result<ModelSettings> {
        let reasoning_effort = match &self.reasoning_effort {
            Some(builder) => builder.build_value(reasoning_effort)?,
            None => None,
        };
        Ok(ModelSettings { reasoning_effort })
    }

    /// Validates that a [`ModelSettings`] instance matches the configured builders.
    ///
    /// Fields without a builder must be unset; fields with a builder must hold
    /// a value that passes validation.
    ///
    /// # Errors
    ///
    /// Returns `TokenizerError::InvalidRequest` if a field is set but has no
    /// builder, or fails its builder's validation.
    pub fn validate_settings(&self, settings: &ModelSettings) -> Result<()> {
        match &self.reasoning_effort {
            None => {
                if settings.reasoning_effort.is_some() {
                    return Err(TokenizerError::InvalidRequest(
                        "reasoning_effort not supported for this model".to_string(),
                    ));
                }
            }
            Some(builder) => match settings.reasoning_effort {
                None => {
                    if !(builder.accepts_none && builder.default.is_none()) {
                        return Err(TokenizerError::InvalidRequest(
                            "reasoning_effort should be set for this model".to_string(),
                        ));
                    }
                }
                Some(value) => {
                    builder.build_value(Some(value))?;
                }
            },
        }
        Ok(())
    }
}
