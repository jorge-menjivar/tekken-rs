# Changelog

## 0.2.0 (unreleased)

### Added

- Support for tokenizer version v15, including the `model_settings_builder`
  section of `tekken.json` (new `tekken::model_settings` module with
  `ModelSettingsBuilder`, `ReasoningEffortBuilder`, `ModelSettings`, and
  `ReasoningEffort`), mirroring `mistral-common`.
- New v15-era special tokens: `[THINK]`, `[/THINK]`, `[MODEL_SETTINGS]`,
  `[/MODEL_SETTINGS]`, `[STREAMING_PAD]`, `[STREAMING_WORD]`,
  `[NEXT_AUDIO_TEXT]`, `[REPEAT_AUDIO_TEXT]`.
- `TokenizerVersion::V15`, `version_num()`, `supports_model_settings()`, and
  `requires_explicit_special_tokens()`.
- `TokenizerError::InvalidRequest` for request-level values rejected by model
  settings constraints (mirrors `InvalidRequestException` in `mistral-common`),
  distinct from `InvalidConfig` which indicates a broken tokenizer file.

### Breaking changes

- Tokenizer files of versions v11 and v13 without a `special_tokens` section
  are now rejected instead of silently falling back to a deprecated built-in
  special token list, matching `mistral-common`. Versions up to v7 keep the
  fallback.
- Model settings sections are parsed strictly (unknown keys are rejected and
  the `default` key of a field builder is required), matching
  `mistral-common`'s pydantic `extra="forbid"` models.
- New public field `ModelData::model_settings_builder` and new
  `SpecialTokens`/`TokenizerVersion` variants break exhaustive struct literals
  and matches in downstream code.
- The minimum supported Rust version is now 1.88.
