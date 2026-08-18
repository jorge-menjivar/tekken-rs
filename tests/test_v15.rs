mod common;

use serde_json::json;
use tekken::config::TokenizerVersion;
use tekken::errors::TokenizerError;
use tekken::model_settings::{ModelSettingsBuilder, ReasoningEffort, ReasoningEffortBuilder};
use tekken::special_tokens::SpecialTokenPolicy;
use tekken::tekkenizer::Tekkenizer;

fn small_vocab_json() -> serde_json::Value {
    serde_json::to_value(common::small_vocab()).unwrap()
}

fn v15_special_tokens_json() -> Vec<serde_json::Value> {
    [
        "<unk>",
        "<s>",
        "</s>",
        "[THINK]",
        "[/THINK]",
        "[MODEL_SETTINGS]",
        "[/MODEL_SETTINGS]",
        "[STREAMING_PAD]",
        "[STREAMING_WORD]",
        "[NEXT_AUDIO_TEXT]",
        "[REPEAT_AUDIO_TEXT]",
    ]
    .iter()
    .enumerate()
    .map(|(rank, token_str)| {
        json!({
            "rank": rank,
            "token_str": token_str,
            "is_control": true,
        })
    })
    .collect()
}

/// Builds a complete tekken.json document for the given version.
fn tokenizer_json(
    version: &str,
    special_tokens: Option<Vec<serde_json::Value>>,
    model_settings_builder: Option<serde_json::Value>,
) -> serde_json::Value {
    let num_special_tokens = 15;
    let mut doc = json!({
        "vocab": small_vocab_json(),
        "config": {
            "pattern": common::PATTERN,
            "num_vocab_tokens": 258,
            "default_vocab_size": 258 + num_special_tokens,
            "default_num_special_tokens": num_special_tokens,
            "version": version,
        },
    });
    if let Some(special_tokens) = special_tokens {
        doc["special_tokens"] = json!(special_tokens);
    }
    if let Some(builder) = model_settings_builder {
        doc["model_settings_builder"] = builder;
    }
    doc
}

fn write_tokenizer_file(doc: &serde_json::Value) -> tempfile::NamedTempFile {
    let file = tempfile::NamedTempFile::new().expect("Failed to create temp file");
    std::fs::write(file.path(), serde_json::to_string(doc).unwrap())
        .expect("Failed to write tokenizer file");
    file
}

fn reasoning_effort_builder_json() -> serde_json::Value {
    json!({
        "reasoning_effort": {
            "type": "enum",
            "accepts_none": true,
            "default": "none",
            "values": ["none", "high"],
        }
    })
}

#[test]
fn test_v15_version_properties() {
    assert_eq!(
        TokenizerVersion::from_string("v15"),
        Some(TokenizerVersion::V15)
    );
    assert_eq!(TokenizerVersion::V15.as_str(), "v15");
    assert_eq!(TokenizerVersion::V15.version_num(), 15);

    // Versions are ordered
    assert!(TokenizerVersion::V13 < TokenizerVersion::V15);
    assert!(TokenizerVersion::V15 > TokenizerVersion::V7);

    // Only v15+ supports model settings
    assert!(TokenizerVersion::V15.supports_model_settings());
    for version in [
        TokenizerVersion::V3,
        TokenizerVersion::V7,
        TokenizerVersion::V11,
        TokenizerVersion::V13,
    ] {
        assert!(!version.supports_model_settings(), "{version:?}");
    }
}

#[test]
fn test_v15_from_file() {
    let doc = tokenizer_json(
        "v15",
        Some(v15_special_tokens_json()),
        Some(reasoning_effort_builder_json()),
    );
    let file = write_tokenizer_file(&doc);

    let tokenizer = Tekkenizer::from_file(file.path()).expect("Failed to load v15 tokenizer");
    assert_eq!(tokenizer.version(), &TokenizerVersion::V15);

    // The new v15-era special tokens resolve to their ranks
    assert_eq!(tokenizer.get_control_token("[THINK]").unwrap(), 3);
    assert_eq!(tokenizer.get_control_token("[/THINK]").unwrap(), 4);
    assert_eq!(tokenizer.get_control_token("[MODEL_SETTINGS]").unwrap(), 5);
    assert_eq!(tokenizer.get_control_token("[/MODEL_SETTINGS]").unwrap(), 6);

    // The model settings builder is parsed
    let builder = tokenizer
        .model_settings_builder()
        .expect("Model settings builder should be present");
    let effort = builder
        .reasoning_effort
        .as_ref()
        .expect("reasoning_effort builder should be present");
    assert!(effort.accepts_none);
    assert_eq!(effort.default, Some(ReasoningEffort::None));
    assert_eq!(
        effort.values,
        vec![ReasoningEffort::None, ReasoningEffort::High]
    );

    // Settings resolution follows the builder's constraints
    assert_eq!(
        builder.build_settings(None).unwrap().reasoning_effort,
        Some(ReasoningEffort::None)
    );
    assert_eq!(
        builder
            .build_settings(Some(ReasoningEffort::High))
            .unwrap()
            .reasoning_effort,
        Some(ReasoningEffort::High)
    );

    // Text round-trips as usual
    let tokens = tokenizer.encode("hello world", true, true).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();
    assert_eq!(decoded, "hello world");
}

#[test]
fn test_v15_to_file_round_trip() {
    let doc = tokenizer_json(
        "v15",
        Some(v15_special_tokens_json()),
        Some(reasoning_effort_builder_json()),
    );
    let file = write_tokenizer_file(&doc);
    let tokenizer = Tekkenizer::from_file(file.path()).expect("Failed to load v15 tokenizer");

    let out_file = tempfile::NamedTempFile::new().expect("Failed to create temp file");
    tokenizer
        .to_file(out_file.path())
        .expect("Failed to save v15 tokenizer");

    let reloaded = Tekkenizer::from_file(out_file.path()).expect("Failed to reload v15 tokenizer");
    assert_eq!(reloaded.version(), &TokenizerVersion::V15);
    assert_eq!(
        reloaded.model_settings_builder(),
        tokenizer.model_settings_builder()
    );
}

#[test]
fn test_model_settings_builder_rejected_below_v15() {
    let doc = tokenizer_json(
        "v13",
        Some(v15_special_tokens_json()),
        Some(reasoning_effort_builder_json()),
    );
    let file = write_tokenizer_file(&doc);

    let err = Tekkenizer::from_file(file.path())
        .err()
        .expect("Loading a v13 file with model_settings_builder should fail");
    assert!(
        err.to_string().contains("model_settings_builder"),
        "Unexpected error: {err}"
    );
}

#[test]
fn test_missing_special_tokens_rejected_above_v7() {
    for version in ["v11", "v13", "v15"] {
        let doc = tokenizer_json(version, None, None);
        let file = write_tokenizer_file(&doc);

        let err = Tekkenizer::from_file(file.path())
            .err()
            .expect("Loading a file without special tokens should fail for versions > v7");
        assert!(
            err.to_string().contains("Special tokens not found"),
            "Unexpected error for {version}: {err}"
        );
    }
}

#[test]
fn test_missing_special_tokens_allowed_up_to_v7() {
    // Versions <= v7 fall back to the deprecated special token list (20 tokens)
    let num_special_tokens = 25;
    let doc = json!({
        "vocab": small_vocab_json(),
        "config": {
            "pattern": common::PATTERN,
            "num_vocab_tokens": 258,
            "default_vocab_size": 258 + num_special_tokens,
            "default_num_special_tokens": num_special_tokens,
            "version": "v7",
        },
    });
    let file = write_tokenizer_file(&doc);

    let tokenizer =
        Tekkenizer::from_file(file.path()).expect("v7 should allow missing special tokens");
    assert_eq!(tokenizer.get_control_token("<s>").unwrap(), 1);
}

#[test]
fn test_reasoning_effort_builder_validation() {
    // Default without accepts_none is rejected
    let builder = ReasoningEffortBuilder {
        accepts_none: false,
        default: Some(ReasoningEffort::None),
        values: vec![ReasoningEffort::None, ReasoningEffort::High],
        ..Default::default()
    };
    assert!(builder.validate().is_err());

    // Duplicate values are rejected
    let builder = ReasoningEffortBuilder {
        accepts_none: true,
        default: None,
        values: vec![ReasoningEffort::High, ReasoningEffort::High],
        ..Default::default()
    };
    assert!(builder.validate().is_err());

    // Empty values without accepts_none are rejected
    let builder = ReasoningEffortBuilder {
        accepts_none: false,
        default: None,
        values: vec![],
        ..Default::default()
    };
    assert!(builder.validate().is_err());

    // Default outside the allowed values is rejected
    let builder = ReasoningEffortBuilder {
        accepts_none: true,
        default: Some(ReasoningEffort::None),
        values: vec![ReasoningEffort::High],
        ..Default::default()
    };
    assert!(builder.validate().is_err());

    // A well-formed builder passes and enforces its constraints; rejected
    // request values surface as InvalidRequest, not InvalidConfig
    let builder = ReasoningEffortBuilder {
        accepts_none: false,
        default: None,
        values: vec![ReasoningEffort::High],
        ..Default::default()
    };
    builder.validate().expect("Builder should be valid");
    assert!(matches!(
        builder.build_value(None),
        Err(TokenizerError::InvalidRequest(_))
    ));
    assert!(matches!(
        builder.build_value(Some(ReasoningEffort::None)),
        Err(TokenizerError::InvalidRequest(_))
    ));
    assert_eq!(
        builder.build_value(Some(ReasoningEffort::High)).unwrap(),
        Some(ReasoningEffort::High)
    );
}

#[test]
fn test_model_settings_builder_strict_parsing() {
    // Unknown keys are rejected, mirroring mistral-common's extra="forbid"
    let unknown_key: Result<ModelSettingsBuilder, _> = serde_json::from_str(
        r#"{"reasoning_effort": {"type": "enum", "accepts_none": true, "default": "none", "values": ["none"], "extra": 1}}"#,
    );
    assert!(unknown_key.is_err());

    let unknown_field_builder: Result<ModelSettingsBuilder, _> = serde_json::from_str(
        r#"{"temperature": {"type": "enum", "accepts_none": true, "default": null, "values": []}}"#,
    );
    assert!(unknown_field_builder.is_err());

    // The `default` key is required, though it may be null
    let missing_default: Result<ModelSettingsBuilder, _> = serde_json::from_str(
        r#"{"reasoning_effort": {"type": "enum", "accepts_none": true, "values": ["none"]}}"#,
    );
    assert!(missing_default.is_err());

    let null_default: ModelSettingsBuilder = serde_json::from_str(
        r#"{"reasoning_effort": {"type": "enum", "accepts_none": true, "default": null, "values": ["none"]}}"#,
    )
    .expect("null default should parse");
    assert_eq!(null_default.reasoning_effort.unwrap().default, None);
}

#[test]
fn test_model_settings_builder_without_fields() {
    // A builder with no field builders rejects set values and builds empty settings
    let builder = ModelSettingsBuilder::none();
    builder.validate().expect("Empty builder should be valid");

    let settings = builder.build_settings(Some(ReasoningEffort::High)).unwrap();
    assert_eq!(settings.reasoning_effort, None);
    assert!(builder.validate_settings(&settings).is_ok());

    let set_settings = tekken::model_settings::ModelSettings {
        reasoning_effort: Some(ReasoningEffort::High),
    };
    assert!(builder.validate_settings(&set_settings).is_err());
}
