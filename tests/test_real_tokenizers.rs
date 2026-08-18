//! Tests against real `tekken.json` files for every supported tokenizer version.
//!
//! The files are downloaded from pinned revisions of public Mistral models by
//! `scripts/download_test_tokenizers.sh` (CI runs it automatically). Locally,
//! each test is skipped if its file has not been downloaded — unless the
//! `TEKKEN_REQUIRE_REAL_TOKENIZERS` environment variable is set, in which case
//! a missing file is a hard failure. The expected token ids below were produced
//! by `mistral-common` 1.11.7 on the same files.

// token ids and vocab sizes are copied verbatim from mistral-common output; separators would make them harder to compare
#![allow(clippy::unreadable_literal)]

use std::path::PathBuf;

use tekken::config::TokenizerVersion;
use tekken::image::ImageConfig;
use tekken::model_settings::ReasoningEffort;
use tekken::special_tokens::SpecialTokenPolicy;
use tekken::tekkenizer::Tekkenizer;

fn real_tokenizer(version: &str) -> Option<Tekkenizer> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("tests/assets/tekken_{version}.json"));
    if !path.is_file() {
        assert!(
            std::env::var_os("TEKKEN_REQUIRE_REAL_TOKENIZERS").is_none(),
            "TEKKEN_REQUIRE_REAL_TOKENIZERS is set but {} does not exist",
            path.display()
        );
        eprintln!(
            "Skipping: {} not found (run scripts/download_test_tokenizers.sh to enable this test)",
            path.display()
        );
        return None;
    }
    Some(Tekkenizer::from_file(&path).unwrap_or_else(|e| {
        panic!("Failed to load {}: {e}", path.display());
    }))
}

/// Checks the image configuration every vision-capable Mistral model ships, and
/// the token grids `mistral-common` lays out for it.
fn check_image_support(tokenizer: &Tekkenizer) {
    assert!(tokenizer.has_image_support());
    assert_eq!(
        tokenizer.image_config(),
        Some(&ImageConfig::new(14, 1540, 2).unwrap())
    );

    // Real files put the image control tokens at these ranks.
    assert_eq!(tokenizer.get_control_token("[IMG]").unwrap(), 10);
    assert_eq!(tokenizer.get_control_token("[IMG_BREAK]").unwrap(), 12);
    assert_eq!(tokenizer.get_control_token("[IMG_END]").unwrap(), 13);

    let encoder = tokenizer.image_encoder().unwrap();
    for (width, height, expected_width, expected_height) in [
        (28, 28, 1, 1),
        (29, 28, 2, 1),
        (1024, 768, 37, 28),
        // Larger than max_image_size, so scaled down to 1540x1155 first
        (4000, 3000, 55, 42),
    ] {
        assert_eq!(
            encoder.image_to_num_tokens(width, height).unwrap(),
            (expected_width, expected_height),
            "{width}x{height} token grid"
        );
        assert_eq!(
            encoder.encode_dimensions(width, height).unwrap().len(),
            (expected_width + 1) * expected_height
        );
    }
}

/// Checks the properties shared by all real Mistral tekken files: sizes, control
/// token ids, and exact token ids as produced by `mistral-common`. All released
/// versions share the same underlying BPE vocabulary, so the expected ids are
/// identical across versions.
fn check_common(tokenizer: &Tekkenizer) {
    assert_eq!(tokenizer.vocab_size(), 131072);
    assert_eq!(tokenizer.num_special_tokens(), 1000);
    assert_eq!(tokenizer.bos_id().unwrap(), 1);
    assert_eq!(tokenizer.eos_id().unwrap(), 2);

    let cases: [(&str, &[u32]); 3] = [
        ("Hello, world!", &[1, 22177, 1044, 4304, 1033]),
        (
            "The quick brown fox jumps over the lazy dog.",
            &[
                1, 1784, 7586, 22980, 94137, 72993, 2136, 1278, 42757, 10575, 1046,
            ],
        ),
        (
            "función común días 日本語 🎉",
            &[
                1, 12127, 1963, 35239, 20608, 30367, 15199, 119685, 1142, 1137,
            ],
        ),
    ];
    for (text, expected) in cases {
        let tokens = tokenizer.encode(text, true, false).unwrap();
        assert_eq!(tokens, expected, "Token mismatch for {text:?}");

        let decoded = tokenizer
            .decode(&tokens, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(decoded, text, "Decode mismatch for {text:?}");
    }
}

#[test]
fn test_real_v3_tokenizer() {
    // Mistral-Nemo-Instruct-2407
    let Some(tokenizer) = real_tokenizer("v3") else {
        return;
    };
    assert_eq!(tokenizer.version(), &TokenizerVersion::V3);

    // v3-era files carry no special_tokens section: the deprecated fallback
    // list applies
    assert_eq!(tokenizer.get_control_token("[INST]").unwrap(), 3);
    assert_eq!(tokenizer.get_control_token("[/INST]").unwrap(), 4);

    // Text-only model: no image section in the file
    assert!(!tokenizer.has_image_support());

    check_common(&tokenizer);
}

#[test]
fn test_real_v7_tokenizer() {
    // Mistral-Small-24B-Instruct-2501
    let Some(tokenizer) = real_tokenizer("v7") else {
        return;
    };
    assert_eq!(tokenizer.version(), &TokenizerVersion::V7);

    // This v7 file also predates in-file special tokens, exercising the
    // deprecated fallback list
    assert_eq!(tokenizer.get_control_token("[SYSTEM_PROMPT]").unwrap(), 17);
    assert_eq!(tokenizer.get_control_token("[TOOL_CONTENT]").unwrap(), 19);

    assert!(!tokenizer.has_image_support());

    check_common(&tokenizer);
}

#[test]
fn test_real_v11_tokenizer() {
    // Mistral-Small-3.2-24B-Instruct-2506
    let Some(tokenizer) = real_tokenizer("v11") else {
        return;
    };
    assert_eq!(tokenizer.version(), &TokenizerVersion::V11);

    assert_eq!(tokenizer.get_control_token("[ARGS]").unwrap(), 32);
    assert_eq!(tokenizer.get_control_token("[CALL_ID]").unwrap(), 33);

    // v11 files still spell the image section "multimodal"
    check_image_support(&tokenizer);

    check_common(&tokenizer);
}

#[test]
fn test_real_v13_tokenizer() {
    // Ministral-3-8B-Instruct-2512
    let Some(tokenizer) = real_tokenizer("v13") else {
        return;
    };
    assert_eq!(tokenizer.version(), &TokenizerVersion::V13);

    assert_eq!(tokenizer.get_control_token("[ARGS]").unwrap(), 32);
    assert!(tokenizer.model_settings_builder().is_none());

    check_image_support(&tokenizer);

    check_common(&tokenizer);
}

#[test]
fn test_real_v15_tokenizer() {
    // Mistral-Small-4-119B-2603
    let Some(tokenizer) = real_tokenizer("v15") else {
        return;
    };
    assert_eq!(tokenizer.version(), &TokenizerVersion::V15);

    assert_eq!(tokenizer.get_control_token("[THINK]").unwrap(), 34);
    assert_eq!(tokenizer.get_control_token("[/THINK]").unwrap(), 35);
    assert_eq!(tokenizer.get_control_token("[MODEL_SETTINGS]").unwrap(), 36);
    assert_eq!(
        tokenizer.get_control_token("[/MODEL_SETTINGS]").unwrap(),
        37
    );

    // The model settings builder as shipped with the model
    let builder = tokenizer
        .model_settings_builder()
        .expect("Real v15 file should carry a model_settings_builder");
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

    check_image_support(&tokenizer);

    check_common(&tokenizer);
}
