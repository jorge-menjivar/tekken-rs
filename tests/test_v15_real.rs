//! Tests against the real v15 `tekken.json` shipped with Mistral-Small-4-119B-2603.
//!
//! The tokenizer file is downloaded by `scripts/download_v15_tokenizer.sh` (CI runs
//! it automatically). Locally, the test is skipped if the file has not been
//! downloaded — unless the `TEKKEN_V15_JSON` environment variable is set, in which
//! case a missing file is a hard failure. The expected token ids below were
//! produced by `mistral-common` 1.11.7 on the same file.

use std::path::PathBuf;

use tekken::config::TokenizerVersion;
use tekken::model_settings::ReasoningEffort;
use tekken::special_tokens::SpecialTokenPolicy;
use tekken::tekkenizer::Tekkenizer;

fn real_v15_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("TEKKEN_V15_JSON") {
        let path = PathBuf::from(path);
        assert!(
            path.is_file(),
            "TEKKEN_V15_JSON is set but {} does not exist",
            path.display()
        );
        return Some(path);
    }

    let default = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/assets/tekken_v15.json");
    if default.is_file() {
        Some(default)
    } else {
        eprintln!(
            "Skipping: {} not found (run scripts/download_v15_tokenizer.sh to enable this test)",
            default.display()
        );
        None
    }
}

#[test]
fn test_real_v15_tokenizer() {
    let Some(path) = real_v15_path() else {
        return;
    };

    let tokenizer = Tekkenizer::from_file(&path).expect("Failed to load real v15 tekken.json");

    assert_eq!(tokenizer.version(), &TokenizerVersion::V15);
    assert_eq!(tokenizer.vocab_size(), 131072);
    assert_eq!(tokenizer.num_special_tokens(), 1000);

    // v15-era special tokens at their real ranks
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

    // Token ids must match mistral-common exactly
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
