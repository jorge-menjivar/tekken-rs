use std::time::Instant;
use tekken::tekkenizer::Tekkenizer;

#[test]
fn test_full_vocab_profile() {
    println!("Profiling full vocabulary loading...");

    let start = Instant::now();
    let content =
        std::fs::read_to_string("tests/assets/tekken.json").expect("Failed to read tekken.json");
    println!("File read took: {:?}", start.elapsed());

    let start = Instant::now();
    let model_data: tekken::config::ModelData =
        serde_json::from_str(&content).expect("Failed to parse JSON");
    println!(
        "JSON parse took: {:?} - {} vocab entries",
        start.elapsed(),
        model_data.vocab.len()
    );

    let config = model_data.config;
    let special_tokens = model_data.special_tokens.unwrap_or_default();

    println!(
        "Using vocab size: {}, special tokens: {}",
        config.default_vocab_size, config.default_num_special_tokens
    );
    assert!(
        config.default_vocab_size > 0,
        "Vocab size should be positive"
    );
    assert!(
        config.default_num_special_tokens > 0,
        "Should have special tokens"
    );

    let start = Instant::now();
    println!(
        "Creating tokenizer with FULL vocabulary ({} tokens)...",
        model_data.vocab.len()
    );

    let tokenizer = Tekkenizer::new(
        model_data.vocab,
        &special_tokens,
        &config.pattern,
        config.default_vocab_size,
        config.default_num_special_tokens,
        tekken::config::TokenizerVersion::from_string(&config.version)
            .expect("Failed to parse version"),
        model_data.audio,
    )
    .expect("Failed to create tokenizer");

    println!("Full tokenizer creation took: {:?}", start.elapsed());
    let final_vocab_size = tokenizer.vocab_size();
    println!("Final vocab size: {final_vocab_size}");
    assert!(final_vocab_size > 0, "Final vocab size should be positive");

    // Test that it works
    let start = Instant::now();
    let tokens = tokenizer
        .encode("Hello world!", true, true)
        .expect("Failed to encode text");
    println!(
        "Encoding took: {:?} - Tokens: {:?}",
        start.elapsed(),
        &tokens[..5.min(tokens.len())]
    );
    assert!(!tokens.is_empty(), "Tokens should not be empty");
}
