use std::time::Instant;
use tekken::tekkenizer::Tekkenizer;

#[test]
fn test_profile_loading() {
    println!("Profiling Tekkenizer::from_file loading...");

    let start = Instant::now();
    println!("Step 1: Reading file...");
    let content =
        std::fs::read_to_string("tests/assets/tekken.json").expect("Failed to read tekken.json");
    println!(
        "Step 1 took: {:?} - File size: {} bytes",
        start.elapsed(),
        content.len()
    );
    assert!(!content.is_empty(), "File content should not be empty");

    let start = Instant::now();
    println!("Step 2: Parsing JSON...");
    let model_data: tekken::config::ModelData =
        serde_json::from_str(&content).expect("Failed to parse JSON");
    println!(
        "Step 2 took: {:?} - Vocab entries: {}",
        start.elapsed(),
        model_data.vocab.len()
    );
    assert!(!model_data.vocab.is_empty(), "Vocab should not be empty");

    let start = Instant::now();
    println!("Step 3: Processing vocabulary (first 1000 tokens only)...");
    let small_vocab: Vec<_> = model_data.vocab.into_iter().take(1000).collect();
    let config = model_data.config;
    let special_tokens = model_data.special_tokens.unwrap_or_default();
    println!("Step 3 took: {:?}", start.elapsed());
    assert_eq!(
        small_vocab.len(),
        1000,
        "Should have exactly 1000 vocab tokens"
    );

    let start = Instant::now();
    println!("Step 4: Creating tokenizer with small vocab...");
    let tokenizer = Tekkenizer::new(
        small_vocab,
        &special_tokens,
        &config.pattern,
        1000 + config.default_num_special_tokens,
        config.default_num_special_tokens,
        tekken::config::TokenizerVersion::from_string(&config.version)
            .expect("Failed to parse version"),
        model_data.audio,
    )
    .expect("Failed to create tokenizer");
    println!("Step 4 took: {:?}", start.elapsed());

    let total_vocab_size = tokenizer.vocab_size();
    println!("Total vocab size: {total_vocab_size}");
    assert!(total_vocab_size > 0, "Total vocab size should be positive");

    // Test encoding with small vocab
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
