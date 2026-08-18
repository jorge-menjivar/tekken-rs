mod common;

use tekken::config::TokenizerVersion;
use tekken::special_tokens::SpecialTokenInfo;
use tekken::tekkenizer::Tekkenizer;

#[test]
fn test_small_vocab() {
    println!("Testing Tekkenizer with small vocabulary...");

    let vocab = common::small_vocab();

    // Create special tokens
    let special_tokens = vec![
        SpecialTokenInfo {
            rank: 0,
            token_str: "<unk>".to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 1,
            token_str: "<s>".to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 2,
            token_str: "</s>".to_string(),
            is_control: true,
        },
    ];

    println!("Creating tokenizer with {} vocab tokens...", vocab.len());
    assert_eq!(
        vocab.len(),
        258,
        "Should have 258 vocab tokens (256 bytes + 2 extra)"
    );

    let tokenizer = Tekkenizer::new(
        vocab,
        &special_tokens,
        common::PATTERN,
        268, // vocab_size (258 + 10)
        10,  // num_special_tokens
        TokenizerVersion::V7,
        None, // no audio config
    )
    .expect("Failed to create tokenizer");

    println!("✓ Tokenizer created successfully!");
    let vocab_size = tokenizer.vocab_size();
    let num_special_tokens = tokenizer.num_special_tokens();
    println!("Vocab size: {vocab_size}");
    println!("Special tokens: {num_special_tokens}");

    assert_eq!(vocab_size, 268, "Vocab size should be 268");
    assert_eq!(num_special_tokens, 10, "Should have 10 special tokens");

    // Test encoding/decoding
    let text = "hello world";
    println!("Testing with text: '{text}'");

    let tokens = tokenizer
        .encode(text, true, true)
        .expect("Failed to encode text");
    println!("Tokens: {tokens:?}");
    assert!(!tokens.is_empty(), "Tokens should not be empty");

    let decoded = tokenizer
        .decode(&tokens, tekken::special_tokens::SpecialTokenPolicy::Keep)
        .expect("Failed to decode tokens");
    println!("Decoded: '{decoded}'");
    assert!(!decoded.is_empty(), "Decoded text should not be empty");

    println!("✓ All tests passed!");
}
