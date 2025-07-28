use base64::{Engine as _, engine::general_purpose};
use tekken::config::{TokenInfo, TokenizerVersion};
use tekken::special_tokens::SpecialTokenInfo;
use tekken::tekkenizer::Tekkenizer;

#[test]
fn test_small_vocab() {
    println!("Testing Tekkenizer with small vocabulary...");

    // Create a minimal vocabulary (just byte tokens)
    let mut vocab = Vec::new();
    for i in 0..256 {
        let token_bytes = general_purpose::STANDARD.encode([i as u8]);
        vocab.push(TokenInfo {
            rank: i,
            token_bytes,
            token_str: Some(format!("byte_{i}")),
        });
    }

    // Add a few more tokens
    vocab.push(TokenInfo {
        rank: 256,
        token_bytes: general_purpose::STANDARD.encode(b"hello"),
        token_str: Some("hello".to_string()),
    });
    vocab.push(TokenInfo {
        rank: 257,
        token_bytes: general_purpose::STANDARD.encode(b"world"),
        token_str: Some("world".to_string()),
    });

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
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        268, // vocab_size (258 + 10)
        10,  // num_special_tokens
        TokenizerVersion::V7,
        None, // no audio config
    ).expect("Failed to create tokenizer");

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
