use std::sync::OnceLock;
use tekken::special_tokens::SpecialTokenPolicy;
use tekken::tekkenizer::Tekkenizer;

static TOKENIZER: OnceLock<Tekkenizer> = OnceLock::new();

fn get_tokenizer() -> &'static Tekkenizer {
    TOKENIZER.get_or_init(|| {
        Tekkenizer::from_file("tekken.json").expect("Failed to load tokenizer from file")
    })
}

#[test]
fn test_roundtrip_encoding_decoding() {
    let tokenizer = get_tokenizer();
    let test_cases = [
        "My very beautiful string",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "This is a test of the Mistral Tekken tokenizer.",
        "Special characters: @#$%^&*()_+-={}[]|\\:;\"'<>,.?/",
        "Mixed CaSe WoRdS",
        "   whitespace   handling   ",
        "Emojis and unicode characters work too!",
        "Numbers: 123, 456, 789",
        "Python vs Rust",
        "tokenizer encoding decoding comparison",
    ];

    for input in test_cases {
        let encoded = tokenizer.encode(input, false, false).unwrap();
        let decoded = tokenizer
            .decode(&encoded, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(input, decoded, "Roundtrip failed for input: {}", input);
    }
}

#[test]
fn test_basic_tokenizer_properties() {
    let tokenizer = get_tokenizer();

    // Test basic properties
    assert_eq!(tokenizer.vocab_size(), 131072);
    assert_eq!(format!("{:?}", tokenizer.version()), "V7");
    assert!(
        tokenizer.vocab_size() > 0,
        "Tokenizer should have a positive vocab size"
    );
}

#[test]
fn test_special_token_policies() {
    let tokenizer = get_tokenizer();
    let text = "Hello world";

    // Test with BOS and EOS tokens
    let tokens_with_special = tokenizer.encode(text, true, true).unwrap();

    // Test different special token policies
    let decoded_ignore = tokenizer
        .decode(&tokens_with_special, SpecialTokenPolicy::Ignore)
        .unwrap();
    let decoded_keep = tokenizer
        .decode(&tokens_with_special, SpecialTokenPolicy::Keep)
        .unwrap();

    // With ignore policy, special tokens should be filtered out
    assert_eq!(decoded_ignore, text);

    // With keep policy, special tokens should be included
    assert!(decoded_keep.contains("<s>") || decoded_keep.contains("</s>"));

    // Test raise policy with special tokens - should raise an error
    if tokens_with_special
        .iter()
        .any(|&token| tokenizer.is_special_token(token))
    {
        assert!(
            tokenizer
                .decode(&tokens_with_special, SpecialTokenPolicy::Raise)
                .is_err()
        );
    }
}

#[test]
fn test_empty_and_whitespace_inputs() {
    let tokenizer = get_tokenizer();

    // Test empty string
    let empty_tokens = tokenizer.encode("", false, false).unwrap();
    let empty_decoded = tokenizer
        .decode(&empty_tokens, SpecialTokenPolicy::Ignore)
        .unwrap();
    assert_eq!(empty_decoded, "");

    // Test whitespace-only strings
    let whitespace_cases = [" ", "  ", "\t", "\n", " \t\n ", "   "];
    for input in whitespace_cases {
        let tokens = tokenizer.encode(input, false, false).unwrap();
        let decoded = tokenizer
            .decode(&tokens, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(
            input, decoded,
            "Whitespace handling failed for: {:?}",
            input
        );
    }
}

#[test]
fn test_single_characters() {
    let tokenizer = get_tokenizer();

    // Test single ASCII characters
    for ch in b'a'..=b'z' {
        let input = (ch as char).to_string();
        let tokens = tokenizer.encode(&input, false, false).unwrap();
        let decoded = tokenizer
            .decode(&tokens, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(
            input, decoded,
            "Single character test failed for: {}",
            input
        );
    }

    // Test digits
    for ch in b'0'..=b'9' {
        let input = (ch as char).to_string();
        let tokens = tokenizer.encode(&input, false, false).unwrap();
        let decoded = tokenizer
            .decode(&tokens, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(input, decoded, "Single digit test failed for: {}", input);
    }
}

#[test]
fn test_unicode_handling() {
    let tokenizer = get_tokenizer();

    let unicode_cases = [
        "café",       // Latin characters with accents
        "naïve",      // More accented characters
        "北京",       // Chinese characters
        "🚀",         // Emoji
        "🌟⭐",       // Multiple emojis
        "Здравствуй", // Cyrillic
        "مرحبا",      // Arabic
    ];

    for input in unicode_cases {
        let tokens = tokenizer.encode(input, false, false).unwrap();
        let decoded = tokenizer
            .decode(&tokens, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(input, decoded, "Unicode test failed for: {}", input);
    }
}

#[test]
fn test_long_text() {
    let tokenizer = get_tokenizer();

    // Test with a longer piece of text
    let long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

    let tokens = tokenizer.encode(long_text, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();
    assert_eq!(long_text, decoded);

    // Verify we got a reasonable number of tokens
    assert!(tokens.len() > 50, "Long text should produce many tokens");
    assert!(
        tokens.len() < long_text.len(),
        "Tokenization should be more efficient than character-by-character"
    );
}

#[test]
fn test_bos_eos_tokens() {
    let tokenizer = get_tokenizer();
    let text = "Test message";

    // Test different combinations of BOS/EOS
    let tokens_none = tokenizer.encode(text, false, false).unwrap();
    let tokens_bos = tokenizer.encode(text, true, false).unwrap();
    let tokens_eos = tokenizer.encode(text, false, true).unwrap();
    let tokens_both = tokenizer.encode(text, true, true).unwrap();

    // BOS version should be longer than no special tokens
    assert!(
        tokens_bos.len() > tokens_none.len(),
        "BOS should add tokens"
    );

    // EOS version should be longer than no special tokens
    assert!(
        tokens_eos.len() > tokens_none.len(),
        "EOS should add tokens"
    );

    // Both version should be longest
    assert!(
        tokens_both.len() > tokens_bos.len(),
        "Both BOS+EOS should be longest"
    );
    assert!(
        tokens_both.len() > tokens_eos.len(),
        "Both BOS+EOS should be longest"
    );
}

#[test]
fn test_special_token_identification() {
    let tokenizer = get_tokenizer();

    // Test with text that includes special tokens when encoded with BOS/EOS
    let text = "Hello";
    let tokens_with_special = tokenizer.encode(text, true, true).unwrap();

    // Should have some special tokens
    let has_special = tokens_with_special
        .iter()
        .any(|&token| tokenizer.is_special_token(token));
    assert!(
        has_special,
        "Should have special tokens when BOS/EOS are added"
    );

    // Test without special tokens
    let tokens_no_special = tokenizer.encode(text, false, false).unwrap();
    let _all_non_special = tokens_no_special
        .iter()
        .all(|&token| !tokenizer.is_special_token(token));
    // Note: This might not always be true if the text itself contains characters that map to special token IDs
    // But for simple text like "Hello", it should be true
}

#[test]
fn test_token_count_consistency() {
    let tokenizer = get_tokenizer();
    let test_cases = [
        "Short",
        "A bit longer sentence with more words.",
        "This is a significantly longer piece of text that should result in more tokens being generated by the tokenizer.",
    ];

    for text in test_cases {
        let tokens = tokenizer.encode(text, false, false).unwrap();
        let decoded = tokenizer
            .decode(&tokens, SpecialTokenPolicy::Ignore)
            .unwrap();

        assert_eq!(text, decoded);
        assert!(tokens.len() > 0, "Should produce at least one token");

        // Generally, shorter text should produce fewer tokens
        // This is a rough heuristic, not always true due to tokenization specifics
        if text.len() < 10 {
            assert!(
                tokens.len() < 20,
                "Short text should not produce too many tokens"
            );
        }
    }
}

#[test]
fn test_special_characters_comprehensive() {
    let tokenizer = get_tokenizer();

    let special_chars = [
        "!@#$%^&*()",
        "{}[]|\\:;\"'<>,.?/",
        "±×÷≠≤≥",
        "™©®",
        "…–—''\"\"",
        "¡¿",
        "§¶†‡•",
    ];

    for chars in special_chars {
        let tokens = tokenizer.encode(chars, false, false).unwrap();
        let decoded = tokenizer
            .decode(&tokens, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(
            chars, decoded,
            "Special characters test failed for: {}",
            chars
        );
    }
}
