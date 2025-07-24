use std::sync::OnceLock;
use tekken::special_tokens::SpecialTokenPolicy;
use tekken::tekkenizer::Tekkenizer;

static TOKENIZER: OnceLock<Tekkenizer> = OnceLock::new();

fn get_tokenizer() -> &'static Tekkenizer {
    TOKENIZER.get_or_init(|| {
        Tekkenizer::from_file("tests/assets/tekken.json")
            .expect("Failed to load tokenizer from file")
    })
}

#[test]
fn test_byte_piece_functionality() {
    let tokenizer = get_tokenizer();

    // Test is_byte functionality with various inputs
    let test_strings = ["hello", "world", "a", "🚀", "café"];

    for text in &test_strings {
        let tokens = tokenizer.encode(text, false, false).unwrap();

        for &token in &tokens {
            let is_byte = tokenizer.is_byte(token);

            // Test id_to_byte_piece with different policies
            if is_byte {
                // For byte tokens, the improved implementation should handle UTF-8 errors gracefully
                let byte_piece_keep = tokenizer.id_to_byte_piece(token, SpecialTokenPolicy::Keep);
                let byte_piece_ignore =
                    tokenizer.id_to_byte_piece(token, SpecialTokenPolicy::Ignore);

                // Both should succeed now with improved error handling
                assert!(
                    byte_piece_keep.is_ok(),
                    "Keep policy should succeed for byte token {token}"
                );
                assert!(
                    byte_piece_ignore.is_ok(),
                    "Ignore policy should succeed for byte token {token}"
                );

                let keep_result = byte_piece_keep.unwrap();
                let _ignore_result = byte_piece_ignore.unwrap();

                // Keep should produce some bytes, ignore might be empty
                assert!(
                    !keep_result.is_empty(),
                    "Keep policy should produce bytes for byte token"
                );
            }
        }
    }
}

#[test]
fn test_special_token_handling() {
    let tokenizer = get_tokenizer();

    // Test getting control tokens
    assert!(
        tokenizer.bos_id().is_ok(),
        "Should be able to get BOS token ID"
    );
    assert!(
        tokenizer.eos_id().is_ok(),
        "Should be able to get EOS token ID"
    );
    assert!(
        tokenizer.pad_id().is_ok(),
        "Should be able to get PAD token ID"
    );
    assert!(
        tokenizer.unk_id().is_ok(),
        "Should be able to get UNK token ID"
    );

    let bos_id = tokenizer.bos_id().unwrap();
    let eos_id = tokenizer.eos_id().unwrap();
    let pad_id = tokenizer.pad_id().unwrap();
    let unk_id = tokenizer.unk_id().unwrap();

    // All special token IDs should be different
    assert_ne!(bos_id, eos_id, "BOS and EOS should have different IDs");
    assert_ne!(bos_id, pad_id, "BOS and PAD should have different IDs");
    assert_ne!(bos_id, unk_id, "BOS and UNK should have different IDs");
    assert_ne!(eos_id, pad_id, "EOS and PAD should have different IDs");
    assert_ne!(eos_id, unk_id, "EOS and UNK should have different IDs");
    assert_ne!(pad_id, unk_id, "PAD and UNK should have different IDs");

    // All should be within special token range
    assert!(
        tokenizer.is_special_token(bos_id),
        "BOS should be a special token"
    );
    assert!(
        tokenizer.is_special_token(eos_id),
        "EOS should be a special token"
    );
    assert!(
        tokenizer.is_special_token(pad_id),
        "PAD should be a special token"
    );
    assert!(
        tokenizer.is_special_token(unk_id),
        "UNK should be a special token"
    );
}

#[test]
fn test_id_to_piece_functionality() {
    let tokenizer = get_tokenizer();

    // Test with regular text
    let text = "Hello world";
    let tokens = tokenizer.encode(text, false, false).unwrap();

    for &token in &tokens {
        let piece = tokenizer.id_to_piece(token).unwrap();
        assert!(!piece.is_empty(), "Token piece should not be empty");
    }

    // Test with special tokens
    let bos_id = tokenizer.bos_id().unwrap();
    let bos_piece = tokenizer.id_to_piece(bos_id).unwrap();
    assert!(
        bos_piece.contains("s>") || bos_piece.contains("BOS") || bos_piece.contains("<s"),
        "BOS piece should look like a BOS token: {bos_piece}"
    );

    let eos_id = tokenizer.eos_id().unwrap();
    let eos_piece = tokenizer.id_to_piece(eos_id).unwrap();
    assert!(
        eos_piece.contains("/s>") || eos_piece.contains("EOS") || eos_piece.contains("</s"),
        "EOS piece should look like an EOS token: {eos_piece}"
    );
}

#[test]
fn test_decode_all_functionality() {
    let tokenizer = get_tokenizer();

    // Test decode_all with mixed special and regular tokens
    let text = "Hello world";
    let tokens = tokenizer.encode(text, true, true).unwrap(); // With BOS/EOS

    // Test with different policies
    let parts_keep = tokenizer
        .decode_all(&tokens, SpecialTokenPolicy::Keep)
        .unwrap();
    let parts_ignore = tokenizer
        .decode_all(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    // Should get some parts back
    assert!(
        !parts_keep.is_empty(),
        "Should get some parts with Keep policy"
    );
    assert!(
        !parts_ignore.is_empty(),
        "Should get some parts with Ignore policy"
    );

    // The concatenated result should match the original text (for ignore policy)
    let concatenated: String = parts_ignore.join("");
    assert_eq!(
        text, concatenated,
        "Concatenated parts should match original text"
    );

    // Test with Raise policy on tokens with special tokens - should fail
    assert!(
        tokenizer
            .decode_all(&tokens, SpecialTokenPolicy::Raise)
            .is_err(),
        "Should fail with Raise policy when special tokens are present"
    );
}

#[test]
fn test_vocabulary_access() {
    let tokenizer = get_tokenizer();

    let vocab = tokenizer.vocab();
    assert!(!vocab.is_empty(), "Vocabulary should not be empty");
    assert_eq!(
        vocab.len(),
        tokenizer.vocab_size(),
        "Vocab length should match vocab_size"
    );

    // Check that we have reasonable vocabulary entries
    let has_common_chars = vocab
        .iter()
        .any(|s| s.contains('a') || s.contains('e') || s.contains('t'));
    assert!(
        has_common_chars,
        "Vocabulary should contain common characters"
    );
}

#[test]
fn test_token_consistency() {
    let tokenizer = get_tokenizer();

    // Test that the same input always produces the same tokens
    let text = "Consistency test message";
    let tokens1 = tokenizer.encode(text, false, false).unwrap();
    let tokens2 = tokenizer.encode(text, false, false).unwrap();
    let tokens3 = tokenizer.encode(text, false, false).unwrap();

    assert_eq!(tokens1, tokens2, "Same input should produce same tokens");
    assert_eq!(tokens2, tokens3, "Same input should produce same tokens");

    // Test with BOS/EOS
    let tokens_bos_eos1 = tokenizer.encode(text, true, true).unwrap();
    let tokens_bos_eos2 = tokenizer.encode(text, true, true).unwrap();

    assert_eq!(
        tokens_bos_eos1, tokens_bos_eos2,
        "Same input with BOS/EOS should produce same tokens"
    );
}

#[test]
fn test_decode_consistency() {
    let tokenizer = get_tokenizer();

    // Test that decoding is consistent
    let text = "Decode consistency test";
    let tokens = tokenizer.encode(text, false, false).unwrap();

    let decoded1 = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();
    let decoded2 = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();
    let decoded3 = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(decoded1, decoded2, "Same tokens should decode to same text");
    assert_eq!(decoded2, decoded3, "Same tokens should decode to same text");
    assert_eq!(text, decoded1, "Decoded text should match original");
}

#[test]
fn test_different_bos_eos_combinations() {
    let tokenizer = get_tokenizer();
    let text = "Test message";

    let tokens_none = tokenizer.encode(text, false, false).unwrap();
    let tokens_bos_only = tokenizer.encode(text, true, false).unwrap();
    let tokens_eos_only = tokenizer.encode(text, false, true).unwrap();
    let tokens_both = tokenizer.encode(text, true, true).unwrap();

    // All should decode back to the original text when ignoring special tokens
    assert_eq!(
        text,
        tokenizer
            .decode(&tokens_none, SpecialTokenPolicy::Ignore)
            .unwrap()
    );
    assert_eq!(
        text,
        tokenizer
            .decode(&tokens_bos_only, SpecialTokenPolicy::Ignore)
            .unwrap()
    );
    assert_eq!(
        text,
        tokenizer
            .decode(&tokens_eos_only, SpecialTokenPolicy::Ignore)
            .unwrap()
    );
    assert_eq!(
        text,
        tokenizer
            .decode(&tokens_both, SpecialTokenPolicy::Ignore)
            .unwrap()
    );

    // Test length relationships
    assert!(
        tokens_bos_only.len() > tokens_none.len(),
        "BOS should add token(s)"
    );
    assert!(
        tokens_eos_only.len() > tokens_none.len(),
        "EOS should add token(s)"
    );
    assert!(
        tokens_both.len() > tokens_bos_only.len(),
        "Both should be longest"
    );
    assert!(
        tokens_both.len() > tokens_eos_only.len(),
        "Both should be longest"
    );

    // When keeping special tokens, BOS versions should start with BOS token
    let bos_id = tokenizer.bos_id().unwrap();
    let eos_id = tokenizer.eos_id().unwrap();

    assert_eq!(
        tokens_bos_only[0], bos_id,
        "BOS-only should start with BOS token"
    );
    assert_eq!(tokens_both[0], bos_id, "Both should start with BOS token");

    assert_eq!(
        tokens_eos_only[tokens_eos_only.len() - 1],
        eos_id,
        "EOS-only should end with EOS token"
    );
    assert_eq!(
        tokens_both[tokens_both.len() - 1],
        eos_id,
        "Both should end with EOS token"
    );
}

#[test]
fn test_special_token_policies_comprehensive() {
    let tokenizer = get_tokenizer();
    let text = "Policy test";

    // Encode with special tokens
    let tokens = tokenizer.encode(text, true, true).unwrap();

    // Test Keep policy
    let decoded_keep = tokenizer.decode(&tokens, SpecialTokenPolicy::Keep).unwrap();
    assert!(
        decoded_keep.len() > text.len(),
        "Keep policy should include special token strings"
    );
    assert!(
        decoded_keep.contains(text),
        "Keep policy should still contain original text"
    );

    // Test Ignore policy
    let decoded_ignore = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();
    assert_eq!(
        text, decoded_ignore,
        "Ignore policy should return only original text"
    );

    // Test Raise policy - should fail
    let raise_result = tokenizer.decode(&tokens, SpecialTokenPolicy::Raise);
    assert!(
        raise_result.is_err(),
        "Raise policy should fail when special tokens are present"
    );

    // Test Raise policy with no special tokens - should succeed
    let tokens_no_special = tokenizer.encode(text, false, false).unwrap();
    let decoded_raise = tokenizer
        .decode(&tokens_no_special, SpecialTokenPolicy::Raise)
        .unwrap();
    assert_eq!(
        text, decoded_raise,
        "Raise policy should work when no special tokens present"
    );
}

#[test]
fn test_get_control_token() {
    let tokenizer = get_tokenizer();

    // Test getting various control tokens
    let test_tokens = [
        ("[INST]", true),
        ("[/INST]", true),
        ("[TOOL_CALLS]", true),
        ("[IMG]", true),
        ("NonexistentToken", false),
        ("", false),
        ("regular text", false),
    ];

    for (token_str, should_exist) in &test_tokens {
        let result = tokenizer.get_control_token(token_str);
        if *should_exist {
            assert!(result.is_ok(), "Should find control token: {token_str}");
            let token_id = result.unwrap();
            assert!(
                (token_id as usize) < tokenizer.num_special_tokens(),
                "Control token should be in special token range"
            );
        } else {
            assert!(
                result.is_err(),
                "Should not find non-existent token: {token_str}"
            );
        }
    }
}

#[test]
fn test_version_and_config() {
    let tokenizer = get_tokenizer();

    // Test version information
    let version = tokenizer.version();
    assert_eq!(format!("{version:?}"), "V7", "Should be version V7");

    // Test size information
    assert_eq!(
        tokenizer.vocab_size(),
        131072,
        "Should have correct vocab size"
    );
    assert!(
        tokenizer.num_special_tokens() > 0,
        "Should have special tokens"
    );
    assert!(
        tokenizer.num_special_tokens() < tokenizer.vocab_size(),
        "Special tokens should be less than total vocab"
    );
}

#[test]
fn test_is_special_token_method() {
    let tokenizer = get_tokenizer();

    // Test that control tokens are identified as special
    let bos_id = tokenizer.bos_id().unwrap();
    let eos_id = tokenizer.eos_id().unwrap();
    let pad_id = tokenizer.pad_id().unwrap();
    let unk_id = tokenizer.unk_id().unwrap();

    assert!(tokenizer.is_special_token(bos_id), "BOS should be special");
    assert!(tokenizer.is_special_token(eos_id), "EOS should be special");
    assert!(tokenizer.is_special_token(pad_id), "PAD should be special");
    assert!(tokenizer.is_special_token(unk_id), "UNK should be special");

    // Test that regular tokens are not special
    let text = "hello";
    let tokens = tokenizer.encode(text, false, false).unwrap();
    for &token in &tokens {
        assert!(
            !tokenizer.is_special_token(token),
            "Regular text token {token} should not be special"
        );
    }

    // Test edge case: token at boundary
    let boundary_token = tokenizer.num_special_tokens() as u32;
    assert!(
        !tokenizer.is_special_token(boundary_token),
        "Token at special/regular boundary should not be special"
    );

    if boundary_token > 0 {
        let last_special = boundary_token - 1;
        assert!(
            tokenizer.is_special_token(last_special),
            "Last special token should be identified as special"
        );
    }
}

#[test]
fn test_error_conditions() {
    let tokenizer = get_tokenizer();

    // Test decoding invalid token IDs
    let invalid_token_id = tokenizer.vocab_size() as u32 + 1000;
    let result = tokenizer.id_to_piece(invalid_token_id);
    assert!(result.is_err(), "Should fail for invalid token ID");

    // Test id_to_byte_piece with invalid token ID
    let byte_result = tokenizer.id_to_byte_piece(invalid_token_id, SpecialTokenPolicy::Keep);
    assert!(
        byte_result.is_err(),
        "Should fail for invalid token ID in byte_piece"
    );

    // Test decode with empty token list
    let empty_tokens: Vec<u32> = vec![];
    let decoded = tokenizer
        .decode(&empty_tokens, SpecialTokenPolicy::Ignore)
        .unwrap();
    assert_eq!(
        decoded, "",
        "Empty token list should decode to empty string"
    );

    // Test decode_all with empty token list
    let parts = tokenizer
        .decode_all(&empty_tokens, SpecialTokenPolicy::Ignore)
        .unwrap();
    assert!(parts.is_empty(), "Empty token list should produce no parts");

    // Test get_control_token with invalid token string
    let invalid_control = tokenizer.get_control_token("NONEXISTENT_TOKEN");
    assert!(
        invalid_control.is_err(),
        "Should fail for nonexistent control token"
    );

    // The error message should be helpful
    if let Err(e) = invalid_control {
        let error_msg = format!("{e}");
        assert!(
            error_msg.contains("NONEXISTENT_TOKEN"),
            "Error should mention the invalid token"
        );
        assert!(
            error_msg.contains("Available special tokens"),
            "Error should list available tokens"
        );
    }
}
