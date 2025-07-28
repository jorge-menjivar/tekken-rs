use std::fs;
use std::path::Path;
use tekken::special_tokens::SpecialTokenPolicy;
use tekken::tekkenizer::Tekkenizer;

#[test]
fn test_to_file() {
    // Load original tokenizer
    println!("Testing Tekkenizer::to_file...");

    let original_tokenizer = Tekkenizer::from_file("tests/assets/tekken.json")
        .expect("Failed to load original tokenizer");
    println!("✓ Successfully loaded original tokenizer");

    // Save to a new file
    let test_output_path = "tests/assets/test_to_file_output.json";
    original_tokenizer
        .to_file(test_output_path)
        .expect("Failed to save tokenizer to file");
    println!("✓ Successfully saved tokenizer to {}", test_output_path);

    // Verify file exists
    assert!(
        Path::new(test_output_path).exists(),
        "Output file should exist"
    );

    // Load the saved tokenizer
    let loaded_tokenizer =
        Tekkenizer::from_file(test_output_path).expect("Failed to load saved tokenizer");
    println!("✓ Successfully loaded saved tokenizer");

    // Compare basic properties
    assert_eq!(
        original_tokenizer.vocab_size(),
        loaded_tokenizer.vocab_size(),
        "Vocab sizes should match"
    );
    assert_eq!(
        original_tokenizer.num_special_tokens(),
        loaded_tokenizer.num_special_tokens(),
        "Number of special tokens should match"
    );
    assert_eq!(
        original_tokenizer.version(),
        loaded_tokenizer.version(),
        "Versions should match"
    );

    // Compare special token IDs
    assert_eq!(
        original_tokenizer.bos_id().unwrap(),
        loaded_tokenizer.bos_id().unwrap(),
        "BOS IDs should match"
    );
    assert_eq!(
        original_tokenizer.eos_id().unwrap(),
        loaded_tokenizer.eos_id().unwrap(),
        "EOS IDs should match"
    );
    assert_eq!(
        original_tokenizer.pad_id().unwrap(),
        loaded_tokenizer.pad_id().unwrap(),
        "PAD IDs should match"
    );
    assert_eq!(
        original_tokenizer.unk_id().unwrap(),
        loaded_tokenizer.unk_id().unwrap(),
        "UNK IDs should match"
    );

    // Test encoding/decoding produces same results
    let test_texts = vec![
        "Hello world!",
        "How are you today?",
        "Testing special characters: @#$%^&*()",
        "Numbers: 123456789",
        "Unicode: 你好世界 🌍",
        "Empty spaces:     ",
        "Newlines\nand\ttabs",
    ];

    for text in &test_texts {
        println!("\nTesting text: '{}'", text);

        // Test encoding
        let original_tokens = original_tokenizer
            .encode(text, true, true)
            .expect("Failed to encode with original tokenizer");
        let loaded_tokens = loaded_tokenizer
            .encode(text, true, true)
            .expect("Failed to encode with loaded tokenizer");

        assert_eq!(
            original_tokens, loaded_tokens,
            "Encoded tokens should match for text: '{}'",
            text
        );

        // Test decoding
        let original_decoded = original_tokenizer
            .decode(&original_tokens, SpecialTokenPolicy::Keep)
            .expect("Failed to decode with original tokenizer");
        let loaded_decoded = loaded_tokenizer
            .decode(&loaded_tokens, SpecialTokenPolicy::Keep)
            .expect("Failed to decode with loaded tokenizer");

        assert_eq!(
            original_decoded, loaded_decoded,
            "Decoded text should match for tokens: {:?}",
            original_tokens
        );
    }

    // Test id_to_piece for a sample of tokens
    println!("\nTesting id_to_piece consistency...");
    for token_id in 0..100 {
        let original_piece = original_tokenizer.id_to_piece(token_id);
        let loaded_piece = loaded_tokenizer.id_to_piece(token_id);

        assert_eq!(
            original_piece.unwrap(),
            loaded_piece.unwrap(),
            "Token piece should match for ID: {}",
            token_id
        );
    }

    // Test audio configuration
    assert_eq!(
        original_tokenizer.has_audio_support(),
        loaded_tokenizer.has_audio_support(),
        "Audio support should match"
    );

    if let (Some(original_audio), Some(loaded_audio)) = (
        original_tokenizer.audio_config(),
        loaded_tokenizer.audio_config(),
    ) {
        assert_eq!(
            original_audio.sampling_rate, loaded_audio.sampling_rate,
            "Audio sampling rates should match"
        );
    }

    // Clean up test file
    fs::remove_file(test_output_path).expect("Failed to clean up test file");
    println!("\n✓ Cleaned up test artifacts");

    println!("\n✓ All to_file tests passed!");
}

#[test]
fn test_to_file_roundtrip_multiple() {
    // Test multiple save/load cycles to ensure consistency
    println!("Testing multiple save/load cycles...");

    let original_tokenizer = Tekkenizer::from_file("tests/assets/tekken.json")
        .expect("Failed to load original tokenizer");

    let test_text = "Testing roundtrip consistency!";
    let original_tokens = original_tokenizer
        .encode(test_text, true, true)
        .expect("Failed to encode text");

    // Perform multiple save/load cycles
    for i in 1..=3 {
        let path = format!("tests/assets/test_roundtrip_{}.json", i);

        if i == 1 {
            original_tokenizer
                .to_file(&path)
                .expect("Failed to save tokenizer");
        } else {
            let prev_path = format!("tests/assets/test_roundtrip_{}.json", i - 1);
            let prev_tokenizer =
                Tekkenizer::from_file(&prev_path).expect("Failed to load previous tokenizer");
            prev_tokenizer
                .to_file(&path)
                .expect("Failed to save tokenizer");

            // Clean up previous file
            fs::remove_file(prev_path).ok();
        }

        let loaded_tokenizer = Tekkenizer::from_file(&path).expect("Failed to load tokenizer");

        let loaded_tokens = loaded_tokenizer
            .encode(test_text, true, true)
            .expect("Failed to encode text");

        assert_eq!(
            original_tokens, loaded_tokens,
            "Tokens should remain consistent after {} save/load cycles",
            i
        );

        if i == 3 {
            // Clean up final file
            fs::remove_file(&path).ok();
        }
    }

    println!("✓ Multiple roundtrip tests passed!");
}
