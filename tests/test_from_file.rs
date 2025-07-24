use tekken::special_tokens::SpecialTokenPolicy;
use tekken::tekkenizer::Tekkenizer;

#[test]
fn test_from_file() {
    // Test loading tokenizer from file
    println!("Testing Tekkenizer::from_file...");

    let tokenizer =
        Tekkenizer::from_file("tekken.json").expect("Failed to load tokenizer from file");
    println!("✓ Successfully loaded tokenizer from tekken.json");

    // Test basic properties
    println!("Vocab size: {}", tokenizer.vocab_size());
    println!(
        "Number of special tokens: {}",
        tokenizer.num_special_tokens()
    );
    println!("Version: {:?}", tokenizer.version());

    assert!(
        tokenizer.vocab_size() > 0,
        "Vocab size should be greater than 0"
    );
    assert!(
        tokenizer.num_special_tokens() > 0,
        "Should have special tokens"
    );

    // Test special token IDs
    let bos_id = tokenizer.bos_id().expect("Failed to get BOS ID");
    let eos_id = tokenizer.eos_id().expect("Failed to get EOS ID");
    let pad_id = tokenizer.pad_id().expect("Failed to get PAD ID");
    let unk_id = tokenizer.unk_id().expect("Failed to get UNK ID");

    println!("BOS ID: {}", bos_id);
    println!("EOS ID: {}", eos_id);
    println!("PAD ID: {}", pad_id);
    println!("UNK ID: {}", unk_id);

    // Test encoding
    let test_text = "Hello world! How are you today?";
    println!("\nTesting encoding of: '{}'", test_text);

    let tokens = tokenizer
        .encode(test_text, true, true)
        .expect("Failed to encode text");
    println!(
        "Encoded tokens ({} tokens): {:?}",
        tokens.len(),
        &tokens[..10.min(tokens.len())]
    );
    assert!(!tokens.is_empty(), "Tokens should not be empty");

    // Test decoding
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Keep)
        .expect("Failed to decode tokens");
    println!("Decoded text: '{}'", decoded);
    assert!(!decoded.is_empty(), "Decoded text should not be empty");

    // Test byte tokens
    println!("\nTesting byte tokens:");
    for i in 0..10 {
        let is_byte = tokenizer.is_byte(i + tokenizer.num_special_tokens() as u32);
        println!(
            "Token {} is byte: {}",
            i + tokenizer.num_special_tokens() as u32,
            is_byte
        );
    }

    // Test id_to_piece
    println!("\nTesting id_to_piece for first few tokens:");
    for i in 0..5 {
        let piece = tokenizer.id_to_piece(i).expect("Failed to get token piece");
        println!("Token {}: '{}'", i, piece);
        assert!(!piece.is_empty(), "Token piece should not be empty");
    }

    // Test audio support
    println!("\nAudio support: {}", tokenizer.has_audio_support());
    if let Some(audio_config) = tokenizer.audio_config() {
        println!(
            "Audio config present: sampling_rate = {}",
            audio_config.sampling_rate
        );
        assert!(
            audio_config.sampling_rate > 0,
            "Sampling rate should be positive"
        );
    }

    println!("\n✓ All tests passed!");
}
