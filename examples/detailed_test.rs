use tekken::special_tokens::SpecialTokenPolicy;
use tekken::tekkenizer::Tekkenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Detailed Tekken Tokenizer Testing ===\n");

    // Load tokenizer
    let tokenizer = Tekkenizer::from_file("tests/assets/tekken.json")?;
    println!("✅ Tokenizer loaded successfully!");
    println!("📊 Vocab size: {}", tokenizer.vocab_size());
    println!("📝 Version: {:?}", tokenizer.version());
    println!("🔢 Special tokens: {}", tokenizer.num_special_tokens());

    // Test special token IDs
    println!("\n🎯 Special Token IDs:");
    println!("  BOS: {}", tokenizer.bos_id()?);
    println!("  EOS: {}", tokenizer.eos_id()?);
    println!("  PAD: {}", tokenizer.pad_id()?);
    println!("  UNK: {}", tokenizer.unk_id()?);

    // Test various text samples
    let test_cases = vec![
        "Hello, world!",
        "This is a test of the tokenizer.",
        "The quick brown fox jumps over the lazy dog.",
        "Testing 123 numbers and symbols @#$%",
        "Multi\nline\ttext with special chars",
        "🚀 Unicode emojis and 日本語 text",
        "",  // Empty string
        " ", // Just space
    ];

    println!("\n📝 Text Encoding/Decoding Tests:");
    for (i, text) in test_cases.iter().enumerate() {
        println!("\n--- Test Case {} ---", i + 1);
        println!("Input: {text:?}");

        // Test with and without BOS/EOS
        let tokens_with_special = tokenizer.encode(text, true, true)?;
        let tokens_without_special = tokenizer.encode(text, false, false)?;

        println!("Tokens (with BOS/EOS): {tokens_with_special:?}");
        println!("Tokens (without): {tokens_without_special:?}");
        println!(
            "Token count (with): {}, (without): {}",
            tokens_with_special.len(),
            tokens_without_special.len()
        );

        // Test decoding with different policies
        let decoded_keep = tokenizer.decode(&tokens_with_special, SpecialTokenPolicy::Keep)?;
        let decoded_ignore = tokenizer.decode(&tokens_with_special, SpecialTokenPolicy::Ignore)?;

        println!("Decoded (keep special): {decoded_keep:?}");
        println!("Decoded (ignore special): {decoded_ignore:?}");

        // Verify round-trip accuracy
        let is_roundtrip_exact = decoded_ignore == *text;
        println!(
            "Round-trip exact: {}",
            if is_roundtrip_exact { "✅" } else { "❌" }
        );

        if !is_roundtrip_exact {
            println!("  Expected: {text:?}");
            println!("  Got:      {decoded_ignore:?}");
        }
    }

    // Test individual token inspection
    println!("\n🔍 Token Inspection:");
    let sample_text = "Hello world";
    let tokens = tokenizer.encode(sample_text, false, false)?;

    for (i, &token_id) in tokens.iter().enumerate() {
        let piece = tokenizer.id_to_piece(token_id)?;
        let is_byte = tokenizer.is_byte(token_id);
        println!("  Token {i}: ID={token_id}, piece={piece:?}, is_byte={is_byte}");
    }

    // Test vocabulary access
    println!("\n📚 Vocabulary Sample (first 10 tokens):");
    let vocab = tokenizer.vocab();
    for i in 0..10.min(vocab.len()) {
        println!("  {}: {:?}", i, vocab[i]);
    }

    // Test byte token range
    println!("\n🔢 Byte Token Analysis:");
    let mut byte_count = 0;
    for token_id in 0..256 {
        if tokenizer.is_byte(token_id) {
            byte_count += 1;
        }
    }
    println!("  Byte tokens in range 0-255: {byte_count}");

    // Test audio support
    println!("\n🎵 Audio Support:");
    if tokenizer.has_audio_support() {
        println!("  ✅ Audio encoding is available");
        if let Some(config) = tokenizer.audio_config() {
            println!(
                "  📊 Audio config: sampling_rate={}, frame_rate={}",
                config.sampling_rate, config.frame_rate
            );
        }
    } else {
        println!("  ❌ Audio encoding not available");
    }

    println!("\n🎉 All tests completed!");
    Ok(())
}
