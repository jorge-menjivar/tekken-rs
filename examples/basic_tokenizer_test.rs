use tekken::tekkenizer::Tekkenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing tekken-rs tokenizer...");

    // Load tokenizer
    let tokenizer = Tekkenizer::from_file("tekken.json")?;
    println!("Tokenizer loaded successfully!");
    println!("Vocab size: {}", tokenizer.vocab_size());
    println!("Version: {:?}", tokenizer.version());

    // Test encoding
    let text = "Hello, world! This is a test.";
    let tokens = tokenizer.encode(text, true, true)?;
    println!("Encoded '{}' -> {:?}", text, tokens);

    // Test decoding
    let decoded = tokenizer.decode(&tokens, tekken::special_tokens::SpecialTokenPolicy::Keep)?;
    println!("Decoded {:?} -> '{}'", tokens, decoded);

    // Test audio support
    if tokenizer.has_audio_support() {
        println!("✅ Audio support is available");
    } else {
        println!("❌ Audio support not available");
    }

    Ok(())
}
