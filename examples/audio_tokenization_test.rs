use serde_json::json;
use std::fs::File;
use std::io::Write;
use tekken::audio::Audio;
use tekken::special_tokens::SpecialTokenPolicy;
use tekken::tekkenizer::Tekkenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Audio Tokenization Test with Tekkenizer ===\n");

    // Load tokenizer with audio support
    let tokenizer = Tekkenizer::from_file("tests/assets/tekken.json")?;
    println!("✅ Tokenizer loaded successfully!");
    println!("📊 Vocab size: {}", tokenizer.vocab_size());
    println!("📝 Version: {:?}", tokenizer.version());

    // Check audio support
    if !tokenizer.has_audio_support() {
        println!("❌ Tokenizer doesn't have audio support");
        return Ok(());
    }

    println!("✅ Audio support available!");
    if let Some(config) = tokenizer.audio_config() {
        println!(
            "📊 Audio config: sampling_rate={}, frame_rate={}",
            config.sampling_rate, config.frame_rate
        );
    }

    // Load and tokenize audio
    println!("\n🎵 Loading test_audio.wav...");
    let audio = Audio::from_file("test_audio.wav")?;
    println!(
        "📊 Audio loaded: {} samples, {}Hz, {:.3}s duration",
        audio.audio_array.len(),
        audio.sampling_rate,
        audio.duration()
    );

    // Encode audio to tokens
    println!("\n🔢 Encoding audio to tokens...");
    let audio_encoding = tokenizer.encode_audio(audio)?;
    println!("✅ Audio encoded to {} tokens", audio_encoding.tokens.len());
    println!(
        "🎯 First 20 tokens: {:?}",
        &audio_encoding.tokens[..std::cmp::min(20, audio_encoding.tokens.len())]
    );

    // Test mixed text + audio tokenization
    println!("\n📝 Testing mixed text and audio...");
    let text = "Here is some audio: ";
    let text_tokens = tokenizer.encode(text, true, false)?; // BOS but no EOS

    // Combine text and audio tokens
    let mut combined_tokens = text_tokens.clone();
    combined_tokens.extend_from_slice(&audio_encoding.tokens);

    // Add EOS token
    let eos_id = tokenizer.eos_id()?;
    combined_tokens.push(eos_id);

    println!(
        "🔗 Combined sequence: {} tokens total",
        combined_tokens.len()
    );
    println!("   Text: {} tokens", text_tokens.len());
    println!("   Audio: {} tokens", audio_encoding.tokens.len());
    println!("   Total with EOS: {} tokens", combined_tokens.len());

    // Try to decode the combined sequence
    println!("\n🔍 Decoding combined sequence...");
    let decoded = tokenizer.decode(&combined_tokens, SpecialTokenPolicy::Keep)?;
    println!("📜 Decoded: {decoded:?}");

    // Save results to JSON
    let results = json!({
        "tokenizer_info": {
            "vocab_size": tokenizer.vocab_size(),
            "version": format!("{:?}", tokenizer.version()),
            "has_audio_support": tokenizer.has_audio_support()
        },
        "audio_info": {
            "duration_seconds": audio_encoding.audio.duration(),
            "sampling_rate": audio_encoding.audio.sampling_rate,
            "sample_count": audio_encoding.audio.audio_array.len()
        },
        "tokenization": {
            "text": text,
            "text_tokens": text_tokens,
            "audio_tokens": audio_encoding.tokens,
            "combined_tokens": combined_tokens,
            "token_counts": {
                "text": text_tokens.len(),
                "audio": audio_encoding.tokens.len(),
                "total": combined_tokens.len()
            }
        },
        "decoded_combined": decoded
    });

    // Save to file
    let mut file = File::create("audio_tokenization_results.json")?;
    file.write_all(serde_json::to_string_pretty(&results)?.as_bytes())?;

    println!("\n💾 Results saved to audio_tokenization_results.json");
    println!("🎉 Audio tokenization test completed!");

    Ok(())
}
