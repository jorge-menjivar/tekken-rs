use serde_json::json;

use tekken::audio::{Audio, AudioConfig, AudioEncoder, AudioSpectrogramConfig, mel_filter_bank};

#[test]
fn test_rust_audio() {
    println!("\n=== Testing Rust Audio Implementation ===");

    // Load audio file
    let audio = Audio::from_file("tests/assets/jfk.wav").unwrap();
    println!(
        "Rust Audio loaded: shape={}, sr={}, format={}",
        audio.audio_array.len(),
        audio.sampling_rate,
        audio.format
    );
    println!("Duration: {:.3}s", audio.duration());

    // Create audio config
    let spectrogram_config = AudioSpectrogramConfig::new(80, 160, 400).unwrap();
    let audio_config = AudioConfig::new(16000, 12.5, spectrogram_config, None).unwrap();

    // Create encoder with dummy token IDs
    let encoder = AudioEncoder::new(audio_config, 1000, 1001);

    // Encode audio
    let encoding = encoder.encode(audio).unwrap();
    println!("Rust encoding: {} tokens", encoding.tokens.len());
    println!(
        "First few tokens: {:?}",
        &encoding.tokens[..std::cmp::min(1000, encoding.tokens.len())]
    );

    // Test mel filter bank
    let filter_bank = mel_filter_bank(201, 80, 0.0, 8000.0, 16000).unwrap();
    println!("Mel Filter Bank: {filter_bank}");
    println!("Rust mel filter bank shape: {:?}", filter_bank.dim());

    let filter_sum: f64 = filter_bank.iter().sum();

    let results = json!({
        "audio_shape": [encoding.audio.audio_array.len()],
        "sampling_rate": encoding.audio.sampling_rate,
        "duration": encoding.audio.duration(),
        "num_tokens": encoding.tokens.len(),
        "tokens": &encoding.tokens[..std::cmp::min(10, encoding.tokens.len())],
        "mel_filter_shape": [filter_bank.nrows(), filter_bank.ncols()],
        "mel_filter_sum": filter_sum
    });

    println!("Rust results: {results}");
}
