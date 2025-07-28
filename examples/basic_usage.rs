use base64::Engine;
use ndarray::Array1;
use tekken::{
    Audio, AudioConfig, AudioSpectrogramConfig, Result, SpecialTokenPolicy, Tekkenizer, config,
    special_tokens,
};

fn main() -> Result<()> {
    env_logger::init();

    println!("=== Mistral Tekken Tokenizer Demo ===\n");

    // Try to load from the actual Mistral tokenizer file if available
    let tokenizer_path = "tests/assets/tekken.json";

    let tokenizer = match Tekkenizer::from_file(tokenizer_path) {
        Ok(t) => {
            println!("✅ Loaded tokenizer from {tokenizer_path}");
            t
        }
        Err(e) => {
            println!("⚠️  Could not load from {tokenizer_path}: {e}");
            println!("Creating a test tokenizer instead...\n");
            create_test_tokenizer()?
        }
    };

    println!("📊 Tokenizer info:");
    println!("   Vocab size: {}", tokenizer.vocab_size());
    println!("   Special tokens: {}", tokenizer.num_special_tokens());
    println!("   Version: {:?}", tokenizer.version());
    println!(
        "   Audio support: {}\n",
        if tokenizer.has_audio_support() {
            "✅"
        } else {
            "❌"
        }
    );

    // Test basic encoding/decoding
    test_basic_tokenization(&tokenizer)?;

    // Test special tokens
    test_special_tokens(&tokenizer)?;

    // Test audio encoding if supported
    if tokenizer.has_audio_support() {
        test_audio_encoding(&tokenizer)?;
    }

    println!("🎉 All tests completed successfully!");
    Ok(())
}

fn create_test_tokenizer() -> Result<Tekkenizer> {
    // Create minimal test vocabulary
    let mut vocab = Vec::new();

    // Byte tokens (0-255)
    for i in 0..256 {
        let bytes = vec![i as u8];
        let token_bytes = base64::engine::general_purpose::STANDARD.encode(&bytes);
        vocab.push(config::TokenInfo {
            rank: i,
            token_bytes,
            token_str: if i < 128 {
                Some(char::from(i as u8).to_string())
            } else {
                None
            },
        });
    }

    // Some common tokens
    let common_tokens: &[&[u8]] = &[b"hello", b"world", b"test", b"and", b"is"];
    for (i, token) in common_tokens.iter().enumerate() {
        vocab.push(config::TokenInfo {
            rank: 256 + i,
            token_bytes: base64::engine::general_purpose::STANDARD.encode(token),
            token_str: Some(String::from_utf8_lossy(token).to_string()),
        });
    }

    // Create special tokens with audio support
    let mut special_tokens = Vec::new();
    let special_token_names = [
        "<unk>",
        "<s>",
        "</s>",
        "[INST]",
        "[/INST]",
        "[AVAILABLE_TOOLS]",
        "[/AVAILABLE_TOOLS]",
        "[TOOL_RESULTS]",
        "[/TOOL_RESULTS]",
        "[TOOL_CALLS]",
        "[IMG]",
        "<pad>",
        "[IMG_BREAK]",
        "[IMG_END]",
        "[PREFIX]",
        "[MIDDLE]",
        "[SUFFIX]",
        "[SYSTEM_PROMPT]",
        "[/SYSTEM_PROMPT]",
        "[TOOL_CONTENT]",
    ];

    for (i, &name) in special_token_names.iter().enumerate() {
        special_tokens.push(special_tokens::SpecialTokenInfo {
            rank: i,
            token_str: name.to_string(),
            is_control: true,
        });
    }

    // Add audio tokens
    special_tokens.push(special_tokens::SpecialTokenInfo {
        rank: 24,
        token_str: "[AUDIO]".to_string(),
        is_control: true,
    });
    special_tokens.push(special_tokens::SpecialTokenInfo {
        rank: 25,
        token_str: "[BEGIN_AUDIO]".to_string(),
        is_control: true,
    });

    // Create audio config
    let audio_config = AudioConfig::new(
        24000, // 24kHz sampling rate
        12.5,  // 12.5 frames per second
        AudioSpectrogramConfig::new(128, 160, 400)?,
        Some(1.0), // 1 second chunks
    )?;

    Tekkenizer::new(
        vocab,
        &special_tokens,
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        300, // vocab_size
        100, // num_special_tokens
        config::TokenizerVersion::V7,
        Some(audio_config),
    )
}

fn test_basic_tokenization(tokenizer: &Tekkenizer) -> Result<()> {
    println!("🔤 Testing basic tokenization...");

    let test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "This is a test of the Mistral Tekken tokenizer.",
        "🚀 Émojis and ünícode characters work too! 🎉",
    ];

    for text in &test_texts {
        println!("   Input: \"{text}\"");

        // Encode without special tokens
        let tokens = tokenizer.encode(text, false, false)?;
        println!("   Tokens: {:?}", &tokens[..tokens.len().min(10)]);

        // Decode back
        let decoded = tokenizer.decode(&tokens, SpecialTokenPolicy::Ignore)?;
        println!("   Decoded: \"{decoded}\"");

        // Check if round-trip works (it might not be exact due to BPE)
        if text.trim() == decoded.trim() {
            println!("   ✅ Perfect round-trip!");
        } else {
            println!("   ⚠️  Round-trip differs (this is normal for BPE)");
        }
        println!();
    }

    Ok(())
}

fn test_special_tokens(tokenizer: &Tekkenizer) -> Result<()> {
    println!("🏷️  Testing special tokens...");

    let text = "Test message";

    // Test with BOS/EOS
    let tokens_with_special = tokenizer.encode(text, true, true)?;
    println!("   With BOS/EOS: {tokens_with_special:?}");

    // Test different decode policies
    let decoded_keep = tokenizer.decode(&tokens_with_special, SpecialTokenPolicy::Keep)?;
    let decoded_ignore = tokenizer.decode(&tokens_with_special, SpecialTokenPolicy::Ignore)?;

    println!("   Decoded (KEEP): \"{decoded_keep}\"");
    println!("   Decoded (IGNORE): \"{decoded_ignore}\"");

    // Test individual special tokens
    if let Ok(bos_id) = tokenizer.bos_id() {
        println!("   BOS token ID: {bos_id}");
    }
    if let Ok(eos_id) = tokenizer.eos_id() {
        println!("   EOS token ID: {eos_id}");
    }

    println!();
    Ok(())
}

fn test_audio_encoding(tokenizer: &Tekkenizer) -> Result<()> {
    println!("🎵 Testing audio encoding...");

    // Generate a simple sine wave (440 Hz A note)
    let duration = 1.0; // 1 second
    let sampling_rate = 24000;
    let frequency = 440.0;

    let signal_length = (duration * sampling_rate as f64) as usize;
    let mut audio_data = Vec::with_capacity(signal_length);

    for i in 0..signal_length {
        let t = i as f64 / sampling_rate as f64;
        let sample = (2.0 * std::f64::consts::PI * frequency * t).sin() as f32;
        audio_data.push(sample * 0.5); // Reduce amplitude
    }

    let audio = Audio::new(
        Array1::from_vec(audio_data),
        sampling_rate,
        "wav".to_string(),
    );

    println!("   Audio duration: {:.2}s", audio.duration());
    println!("   Audio samples: {}", audio.audio_array.len());

    // Encode the audio
    let encoding = tokenizer.encode_audio(audio)?;
    println!("   Audio tokens: {}", encoding.tokens.len());
    println!(
        "   First few tokens: {:?}",
        &encoding.tokens[..encoding.tokens.len().min(5)]
    );

    // Verify token structure
    if !encoding.tokens.is_empty() {
        let begin_audio_id = tokenizer.get_control_token("[BEGIN_AUDIO]")?;
        let audio_token_id = tokenizer.get_control_token("[AUDIO]")?;

        if encoding.tokens[0] == begin_audio_id {
            println!("   ✅ First token is BEGIN_AUDIO");
        }

        let audio_token_count = encoding.tokens[1..]
            .iter()
            .filter(|&&t| t == audio_token_id)
            .count();
        println!("   Audio content tokens: {audio_token_count}");
    }

    println!();
    Ok(())
}
