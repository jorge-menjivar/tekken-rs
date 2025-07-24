# tekken-rs

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

A Rust implementation of the Mistral Tekken tokenizer with audio support. This library provides fast and efficient tokenization capabilities for text and audio data, fully compatible with Mistral AI's tokenizer.

## Features

- **Text Tokenization**: Full compatibility with Mistral's Tekken tokenizer
- **Audio Support**: Encode and decode audio data with mel-scale spectrogram processing
- **Multiple Versions**: Support for various tokenizer versions (V7, etc.)
- **Special Tokens**: Complete handling of special tokens (BOS, EOS, audio tokens, etc.)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
tekken = "0.1.0"
```

Or use the Git repository directly:

```toml
[dependencies]
tekken = { git = "https://github.com/jorge-menjivar/tekken-rs" }
```

## Quick Start

### Basic Text Tokenization

```rust
use tekken::tekkenizer::Tekkenizer;
use tekken::special_tokens::SpecialTokenPolicy;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load tokenizer
    let tokenizer = Tekkenizer::from_file("tekken.json")?;

    // Encode text
    let text = "Hello, world!";
    let tokens = tokenizer.encode(text, true, true)?; // add_bos=true, add_eos=true

    // Decode tokens
    let decoded = tokenizer.decode(&tokens, SpecialTokenPolicy::Keep)?;
    println!("Original: {}", text);
    println!("Tokens: {:?}", tokens);
    println!("Decoded: {}", decoded);

    Ok(())
}
```

### Audio Processing

```rust
use tekken::audio::{Audio, AudioConfig, AudioSpectrogramConfig, AudioEncoder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load audio
    let audio = Audio::from_file("audio.wav")?;

    // Create audio configuration
    let spectrogram_config = AudioSpectrogramConfig::new(80, 160, 400)?;
    let audio_config = AudioConfig::new(16000, 12.5, spectrogram_config, None)?;

    // Encode audio to tokens
    let encoder = AudioEncoder::new(audio_config, 1000, 1001); // audio_token_id, begin_audio_token_id
    let encoding = encoder.encode(audio)?;

    println!("Audio encoded to {} tokens", encoding.tokens.len());

    Ok(())
}
```

## Examples

Run the examples to see the tokenizer in action:

```bash
# Basic tokenizer test
cargo run --example basic_tokenizer_test

# Audio processing test
cargo run --bin test_audio
```

## Testing

Run the test suite:

```bash
cargo test
```

## Architecture

The tokenizer consists of several key components:

- **`tokenizer.rs`**: Main tokenizer implementation
- **`audio.rs`**: Audio processing and encoding functionality
- **`special_tokens.rs`**: Special token definitions and handling
- **`config.rs`**: Configuration structures
- **`errors.rs`**: Error handling

## Audio Support

The audio implementation includes:

- WAV file loading and processing
- Mel-scale spectrogram computation
- Audio chunk encoding to tokens
- Compatible with Python implementation

### Audio Token Flow

1. **Load Audio**: Load WAV files or audio data
2. **Resample**: Convert to target sampling rate (16kHz)
3. **Pad**: Ensure minimum length for processing
4. **Tokenize**: Convert to token sequence with special audio markers

## Compatibility

This Rust implementation is designed to be fully compatible with the Python version:

- Same tokenization results
- Identical audio processing
- Compatible special token handling
- Same mel filter bank computations

## Requirements

- Rust 1.70 or higher
- For audio support: audio files in WAV format

## Project Structure

```
tekken-rs/
├── src/
│   ├── lib.rs          # Library entry point
│   ├── tokenizer.rs    # Main tokenizer implementation
│   ├── audio.rs        # Audio processing functionality
│   ├── special_tokens.rs # Special token definitions
│   ├── config.rs       # Configuration structures
│   └── errors.rs       # Error types
├── examples/           # Example usage
├── tests/             # Integration tests
└── benches/           # Performance benchmarks
```

## Performance

The Rust implementation provides significant performance improvements over the Python version:

- Fast tokenization using efficient data structures
- Zero-copy string handling where possible
- Optimized audio processing with SIMD operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to:
- Update tests as appropriate
- Follow Rust coding conventions
- Run `cargo fmt` and `cargo clippy` before submitting

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This is an original Rust implementation designed to be compatible with Mistral AI's Tekken tokenizer format.

The project uses tokenizer model files (tekken_240718.json, tekken_240911.json) from Mistral AI to ensure compatibility.

See [NOTICE](NOTICE) file for detailed attribution.
