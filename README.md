# tekken

[![crates.io](https://img.shields.io/crates/v/tekken.svg)](https://crates.io/crates/tekken)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fjorge-menjivar.github.io%2Ftekken-rs%2Fcoverage.json)](https://jorge-menjivar.github.io/tekken-rs/)

A Rust implementation of the Mistral Tekken tokenizer with image and audio support. This library provides fast and efficient tokenization capabilities for text, image and audio data, fully compatible with Mistral AI's tokenizer.

## Features

- **Text Tokenization**: Full compatibility with Mistral's Tekken tokenizer
- **Image Support**: Resize and normalize images into `[IMG]` token grids (`image` feature)
- **Audio Support**: Encode and decode audio data with mel-scale spectrogram processing (`audio` feature)
- **Multiple Versions**: Support for tokenizer versions V3, V7, V11, V13, and V15
- **Special Tokens**: Complete handling of special tokens (BOS, EOS, audio tokens, etc.)
- **Model Settings**: Parsing and validation of `model_settings_builder` constraints (V15+)

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

### Feature Flags

The multimodal features are enabled by default. Turn them off to drop their
dependencies:

| Feature | Default | Enables | Pulls in |
| --- | --- | --- | --- |
| `image` | yes | Image decoding and preprocessing | `image`, `ndarray` |
| `audio` | yes | Audio decoding and preprocessing | `hound`, `rubato`, `rustfft`, `ndarray` |

```toml
[dependencies]
# Text only
tekken = { version = "0.1.0", default-features = false }

# Text and images
tekken = { version = "0.1.0", default-features = false, features = ["image"] }
```

Tokenizer files parse the same either way: `ImageConfig` and `AudioConfig` are
always available, and `ImageEncoder` still lays out image tokens from image
dimensions. Only decoding and preprocessing actual media need the features.

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

### Image Processing

```rust
use tekken::image::Image;
use tekken::tekkenizer::Tekkenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A tokenizer whose tekken.json carries an image section
    let tokenizer = Tekkenizer::from_file("tekken.json")?;

    let image = Image::from_file("picture.png")?;
    let encoding = tokenizer.encode_image(&image)?;

    // Rows of [IMG] tokens delimited by [IMG_BREAK] / [IMG_END]
    println!("Tokens: {}", encoding.tokens.len());
    // Normalized pixels, shape (3, height, width)
    println!("Pixels: {:?}", encoding.image.dim());

    Ok(())
}
```

To lay out image tokens without decoding pixels, use the encoder directly:

```rust
use tekken::image::{ImageConfig, ImageEncoder, SpecialImageIds};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let encoder = ImageEncoder::new(
        ImageConfig::new(14, 1540, 2)?,
        SpecialImageIds { img: 10, img_break: 12, img_end: 13 },
    );

    let (width_tokens, height_tokens) = encoder.image_to_num_tokens(1024, 768)?;
    println!("{width_tokens}x{height_tokens} token grid");

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
- **`image.rs`**: Image loading, preprocessing and encoding functionality
- **`audio.rs`**: Audio processing and encoding functionality
- **`special_tokens.rs`**: Special token definitions and handling
- **`config.rs`**: Configuration structures
- **`errors.rs`**: Error handling

## Image Support

The image implementation includes:

- PNG, JPEG, GIF, WebP, BMP and TIFF decoding, with transparency composited onto white
- Aspect-ratio-preserving resizing to a whole number of vision patches
- Normalization with the CLIP dataset statistics into a `(3, height, width)` array
- `[IMG]` / `[IMG_BREAK]` / `[IMG_END]` token grids

### Image Token Flow

1. **Load Image**: Decode an image from a file, bytes, a base64 string or raw RGB
2. **Fit**: Scale down so neither side exceeds `max_image_size`, keeping the aspect ratio
3. **Resize**: Resample to a whole number of tokens with bicubic interpolation
4. **Normalize**: Scale to `[0, 1]` and standardize per channel
5. **Tokenize**: Emit one `[IMG]` per token-sized block, ending each row with a break

The resize is a port of OpenCV's `INTER_CUBIC`, which is what `mistral-common` uses.

Verified against `mistral-common` 1.11.7's `ImageEncoder`: token sequences are
identical, and pixel values agree to within float32 rounding (OpenCV's own output
varies in the last bits between its scalar, SIMD and IPP code paths). PNG, WebP,
BMP and TIFF decode byte-for-byte identically to Pillow.

The exception is JPEG: decoding uses the `image` crate, whose baseline JPEG
decoder differs from Pillow's libjpeg-turbo in its IDCT and chroma upsampling, so
decoded samples can differ by a few 8-bit levels. Token layouts are unaffected,
since they depend only on the image dimensions. To pin a specific decoder, decode
yourself and pass the pixels to `Image::new`.

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
- Identical image token layouts and preprocessing
- Identical audio processing
- Compatible special token handling
- Same mel filter bank computations

## Requirements

- Rust 1.70 or higher
- For image support: PNG, JPEG, GIF, WebP, BMP or TIFF files
- For audio support: audio files in WAV format

## Project Structure

```
tekken-rs/
├── src/
│   ├── lib.rs          # Library entry point
│   ├── tokenizer.rs    # Main tokenizer implementation
│   ├── image.rs        # Image processing functionality
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

See [NOTICE](NOTICE) file for detailed attribution.
