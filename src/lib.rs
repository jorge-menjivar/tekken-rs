//! # Tekken - Rust Implementation of Mistral's Multimodal Tokenizer
//!
//! `tekken` is a Rust implementation of Mistral's Tekken tokenizer with full support
//! for text, image and audio tokenization. It provides high-performance, memory-safe
//! tokenization that is fully compatible with the Python implementation.
//!
//! ## Features
//!
//! - **Text Tokenization**: Full BPE (Byte Pair Encoding) support with special tokens
//! - **Image Processing**: Resize and normalize images into `[IMG]` token grids (`image` feature)
//! - **Audio Processing**: Convert audio waveforms to token sequences using mel-scale spectrograms (`audio` feature)
//! - **Multimodal Support**: Mix text, image and audio tokens in a single sequence
//! - **Version Compatibility**: Support for multiple tokenizer versions (V3, V7, V11, V13, V15)
//! - **Special Tokens**: Comprehensive handling of control, instruction, tool, and media tokens
//! - **Model Settings**: Parsing and validation of model settings constraints (V15+)
//!
//! ## Feature Flags
//!
//! Both multimodal features are enabled by default and can be turned off to
//! drop their dependencies:
//!
//! - `image` - image decoding and preprocessing, via the `image` crate
//! - `audio` - audio decoding and preprocessing, via `hound`
//!
//! Tokenizer files are parsed the same either way: [`ImageConfig`] and
//! [`AudioConfig`] are always available, as are the token layouts
//! [`ImageEncoder`] computes from image dimensions. Only decoding and
//! preprocessing actual media need the features.
//!
//! ## Quick Start
//!
//! ### Basic Text Tokenization
//!
//! ```rust,no_run
//! use tekken::{Tekkenizer, SpecialTokenPolicy};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load tokenizer from configuration file
//! let tokenizer = Tekkenizer::from_file("tekken.json")?;
//!
//! // Encode text with BOS/EOS tokens
//! let text = "Hello, world!";
//! let tokens = tokenizer.encode(text, true, true)?;
//! println!("Tokens: {:?}", tokens);
//!
//! // Decode back to text
//! let decoded = tokenizer.decode(&tokens, SpecialTokenPolicy::Keep)?;
//! println!("Decoded: {}", decoded);
//! # Ok(())
//! # }
//! ```
//!
//! ### Image Tokenization
//!
//! ```rust,no_run
//! # #[cfg(feature = "image")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use tekken::{Image, ImageConfig, ImageEncoder, SpecialImageIds};
//!
//! // Load an image
//! let image = Image::from_file("picture.png")?;
//!
//! // Configure image processing
//! let config = ImageConfig::new(14, 1540, 2)?;
//!
//! // Create encoder and process the image
//! let encoder = ImageEncoder::new(
//!     config,
//!     SpecialImageIds { img: 10, img_break: 12, img_end: 13 },
//! );
//! let encoding = encoder.encode(&image)?;
//!
//! println!("Image encoded to {} tokens", encoding.tokens.len());
//! println!("Preprocessed shape: {:?}", encoding.image.dim());
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "image"))]
//! # fn main() {}
//! ```
//!
//! ### Audio Tokenization
//!
//! ```rust,no_run
//! # #[cfg(feature = "audio")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use tekken::{Audio, AudioConfig, AudioSpectrogramConfig, AudioEncoder};
//!
//! // Load audio file
//! let audio = Audio::from_file("audio.wav")?;
//!
//! // Configure audio processing
//! let spectrogram_config = AudioSpectrogramConfig::new(80, 160, 400)?;
//! let audio_config = AudioConfig::new(16000, 12.5, spectrogram_config, None)?;
//!
//! // Create encoder and process audio
//! let encoder = AudioEncoder::new(audio_config, 1000, 1001); // audio_token_id, begin_audio_token_id
//! let encoding = encoder.encode(audio)?;
//!
//! println!("Audio encoded to {} tokens", encoding.tokens.len());
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "audio"))]
//! # fn main() {}
//! ```
//!
//! ### Multimodal Tokenization
//!
//! ```rust,no_run
//! # #[cfg(all(feature = "audio", feature = "image"))]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use tekken::{Tekkenizer, Audio, Image, SpecialTokenPolicy};
//!
//! let tokenizer = Tekkenizer::from_file("tekken.json")?;
//!
//! // Text tokens
//! let text_tokens = tokenizer.encode("Please transcribe this audio:", true, false)?;
//!
//! // Image tokens (if tokenizer has image support)
//! if tokenizer.has_image_support() {
//!     let image = Image::from_file("picture.png")?;
//!     let image_encoding = tokenizer.encode_image(&image)?;
//!     println!("Image tokens: {}", image_encoding.tokens.len());
//! }
//!
//! // Audio tokens (if tokenizer has audio support)
//! if tokenizer.has_audio_support() {
//!     let audio = Audio::from_file("speech.wav")?;
//!     let audio_encoding = tokenizer.encode_audio(audio)?;
//!     
//!     // Combine text and audio tokens
//!     let mut combined_tokens = text_tokens;
//!     combined_tokens.extend(audio_encoding.tokens);
//!     
//!     println!("Combined sequence: {} tokens", combined_tokens.len());
//! }
//! # Ok(())
//! # }
//! # #[cfg(not(all(feature = "audio", feature = "image")))]
//! # fn main() {}
//! ```
//!
//! ## Architecture
//!
//! The library is organized into several modules:
//!
//! - [`tekkenizer`]: Main tokenizer implementation and text processing
//! - [`image`]: Image loading, preprocessing, and image tokenization
//! - [`audio`]: Audio processing, mel-scale spectrograms, and audio tokenization
//! - [`special_tokens`]: Special token definitions and handling policies
//! - [`config`]: Configuration structures and version management
//! - [`model_settings`]: Model settings constraints for v15+ tokenizers
//! - [`errors`]: Comprehensive error handling
//!
//! ## Compatibility
//!
//! This Rust implementation is designed to be fully compatible with Mistral's Python
//! tokenizer implementation:
//!
//! - Identical tokenization results for text
//! - Same image resizing, normalization and token layout
//! - Same audio processing pipeline and token generation
//! - Compatible special token handling
//! - Matching mel filter bank computations
//!
//! ## Performance
//!
//! The Rust implementation provides significant performance improvements over Python:
//!
//! - Memory-safe processing with zero-copy operations where possible
//! - Efficient audio processing with optimized mel-scale computations
//! - Fast BPE tokenization using proven algorithms
//! - Minimal allocations and efficient data structures

pub mod audio;
pub mod config;
pub mod errors;
pub mod image;
pub mod model_settings;
pub mod special_tokens;
pub mod tekkenizer;

// Re-export commonly used types for convenience
#[cfg(feature = "audio")]
pub use audio::{Audio, AudioEncoding};
pub use audio::{AudioConfig, AudioEncoder, AudioSpectrogramConfig};
pub use config::{TekkenConfig, TokenInfo};
pub use errors::{Result, TokenizerError};
#[cfg(feature = "image")]
pub use image::{Image, ImageEncoding};
pub use image::{ImageConfig, ImageEncoder, SpecialImageIds};
pub use model_settings::{ModelSettings, ModelSettingsBuilder, ReasoningEffort};
pub use special_tokens::SpecialTokenInfo;
pub use special_tokens::{SpecialTokenPolicy, SpecialTokens};
pub use tekkenizer::Tekkenizer;
