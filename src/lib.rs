//! # Tekken - Rust Implementation of Mistral's Multimodal Tokenizer
//!
//! `tekken` is a Rust implementation of Mistral's Tekken tokenizer with full support
//! for both text and audio tokenization. It provides high-performance, memory-safe
//! tokenization that is fully compatible with the Python implementation.
//!
//! ## Features
//!
//! - **Text Tokenization**: Full BPE (Byte Pair Encoding) support with special tokens
//! - **Audio Processing**: Convert audio waveforms to token sequences using mel-scale spectrograms
//! - **Multimodal Support**: Mix text and audio tokens in a single sequence
//! - **Version Compatibility**: Support for multiple tokenizer versions (V3, V7, V11, V13)
//! - **Special Tokens**: Comprehensive handling of control, instruction, tool, and media tokens
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
//! ### Audio Tokenization
//!
//! ```rust,no_run
//! use tekken::{Audio, AudioConfig, AudioSpectrogramConfig, AudioEncoder};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
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
//! ```
//!
//! ### Multimodal Tokenization
//!
//! ```rust,no_run
//! use tekken::{Tekkenizer, Audio, SpecialTokenPolicy};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let tokenizer = Tekkenizer::from_file("tekken.json")?;
//!
//! // Text tokens
//! let text_tokens = tokenizer.encode("Please transcribe this audio:", true, false)?;
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
//! ```
//!
//! ## Architecture
//!
//! The library is organized into several modules:
//!
//! - [`tekkenizer`]: Main tokenizer implementation and text processing
//! - [`audio`]: Audio processing, mel-scale spectrograms, and audio tokenization  
//! - [`special_tokens`]: Special token definitions and handling policies
//! - [`config`]: Configuration structures and version management
//! - [`errors`]: Comprehensive error handling
//!
//! ## Compatibility
//!
//! This Rust implementation is designed to be fully compatible with Mistral's Python
//! tokenizer implementation:
//!
//! - Identical tokenization results for text
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
pub mod special_tokens;
pub mod tekkenizer;

// Re-export commonly used types for convenience
pub use audio::{Audio, AudioConfig, AudioEncoder, AudioSpectrogramConfig};
pub use config::{TekkenConfig, TokenInfo};
pub use errors::{Result, TokenizerError};
pub use special_tokens::SpecialTokenInfo;
pub use special_tokens::{SpecialTokenPolicy, SpecialTokens};
pub use tekkenizer::Tekkenizer;
