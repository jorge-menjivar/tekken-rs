use thiserror::Error;

/// Type alias for Results with `TokenizerError`.
///
/// This provides a convenient shorthand for Result types throughout the library.
pub type Result<T> = std::result::Result<T, TokenizerError>;

/// Comprehensive error type for tokenizer operations.
///
/// This enum covers all possible error conditions that can occur during
/// tokenizer initialization, text/audio processing, and other operations.
/// It uses the `thiserror` crate to provide detailed error messages and
/// automatic conversion from underlying error types.
///
/// # Error Categories
///
/// * **I/O Errors**: File reading/writing operations
/// * **Parsing Errors**: JSON deserialization and Base64 decoding
/// * **Processing Errors**: Tokenization and audio processing failures
/// * **Configuration Errors**: Invalid parameters or missing tokens
/// * **Policy Errors**: Special token handling violations
#[derive(Error, Debug)]
pub enum TokenizerError {
    /// I/O operation failed (file reading, writing, etc.).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing or serialization failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Base64 decoding failed.
    #[error("Base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),

    /// Error in the underlying tokenization engine.
    #[error("Tokenizers error: {0}")]
    Tokenizers(String),

    /// Audio processing operation failed.
    #[error("Audio error: {0}")]
    Audio(String),

    /// Configuration parameters are invalid or inconsistent.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Required token (usually special token) was not found in vocabulary.
    #[error("Token not found: {0}")]
    TokenNotFound(String),

    /// Operation violated the specified special token policy.
    #[error("Special token policy violation: {0}")]
    SpecialTokenPolicy(String),

    /// File format or data format is not supported.
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}
