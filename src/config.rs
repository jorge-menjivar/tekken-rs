use crate::audio::AudioConfig;
use crate::special_tokens::SpecialTokenInfo;
use serde::{Deserialize, Serialize};

/// Information about a vocabulary token.
///
/// This struct contains metadata about a single token in the vocabulary,
/// including its rank (position), byte representation, and optional string form.
///
/// # Fields
///
/// * `rank` - Position of the token in the vocabulary (used as token ID)
/// * `token_bytes` - Base64-encoded byte representation of the token
/// * `token_str` - Optional human-readable string representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    /// The position of this token in the vocabulary (used as token ID).
    pub rank: usize,
    /// Base64-encoded byte representation of the token.
    pub token_bytes: String,
    /// Optional human-readable string representation of the token.
    pub token_str: Option<String>,
}

/// Configuration parameters for a Tekken tokenizer.
///
/// This struct contains the core configuration needed to initialize a tokenizer,
/// including the regex pattern for tokenization, vocabulary sizes, and version information.
///
/// # Fields
///
/// * `pattern` - Regex pattern used for tokenization
/// * `num_vocab_tokens` - Number of regular vocabulary tokens
/// * `default_vocab_size` - Default total vocabulary size including special tokens
/// * `default_num_special_tokens` - Default number of special tokens
/// * `version` - Tokenizer version string (e.g., "v7")
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TekkenConfig {
    /// Regex pattern used for tokenization.
    pub pattern: String,
    /// Number of regular vocabulary tokens (excluding special tokens).
    pub num_vocab_tokens: usize,
    /// Default total vocabulary size including special tokens.
    pub default_vocab_size: usize,
    /// Default number of special tokens.
    pub default_num_special_tokens: usize,
    /// Tokenizer version string (e.g., "v7", "v11", "v13").
    pub version: String,
}

/// Configuration for image processing (placeholder).
///
/// This struct is reserved for future image processing capabilities.
/// Currently minimal as audio processing is the primary multimodal focus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageConfig {
    // Image config fields would go here
    // For now, we'll keep it minimal as audio is the focus
}

/// Complete model data loaded from a tokenizer configuration file.
///
/// This struct represents the entire configuration and data needed to initialize
/// a Tekken tokenizer, typically loaded from a JSON file like `tekken.json`.
///
/// # Fields
///
/// * `vocab` - All vocabulary tokens with their metadata
/// * `special_tokens` - Optional special token definitions
/// * `config` - Core tokenizer configuration
/// * `audio` - Optional audio processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelData {
    /// All vocabulary tokens with their metadata.
    pub vocab: Vec<TokenInfo>,
    /// Optional special token definitions (uses defaults if None).
    pub special_tokens: Option<Vec<SpecialTokenInfo>>,
    /// Core tokenizer configuration parameters.
    pub config: TekkenConfig,
    /// Optional audio processing configuration for multimodal support.
    pub audio: Option<AudioConfig>,
}

/// Enumeration of supported tokenizer versions.
///
/// Different versions may have different vocabulary sizes, special tokens,
/// and processing capabilities. This enum provides a type-safe way to
/// handle version-specific behavior.
///
/// # Supported Versions
///
/// * `V3` - Early version with basic functionality
/// * `V7` - Version with enhanced special tokens and audio support
/// * `V11` - Updated version with additional features
/// * `V13` - Latest version with full multimodal capabilities
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerVersion {
    V3,
    V7,
    V11,
    V13,
}

impl TokenizerVersion {
    /// Parses a version string into a `TokenizerVersion`.
    ///
    /// # Arguments
    ///
    /// * `s` - Version string (e.g., "v7", "v11")
    ///
    /// # Returns
    ///
    /// The corresponding `TokenizerVersion` if recognized, None otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tekken::config::TokenizerVersion;
    ///
    /// assert_eq!(TokenizerVersion::from_string("v7"), Some(TokenizerVersion::V7));
    /// assert_eq!(TokenizerVersion::from_string("invalid"), None);
    /// ```
    #[must_use]
    pub fn from_string(s: &str) -> Option<Self> {
        match s {
            "v3" => Some(Self::V3),
            "v7" => Some(Self::V7),
            "v11" => Some(Self::V11),
            "v13" => Some(Self::V13),
            _ => None,
        }
    }

    /// Returns the string representation of the version.
    ///
    /// # Returns
    ///
    /// The version string (e.g., "v7", "v11").
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tekken::config::TokenizerVersion;
    ///
    /// assert_eq!(TokenizerVersion::V7.as_str(), "v7");
    /// assert_eq!(TokenizerVersion::V13.as_str(), "v13");
    /// ```
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::V3 => "v3",
            Self::V7 => "v7",
            Self::V11 => "v11",
            Self::V13 => "v13",
        }
    }
}
