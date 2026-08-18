use crate::audio::AudioConfig;
use crate::image::ImageConfig;
use crate::model_settings::ModelSettingsBuilder;
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
/// * `image` - Optional image processing configuration
/// * `multimodal` - Deprecated spelling of `image`, only valid up to v11
/// * `audio` - Optional audio processing configuration
/// * `model_settings_builder` - Optional model settings constraints (v15+ only)
///
/// Deserialization tolerates a repeated key by keeping its last occurrence,
/// matching Python's `json.load`. Some released tokenizer files rely on this:
/// the v11 `tekken.json` of Mistral-Small-3.2 lists `multimodal` twice.
#[derive(Debug, Clone, Serialize)]
pub struct ModelData {
    /// All vocabulary tokens with their metadata.
    pub vocab: Vec<TokenInfo>,
    /// Optional special token definitions (uses defaults if None).
    pub special_tokens: Option<Vec<SpecialTokenInfo>>,
    /// Core tokenizer configuration parameters.
    pub config: TekkenConfig,
    /// Optional image processing configuration for multimodal support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<ImageConfig>,
    /// Deprecated spelling of [`ModelData::image`], only accepted up to v11.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multimodal: Option<ImageConfig>,
    /// Optional audio processing configuration for multimodal support.
    pub audio: Option<AudioConfig>,
    /// Optional model settings constraints (only valid for tokenizers v15+).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_settings_builder: Option<ModelSettingsBuilder>,
}

impl<'de> Deserialize<'de> for ModelData {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ModelDataVisitor;

        impl<'de> serde::de::Visitor<'de> for ModelDataVisitor {
            type Value = ModelData;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a tekken tokenizer document")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<ModelData, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                use serde::de::Error as _;

                let mut vocab = None;
                let mut special_tokens = None;
                let mut config = None;
                let mut image = None;
                let mut multimodal = None;
                let mut audio = None;
                let mut model_settings_builder = None;

                // Repeated keys overwrite earlier ones instead of being an
                // error, and unknown keys are skipped.
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "vocab" => vocab = Some(map.next_value()?),
                        "special_tokens" => special_tokens = map.next_value()?,
                        "config" => config = Some(map.next_value()?),
                        "image" => image = map.next_value()?,
                        "multimodal" => multimodal = map.next_value()?,
                        "audio" => audio = map.next_value()?,
                        "model_settings_builder" => model_settings_builder = map.next_value()?,
                        _ => {
                            map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }

                Ok(ModelData {
                    vocab: vocab.ok_or_else(|| A::Error::missing_field("vocab"))?,
                    special_tokens,
                    config: config.ok_or_else(|| A::Error::missing_field("config"))?,
                    image,
                    multimodal,
                    audio,
                    model_settings_builder,
                })
            }
        }

        deserializer.deserialize_map(ModelDataVisitor)
    }
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
/// * `V13` - Version with no call id tokenization and better prompt caching
/// * `V15` - Latest version with model settings support
///
/// Variants are declared in ascending order, so comparison operators
/// (`<`, `>=`, ...) follow the version ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TokenizerVersion {
    V3,
    V7,
    V11,
    V13,
    V15,
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
            "v15" => Some(Self::V15),
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
            Self::V15 => "v15",
        }
    }

    /// Returns the numeric part of the version (e.g., 15 for `V15`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tekken::config::TokenizerVersion;
    ///
    /// assert_eq!(TokenizerVersion::V15.version_num(), 15);
    /// ```
    #[must_use]
    pub const fn version_num(&self) -> u32 {
        match self {
            Self::V3 => 3,
            Self::V7 => 7,
            Self::V11 => 11,
            Self::V13 => 13,
            Self::V15 => 15,
        }
    }

    /// Returns whether this version supports model settings (v15 and later).
    ///
    /// Tokenizer files of these versions may contain a `model_settings_builder`
    /// section constraining request-level model settings such as reasoning effort.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tekken::config::TokenizerVersion;
    ///
    /// assert!(TokenizerVersion::V15.supports_model_settings());
    /// assert!(!TokenizerVersion::V13.supports_model_settings());
    /// ```
    #[must_use]
    pub const fn supports_model_settings(&self) -> bool {
        self.version_num() >= 15
    }

    /// Returns whether this version requires special tokens to be listed in
    /// the tokenizer file (versions after v7).
    ///
    /// Files of earlier versions may omit the `special_tokens` section, in
    /// which case a deprecated built-in list is used.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tekken::config::TokenizerVersion;
    ///
    /// assert!(TokenizerVersion::V11.requires_explicit_special_tokens());
    /// assert!(!TokenizerVersion::V7.requires_explicit_special_tokens());
    /// ```
    #[must_use]
    pub const fn requires_explicit_special_tokens(&self) -> bool {
        self.version_num() > 7
    }

    /// Returns whether this version still accepts the deprecated `multimodal`
    /// key for the image configuration (v11 and earlier).
    ///
    /// Later versions must spell that section `image`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tekken::config::TokenizerVersion;
    ///
    /// assert!(TokenizerVersion::V11.allows_deprecated_multimodal_key());
    /// assert!(!TokenizerVersion::V13.allows_deprecated_multimodal_key());
    /// ```
    #[must_use]
    pub const fn allows_deprecated_multimodal_key(&self) -> bool {
        self.version_num() <= 11
    }
}
