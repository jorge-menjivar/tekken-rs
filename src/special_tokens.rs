use serde::{Deserialize, Serialize};

/// Enumeration of special tokens used in the Tekken tokenizer.
///
/// Special tokens are used to mark different types of content and control sequences
/// in text and multimodal processing. Each variant corresponds to a specific
/// token string that has special meaning in the tokenization process.
///
/// # Token Categories
///
/// - **Control tokens**: UNK, BOS, EOS, PAD for general sequence control
/// - **Instruction tokens**: `BeginInst`, `EndInst` for instruction formatting
/// - **Tool tokens**: Various tokens for tool call formatting and results
/// - **Image tokens**: IMG, `ImgBreak`, `ImgEnd` for image content
/// - **Audio tokens**: Audio, `BeginAudio`, Transcribe for audio content
/// - **Code tokens**: Prefix, Middle, Suffix for code completion
/// - **System tokens**: `BeginSystem`, `EndSystem` for system prompts
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecialTokens {
    Unk,
    Bos,
    Eos,
    BeginInst,
    EndInst,
    BeginTools,
    EndTools,
    BeginToolResults,
    EndToolResults,
    ToolCalls,
    Img,
    Pad,
    ImgBreak,
    ImgEnd,
    Prefix,
    Middle,
    Suffix,
    BeginSystem,
    EndSystem,
    BeginToolContent,
    Audio,
    BeginAudio,
    Transcribe,
    Args,
    CallId,
}

impl SpecialTokens {
    /// Returns the string representation of the special token.
    ///
    /// Each special token has a corresponding string representation that is used
    /// in the actual tokenization process. This method provides the mapping from
    /// the enum variant to its string form.
    ///
    /// # Returns
    ///
    /// The string representation of the token.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tekken::special_tokens::SpecialTokens;
    ///
    /// assert_eq!(SpecialTokens::Bos.as_str(), "<s>");
    /// assert_eq!(SpecialTokens::Eos.as_str(), "</s>");
    /// assert_eq!(SpecialTokens::BeginAudio.as_str(), "[BEGIN_AUDIO]");
    /// ```
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Unk => "<unk>",
            Self::Bos => "<s>",
            Self::Eos => "</s>",
            Self::BeginInst => "[INST]",
            Self::EndInst => "[/INST]",
            Self::BeginTools => "[AVAILABLE_TOOLS]",
            Self::EndTools => "[/AVAILABLE_TOOLS]",
            Self::BeginToolResults => "[TOOL_RESULTS]",
            Self::EndToolResults => "[/TOOL_RESULTS]",
            Self::ToolCalls => "[TOOL_CALLS]",
            Self::Img => "[IMG]",
            Self::Pad => "<pad>",
            Self::ImgBreak => "[IMG_BREAK]",
            Self::ImgEnd => "[IMG_END]",
            Self::Prefix => "[PREFIX]",
            Self::Middle => "[MIDDLE]",
            Self::Suffix => "[SUFFIX]",
            Self::BeginSystem => "[SYSTEM_PROMPT]",
            Self::EndSystem => "[/SYSTEM_PROMPT]",
            Self::BeginToolContent => "[TOOL_CONTENT]",
            Self::Audio => "[AUDIO]",
            Self::BeginAudio => "[BEGIN_AUDIO]",
            Self::Transcribe => "[TRANSCRIBE]",
            Self::Args => "[ARGS]",
            Self::CallId => "[CALL_ID]",
        }
    }
}

/// Policy for handling special tokens during decoding.
///
/// This enum defines how special tokens should be treated when converting
/// token sequences back to text. Different policies allow for different
/// use cases and output formats.
///
/// # Variants
///
/// - `Ignore`: Skip special tokens completely in the output
/// - `Keep`: Include special tokens in their string form in the output
/// - `Raise`: Raise an error if special tokens are encountered during decoding
///
/// # Examples
///
/// ```rust,no_run
/// use tekken::special_tokens::SpecialTokenPolicy;
/// use tekken::tekkenizer::Tekkenizer;
///
/// let tokenizer = Tekkenizer::from_file("tekken.json")?;
/// let tokens = vec![1, 22177, 2]; // [BOS, "Hello", EOS]
///
/// // Keep special tokens: "<s>Hello</s>"
/// let with_special = tokenizer.decode(&tokens, SpecialTokenPolicy::Keep)?;
///
/// // Ignore special tokens: "Hello"
/// let without_special = tokenizer.decode(&tokens, SpecialTokenPolicy::Ignore)?;
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialTokenPolicy {
    /// Skip special tokens during decoding, excluding them from the output.
    Ignore,
    /// Include special tokens in their string representation in the output.
    Keep,
    /// Raise an error if special tokens are encountered during decoding.
    Raise,
}

/// Information about a special token including its rank and properties.
///
/// This struct contains metadata about a special token, including its position
/// in the vocabulary (rank), string representation, and whether it's a control token.
///
/// # Fields
///
/// * `rank` - Position of the token in the vocabulary (token ID)
/// * `token_str` - String representation of the token
/// * `is_control` - Whether this is a control token (affects processing behavior)
///
/// # Examples
///
/// ```rust
/// use tekken::special_tokens::SpecialTokenInfo;
///
/// let bos_token = SpecialTokenInfo {
///     rank: 1,
///     token_str: "<s>".to_string(),
///     is_control: true,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokenInfo {
    /// The position of this token in the vocabulary (used as token ID).
    pub rank: usize,
    /// The string representation of this special token.
    pub token_str: String,
    /// Whether this token is a control token that affects tokenizer behavior.
    pub is_control: bool,
}
