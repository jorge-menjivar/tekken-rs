use base64::{Engine as _, engine::general_purpose};
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::path::Path;
use tiktoken_rs::CoreBPE;

use crate::audio::{Audio, AudioConfig, AudioEncoder, AudioEncoding};
use crate::config::{ModelData, TekkenConfig, TokenInfo, TokenizerVersion};
use crate::errors::{Result, TokenizerError};
use crate::special_tokens::{SpecialTokenInfo, SpecialTokenPolicy, SpecialTokens};

/// A Tekken tokenizer that supports both text and audio tokenization.
///
/// The Tekkenizer is designed to handle multimodal input, supporting both text
/// tokenization using BPE (Byte Pair Encoding) and audio tokenization using mel-scale
/// spectrograms.
///
/// # Examples
///
/// ```rust,no_run
/// use tekken::tekkenizer::Tekkenizer;
/// use tekken::special_tokens::SpecialTokenPolicy;
///
/// // Load tokenizer from file
/// let tokenizer = Tekkenizer::from_file("tekken.json")?;
///
/// // Encode text
/// let tokens = tokenizer.encode("Hello world!", true, true)?;
///
/// // Decode tokens
/// let text = tokenizer.decode(&tokens, SpecialTokenPolicy::Keep)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Tekkenizer {
    tekkenizer: CoreBPE,
    vocab_size: usize,
    num_special_tokens: usize,
    version: TokenizerVersion,
    pattern: String,
    special_tokens: Vec<SpecialTokenInfo>,
    special_tokens_map: HashMap<String, usize>,
    vocab: Vec<String>,
    vocab_tokens: Vec<TokenInfo>,
    audio_config: Option<AudioConfig>,
    audio_encoder: Option<AudioEncoder>,
}

impl Tekkenizer {
    /// Creates a new Tekkenizer with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `vocab` - The vocabulary tokens with their byte representations
    /// * `special_tokens` - Special tokens used for control sequences
    /// * `pattern` - Regex pattern for tokenization
    /// * `vocab_size` - Total vocabulary size including special tokens
    /// * `num_special_tokens` - Number of special tokens
    /// * `version` - Tokenizer version
    /// * `audio_config` - Optional audio configuration for multimodal support
    ///
    /// # Returns
    ///
    /// A new `Tekkenizer` instance or an error if configuration is invalid.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Vocabulary size is inconsistent with provided tokens
    /// - Special tokens contain duplicates
    /// - Audio configuration is invalid
    /// - Core BPE creation fails
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(
        vocab: Vec<TokenInfo>,
        special_tokens: &Vec<SpecialTokenInfo>,
        pattern: &str,
        vocab_size: usize,
        num_special_tokens: usize,
        version: TokenizerVersion,
        audio_config: Option<AudioConfig>,
    ) -> Result<Self> {
        if vocab_size > vocab.len() + num_special_tokens {
            return Err(TokenizerError::InvalidConfig(format!(
                "vocab_size ({}) must be <= vocab.len() ({}) + num_special_tokens ({})",
                vocab_size,
                vocab.len(),
                num_special_tokens
            )));
        }

        // Check special tokens are unique
        let mut token_strings = std::collections::HashSet::new();
        for token in special_tokens {
            if !token_strings.insert(&token.token_str) {
                return Err(TokenizerError::InvalidConfig(format!(
                    "Duplicate special token: {}",
                    token.token_str
                )));
            }
        }

        if special_tokens.len() > num_special_tokens {
            return Err(TokenizerError::InvalidConfig(format!(
                "special_tokens.len() ({}) must be <= num_special_tokens ({})",
                special_tokens.len(),
                num_special_tokens
            )));
        }

        // Fill missing special tokens
        let mut all_special_tokens = special_tokens.clone();
        for i in special_tokens.len()..num_special_tokens {
            all_special_tokens.push(SpecialTokenInfo {
                rank: i,
                token_str: format!("<SPECIAL_{i}>"),
                is_control: true,
            });
        }

        let inner_vocab_size = vocab_size - num_special_tokens;
        let vocab_tokens_copy = vocab.clone();
        let mergeable_ranks = reload_mergeable_ranks(vocab, inner_vocab_size)?;

        // Create tiktoken CoreBPE from mergeable ranks
        let special_tokens: FxHashMap<String, u32> = FxHashMap::default();

        let tekkenizer = CoreBPE::new(mergeable_ranks.clone(), special_tokens, pattern)
            .map_err(|e| TokenizerError::InvalidConfig(format!("Failed to create CoreBPE: {e}")))?;

        // Create special tokens map
        let special_tokens_map: HashMap<String, usize> = all_special_tokens
            .iter()
            .map(|token| (token.token_str.clone(), token.rank))
            .collect();

        // Create reverse lookup map for efficient vocabulary string creation
        let rank_to_bytes: FxHashMap<u32, &Vec<u8>> = mergeable_ranks
            .iter()
            .map(|(bytes, &rank)| (rank, bytes))
            .collect();

        // Create vocabulary
        let vocab_strings: Vec<String> = (0..vocab_size)
            .map(|i| {
                if i < num_special_tokens {
                    all_special_tokens[i].token_str.clone()
                } else {
                    // Get token string from tiktoken using efficient lookup
                    #[allow(clippy::cast_possible_truncation)]
                    let token_id = (i - num_special_tokens) as u32;
                    rank_to_bytes.get(&token_id).map_or_else(
                        || "<?>".to_string(),
                        |bytes| String::from_utf8_lossy(bytes).to_string(),
                    )
                }
            })
            .collect();

        // Set up audio encoder if audio config is provided
        let audio_encoder = if let Some(ref config) = audio_config {
            let audio_token_id = special_tokens_map
                .get(SpecialTokens::Audio.as_str())
                .ok_or_else(|| {
                    TokenizerError::TokenNotFound("Audio token not found".to_string())
                })?;
            let begin_audio_token_id = special_tokens_map
                .get(SpecialTokens::BeginAudio.as_str())
                .ok_or_else(|| {
                TokenizerError::TokenNotFound("BeginAudio token not found".to_string())
            })?;

            #[allow(clippy::cast_possible_truncation)]
            Some(AudioEncoder::new(
                config.clone(),
                *audio_token_id as u32,
                *begin_audio_token_id as u32,
            ))
        } else {
            None
        };

        Ok(Self {
            tekkenizer,
            vocab_size,
            num_special_tokens,
            version,
            pattern: pattern.to_string(),
            special_tokens: all_special_tokens,
            special_tokens_map,
            vocab: vocab_strings,
            vocab_tokens: vocab_tokens_copy,
            audio_config,
            audio_encoder,
        })
    }

    /// Loads a tokenizer from a JSON configuration file.
    ///
    /// The file should contain tokenizer configuration including vocabulary,
    /// special tokens, patterns, and optional audio configuration.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the tokenizer configuration file (typically `tekken.json`)
    ///
    /// # Returns
    ///
    /// A new `Tekkenizer` instance loaded from the file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be read
    /// - JSON parsing fails
    /// - Configuration is invalid
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use tekken::tekkenizer::Tekkenizer;
    ///
    /// let tokenizer = Tekkenizer::from_file("tekken.json")?;
    /// println!("Loaded tokenizer version: {:?}", tokenizer.version());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let model_data: ModelData = serde_json::from_str(&content)?;

        let version =
            TokenizerVersion::from_string(&model_data.config.version).ok_or_else(|| {
                TokenizerError::InvalidConfig(format!(
                    "Unknown version: {}",
                    model_data.config.version
                ))
            })?;

        let special_tokens = model_data.special_tokens.unwrap_or_else(|| {
            // Use deprecated special tokens for older versions
            get_deprecated_special_tokens()
        });

        Self::new(
            model_data.vocab,
            &special_tokens,
            &model_data.config.pattern,
            model_data.config.default_vocab_size,
            model_data.config.default_num_special_tokens,
            version,
            model_data.audio,
        )
    }

    /// Returns the total vocabulary size including special tokens.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use tekken::tekkenizer::Tekkenizer;
    /// # let tokenizer = Tekkenizer::from_file("tekken.json")?;
    /// println!("Vocabulary size: {}", tokenizer.vocab_size());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns the number of special tokens in the vocabulary.
    ///
    /// Special tokens include control tokens like BOS, EOS, and audio tokens.
    #[must_use]
    pub const fn num_special_tokens(&self) -> usize {
        self.num_special_tokens
    }

    /// Returns the tokenizer version.
    ///
    /// Different versions may have different vocabulary sizes and special tokens.
    #[must_use]
    pub const fn version(&self) -> &TokenizerVersion {
        &self.version
    }

    /// Returns the token ID (u32) for the Beginning of Sequence (BOS) token.
    ///
    /// # Errors
    ///
    /// Returns an error if the BOS token is not found in the vocabulary.
    pub fn bos_id(&self) -> Result<u32> {
        self.get_control_token(SpecialTokens::Bos.as_str())
    }

    /// Returns the token ID (u32) for the End of Sequence (EOS) token.
    ///
    /// # Errors
    ///
    /// Returns an error if the EOS token is not found in the vocabulary.
    pub fn eos_id(&self) -> Result<u32> {
        self.get_control_token(SpecialTokens::Eos.as_str())
    }

    /// Returns the token ID (u32) for the padding (PAD) token.
    ///
    /// # Errors
    ///
    /// Returns an error if the PAD token is not found in the vocabulary.
    pub fn pad_id(&self) -> Result<u32> {
        self.get_control_token(SpecialTokens::Pad.as_str())
    }

    /// Returns the token ID (u32) for the Unknown (UNK) token.
    ///
    /// # Errors
    ///
    /// Returns an error if the UNK token is not found in the vocabulary.
    pub fn unk_id(&self) -> Result<u32> {
        self.get_control_token(SpecialTokens::Unk.as_str())
    }

    /// Returns the token ID (u32) for a specific control token by its string representation.
    ///
    /// # Arguments
    ///
    /// * `token_str` - The string representation of the control token (e.g., "<s>", "</s>")
    ///
    /// # Returns
    ///
    /// The token ID (u32) if found.
    ///
    /// # Errors
    ///
    /// Returns an error if the control token is not found in the vocabulary.
    #[allow(clippy::cast_possible_truncation)]
    pub fn get_control_token(&self, token_str: &str) -> Result<u32> {
        self.special_tokens_map
            .get(token_str)
            .map(|&id| id as u32)
            .ok_or_else(|| {
                let available_tokens: Vec<&String> = self.special_tokens_map.keys().collect();
                TokenizerError::TokenNotFound(format!(
                    "Unknown control token: '{token_str}'. Available special tokens: {available_tokens:?}",
                ))
            })
    }

    /// Returns a reference to the complete vocabulary as a slice of strings.
    ///
    /// The vocabulary includes both special tokens and regular tokens.
    /// Token IDs (u32) correspond to indices in this slice.
    #[must_use]
    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }

    /// Encodes text into a sequence of token IDs.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to tokenize
    /// * `add_bos` - Whether to add a Beginning of Sequence token at the start
    /// * `add_eos` - Whether to add an End of Sequence token at the end
    ///
    /// # Returns
    ///
    /// A vector of token IDs (u32) representing the encoded text.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use tekken::tekkenizer::Tekkenizer;
    /// # let tokenizer = Tekkenizer::from_file("tekken.json")?;
    /// let tokens = tokenizer.encode("Hello world!", true, true)?;
    /// println!("Encoded tokens: {:?}", tokens);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer is not initialized.
    #[allow(clippy::cast_possible_truncation)]
    pub fn encode(
        &self,
        text: &str,
        add_beginning_of_sequence: bool,
        add_end_of_sequence: bool,
    ) -> Result<Vec<u32>> {
        let (tokens, _) = self
            .tekkenizer
            .encode(text, &std::collections::HashSet::new());
        let mut tokens: Vec<u32> = tokens;

        // Shift tokens to account for special tokens
        for token in &mut tokens {
            *token += self.num_special_tokens as u32;
        }

        if add_beginning_of_sequence {
            let bos_id = self.bos_id()?;
            tokens.insert(0, bos_id);
        }

        if add_end_of_sequence {
            let eos_id = self.eos_id()?;
            tokens.push(eos_id);
        }

        Ok(tokens)
    }

    /// Decodes a sequence of token IDs back into text.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The token IDs (u32) to decode
    /// * `special_token_policy` - How to handle special tokens during decoding:
    ///   - `Keep`: Include special tokens in the output
    ///   - `Ignore`: Skip special tokens
    ///   - `Raise`: Error if special tokens are encountered
    ///
    /// # Returns
    ///
    /// The decoded text string.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use tekken::tekkenizer::Tekkenizer;
    /// # use tekken::special_tokens::SpecialTokenPolicy;
    /// # let tokenizer = Tekkenizer::from_file("tekken.json")?;
    /// # let tokens = vec![1, 22177, 1044, 4304, 2];
    /// let text = tokenizer.decode(&tokens, SpecialTokenPolicy::Keep)?;
    /// println!("Decoded text: {}", text);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// If the token IDs are invalid or the special token policy is not recognized.
    pub fn decode(
        &self,
        tokens: &[u32],
        special_token_policy: SpecialTokenPolicy,
    ) -> Result<String> {
        let decoded_parts = self.decode_all(tokens, special_token_policy)?;
        Ok(decoded_parts.join(""))
    }

    /// Decodes token IDs into separate strings, grouping consecutive special/non-special tokens.
    ///
    /// This method preserves the grouping of tokens, returning a vector where each element
    /// represents a contiguous group of either special tokens or regular tokens.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The token IDs (u32) to decode
    /// * `special_token_policy` - How to handle special tokens during decoding
    ///
    /// # Returns
    ///
    /// A vector of decoded string segments.
    ///
    /// # Errors
    ///
    /// If the token IDs are invalid or the special token policy is not recognized.
    #[allow(clippy::cast_possible_truncation)]
    pub fn decode_all(
        &self,
        tokens: &[u32],
        special_token_policy: SpecialTokenPolicy,
    ) -> Result<Vec<String>> {
        let mut decoded = Vec::new();
        let mut current_group = Vec::new();
        let mut current_is_special = None;

        for &token_id in tokens {
            #[allow(clippy::cast_possible_truncation)]
            let is_special = token_id < self.num_special_tokens as u32;

            if current_is_special.is_none() {
                current_is_special = Some(is_special);
            }

            if current_is_special == Some(is_special) {
                current_group.push(token_id);
            } else {
                // Process the current group
                if let Some(was_special) = current_is_special {
                    self.decode_group(
                        &current_group,
                        was_special,
                        &mut decoded,
                        special_token_policy,
                    )?;
                }

                // Start new group
                current_group.clear();
                current_group.push(token_id);
                current_is_special = Some(is_special);
            }
        }

        // Process the last group
        if let Some(was_special) = current_is_special {
            self.decode_group(
                &current_group,
                was_special,
                &mut decoded,
                special_token_policy,
            )?;
        }

        Ok(decoded)
    }

    /// Helper method to decode a group of tokens that are all special or all non-special.
    ///
    /// # Arguments
    ///
    /// * `group` - The token IDs (u32) in the current group
    /// * `is_special` - Whether this group contains special tokens
    /// * `decoded` - Output vector to append decoded strings to
    /// * `special_token_policy` - How to handle special tokens
    #[allow(clippy::cast_possible_truncation)]
    fn decode_group(
        &self,
        group: &[u32],
        is_special: bool,
        decoded: &mut Vec<String>,
        special_token_policy: SpecialTokenPolicy,
    ) -> Result<()> {
        if is_special {
            match special_token_policy {
                SpecialTokenPolicy::Raise => {
                    return Err(TokenizerError::SpecialTokenPolicy(format!(
                        "Decoding tokens that contain special tokens ({group:?}) is not allowed",
                    )));
                }
                SpecialTokenPolicy::Keep => {
                    for &token_id in group {
                        decoded.push(self.special_tokens[token_id as usize].token_str.clone());
                    }
                }
                SpecialTokenPolicy::Ignore => {
                    // Skip special tokens
                }
            }
        } else {
            // Decode non-special tokens
            #[allow(clippy::cast_possible_truncation)]
            let shifted_tokens: Vec<u32> = group
                .iter()
                .map(|&t| t - self.num_special_tokens as u32)
                .collect();
            let decoded_text = self
                .tekkenizer
                .decode(shifted_tokens)
                .map_err(|e| TokenizerError::Tokenizers(format!("{e:?}")))?;
            decoded.push(decoded_text);
        }

        Ok(())
    }

    /// Checks if a token ID represents a special token.
    ///
    /// Special tokens include control tokens like BOS, EOS, instruction tokens, etc.
    ///
    /// # Arguments
    ///
    /// * `token_id` - The token ID (u32) to check
    ///
    /// # Returns
    ///
    /// `true` if the token is a special token, `false` otherwise.
    #[must_use]
    pub const fn is_special_token(&self, token_id: u32) -> bool {
        (token_id as usize) < self.num_special_tokens
    }

    /// Checks if a token ID represents a single byte token.
    ///
    /// In BPE tokenization, the first 256 tokens typically represent individual bytes.
    ///
    /// # Arguments
    ///
    /// * `token_id` - The token ID (u32) to check
    ///
    /// # Returns
    ///
    /// `true` if the token represents a single byte, `false` otherwise.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn is_byte(&self, token_id: u32) -> bool {
        #[allow(clippy::cast_possible_truncation)]
        if token_id < self.num_special_tokens as u32 {
            false
        } else {
            #[allow(clippy::cast_possible_truncation)]
            let shifted_id = token_id - self.num_special_tokens as u32;
            shifted_id < 256
        }
    }

    /// Converts a single token ID to its string representation.
    ///
    /// This method includes special tokens in the output.
    ///
    /// # Arguments
    ///
    /// * `token_id` - The token ID (u32) to convert
    ///
    /// # Returns
    ///
    /// The string representation of the token.
    ///
    /// # Errors
    ///
    /// Returns an error if the token ID is out of vocabulary range.
    pub fn id_to_piece(&self, token_id: u32) -> Result<String> {
        // Validate token ID is within vocabulary range
        if token_id as usize >= self.vocab_size {
            return Err(TokenizerError::InvalidConfig(format!(
                "Token ID {} is out of vocabulary range (0-{})",
                token_id,
                self.vocab_size - 1
            )));
        }

        self.decode(&[token_id], SpecialTokenPolicy::Keep)
    }

    /// Converts a single token ID to its byte representation.
    ///
    /// # Arguments
    ///
    /// * `token_id` - The token ID (u32) to convert
    /// * `special_token_policy` - How to handle special tokens
    ///
    /// # Returns
    ///
    /// The byte representation of the token.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Token ID is invalid (out of vocabulary range)
    /// - Special token policy is Raise and token is special
    /// - Token cannot be converted to bytes (should be rare)
    #[allow(clippy::cast_possible_truncation)]
    pub fn id_to_byte_piece(
        &self,
        token_id: u32,
        special_token_policy: SpecialTokenPolicy,
    ) -> Result<Vec<u8>> {
        // Validate token ID is within vocabulary range
        if token_id as usize >= self.vocab_size {
            return Err(TokenizerError::InvalidConfig(format!(
                "Token ID {} is out of vocabulary range (0-{})",
                token_id,
                self.vocab_size - 1
            )));
        }

        #[allow(clippy::cast_possible_truncation)]
        if token_id < self.num_special_tokens as u32 {
            match special_token_policy {
                SpecialTokenPolicy::Keep => Ok(self.special_tokens[token_id as usize]
                    .token_str
                    .as_bytes()
                    .to_vec()),
                SpecialTokenPolicy::Raise => Err(TokenizerError::SpecialTokenPolicy(format!(
                    "Token ID {} is a special token ({}), cannot convert to byte piece with Raise policy",
                    token_id, self.special_tokens[token_id as usize].token_str
                ))),
                SpecialTokenPolicy::Ignore => Ok(vec![]),
            }
        } else {
            #[allow(clippy::cast_possible_truncation)]
            let shifted_id = token_id - self.num_special_tokens as u32;

            // Try to decode the token - this might fail for some byte tokens
            match self.tekkenizer.decode(vec![shifted_id]) {
                Ok(decoded) => Ok(decoded.as_bytes().to_vec()),
                Err(e) => {
                    // If decoding fails, try to get the raw bytes from the vocabulary
                    // This can happen with incomplete UTF-8 sequences in individual byte tokens
                    if let Some(vocab_entry) = self.vocab.get(token_id as usize) {
                        Ok(vocab_entry.as_bytes().to_vec())
                    } else {
                        Err(TokenizerError::Tokenizers(format!(
                            "Failed to decode token ID {token_id} to bytes: {e:?}. Token may represent invalid UTF-8 sequence.",
                        )))
                    }
                }
            }
        }
    }

    /// Encodes audio data into tokens that can be mixed with text tokens.
    ///
    /// This method converts audio waveforms into token sequences using mel-scale
    /// spectrogram processing and audio-specific special tokens.
    ///
    /// # Arguments
    ///
    /// * `audio` - The audio data to encode
    ///
    /// # Returns
    ///
    /// An `AudioEncoding` containing the token sequence and processed audio data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Audio encoder is not configured
    /// - Audio processing fails
    /// - Resampling fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use tekken::tekkenizer::Tekkenizer;
    /// # use tekken::audio::Audio;
    /// # let tokenizer = Tekkenizer::from_file("tekken.json")?;
    /// let audio = Audio::from_file("audio.wav")?;
    /// let encoding = tokenizer.encode_audio(audio)?;
    /// println!("Audio encoded to {} tokens", encoding.tokens.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn encode_audio(&self, audio: Audio) -> Result<AudioEncoding> {
        self.audio_encoder.as_ref().map_or_else(
            || {
                Err(TokenizerError::Audio(
                    "Audio encoder not configured".to_string(),
                ))
            },
            |encoder| encoder.encode(audio),
        )
    }

    /// Checks if this tokenizer instance supports audio processing.
    ///
    /// Audio support depends on the tokenizer configuration containing audio settings
    /// and the presence of required audio special tokens.
    ///
    /// # Returns
    ///
    /// `true` if audio encoding is available, `false` otherwise.
    #[must_use]
    pub const fn has_audio_support(&self) -> bool {
        self.audio_encoder.is_some()
    }

    /// Returns a reference to the audio configuration, if available.
    ///
    /// # Returns
    ///
    /// An optional reference to the `AudioConfig` if audio support is configured.
    /// Returns `None` if the tokenizer was not initialized with audio support.
    #[must_use]
    pub const fn audio_config(&self) -> Option<&AudioConfig> {
        self.audio_config.as_ref()
    }

    /// Saves the tokenizer to a file in the same format as `ModelData`.
    ///
    /// This method serializes the tokenizer configuration, vocabulary, and special tokens
    /// to a JSON file that can be loaded later using `from_file`.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the tokenizer configuration file will be written
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful write, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be created or written to
    /// - JSON serialization fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use tekken::tekkenizer::Tekkenizer;
    /// # let tokenizer = Tekkenizer::from_file("tekken.json")?;
    /// tokenizer.to_file("output_tekken.json")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        // Create the model data structure
        let model_data = ModelData {
            vocab: self.vocab_tokens.clone(),
            special_tokens: Some(self.special_tokens.clone()),
            config: TekkenConfig {
                pattern: self.pattern.clone(),
                num_vocab_tokens: self.vocab_size - self.num_special_tokens,
                default_vocab_size: self.vocab_size,
                default_num_special_tokens: self.num_special_tokens,
                version: self.version.as_str().to_string(),
            },
            audio: self.audio_config.clone(),
        };

        // Serialize to JSON
        let json_content = serde_json::to_string_pretty(&model_data)?;

        // Write to file
        std::fs::write(path, json_content)?;

        Ok(())
    }
}

/// Processes vocabulary tokens into a format suitable for tiktoken encoding.
///
/// This function converts token information into the mergeable ranks format
/// required by tiktoken, validating byte tokens and ensuring rank contiguity.
///
/// # Arguments
///
/// * `vocab` - The vocabulary tokens with their byte representations
/// * `max_vocab` - Maximum number of vocabulary tokens to process
///
/// # Returns
///
/// A hash map from byte sequences to token ranks (u32 for tiktoken).
#[allow(clippy::cast_possible_truncation)]
fn reload_mergeable_ranks(
    vocab: Vec<TokenInfo>,
    max_vocab: usize,
) -> Result<FxHashMap<Vec<u8>, u32>> {
    let vocab = if vocab.len() > max_vocab {
        vocab.into_iter().take(max_vocab).collect()
    } else {
        vocab
    };

    let mut ranks = FxHashMap::default();

    for token in vocab {
        let token_bytes = general_purpose::STANDARD.decode(&token.token_bytes)?;

        // Verify byte tokens for first 256 tokens
        #[allow(clippy::cast_possible_truncation)]
        if token.rank < 256 && token_bytes != vec![token.rank as u8] {
            return Err(TokenizerError::InvalidConfig(format!(
                "Expected byte token at rank {} to be [{}], got {:?}",
                token.rank, token.rank, token_bytes
            )));
        }

        #[allow(clippy::cast_possible_truncation)]
        ranks.insert(token_bytes, token.rank as u32);
    }

    // Verify ranks are contiguous
    #[allow(clippy::cast_possible_truncation)]
    let expected_ranks: std::collections::HashSet<_> = (0..ranks.len() as u32).collect();
    let actual_ranks: std::collections::HashSet<_> = ranks.values().copied().collect();

    if expected_ranks != actual_ranks {
        return Err(TokenizerError::InvalidConfig(
            "Vocabulary ranks are not contiguous".to_string(),
        ));
    }

    Ok(ranks)
}

/// Returns the default special tokens for older tokenizer versions.
///
/// This function provides backward compatibility with tokenizer versions
/// that don't specify special tokens in their configuration files.
///
/// # Returns
///
/// A vector of special token information for legacy tokenizers.
#[allow(clippy::too_many_lines)]
fn get_deprecated_special_tokens() -> Vec<SpecialTokenInfo> {
    vec![
        SpecialTokenInfo {
            rank: 0,
            token_str: SpecialTokens::Unk.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 1,
            token_str: SpecialTokens::Bos.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 2,
            token_str: SpecialTokens::Eos.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 3,
            token_str: SpecialTokens::BeginInst.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 4,
            token_str: SpecialTokens::EndInst.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 5,
            token_str: SpecialTokens::BeginTools.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 6,
            token_str: SpecialTokens::EndTools.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 7,
            token_str: SpecialTokens::BeginToolResults.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 8,
            token_str: SpecialTokens::EndToolResults.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 9,
            token_str: SpecialTokens::ToolCalls.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 10,
            token_str: SpecialTokens::Img.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 11,
            token_str: SpecialTokens::Pad.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 12,
            token_str: SpecialTokens::ImgBreak.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 13,
            token_str: SpecialTokens::ImgEnd.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 14,
            token_str: SpecialTokens::Prefix.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 15,
            token_str: SpecialTokens::Middle.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 16,
            token_str: SpecialTokens::Suffix.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 17,
            token_str: SpecialTokens::BeginSystem.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 18,
            token_str: SpecialTokens::EndSystem.as_str().to_string(),
            is_control: true,
        },
        SpecialTokenInfo {
            rank: 19,
            token_str: SpecialTokens::BeginToolContent.as_str().to_string(),
            is_control: true,
        },
    ]
}
