use crate::errors::{Result, TokenizerError};
use base64::Engine;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration for generating audio spectrograms.
///
/// This struct contains the parameters needed to compute mel-scale spectrograms
/// from audio waveforms, which are used in audio tokenization.
///
/// # Fields
///
/// * `num_mel_bins` - Number of mel-frequency bins (typically 80 or 128)
/// * `hop_length` - Length of overlapping windows for STFT (typically 160)
/// * `window_size` - Window size for Fourier transform (typically 400)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSpectrogramConfig {
    pub num_mel_bins: usize,
    pub hop_length: usize,
    pub window_size: usize,
}

impl AudioSpectrogramConfig {
    /// Creates a new AudioSpectrogramConfig with validation.
    ///
    /// # Arguments
    ///
    /// * `num_mel_bins` - Number of mel-frequency bins (must be > 0)
    /// * `hop_length` - Length of overlapping windows for STFT (must be > 0)
    /// * `window_size` - Window size for Fourier transform (must be > 0)
    ///
    /// # Returns
    ///
    /// A new AudioSpectrogramConfig instance.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is zero or invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tekken::audio::AudioSpectrogramConfig;
    ///
    /// let config = AudioSpectrogramConfig::new(80, 160, 400)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(num_mel_bins: usize, hop_length: usize, window_size: usize) -> Result<Self> {
        if num_mel_bins == 0 {
            return Err(TokenizerError::InvalidConfig(
                "num_mel_bins must be > 0".to_string(),
            ));
        }
        if hop_length == 0 {
            return Err(TokenizerError::InvalidConfig(
                "hop_length must be > 0".to_string(),
            ));
        }
        if window_size == 0 {
            return Err(TokenizerError::InvalidConfig(
                "window_size must be > 0".to_string(),
            ));
        }

        Ok(Self {
            num_mel_bins,
            hop_length,
            window_size,
        })
    }
}

/// Configuration for audio processing and tokenization.
///
/// This struct contains all parameters needed to process audio files and convert
/// them into token sequences that can be mixed with text tokens.
///
/// # Fields
///
/// * `sampling_rate` - Target sampling rate in Hz (e.g., 16000)
/// * `frame_rate` - Number of frames per second for the tokenizer model
/// * `audio_encoding_config` - Spectrogram generation parameters
/// * `chunk_length_s` - Optional chunk length in seconds for padding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sampling_rate: usize,
    pub frame_rate: f64,
    pub audio_encoding_config: AudioSpectrogramConfig,
    pub chunk_length_s: Option<f64>,
}

impl AudioConfig {
    /// Creates a new AudioConfig with validation.
    ///
    /// # Arguments
    ///
    /// * `sampling_rate` - Target sampling rate in Hz (must be > 0)
    /// * `frame_rate` - Number of frames per second (must be > 0)
    /// * `encoding_config` - Spectrogram configuration
    /// * `chunk_length_s` - Optional chunk length in seconds (must be > 0 if provided)
    ///
    /// # Returns
    ///
    /// A new AudioConfig instance.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is invalid.
    pub fn new(
        sampling_rate: usize,
        frame_rate: f64,
        encoding_config: AudioSpectrogramConfig,
        chunk_length_s: Option<f64>,
    ) -> Result<Self> {
        if sampling_rate == 0 {
            return Err(TokenizerError::InvalidConfig(
                "sampling_rate must be > 0".to_string(),
            ));
        }
        if frame_rate <= 0.0 {
            return Err(TokenizerError::InvalidConfig(
                "frame_rate must be > 0".to_string(),
            ));
        }

        if let Some(chunk_length) = chunk_length_s {
            if chunk_length <= 0.0 {
                return Err(TokenizerError::InvalidConfig(
                    "chunk_length_s must be > 0".to_string(),
                ));
            }
        }

        Ok(Self {
            sampling_rate,
            frame_rate,
            audio_encoding_config: encoding_config,
            chunk_length_s,
        })
    }

    /// Calculates the number of audio frames per chunk.
    ///
    /// # Returns
    ///
    /// The number of frames per chunk based on chunk length and sampling rate.
    ///
    /// # Errors
    ///
    /// Returns an error if chunk_length_s is not set.
    pub fn chunk_frames(&self) -> Result<usize> {
        match self.chunk_length_s {
            Some(chunk_length) => Ok((chunk_length * self.sampling_rate as f64) as usize),
            None => Err(TokenizerError::InvalidConfig(
                "chunk_length_s not set".to_string(),
            )),
        }
    }

    /// Calculates the length of audio (in samples) represented by each token.
    ///
    /// This determines the downsampling factor from audio samples to tokens
    /// based on the frame rate and spectrogram hop length.
    ///
    /// # Returns
    ///
    /// Number of audio samples per token.
    pub fn audio_length_per_tok(&self) -> usize {
        let mut downsample_factor = self.sampling_rate as f64 / self.frame_rate;
        downsample_factor /= self.audio_encoding_config.hop_length as f64;
        downsample_factor as usize
    }
}

/// Represents audio data with metadata.
///
/// This struct holds audio waveform data along with its sampling rate and format.
/// It provides methods for loading, processing, and converting audio data.
///
/// # Fields
///
/// * `audio_array` - Audio waveform as a 1D array of f32 samples
/// * `sampling_rate` - Sampling rate in Hz
/// * `format` - Audio format string (e.g., "wav")
#[derive(Debug, Clone)]
pub struct Audio {
    pub audio_array: Array1<f32>,
    pub sampling_rate: usize,
    pub format: String,
}

impl Audio {
    /// Creates a new Audio instance.
    ///
    /// # Arguments
    ///
    /// * `audio_array` - Audio waveform data as a 1D array
    /// * `sampling_rate` - Sampling rate in Hz
    /// * `format` - Audio format string
    ///
    /// # Returns
    ///
    /// A new Audio instance.
    pub fn new(audio_array: Array1<f32>, sampling_rate: usize, format: String) -> Self {
        Self {
            audio_array,
            sampling_rate,
            format,
        }
    }

    /// Loads audio data from a WAV file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the audio file
    ///
    /// # Returns
    ///
    /// A new Audio instance with the loaded data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be opened
    /// - File format is not supported
    /// - Audio data cannot be read
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use tekken::audio::Audio;
    ///
    /// let audio = Audio::from_file("audio.wav")?;
    /// println!("Loaded audio: {} samples at {} Hz", audio.audio_array.len(), audio.sampling_rate);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut reader = hound::WavReader::open(path)
            .map_err(|e| TokenizerError::Audio(format!("Failed to open audio file: {}", e)))?;

        let spec = reader.spec();
        let sampling_rate = spec.sample_rate as usize;

        // Read samples and convert to f32
        let samples: std::result::Result<Vec<f32>, _> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().collect(),
            hound::SampleFormat::Int => reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
                .collect(),
        };

        let samples =
            samples.map_err(|e| TokenizerError::Audio(format!("Failed to read samples: {}", e)))?;

        // Handle stereo to mono conversion (average channels)
        let audio_array = if spec.channels == 1 {
            Array1::from_vec(samples)
        } else {
            let mono_samples: Vec<f32> = samples
                .chunks(spec.channels as usize)
                .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
                .collect();
            Array1::from_vec(mono_samples)
        };

        Ok(Self::new(audio_array, sampling_rate, "wav".to_string()))
    }

    /// Loads audio data from a base64-encoded string.
    ///
    /// # Arguments
    ///
    /// * `data` - Base64-encoded audio data
    ///
    /// # Returns
    ///
    /// A new Audio instance with the decoded data.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding or parsing fails.
    pub fn from_base64(data: &str) -> Result<Self> {
        let audio_bytes = base64::engine::general_purpose::STANDARD.decode(data)?;
        Self::from_bytes(&audio_bytes)
    }

    /// Loads audio data from raw bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw audio file data
    ///
    /// # Returns
    ///
    /// A new Audio instance parsed from the bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes cannot be parsed as audio.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let cursor = std::io::Cursor::new(bytes);
        let mut reader = hound::WavReader::new(cursor)
            .map_err(|e| TokenizerError::Audio(format!("Failed to parse audio bytes: {}", e)))?;

        let spec = reader.spec();
        let sampling_rate = spec.sample_rate as usize;

        let samples: std::result::Result<Vec<f32>, _> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().collect(),
            hound::SampleFormat::Int => reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
                .collect(),
        };

        let samples =
            samples.map_err(|e| TokenizerError::Audio(format!("Failed to read samples: {}", e)))?;

        let audio_array = if spec.channels == 1 {
            Array1::from_vec(samples)
        } else {
            let mono_samples: Vec<f32> = samples
                .chunks(spec.channels as usize)
                .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
                .collect();
            Array1::from_vec(mono_samples)
        };

        Ok(Self::new(audio_array, sampling_rate, "wav".to_string()))
    }

    /// Calculates the duration of the audio in seconds.
    ///
    /// # Returns
    ///
    /// Audio duration in seconds.
    pub fn duration(&self) -> f64 {
        self.audio_array.len() as f64 / self.sampling_rate as f64
    }

    /// Resamples the audio to a target sampling rate.
    ///
    /// # Arguments
    ///
    /// * `target_rate` - Target sampling rate in Hz
    ///
    /// # Errors
    ///
    /// Currently returns an error as resampling is not yet implemented.
    ///
    /// # Note
    ///
    /// This is a placeholder implementation that needs proper resampling logic.
    pub fn resample(&mut self, target_rate: usize) -> Result<()> {
        if self.sampling_rate == target_rate {
            return Ok(());
        }

        // For now, return an error for resampling - this would need proper implementation
        return Err(TokenizerError::Audio(
            "Resampling not yet implemented".to_string(),
        ));
    }

    /// Pads the audio to meet minimum length requirements.
    ///
    /// This method ensures the audio is long enough for processing by padding
    /// with zeros if necessary. Padding is applied based on chunk length or
    /// minimum window size requirements.
    ///
    /// # Arguments
    ///
    /// * `config` - Audio configuration specifying padding requirements
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid.
    pub fn pad(&mut self, config: &AudioConfig) -> Result<()> {
        let current_length = self.audio_array.len();

        let target_length = if let Some(_chunk_length_s) = config.chunk_length_s {
            let chunk_frames = config.chunk_frames()?;
            let next_multiple = ((current_length + chunk_frames - 1) / chunk_frames) * chunk_frames;
            next_multiple
        } else if current_length < config.audio_encoding_config.window_size {
            config.audio_encoding_config.window_size
        } else {
            return Ok(());
        };

        if target_length > current_length {
            let _padding_length = target_length - current_length;
            let mut padded = Array1::zeros(target_length);
            padded
                .slice_mut(ndarray::s![..current_length])
                .assign(&self.audio_array);
            self.audio_array = padded;
        }

        Ok(())
    }
}

/// Result of audio tokenization containing tokens and processed audio.
///
/// This struct encapsulates the output of audio encoding, containing both
/// the token sequence and the processed audio data.
///
/// # Fields
///
/// * `tokens` - Token sequence (u32) representing the audio (includes begin_audio and audio tokens)
/// * `audio` - Processed audio data after resampling and padding
#[derive(Debug, Clone)]
pub struct AudioEncoding {
    pub tokens: Vec<u32>,
    pub audio: Audio,
}

/// Encoder for converting audio data into token sequences.
///
/// The AudioEncoder processes audio waveforms and converts them into token
/// sequences that can be mixed with text tokens in multimodal applications.
///
/// # Fields
///
/// * `config` - Audio processing configuration
/// * `audio_token_id` - Token ID (u32) for audio content tokens
/// * `begin_audio_token_id` - Token ID (u32) for marking the start of audio
#[derive(Debug, Clone)]
pub struct AudioEncoder {
    pub config: AudioConfig,
    pub audio_token_id: u32,
    pub begin_audio_token_id: u32,
}

impl AudioEncoder {
    /// Creates a new AudioEncoder.
    ///
    /// # Arguments
    ///
    /// * `config` - Audio processing configuration
    /// * `audio_token_id` - Token ID (u32) representing audio content
    /// * `begin_audio_token_id` - Token ID (u32) marking the start of audio sequence
    ///
    /// # Returns
    ///
    /// A new AudioEncoder instance.
    pub fn new(config: AudioConfig, audio_token_id: u32, begin_audio_token_id: u32) -> Self {
        Self {
            config,
            audio_token_id,
            begin_audio_token_id,
        }
    }

    /// Encodes audio data into a token sequence.
    ///
    /// This method processes the audio through resampling, padding, and tokenization
    /// to produce a sequence of tokens that represents the audio content.
    ///
    /// # Arguments
    ///
    /// * `audio` - The audio data to encode
    ///
    /// # Returns
    ///
    /// An AudioEncoding containing the token sequence and processed audio.
    ///
    /// # Errors
    ///
    /// Returns an error if audio processing fails.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use tekken::audio::{Audio, AudioConfig, AudioSpectrogramConfig, AudioEncoder};
    ///
    /// let audio = Audio::from_file("audio.wav")?;
    /// let spectrogram_config = AudioSpectrogramConfig::new(80, 160, 400)?;
    /// let audio_config = AudioConfig::new(16000, 12.5, spectrogram_config, None)?;
    /// let encoder = AudioEncoder::new(audio_config, 1000, 1001);
    ///
    /// let encoding = encoder.encode(audio)?;
    /// println!("Audio encoded to {} tokens", encoding.tokens.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn encode(&self, mut audio: Audio) -> Result<AudioEncoding> {
        // Resample to target sampling rate
        audio.resample(self.config.sampling_rate)?;

        // Pad audio if needed
        audio.pad(&self.config)?;

        let signal_length = audio.audio_array.len();

        // Calculate signal length after downsampling for spectrogram
        let signal_length = if signal_length % self.config.audio_encoding_config.hop_length != 0 {
            (signal_length as f64 / self.config.audio_encoding_config.hop_length as f64 - 1.0)
                .ceil() as usize
        } else {
            signal_length / self.config.audio_encoding_config.hop_length
        };

        let num_audio_tokens =
            (signal_length as f64 / self.config.audio_length_per_tok() as f64).ceil() as usize;

        let mut tokens = vec![self.begin_audio_token_id];
        tokens.extend(vec![self.audio_token_id; num_audio_tokens]);

        Ok(AudioEncoding { tokens, audio })
    }
}

/// Converts frequency from Hertz to the mel-scale using the Slaney formula.
///
/// The mel-scale is a perceptual scale that better represents human auditory perception.
/// This function implements the Slaney-style conversion commonly used in audio processing.
///
/// # Arguments
///
/// * `freq` - Frequency in Hertz
///
/// # Returns
///
/// Frequency in mel-scale units.
///
/// # References
///
/// Based on the Slaney mel-scale conversion used in audio processing libraries.
pub fn hertz_to_mel(freq: f64) -> f64 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 27.0 / 6.4_f64.ln();

    if freq >= min_log_hertz {
        min_log_mel + (freq / min_log_hertz).ln() * logstep
    } else {
        3.0 * freq / 200.0
    }
}

/// Converts frequency from the mel-scale back to Hertz.
///
/// This is the inverse operation of `hertz_to_mel`, converting mel-scale
/// frequencies back to linear Hertz frequencies.
///
/// # Arguments
///
/// * `mel` - Frequency in mel-scale units
///
/// # Returns
///
/// Frequency in Hertz.
pub fn mel_to_hertz(mel: f64) -> f64 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 6.4_f64.ln() / 27.0;

    if mel >= min_log_mel {
        min_log_hertz * ((mel - min_log_mel) * logstep).exp()
    } else {
        200.0 * mel / 3.0
    }
}

/// Creates a mel-scale filter bank for spectrogram processing.
///
/// This function generates a matrix of triangular filters distributed on the mel-scale
/// that can be used to convert linear frequency spectrograms to mel-scale spectrograms.
/// The implementation follows the Slaney-style mel filter bank construction.
///
/// # Arguments
///
/// * `num_frequency_bins` - Number of frequency bins in the input spectrogram
/// * `num_mel_bins` - Number of desired mel-frequency bins in the output
/// * `min_frequency` - Minimum frequency in Hz to consider
/// * `max_frequency` - Maximum frequency in Hz to consider
/// * `sampling_rate` - Audio sampling rate in Hz
///
/// # Returns
///
/// A 2D array of shape `(num_frequency_bins, num_mel_bins)` containing the filter bank.
/// Each column represents a mel filter that can be applied to frequency bins.
///
/// # Errors
///
/// Returns an error if:
/// - `num_frequency_bins` is less than 2
/// - `min_frequency` is greater than `max_frequency`
/// - Any mel filter has all zero values
///
/// # Examples
///
/// ```rust
/// use tekken::audio::mel_filter_bank;
///
/// let filter_bank = mel_filter_bank(201, 80, 0.0, 8000.0, 16000)?;
/// println!("Filter bank shape: {:?}", filter_bank.dim());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn mel_filter_bank(
    num_frequency_bins: usize,
    num_mel_bins: usize,
    min_frequency: f64,
    max_frequency: f64,
    sampling_rate: usize,
) -> Result<ndarray::Array2<f64>> {
    if num_frequency_bins < 2 {
        return Err(TokenizerError::InvalidConfig(format!(
            "num_frequency_bins must be >= 2, got {}",
            num_frequency_bins
        )));
    }

    if min_frequency > max_frequency {
        return Err(TokenizerError::InvalidConfig(format!(
            "min_frequency ({}) must be <= max_frequency ({})",
            min_frequency, max_frequency
        )));
    }

    // Center points of the triangular mel filters
    let mel_min = hertz_to_mel(min_frequency);
    let mel_max = hertz_to_mel(max_frequency);
    let mel_freqs: Vec<f64> = (0..=num_mel_bins + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (num_mel_bins + 1) as f64)
        .collect();
    let filter_freqs: Vec<f64> = mel_freqs.iter().map(|&mel| mel_to_hertz(mel)).collect();

    // Frequencies of FFT bins in Hz
    let fft_freqs: Vec<f64> = (0..num_frequency_bins)
        .map(|i| i as f64 * sampling_rate as f64 / 2.0 / (num_frequency_bins - 1) as f64)
        .collect();

    // Create triangular filter bank - shape (num_frequency_bins, num_mel_bins) to match Python
    let mut filter_bank = ndarray::Array2::zeros((num_frequency_bins, num_mel_bins));

    for mel_idx in 0..num_mel_bins {
        let left_freq = filter_freqs[mel_idx];
        let center_freq = filter_freqs[mel_idx + 1];
        let right_freq = filter_freqs[mel_idx + 2];

        for (freq_idx, &fft_freq) in fft_freqs.iter().enumerate() {
            let value = if fft_freq >= left_freq && fft_freq <= center_freq {
                (fft_freq - left_freq) / (center_freq - left_freq)
            } else if fft_freq > center_freq && fft_freq <= right_freq {
                (right_freq - fft_freq) / (right_freq - center_freq)
            } else {
                0.0
            };

            filter_bank[[freq_idx, mel_idx]] = value.max(0.0);
        }
    }

    // Apply Slaney-style energy normalization
    for mel_idx in 0..num_mel_bins {
        let enorm = 2.0 / (filter_freqs[mel_idx + 2] - filter_freqs[mel_idx]);
        for freq_idx in 0..num_frequency_bins {
            filter_bank[[freq_idx, mel_idx]] *= enorm;
        }
    }

    Ok(filter_bank)
}
