//! Image loading, preprocessing and tokenization.
//!
//! Decoding and pixel preprocessing require the `image` feature (enabled by
//! default). Without it, [`ImageConfig`] is still parsed from tokenizer files
//! and [`ImageEncoder`] still lays out image tokens from image dimensions.
//!
//! This module mirrors `mistral_common.tokens.tokenizers.image`: images are
//! resized so that both sides fit within `max_image_size` while preserving the
//! aspect ratio and snapping to a whole number of patches, normalized with the
//! CLIP dataset statistics, and turned into a grid of `[IMG]` tokens delimited
//! by `[IMG_BREAK]` and `[IMG_END]`.
//!
//! The pixel pipeline reproduces the Python one: transparent pixels are
//! composited onto a white background the way `PIL.Image.paste` does, and the
//! resize is a port of `OpenCV`'s `INTER_CUBIC` (the interpolation
//! `mistral-common` uses), including its border handling. Output matches the
//! Python encoder to within float32 rounding.
//!
//! The one exception is JPEG input. Decoding is done by the `image` crate,
//! whose baseline JPEG decoder differs from Pillow's libjpeg-turbo in its IDCT
//! and chroma upsampling, so decoded samples can differ by a few 8-bit levels.
//! Token layouts are unaffected, since they depend only on the image
//! dimensions. Callers who need a specific decoder can decode themselves and
//! pass the pixels to [`Image::new`].

#[cfg(feature = "image")]
use base64::Engine;
#[cfg(feature = "image")]
use ndarray::Array3;
use serde::{Deserialize, Serialize};
#[cfg(feature = "image")]
use std::path::Path;

use crate::errors::{Result, TokenizerError};

/// Per-channel RGB mean used to normalize images, in `[0, 1]` space.
#[cfg(feature = "image")]
pub const DATASET_MEAN: [f64; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];

/// Per-channel RGB standard deviation used to normalize images, in `[0, 1]` space.
#[cfg(feature = "image")]
pub const DATASET_STD: [f64; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

const fn default_spatial_merge_size() -> usize {
    1
}

/// Configuration for image tokenization.
///
/// Loaded from the `image` section of a `tekken.json` file (or the deprecated
/// `multimodal` section for tokenizers up to v11).
///
/// # Fields
///
/// * `image_patch_size` - Side length in pixels of a single vision patch
/// * `max_image_size` - Maximum side length in pixels after resizing
/// * `spatial_merge_size` - Number of patches merged per token along each axis
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageConfig {
    /// Side length in pixels of a single vision patch.
    pub image_patch_size: usize,
    /// Maximum side length in pixels an image is resized to.
    pub max_image_size: usize,
    /// Number of patches merged into one token along each axis.
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
}

impl ImageConfig {
    /// Creates a new `ImageConfig` with validation.
    ///
    /// # Arguments
    ///
    /// * `image_patch_size` - Side length in pixels of a vision patch (must be > 0)
    /// * `max_image_size` - Maximum side length in pixels after resizing (must be > 0)
    /// * `spatial_merge_size` - Patches merged per token along each axis (must be > 0)
    ///
    /// # Errors
    ///
    /// Returns `TokenizerError::InvalidConfig` if any parameter is zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tekken::image::ImageConfig;
    ///
    /// let config = ImageConfig::new(14, 1540, 2)?;
    /// assert_eq!(config.pixels_per_token(), 28);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        image_patch_size: usize,
        max_image_size: usize,
        spatial_merge_size: usize,
    ) -> Result<Self> {
        if image_patch_size == 0 {
            return Err(TokenizerError::InvalidConfig(
                "image_patch_size must be > 0".to_string(),
            ));
        }
        if max_image_size == 0 {
            return Err(TokenizerError::InvalidConfig(
                "max_image_size must be > 0".to_string(),
            ));
        }
        if spatial_merge_size == 0 {
            return Err(TokenizerError::InvalidConfig(
                "spatial_merge_size must be > 0".to_string(),
            ));
        }

        Ok(Self {
            image_patch_size,
            max_image_size,
            spatial_merge_size,
        })
    }

    /// Returns the number of pixels along each axis covered by a single token.
    ///
    /// This is `image_patch_size * spatial_merge_size`.
    #[must_use]
    pub const fn pixels_per_token(&self) -> usize {
        self.image_patch_size * self.spatial_merge_size
    }

    /// Validates the invariants of a configuration loaded from a file.
    ///
    /// # Errors
    ///
    /// Returns `TokenizerError::InvalidConfig` if any field is zero.
    pub fn validate(&self) -> Result<()> {
        Self::new(
            self.image_patch_size,
            self.max_image_size,
            self.spatial_merge_size,
        )
        .map(|_| ())
    }
}

/// Token IDs used to delimit an image in a token sequence.
///
/// # Fields
///
/// * `img` - The `[IMG]` token, one per token-sized patch block
/// * `img_break` - The `[IMG_BREAK]` token, ending every row but the last
/// * `img_end` - The `[IMG_END]` token, ending the final row
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialImageIds {
    /// Token ID emitted for each token-sized block of the image.
    pub img: u32,
    /// Token ID ending each image row except the last.
    pub img_break: u32,
    /// Token ID ending the last image row.
    pub img_end: u32,
}

/// An RGB image.
///
/// Pixels are stored row-major as interleaved 8-bit RGB triplets, matching a
/// PIL image in `"RGB"` mode. Images decoded from a file or from bytes have any
/// transparency composited onto a white background, as `mistral-common` does.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(feature = "image")]
pub struct Image {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

#[cfg(feature = "image")]
impl Image {
    /// Creates an image from raw row-major RGB bytes.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width in pixels (must be > 0)
    /// * `height` - Image height in pixels (must be > 0)
    /// * `pixels` - `width * height * 3` bytes of interleaved RGB data
    ///
    /// # Errors
    ///
    /// Returns `TokenizerError::Image` if a dimension is zero or the buffer
    /// length does not match the dimensions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tekken::image::Image;
    ///
    /// let image = Image::new(2, 1, vec![255, 0, 0, 0, 255, 0])?;
    /// assert_eq!((image.width(), image.height()), (2, 1));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(TokenizerError::Image(format!(
                "Image dimensions must be > 0, got {width}x{height}"
            )));
        }

        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| {
                TokenizerError::Image(format!("Image dimensions overflow: {width}x{height}"))
            })?;
        if pixels.len() != expected {
            return Err(TokenizerError::Image(format!(
                "Expected {expected} bytes for a {width}x{height} RGB image, got {}",
                pixels.len()
            )));
        }

        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Loads an image from a file, decoding it by its content.
    ///
    /// PNG, WebP, BMP and TIFF decode byte-for-byte identically to Pillow;
    /// JPEG can differ by a few 8-bit levels (see the [module docs](self)).
    ///
    /// # Arguments
    ///
    /// * `path` - Path to an image file (PNG, JPEG, GIF, WebP, BMP or TIFF)
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or decoded.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use tekken::image::Image;
    ///
    /// let image = Image::from_file("picture.png")?;
    /// println!("{}x{}", image.width(), image.height());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Decodes an image from raw encoded bytes.
    ///
    /// Transparency is composited onto a white background. PNG, WebP, BMP and
    /// TIFF decode byte-for-byte identically to Pillow; JPEG can differ by a
    /// few 8-bit levels (see the [module docs](self)).
    ///
    /// # Arguments
    ///
    /// * `bytes` - Encoded image data (PNG, JPEG, GIF, WebP, BMP or TIFF)
    ///
    /// # Errors
    ///
    /// Returns `TokenizerError::Image` if the bytes cannot be decoded.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let decoded = ::image::load_from_memory(bytes)
            .map_err(|e| TokenizerError::Image(format!("Failed to decode image: {e}")))?;
        Self::from_dynamic_image(&decoded)
    }

    /// Decodes an image from a base64 string.
    ///
    /// A `data:image/...;base64,` prefix, as used in `image_url` chunks, is
    /// accepted and stripped.
    ///
    /// # Arguments
    ///
    /// * `data` - Base64-encoded image data, with or without a data URL prefix
    ///
    /// # Errors
    ///
    /// Returns an error if the base64 is malformed or the image cannot be decoded.
    pub fn from_base64(data: &str) -> Result<Self> {
        let payload = data
            .strip_prefix("data:")
            .and_then(|rest| rest.split_once(";base64,"))
            .map_or(data, |(_, encoded)| encoded);
        let bytes = base64::engine::general_purpose::STANDARD.decode(payload.trim())?;
        Self::from_bytes(&bytes)
    }

    /// Converts a decoded image to RGB, compositing transparency onto white.
    fn from_dynamic_image(decoded: &::image::DynamicImage) -> Result<Self> {
        use ::image::GenericImageView;

        let (width, height) = decoded.dimensions();
        let pixels = if decoded.color().has_alpha() {
            let rgba = decoded.to_rgba8();
            let mut pixels = Vec::with_capacity(rgba.len() / 4 * 3);
            for pixel in rgba.pixels() {
                let [red, green, blue, alpha] = pixel.0;
                pixels.push(blend_onto_white(red, alpha));
                pixels.push(blend_onto_white(green, alpha));
                pixels.push(blend_onto_white(blue, alpha));
            }
            pixels
        } else {
            decoded.to_rgb8().into_raw()
        };

        Self::new(width as usize, height as usize, pixels)
    }

    /// Returns the image width in pixels.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns the image height in pixels.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Returns the raw row-major interleaved RGB pixel data.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

/// Composites one 8-bit channel value over a white background.
///
/// Reproduces the fixed-point blend `PIL.Image.paste` performs when pasting an
/// RGBA image onto a white RGBA background using its own alpha as the mask.
#[cfg(feature = "image")]
const fn blend_onto_white(channel: u8, alpha: u8) -> u8 {
    muldiv255(255, 255 - alpha) + muldiv255(channel, alpha)
}

/// Computes `round(a * b / 255)` the way PIL's `MULDIV255` macro does.
#[allow(clippy::cast_possible_truncation)]
#[cfg(feature = "image")]
const fn muldiv255(a: u8, b: u8) -> u8 {
    let tmp = a as u32 * b as u32 + 128;
    // The result of rounding a product of two bytes down by 255 is a byte.
    (((tmp >> 8) + tmp) >> 8) as u8
}

/// A tokenized image.
///
/// # Fields
///
/// * `tokens` - The token sequence representing the image
/// * `image` - The preprocessed pixel data with shape `(3, height, width)`
#[derive(Debug, Clone)]
#[cfg(feature = "image")]
pub struct ImageEncoding {
    /// Token sequence: rows of `[IMG]` tokens delimited by `[IMG_BREAK]`/`[IMG_END]`.
    pub tokens: Vec<u32>,
    /// Normalized image with shape `(3, height, width)` in channel-first order.
    pub image: Array3<f32>,
}

/// Converts images into token sequences and normalized pixel arrays.
///
/// # Fields
///
/// * `config` - Image tokenization parameters
/// * `special_ids` - The `[IMG]`, `[IMG_BREAK]` and `[IMG_END]` token IDs
#[derive(Debug, Clone)]
pub struct ImageEncoder {
    /// Image tokenization parameters.
    pub config: ImageConfig,
    /// Token IDs used to delimit the image.
    pub special_ids: SpecialImageIds,
}

impl ImageEncoder {
    /// Creates a new `ImageEncoder`.
    ///
    /// # Arguments
    ///
    /// * `config` - Image tokenization parameters
    /// * `special_ids` - The `[IMG]`, `[IMG_BREAK]` and `[IMG_END]` token IDs
    #[must_use]
    pub const fn new(config: ImageConfig, special_ids: SpecialImageIds) -> Self {
        Self {
            config,
            special_ids,
        }
    }

    /// Returns the `[IMG]` token ID.
    #[must_use]
    pub const fn image_token(&self) -> u32 {
        self.special_ids.img
    }

    /// Computes the token grid size for an image of the given dimensions.
    ///
    /// The image is first scaled down so that neither side exceeds
    /// `max_image_size`, then each axis is covered by whole tokens.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width in pixels (must be > 0)
    /// * `height` - Image height in pixels (must be > 0)
    ///
    /// # Returns
    ///
    /// The number of tokens along the width and the height, in that order.
    ///
    /// # Errors
    ///
    /// Returns `TokenizerError::Image` if a dimension is zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tekken::image::{ImageConfig, ImageEncoder, SpecialImageIds};
    ///
    /// let encoder = ImageEncoder::new(
    ///     ImageConfig::new(16, 128, 1)?,
    ///     SpecialImageIds { img: 10, img_break: 12, img_end: 13 },
    /// );
    /// assert_eq!(encoder.image_to_num_tokens(128, 64)?, (8, 4));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    pub fn image_to_num_tokens(&self, width: usize, height: usize) -> Result<(usize, usize)> {
        let (width, height) = self.resized_dimensions(width, height)?;
        let pixels_per_token = self.config.pixels_per_token();

        Ok((
            (width - 1) / pixels_per_token + 1,
            (height - 1) / pixels_per_token + 1,
        ))
    }

    /// Scales the dimensions down so neither side exceeds `max_image_size`.
    ///
    /// Mirrors the aspect-ratio-preserving downscale of `_image_to_num_tokens`,
    /// including Python's round-half-to-even rounding.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    fn resized_dimensions(&self, width: usize, height: usize) -> Result<(usize, usize)> {
        if width == 0 || height == 0 {
            return Err(TokenizerError::Image(format!(
                "Image dimensions must be > 0, got {width}x{height}"
            )));
        }

        let max_size = self.config.max_image_size as f64;
        let ratio = f64::max(height as f64 / max_size, width as f64 / max_size);
        if ratio <= 1.0 {
            return Ok((width, height));
        }

        // Python's round() breaks ties to even; f64::round_ties_even matches it.
        let width = (width as f64 / ratio).round_ties_even() as usize;
        let height = (height as f64 / ratio).round_ties_even() as usize;
        if width == 0 || height == 0 {
            return Err(TokenizerError::Image(
                "Image is too elongated to be resized to a whole number of pixels".to_string(),
            ));
        }

        Ok((width, height))
    }

    /// Builds the token sequence for an image of the given dimensions.
    ///
    /// This performs no pixel processing, so it can be used to count or lay out
    /// tokens without decoding the image itself.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Returns
    ///
    /// `height_tokens` rows of `width_tokens` `[IMG]` tokens, each row ending in
    /// `[IMG_BREAK]` except the last, which ends in `[IMG_END]`.
    ///
    /// # Errors
    ///
    /// Returns `TokenizerError::Image` if a dimension is zero.
    pub fn encode_dimensions(&self, width: usize, height: usize) -> Result<Vec<u32>> {
        let (width_tokens, height_tokens) = self.image_to_num_tokens(width, height)?;

        let mut tokens = Vec::with_capacity((width_tokens + 1) * height_tokens);
        for _ in 0..height_tokens {
            tokens.extend(std::iter::repeat_n(self.special_ids.img, width_tokens));
            tokens.push(self.special_ids.img_break);
        }
        // The image ends rather than breaks after its last row.
        if let Some(last) = tokens.last_mut() {
            *last = self.special_ids.img_end;
        }

        Ok(tokens)
    }

    /// Encodes an image into tokens and normalized pixels.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to encode
    ///
    /// # Returns
    ///
    /// An [`ImageEncoding`] holding the token sequence and the preprocessed
    /// image with shape `(3, height, width)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the image has a zero dimension.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tekken::image::{Image, ImageConfig, ImageEncoder, SpecialImageIds};
    ///
    /// let encoder = ImageEncoder::new(
    ///     ImageConfig::new(16, 128, 1)?,
    ///     SpecialImageIds { img: 10, img_break: 12, img_end: 13 },
    /// );
    /// let image = Image::new(32, 16, vec![128; 32 * 16 * 3])?;
    ///
    /// let encoding = encoder.encode(&image)?;
    /// assert_eq!(encoding.tokens.len(), (2 + 1) * 1);
    /// assert_eq!(encoding.image.dim(), (3, 16, 32));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg(feature = "image")]
    pub fn encode(&self, image: &Image) -> Result<ImageEncoding> {
        let (width_tokens, height_tokens) =
            self.image_to_num_tokens(image.width(), image.height())?;
        let tokens = self.encode_dimensions(image.width(), image.height())?;

        let pixels_per_token = self.config.pixels_per_token();
        let processed = transform_image(
            image,
            width_tokens * pixels_per_token,
            height_tokens * pixels_per_token,
        )?;

        Ok(ImageEncoding {
            tokens,
            image: processed,
        })
    }
}

/// Resizes an image and normalizes it with the CLIP dataset statistics.
///
/// # Arguments
///
/// * `image` - The source image
/// * `new_width` - Target width in pixels (must be > 0)
/// * `new_height` - Target height in pixels (must be > 0)
///
/// # Returns
///
/// An array of shape `(3, new_height, new_width)` holding the normalized image
/// in channel-first order.
///
/// # Errors
///
/// Returns `TokenizerError::Image` if a target dimension is zero.
///
/// # Examples
///
/// ```rust
/// use tekken::image::{Image, transform_image};
///
/// let image = Image::new(4, 4, vec![255; 4 * 4 * 3])?;
/// let transformed = transform_image(&image, 2, 2)?;
/// assert_eq!(transformed.dim(), (3, 2, 2));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "image")]
pub fn transform_image(image: &Image, new_width: usize, new_height: usize) -> Result<Array3<f32>> {
    if new_width == 0 || new_height == 0 {
        return Err(TokenizerError::Image(format!(
            "Target dimensions must be > 0, got {new_width}x{new_height}"
        )));
    }

    // Sampled straight out of the 8-bit buffer, so no float copy of the whole
    // source image is ever materialized.
    let pixels = image.pixels();
    let stride = image.width() * 3;
    let resized = resize_bicubic_with(
        |row, index| f32::from(pixels[row * stride + index]),
        image.width(),
        image.height(),
        new_width,
        new_height,
        3,
    );

    Ok(normalize(
        &resized,
        new_width,
        new_height,
        DATASET_MEAN,
        DATASET_STD,
    ))
}

/// Normalizes interleaved RGB samples and converts them to channel-first order.
///
/// Samples are scaled from `[0, 255]` to `[0, 1]`, then standardized per channel.
///
/// The scaling is done in `f32` and the standardization in `f64` before the
/// result is rounded back to `f32`, which is the precision `NumPy` uses for the
/// same expression in `mistral-common` (a `f32` array divided by a Python float
/// stays `f32`, but subtracting a tuple of floats promotes it to `f64`).
///
/// # Arguments
///
/// * `samples` - `width * height * 3` interleaved RGB samples in `[0, 255]`
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `mean` - Per-channel mean in `[0, 1]` space
/// * `std` - Per-channel standard deviation in `[0, 1]` space
///
/// # Returns
///
/// An array of shape `(3, height, width)`.
///
/// # Panics
///
/// Panics if `samples` does not hold exactly `width * height * 3` values.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
#[cfg(feature = "image")]
pub fn normalize(
    samples: &[f32],
    width: usize,
    height: usize,
    mean: [f64; 3],
    std: [f64; 3],
) -> Array3<f32> {
    assert_eq!(
        samples.len(),
        width * height * 3,
        "expected {} samples for a {width}x{height} RGB image, got {}",
        width * height * 3,
        samples.len()
    );

    Array3::from_shape_fn((3, height, width), |(channel, y, x)| {
        let scaled = samples[(y * width + x) * 3 + channel] / 255.0;
        ((f64::from(scaled) - mean[channel]) / std[channel]) as f32
    })
}

/// Offsets of the 4-tap cubic support relative to the mapped source pixel.
#[cfg(feature = "image")]
const TAPS: [isize; 4] = [-1, 0, 1, 2];

/// The source positions and kernel weights feeding one destination position.
#[cfg(feature = "image")]
struct CubicTaps {
    positions: [usize; 4],
    weights: [f32; 4],
}

/// Precomputes the cubic taps mapping a source axis onto a destination axis.
///
/// Positions use `OpenCV`'s half-pixel mapping and are clamped to the source
/// range, which replicates the edge sample exactly as `OpenCV` does.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#[cfg(feature = "image")]
fn cubic_taps(src_len: usize, dst_len: usize) -> Vec<CubicTaps> {
    let scale = src_len as f64 / dst_len as f64;

    (0..dst_len)
        .map(|index| {
            let mapped = ((index as f64 + 0.5) * scale - 0.5) as f32;
            let base = mapped.floor();
            CubicTaps {
                // `base` is integral, so the cast is exact.
                positions: TAPS.map(|tap| clamp_index(base as isize + tap, src_len)),
                weights: cubic_weights(mapped - base),
            }
        })
        .collect()
}

/// Resizes interleaved multi-channel samples using bicubic interpolation.
///
/// This is a port of `OpenCV`'s `cv2.resize(..., interpolation=cv2.INTER_CUBIC)`
/// for 32-bit float input, which is what `mistral-common` uses: the same
/// half-pixel coordinate mapping, the same `a = -0.75` cubic kernel, the same
/// horizontal-then-vertical pass order, and the same replicated borders.
///
/// # Arguments
///
/// * `source` - `src_width * src_height * channels` interleaved samples
/// * `src_width` - Source width in pixels
/// * `src_height` - Source height in pixels
/// * `dst_width` - Target width in pixels
/// * `dst_height` - Target height in pixels
/// * `channels` - Number of interleaved channels
///
/// # Returns
///
/// `dst_width * dst_height * channels` interleaved samples.
///
/// # Panics
///
/// Panics if `source` does not hold exactly `src_width * src_height * channels`
/// values, or if any dimension is zero.
#[must_use]
#[cfg(feature = "image")]
pub fn resize_bicubic(
    source: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    channels: usize,
) -> Vec<f32> {
    assert_eq!(
        source.len(),
        src_width * src_height * channels,
        "source length does not match the given dimensions"
    );

    let stride = src_width * channels;
    resize_bicubic_with(
        |row, index| source[row * stride + index],
        src_width,
        src_height,
        dst_width,
        dst_height,
        channels,
    )
}

/// Resizes samples read through `sample`, called as `sample(row, index)` with
/// `index` an offset within that row's `src_width * channels` samples.
///
/// Only the four horizontally resampled rows the current output row needs are
/// held at once, so memory does not grow with the source height.
///
/// # Panics
///
/// Panics if any dimension is zero.
#[cfg(feature = "image")]
fn resize_bicubic_with<F>(
    sample: F,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    channels: usize,
) -> Vec<f32>
where
    F: Fn(usize, usize) -> f32,
{
    assert!(
        src_width > 0 && src_height > 0 && dst_width > 0 && dst_height > 0 && channels > 0,
        "resize dimensions must be > 0"
    );

    let columns = cubic_taps(src_width, dst_width);
    let rows = cubic_taps(src_height, dst_height);

    let row_len = dst_width * channels;
    // A window of horizontally resampled source rows, keyed by source row
    // index. Consecutive output rows usually share most of their four taps.
    let mut window: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0; row_len]);
    let mut window_rows = [usize::MAX; 4];

    let mut resized = vec![0.0f32; dst_height * row_len];
    for (y, row_taps) in rows.iter().enumerate() {
        let mut slots = [usize::MAX; 4];
        let mut taken = [false; 4];

        // Claim the slots already holding a row this output row needs, so the
        // pass below cannot overwrite one before it is read.
        for (tap, &row) in row_taps.positions.iter().enumerate() {
            if let Some(slot) = window_rows.iter().position(|&cached| cached == row) {
                slots[tap] = slot;
                taken[slot] = true;
            }
        }
        for (tap, &row) in row_taps.positions.iter().enumerate() {
            if slots[tap] != usize::MAX {
                continue;
            }
            // A row can repeat across taps at the borders; resample it once.
            if let Some(slot) = window_rows.iter().position(|&cached| cached == row) {
                slots[tap] = slot;
                continue;
            }
            let slot = taken
                .iter()
                .position(|&used| !used)
                .expect("four taps never need more than four rows");
            taken[slot] = true;
            window_rows[slot] = row;
            resample_row(&sample, row, &columns, channels, &mut window[slot]);
            slots[tap] = slot;
        }

        let resized_row = &mut resized[y * row_len..(y + 1) * row_len];
        for (slot, weight) in slots.iter().zip(&row_taps.weights) {
            for (destination, source) in resized_row.iter_mut().zip(&window[*slot]) {
                *destination += source * weight;
            }
        }
    }

    resized
}

/// Resamples one source row onto the destination width.
#[cfg(feature = "image")]
fn resample_row<F>(
    sample: &F,
    row: usize,
    columns: &[CubicTaps],
    channels: usize,
    resampled: &mut [f32],
) where
    F: Fn(usize, usize) -> f32,
{
    for (x, taps) in columns.iter().enumerate() {
        for channel in 0..channels {
            let mut value = 0.0f32;
            for (column, weight) in taps.positions.iter().zip(&taps.weights) {
                value += sample(row, column * channels + channel) * weight;
            }
            resampled[x * channels + channel] = value;
        }
    }
}

/// Clamps a source index to the valid range, replicating the edge sample.
#[allow(clippy::cast_sign_loss)]
#[cfg(feature = "image")]
const fn clamp_index(index: isize, len: usize) -> usize {
    if index < 0 {
        0
    // Negative indices are handled above, so the cast keeps the value.
    } else if index as usize >= len {
        len - 1
    } else {
        index as usize
    }
}

/// Computes `OpenCV`'s four cubic interpolation weights for a fractional offset.
#[cfg(feature = "image")]
fn cubic_weights(fraction: f32) -> [f32; 4] {
    const A: f32 = -0.75;

    let x = fraction;
    let w0 = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
    let w1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let w2 = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    let w3 = 1.0 - w0 - w1 - w2;

    [w0, w1, w2, w3]
}
