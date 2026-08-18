//! Image tokenization tests.
//!
//! The expected values were produced by running `mistral-common` 1.11.7's
//! `ImageEncoder` on the same inputs, and by the `OpenCV` / Pillow primitives it
//! builds on. Token layouts and grid sizes are exact; pixel values are compared
//! with a relative tolerance, because `OpenCV`'s own `INTER_CUBIC` output varies
//! in the last float bits between its scalar, SIMD and IPP code paths.
#![cfg(feature = "image")]

mod common;

use approx::assert_relative_eq;
use serde_json::json;
use tekken::config::TokenizerVersion;
use tekken::errors::TokenizerError;
use tekken::image::{Image, ImageConfig, ImageEncoder, SpecialImageIds, resize_bicubic};
use tekken::tekkenizer::Tekkenizer;

const IMG: u32 = 10;
const IMG_BREAK: u32 = 12;
const IMG_END: u32 = 13;

const SPECIAL_IDS: SpecialImageIds = SpecialImageIds {
    img: IMG,
    img_break: IMG_BREAK,
    img_end: IMG_END,
};

fn encoder(
    image_patch_size: usize,
    max_image_size: usize,
    spatial_merge_size: usize,
) -> ImageEncoder {
    ImageEncoder::new(
        ImageConfig::new(image_patch_size, max_image_size, spatial_merge_size).unwrap(),
        SPECIAL_IDS,
    )
}

/// Deterministic pseudo-random bytes, matching the generator used to produce
/// the reference values with `mistral-common`.
fn lcg_bytes(count: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(count);
    let mut state: u64 = 12345;
    for _ in 0..count {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        bytes.push(((state >> 16) & 0xFF) as u8);
    }
    bytes
}

fn noise_image(width: usize, height: usize) -> Image {
    Image::new(width, height, lcg_bytes(width * height * 3)).unwrap()
}

fn solid_image(width: usize, height: usize, color: [u8; 3]) -> Image {
    let pixels = color
        .iter()
        .copied()
        .cycle()
        .take(width * height * 3)
        .collect();
    Image::new(width, height, pixels).unwrap()
}

#[test]
fn test_image_to_num_tokens_matches_mistral_common() {
    for spatial_merge_size in [1, 2] {
        let encoder = encoder(16 / spatial_merge_size, 128, spatial_merge_size);

        for (size, expected) in [(4, 1), (16, 1), (128, 8), (512, 8), (2048, 8)] {
            assert_eq!(
                encoder.image_to_num_tokens(size, size).unwrap(),
                (expected, expected),
                "square {size}px at merge {spatial_merge_size}"
            );
        }

        for (width, height, expected_width, expected_height) in [
            (4, 2, 1, 1),
            (8, 16, 1, 1),
            (128, 64, 8, 4),
            (512, 1024, 4, 8),
        ] {
            assert_eq!(
                encoder.image_to_num_tokens(width, height).unwrap(),
                (expected_width, expected_height),
                "{width}x{height} at merge {spatial_merge_size}"
            );
        }
    }
}

#[test]
fn test_image_to_num_tokens_rounds_ties_to_even() {
    // Downscaling by exactly 2 leaves a half pixel, which Python's round()
    // breaks towards the even value: 2.5 -> 2 but 3.5 -> 4.
    let encoder = encoder(1, 4, 1);
    assert_eq!(encoder.image_to_num_tokens(5, 8).unwrap(), (2, 4));
    assert_eq!(encoder.image_to_num_tokens(7, 8).unwrap(), (4, 4));
}

#[test]
fn test_image_to_num_tokens_rejects_empty_images() {
    let encoder = encoder(16, 1024, 1);
    assert!(matches!(
        encoder.image_to_num_tokens(0, 10),
        Err(TokenizerError::Image(_))
    ));
    assert!(matches!(
        encoder.image_to_num_tokens(10, 0),
        Err(TokenizerError::Image(_))
    ));
}

#[test]
fn test_token_layout() {
    let encoder = encoder(16, 1024, 1);

    // 3 tokens wide, 2 tokens tall: every row ends in a break, except the last.
    let tokens = encoder.encode_dimensions(40, 20).unwrap();
    assert_eq!(
        tokens,
        vec![
            IMG, IMG, IMG, IMG_BREAK, //
            IMG, IMG, IMG, IMG_END,
        ]
    );

    // A single-token image is just [IMG][IMG_END].
    assert_eq!(encoder.encode_dimensions(1, 1).unwrap(), vec![IMG, IMG_END]);
}

#[test]
fn test_token_layout_matches_encode() {
    let encoder = encoder(16, 128, 2);
    let image = noise_image(129, 96);

    let encoding = encoder.encode(&image).unwrap();
    let (width_tokens, height_tokens) = encoder.image_to_num_tokens(129, 96).unwrap();

    assert_eq!(
        encoding.tokens,
        encoder.encode_dimensions(129, 96).unwrap(),
        "encode and encode_dimensions must agree"
    );
    assert_eq!(encoding.tokens.len(), (width_tokens + 1) * height_tokens);
    assert_eq!(
        encoding.tokens.iter().filter(|&&t| t == IMG).count(),
        width_tokens * height_tokens
    );
    assert_eq!(
        encoding.tokens.iter().filter(|&&t| t == IMG_BREAK).count(),
        height_tokens - 1
    );
    assert_eq!(encoding.tokens.last(), Some(&IMG_END));
}

/// width, height, patch size, max size, merge size, width tokens, height
/// tokens, and the sum of the absolute values of the processed pixels.
type EncodeCase = (usize, usize, usize, usize, usize, usize, usize, f64);

#[test]
fn test_encode_matches_mistral_common() {
    let cases: [EncodeCase; 5] = [
        (37, 23, 16, 1024, 1, 3, 2, 3_642.175_220),
        (200, 311, 16, 1024, 1, 13, 20, 157_487.636_499),
        (300, 212, 8, 1024, 2, 19, 14, 161_122.434_562),
        (64, 64, 14, 1540, 2, 3, 3, 16_702.438_935),
        (2048, 3, 16, 1024, 1, 64, 1, 35_680.966_946),
    ];

    for (width, height, patch, max_size, merge, width_tokens, height_tokens, abs_sum) in cases {
        let encoder = encoder(patch, max_size, merge);
        let encoding = encoder.encode(&noise_image(width, height)).unwrap();

        assert_eq!(
            encoder.image_to_num_tokens(width, height).unwrap(),
            (width_tokens, height_tokens),
            "{width}x{height} token grid"
        );
        assert_eq!(
            encoding.image.dim(),
            (
                3,
                height_tokens * patch * merge,
                width_tokens * patch * merge
            ),
            "{width}x{height} processed shape"
        );
        assert_eq!(encoding.tokens.len(), (width_tokens + 1) * height_tokens);

        let actual: f64 = encoding.image.iter().map(|v| f64::from(v.abs())).sum();
        assert_relative_eq!(actual, abs_sum, max_relative = 1e-5);
    }
}

#[test]
fn test_normalization_of_a_solid_image() {
    // 32x16 maps to exactly 2x1 tokens of 16 pixels, so the resize is a no-op
    // and every pixel is just the normalized source color.
    let encoder = encoder(16, 1024, 1);
    let encoding = encoder
        .encode(&solid_image(32, 16, [128, 64, 192]))
        .unwrap();

    assert_eq!(encoding.image.dim(), (3, 16, 32));
    for (channel, expected) in [0.076_336_18_f32, -0.791_599_87, 1.250_032_9]
        .into_iter()
        .enumerate()
    {
        for value in encoding.image.index_axis(ndarray::Axis(0), channel) {
            assert_relative_eq!(*value, expected, max_relative = 1e-6);
        }
    }
}

#[test]
fn test_resize_bicubic_matches_opencv() {
    // Upsampling: the cubic kernel overshoots past the [0, 255] input range and
    // the edge taps come from a replicated border, exactly as in OpenCV.
    let source = [0.0_f32, 255.0, 0.0, 255.0, 0.0, 255.0];
    let expected = [
        -25.953_82_f32,
        89.308_68,
        245.139_3,
        161.889_5,
        1.532_564,
        127.5,
        253.467_3,
        93.110_47,
        9.860_64,
        165.691_3,
        280.953_9,
    ];
    let resized = resize_bicubic(&source, 6, 1, 11, 1, 1);
    for (actual, expected) in resized.iter().zip(expected) {
        assert_relative_eq!(*actual, expected, max_relative = 1e-6);
    }

    // Downsampling uses the same kernel (no area averaging).
    let resized = resize_bicubic(&source, 6, 1, 3, 1, 1);
    for (actual, expected) in resized.iter().zip([151.406_2_f32, 127.5, 103.593_8]) {
        assert_relative_eq!(*actual, expected, max_relative = 1e-6);
    }

    // Both axes at once, on a non-square target.
    let source = [0.0_f32, 255.0, 0.0, 255.0, 0.0, 255.0, 0.0, 255.0, 0.0];
    let expected = [
        -46.297_53_f32,
        115.835_7,
        273.303_2,
        115.835_7,
        -46.297_43,
        203.341_6,
        132.59,
        63.874_51,
        132.59,
        203.341_6,
        203.341_6,
        132.59,
        63.874_51,
        132.59,
        203.341_6,
        -46.297_53,
        115.835_7,
        273.303_2,
        115.835_7,
        -46.297_43,
    ];
    let resized = resize_bicubic(&source, 3, 3, 5, 4, 1);
    for (actual, expected) in resized.iter().zip(expected) {
        assert_relative_eq!(*actual, expected, max_relative = 1e-6);
    }
}

/// A 3x2 RGB PNG with known pixel values.
const RGB_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAIAAAASFvFNAAAAGElEQVR4nGNkZGKGAIZfv//8/fefoeE/ACnqB4QqblbnAAAAAElFTkSuQmCC";

/// A 4x3 RGBA PNG whose pixels cover the full alpha range.
const RGBA_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAYAAAC09K7GAAAAN0lEQVR4nAXBoRHAIBQFwcNGo98MLWBiUwyF/MrQlEEnl10EkQJRvvb0nCRvkpHkctYUUG1VtX+0KxKeaQerSAAAAABJRU5ErkJggg==";

#[test]
fn test_decode_rgb_png() {
    let image = Image::from_base64(RGB_PNG).unwrap();

    assert_eq!((image.width(), image.height()), (3, 2));
    assert_eq!(
        image.pixels(),
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 250, 251, 252, 253, 254, 255, 0, 128, 255
        ]
    );
}

#[test]
fn test_decode_rgba_png_composites_onto_white() {
    let image = Image::from_base64(RGBA_PNG).unwrap();

    assert_eq!((image.width(), image.height()), (4, 3));
    // Reference values from Pillow: fully opaque pixels are untouched, fully
    // transparent ones become white, and the rest are blended towards white.
    assert_eq!(
        image.pixels(),
        &[
            255, 0, 0, 127, 255, 127, 255, 255, 255, 255, 255, 191, //
            63, 71, 79, 40, 50, 60, 232, 233, 234, 255, 255, 255, //
            200, 100, 50, 0, 0, 0, 255, 255, 255, 160, 160, 160,
        ]
    );
}

#[test]
fn test_decode_accepts_data_urls_and_files() {
    let expected = Image::from_base64(RGB_PNG).unwrap();

    let data_url = format!("data:image/png;base64,{RGB_PNG}");
    assert_eq!(Image::from_base64(&data_url).unwrap(), expected);

    let bytes = base64_decode(RGB_PNG);
    assert_eq!(Image::from_bytes(&bytes).unwrap(), expected);

    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), &bytes).unwrap();
    assert_eq!(Image::from_file(file.path()).unwrap(), expected);
}

#[test]
fn test_decode_rejects_garbage() {
    assert!(matches!(
        Image::from_bytes(b"not an image"),
        Err(TokenizerError::Image(_))
    ));
    assert!(Image::from_base64("!!! not base64 !!!").is_err());
}

#[test]
fn test_image_rejects_inconsistent_buffers() {
    assert!(matches!(
        Image::new(0, 4, vec![]),
        Err(TokenizerError::Image(_))
    ));
    assert!(matches!(
        Image::new(2, 2, vec![0; 11]),
        Err(TokenizerError::Image(_))
    ));
    assert!(Image::new(2, 2, vec![0; 12]).is_ok());
}

#[test]
fn test_image_config_rejects_zero_fields() {
    assert!(ImageConfig::new(0, 1024, 1).is_err());
    assert!(ImageConfig::new(16, 0, 1).is_err());
    assert!(ImageConfig::new(16, 1024, 0).is_err());
}

#[test]
fn test_image_config_defaults_spatial_merge_size() {
    let config: ImageConfig =
        serde_json::from_str(r#"{"image_patch_size": 16, "max_image_size": 1024}"#).unwrap();
    assert_eq!(config.spatial_merge_size, 1);
    assert_eq!(config.pixels_per_token(), 16);
}

// ---------------------------------------------------------------------------
// Tokenizer integration
// ---------------------------------------------------------------------------

fn image_special_tokens_json() -> Vec<serde_json::Value> {
    ["<unk>", "<s>", "</s>", "[IMG]", "[IMG_BREAK]", "[IMG_END]"]
        .iter()
        .enumerate()
        .map(|(rank, token_str)| json!({"rank": rank, "token_str": token_str, "is_control": true}))
        .collect()
}

fn tokenizer_json(
    version: &str,
    image_key: Option<(&str, serde_json::Value)>,
) -> serde_json::Value {
    let num_special_tokens = 10;
    let mut doc = json!({
        "vocab": serde_json::to_value(common::small_vocab()).unwrap(),
        "special_tokens": image_special_tokens_json(),
        "config": {
            "pattern": common::PATTERN,
            "num_vocab_tokens": 258,
            "default_vocab_size": 258 + num_special_tokens,
            "default_num_special_tokens": num_special_tokens,
            "version": version,
        },
    });
    if let Some((key, value)) = image_key {
        doc[key] = value;
    }
    doc
}

fn image_config_json() -> serde_json::Value {
    json!({"image_patch_size": 14, "max_image_size": 1540, "spatial_merge_size": 2})
}

fn write_tokenizer_file(doc: &serde_json::Value) -> tempfile::NamedTempFile {
    let file = tempfile::NamedTempFile::new().expect("Failed to create temp file");
    std::fs::write(file.path(), serde_json::to_string(doc).unwrap())
        .expect("Failed to write tokenizer file");
    file
}

#[test]
fn test_tokenizer_loads_image_config() {
    for version in ["v11", "v13", "v15"] {
        let file = write_tokenizer_file(&tokenizer_json(
            version,
            Some(("image", image_config_json())),
        ));
        let tokenizer = Tekkenizer::from_file(file.path()).unwrap();

        assert!(tokenizer.has_image_support(), "{version}");
        assert_eq!(
            tokenizer.image_config(),
            Some(&ImageConfig::new(14, 1540, 2).unwrap()),
            "{version}"
        );
    }
}

#[test]
fn test_tokenizer_accepts_deprecated_multimodal_key_up_to_v11() {
    for version in ["v3", "v7", "v11"] {
        let file = write_tokenizer_file(&tokenizer_json(
            version,
            Some(("multimodal", image_config_json())),
        ));
        let tokenizer = Tekkenizer::from_file(file.path()).unwrap();

        assert!(tokenizer.has_image_support(), "{version}");
        assert_eq!(
            tokenizer.image_config(),
            Some(&ImageConfig::new(14, 1540, 2).unwrap()),
            "{version}"
        );
    }
}

#[test]
fn test_tokenizer_rejects_deprecated_multimodal_key_after_v11() {
    for version in ["v13", "v15"] {
        let file = write_tokenizer_file(&tokenizer_json(
            version,
            Some(("multimodal", image_config_json())),
        ));
        let error = Tekkenizer::from_file(file.path())
            .err()
            .expect("multimodal key should be rejected");

        assert!(
            matches!(&error, TokenizerError::InvalidConfig(message) if message.contains("'image'")),
            "{version}: unexpected error {error}"
        );
    }
    assert!(TokenizerVersion::V11.allows_deprecated_multimodal_key());
    assert!(!TokenizerVersion::V13.allows_deprecated_multimodal_key());
}

#[test]
fn test_tokenizer_without_image_config_has_no_image_support() {
    let file = write_tokenizer_file(&tokenizer_json("v13", None));
    let tokenizer = Tekkenizer::from_file(file.path()).unwrap();

    assert!(!tokenizer.has_image_support());
    assert_eq!(tokenizer.image_config(), None);
    assert!(matches!(
        tokenizer.encode_image(&solid_image(16, 16, [0, 0, 0])),
        Err(TokenizerError::Image(_))
    ));
}

#[test]
fn test_tokenizer_requires_image_control_tokens() {
    let mut doc = tokenizer_json("v13", Some(("image", image_config_json())));
    doc["special_tokens"] = json!(
        ["<unk>", "<s>", "</s>"]
            .iter()
            .enumerate()
            .map(|(rank, token_str)| json!({"rank": rank, "token_str": token_str, "is_control": true}))
            .collect::<Vec<_>>()
    );
    let file = write_tokenizer_file(&doc);

    let error = Tekkenizer::from_file(file.path())
        .err()
        .expect("missing image control tokens should be rejected");
    assert!(
        matches!(&error, TokenizerError::TokenNotFound(message) if message.contains("[IMG]")),
        "unexpected error {error}"
    );
}

#[test]
fn test_tokenizer_encodes_images_with_its_own_control_tokens() {
    let file = write_tokenizer_file(&tokenizer_json("v13", Some(("image", image_config_json()))));
    let tokenizer = Tekkenizer::from_file(file.path()).unwrap();

    let img = tokenizer.get_control_token("[IMG]").unwrap();
    let img_break = tokenizer.get_control_token("[IMG_BREAK]").unwrap();
    let img_end = tokenizer.get_control_token("[IMG_END]").unwrap();

    // 56x28 pixels at 14px patches merged 2x2 is a 2x1 token grid.
    let encoding = tokenizer
        .encode_image(&solid_image(56, 28, [10, 20, 30]))
        .unwrap();
    assert_eq!(encoding.tokens, vec![img, img, img_end]);
    assert_eq!(encoding.image.dim(), (3, 28, 56));

    let encoding = tokenizer
        .encode_image(&solid_image(56, 56, [10, 20, 30]))
        .unwrap();
    assert_eq!(
        encoding.tokens,
        vec![img, img, img_break, img, img, img_end]
    );
}

#[test]
fn test_tokenizer_tolerates_a_repeated_image_section() {
    // The released v11 tekken.json of Mistral-Small-3.2 lists "multimodal"
    // twice; Python's json.load keeps the last occurrence.
    let last = json!({"image_patch_size": 16, "max_image_size": 512, "spatial_merge_size": 1});
    let doc = serde_json::to_string(&tokenizer_json("v11", Some(("multimodal", last)))).unwrap();
    let with_duplicate = format!(
        "{{\"multimodal\": {},{}",
        image_config_json(),
        doc.strip_prefix('{').unwrap()
    );

    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), &with_duplicate).unwrap();

    let tokenizer = Tekkenizer::from_file(file.path()).unwrap();
    assert_eq!(
        tokenizer.image_config(),
        Some(&ImageConfig::new(16, 512, 1).unwrap()),
        "the last occurrence of a repeated key should win"
    );
}

#[test]
fn test_image_config_survives_a_file_roundtrip() {
    let file = write_tokenizer_file(&tokenizer_json(
        "v11",
        Some(("multimodal", image_config_json())),
    ));
    let tokenizer = Tekkenizer::from_file(file.path()).unwrap();

    let written = tempfile::NamedTempFile::new().unwrap();
    tokenizer.to_file(written.path()).unwrap();

    // The config is always written under the current name.
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(written.path()).unwrap()).unwrap();
    assert_eq!(doc["image"], image_config_json());
    assert!(doc.get("multimodal").is_none());

    let reloaded = Tekkenizer::from_file(written.path()).unwrap();
    assert_eq!(reloaded.image_config(), tokenizer.image_config());
}

#[test]
fn test_with_image_config_toggles_support() {
    let file = write_tokenizer_file(&tokenizer_json("v13", None));
    let tokenizer = Tekkenizer::from_file(file.path()).unwrap();
    assert!(!tokenizer.has_image_support());

    let tokenizer = tokenizer
        .with_image_config(Some(ImageConfig::new(16, 1024, 1).unwrap()))
        .unwrap();
    assert!(tokenizer.has_image_support());

    let tokenizer = tokenizer.with_image_config(None).unwrap();
    assert!(!tokenizer.has_image_support());
    assert_eq!(tokenizer.image_config(), None);
}

fn base64_decode(data: &str) -> Vec<u8> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .unwrap()
}
