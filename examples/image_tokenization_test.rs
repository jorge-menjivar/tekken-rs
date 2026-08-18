use tekken::image::{Image, ImageConfig, ImageEncoder, SpecialImageIds};
use tekken::tekkenizer::Tekkenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Image Tokenization Test with Tekkenizer ===\n");

    // Any vision-capable tekken.json carries an image section; the bundled
    // v7 test asset is audio-only, so fall back to a stand-alone encoder.
    let tokenizer = Tekkenizer::from_file("tests/assets/tekken.json")?;
    println!("✅ Tokenizer loaded successfully!");
    println!("📊 Vocab size: {}", tokenizer.vocab_size());
    println!("📝 Version: {:?}", tokenizer.version());

    let encoder = match tokenizer.image_encoder() {
        Some(encoder) => {
            let config = tokenizer.image_config().unwrap();
            println!(
                "✅ Image support available: patch={}, max_size={}, merge={}",
                config.image_patch_size, config.max_image_size, config.spatial_merge_size
            );
            encoder.clone()
        }
        None => {
            println!("ℹ️  This tokenizer has no image section; using Pixtral's settings instead");
            ImageEncoder::new(
                ImageConfig::new(14, 1540, 2)?,
                SpecialImageIds {
                    img: tokenizer.get_control_token("[IMG]")?,
                    img_break: tokenizer.get_control_token("[IMG_BREAK]")?,
                    img_end: tokenizer.get_control_token("[IMG_END]")?,
                },
            )
        }
    };

    // Token grids can be computed from dimensions alone, without any pixels.
    println!("\n📐 Token grids by image size:");
    for (width, height) in [(28, 28), (640, 480), (1024, 768), (4000, 3000)] {
        let (width_tokens, height_tokens) = encoder.image_to_num_tokens(width, height)?;
        println!(
            "   {width}x{height} px -> {width_tokens}x{height_tokens} tokens ({} total)",
            (width_tokens + 1) * height_tokens
        );
    }

    // Encode a synthetic gradient; use Image::from_file for a real picture.
    let (width, height) = (200, 120);
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(u8::try_from(x * 255 / width).unwrap());
            pixels.push(u8::try_from(y * 255 / height).unwrap());
            pixels.push(128);
        }
    }
    let image = Image::new(width, height, pixels)?;

    println!("\n🖼️  Encoding a {width}x{height} gradient...");
    let encoding = encoder.encode(&image)?;
    println!("📊 Tokens: {}", encoding.tokens.len());
    println!("📊 Processed pixels: {:?}", encoding.image.dim());
    println!(
        "📊 First tokens: {:?}",
        &encoding.tokens[..8.min(encoding.tokens.len())]
    );
    println!("📊 Last token: {:?}", encoding.tokens.last());

    Ok(())
}
