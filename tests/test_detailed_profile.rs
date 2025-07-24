use base64::{Engine as _, engine::general_purpose};
use rustc_hash::FxHashMap;
use std::time::Instant;
use tiktoken_rs::CoreBPE;

#[test]
fn test_detailed_profile() {
    println!("Detailed profiling of tokenizer creation steps...");

    let start = Instant::now();
    let content = std::fs::read_to_string("tekken.json").expect("Failed to read tekken.json");
    let model_data: tekken::config::ModelData =
        serde_json::from_str(&content).expect("Failed to parse JSON");
    println!("File read + JSON parse: {:?}", start.elapsed());

    let config = model_data.config;
    let vocab = model_data.vocab;

    println!("Processing {} vocab tokens...", vocab.len());
    assert!(!vocab.is_empty(), "Vocab should not be empty");

    // Mimic reload_mergeable_ranks function
    let start = Instant::now();
    let inner_vocab_size = config.default_vocab_size - config.default_num_special_tokens;
    let vocab_to_process = if vocab.len() > inner_vocab_size {
        vocab.into_iter().take(inner_vocab_size).collect()
    } else {
        vocab
    };
    println!("Vocab filtering took: {:?}", start.elapsed());

    let start = Instant::now();
    let mut ranks = FxHashMap::default();

    for (i, token) in vocab_to_process.iter().enumerate() {
        if i % 10000 == 0 {
            println!("Processed {} tokens in {:?}", i, start.elapsed());
        }

        let token_bytes = general_purpose::STANDARD
            .decode(&token.token_bytes)
            .expect("Failed to decode base64 token bytes");

        // Verify byte tokens for first 256 tokens
        if token.rank < 256 {
            assert_eq!(
                token_bytes,
                vec![token.rank as u8],
                "Invalid byte token at rank {}",
                token.rank
            );
        }

        ranks.insert(token_bytes, token.rank as u32);
    }
    println!("Base64 decoding and validation took: {:?}", start.elapsed());

    // Verify ranks are contiguous
    let start = Instant::now();
    let expected_ranks: std::collections::HashSet<_> = (0..ranks.len() as u32).collect();
    let actual_ranks: std::collections::HashSet<_> = ranks.values().copied().collect();

    assert_eq!(
        expected_ranks, actual_ranks,
        "Vocabulary ranks are not contiguous"
    );
    println!("Rank validation took: {:?}", start.elapsed());

    // Create CoreBPE
    let start = Instant::now();
    let special_tokens_map: FxHashMap<String, u32> = FxHashMap::default();
    let pattern = r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;

    println!("Creating CoreBPE with {} mergeable ranks...", ranks.len());
    let tokenizer =
        CoreBPE::new(ranks, special_tokens_map, pattern).expect("Failed to create CoreBPE");

    println!("CoreBPE creation took: {:?}", start.elapsed());

    // Test encoding
    let start = Instant::now();
    let (tokens, _) = tokenizer.encode("Hello world!", &std::collections::HashSet::new());
    println!(
        "Encoding test took: {:?} - Tokens: {:?}",
        start.elapsed(),
        &tokens[..5.min(tokens.len())]
    );
    assert!(!tokens.is_empty(), "Tokens should not be empty");
}
