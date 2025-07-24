use tekken::special_tokens::SpecialTokenPolicy;
use tekken::tekkenizer::Tekkenizer;

#[test]
fn test_specific_tokens() {
    println!("=== TESTING SPECIFIC TOKEN DECODING ===");

    // Load tokenizer
    let tokenizer_path = "tekken.json";
    let tokenizer =
        Tekkenizer::from_file(tokenizer_path).expect("Failed to load tokenizer from file");

    // Test problematic tokens
    let test_tokens = [4998, 20999];

    for &token_id in &test_tokens {
        let decoded = tokenizer
            .decode(&[token_id], SpecialTokenPolicy::Ignore)
            .expect("Failed to decode token");
        println!("Token {}: '{}'", token_id, decoded);
        assert!(!decoded.is_empty(), "Decoded token should not be empty");
    }

    // Test full sequence
    let full_tokens = vec![4998, 1878, 1044, 2036, 20574, 20999];
    let full_decoded = tokenizer
        .decode(&full_tokens, SpecialTokenPolicy::Ignore)
        .expect("Failed to decode full sequence");
    println!("Full sequence: '{}'", full_decoded);
    assert!(
        !full_decoded.is_empty(),
        "Full decoded sequence should not be empty"
    );
}
