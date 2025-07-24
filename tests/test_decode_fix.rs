use tekken::special_tokens::SpecialTokenPolicy;
use tekken::tekkenizer::Tekkenizer;

#[test]
fn test_decode_fix() {
    println!("=== TESTING POTENTIAL DECODE FIXES ===");

    // Load tokenizer
    let tokenizer_path = "tekken.json";
    let tokenizer = Tekkenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");

    // Test tokens
    let test_tokens: Vec<u32> = vec![4998, 1878, 1044, 2036, 20574, 20999];
    println!("Test tokens: {:?}", test_tokens);

    // Current behavior
    let current_result = tokenizer
        .decode(&test_tokens, SpecialTokenPolicy::Ignore)
        .expect("Failed to decode");
    println!("Current result: '{}'", current_result);

    // Test individual token decoding and manual joining
    println!("\n=== MANUAL JOINING TEST ===");
    let mut manual_result = String::new();
    for &token_id in &test_tokens {
        let piece = tokenizer
            .id_to_piece(token_id)
            .expect("Failed to get piece");
        manual_result.push_str(&piece);
    }
    println!("Manual join result: '{}'", manual_result);

    // Test decode_all to see the individual segments
    let segments = tokenizer
        .decode_all(&test_tokens, SpecialTokenPolicy::Ignore)
        .expect("Failed to decode all");
    println!("Decode segments: {:?}", segments);
    println!("Segments joined: '{}'", segments.join(""));

    // Compare with expected
    let expected = "And so, my fellow Americans";
    println!("\nExpected: '{}'", expected);
    println!("Current:  '{}'", current_result);
    println!("Manual:   '{}'", manual_result);
    println!("Match (manual): {}", expected == manual_result);

    // Test smaller groups
    println!("\n=== SMALL GROUP TESTS ===");

    // Single token
    let single = tokenizer
        .decode(&[4998], SpecialTokenPolicy::Ignore)
        .expect("Failed to decode single");
    println!("Single token [4998]: '{}'", single);

    // Two tokens
    let two = tokenizer
        .decode(&[4998, 1878], SpecialTokenPolicy::Ignore)
        .expect("Failed to decode two");
    println!("Two tokens [4998, 1878]: '{}'", two);

    // Three tokens
    let three = tokenizer
        .decode(&[4998, 1878, 1044], SpecialTokenPolicy::Ignore)
        .expect("Failed to decode three");
    println!("Three tokens [4998, 1878, 1044]: '{}'", three);

    // Basic assertions
    assert!(
        !current_result.is_empty(),
        "Decoded result should not be empty"
    );
    assert!(
        !manual_result.is_empty(),
        "Manual result should not be empty"
    );
}
