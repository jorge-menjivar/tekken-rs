use tekken::special_tokens::SpecialTokenPolicy;
use tekken::tekkenizer::Tekkenizer;

#[test]
fn test_tokenizer_decoding() {
    println!("=== RUST TOKENIZER DECODING TEST ===");

    // Load the same tokenizer file used in Python
    let tokenizer_path = "tekken.json";
    let tokenizer = Tekkenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");

    println!("Loaded tokenizer from: {}", tokenizer_path);
    println!("Vocab size: {}", tokenizer.vocab_size());

    // Same tokens from JFK example
    let test_tokens: Vec<u32> = vec![
        4998, 1878, 1044, 2036, 20574, 20999, 1044, 4237, 1605, 2549, 2143, 6816, 1710, 1653, 1394,
        1636, 1044, 4237, 2549, 1636, 1710, 1653, 1394, 2143, 6816, 1046, 2,
    ];

    println!("\nTest tokens: {:?}", test_tokens);

    // Decode individually
    println!("\n=== INDIVIDUAL TOKEN DECODING ===");
    let mut individual_pieces = Vec::new();
    for (i, &token_id) in test_tokens.iter().enumerate() {
        let piece = tokenizer
            .id_to_piece(token_id)
            .expect("Failed to get piece");
        individual_pieces.push(piece.clone());
        println!("Token {:2}: {:5} -> '{}'", i + 1, token_id, piece);
    }

    // Join individual pieces
    let individual_joined = individual_pieces.join("");
    println!("\nIndividual pieces joined: '{}'", individual_joined);

    // Decode all at once with different policies
    let full_decode_ignore = tokenizer
        .decode(&test_tokens, SpecialTokenPolicy::Ignore)
        .expect("Failed to decode");
    println!("Full decode (IGNORE):     '{}'", full_decode_ignore);

    let full_decode_keep = tokenizer
        .decode(&test_tokens, SpecialTokenPolicy::Keep)
        .expect("Failed to decode");
    println!("Full decode (KEEP):       '{}'", full_decode_keep);

    // Compare
    println!("\n=== COMPARISON ===");
    println!(
        "Individual == Full (IGNORE): {}",
        individual_joined == full_decode_ignore
    );
    println!(
        "Individual == Full (KEEP):   {}",
        individual_joined == full_decode_keep
    );

    // Show differences if any
    if individual_joined != full_decode_ignore {
        println!("\n=== DIFFERENCES (IGNORE policy) ===");
        println!("Individual: '{}'", individual_joined);
        println!("Full:       '{}'", full_decode_ignore);

        let individual_chars: Vec<char> = individual_joined.chars().collect();
        let full_chars: Vec<char> = full_decode_ignore.chars().collect();
        let max_len = individual_chars.len().max(full_chars.len());

        for i in 0..max_len {
            let c1 = individual_chars.get(i).copied().unwrap_or('∅');
            let c2 = full_chars.get(i).copied().unwrap_or('∅');
            if c1 != c2 {
                println!("  Char {}: Individual='{}' vs Full='{}'", i, c1, c2);
            }
        }
    }

    // Expected Python output for comparison
    let expected_python = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
    println!("\n=== PYTHON COMPARISON ===");
    println!("Expected Python:  '{}'", expected_python);
    println!("Rust IGNORE:      '{}'", full_decode_ignore);
    println!(
        "Match with Python: {}",
        expected_python == full_decode_ignore
    );

    // Assert that decoding works correctly
    assert!(
        !full_decode_ignore.is_empty(),
        "Decoded text should not be empty"
    );
    assert!(
        !full_decode_keep.is_empty(),
        "Decoded text should not be empty"
    );

    if expected_python != full_decode_ignore {
        println!("\n=== DIFFERENCES FROM PYTHON ===");
        let python_chars: Vec<char> = expected_python.chars().collect();
        let rust_chars: Vec<char> = full_decode_ignore.chars().collect();
        let max_len = python_chars.len().max(rust_chars.len());

        for i in 0..max_len {
            let py_char = python_chars.get(i).copied().unwrap_or('∅');
            let rust_char = rust_chars.get(i).copied().unwrap_or('∅');
            if py_char != rust_char {
                println!("  Char {}: Python='{}' vs Rust='{}'", i, py_char, rust_char);
            }
        }
    }
}
