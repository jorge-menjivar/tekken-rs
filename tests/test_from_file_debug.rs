use tekken::tekkenizer::Tekkenizer;

#[test]
fn test_from_file_debug() {
    println!("Testing Tekkenizer::from_file with debug info...");

    println!("Step 1: Reading file...");
    let content = std::fs::read_to_string("tests/assets/tekken.json")
        .expect("Failed to read tekken.json file");
    println!(
        "Step 2: File read successfully, size: {} bytes",
        content.len()
    );
    assert!(!content.is_empty(), "File content should not be empty");

    println!("Step 3: Parsing JSON...");
    let model_data: serde_json::Value =
        serde_json::from_str(&content).expect("Failed to parse JSON");
    println!("Step 4: JSON parsed successfully");
    assert!(model_data.is_object(), "JSON should be an object");

    println!("Step 5: Creating tokenizer...");
    let tokenizer = Tekkenizer::from_file("tests/assets/tekken.json")
        .expect("Failed to create tokenizer from file");
    println!("Step 6: Tokenizer created successfully!");

    let vocab_size = tokenizer.vocab_size();
    println!("Vocab size: {vocab_size}");
    assert!(vocab_size > 0, "Vocab size should be greater than 0");
}
