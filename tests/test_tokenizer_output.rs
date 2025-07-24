use std::sync::OnceLock;
use tekken::special_tokens::SpecialTokenPolicy;
use tekken::tekkenizer::Tekkenizer;

static TOKENIZER: OnceLock<Tekkenizer> = OnceLock::new();

fn get_tokenizer() -> &'static Tekkenizer {
    TOKENIZER.get_or_init(|| {
        Tekkenizer::from_file("tests/assets/tekken.json")
            .expect("Failed to load tokenizer from file")
    })
}

#[test]
fn test_tokenizer_info() {
    let tokenizer = get_tokenizer();

    assert_eq!(tokenizer.vocab_size(), 131072);
    assert_eq!(format!("{:?}", tokenizer.version()), "V7");
}

#[test]
fn test_hello_world() {
    let tokenizer = get_tokenizer();
    let input = "Hello, world!";
    let expected_tokens = vec![22177, 1044, 4304, 1033];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "Hello, world!");
    assert_eq!(tokens.len(), 4);
    assert_eq!(input, decoded);
}

#[test]
fn test_quick_brown_fox() {
    let tokenizer = get_tokenizer();
    let input = "The quick brown fox jumps over the lazy dog.";
    let expected_tokens = vec![
        1784, 7586, 22980, 94137, 72993, 2136, 1278, 42757, 10575, 1046,
    ];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "The quick brown fox jumps over the lazy dog.");
    assert_eq!(tokens.len(), 10);
    assert_eq!(input, decoded);
}

#[test]
fn test_mistral_tekken_tokenizer() {
    let tokenizer = get_tokenizer();
    let input = "This is a test of the Mistral Tekken tokenizer.";
    let expected_tokens = vec![
        4380, 1395, 1261, 2688, 1307, 1278, 42301, 2784, 47213, 3569, 128405, 1046,
    ];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "This is a test of the Mistral Tekken tokenizer.");
    assert_eq!(tokens.len(), 12);
    assert_eq!(input, decoded);
}

#[test]
fn test_emojis_unicode() {
    let tokenizer = get_tokenizer();
    let input = "Emojis and unicode characters work too!";
    let expected_tokens = vec![5969, 3659, 1275, 1321, 79219, 11084, 2196, 4382, 1033];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "Emojis and unicode characters work too!");
    assert_eq!(tokens.len(), 9);
    assert_eq!(input, decoded);
}

#[test]
fn test_single_word_hello() {
    let tokenizer = get_tokenizer();
    let input = "Hello";
    let expected_tokens = vec![22177];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "Hello");
    assert_eq!(tokens.len(), 1);
    assert_eq!(input, decoded);
}

#[test]
fn test_single_word_world() {
    let tokenizer = get_tokenizer();
    let input = "world";
    let expected_tokens = vec![34049];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "world");
    assert_eq!(tokens.len(), 1);
    assert_eq!(input, decoded);
}

#[test]
fn test_single_word_test() {
    let tokenizer = get_tokenizer();
    let input = "test";
    let expected_tokens = vec![4417];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "test");
    assert_eq!(tokens.len(), 1);
    assert_eq!(input, decoded);
}

#[test]
fn test_single_char_a() {
    let tokenizer = get_tokenizer();
    let input = "a";
    let expected_tokens = vec![1097];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "a");
    assert_eq!(tokens.len(), 1);
    assert_eq!(input, decoded);
}

#[test]
fn test_single_word_the() {
    let tokenizer = get_tokenizer();
    let input = "the";
    let expected_tokens = vec![3265];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "the");
    assert_eq!(tokens.len(), 1);
    assert_eq!(input, decoded);
}

#[test]
fn test_single_word_python() {
    let tokenizer = get_tokenizer();
    let input = "Python";
    let expected_tokens = vec![46728];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "Python");
    assert_eq!(tokens.len(), 1);
    assert_eq!(input, decoded);
}

#[test]
fn test_single_word_rust() {
    let tokenizer = get_tokenizer();
    let input = "Rust";
    let expected_tokens = vec![1082, 1616];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "Rust");
    assert_eq!(tokens.len(), 2);
    assert_eq!(input, decoded);
}

#[test]
fn test_single_word_tokenizer() {
    let tokenizer = get_tokenizer();
    let input = "tokenizer";
    let expected_tokens = vec![15017, 7463];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "tokenizer");
    assert_eq!(tokens.len(), 2);
    assert_eq!(input, decoded);
}

#[test]
fn test_single_word_encoding() {
    let tokenizer = get_tokenizer();
    let input = "encoding";
    let expected_tokens = vec![47130];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "encoding");
    assert_eq!(tokens.len(), 1);
    assert_eq!(input, decoded);
}

#[test]
fn test_single_word_decoding() {
    let tokenizer = get_tokenizer();
    let input = "decoding";
    let expected_tokens = vec![18888, 7967];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "decoding");
    assert_eq!(tokens.len(), 2);
    assert_eq!(input, decoded);
}

#[test]
fn test_single_word_comparison() {
    let tokenizer = get_tokenizer();
    let input = "comparison";
    let expected_tokens = vec![69959, 3693];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "comparison");
    assert_eq!(tokens.len(), 2);
    assert_eq!(input, decoded);
}

#[test]
fn test_simple_sentence() {
    let tokenizer = get_tokenizer();
    let input = "Simple sentence.";
    let expected_tokens = vec![28683, 19286, 1046];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "Simple sentence.");
    assert_eq!(tokens.len(), 3);
    assert_eq!(input, decoded);
}

#[test]
fn test_numbers() {
    let tokenizer = get_tokenizer();
    let input = "Another test case with numbers: 123, 456, 789.";
    let expected_tokens = vec![
        18661, 2688, 2937, 1454, 8091, 1058, 1032, 1049, 1050, 1051, 1044, 1032, 1052, 1053, 1054,
        1044, 1032, 1055, 1056, 1057, 1046,
    ];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "Another test case with numbers: 123, 456, 789.");
    assert_eq!(tokens.len(), 21);
    assert_eq!(input, decoded);
}

#[test]
fn test_special_characters() {
    let tokenizer = get_tokenizer();
    let input = "Special characters: @#$%^&*()_+-={}[]|\\:;\"'<>,.?/";
    let expected_tokens = vec![
        40124, 11084, 1058, 2126, 1035, 1036, 1037, 1094, 1038, 1042, 1690, 1095, 104799, 3181,
        1125, 4344, 17743, 1058, 36211, 96726, 24482, 1046, 1063, 1047,
    ];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(
        decoded,
        "Special characters: @#$%^&*()_+-={}[]|\\:;\"'<>,.?/"
    );
    assert_eq!(tokens.len(), 24);
    assert_eq!(input, decoded);
}

#[test]
fn test_mixed_case() {
    let tokenizer = get_tokenizer();
    let input = "Mixed CaSe WoRdS";
    let expected_tokens = vec![1077, 5422, 10645, 3201, 18739, 1082, 1100, 1083];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "Mixed CaSe WoRdS");
    assert_eq!(tokens.len(), 8);
    assert_eq!(input, decoded);
}

#[test]
fn test_whitespace_handling() {
    let tokenizer = get_tokenizer();
    let input = "   whitespace   handling   ";
    let expected_tokens = vec![1256, 81024, 1256, 21490, 1293];

    let tokens = tokenizer.encode(input, false, false).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();

    assert_eq!(tokens, expected_tokens);
    assert_eq!(decoded, "   whitespace   handling   ");
    assert_eq!(tokens.len(), 5);
    assert_eq!(input, decoded);
}
