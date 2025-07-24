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
fn test_no_tools_conversation_structure() {
    let tokenizer = get_tokenizer();

    // Test a conversation structure similar to the Python integration tests
    let messages = [
        ("user", "What's the result of 5 + 5?"),
        ("assistant", "The result of 5 + 5 is 10."),
        ("user", "What is the square root of 64?"),
        (
            "assistant",
            "The square root of 64 is 8, because 8 x 8 equals 64.",
        ),
        (
            "user",
            "Can you multiply the results of the previous two questions?",
        ),
        ("assistant", "Sure! The result of 10 x 8 is 80."),
        ("user", "Thanks"),
    ];

    // Test individual message encoding
    for (role, content) in &messages {
        let tokens = tokenizer.encode(content, false, false).unwrap();
        let decoded = tokenizer
            .decode(&tokens, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(*content, decoded, "Failed for {role} message: {content}");

        // Ensure we get reasonable token counts
        assert!(
            !tokens.is_empty(),
            "Should produce at least one token for: {content}"
        );
        assert!(
            tokens.len() < content.len() + 10,
            "Token count seems unreasonable for: {content}"
        );
    }
}

#[test]
fn test_mathematical_expressions() {
    let tokenizer = get_tokenizer();

    let math_expressions = [
        "5 + 5",
        "10 x 8",
        "sqrt(64)",
        "8 x 8 = 64",
        "result = 80",
        "2^3 = 8",
        "π ≈ 3.14159",
        "∫ x dx = x²/2 + C",
    ];

    for expr in &math_expressions {
        let tokens = tokenizer.encode(expr, false, false).unwrap();
        let decoded = tokenizer
            .decode(&tokens, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(*expr, decoded, "Mathematical expression failed: {expr}");
    }
}

#[test]
fn test_conversation_with_bos_eos() {
    let tokenizer = get_tokenizer();

    // Test encoding with BOS/EOS like a real conversation would
    let user_messages = [
        "Hello, how are you?",
        "Can you help me with math?",
        "What's 15 * 23?",
    ];

    let assistant_responses = [
        "Hello! I'm doing well, thank you for asking. How can I help you today?",
        "Of course! I'd be happy to help you with math problems.",
        "15 * 23 = 345",
    ];

    for (user_msg, assistant_msg) in user_messages.iter().zip(assistant_responses.iter()) {
        // Test user message with BOS
        let user_tokens = tokenizer.encode(user_msg, true, false).unwrap();
        let user_decoded = tokenizer
            .decode(&user_tokens, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(*user_msg, user_decoded);

        // Test assistant response with EOS
        let assistant_tokens = tokenizer.encode(assistant_msg, false, true).unwrap();
        let assistant_decoded = tokenizer
            .decode(&assistant_tokens, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(*assistant_msg, assistant_decoded);

        // Verify BOS/EOS tokens are present when using Keep policy
        let user_with_special = tokenizer
            .decode(&user_tokens, SpecialTokenPolicy::Keep)
            .unwrap();
        let assistant_with_special = tokenizer
            .decode(&assistant_tokens, SpecialTokenPolicy::Keep)
            .unwrap();

        // Should contain special tokens when keeping them
        assert!(user_with_special.len() > user_msg.len());
        assert!(assistant_with_special.len() > assistant_msg.len());
    }
}

#[test]
fn test_long_conversation_context() {
    let tokenizer = get_tokenizer();

    // Test a longer conversation that might appear in real usage
    let conversation = "User: I'm working on a machine learning project and need help understanding neural networks. Can you explain the basic concepts?\n\nAssistant: I'd be happy to explain neural networks! A neural network is a computational model inspired by biological neural networks. Here are the key concepts:\n\n1. **Neurons (Nodes)**: The basic processing units that receive inputs, apply a function, and produce an output.\n\n2. **Layers**: Neurons are organized in layers:\n   - Input layer: Receives the initial data\n   - Hidden layers: Process the information\n   - Output layer: Produces the final result\n\n3. **Weights and Biases**: Parameters that determine how much influence each input has on the output.\n\n4. **Activation Functions**: Functions like ReLU, sigmoid, or tanh that introduce non-linearity.\n\n5. **Training**: The process of adjusting weights and biases using algorithms like backpropagation.\n\nWould you like me to elaborate on any of these concepts?\n\nUser: Yes, can you explain backpropagation in more detail?\n\nAssistant: Certainly! Backpropagation is the key algorithm for training neural networks. Here's how it works:\n\n**The Process:**\n1. **Forward Pass**: Input data flows through the network, layer by layer, until it reaches the output.\n\n2. **Loss Calculation**: Compare the predicted output with the actual target to calculate the error (loss).\n\n3. **Backward Pass**: The error is propagated backward through the network:\n   - Calculate how much each weight contributed to the error\n   - Use the chain rule of calculus to compute gradients\n   - Update weights to minimize the error\n\n**Key Points:**\n- It's called \"backpropagation\" because we propagate the error backward\n- Uses gradient descent to find the optimal weights\n- The learning rate controls how big steps we take during updates\n- Multiple iterations gradually improve the network's performance\n\nThis iterative process continues until the network learns to make accurate predictions!";

    let tokens = tokenizer.encode(conversation, true, true).unwrap();
    let decoded = tokenizer
        .decode(&tokens, SpecialTokenPolicy::Ignore)
        .unwrap();
    assert_eq!(conversation, decoded);

    // Verify we get a reasonable number of tokens for this long text
    assert!(
        tokens.len() > 100,
        "Long conversation should produce many tokens"
    );
    assert!(
        tokens.len() < conversation.len(),
        "Tokenization should be more efficient than character-level"
    );

    // Test with different special token policies
    let decoded_keep = tokenizer.decode(&tokens, SpecialTokenPolicy::Keep).unwrap();
    assert!(
        decoded_keep.len() > conversation.len(),
        "Keep policy should include special tokens"
    );
}

#[test]
fn test_code_examples() {
    let tokenizer = get_tokenizer();

    let code_examples = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "function quickSort(arr) {\n    if (arr.length <= 1) return arr;\n    const pivot = arr[0];\n    const left = arr.slice(1).filter(x => x < pivot);\n    const right = arr.slice(1).filter(x => x >= pivot);\n    return [...quickSort(left), pivot, ...quickSort(right)];\n}",
        "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}",
        "fn main() {\n    let numbers = vec![1, 2, 3, 4, 5];\n    let sum: i32 = numbers.iter().sum();\n    println!(\"Sum: {}\", sum);\n}",
    ];

    for code in &code_examples {
        let tokens = tokenizer.encode(code, false, false).unwrap();
        let decoded = tokenizer
            .decode(&tokens, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(*code, decoded, "Code example failed to roundtrip");

        // Code should typically tokenize efficiently
        assert!(tokens.len() > 10, "Code should produce multiple tokens");
    }
}

#[test]
fn test_multilingual_content() {
    let tokenizer = get_tokenizer();

    let multilingual_examples = [
        "Hello, world! | Hola, mundo! | Bonjour, le monde!",
        "English text mixed with 中文字符 and العربية",
        "Русский текст с English words и العربية أيضاً",
        "Japanese: こんにちは世界 Korean: 안녕하세요 Thai: สวัสดีชาวโลก",
        "Mathematical symbols: ∑∫∂∇∞ Greek: αβγδε Currency: $€¥£₹",
    ];

    for text in &multilingual_examples {
        let tokens = tokenizer.encode(text, false, false).unwrap();
        let decoded = tokenizer
            .decode(&tokens, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(*text, decoded, "Multilingual text failed: {text}");
    }
}

#[test]
fn test_edge_cases() {
    let tokenizer = get_tokenizer();

    let long_repeated = "a".repeat(1000);
    let edge_cases = [
        "",                             // Empty string
        " ",                            // Single space
        "\n",                           // Single newline
        "\t",                           // Single tab
        "   \n\t   ",                   // Mixed whitespace
        "a",                            // Single character
        "🚀",                           // Single emoji
        long_repeated.as_str(),         // Very long repeated character
        "Hello\0World",                 // Null character (if supported)
        "Line1\nLine2\rLine3\r\nLine4", // Different line endings
    ];

    for case in &edge_cases {
        let tokens = tokenizer.encode(case, false, false).unwrap();
        let decoded = tokenizer
            .decode(&tokens, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(*case, decoded, "Edge case failed: {case:?}");
    }
}

#[test]
fn test_token_efficiency() {
    let tokenizer = get_tokenizer();

    // Test that common words/phrases tokenize efficiently
    let efficiency_tests = [
        ("the", 1),                     // Very common word should be 1 token
        ("hello", 1),                   // Common greeting should be 1 token
        ("world", 1),                   // Common word should be 1 token
        ("python", 1),                  // Programming language should be 1 token
        ("function", 2),                // Longer word might be 1-2 tokens
        ("artificial intelligence", 4), // Phrase should be reasonably tokenized
    ];

    for (text, max_expected_tokens) in &efficiency_tests {
        let tokens = tokenizer.encode(text, false, false).unwrap();
        assert!(
            tokens.len() <= *max_expected_tokens,
            "Text '{}' produced {} tokens, expected <= {}",
            text,
            tokens.len(),
            max_expected_tokens
        );

        let decoded = tokenizer
            .decode(&tokens, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(*text, decoded);
    }
}

#[test]
fn test_special_token_injection_safety() {
    let tokenizer = get_tokenizer();

    // Test that user input containing special token strings doesn't break tokenization
    let potentially_problematic_inputs = [
        "<s>User input with BOS token</s>",
        "[INST]Instruction-like input[/INST]",
        "[TOOL_CALLS]Fake tool call[/TOOL_CALLS]",
        "Text with <unk> token string",
        "Mixed <s>content</s> with [INST]multiple[/INST] token strings",
    ];

    for input in &potentially_problematic_inputs {
        let tokens = tokenizer.encode(input, false, false).unwrap();
        let decoded = tokenizer
            .decode(&tokens, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(
            *input, decoded,
            "Special token string handling failed: {input}"
        );

        // Also test with BOS/EOS
        let tokens_with_special = tokenizer.encode(input, true, true).unwrap();
        let decoded_with_special = tokenizer
            .decode(&tokens_with_special, SpecialTokenPolicy::Ignore)
            .unwrap();
        assert_eq!(
            *input, decoded_with_special,
            "BOS/EOS + special token strings failed: {input}"
        );
    }
}
