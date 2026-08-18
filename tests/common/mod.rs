use base64::{Engine as _, engine::general_purpose};
use tekken::config::TokenInfo;

/// The standard Tekken tokenization regex, shared by small-vocab test fixtures.
pub const PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Builds a minimal vocabulary of the 256 byte tokens plus "hello" and "world".
pub fn small_vocab() -> Vec<TokenInfo> {
    let mut vocab: Vec<TokenInfo> = (0..256usize)
        .map(|i| TokenInfo {
            rank: i,
            token_bytes: general_purpose::STANDARD.encode([u8::try_from(i).unwrap()]),
            token_str: Some(format!("byte_{i}")),
        })
        .collect();
    vocab.push(TokenInfo {
        rank: 256,
        token_bytes: general_purpose::STANDARD.encode(b"hello"),
        token_str: Some("hello".to_string()),
    });
    vocab.push(TokenInfo {
        rank: 257,
        token_bytes: general_purpose::STANDARD.encode(b"world"),
        token_str: Some("world".to_string()),
    });
    vocab
}
