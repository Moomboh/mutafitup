/// Tokenizer loading and batch encoding using the HuggingFace
/// `tokenizers` crate.
use anyhow::{Context, Result};
use tokenizers::tokenizer::Tokenizer;
use tokenizers::utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy};

/// Result of batch tokenization: flattened token IDs and attention masks
/// plus the batch dimensions.
pub struct TokenizerOutput {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub batch_size: usize,
    pub seq_len: usize,
}

/// Load a tokenizer from a `tokenizer.json` file on the filesystem
/// and configure padding.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_tokenizer_from_file(path: &str) -> Result<Tokenizer> {
    let mut tokenizer =
        Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
    configure_padding(&mut tokenizer);
    Ok(tokenizer)
}

/// Load a tokenizer from raw JSON bytes and configure padding.
#[cfg(target_arch = "wasm32")]
pub fn load_tokenizer_from_bytes(json_bytes: &[u8]) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_bytes(json_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to parse tokenizer JSON: {e}"))?;
    configure_padding(&mut tokenizer);
    Ok(tokenizer)
}

/// Set up batch padding (pad to longest in batch, right-padded).
fn configure_padding(tokenizer: &mut Tokenizer) {
    let pad_id = tokenizer.token_to_id("<pad>").unwrap_or(0);
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        direction: PaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id,
        pad_type_id: 0,
        pad_token: "<pad>".to_string(),
    }));
}

/// Encode a batch of preprocessed protein sequences.
///
/// Padding is already configured on the tokenizer during loading.
/// Returns flattened i64 arrays for `input_ids` and `attention_mask`,
/// plus the batch dimensions (batch_size, seq_len).
pub fn encode_batch(tokenizer: &Tokenizer, sequences: &[String]) -> Result<TokenizerOutput> {
    let inputs: Vec<tokenizers::EncodeInput> = sequences
        .iter()
        .map(|s| tokenizers::EncodeInput::Single(s.as_str().into()))
        .collect();

    let encodings = tokenizer
        .encode_batch(inputs, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))
        .context("Failed to encode batch")?;

    let batch_size = encodings.len();
    let seq_len = encodings.first().map(|e| e.get_ids().len()).unwrap_or(0);

    let mut input_ids = Vec::with_capacity(batch_size * seq_len);
    let mut attention_mask = Vec::with_capacity(batch_size * seq_len);

    for encoding in &encodings {
        input_ids.extend(encoding.get_ids().iter().map(|&id| id as i64));
        attention_mask.extend(encoding.get_attention_mask().iter().map(|&v| v as i64));
    }

    Ok(TokenizerOutput {
        input_ids,
        attention_mask,
        batch_size,
        seq_len,
    })
}
