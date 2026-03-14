/**
 * Shared tokenizer utilities for mutafitup packages.
 *
 * Provides the `tokenize()` function and `TokenizerOutput` interface
 * that are common to both Node.js and browser environments.
 */

import type { PreTrainedTokenizer } from "@huggingface/transformers";

export interface TokenizerOutput {
  /** Token IDs: flat Int64 BigInt array, shape [batchSize, seqLen]. */
  inputIds: BigInt64Array;
  /** Attention mask: flat Int64 BigInt array, shape [batchSize, seqLen]. */
  attentionMask: BigInt64Array;
  /** Batch size. */
  batchSize: number;
  /** Sequence length (with padding). */
  seqLen: number;
}

/**
 * Tokenize a batch of (preprocessed) sequences and return typed arrays
 * ready for ONNX inference.
 *
 * Sequences are padded to the longest sequence in the batch.
 */
export function tokenize(
  tokenizer: PreTrainedTokenizer,
  sequences: string[],
): TokenizerOutput {
  const encoded = tokenizer(sequences, {
    padding: true,
    truncation: false,
    return_tensors: "js",
  });

  const inputIdsData = encoded.input_ids.data as BigInt64Array;
  const attentionMaskData = encoded.attention_mask.data as BigInt64Array;

  const batchSize = sequences.length;
  const seqLen = inputIdsData.length / batchSize;

  return {
    inputIds: inputIdsData,
    attentionMask: attentionMaskData,
    batchSize,
    seqLen,
  };
}
