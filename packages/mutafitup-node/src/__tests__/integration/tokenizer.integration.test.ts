import { describe, it, expect } from "vitest";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { loadTokenizer, tokenize } from "../../tokenizer.js";

// Path to the real ONNX export from the pipeline
const EXPORT_DIR = resolve(
  import.meta.dirname,
  "../../../../../results/onnx_export/accgrad_lora/esmc_300m_all_r4/best_overall",
);
const TOKENIZER_DIR = resolve(EXPORT_DIR, "tokenizer");

const hasTokenizer = existsSync(resolve(TOKENIZER_DIR, "tokenizer.json"));

describe.skipIf(!hasTokenizer)("tokenizer integration (ESMc)", () => {
  it("loads the tokenizer from disk", async () => {
    const tokenizer = await loadTokenizer(TOKENIZER_DIR);
    expect(tokenizer).toBeDefined();
  });

  it("tokenizes a single sequence", async () => {
    const tokenizer = await loadTokenizer(TOKENIZER_DIR);
    const result = tokenize(tokenizer, ["MAKLVFG"]);

    expect(result.batchSize).toBe(1);
    // ESMc adds <cls> prefix and <eos> suffix: 7 AAs + 2 special = 9
    expect(result.seqLen).toBe(9);
    expect(result.inputIds.length).toBe(9);
    expect(result.attentionMask.length).toBe(9);

    // All attention mask values should be 1 (no padding for single sequence)
    for (let i = 0; i < result.attentionMask.length; i++) {
      expect(result.attentionMask[i]).toBe(1n);
    }
  });

  it("tokenizes a batch with padding", async () => {
    const tokenizer = await loadTokenizer(TOKENIZER_DIR);
    const result = tokenize(tokenizer, ["MAKLVFG", "MA"]);

    expect(result.batchSize).toBe(2);
    // Padded to the longer sequence: 7 + 2 special = 9
    expect(result.seqLen).toBe(9);
    expect(result.inputIds.length).toBe(18); // 2 * 9

    // Second sequence should have padding (attention_mask = 0) at the end
    // seq2 "MA" = 2 AAs + 2 special = 4 real tokens, 5 padding
    const seq2MaskStart = 9; // start of second sequence in flat array
    // First 4 tokens should be attended
    for (let i = 0; i < 4; i++) {
      expect(result.attentionMask[seq2MaskStart + i]).toBe(1n);
    }
    // Remaining 5 should be padded
    for (let i = 4; i < 9; i++) {
      expect(result.attentionMask[seq2MaskStart + i]).toBe(0n);
    }
  });

  it("produces consistent token IDs for the same sequence", async () => {
    const tokenizer = await loadTokenizer(TOKENIZER_DIR);
    const result1 = tokenize(tokenizer, ["MAKLVFG"]);
    const result2 = tokenize(tokenizer, ["MAKLVFG"]);

    expect(result1.inputIds).toEqual(result2.inputIds);
  });
});
