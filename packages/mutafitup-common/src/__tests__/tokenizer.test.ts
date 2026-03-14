import { describe, it, expect, vi } from "vitest";
import { tokenize } from "../tokenizer.js";

// We test tokenize() with a mock tokenizer (platform-specific loadTokenizer
// is tested via integration tests in mutafitup-node / mutafitup-web).

function createMockTokenizer(
  inputIds: bigint[],
  attentionMask: bigint[],
  batchSize: number,
) {
  const seqLen = inputIds.length / batchSize;

  return vi.fn().mockReturnValue({
    input_ids: {
      data: BigInt64Array.from(inputIds),
      dims: [batchSize, seqLen],
    },
    attention_mask: {
      data: BigInt64Array.from(attentionMask),
      dims: [batchSize, seqLen],
    },
  });
}

describe("tokenize", () => {
  it("returns correct batchSize and seqLen for a single sequence", () => {
    // Simulate tokenizing "MAKLV" -> [0, 4, 5, 6, 7, 8, 2] (cls + 5 AAs + eos)
    const ids = [0n, 4n, 5n, 6n, 7n, 8n, 2n];
    const mask = [1n, 1n, 1n, 1n, 1n, 1n, 1n];
    const mockTokenizer = createMockTokenizer(ids, mask, 1);

    const result = tokenize(mockTokenizer as any, ["MAKLV"]);

    expect(result.batchSize).toBe(1);
    expect(result.seqLen).toBe(7);
    expect(result.inputIds).toBeInstanceOf(BigInt64Array);
    expect(result.attentionMask).toBeInstanceOf(BigInt64Array);
    expect(result.inputIds.length).toBe(7);
    expect(result.attentionMask.length).toBe(7);
  });

  it("returns correct batchSize and seqLen for a batch with padding", () => {
    // Batch of 2 sequences, padded to length 5
    // seq1: [0, 4, 5, 6, 2]  mask: [1, 1, 1, 1, 1]
    // seq2: [0, 7, 2, 1, 1]  mask: [1, 1, 1, 0, 0]
    const ids = [0n, 4n, 5n, 6n, 2n, 0n, 7n, 2n, 1n, 1n];
    const mask = [1n, 1n, 1n, 1n, 1n, 1n, 1n, 1n, 0n, 0n];
    const mockTokenizer = createMockTokenizer(ids, mask, 2);

    const result = tokenize(mockTokenizer as any, ["MAKLV", "AB"]);

    expect(result.batchSize).toBe(2);
    expect(result.seqLen).toBe(5);
    expect(result.inputIds.length).toBe(10);
    expect(result.attentionMask.length).toBe(10);
  });

  it("passes correct options to the tokenizer", () => {
    const ids = [0n, 4n, 2n];
    const mask = [1n, 1n, 1n];
    const mockTokenizer = createMockTokenizer(ids, mask, 1);

    tokenize(mockTokenizer as any, ["M"]);

    expect(mockTokenizer).toHaveBeenCalledWith(["M"], {
      padding: true,
      truncation: false,
      return_tensors: "js",
    });
  });

  it("preserves token ID values correctly", () => {
    const ids = [0n, 100n, 200n, 2n];
    const mask = [1n, 1n, 1n, 1n];
    const mockTokenizer = createMockTokenizer(ids, mask, 1);

    const result = tokenize(mockTokenizer as any, ["XY"]);

    expect(result.inputIds[0]).toBe(0n);
    expect(result.inputIds[1]).toBe(100n);
    expect(result.inputIds[2]).toBe(200n);
    expect(result.inputIds[3]).toBe(2n);
  });
});
