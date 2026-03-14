/**
 * Node.js-specific tokenizer loader.
 *
 * Loads a HuggingFace tokenizer from a local directory using
 * `local_files_only: true`.
 */

import { AutoTokenizer, type PreTrainedTokenizer } from "@huggingface/transformers";

export type { TokenizerOutput } from "mutafitup-common";
export { tokenize } from "mutafitup-common";

/**
 * Load a HuggingFace tokenizer from a local directory.
 *
 * @param tokenizerDir Absolute or relative path to the directory containing
 *   `tokenizer.json`, `tokenizer_config.json`, etc.
 */
export async function loadTokenizer(
  tokenizerDir: string,
): Promise<PreTrainedTokenizer> {
  return await AutoTokenizer.from_pretrained(tokenizerDir, {
    local_files_only: true,
  });
}
