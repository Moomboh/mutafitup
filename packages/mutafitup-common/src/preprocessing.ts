/**
 * Sequence preprocessing matching the Python backbone's
 * `preprocess_sequences()` logic.
 *
 * Transforms are driven by the `preprocessing` field in
 * `export_metadata.json` so the JS side stays in sync with
 * the Python export without hard-coding backbone-specific rules.
 */

import type { PreprocessingConfig } from "./types.js";

/**
 * Apply preprocessing transforms to raw protein sequences.
 *
 * Steps (in order):
 * 1. Character replacements (e.g. rare amino acids O/B/U/Z/J -> X)
 * 2. Space-separate each character (e.g. "MAKLV" -> "M A K L V")
 * 3. Prepend prefix (e.g. "<AA2fold> M A K L V")
 */
export function preprocessSequences(
  sequences: string[],
  config: PreprocessingConfig,
): string[] {
  return sequences.map((seq) => preprocessSequence(seq, config));
}

function preprocessSequence(
  sequence: string,
  config: PreprocessingConfig,
): string {
  let result = sequence;

  if (Object.keys(config.char_replacements).length > 0) {
    const chars = result.split("");
    for (let i = 0; i < chars.length; i++) {
      const replacement = config.char_replacements[chars[i]];
      if (replacement !== undefined) {
        chars[i] = replacement;
      }
    }
    result = chars.join("");
  }

  if (config.space_separate) {
    result = result.split("").join(" ");
  }

  if (config.prefix !== null) {
    result = config.space_separate
      ? `${config.prefix} ${result}`
      : `${config.prefix}${result}`;
  }

  return result;
}
