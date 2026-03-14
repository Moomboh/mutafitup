/**
 * Browser-specific tokenizer loader.
 *
 * When `source` is an HTTP(S) URL, the tokenizer files are fetched
 * directly and the tokenizer is constructed manually. This works around
 * `AutoTokenizer.from_pretrained()` rejecting arbitrary URLs (it only
 * accepts HuggingFace Hub model IDs or local filesystem paths).
 *
 * When `source` is a Hub model ID, `from_pretrained()` is used directly.
 */

import {
  AutoTokenizer,
  PreTrainedTokenizer,
  type PreTrainedTokenizer as PreTrainedTokenizerType,
} from "@huggingface/transformers";

export type { TokenizerOutput } from "mutafitup-common";
export { tokenize } from "mutafitup-common";

/** Check whether a string looks like an HTTP(S) URL. */
function isHttpUrl(s: string): boolean {
  try {
    const url = new URL(s);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

/**
 * Fetch a JSON file from a URL, throwing a descriptive error on failure.
 */
async function fetchJson<T>(url: string): Promise<T> {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(
      `Failed to fetch ${url}: ${resp.status} ${resp.statusText}`,
    );
  }
  return resp.json() as Promise<T>;
}

/**
 * Resolve the correct tokenizer class from the `tokenizer_class` field
 * in `tokenizer_config.json`, matching the logic of
 * `AutoTokenizer.from_pretrained()`.
 */
function resolveTokenizerClass(
  tokenizerConfig: Record<string, unknown>,
): typeof PreTrainedTokenizer {
  // The config may include a "Fast" suffix — strip it to match the mapping.
  const rawName =
    (tokenizerConfig.tokenizer_class as string | undefined)?.replace(
      /Fast$/,
      "",
    ) ?? "PreTrainedTokenizer";

  const mapping = AutoTokenizer.TOKENIZER_CLASS_MAPPING as Record<
    string,
    typeof PreTrainedTokenizer | undefined
  >;
  const cls = mapping[rawName];

  if (!cls) {
    console.warn(
      `Unknown tokenizer class "${rawName}", falling back to PreTrainedTokenizer.`,
    );
    return PreTrainedTokenizer;
  }

  return cls;
}

/**
 * Load a HuggingFace tokenizer from a URL or model identifier.
 *
 * @param source URL prefix (e.g. "https://example.com/models/my-model/tokenizer")
 *   or a HuggingFace Hub model ID (e.g. "facebook/esm2_t6_8M_UR50D").
 */
export async function loadTokenizer(
  source: string,
): Promise<PreTrainedTokenizerType> {
  if (isHttpUrl(source)) {
    return loadTokenizerFromUrl(source);
  }

  return await AutoTokenizer.from_pretrained(source);
}

/**
 * Load a tokenizer by fetching `tokenizer.json` and
 * `tokenizer_config.json` directly from an HTTP URL prefix, then
 * constructing the tokenizer instance manually.
 */
async function loadTokenizerFromUrl(
  urlPrefix: string,
): Promise<PreTrainedTokenizerType> {
  const base = urlPrefix.endsWith("/") ? urlPrefix.slice(0, -1) : urlPrefix;

  const [tokenizerJSON, tokenizerConfig] = await Promise.all([
    fetchJson<Record<string, unknown>>(`${base}/tokenizer.json`),
    fetchJson<Record<string, unknown>>(`${base}/tokenizer_config.json`),
  ]);

  const TokenizerClass = resolveTokenizerClass(tokenizerConfig);
  return new TokenizerClass(tokenizerJSON, tokenizerConfig);
}
