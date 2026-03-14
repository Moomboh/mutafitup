/**
 * mutafitup-web
 *
 * Load an ONNX-exported mutafitup model in the browser and run inference
 * on raw protein sequences end-to-end (preprocessing -> tokenization ->
 * ONNX inference).
 *
 * Uses `onnxruntime-web/all` for WebNN + WebGPU + WASM support.
 */

import * as ort from "onnxruntime-web/all";
import type { PreTrainedTokenizer } from "@huggingface/transformers";

import type { ExportMetadata, Predictions } from "mutafitup-common";
import { preprocessSequences, tokenize } from "mutafitup-common";
import { loadTokenizer } from "./tokenizer.js";

export type {
  ExportMetadata,
  Predictions,
  PreprocessingConfig,
  TaskConfig,
  TokenizerOutput,
} from "mutafitup-common";
export { preprocessSequences, tokenize } from "mutafitup-common";

/**
 * Pre-loaded model assets, as an alternative to providing a URL
 * for `MutafitupModel.load()`.
 */
export interface LoadedModel {
  /** Parsed export metadata. */
  metadata: ExportMetadata;
  /** ONNX model as an ArrayBuffer or a URL string to the .onnx file. */
  model: ArrayBuffer | string;
  /**
   * Tokenizer source: a URL prefix to the directory containing
   * `tokenizer.json`, or a HuggingFace Hub model ID.
   */
  tokenizerSource: string;
}

export interface PredictOptions {
  /**
   * Maximum number of sequences per ONNX inference call. When the
   * input contains more sequences than this, they are processed in
   * chunks and results are concatenated.
   *
   * Leave unset (or set to `Infinity`) to process all sequences in a
   * single call.
   */
  batchSize?: number;
}

export interface MutafitupModelOptions {
  /**
   * ONNX execution providers to use, in order of preference.
   *
   * @default [
   *   { name: "webnn", deviceType: "gpu", powerPreference: "high-performance" },
   *   { name: "webgpu" },
   *   { name: "wasm" },
   * ]
   */
  executionProviders?: ort.InferenceSession.ExecutionProviderConfig[];
}

/** Default execution provider preference order. */
const DEFAULT_EXECUTION_PROVIDERS: ort.InferenceSession.ExecutionProviderConfig[] = [
  { name: "webnn", deviceType: "gpu", powerPreference: "high-performance" } as ort.InferenceSession.ExecutionProviderConfig,
  { name: "webgpu" },
  { name: "wasm" },
];

/**
 * A loaded mutafitup multitask model that provides end-to-end inference
 * from raw protein sequences to per-task predictions in the browser.
 *
 * @example
 * ```ts
 * // Load from a URL prefix
 * const model = await MutafitupModel.load("https://example.com/models/my-model");
 * const predictions = await model.predict(["MAKLVFG", "PEPTIDE"]);
 * console.log(predictions.meltome); // Float32Array
 * model.dispose();
 * ```
 *
 * @example
 * ```ts
 * // Load from pre-fetched assets
 * const model = await MutafitupModel.load({
 *   metadata: myParsedMetadata,
 *   model: myOnnxArrayBuffer,
 *   tokenizerSource: "https://example.com/models/my-model/tokenizer",
 * });
 * ```
 */
export class MutafitupModel {
  private constructor(
    private session: ort.InferenceSession,
    private tokenizer: PreTrainedTokenizer,
    private _metadata: ExportMetadata,
  ) {}

  /** Export metadata describing tasks, preprocessing, and I/O schema. */
  get metadata(): ExportMetadata {
    return this._metadata;
  }

  /**
   * Load a mutafitup ONNX model.
   *
   * @param source Either a URL prefix string (the directory containing
   *   `model.onnx`, `tokenizer/`, and `export_metadata.json`) or a
   *   pre-loaded `LoadedModel` object.
   * @param options Optional configuration (execution providers, etc.).
   */
  static async load(
    source: string | LoadedModel,
    options: MutafitupModelOptions = {},
  ): Promise<MutafitupModel> {
    const {
      executionProviders = DEFAULT_EXECUTION_PROVIDERS,
    } = options;

    if (typeof source === "string") {
      return MutafitupModel._loadFromUrl(source, executionProviders);
    }
    return MutafitupModel._loadFromObject(source, executionProviders);
  }

  /** Load from a URL prefix by fetching all assets. */
  private static async _loadFromUrl(
    urlPrefix: string,
    executionProviders: ort.InferenceSession.ExecutionProviderConfig[],
  ): Promise<MutafitupModel> {
    const base = urlPrefix.endsWith("/") ? urlPrefix.slice(0, -1) : urlPrefix;

    const metadataResp = await fetch(`${base}/export_metadata.json`);
    if (!metadataResp.ok) {
      throw new Error(
        `Failed to fetch export_metadata.json: ${metadataResp.status} ${metadataResp.statusText}`,
      );
    }
    const metadata: ExportMetadata = await metadataResp.json();

    const tokenizer = await loadTokenizer(`${base}/tokenizer`);

    const modelUrl = `${base}/model.onnx`;
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders,
    });

    return new MutafitupModel(session, tokenizer, metadata);
  }

  /** Load from a pre-loaded object with metadata, model data, and tokenizer source. */
  private static async _loadFromObject(
    loaded: LoadedModel,
    executionProviders: ort.InferenceSession.ExecutionProviderConfig[],
  ): Promise<MutafitupModel> {
    const { metadata, model, tokenizerSource } = loaded;

    const tokenizer = await loadTokenizer(tokenizerSource);

    let session: ort.InferenceSession;
    if (typeof model === "string") {
      session = await ort.InferenceSession.create(model, {
        executionProviders,
      });
    } else {
      session = await ort.InferenceSession.create(new Uint8Array(model), {
        executionProviders,
      });
    }

    return new MutafitupModel(session, tokenizer, metadata);
  }

  /**
   * Run inference on one or more raw protein sequences.
   *
   * Handles preprocessing, tokenization, and ONNX inference. Returns
   * a record mapping task names to their output tensors as flat
   * `Float32Array`s.
   *
   * For **per-protein** tasks the array has shape `[batchSize * numOutputs]`.
   * For **per-residue** tasks the array has shape `[batchSize * seqLen * numOutputs]`.
   *
   * @param sequences Raw amino acid sequences (e.g. `["MAKLVFG"]`).
   * @param options Optional predict configuration (batch size, etc.).
   */
  async predict(
    sequences: string[],
    options: PredictOptions = {},
  ): Promise<Predictions> {
    const { batchSize: maxBatch = Infinity } = options;

    if (sequences.length > maxBatch) {
      const accumulated: Record<string, Float32Array[]> = {};

      for (let i = 0; i < sequences.length; i += maxBatch) {
        const chunk = sequences.slice(i, i + maxBatch);
        const chunkPreds = await this._predictBatch(chunk);

        for (const [taskName, data] of Object.entries(chunkPreds)) {
          (accumulated[taskName] ??= []).push(data);
        }
      }

      const predictions: Predictions = {};
      for (const [taskName, arrays] of Object.entries(accumulated)) {
        const totalLen = arrays.reduce((s, a) => s + a.length, 0);
        const merged = new Float32Array(totalLen);
        let offset = 0;
        for (const arr of arrays) {
          merged.set(arr, offset);
          offset += arr.length;
        }
        predictions[taskName] = merged;
      }

      return predictions;
    }

    return this._predictBatch(sequences);
  }

  /** Run a single batch through the ONNX session (no chunking). */
  private async _predictBatch(sequences: string[]): Promise<Predictions> {
    const preprocessed = preprocessSequences(
      sequences,
      this._metadata.preprocessing,
    );

    const { inputIds, attentionMask, batchSize, seqLen } = tokenize(
      this.tokenizer,
      preprocessed,
    );

    const feeds: Record<string, ort.Tensor> = {
      input_ids: new ort.Tensor("int64", inputIds, [batchSize, seqLen]),
      attention_mask: new ort.Tensor("int64", attentionMask, [batchSize, seqLen]),
    };

    const results = await this.session.run(feeds);

    const predictions: Predictions = {};
    for (const [taskName, taskConfig] of Object.entries(this._metadata.tasks)) {
      const outputTensor = results[taskConfig.output_name];
      if (outputTensor) {
        predictions[taskName] = new Float32Array(outputTensor.data as Float32Array);
      }
    }

    return predictions;
  }

  /**
   * Release ONNX session resources. The model instance should not be
   * used after calling dispose.
   */
  dispose(): void {
    this.session.release();
  }
}
