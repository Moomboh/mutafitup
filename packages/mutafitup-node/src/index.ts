/**
 * mutafitup-node
 *
 * Load an ONNX-exported mutafitup model and run inference on raw protein
 * sequences end-to-end (preprocessing -> tokenization -> ONNX inference).
 */

import { readFile } from "node:fs/promises";
import { join } from "node:path";

import * as ort from "onnxruntime-node";
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
   * @default ["webgpu", "cpu"]
   */
  executionProviders?: ort.InferenceSession.ExecutionProviderConfig[];
}

/**
 * A loaded mutafitup multitask model that provides end-to-end inference
 * from raw protein sequences to per-task predictions.
 *
 * @example
 * ```ts
 * const model = await MutafitupModel.load("path/to/onnx_export");
 * const predictions = await model.predict(["MAKLVFG", "PEPTIDE"]);
 * console.log(predictions.meltome); // Float32Array
 * model.dispose();
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
   * Load a mutafitup ONNX model from an export directory.
   *
   * @param exportDir Path to the directory containing `model.onnx`,
   *   `tokenizer/`, and `export_metadata.json`.
   * @param options Optional configuration (execution providers, etc.).
   */
  static async load(
    exportDir: string,
    options: MutafitupModelOptions = {},
  ): Promise<MutafitupModel> {
    const {
      executionProviders = ["webgpu", "cpu"],
    } = options;

    const metadataPath = join(exportDir, "export_metadata.json");
    const metadataJson = await readFile(metadataPath, "utf-8");
    const metadata: ExportMetadata = JSON.parse(metadataJson);

    const tokenizerDir = join(exportDir, "tokenizer");
    const tokenizer = await loadTokenizer(tokenizerDir);

    const modelPath = join(exportDir, "model.onnx");
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders,
    });

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
