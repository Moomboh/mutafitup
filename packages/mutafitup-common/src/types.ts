/**
 * TypeScript interfaces matching the `export_metadata.json` produced by
 * the Python ONNX export pipeline.
 */

export interface PreprocessingConfig {
  /** Whether to space-separate amino acid characters before tokenization (e.g. ProtT5). */
  space_separate: boolean;
  /** Optional prefix to prepend to each sequence (e.g. "<AA2fold>" for ProstT5). */
  prefix: string | null;
  /** Character replacements applied before tokenization (e.g. {"O": "X", "B": "X"}). */
  char_replacements: Record<string, string>;
}

export interface TaskConfig {
  /** "regression" or "classification". */
  problem_type: "regression" | "classification";
  /** "per_protein" (single output per sequence) or "per_residue" (output per position). */
  level: "per_protein" | "per_residue";
  /** Number of output values (1 for regression, num_classes for classification). */
  num_outputs: number;
  /** Name of the corresponding ONNX output tensor (e.g. "meltome_logits"). */
  output_name: string;
}

export interface ExportMetadata {
  /** HuggingFace checkpoint identifier for the base backbone. */
  base_checkpoint: string;
  /** Fully qualified Python class name of the backbone. */
  backbone_class: string;
  /** Preprocessing instructions for raw protein sequences. */
  preprocessing: PreprocessingConfig;
  /** Task definitions keyed by task name. */
  tasks: Record<string, TaskConfig>;
  /** ONNX model input names (e.g. ["input_ids", "attention_mask"]). */
  inputs: string[];
  /** ONNX model output names (e.g. ["meltome_logits", "subloc_logits"]). */
  outputs: string[];
  /** ONNX opset version used during export. */
  opset_version: number;
  /** SPDX-style license identifier (e.g. "cambrian-open"). */
  license?: string;
  /** URL to the full license text. */
  license_url?: string;
  /** Short license attribution notice. */
  license_notice?: string;
  /** Required attribution text (e.g. "Built with ESM"). */
  attribution?: string;
}

/** Per-task prediction results keyed by task name. */
export type Predictions = Record<string, Float32Array>;
