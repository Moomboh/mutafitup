/**
 * Validate Node.js ONNX predictions against PyTorch predictions.
 *
 * Reads the PyTorch prediction JSONL files produced by the Snakemake
 * predict rule, runs the same sequences through the ONNX model via
 * mutafitup-node, post-processes identically, and compares results.
 *
 * Validation uses correlation-based metrics rather than absolute
 * tolerances, since different GPU backends (MPS vs WebGPU) produce
 * slightly different floating-point results for large models.
 *
 * Only per-protein tasks are validated (per-residue tasks require
 * label-based masking that the JS side does not have access to).
 *
 * Usage:
 *   npx tsx validate_onnx_node.ts <export_dir> <predictions_dir> <report_path>
 *
 * Exit code:
 *   0 — all per-protein tasks pass
 *   1 — at least one task failed or an error occurred
 */

import { readFileSync, writeFileSync, readdirSync, mkdirSync } from "node:fs";
import { join, dirname, basename } from "node:path";
import { MutafitupModel } from "mutafitup-node";
import { listSupportedBackends } from "onnxruntime-node";
import type { ExportMetadata, TaskConfig } from "mutafitup-node";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/** Minimum Pearson correlation for regression tasks to pass. */
const REGRESSION_MIN_PEARSON = 0.8;

/** Minimum Spearman rank correlation for regression tasks to pass. */
const REGRESSION_MIN_SPEARMAN = 0.7;

/** Minimum argmax accuracy for classification tasks to pass. */
const CLASSIFICATION_MIN_ACCURACY = 0.8;

/** Maximum number of sequences to process per batch. */
const BATCH_SIZE = 8;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface PytorchPrediction {
  id: number;
  prediction: number; // scalar for per-protein
  target: number;
  sequence: string;
}

interface RegressionTaskReport {
  task: string;
  problem_type: "regression";
  level: string;
  total_samples: number;
  pearson_r: number;
  spearman_r: number;
  max_abs_diff: number;
  mean_abs_diff: number;
  passed: boolean;
}

interface ClassificationTaskReport {
  task: string;
  problem_type: "classification";
  level: string;
  total_samples: number;
  accuracy: number;
  matched: number;
  mismatched: number;
  passed: boolean;
  mismatched_samples?: Array<{
    id: number;
    sequence: string;
    pytorch: number;
    onnx: number;
  }>;
}

interface SkippedTaskReport {
  task: string;
  problem_type: string;
  level: string;
  total_samples: 0;
  skipped_reason: string;
  passed: true;
}

type TaskReport = RegressionTaskReport | ClassificationTaskReport | SkippedTaskReport;

interface Report {
  export_dir: string;
  predictions_dir: string;
  execution_provider: string;
  tasks: TaskReport[];
  passed: boolean;
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

function pearsonCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 2) return NaN;

  const meanX = x.reduce((s, v) => s + v, 0) / n;
  const meanY = y.reduce((s, v) => s + v, 0) / n;

  let num = 0;
  let denomX = 0;
  let denomY = 0;

  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    num += dx * dy;
    denomX += dx * dx;
    denomY += dy * dy;
  }

  const denom = Math.sqrt(denomX * denomY);
  return denom === 0 ? NaN : num / denom;
}

/**
 * Assign fractional ranks (average of tied positions) to an array of
 * values, matching the default behaviour of scipy.stats.rankdata.
 */
function fractionalRanks(values: number[]): number[] {
  const n = values.length;
  const indexed = values.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);

  const ranks = new Array<number>(n);
  let i = 0;
  while (i < n) {
    let j = i;
    // Find the end of the group of ties
    while (j < n && indexed[j].v === indexed[i].v) j++;
    // Average rank for the tie group (1-based)
    const avgRank = (i + 1 + j) / 2;
    for (let k = i; k < j; k++) {
      ranks[indexed[k].i] = avgRank;
    }
    i = j;
  }

  return ranks;
}

function spearmanCorrelation(x: number[], y: number[]): number {
  return pearsonCorrelation(fractionalRanks(x), fractionalRanks(y));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function readJsonl(path: string): PytorchPrediction[] {
  const content = readFileSync(path, "utf-8").trim();
  if (!content) return [];
  return content.split("\n").map((line) => JSON.parse(line));
}

/**
 * Post-process raw ONNX logits for a single per-protein sample,
 * mirroring the Python predict.py logic.
 */
function postProcessArgmax(
  logits: Float32Array,
  offset: number,
  numOutputs: number,
): number {
  let bestIdx = 0;
  let bestVal = logits[offset];
  for (let i = 1; i < numOutputs; i++) {
    if (logits[offset + i] > bestVal) {
      bestVal = logits[offset + i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const [exportDir, predictionsDir, reportPath] = process.argv.slice(2);

  if (!exportDir || !predictionsDir || !reportPath) {
    console.error(
      "Usage: validate_onnx_node.ts <export_dir> <predictions_dir> <report_path>",
    );
    process.exit(1);
  }

  console.log(`Export dir:      ${exportDir}`);
  console.log(`Predictions dir: ${predictionsDir}`);
  console.log(`Report path:     ${reportPath}`);

  // Log available ONNX backends
  const backends = listSupportedBackends();
  console.log(
    `\nAvailable ONNX backends: ${backends.map((b) => b.name).join(", ")}`,
  );
  console.log(`Execution provider: webgpu`);

  // Load model
  console.log("Loading ONNX model ...");
  const model = await MutafitupModel.load(exportDir, {
    executionProviders: ["webgpu"],
  });
  const metadata: ExportMetadata = model.metadata;

  // Discover which JSONL files exist in the predictions directory
  const jsonlFiles = readdirSync(predictionsDir).filter((f) =>
    f.endsWith(".jsonl"),
  );
  const availableTasks = new Set(
    jsonlFiles.map((f) => basename(f, ".jsonl")),
  );

  console.log(`Available prediction files: ${[...availableTasks].join(", ")}`);
  console.log(
    `Model tasks: ${Object.keys(metadata.tasks).join(", ")}`,
  );

  const taskReports: TaskReport[] = [];
  let allPassed = true;

  for (const [taskName, taskConfig] of Object.entries(metadata.tasks)) {
    console.log(`\n--- Task: ${taskName} (${taskConfig.problem_type}) ---`);

    // Skip per-residue tasks
    if (taskConfig.level !== "per_protein") {
      console.log(`  Skipping (level=${taskConfig.level}, only per_protein supported)`);
      taskReports.push({
        task: taskName,
        problem_type: taskConfig.problem_type,
        level: taskConfig.level,
        total_samples: 0,
        skipped_reason: "per-residue validation not implemented",
        passed: true,
      });
      continue;
    }

    // Check that a JSONL file exists for this task
    if (!availableTasks.has(taskName)) {
      console.log(`  Skipping (no ${taskName}.jsonl found in predictions dir)`);
      taskReports.push({
        task: taskName,
        problem_type: taskConfig.problem_type,
        level: taskConfig.level,
        total_samples: 0,
        skipped_reason: `no ${taskName}.jsonl in predictions directory`,
        passed: true,
      });
      continue;
    }

    // Read PyTorch predictions
    const jsonlPath = join(predictionsDir, `${taskName}.jsonl`);
    const pytorchPreds = readJsonl(jsonlPath);
    console.log(`  PyTorch predictions: ${pytorchPreds.length} samples`);

    if (pytorchPreds.length === 0) {
      taskReports.push({
        task: taskName,
        problem_type: taskConfig.problem_type,
        level: taskConfig.level,
        total_samples: 0,
        skipped_reason: "empty JSONL file",
        passed: true,
      });
      continue;
    }

    // Extract sequences (preserving order matching PyTorch predictions)
    const sequences = pytorchPreds.map((p) => p.sequence);

    // Run ONNX predictions in batches, collecting raw logits
    const allLogits: Float32Array[] = [];
    const totalBatches = Math.ceil(sequences.length / BATCH_SIZE);

    for (let i = 0; i < sequences.length; i += BATCH_SIZE) {
      const batchSeqs = sequences.slice(i, i + BATCH_SIZE);
      const predictions = await model.predict(batchSeqs);
      const taskLogits = predictions[taskName];

      if (!taskLogits) {
        throw new Error(
          `ONNX model did not produce output for task "${taskName}"`,
        );
      }

      allLogits.push(taskLogits);

      const done = Math.min(i + BATCH_SIZE, sequences.length);
      const batchNum = Math.floor(i / BATCH_SIZE) + 1;
      process.stdout.write(
        `\r  Predicting: ${done}/${sequences.length} sequences (batch ${batchNum}/${totalBatches})`,
      );
    }
    process.stdout.write("\n");

    // Concatenate all logits
    const totalLogitLen = allLogits.reduce((s, a) => s + a.length, 0);
    const onnxLogits = new Float32Array(totalLogitLen);
    let offset = 0;
    for (const chunk of allLogits) {
      onnxLogits.set(chunk, offset);
      offset += chunk.length;
    }

    // -----------------------------------------------------------------------
    // Evaluate based on problem type
    // -----------------------------------------------------------------------

    if (taskConfig.problem_type === "regression") {
      // Extract per-sample scalar values
      const pytorchVals = pytorchPreds.map((p) => p.prediction);
      const onnxVals: number[] = [];
      for (let i = 0; i < sequences.length; i++) {
        onnxVals.push(onnxLogits[i * taskConfig.num_outputs]);
      }

      // Compute statistics
      const diffs = pytorchVals.map((v, i) => Math.abs(v - onnxVals[i]));
      const maxAbsDiff = Math.max(...diffs);
      const meanAbsDiff = diffs.reduce((s, d) => s + d, 0) / diffs.length;
      const pR = pearsonCorrelation(pytorchVals, onnxVals);
      const sR = spearmanCorrelation(pytorchVals, onnxVals);

      const passed =
        pR >= REGRESSION_MIN_PEARSON && sR >= REGRESSION_MIN_SPEARMAN;
      if (!passed) allPassed = false;

      console.log(
        `  Pearson r:  ${pR.toFixed(6)} (threshold: ${REGRESSION_MIN_PEARSON})\n` +
        `  Spearman r: ${sR.toFixed(6)} (threshold: ${REGRESSION_MIN_SPEARMAN})\n` +
        `  Max diff:   ${maxAbsDiff.toFixed(6)}\n` +
        `  Mean diff:  ${meanAbsDiff.toFixed(6)}\n` +
        `  ${passed ? "PASSED" : "FAILED"}`,
      );

      taskReports.push({
        task: taskName,
        problem_type: "regression",
        level: taskConfig.level,
        total_samples: pytorchPreds.length,
        pearson_r: pR,
        spearman_r: sR,
        max_abs_diff: maxAbsDiff,
        mean_abs_diff: meanAbsDiff,
        passed,
      });

    } else {
      // Classification — compare argmax predictions
      const pytorchClasses = pytorchPreds.map((p) => p.prediction);
      const onnxClasses: number[] = [];
      for (let i = 0; i < sequences.length; i++) {
        onnxClasses.push(
          postProcessArgmax(
            onnxLogits,
            i * taskConfig.num_outputs,
            taskConfig.num_outputs,
          ),
        );
      }

      let matched = 0;
      let mismatched = 0;
      const mismatchedSamples: ClassificationTaskReport["mismatched_samples"] = [];

      for (let i = 0; i < pytorchClasses.length; i++) {
        if (pytorchClasses[i] === onnxClasses[i]) {
          matched++;
        } else {
          mismatched++;
          if (mismatchedSamples!.length < 10) {
            mismatchedSamples!.push({
              id: pytorchPreds[i].id,
              sequence: pytorchPreds[i].sequence.substring(0, 50),
              pytorch: pytorchClasses[i],
              onnx: onnxClasses[i],
            });
          }
        }
      }

      const accuracy = matched / pytorchClasses.length;
      const passed = accuracy >= CLASSIFICATION_MIN_ACCURACY;
      if (!passed) allPassed = false;

      console.log(
        `  Accuracy:   ${(accuracy * 100).toFixed(1)}% (${matched}/${pytorchClasses.length}, threshold: ${(CLASSIFICATION_MIN_ACCURACY * 100).toFixed(0)}%)\n` +
        `  ${passed ? "PASSED" : "FAILED"}`,
      );

      taskReports.push({
        task: taskName,
        problem_type: "classification",
        level: taskConfig.level,
        total_samples: pytorchPreds.length,
        accuracy,
        matched,
        mismatched,
        passed,
        ...(mismatchedSamples.length > 0 && { mismatched_samples: mismatchedSamples }),
      });
    }
  }

  // Write report
  const report: Report = {
    export_dir: exportDir,
    predictions_dir: predictionsDir,
    execution_provider: "webgpu",
    tasks: taskReports,
    passed: allPassed,
  };

  mkdirSync(dirname(reportPath), { recursive: true });
  writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");
  console.log(`\nReport written to ${reportPath}`);

  // Clean up
  model.dispose();

  if (!allPassed) {
    console.error("\nONNX validation FAILED");
    process.exit(1);
  }

  console.log("\nONNX validation PASSED");
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
