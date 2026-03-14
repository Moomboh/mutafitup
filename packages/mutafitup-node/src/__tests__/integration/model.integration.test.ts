import { describe, it, expect, afterAll } from "vitest";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { MutafitupModel } from "../../index.js";

// Path to the real ONNX export from the pipeline
const EXPORT_DIR = resolve(
  import.meta.dirname,
  "../../../../../results/onnx_export/accgrad_lora/esmc_300m_all_r4/best_overall",
);

const hasModel =
  existsSync(resolve(EXPORT_DIR, "model.onnx")) &&
  existsSync(resolve(EXPORT_DIR, "export_metadata.json")) &&
  existsSync(resolve(EXPORT_DIR, "tokenizer", "tokenizer.json"));

describe.skipIf(!hasModel)("MutafitupModel integration (ESMc 300M)", () => {
  let model: MutafitupModel;

  // Load once for all tests in this suite (the model is 1.2 GB)
  beforeAll(async () => {
    model = await MutafitupModel.load(EXPORT_DIR, {
      // Use CPU for deterministic integration tests
      executionProviders: ["cpu"],
    });
  }, 60_000); // 60s timeout for model loading

  afterAll(() => {
    model?.dispose();
  });

  it("loads the model and exposes correct metadata", () => {
    expect(model.metadata.base_checkpoint).toBe("esmc_300m");
    expect(model.metadata.tasks).toHaveProperty("meltome");
    expect(model.metadata.tasks).toHaveProperty("subloc");
    expect(model.metadata.tasks.meltome.problem_type).toBe("regression");
    expect(model.metadata.tasks.meltome.level).toBe("per_protein");
    expect(model.metadata.tasks.meltome.num_outputs).toBe(1);
    expect(model.metadata.tasks.subloc.problem_type).toBe("classification");
    expect(model.metadata.tasks.subloc.level).toBe("per_protein");
    expect(model.metadata.tasks.subloc.num_outputs).toBe(10);
  });

  it(
    "predicts on a single sequence with correct output shapes",
    async () => {
      const predictions = await model.predict(["MAKLVFG"]);

      // meltome: per-protein regression -> 1 float per sequence
      expect(predictions.meltome).toBeInstanceOf(Float32Array);
      expect(predictions.meltome.length).toBe(1);
      expect(Number.isFinite(predictions.meltome[0])).toBe(true);

      // subloc: per-protein classification -> 10 logits per sequence
      expect(predictions.subloc).toBeInstanceOf(Float32Array);
      expect(predictions.subloc.length).toBe(10);
      for (let i = 0; i < 10; i++) {
        expect(Number.isFinite(predictions.subloc[i])).toBe(true);
      }
    },
    30_000,
  );

  it(
    "predicts on a batch of sequences",
    async () => {
      const predictions = await model.predict(["MAKLVFG", "PEPTIDE", "ACD"]);

      // meltome: 3 sequences * 1 output = 3
      expect(predictions.meltome.length).toBe(3);
      for (let i = 0; i < 3; i++) {
        expect(Number.isFinite(predictions.meltome[i])).toBe(true);
      }

      // subloc: 3 sequences * 10 classes = 30
      expect(predictions.subloc.length).toBe(30);
      for (let i = 0; i < 30; i++) {
        expect(Number.isFinite(predictions.subloc[i])).toBe(true);
      }
    },
    30_000,
  );

  it(
    "produces deterministic outputs for the same input",
    async () => {
      const predictions1 = await model.predict(["MAKLVFG"]);
      const predictions2 = await model.predict(["MAKLVFG"]);

      expect(predictions1.meltome[0]).toBeCloseTo(predictions2.meltome[0], 5);
      for (let i = 0; i < 10; i++) {
        expect(predictions1.subloc[i]).toBeCloseTo(predictions2.subloc[i], 5);
      }
    },
    30_000,
  );
});
