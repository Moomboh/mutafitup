/**
 * Playwright e2e tests for mutafitup-web.
 *
 * Loads the real ONNX export in a browser context and runs inference
 * end-to-end. Model files are served from the local filesystem via
 * the Vite dev server's `serveModelFiles` plugin (streaming, handles
 * the 1.2 GB external data file).
 *
 * Skips if the ONNX export files are not present on disk.
 */

import { test, expect, type Page } from "@playwright/test";
import { existsSync } from "node:fs";
import { resolve, join } from "node:path";

// ── Model file paths ────────────────────────────────────────────────

const EXPORT_DIR = resolve(
  import.meta.dirname,
  "../../../results/onnx_export/accgrad_lora/esmc_300m_all_r4/best_overall",
);

const hasModel =
  existsSync(join(EXPORT_DIR, "model.onnx")) &&
  existsSync(join(EXPORT_DIR, "export_metadata.json")) &&
  existsSync(join(EXPORT_DIR, "tokenizer", "tokenizer.json"));

// ── Model URL that the test app will request ────────────────────────

// Model files are served by the Vite dev server's `serveModelFiles`
// plugin at /__model__/*, which streams files from the local export
// directory.
const MODEL_URL = "http://localhost:5199/__model__";

// ── Helpers ─────────────────────────────────────────────────────────

/**
 * Wait for the test app to be ready (main.ts has executed and set
 * window.__ready = true).
 */
async function waitForReady(page: Page): Promise<void> {
  await page.waitForFunction(() => (window as any).__ready === true, null, {
    timeout: 15_000,
  });
}

/**
 * Load the model in the browser with given execution providers.
 */
async function loadModel(
  page: Page,
  executionProviders: Array<string | Record<string, unknown>>,
): Promise<void> {
  await page.evaluate(
    async ({ url, eps }) => {
      await (window as any).__loadModel(url, eps);
    },
    { url: MODEL_URL, eps: executionProviders },
  );
}

/**
 * Run inference and return predictions as plain number arrays.
 */
async function predict(
  page: Page,
  sequences: string[],
  options?: { batchSize?: number },
): Promise<Record<string, number[]>> {
  return await page.evaluate(
    async ({ seqs, opts }) => {
      return await (window as any).__predict(seqs, opts);
    },
    { seqs: sequences, opts: options },
  );
}

function getMetadata(page: Page): Promise<any> {
  return page.evaluate(() => (window as any).__getMetadata());
}

function dispose(page: Page): Promise<void> {
  return page.evaluate(() => (window as any).__dispose());
}

// ── WASM Tests ──────────────────────────────────────────────────────

test.describe("mutafitup-web WASM EP", () => {
  test.skip(!hasModel, "ONNX export files not found — skipping e2e tests");

  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForReady(page);
  });

  test.afterEach(async ({ page }) => {
    await dispose(page);
  });

  test("loads the model and exposes correct metadata", async ({ page }) => {
    await loadModel(page, [{ name: "wasm" }]);
    const metadata = await getMetadata(page);

    expect(metadata.base_checkpoint).toBe("esmc_300m");
    expect(metadata.tasks).toHaveProperty("meltome");
    expect(metadata.tasks).toHaveProperty("subloc");
    expect(metadata.tasks.meltome.problem_type).toBe("regression");
    expect(metadata.tasks.meltome.level).toBe("per_protein");
    expect(metadata.tasks.meltome.num_outputs).toBe(1);
    expect(metadata.tasks.subloc.problem_type).toBe("classification");
    expect(metadata.tasks.subloc.level).toBe("per_protein");
    expect(metadata.tasks.subloc.num_outputs).toBe(10);
  });

  test("predicts on a single sequence with correct output shapes", async ({
    page,
  }) => {
    await loadModel(page, [{ name: "wasm" }]);
    const predictions = await predict(page, ["MAKLVFG"]);

    // meltome: per-protein regression -> 1 float per sequence
    expect(predictions.meltome).toHaveLength(1);
    expect(Number.isFinite(predictions.meltome[0])).toBe(true);

    // subloc: per-protein classification -> 10 logits per sequence
    expect(predictions.subloc).toHaveLength(10);
    for (let i = 0; i < 10; i++) {
      expect(Number.isFinite(predictions.subloc[i])).toBe(true);
    }
  });

  test("predicts on a batch of sequences", async ({ page }) => {
    await loadModel(page, [{ name: "wasm" }]);
    const predictions = await predict(page, ["MAKLVFG", "PEPTIDE", "ACD"]);

    // meltome: 3 sequences * 1 output = 3
    expect(predictions.meltome).toHaveLength(3);
    for (let i = 0; i < 3; i++) {
      expect(Number.isFinite(predictions.meltome[i])).toBe(true);
    }

    // subloc: 3 sequences * 10 classes = 30
    expect(predictions.subloc).toHaveLength(30);
    for (let i = 0; i < 30; i++) {
      expect(Number.isFinite(predictions.subloc[i])).toBe(true);
    }
  });

  test("produces deterministic outputs for the same input", async ({
    page,
  }) => {
    await loadModel(page, [{ name: "wasm" }]);
    const predictions1 = await predict(page, ["MAKLVFG"]);
    const predictions2 = await predict(page, ["MAKLVFG"]);

    expect(predictions1.meltome[0]).toBeCloseTo(predictions2.meltome[0], 5);
    for (let i = 0; i < 10; i++) {
      expect(predictions1.subloc[i]).toBeCloseTo(predictions2.subloc[i], 5);
    }
  });
});

// ── WebGPU Tests ────────────────────────────────────────────────────

test.describe("mutafitup-web WebGPU EP", () => {
  test.skip(!hasModel, "ONNX export files not found — skipping e2e tests");

  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForReady(page);
  });

  test.afterEach(async ({ page }) => {
    await dispose(page);
  });

  test("loads and predicts with WebGPU", async ({ page }) => {
    // WebGPU shader compilation can be slow the first time for a
    // 300M parameter model. Allow 5 minutes.
    test.setTimeout(300_000);

    // Check that a GPU adapter is actually obtainable, not just that
    // the API exists. The headless shell exposes navigator.gpu but
    // requestAdapter() returns null without real GPU access.
    const hasAdapter = await page.evaluate(async () => {
      if (!("gpu" in navigator)) return false;
      const adapter = await (navigator as any).gpu.requestAdapter();
      return adapter != null;
    });
    if (!hasAdapter) {
      test.skip(true, "WebGPU adapter not available in this browser configuration");
      return;
    }

    // WebGPU only — no WASM fallback. This proves that GPU inference
    // actually works rather than silently falling back.
    await loadModel(page, [{ name: "webgpu" }]);

    const predictions = await predict(page, ["MAKLVFG"]);

    expect(predictions.meltome).toHaveLength(1);
    expect(Number.isFinite(predictions.meltome[0])).toBe(true);
    expect(predictions.subloc).toHaveLength(10);
    for (let i = 0; i < 10; i++) {
      expect(Number.isFinite(predictions.subloc[i])).toBe(true);
    }
  });
});
