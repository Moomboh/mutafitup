/**
 * E2E test harness for mutafitup-web.
 *
 * Imports mutafitup-web and exposes helper functions on `window`
 * for Playwright to call via `page.evaluate()`.
 */

import * as ort from "onnxruntime-web/all";
import { MutafitupModel } from "mutafitup-web";
import type { ExportMetadata, Predictions } from "mutafitup-web";

// Disable multi-threading to simplify test setup
// (avoids SharedArrayBuffer requirements in some contexts)
ort.env.wasm.numThreads = 1;

let model: MutafitupModel | null = null;

declare global {
  interface Window {
    __loadModel: (
      url: string,
      executionProviders: ort.InferenceSession.ExecutionProviderConfig[],
    ) => Promise<void>;
    __predict: (
      sequences: string[],
      options?: { batchSize?: number },
    ) => Promise<Record<string, number[]>>;
    __getMetadata: () => ExportMetadata | null;
    __dispose: () => void;
    __ready: boolean;
  }
}

window.__loadModel = async (
  url: string,
  executionProviders: ort.InferenceSession.ExecutionProviderConfig[],
) => {
  if (model) {
    model.dispose();
    model = null;
  }

  // Wrap model loading in a timeout. onnxruntime-web has no internal
  // timeout for navigator.ml.createContext(), which can hang
  // indefinitely if the WebNN backend doesn't support the requested
  // device type. This gives a clear error instead of waiting for the
  // full Playwright test timeout.
  const LOAD_TIMEOUT_MS = 120_000; // 2 minutes

  const loadPromise = MutafitupModel.load(url, { executionProviders });
  const timeoutPromise = new Promise<never>((_resolve, reject) => {
    setTimeout(
      () => reject(new Error(
        `Model loading timed out after ${LOAD_TIMEOUT_MS}ms. ` +
        `This likely means navigator.ml.createContext() hung — ` +
        `the requested WebNN device type may not be available.`,
      )),
      LOAD_TIMEOUT_MS,
    );
  });

  model = await Promise.race([loadPromise, timeoutPromise]);
};

window.__predict = async (
  sequences: string[],
  options?: { batchSize?: number },
): Promise<Record<string, number[]>> => {
  if (!model) throw new Error("Model not loaded");
  const predictions: Predictions = await model.predict(sequences, options);

  // Convert Float32Arrays to plain number arrays for serialization
  // (page.evaluate can't transfer typed arrays directly)
  const result: Record<string, number[]> = {};
  for (const [key, arr] of Object.entries(predictions)) {
    result[key] = Array.from(arr);
  }
  return result;
};

window.__getMetadata = (): ExportMetadata | null => {
  return model?.metadata ?? null;
};

window.__dispose = () => {
  if (model) {
    model.dispose();
    model = null;
  }
};

window.__ready = true;

document.getElementById("status")!.textContent = "ready";
