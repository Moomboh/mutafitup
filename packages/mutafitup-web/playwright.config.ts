import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright configuration for mutafitup-web e2e tests.
 *
 * Two projects:
 * - chromium-wasm: Standard Chromium (headless shell), WASM EP only.
 * - chromium-gpu:  Full Chromium with GPU flags for WebGPU EP tests.
 *                  Uses --headless=new (not the headless shell) so GPU
 *                  features (ANGLE, WebGPU, WebNN) are available.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false, // model loading is heavy, run sequentially
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1, // single worker — model is 1.2 GB
  reporter: process.env.CI ? "github" : "list",
  timeout: 180_000, // 3 minutes per test (model loading is slow)

  use: {
    baseURL: "http://localhost:5199",
    trace: "on-first-retry",
  },

  projects: [
    {
      name: "chromium-wasm",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          args: [
            "--disable-gpu-sandbox",
          ],
        },
      },
    },
    {
      name: "chromium-gpu",
      use: {
        ...devices["Desktop Chrome"],
        // Use the full chromium binary (not chromium-headless-shell) so
        // that GPU/ANGLE/WebGPU are available. Playwright picks the
        // stripped-down headless shell when `headless: true`, which
        // lacks GPU support entirely.
        headless: false,
        launchOptions: {
          args: [
            // New headless mode: runs the full browser headlessly with
            // GPU support (unlike the old --headless / headless shell).
            "--headless=new",
            "--disable-gpu-sandbox",
            // WebGPU — may be needed in some Chromium builds.
            "--enable-unsafe-webgpu",
            // ANGLE is required for GPU access in headless mode.
            "--use-gl=angle",
            "--use-angle=default",
          ],
        },
      },
    },
  ],

  webServer: {
    command: "pnpm run dev:e2e",
    url: "http://localhost:5199",
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
