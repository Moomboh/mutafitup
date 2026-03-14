import { defineConfig, type Plugin } from "vite";
import { dirname, resolve, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createReadStream, existsSync, statSync } from "node:fs";

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Path to the ONNX export directory. The e2e tests serve model files
 * from this directory at `/__model__/*`.
 */
const EXPORT_DIR = resolve(
  __dirname,
  "../../../../results/onnx_export/accgrad_lora/esmc_300m_all_r4/best_overall",
);

/**
 * Vite plugin that serves model files from the ONNX export directory
 * at `/__model__/*` using streaming.
 */
function serveModelFiles(): Plugin {
  return {
    name: "serve-model-files",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith("/__model__/")) {
          return next();
        }

        const relativePath = req.url.slice("/__model__/".length);
        const localPath = join(EXPORT_DIR, relativePath);

        if (!existsSync(localPath)) {
          res.statusCode = 404;
          res.end(`Not found: ${relativePath}`);
          return;
        }

        const stat = statSync(localPath);

        let contentType = "application/octet-stream";
        if (localPath.endsWith(".json")) contentType = "application/json";
        else if (localPath.endsWith(".onnx")) contentType = "application/octet-stream";

        res.setHeader("Content-Type", contentType);
        res.setHeader("Content-Length", stat.size);
        // Required by COEP
        res.setHeader("Cross-Origin-Resource-Policy", "same-origin");

        createReadStream(localPath).pipe(res);
      });
    },
  };
}

/**
 * Vite config for the e2e test app.
 *
 * Key considerations:
 * - `onnxruntime-web` must be excluded from dep optimization because
 *   Vite's pre-bundling rewrites `import.meta.url`, which breaks ort's
 *   automatic WASM file resolution (it uses `import.meta.url` to infer
 *   where to load `.wasm` and `.mjs` glue files from). Excluding it
 *   keeps the original ESM imports intact so the resolution works.
 * - Cross-Origin headers are required for SharedArrayBuffer
 *   (multi-threaded WASM).
 * - Model files are served from the local `results/` directory via
 *   the `serveModelFiles` plugin (streaming, handles large files).
 */
export default defineConfig({
  plugins: [serveModelFiles()],

  server: {
    port: 5199,
    strictPort: true,
    headers: {
      // Required for SharedArrayBuffer (multi-threaded WASM)
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },

  optimizeDeps: {
    // Preserve import.meta.url resolution that ort uses for .wasm files
    exclude: ["onnxruntime-web"],
  },
});
