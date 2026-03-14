/// Browser backend using `ort-web` (onnxruntime-web via wasm-bindgen).
///
/// Loads a model from pre-fetched bytes (provided by `model_store::web`)
/// and runs async inference via `Session::run_async` with WebNN/WebGPU/WASM
/// execution providers.
use anyhow::{Context, Result};
use ort::ep;
use ort::session::{RunOptions, Session};
use ort::value::Tensor;
use tokenizers::tokenizer::Tokenizer;
use wasm_bindgen_futures::JsFuture;

use super::preprocessing::preprocess_sequences;
use super::tokenizer::{encode_batch, load_tokenizer_from_bytes};
use super::{
    denormalize_predictions, split_batch_predictions, EpChoice, ExportMetadata,
    NormalizationStats, Predictions,
};

use crate::model_store::ModelFiles;

/// URL prefix where self-hosted onnxruntime-web JS/WASM files are served.
/// Proxied via `[[web.proxy]]` in Dioxus.toml to a local miniserve instance.
const ORT_RUNTIME_URL: &str = "/__ort__/";

pub struct WebBackend {
    session: Session,
    tokenizer: Tokenizer,
    metadata: ExportMetadata,
    norm_stats: Option<NormalizationStats>,
}

// WASM is single-threaded, so Send/Sync are sound.
unsafe impl Send for WebBackend {}
unsafe impl Sync for WebBackend {}

/// Log a message to the browser console.
fn console_log(msg: &str) {
    web_sys::console::log_1(&msg.into());
}

/// Detect which ONNX Runtime backend will be used.
///
/// Mirrors the checks in onnxruntime-web's WebGPU EP initialization:
/// 1. `navigator.gpu` must exist
/// 2. `navigator.gpu.requestAdapter()` must return a non-null adapter
///
/// Returns a human-readable backend name (e.g. "WebGPU", "WASM (CPU)").
async fn detect_backend() -> String {
    use wasm_bindgen::JsCast;

    let Some(window) = web_sys::window() else {
        return "WASM (CPU)".into();
    };
    let navigator = window.navigator();
    let has_gpu_api = js_sys::Reflect::has(&navigator, &"gpu".into()).unwrap_or(false);
    if !has_gpu_api {
        return "WASM (CPU)".into();
    }
    let gpu = match js_sys::Reflect::get(&navigator, &"gpu".into()) {
        Ok(gpu) if !gpu.is_undefined() && !gpu.is_null() => gpu,
        _ => return "WASM (CPU)".into(),
    };
    let request_adapter = match js_sys::Reflect::get(&gpu, &"requestAdapter".into()) {
        Ok(f) if f.is_function() => js_sys::Function::from(f),
        _ => return "WASM (CPU)".into(),
    };
    let promise = match request_adapter.call0(&gpu) {
        Ok(p) => p,
        Err(_) => return "WASM (CPU)".into(),
    };
    match JsFuture::from(js_sys::Promise::from(promise)).await {
        Ok(adapter) if !adapter.is_null() && !adapter.is_undefined() => "WebGPU".into(),
        _ => "WASM (CPU)".into(),
    }
}

/// Probe which execution providers are available in the browser.
/// Returns them in priority order (best first), always ending with CPU.
pub async fn available_eps() -> Vec<EpChoice> {
    let mut eps = Vec::new();
    let backend = detect_backend().await;
    if backend == "WebGPU" {
        eps.push(EpChoice::WebGPU);
    }
    eps.push(EpChoice::Cpu);
    eps
}

impl WebBackend {
    /// Create a WebBackend from pre-fetched model files.
    ///
    /// The ORT runtime is initialized on the first call; model bytes
    /// are loaded into the session via `commit_from_memory`.
    pub async fn load_from_files(
        files: ModelFiles,
        ep_choice: &EpChoice,
    ) -> Result<Self> {
        // Load onnxruntime-web from self-hosted files (WebGPU-capable build
        // includes WebNN + WASM CPU support).
        console_log("[mutafitup] Loading onnxruntime-web runtime...");
        let origin = web_sys::window()
            .context("No window object")?
            .location()
            .origin()
            .map_err(|_| anyhow::anyhow!("Failed to get window origin"))?;
        let ort_base_url = format!("{origin}{ORT_RUNTIME_URL}");
        let dist = ort_web::Dist::new(&ort_base_url)
            .with_script_name("ort.webgpu.min.js")
            .with_binary_name("ort-wasm-simd-threaded.jsep.wasm");
        let api = ort_web::api(dist)
            .await
            .map_err(|e| anyhow::anyhow!("ort-web init failed: {e}"))?;
        ort::set_api(api);
        console_log("[mutafitup] onnxruntime-web runtime loaded");

        console_log(&format!("[mutafitup] Requested EP: {ep_choice}"));

        // Parse metadata
        let metadata: ExportMetadata = serde_json::from_str(&files.metadata_json)
            .context("Failed to parse export_metadata.json")?;

        // Parse normalization stats (optional)
        let norm_stats = match files.normalization_stats_json {
            Some(json) => {
                let stats: NormalizationStats = serde_json::from_str(&json)
                    .context("Failed to parse normalization_stats.json")?;
                console_log(&format!(
                    "[mutafitup] Loaded normalization stats for {} task(s)",
                    stats.len()
                ));
                Some(stats)
            }
            None => {
                console_log(
                    "[mutafitup] No normalization_stats.json found, predictions will be raw normalized values",
                );
                None
            }
        };

        // Load tokenizer
        let tokenizer = load_tokenizer_from_bytes(&files.tokenizer_bytes)?;

        // Create ORT session from downloaded bytes
        console_log("[mutafitup] Creating ONNX Runtime session...");
        let session = match ep_choice {
            EpChoice::WebGPU => Session::builder()
                .context("Failed to create session builder")?
                .with_execution_providers([ep::WebGPU::default().build()])
                .map_err(|e| anyhow::anyhow!("Failed to configure execution providers: {e}"))?
                .commit_from_memory(&files.model_bytes)
                .await
                .context("Failed to create ONNX session")?,
            _ => Session::builder()
                .context("Failed to create session builder")?
                .commit_from_memory(&files.model_bytes)
                .await
                .context("Failed to create ONNX session")?,
        };
        console_log("[mutafitup] Model loaded successfully");

        Ok(Self {
            session,
            tokenizer,
            metadata,
            norm_stats,
        })
    }

    /// Get the export metadata.
    pub fn metadata(&self) -> &ExportMetadata {
        &self.metadata
    }

    /// Run inference on one or more raw protein sequences.
    ///
    /// Retries once on "detached ArrayBuffer" errors, which can occur when
    /// the WebGPU EP's internal buffer pool races with WASM memory growth.
    pub async fn predict(&mut self, sequences: &[String]) -> Result<Vec<Predictions>> {
        match self.predict_inner(sequences).await {
            Err(e) if e.to_string().contains("detached ArrayBuffer") => {
                console_log(
                    "[mutafitup] Detached ArrayBuffer error, retrying after delay...",
                );
                gloo_timers::future::TimeoutFuture::new(50).await;
                self.predict_inner(sequences).await
            }
            result => result,
        }
    }

    /// Inner prediction logic. Separated so `predict` can retry on transient
    /// WebGPU buffer errors.
    async fn predict_inner(&mut self, sequences: &[String]) -> Result<Vec<Predictions>> {
        let preprocessed = preprocess_sequences(sequences, &self.metadata.preprocessing);

        let tok_output = encode_batch(&self.tokenizer, &preprocessed)?;
        let batch_size = tok_output.batch_size;
        let seq_len = tok_output.seq_len;

        // Force any needed WASM memory growth *before* creating tensor views.
        // ort-web's Tensor::from_ptr uses js_sys::*Array::view() which
        // references the current WebAssembly.Memory buffer. If memory.grow()
        // is triggered between view creation and the JS ort.Tensor constructor
        // consuming it, the old ArrayBuffer is detached and the constructor
        // throws "Cannot perform Construct on a detached ArrayBuffer".
        let tensor_bytes = batch_size * seq_len * 8 * 2; // two i64 tensors
        let _pre_grow = vec![0u8; tensor_bytes + 1024 * 1024];
        drop(_pre_grow);

        let input_ids = Tensor::from_array((
            [batch_size, seq_len],
            tok_output.input_ids.into_boxed_slice(),
        ))
        .context("Failed to create input_ids tensor")?;

        let attention_mask = Tensor::from_array((
            [batch_size, seq_len],
            tok_output.attention_mask.into_boxed_slice(),
        ))
        .context("Failed to create attention_mask tensor")?;

        console_log(&format!(
            "[mutafitup] Running inference (batch={batch_size}, seq_len={seq_len})...",
        ));
        let start = js_sys::Date::now();

        let run_options = RunOptions::new().context("Failed to create RunOptions")?;
        let mut outputs = self
            .session
            .run_async(
                ort::inputs! {
                    "input_ids" => input_ids,
                    "attention_mask" => attention_mask,
                },
                &run_options,
            )
            .await
            .context("ONNX async inference failed")?;

        // ort-web requires explicit output synchronization
        ort_web::sync_outputs(&mut outputs)
            .await
            .map_err(|e| anyhow::anyhow!("sync_outputs failed: {e}"))?;

        let elapsed = js_sys::Date::now() - start;
        console_log(&format!(
            "[mutafitup] Inference completed in {elapsed:.0}ms"
        ));

        let mut flat_predictions = Predictions::new();
        for (task_name, task_config) in &self.metadata.tasks {
            if let Some(output_tensor) = outputs.get(task_config.output_name.as_str()) {
                let (_shape, data): (&ort::value::Shape, &[f32]) = output_tensor
                    .try_extract_tensor::<f32>()
                    .with_context(|| format!("Failed to extract output for task {task_name}"))?;
                flat_predictions.insert(task_name.clone(), data.to_vec());
            }
        }

        // Release GPU buffer references before the next inference call.
        // The microtask yield lets the browser GC reclaim WebGPU buffers
        // that onnxruntime-web's buffer pool may otherwise race with.
        drop(outputs);
        gloo_timers::future::TimeoutFuture::new(0).await;

        if let Some(ref stats) = self.norm_stats {
            denormalize_predictions(&mut flat_predictions, stats);
        }

        Ok(split_batch_predictions(
            &flat_predictions,
            &self.metadata,
            batch_size,
            seq_len,
        ))
    }
}
