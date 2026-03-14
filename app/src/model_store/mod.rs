/// Platform-abstracted model cache.
///
/// On desktop, models are cached in the filesystem under
/// `~/.cache/mutafitup/models/`. On web, models are persisted in the
/// Origin Private File System (OPFS). Both backends expose the same
/// high-level operations: fetch manifest, ensure model files are
/// available, and return bytes or paths for the inference backend.

#[cfg(not(target_arch = "wasm32"))]
pub mod native;

#[cfg(target_arch = "wasm32")]
pub mod web;

/// Progress callback signature: `(phase_label, bytes_received, total_bytes)`.
///
/// `total_bytes` is `None` when the server does not send `Content-Length`.
pub type ProgressFn = dyn Fn(&str, u64, Option<u64>);

/// Files required to load a model on web (all held in memory).
///
/// Desktop does not use this — it passes a directory path to the
/// worker process instead.
#[cfg(target_arch = "wasm32")]
pub struct ModelFiles {
    pub metadata_json: String,
    pub normalization_stats_json: Option<String>,
    pub tokenizer_bytes: Vec<u8>,
    pub model_bytes: Vec<u8>,
}
