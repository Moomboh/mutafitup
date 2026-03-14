/// Browser model cache backed by the Origin Private File System (OPFS).
///
/// Files are persisted across page reloads via `navigator.storage.getDirectory()`.
/// If OPFS is unavailable the store falls back to transient in-memory downloads
/// (the model still works for that session, it just re-downloads on next visit).
use anyhow::{Context, Result};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use super::ProgressFn;
pub use super::ModelFiles;

/// Default HuggingFace CDN URL for model downloads (web).
///
/// Override at compile time via `MUTAFITUP_MODELS_URL`. Falls back to
/// `"/__models__"` for local development (proxied via Dioxus.toml).
pub const DEFAULT_MODELS_URL: &str = match option_env!("MUTAFITUP_MODELS_URL") {
    Some(url) => url,
    None => "/__models__",
};

// ── Public API ───────────────────────────────────────────────────────

/// Fetch the model manifest JSON string.
///
/// Always fetches from the network and caches the result in OPFS.
pub async fn fetch_manifest(base_url: &str) -> Result<String> {
    let url = format!("{base_url}/models.json");
    let json = fetch_text(&url).await?;
    // Best-effort cache in OPFS
    let _ = opfs_write("models.json", json.as_bytes()).await;
    Ok(json)
}

/// Ensure all model files are available and return them as bytes.
///
/// Checks OPFS first; downloads and persists any missing files.
pub async fn ensure_model_files(
    base_url: &str,
    model_id: &str,
    on_progress: &ProgressFn,
) -> Result<ModelFiles> {
    let base = base_url.trim_end_matches('/');
    let prefix = opfs_model_prefix(model_id);

    // metadata (small — only report on cache miss)
    let metadata_json = ensure_text_cached(
        &format!("{base}/{model_id}/export_metadata.json"),
        &format!("{prefix}/export_metadata.json"),
        Some(("Downloading metadata...", on_progress)),
    )
    .await?;

    // normalization stats (optional, small)
    let norm_key = format!("{prefix}/normalization_stats.json");
    let normalization_stats_json = match opfs_read_string(&norm_key).await {
        Some(s) => Some(s),
        None => {
            let url = format!("{base}/{model_id}/normalization_stats.json");
            match fetch_text_optional(&url).await {
                Some(json) => {
                    let _ = opfs_write(&norm_key, json.as_bytes()).await;
                    Some(json)
                }
                None => None,
            }
        }
    };

    // tokenizer (small — only report on cache miss)
    let tokenizer_bytes = ensure_bytes_cached(
        &format!("{base}/{model_id}/tokenizer/tokenizer.json"),
        &format!("{prefix}/tokenizer.json"),
        Some(("Downloading tokenizer...", on_progress)),
    )
    .await?;

    // model.onnx (large — stream with progress)
    let model_key = format!("{prefix}/model.onnx");
    let model_bytes = match opfs_read_bytes(&model_key).await {
        Some(bytes) => {
            on_progress("Loading from cache...", bytes.len() as u64, Some(bytes.len() as u64));
            console_log(&format!(
                "[mutafitup] Loaded model.onnx from OPFS cache ({} bytes)",
                bytes.len()
            ));
            bytes
        }
        None => {
            let url = format!("{base}/{model_id}/model.onnx");
            console_log(&format!("[mutafitup] Downloading model from {url}..."));
            let bytes = fetch_bytes_with_progress(&url, &|done, total| {
                on_progress("Downloading model...", done, total);
            })
            .await?;
            console_log(&format!(
                "[mutafitup] Model downloaded ({} bytes), caching in OPFS...",
                bytes.len()
            ));
            on_progress("Caching model...", 0, None);
            let _ = opfs_write(&model_key, &bytes).await;
            bytes
        }
    };

    Ok(ModelFiles {
        metadata_json,
        normalization_stats_json,
        tokenizer_bytes,
        model_bytes,
    })
}

// ── Fetch helpers ────────────────────────────────────────────────────

/// Fetch a URL and return the response body as a String.
pub async fn fetch_text(url: &str) -> Result<String> {
    let window = web_sys::window().context("No window object")?;
    let resp_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|e| anyhow::anyhow!("fetch failed: {e:?}"))?;
    let resp: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("Response cast failed"))?;
    if !resp.ok() {
        anyhow::bail!("HTTP {} {} for {}", resp.status(), resp.status_text(), url);
    }
    let text = JsFuture::from(
        resp.text()
            .map_err(|e| anyhow::anyhow!("text() failed: {e:?}"))?,
    )
    .await
    .map_err(|e| anyhow::anyhow!("text() promise failed: {e:?}"))?;
    text.as_string().context("Response body is not a string")
}

/// Fetch a URL and return the response body as a String, or `None` on 404.
fn fetch_text_optional(url: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + '_>> {
    Box::pin(async move {
        let window = web_sys::window()?;
        let resp_value = JsFuture::from(window.fetch_with_str(url)).await.ok()?;
        let resp: web_sys::Response = resp_value.dyn_into().ok()?;
        if resp.status() == 404 {
            return None;
        }
        if !resp.ok() {
            return None;
        }
        let text = JsFuture::from(resp.text().ok()?).await.ok()?;
        text.as_string()
    })
}

/// Fetch a URL with streaming progress and return the response as bytes.
pub async fn fetch_bytes_with_progress(
    url: &str,
    on_progress: &dyn Fn(u64, Option<u64>),
) -> Result<Vec<u8>> {
    let window = web_sys::window().context("No window object")?;
    let resp_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|e| anyhow::anyhow!("fetch failed: {e:?}"))?;
    let resp: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("Response cast failed"))?;
    if !resp.ok() {
        anyhow::bail!("HTTP {} {} for {}", resp.status(), resp.status_text(), url);
    }

    let total: Option<u64> = resp
        .headers()
        .get("content-length")
        .ok()
        .flatten()
        .and_then(|s| s.parse().ok());

    let body = resp.body().context("Response has no body")?;
    let reader: web_sys::ReadableStreamDefaultReader = body
        .get_reader()
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("Failed to get stream reader"))?;

    let capacity = total.unwrap_or(1024 * 1024) as usize;
    let mut buf = Vec::with_capacity(capacity);
    let mut received: u64 = 0;

    loop {
        let chunk = JsFuture::from(reader.read())
            .await
            .map_err(|e| anyhow::anyhow!("stream read failed: {e:?}"))?;
        let done = js_sys::Reflect::get(&chunk, &"done".into())
            .unwrap_or(true.into())
            .as_bool()
            .unwrap_or(true);
        if done {
            break;
        }
        let value = js_sys::Reflect::get(&chunk, &"value".into())
            .map_err(|_| anyhow::anyhow!("Missing value in stream chunk"))?;
        let array = js_sys::Uint8Array::new(&value);
        let len = array.length() as usize;
        let offset = buf.len();
        buf.resize(offset + len, 0);
        array.copy_to(&mut buf[offset..]);
        received += len as u64;
        on_progress(received, total);
    }

    Ok(buf)
}

// ── OPFS helpers ─────────────────────────────────────────────────────

fn console_log(msg: &str) {
    web_sys::console::log_1(&msg.into());
}

/// Flat key for a model file inside OPFS, e.g.
/// `"accgrad_lora__esmc_300m_all_r4__best_overall"`.
fn opfs_model_prefix(model_id: &str) -> String {
    model_id.replace('/', "__")
}

/// Get the OPFS root directory handle.
///
/// Returns `None` if the browser does not support OPFS.
async fn opfs_root() -> Option<web_sys::FileSystemDirectoryHandle> {
    let window = web_sys::window()?;
    let navigator = window.navigator();
    let storage = navigator.storage();
    let promise = web_sys::StorageManager::get_directory(&storage);
    let dir = JsFuture::from(promise).await.ok()?;
    dir.dyn_into().ok()
}

/// Get or create a sub-directory inside an OPFS directory handle.
async fn opfs_subdir(
    parent: &web_sys::FileSystemDirectoryHandle,
    name: &str,
) -> Option<web_sys::FileSystemDirectoryHandle> {
    let opts = web_sys::FileSystemGetDirectoryOptions::new();
    opts.set_create(true);
    let promise = parent.get_directory_handle_with_options(name, &opts);
    let dir = JsFuture::from(promise).await.ok()?;
    dir.dyn_into().ok()
}

/// Resolve a `/`-separated key to a `(directory_handle, file_name)` pair,
/// creating intermediate directories as needed.
///
/// E.g. `"accgrad_lora__esmc_300m_all_r4__best_overall/model.onnx"` →
/// subdirectory `accgrad_lora__esmc_300m_all_r4__best_overall`, file `model.onnx`.
async fn opfs_resolve_path(
    key: &str,
) -> Option<(web_sys::FileSystemDirectoryHandle, String)> {
    let root = opfs_root().await?;
    let app_dir = opfs_subdir(&root, "mutafitup-models").await?;

    let parts: Vec<&str> = key.split('/').collect();
    if parts.is_empty() {
        return None;
    }
    let (dir_parts, file_name) = parts.split_at(parts.len() - 1);
    let file_name = file_name[0].to_string();

    let mut current = app_dir;
    for part in dir_parts {
        current = opfs_subdir(&current, part).await?;
    }
    Some((current, file_name))
}

/// Read a file from OPFS and return its bytes, or `None` if it does not exist.
async fn opfs_read_bytes(key: &str) -> Option<Vec<u8>> {
    let (dir, name) = opfs_resolve_path(key).await?;
    // Try to get the file handle (without create) — returns an error if missing
    let promise = dir.get_file_handle(&name);
    let handle: web_sys::FileSystemFileHandle =
        JsFuture::from(promise).await.ok()?.dyn_into().ok()?;
    let file: web_sys::File = JsFuture::from(
        web_sys::FileSystemFileHandle::get_file(&handle),
    )
    .await
    .ok()?
    .dyn_into()
    .ok()?;
    let ab = JsFuture::from(file.array_buffer()).await.ok()?;
    let arr = js_sys::Uint8Array::new(&ab);
    Some(arr.to_vec())
}

/// Read a file from OPFS and return it as a UTF-8 string, or `None`.
async fn opfs_read_string(key: &str) -> Option<String> {
    let bytes = opfs_read_bytes(key).await?;
    String::from_utf8(bytes).ok()
}

/// Write bytes to a file in OPFS, creating directories as needed.
///
/// Returns `Ok(())` on success, `Err` if OPFS is unavailable or write fails.
async fn opfs_write(key: &str, data: &[u8]) -> Result<()> {
    let (dir, name) = opfs_resolve_path(key)
        .await
        .context("OPFS not available")?;
    let opts = web_sys::FileSystemGetFileOptions::new();
    opts.set_create(true);
    let handle: web_sys::FileSystemFileHandle = JsFuture::from(
        dir.get_file_handle_with_options(&name, &opts),
    )
    .await
    .map_err(|e| anyhow::anyhow!("getFileHandle failed: {e:?}"))?
    .dyn_into()
    .map_err(|_| anyhow::anyhow!("FileSystemFileHandle cast failed"))?;

    let writable: web_sys::FileSystemWritableFileStream = JsFuture::from(
        web_sys::FileSystemFileHandle::create_writable(&handle),
    )
    .await
    .map_err(|e| anyhow::anyhow!("createWritable failed: {e:?}"))?
    .dyn_into()
    .map_err(|_| anyhow::anyhow!("WritableFileStream cast failed"))?;

    let uint8 = js_sys::Uint8Array::from(data);
    JsFuture::from(
        writable
            .write_with_buffer_source(&uint8)
            .map_err(|e| anyhow::anyhow!("write failed: {e:?}"))?,
    )
    .await
    .map_err(|e| anyhow::anyhow!("write promise failed: {e:?}"))?;

    JsFuture::from(writable.close())
        .await
        .map_err(|e| anyhow::anyhow!("close failed: {e:?}"))?;

    Ok(())
}

// ── Cache-first helpers ──────────────────────────────────────────────

/// Return a text file from OPFS cache, or download and cache it.
///
/// If `on_miss` is provided, the progress callback is invoked only when
/// the file is not in the cache (i.e. a network download is needed).
async fn ensure_text_cached(
    url: &str,
    key: &str,
    on_miss: Option<(&str, &ProgressFn)>,
) -> Result<String> {
    if let Some(cached) = opfs_read_string(key).await {
        console_log(&format!("[mutafitup] Cache hit: {key}"));
        return Ok(cached);
    }
    console_log(&format!("[mutafitup] Cache miss: {key}, fetching {url}"));
    if let Some((label, progress)) = on_miss {
        progress(label, 0, None);
    }
    let text = fetch_text(url).await?;
    let _ = opfs_write(key, text.as_bytes()).await;
    Ok(text)
}

/// Return a binary file from OPFS cache, or download and cache it.
///
/// If `on_miss` is provided, the progress callback is invoked only when
/// the file is not in the cache (i.e. a network download is needed).
async fn ensure_bytes_cached(
    url: &str,
    key: &str,
    on_miss: Option<(&str, &ProgressFn)>,
) -> Result<Vec<u8>> {
    if let Some(cached) = opfs_read_bytes(key).await {
        console_log(&format!("[mutafitup] Cache hit: {key}"));
        return Ok(cached);
    }
    console_log(&format!("[mutafitup] Cache miss: {key}, fetching {url}"));
    if let Some((label, progress)) = on_miss {
        progress(label, 0, None);
    }
    let bytes = fetch_bytes_with_progress(url, &|_, _| {}).await?;
    let _ = opfs_write(key, &bytes).await;
    Ok(bytes)
}
