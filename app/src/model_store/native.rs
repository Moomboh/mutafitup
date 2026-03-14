/// Desktop model cache backed by the local filesystem.
///
/// Resolution order for the model source:
///
/// 1. `MUTAFITUP_MODELS_DIR` environment variable (explicit override)
/// 2. `../results/onnx_export` relative to the binary (development)
/// 3. HuggingFace CDN → downloaded to `~/.cache/mutafitup/models/`
use std::io::{Read, Write};
use std::path::Path;

use anyhow::Result;

use super::ProgressFn;

/// Root directory for all ONNX model exports (native desktop).
/// Override with the `MUTAFITUP_MODELS_DIR` environment variable.
const DEFAULT_MODELS_DIR: &str = "../results/onnx_export";

/// HuggingFace CDN URL for downloading models on native desktop.
///
/// Override at compile time via `MUTAFITUP_MODELS_URL`.
const DEFAULT_HF_MODELS_URL: &str = match option_env!("MUTAFITUP_MODELS_URL") {
    Some(url) => url,
    None => "https://huggingface.co/Moomboh/ESMC-300M-mutafitup/resolve/main",
};

/// Where models come from.
pub enum ModelSource {
    /// Models are available in a local directory (dev or explicit override).
    Local(String),
    /// Models are downloaded from HuggingFace and cached locally.
    HuggingFace { base_url: String, cache_dir: String },
}

/// Determine the model source.
///
/// Set `MUTAFITUP_FORCE_HF=1` to skip local sources and always
/// download from HuggingFace (useful for testing HF integration).
pub fn resolve_source() -> Result<ModelSource> {
    // 0. Force HuggingFace mode
    if std::env::var("MUTAFITUP_FORCE_HF").is_ok() {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
            .join("mutafitup")
            .join("models");
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| anyhow::anyhow!("Failed to create cache dir: {e}"))?;
        eprintln!(
            "[mutafitup] Forced HuggingFace mode (cache: {})",
            cache_dir.display()
        );
        return Ok(ModelSource::HuggingFace {
            base_url: DEFAULT_HF_MODELS_URL.to_string(),
            cache_dir: cache_dir.to_string_lossy().to_string(),
        });
    }

    // 1. Explicit override via env var
    if let Ok(dir) = std::env::var("MUTAFITUP_MODELS_DIR") {
        let path = Path::new(&dir);
        if path.join("models.json").exists() {
            eprintln!("[mutafitup] Using models from MUTAFITUP_MODELS_DIR: {dir}");
            return Ok(ModelSource::Local(dir));
        }
        eprintln!("[mutafitup] Warning: MUTAFITUP_MODELS_DIR={dir} does not contain models.json");
    }

    // 2. Local development directory
    let default_path = Path::new(DEFAULT_MODELS_DIR);
    if default_path.join("models.json").exists() {
        eprintln!(
            "[mutafitup] Using models from local directory: {}",
            DEFAULT_MODELS_DIR
        );
        return Ok(ModelSource::Local(DEFAULT_MODELS_DIR.to_string()));
    }

    // 3. HuggingFace cache
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
        .join("mutafitup")
        .join("models");
    std::fs::create_dir_all(&cache_dir)
        .map_err(|e| anyhow::anyhow!("Failed to create cache dir: {e}"))?;
    eprintln!(
        "[mutafitup] Using HuggingFace models (cache: {})",
        cache_dir.display()
    );
    Ok(ModelSource::HuggingFace {
        base_url: DEFAULT_HF_MODELS_URL.to_string(),
        cache_dir: cache_dir.to_string_lossy().to_string(),
    })
}

/// Fetch the model manifest JSON string.
///
/// For local sources, reads from disk. For HuggingFace sources,
/// always re-fetches from the network (to pick up new models) and
/// caches the result locally.
pub fn fetch_manifest(source: &ModelSource) -> Result<String> {
    match source {
        ModelSource::Local(dir) => {
            let manifest_path = format!("{dir}/models.json");
            std::fs::read_to_string(&manifest_path)
                .map_err(|e| anyhow::anyhow!("Failed to read {manifest_path}: {e}"))
        }
        ModelSource::HuggingFace {
            base_url,
            cache_dir,
        } => {
            let url = format!("{base_url}/models.json");
            let resp = ureq::get(&url)
                .call()
                .map_err(|e| anyhow::anyhow!("Failed to fetch {url}: {e}"))?;
            let json = resp
                .into_body()
                .read_to_string()
                .map_err(|e| anyhow::anyhow!("Failed to read manifest response: {e}"))?;
            // Cache the manifest locally
            let manifest_path = Path::new(cache_dir).join("models.json");
            let _ = std::fs::write(&manifest_path, &json);
            Ok(json)
        }
    }
}

/// Ensure all model files are cached locally and return the directory path.
///
/// For local sources, simply returns the subdirectory. For HuggingFace
/// sources, downloads any missing files to the cache.
pub fn ensure_model_dir(
    source: &ModelSource,
    model_id: &str,
    on_progress: &ProgressFn,
) -> Result<String> {
    match source {
        ModelSource::Local(dir) => Ok(format!("{dir}/{model_id}")),
        ModelSource::HuggingFace {
            base_url,
            cache_dir,
        } => ensure_model_cached(base_url, cache_dir, model_id, on_progress),
    }
}

// ── Internal helpers ─────────────────────────────────────────────────

/// Download a file from a URL to a local path with chunked progress.
/// Skips download if the file already exists.
fn download_file(url: &str, dest: &Path, on_progress: &ProgressFn) -> Result<()> {
    if dest.exists() {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("Failed to create dir {}: {e}", parent.display()))?;
    }
    let label = dest.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    eprintln!("[mutafitup] Downloading {url}");
    let resp = ureq::get(url)
        .call()
        .map_err(|e| anyhow::anyhow!("Failed to download {url}: {e}"))?;
    let total: Option<u64> = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok()?.parse().ok());
    let mut reader = resp.into_body().into_reader();
    let tmp = format!("{}.tmp", dest.display());
    {
        let mut file = std::fs::File::create(&tmp)
            .map_err(|e| anyhow::anyhow!("Failed to create {tmp}: {e}"))?;
        let mut received: u64 = 0;
        let mut buf = [0u8; 256 * 1024]; // 256 KB chunks
        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| anyhow::anyhow!("Read error: {e}"))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])
                .map_err(|e| anyhow::anyhow!("Write error: {e}"))?;
            received += n as u64;
            on_progress(label, received, total);
        }
    }
    std::fs::rename(&tmp, dest)
        .map_err(|e| anyhow::anyhow!("Failed to rename {tmp} -> {}: {e}", dest.display()))?;
    Ok(())
}

/// Download all files needed to load a model from HuggingFace.
fn ensure_model_cached(
    base_url: &str,
    cache_dir: &str,
    model_id: &str,
    on_progress: &ProgressFn,
) -> Result<String> {
    let model_cache = Path::new(cache_dir).join(model_id);
    let files = [
        "export_metadata.json",
        "normalization_stats.json",
        "tokenizer/tokenizer.json",
        "model.onnx", // largest file last for best UX
    ];
    for file in &files {
        let url = format!("{base_url}/{model_id}/{file}");
        let dest = model_cache.join(file);
        // normalization_stats.json is optional — skip on 404
        if *file == "normalization_stats.json" {
            if !dest.exists() {
                match ureq::get(&url).call() {
                    Ok(resp) => {
                        if let Some(parent) = dest.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        let mut reader = resp.into_body().into_reader();
                        let tmp = format!("{}.tmp", dest.display());
                        if let Ok(mut f) = std::fs::File::create(&tmp) {
                            let _ = std::io::copy(&mut reader, &mut f);
                            let _ = std::fs::rename(&tmp, &dest);
                        }
                    }
                    Err(_) => {
                        eprintln!("[mutafitup] normalization_stats.json not found (optional)");
                    }
                }
            }
            continue;
        }
        download_file(&url, &dest, on_progress)?;
    }
    Ok(model_cache.to_string_lossy().to_string())
}
