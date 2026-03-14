/// Native desktop backend using `ort` (ONNX Runtime).
///
/// ONNX inference runs in a **separate child process** to fully isolate
/// CoreML EP's process-global state. The CoreML execution provider
/// segfaults when a second `Session` is created after the first is
/// destroyed — even sequentially on the same thread. By running each
/// model in its own process and killing the process on model switch,
/// the OS fully reclaims all CoreML resources.
///
/// The same binary is reused for the worker: `main()` checks for the
/// `--worker` flag and calls [`worker_main`] instead of launching Dioxus.
///
/// IPC uses bincode over stdin/stdout pipes (length-prefixed framing).
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;

use anyhow::{Context, Result};
use ort::ep::ExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::Tokenizer;

use super::preprocessing::preprocess_sequences;
use super::tokenizer::{encode_batch, load_tokenizer_from_file};
use super::{
    denormalize_predictions, split_batch_predictions, EpChoice, ExportMetadata, NormalizationStats,
    Predictions,
};

// ── IPC protocol types ──

#[derive(Serialize, Deserialize)]
enum WorkerRequest {
    Load { model_dir: String, ep: EpChoice },
    Predict { sequences: Vec<String> },
}

#[derive(Serialize, Deserialize)]
enum WorkerResponse {
    Loaded(Result<ModelInfo, String>),
    Predicted(Result<Vec<Predictions>, String>),
}

/// Metadata snapshot returned after a successful load so the UI can
/// display model info without reaching into the backend.
#[derive(Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub metadata: ExportMetadata,
}

// ── IPC framing: [4-byte LE length][bincode payload] ──

fn write_message<W: Write>(writer: &mut W, msg: &impl Serialize) -> io::Result<()> {
    let payload =
        bincode::serialize(msg).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let len = payload.len() as u32;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()
}

fn read_message<R: Read, T: serde::de::DeserializeOwned>(reader: &mut R) -> io::Result<T> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    bincode::deserialize(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

// ══════════════════════════════════════════════════════════════════════
//  Worker process (runs when the binary is invoked with `--worker`)
// ══════════════════════════════════════════════════════════════════════

/// Internal backend that lives exclusively in the worker process.
struct NativeBackend {
    session: Session,
    tokenizer: Tokenizer,
    metadata: ExportMetadata,
    norm_stats: Option<NormalizationStats>,
}

unsafe impl Send for NativeBackend {}

/// Probe which ONNX Runtime execution providers are available on this
/// platform. Returns them in priority order (best first), always
/// ending with CPU. Safe to call from the parent process — does not
/// create any ONNX sessions.
pub fn available_eps() -> Vec<EpChoice> {
    let mut eps = Vec::new();

    let coreml = ort::ep::CoreML::default();
    if coreml.supported_by_platform() {
        if let Ok(true) = coreml.is_available() {
            eprintln!("[mutafitup] CoreML EP available");
            eps.push(EpChoice::CoreML);
        }
    }

    let directml = ort::ep::DirectML::default();
    if directml.supported_by_platform() {
        if let Ok(true) = directml.is_available() {
            eprintln!("[mutafitup] DirectML EP available");
            eps.push(EpChoice::DirectML);
        }
    }

    eps.push(EpChoice::Cpu);
    eps
}

/// Build a platform-appropriate cache directory path for compiled CoreML
/// models. Returns `None` if the home directory can't be determined.
///
/// Layout: `~/.cache/mutafitup/coreml/<sanitized_export_dir>/`
/// where slashes in `export_dir` become underscores.
fn coreml_cache_path(export_dir: &str) -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let sanitized = export_dir.trim_matches('/').replace('/', "_");
    Some(
        Path::new(&home)
            .join(".cache")
            .join("mutafitup")
            .join("coreml")
            .join(sanitized),
    )
}

impl NativeBackend {
    fn load(export_dir: &str, ep: &EpChoice) -> Result<Self> {
        let base = Path::new(export_dir);

        let metadata_path = base.join("export_metadata.json");
        let metadata_json = std::fs::read_to_string(&metadata_path)
            .with_context(|| format!("Failed to read {}", metadata_path.display()))?;
        let metadata: ExportMetadata =
            serde_json::from_str(&metadata_json).context("Failed to parse export_metadata.json")?;

        // Normalization stats are co-located with the model.
        let norm_stats_path = base.join("normalization_stats.json");

        let norm_stats = if norm_stats_path.exists() {
            let json = std::fs::read_to_string(&norm_stats_path)
                .with_context(|| format!("Failed to read {}", norm_stats_path.display()))?;
            let stats: NormalizationStats =
                serde_json::from_str(&json).context("Failed to parse normalization_stats.json")?;
            eprintln!(
                "[mutafitup-worker] Loaded normalization stats from {}",
                norm_stats_path.display()
            );
            Some(stats)
        } else {
            eprintln!("[mutafitup-worker] No normalization_stats.json found, predictions will be raw normalized values");
            None
        };

        let tokenizer_path = base.join("tokenizer").join("tokenizer.json");
        let tokenizer =
            load_tokenizer_from_file(tokenizer_path.to_str().context("Non-UTF8 tokenizer path")?)?;

        let model_path = base.join("model.onnx");
        let backend_name = ep.to_string();
        eprintln!("[mutafitup-worker] Requested EP: {backend_name}");
        eprintln!(
            "[mutafitup-worker] Creating session from {}",
            model_path.display()
        );

        let t0 = std::time::Instant::now();
        let session = match ep {
            EpChoice::CoreML => {
                // CoreML cache directory: reuse compiled CoreML models across
                // session loads to avoid re-compiling on every process spawn.
                let coreml_cache_dir = coreml_cache_path(export_dir);
                if let Some(ref dir) = coreml_cache_dir {
                    if let Err(e) = std::fs::create_dir_all(dir) {
                        eprintln!(
                            "[mutafitup-worker] Warning: failed to create CoreML cache dir: {e}"
                        );
                    } else {
                        eprintln!("[mutafitup-worker] CoreML cache dir: {}", dir.display());
                    }
                }

                let mut coreml_ep = ort::ep::CoreML::default()
                    // MLProgram format (Core ML 5+, macOS 12+) stores weights
                    // in a separate file, supports LayerNormalization/Erf/Gelu,
                    // and avoids the 2 GB NeuralNetwork protobuf limit.
                    .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                    // Static shapes let the converter correctly infer
                    // intermediate tensor ranks during MLProgram compilation.
                    .with_static_input_shapes(true);

                if let Some(ref dir) = coreml_cache_dir {
                    coreml_ep = coreml_ep.with_model_cache_dir(dir.to_string_lossy());
                }

                Session::builder()
                    .context("Failed to create session builder")?
                    .with_execution_providers([coreml_ep.build()])
                    .map_err(|e| anyhow::anyhow!("Failed to configure execution providers: {e}"))?
                    .commit_from_file(&model_path)
                    .with_context(|| {
                        format!("Failed to load ONNX model from {}", model_path.display())
                    })?
            }
            EpChoice::DirectML => Session::builder()
                .context("Failed to create session builder")?
                .with_execution_providers([ort::ep::DirectML::default().build()])
                .map_err(|e| anyhow::anyhow!("Failed to configure execution providers: {e}"))?
                .commit_from_file(&model_path)
                .with_context(|| {
                    format!("Failed to load ONNX model from {}", model_path.display())
                })?,
            // CPU (or any unrecognised EP on native) — no EPs registered,
            // ORT uses the built-in CPU execution provider.
            _ => Session::builder()
                .context("Failed to create session builder")?
                .commit_from_file(&model_path)
                .with_context(|| {
                    format!("Failed to load ONNX model from {}", model_path.display())
                })?,
        };
        eprintln!(
            "[mutafitup-worker] Session created in {:.1}s ({backend_name})",
            t0.elapsed().as_secs_f64()
        );

        Ok(Self {
            session,
            tokenizer,
            metadata,
            norm_stats,
        })
    }

    fn predict(&mut self, sequences: &[String]) -> Result<Vec<Predictions>> {
        let preprocessed = preprocess_sequences(sequences, &self.metadata.preprocessing);

        let tok_output = encode_batch(&self.tokenizer, &preprocessed)?;
        let batch_size = tok_output.batch_size;
        let seq_len = tok_output.seq_len;

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

        let outputs = self
            .session
            .run(ort::inputs! {
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
            })
            .context("ONNX inference failed")?;

        let mut flat_predictions = Predictions::new();
        for (task_name, task_config) in &self.metadata.tasks {
            if let Some(output_tensor) = outputs.get(task_config.output_name.as_str()) {
                let (_, data) = output_tensor
                    .try_extract_tensor::<f32>()
                    .with_context(|| format!("Failed to extract output for task {task_name}"))?;
                flat_predictions.insert(task_name.clone(), data.to_vec());
            }
        }

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

/// Entry point for the worker process. Called from `main()` when
/// `--worker` is passed. Reads requests from stdin, processes them,
/// writes responses to stdout. Exits when stdin closes.
pub fn worker_main() {
    eprintln!(
        "[mutafitup-worker] Worker process started (pid {})",
        std::process::id()
    );

    let mut reader = BufReader::new(io::stdin().lock());
    let mut writer = BufWriter::new(io::stdout().lock());
    let mut backend: Option<NativeBackend> = None;

    loop {
        let request: WorkerRequest = match read_message(&mut reader) {
            Ok(req) => req,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                eprintln!("[mutafitup-worker] Parent closed stdin, exiting");
                break;
            }
            Err(e) => {
                eprintln!("[mutafitup-worker] Failed to read request: {e}");
                break;
            }
        };

        let response = match request {
            WorkerRequest::Load { model_dir, ep } => {
                eprintln!("[mutafitup-worker] Loading model from {model_dir} (ep={ep})");
                backend = None;
                match NativeBackend::load(&model_dir, &ep) {
                    Ok(b) => {
                        let info = ModelInfo {
                            metadata: b.metadata.clone(),
                        };
                        backend = Some(b);
                        WorkerResponse::Loaded(Ok(info))
                    }
                    Err(e) => WorkerResponse::Loaded(Err(format!("{e:#}"))),
                }
            }
            WorkerRequest::Predict { sequences } => match backend.as_mut() {
                Some(b) => match b.predict(&sequences) {
                    Ok(preds) => WorkerResponse::Predicted(Ok(preds)),
                    Err(e) => WorkerResponse::Predicted(Err(format!("{e:#}"))),
                },
                None => WorkerResponse::Predicted(Err("No model loaded".into())),
            },
        };

        if let Err(e) = write_message(&mut writer, &response) {
            eprintln!("[mutafitup-worker] Failed to write response: {e}");
            break;
        }
    }

    eprintln!("[mutafitup-worker] Worker process exiting");
}

// ══════════════════════════════════════════════════════════════════════
//  Parent-side handle (used by the Dioxus app)
// ══════════════════════════════════════════════════════════════════════

/// Commands sent from `NativeModelHandle` methods to the dispatcher thread.
enum DispatchCmd {
    /// Kill the current worker process, spawn a fresh one, and load a model.
    Load {
        model_dir: String,
        ep: EpChoice,
        reply: mpsc::SyncSender<Result<ModelInfo>>,
    },
    /// Send a predict request to the running worker process.
    Predict {
        sequences: Vec<String>,
        reply: mpsc::SyncSender<Result<Vec<Predictions>>>,
    },
}

/// Managed child process with typed IPC pipes.
struct WorkerProcess {
    child: Child,
    reader: BufReader<ChildStdout>,
    writer: BufWriter<ChildStdin>,
}

impl WorkerProcess {
    /// Spawn a new worker child process from the current executable.
    fn spawn_new() -> Result<Self> {
        let exe = std::env::current_exe().context("Failed to determine current executable path")?;
        eprintln!("[mutafitup] Spawning worker process: {}", exe.display());

        let mut child = Command::new(&exe)
            .arg("--worker")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("Failed to spawn worker process: {}", exe.display()))?;

        let stdin = child.stdin.take().context("Worker stdin not captured")?;
        let stdout = child.stdout.take().context("Worker stdout not captured")?;

        Ok(Self {
            child,
            reader: BufReader::new(stdout),
            writer: BufWriter::new(stdin),
        })
    }

    fn send_request(&mut self, req: &WorkerRequest) -> Result<()> {
        write_message(&mut self.writer, req).context("Failed to send request to worker")
    }

    fn read_response(&mut self) -> Result<WorkerResponse> {
        read_message(&mut self.reader).context("Failed to read response from worker")
    }

    /// Kill the child process. Errors are logged but not propagated
    /// since we're about to spawn a replacement.
    fn kill(&mut self) {
        eprintln!(
            "[mutafitup] Killing worker process (pid {:?})",
            self.child.id()
        );
        if let Err(e) = self.child.kill() {
            eprintln!("[mutafitup] Warning: failed to kill worker: {e}");
        }
        // Reap the child to avoid zombies.
        let _ = self.child.wait();
    }
}

/// Lightweight, cloneable handle to the inference worker process.
///
/// All ONNX sessions live in a separate process. On model switch, the
/// old process is killed and a new one is spawned, fully isolating
/// CoreML EP's process-global state.
///
/// The public API (`spawn`, `load`, `predict`) is identical to the
/// previous in-process thread design — callers are unaware of the
/// process boundary.
#[derive(Clone)]
pub struct NativeModelHandle {
    tx: mpsc::SyncSender<DispatchCmd>,
}

impl NativeModelHandle {
    /// Start the dispatcher thread (which manages the worker child
    /// process) and return a handle.
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::sync_channel::<DispatchCmd>(1);

        std::thread::Builder::new()
            .name("mutafitup-dispatcher".into())
            .spawn(move || Self::dispatcher_loop(rx))
            .expect("failed to spawn dispatcher thread");

        Self { tx }
    }

    /// Load a model by killing the current worker process and spawning
    /// a fresh one. Returns a receiver that will contain the model
    /// metadata on success.
    pub fn load(&self, model_dir: String, ep: EpChoice) -> mpsc::Receiver<Result<ModelInfo>> {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        let _ = self.tx.send(DispatchCmd::Load {
            model_dir,
            ep,
            reply: reply_tx,
        });
        reply_rx
    }

    /// Run inference on the currently loaded model. Returns a receiver
    /// that will contain the per-sequence predictions.
    pub fn predict(&self, sequences: Vec<String>) -> mpsc::Receiver<Result<Vec<Predictions>>> {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        let _ = self.tx.send(DispatchCmd::Predict {
            sequences,
            reply: reply_tx,
        });
        reply_rx
    }

    /// Dispatcher loop running on a dedicated thread. Owns the child
    /// process and serialises all IPC through it.
    fn dispatcher_loop(rx: mpsc::Receiver<DispatchCmd>) {
        let mut worker: Option<WorkerProcess> = None;

        while let Ok(cmd) = rx.recv() {
            match cmd {
                DispatchCmd::Load {
                    model_dir,
                    ep,
                    reply,
                } => {
                    // Kill the old worker process (full CoreML cleanup).
                    if let Some(mut old) = worker.take() {
                        old.kill();
                    }

                    // Spawn a fresh worker process.
                    let mut proc = match WorkerProcess::spawn_new() {
                        Ok(p) => p,
                        Err(e) => {
                            let _ = reply.send(Err(e));
                            continue;
                        }
                    };

                    // Send the Load request to the new worker.
                    if let Err(e) = proc.send_request(&WorkerRequest::Load {
                        model_dir: model_dir.clone(),
                        ep,
                    }) {
                        let _ = reply.send(Err(e));
                        // Worker is broken — don't keep it.
                        proc.kill();
                        continue;
                    }

                    // Read the response.
                    match proc.read_response() {
                        Ok(WorkerResponse::Loaded(Ok(info))) => {
                            worker = Some(proc);
                            let _ = reply.send(Ok(info));
                        }
                        Ok(WorkerResponse::Loaded(Err(msg))) => {
                            // Worker is alive but model failed to load.
                            // Keep the worker alive for a potential retry
                            // with a different model.
                            worker = Some(proc);
                            let _ = reply.send(Err(anyhow::anyhow!("{msg}")));
                        }
                        Ok(_) => {
                            // Unexpected response type.
                            proc.kill();
                            let _ = reply.send(Err(anyhow::anyhow!(
                                "Worker sent unexpected response to Load"
                            )));
                        }
                        Err(e) => {
                            proc.kill();
                            let _ = reply.send(Err(e));
                        }
                    }
                }

                DispatchCmd::Predict { sequences, reply } => {
                    let Some(ref mut proc) = worker else {
                        let _ = reply.send(Err(anyhow::anyhow!("No worker process running")));
                        continue;
                    };

                    if let Err(e) = proc.send_request(&WorkerRequest::Predict { sequences }) {
                        let _ = reply.send(Err(e));
                        // Worker pipe is broken — tear it down.
                        if let Some(mut dead) = worker.take() {
                            dead.kill();
                        }
                        continue;
                    }

                    match proc.read_response() {
                        Ok(WorkerResponse::Predicted(Ok(preds))) => {
                            let _ = reply.send(Ok(preds));
                        }
                        Ok(WorkerResponse::Predicted(Err(msg))) => {
                            let _ = reply.send(Err(anyhow::anyhow!("{msg}")));
                        }
                        Ok(_) => {
                            let _ = reply.send(Err(anyhow::anyhow!(
                                "Worker sent unexpected response to Predict"
                            )));
                        }
                        Err(e) => {
                            let _ = reply.send(Err(e));
                            // Worker is broken.
                            if let Some(mut dead) = worker.take() {
                                dead.kill();
                            }
                        }
                    }
                }
            }
        }

        // Parent dropped all handles — clean up.
        if let Some(mut proc) = worker.take() {
            proc.kill();
        }
        eprintln!("[mutafitup] Dispatcher thread exiting");
    }
}
