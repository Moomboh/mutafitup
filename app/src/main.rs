use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use dioxus::prelude::*;

/// Turn a blocking `mpsc::Receiver` into a future.
///
/// Spawns a helper thread that blocks on `recv()` and wakes the async
/// task when the value arrives. This keeps the main-thread async
/// runtime free for UI updates while waiting for background work.
#[cfg(not(target_arch = "wasm32"))]
fn async_recv<T: Send + 'static>(
    rx: std::sync::mpsc::Receiver<T>,
) -> impl std::future::Future<Output = Option<T>> {
    use std::pin::Pin;
    use std::task::{Context, Poll};

    struct RecvFuture<T: Send + 'static> {
        /// Holds the receiver until the waiter thread takes it.
        rx: Option<std::sync::mpsc::Receiver<T>>,
        result: Arc<Mutex<Option<T>>>,
    }

    impl<T: Send + 'static> std::future::Future for RecvFuture<T> {
        type Output = Option<T>;

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            // Check if result is ready
            if let Ok(mut guard) = self.result.try_lock() {
                if guard.is_some() {
                    return Poll::Ready(guard.take());
                }
            }

            // First poll: take the receiver and spawn the blocking waiter thread
            if let Some(rx) = self.rx.take() {
                let result = self.result.clone();
                let waker = cx.waker().clone();
                std::thread::spawn(move || {
                    let val = rx.recv().ok();
                    *result.lock().unwrap() = val;
                    waker.wake();
                });
            }

            Poll::Pending
        }
    }

    RecvFuture {
        rx: Some(rx),
        result: Arc::new(Mutex::new(None)),
    }
}

/// Yield once to the async runtime, allowing queued UI updates to flush.
#[cfg(not(target_arch = "wasm32"))]
fn async_yield() -> impl std::future::Future<Output = ()> {
    use std::pin::Pin;
    use std::task::{Context, Poll};

    struct YieldOnce(bool);

    impl std::future::Future for YieldOnce {
        type Output = ();
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.0 {
                Poll::Ready(())
            } else {
                self.0 = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    YieldOnce(false)
}

mod components;
mod fasta;
mod inference;
mod model_store;
mod portal;
mod prediction_controls;
mod prediction_view;
mod sequence_input;
mod sequence_list;
mod tooltip;

use components::button::{Button, ButtonVariant};
use components::card::{Card, CardContent};
use components::modal::Modal;
use components::progress::{Progress, ProgressIndicator};
use inference::{EpChoice, ExportMetadata, ModelManifest, ModelManifestEntry, Predictions, TaskLabels};
use portal::{PortalOut, TooltipPortal};
use prediction_controls::PredictionControls;
use sequence_input::SequenceInput;
use sequence_list::{
    ModelResult, SequenceEntry, SequenceItems, SequencePagination, SequenceStatus, SequenceToolbar,
};

/// Task label metadata embedded at compile time.
const TASK_LABELS_JSON: &str = include_str!("../assets/task_labels.json");

const VERSION: &str = env!("CARGO_PKG_VERSION");

const FAVICON: Asset = asset!("/assets/favicon.ico");
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");
const THEME_CSS: Asset = asset!("/assets/dx-components-theme.css");
const STYLES_CSS: Asset = asset!("/assets/styles.css");



fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::args().any(|a| a == "--worker") {
        inference::native::worker_main();
        return;
    }

    dioxus::launch(App);
}

// ── Shared model state ──

/// On native, wraps a `NativeModelHandle` (channel to the persistent
/// worker thread). On web, wraps the `WebBackend` in `Arc<Mutex>`.
#[derive(Clone)]
struct ModelState {
    #[cfg(not(target_arch = "wasm32"))]
    inner: inference::native::NativeModelHandle,
    #[cfg(target_arch = "wasm32")]
    inner: Arc<Mutex<inference::web::WebBackend>>,
}

// ── Status types ──

/// Progress snapshot for model download / initialization.
#[derive(Clone, PartialEq)]
struct DownloadProgress {
    /// Human-readable label for the current phase, e.g. "model.onnx".
    phase: String,
    /// Bytes received so far.
    completed_bytes: u64,
    /// Total bytes expected (`None` when Content-Length is unavailable).
    total_bytes: Option<u64>,
}

#[derive(Clone, PartialEq)]
enum ModelStatus {
    /// Fetching the model manifest.
    LoadingManifest,
    /// Downloading model files (with optional byte-level progress).
    Downloading(DownloadProgress),
    /// Model bytes downloaded; building the ORT session.
    Initializing,
    /// Ready for inference.
    Ready,
    /// Something went wrong.
    Error(String),
}

// ── App ──

#[component]
fn App() -> Element {
    use_context_provider(TooltipPortal::new);

    rsx! {
        document::Link { rel: "icon", href: FAVICON }
        document::Stylesheet { href: TAILWIND_CSS }
        document::Stylesheet { href: THEME_CSS }
        document::Stylesheet { href: STYLES_CSS }
        PredictionPage {}

        // Tooltip portal outlet — sits at the root of the DOM tree,
        // outside any overflow / transform containers.
        PortalOut {}
    }
}

// ── Main prediction page ──

#[component]
fn PredictionPage() -> Element {
    let text_input = use_signal(|| String::new());
    let mut model_status = use_signal(|| ModelStatus::LoadingManifest);
    let mut download_started = use_signal(|| None::<web_time::Instant>);
    let mut metadata = use_signal(|| None::<ExportMetadata>);

    let mut model_state: Signal<Option<ModelState>> = use_signal(|| None);

    // Per-model metadata map: preserved across model switches so
    // prediction views can look up the metadata for each model's results.
    let mut model_metadata: Signal<HashMap<String, ExportMetadata>> =
        use_signal(HashMap::new);

    // Persistent worker thread for native ONNX inference. Spawned once
    // and reused for all model loads / predictions to avoid CoreML EP
    // crashes from overlapping sessions.
    #[cfg(not(target_arch = "wasm32"))]
    let worker_handle = use_hook(|| inference::native::NativeModelHandle::spawn());

    // Model manifest and selection
    let mut manifest = use_signal(|| None::<ModelManifest>);
    let mut selected_model_id = use_signal(|| None::<String>);

    // Execution provider selection
    let mut available_backends: Signal<Vec<EpChoice>> = use_signal(Vec::new);
    let mut selected_ep = use_signal(|| None::<EpChoice>);

    // Detect available EPs once at startup (native: synchronous, web: deferred to manifest loader)
    #[cfg(not(target_arch = "wasm32"))]
    use_hook(|| {
        let eps = inference::native::available_eps();
        available_backends.set(eps.clone());
        if let Some(best) = eps.into_iter().next() {
            selected_ep.set(Some(best));
        }
    });

    let task_labels = use_signal(|| {
        serde_json::from_str::<TaskLabels>(TASK_LABELS_JSON).ok()
    });

    let mut entries: Signal<Vec<SequenceEntry>> = use_signal(Vec::new);
    let batch_size = use_signal(|| 1usize);
    let mut is_predicting = use_signal(|| false);
    let mut completed_count = use_signal(|| 0usize);
    let mut elapsed_ms = use_signal(|| None::<f64>);

    // Confirmation dialog state for "Load sequences" when predictions exist
    let mut show_load_confirm = use_signal(|| false);

    // Pagination state (lifted here so toolbar, items, and footer can share it)
    let mut current_page = use_signal(|| 0usize);
    let page_size = use_signal(|| 10usize);

    // Load manifest (and detect EPs on web) on mount
    let _manifest_loader = use_resource(move || async move {
        // On web, EP detection is async (probes WebGPU adapter).
        #[cfg(target_arch = "wasm32")]
        {
            let eps = inference::web::available_eps().await;
            if let Some(best) = eps.first().cloned() {
                selected_ep.set(Some(best));
            }
            available_backends.set(eps);
        }

        match load_manifest().await {
            Ok(m) => {
                let compatible = platform_models(&m.models);
                if let Some(first) = compatible.first() {
                    let first_id = first.id.clone();
                    manifest.set(Some(m));
                    selected_model_id.set(Some(first_id));
                } else {
                    model_status.set(ModelStatus::Error("No compatible models in manifest".into()));
                }
            }
            Err(e) => {
                model_status.set(ModelStatus::Error(format!("Failed to load manifest: {e:#}")));
            }
        }
    });

    // Load model whenever selected_model_id or selected_ep changes
    let _model_loader = use_resource(move || {
        #[cfg(not(target_arch = "wasm32"))]
        let worker = worker_handle.clone();

        async move {
            let Some(model_id) = selected_model_id.read().clone() else {
                return;
            };
            // .read() subscribes to selected_ep so EP changes trigger reload
            let Some(ep) = selected_ep.read().clone() else {
                return;
            };

            // Never switch models while a prediction is in-flight.
            // peek() avoids subscribing to is_predicting — otherwise
            // the resource would re-run (and reload the model) every
            // time a prediction finishes.
            if *is_predicting.peek() {
                return;
            }

            model_status.set(ModelStatus::Downloading(DownloadProgress {
                phase: "Preparing...".into(),
                completed_bytes: 0,
                total_bytes: None,
            }));
            download_started.set(Some(web_time::Instant::now()));
            metadata.set(None);
            model_state.set(None);

            // Do NOT clear prediction results — they are preserved for
            // multi-model comparison.
            completed_count.set(0);
            elapsed_ms.set(None);

            match load_model(
                &model_id,
                &ep,
                model_status,
                #[cfg(not(target_arch = "wasm32"))]
                &worker,
            )
            .await
            {
                Ok((state, info)) => {
                    // Store metadata both as "current" and in the per-model map
                    model_metadata.write().insert(model_id.clone(), info.metadata.clone());
                    metadata.set(Some(info.metadata));
                    model_state.set(Some(state));
                    model_status.set(ModelStatus::Ready);
                }
                Err(e) => {
                    model_status.set(ModelStatus::Error(format!("{e:#}")));
                }
            }
        }
    });

    // Global track stats: recomputed whenever model_metadata or entries change.
    let global_track_stats = use_memo(move || {
        let meta_guard = model_metadata.read();
        let entries_guard = entries.read();
        prediction_view::compute_global_track_stats(&meta_guard, &entries_guard)
    });

    let is_ready = matches!(*model_status.read(), ModelStatus::Ready);
    let has_entries = !entries.read().is_empty();
    let total = entries.read().len();
    let predicting = *is_predicting.read();

    // Pagination computations
    let items_per_page = *page_size.read();
    let total_pages = if items_per_page > 0 {
        (total + items_per_page - 1) / items_per_page
    } else {
        1
    };
    let page = (*current_page.read()).min(total_pages.saturating_sub(1));
    let start = page * items_per_page;
    let end = (start + items_per_page).min(total);
    let page_entries: Vec<SequenceEntry> = entries.read()[start..end].to_vec();

    // Actually load the sequences (called directly or after confirmation)
    let mut do_load_sequences = move || {
        let text = text_input.read().clone();
        let parsed = fasta::parse_input(&text);

        let new_entries: Vec<SequenceEntry> = parsed
            .into_iter()
            .enumerate()
            .map(|(i, seq)| SequenceEntry {
                index: i,
                header: seq.header,
                sequence: seq.sequence,
                model_results: Vec::new(),
            })
            .collect();

        entries.set(new_entries);
        model_metadata.write().clear();
        // Re-add the current model's metadata if loaded
        if let Some(mid) = selected_model_id.peek().clone() {
            if let Some(meta) = metadata.peek().clone() {
                model_metadata.write().insert(mid, meta);
            }
        }
        completed_count.set(0);
        elapsed_ms.set(None);
        current_page.set(0);
    };

    // "Load sequences" handler: shows confirmation if predictions exist
    let on_load_sequences = move |_: ()| {
        let any_predictions = entries.read().iter().any(|e| e.has_any_predictions());
        if any_predictions {
            show_load_confirm.set(true);
        } else {
            do_load_sequences();
        }
    };

    let on_predict = move |_: ()| {
        let model = model_state.read().clone();
        let Some(model) = model else { return };
        let bs = *batch_size.read();
        let Some(model_id) = selected_model_id.read().clone() else { return };

        // Look up the model label from the manifest
        let model_label = manifest
            .read()
            .as_ref()
            .and_then(|m| m.models.iter().find(|e| e.id == model_id))
            .map(|e| e.label.clone())
            .unwrap_or_else(|| model_id.clone());

        spawn(async move {
            is_predicting.set(true);
            completed_count.set(0);
            let start = web_time::Instant::now();

            let total_seqs = entries.read().len();

            // Initialize or reset ModelResult for this model on every entry
            {
                let mut e = entries.write();
                for entry in e.iter_mut() {
                    if let Some(existing) = entry.model_results.iter_mut().find(|r| r.model_id == model_id) {
                        existing.status = SequenceStatus::Pending;
                        existing.predictions = None;
                    } else {
                        entry.model_results.push(ModelResult {
                            model_id: model_id.clone(),
                            model_label: model_label.clone(),
                            status: SequenceStatus::Pending,
                            predictions: None,
                        });
                    }
                }
            }

            let mut offset = 0;

            while offset < total_seqs {
                let end = (offset + bs).min(total_seqs);

                // Mark batch as running
                {
                    let mut e = entries.write();
                    for i in offset..end {
                        if let Some(r) = e[i].model_results.iter_mut().find(|r| r.model_id == model_id) {
                            r.status = SequenceStatus::Running;
                        }
                    }
                }

                // Collect batch sequences
                let batch_sequences: Vec<String> = {
                    let e = entries.read();
                    (offset..end).map(|i| e[i].sequence.clone()).collect()
                };

                // Run prediction on the worker thread (native) or inline (web)
                let result = run_prediction(&model, batch_sequences).await;

                // Update entries with results
                match result {
                    Ok(batch_preds) => {
                        let mut e = entries.write();
                        for (j, preds) in batch_preds.into_iter().enumerate() {
                            let idx = offset + j;
                            if idx < e.len() {
                                if let Some(r) = e[idx].model_results.iter_mut().find(|r| r.model_id == model_id) {
                                    r.status = SequenceStatus::Done;
                                    r.predictions = Some(preds);
                                }
                            }
                        }
                    }
                    Err(err) => {
                        let msg = format!("{err:#}");
                        let mut e = entries.write();
                        for i in offset..end {
                            if i < e.len() {
                                if let Some(r) = e[i].model_results.iter_mut().find(|r| r.model_id == model_id) {
                                    r.status = SequenceStatus::Error(msg.clone());
                                }
                            }
                        }
                    }
                }

                completed_count.set(end);
                elapsed_ms.set(Some(start.elapsed().as_millis() as f64));

                offset = end;

                // Yield to the event loop between batches so the UI updates.
                #[cfg(target_arch = "wasm32")]
                {
                    gloo_timers::future::TimeoutFuture::new(0).await;
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    async_yield().await;
                }
            }

            is_predicting.set(false);
        });
    };

    // Build manifest entries for the selector (filtered by platform)
    let manifest_models: Vec<ModelManifestEntry> = manifest
        .read()
        .as_ref()
        .map(|m| platform_models(&m.models))
        .unwrap_or_default();
    let current_model_id = selected_model_id.read().clone().unwrap_or_default();
    let backends = available_backends.read().clone();
    let current_ep_id = selected_ep
        .read()
        .as_ref()
        .map(|ep| ep.id().to_string())
        .unwrap_or_default();

    rsx! {
        // Confirmation dialog for loading new sequences
        Modal {
            open: *show_load_confirm.read(),
            on_close: move |_| show_load_confirm.set(false),

            div {
                class: "flex flex-col gap-4",
                div {
                    class: "text-sm font-medium",
                    "Replace sequences?"
                }
                div {
                    class: "text-xs opacity-60",
                    "Loading new sequences will clear all current predictions. This cannot be undone."
                }
                div {
                    class: "flex items-center justify-end gap-2 mt-2",
                    Button {
                        variant: ButtonVariant::Outline,
                        onclick: move |_| show_load_confirm.set(false),
                        "Cancel"
                    }
                    Button {
                        variant: ButtonVariant::Destructive,
                        onclick: move |_| {
                            show_load_confirm.set(false);
                            do_load_sequences();
                        },
                        "Replace"
                    }
                }
            }
        }

        div {
            class: "bg-surface text-white min-h-screen font-sans flex",

            // ── Left column: fixed sidebar ──
            aside {
                class: "w-[300px] shrink-0 h-screen sticky top-0 overflow-y-auto p-4 flex flex-col gap-4 border-r border-border-subtle",

                // Header (no card — plain text aligned with sidebar padding)
                div {
                    class: "flex flex-col gap-1",
                    // Row 1: title + GitHub icon
                    div {
                        class: "flex items-center justify-between",
                        div { class: "text-base font-semibold leading-none", "mutafitup" }
                        a {
                            href: "https://github.com/Moomboh/mutafitup",
                            target: "_blank",
                            class: "text-neutral-400 hover:text-neutral-200 transition-colors",
                            title: "View on GitHub",
                            svg {
                                xmlns: "http://www.w3.org/2000/svg",
                                width: "16",
                                height: "16",
                                view_box: "0 0 24 24",
                                fill: "currentColor",
                                // GitHub mark (Simple Icons)
                                path {
                                    d: "M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12",
                                }
                            }
                        }
                    }
                    // Row 2: subtitle + version
                    div {
                        class: "flex items-center justify-between",
                        div { class: "text-sm text-neutral-400", "Protein property prediction" }
                        div { class: "text-xs text-neutral-500", "v{VERSION}" }
                    }
                    // Row 3: platform-specific link (Download desktop / Open web version)
                    if cfg!(target_arch = "wasm32") {
                        a {
                            href: "https://github.com/Moomboh/mutafitup/releases/latest",
                            target: "_blank",
                            class: "flex items-center gap-1.5 text-xs text-neutral-500 hover:text-neutral-300 transition-colors",
                            // Lucide Download icon
                            svg {
                                xmlns: "http://www.w3.org/2000/svg",
                                width: "12",
                                height: "12",
                                view_box: "0 0 24 24",
                                fill: "none",
                                stroke: "currentColor",
                                stroke_width: "2",
                                stroke_linecap: "round",
                                stroke_linejoin: "round",
                                path { d: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" }
                                polyline { points: "7 10 12 15 17 10" }
                                line { x1: "12", y1: "15", x2: "12", y2: "3" }
                            }
                            "Download desktop app"
                        }
                    } else {
                        a {
                            href: "https://mutafitup.moomboh.com",
                            target: "_blank",
                            class: "flex items-center gap-1.5 text-xs text-neutral-500 hover:text-neutral-300 transition-colors",
                            // Lucide Globe icon
                            svg {
                                xmlns: "http://www.w3.org/2000/svg",
                                width: "12",
                                height: "12",
                                view_box: "0 0 24 24",
                                fill: "none",
                                stroke: "currentColor",
                                stroke_width: "2",
                                stroke_linecap: "round",
                                stroke_linejoin: "round",
                                circle { cx: "12", cy: "12", r: "10" }
                                path { d: "M2 12h20" }
                                path { d: "M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" }
                            }
                            "Open web version"
                        }
                    }
                }

                // Sequence input
                Card {
                    class: "w-full",
                    CardContent {
                        class: "pt-6",
                        SequenceInput {
                            text_value: text_input,
                            on_load: on_load_sequences,
                            disabled: !is_ready || predicting,
                        }
                    }
                }

                // Model selector + info card
                Card {
                    class: "w-full",
                    style: "background-color: #1a1b23; border: 1px solid #2d2e3a; padding: 0.75rem 0; gap: 0; border-radius: 0.75rem;",
                    CardContent {
                        style: "padding: 0 0.75rem;",

                        // Model selector dropdown
                        if !manifest_models.is_empty() {
                            div {
                                class: "mb-3",
                                label {
                                    class: "block text-xs text-neutral-400 mb-1",
                                    "Model"
                                }
                                select {
                                    class: "model-select",
                                    disabled: predicting,
                                    value: "{current_model_id}",
                                    onchange: move |evt: Event<FormData>| {
                                        let new_id = evt.value();
                                        if !new_id.is_empty() {
                                            selected_model_id.set(Some(new_id));
                                        }
                                    },
                                    for model in manifest_models.iter() {
                                        option {
                                            key: "{model.id}",
                                            value: "{model.id}",
                                            selected: model.id == current_model_id,
                                            "{model.label}"
                                        }
                                    }
                                }
                            }
                        }

                        // Backend (EP) selector dropdown
                        if backends.len() > 1 {
                            div {
                                class: "mb-3",
                                label {
                                    class: "block text-xs text-neutral-400 mb-1",
                                    "Backend"
                                }
                                select {
                                    class: "model-select",
                                    disabled: predicting,
                                    value: "{current_ep_id}",
                                    onchange: move |evt: Event<FormData>| {
                                        if let Some(ep) = EpChoice::from_id(&evt.value()) {
                                            selected_ep.set(Some(ep));
                                        }
                                    },
                                    for ep in backends.iter() {
                                        option {
                                            key: "{ep.id()}",
                                            value: "{ep.id()}",
                                            selected: ep.id() == current_ep_id,
                                            "{ep}"
                                        }
                                    }
                                }
                            }
                        }

                        // Status indicator
                        div {
                            class: "flex flex-col gap-2 text-sm",
                            match &*model_status.read() {
                                ModelStatus::LoadingManifest => rsx! {
                                    div {
                                        class: "flex items-center gap-2",
                                        div { class: "w-2 h-2 rounded-full bg-amber-500 animate-pulse" }
                                        span { class: "text-amber-500", "Loading manifest..." }
                                    }
                                },
                                ModelStatus::Downloading(progress) => {
                                    let pct = match (progress.completed_bytes, progress.total_bytes) {
                                        (done, Some(total)) if total > 0 => Some((done as f64 / total as f64) * 100.0),
                                        _ => None,
                                    };
                                    let has_size = progress.completed_bytes > 0 || progress.total_bytes.is_some();
                                    let size_label = format_download_size(progress.completed_bytes, progress.total_bytes);
                                    let eta_label = download_started
                                        .read()
                                        .and_then(|started| {
                                            let elapsed = started.elapsed().as_secs_f64();
                                            format_eta(progress.completed_bytes, progress.total_bytes, elapsed)
                                        })
                                        .unwrap_or_default();
                                    rsx! {
                                        div {
                                            class: "flex items-center gap-2",
                                            div { class: "w-2 h-2 rounded-full bg-amber-500 animate-pulse" }
                                            span { class: "text-amber-500", "{progress.phase}" }
                                        }
                                        if has_size {
                                            Progress {
                                                value: pct,
                                                max: 100.0,
                                                style: "width: 100%;",
                                                ProgressIndicator {}
                                            }
                                            div {
                                                class: "flex justify-between text-xs opacity-60",
                                                span { "{eta_label}" }
                                                span { "{size_label}" }
                                            }
                                        }
                                    }
                                },
                                ModelStatus::Initializing => rsx! {
                                    div {
                                        class: "flex items-center gap-2",
                                        div { class: "w-2 h-2 rounded-full bg-amber-500 animate-pulse" }
                                        span { class: "text-amber-500", "Initializing runtime..." }
                                    }
                                },
                                ModelStatus::Ready => rsx! {
                                    div {
                                        class: "flex items-center gap-2",
                                        div { class: "w-2 h-2 rounded-full bg-green-500" }
                                        span { class: "text-green-500", "Model ready" }
                                    }
                                },
                                ModelStatus::Error(e) => rsx! {
                                    div {
                                        class: "flex items-center gap-2",
                                        div { class: "w-2 h-2 rounded-full bg-red-500" }
                                        span { class: "text-red-500 break-words", "Error: {e}" }
                                    }
                                },
                            }
                        }

                        if let Some(meta) = metadata.read().as_ref() {
                            div {
                                class: "flex flex-col gap-1 mt-3 text-xs opacity-60",
                                span { "Backbone: {meta.base_checkpoint}" }
                                span {
                                    "Tasks: "
                                    {meta.tasks.keys().cloned().collect::<Vec<_>>().join(", ")}
                                }
                            }
                        }
                    }
                }

                // Prediction controls (batch size + predict button)
                Card {
                    class: "w-full",
                    CardContent {
                        class: "pt-6",
                        PredictionControls {
                            batch_size: batch_size,
                            on_predict: on_predict,
                            is_predicting: predicting,
                            can_predict: is_ready && has_entries,
                        }
                    }
                }

                // ESM attribution (required by Cambrian Open License)
                div {
                    class: "mt-auto pt-4 text-xs text-neutral-500",
                    a {
                        href: "https://www.evolutionaryscale.ai/blog/esm-cambrian",
                        target: "_blank",
                        class: "hover:text-neutral-300 transition-colors",
                        "Built with ESM"
                    }
                }
            }

            // ── Right column: 3-zone flex layout ──
            main {
                class: "flex-1 min-w-0 h-screen flex flex-col",

                // Zone 1: Fixed toolbar (always visible)
                SequenceToolbar {
                    entries: entries,
                    model_metadata: model_metadata.read().clone(),
                    current_model_id: current_model_id.clone(),
                    is_predicting: predicting,
                    prediction_completed: *completed_count.read(),
                    prediction_total: total,
                    prediction_elapsed_ms: *elapsed_ms.read(),
                }

                // Zone 2: Scrollable sequence items
                div {
                    class: "flex-1 min-h-0 overflow-y-auto p-5",

                    if has_entries {
                        SequenceItems {
                            page_entries: page_entries,
                            model_metadata: model_metadata.read().clone(),
                            task_labels: task_labels,
                            global_track_stats: global_track_stats.read().clone(),
                            current_model_id: current_model_id.clone(),
                        }
                    } else {
                        // Empty state
                        div {
                            class: "flex items-center justify-center h-full opacity-30",
                            div {
                                class: "text-center",
                                div { class: "text-4xl mb-3 font-mono opacity-50", "{{  }}" }
                                div { class: "text-sm", "Paste or upload sequences to get started" }
                            }
                        }
                    }
                }

                // Legend grid: shown when we have any model metadata and at least one entry
                if !model_metadata.read().is_empty() && has_entries {
                    prediction_view::TrackLegendGrid {
                        model_metadata: model_metadata.read().clone(),
                        task_labels: task_labels,
                        global_track_stats: global_track_stats.read().clone(),
                    }
                }

                // Zone 3: Fixed pagination footer (only when there are entries)
                if has_entries {
                    SequencePagination {
                        current_page: current_page,
                        page_size: page_size,
                        total_pages: total_pages,
                        total: total,
                    }
                }
            }
        }
    }
}

// ── Helper functions ──

/// Return only the models loadable on the current platform.
///
/// On web, models with external data files (>2 GB) cannot be loaded by
/// onnxruntime-web. On desktop, all models are supported.
/// Format a byte-level progress label, e.g. "812 MB / 1.30 GB".
fn format_download_size(done: u64, total: Option<u64>) -> String {
    fn human(bytes: u64) -> String {
        if bytes >= 1_000_000_000 {
            format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
        } else if bytes >= 1_000_000 {
            format!("{:.1} MB", bytes as f64 / 1_000_000.0)
        } else if bytes >= 1_000 {
            format!("{:.0} KB", bytes as f64 / 1_000.0)
        } else {
            format!("{bytes} B")
        }
    }
    match total {
        Some(t) => format!("{} / {}", human(done), human(t)),
        None => human(done),
    }
}

/// Format an estimated time remaining for a download.
///
/// Returns `None` if there is insufficient data to estimate (too early,
/// no total size, or no bytes received yet).
fn format_eta(completed: u64, total: Option<u64>, elapsed_secs: f64) -> Option<String> {
    let total = total?;
    if completed == 0 || elapsed_secs < 0.5 || total == 0 || completed >= total {
        return None;
    }
    let rate = completed as f64 / elapsed_secs;
    let remaining_secs = (total - completed) as f64 / rate;
    let secs = remaining_secs.ceil() as u64;
    if secs < 60 {
        Some(format!("~{secs}s"))
    } else if secs < 3600 {
        Some(format!("~{}m {:02}s", secs / 60, secs % 60))
    } else {
        Some(format!("~{}h {:02}m", secs / 3600, (secs % 3600) / 60))
    }
}

fn platform_models(models: &[ModelManifestEntry]) -> Vec<ModelManifestEntry> {
    #[cfg(target_arch = "wasm32")]
    {
        models.iter().filter(|m| m.web_compatible).cloned().collect()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        models.to_vec()
    }
}

/// Load the model manifest listing all available ONNX exports.
async fn load_manifest() -> anyhow::Result<ModelManifest> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let source = model_store::native::resolve_source()?;
        let json = tokio::task::spawn_blocking(move || {
            model_store::native::fetch_manifest(&source)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Manifest fetch thread panicked: {e}"))??;
        let manifest: ModelManifest = serde_json::from_str(&json)
            .map_err(|e| anyhow::anyhow!("Invalid manifest: {e}"))?;
        Ok(manifest)
    }

    #[cfg(target_arch = "wasm32")]
    {
        let json = model_store::web::fetch_manifest(
            model_store::web::DEFAULT_MODELS_URL,
        )
        .await?;
        let manifest: ModelManifest = serde_json::from_str(&json)
            .map_err(|e| anyhow::anyhow!("Invalid manifest: {e}"))?;
        Ok(manifest)
    }
}

/// Lightweight info bundle returned alongside `ModelState` after loading.
#[cfg(not(target_arch = "wasm32"))]
struct LoadResult {
    metadata: ExportMetadata,
}

#[cfg(target_arch = "wasm32")]
struct LoadResult {
    metadata: ExportMetadata,
}

/// Load the inference backend for a specific model (identified by its
/// manifest `id`, e.g. `"accgrad_lora/esmc_300m_all_r4/best_overall"`).
///
/// On native, ensures model files are cached locally and sends a `Load`
/// command to the persistent worker thread. On web, fetches files via
/// the OPFS-backed model store and creates the session from bytes.
/// Progress is reported through `status`.
async fn load_model(
    model_id: &str,
    ep: &EpChoice,
    mut status: Signal<ModelStatus>,
    #[cfg(not(target_arch = "wasm32"))] worker: &inference::native::NativeModelHandle,
) -> anyhow::Result<(ModelState, LoadResult)> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let source = model_store::native::resolve_source()?;
        let model_id_owned = model_id.to_string();
        let (tx, mut rx) =
            tokio::sync::mpsc::unbounded_channel::<(String, u64, Option<u64>)>();

        let mut download_handle = tokio::task::spawn_blocking(move || {
            model_store::native::ensure_model_dir(
                &source,
                &model_id_owned,
                &move |label, done, total| {
                    let _ = tx.send((label.to_string(), done, total));
                },
            )
        });

        // Poll progress while download runs on a background thread,
        // keeping the UI event loop responsive.
        let model_dir = loop {
            tokio::select! {
                msg = rx.recv() => {
                    if let Some((label, done, total)) = msg {
                        let mut s = status;
                        s.set(ModelStatus::Downloading(DownloadProgress {
                            phase: format!("Downloading {label}..."),
                            completed_bytes: done,
                            total_bytes: total,
                        }));
                    }
                }
                result = &mut download_handle => {
                    break result
                        .map_err(|e| anyhow::anyhow!("Download thread panicked: {e}"))??;
                }
            }
        };

        status.set(ModelStatus::Initializing);
        let reply_rx = worker.load(model_dir, ep.clone());
        let info = async_recv(reply_rx)
            .await
            .ok_or_else(|| anyhow::anyhow!("Model worker thread died"))??;

        let state = ModelState {
            inner: worker.clone(),
        };
        let result = LoadResult {
            metadata: info.metadata,
        };
        Ok((state, result))
    }

    #[cfg(target_arch = "wasm32")]
    {
        let status_dl = status;
        let files = model_store::web::ensure_model_files(
            model_store::web::DEFAULT_MODELS_URL,
            model_id,
            &move |phase, done, total| {
                let mut s = status_dl;
                s.set(ModelStatus::Downloading(DownloadProgress {
                    phase: phase.to_string(),
                    completed_bytes: done,
                    total_bytes: total,
                }));
            },
        )
        .await?;

        status.set(ModelStatus::Initializing);
        let backend = inference::web::WebBackend::load_from_files(files, ep).await?;
        let result = LoadResult {
            metadata: backend.metadata().clone(),
        };
        let state = ModelState {
            inner: Arc::new(Mutex::new(backend)),
        };
        Ok((state, result))
    }
}

/// Run prediction (platform-specific).
///
/// On native, sends sequences to the persistent worker thread. On web,
/// runs inference inline.
async fn run_prediction(
    model: &ModelState,
    sequences: Vec<String>,
) -> anyhow::Result<Vec<Predictions>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let reply_rx = model.inner.predict(sequences);
        let result = async_recv(reply_rx).await;
        result.unwrap_or_else(|| Err(anyhow::anyhow!("Model worker thread died")))
    }

    #[cfg(target_arch = "wasm32")]
    {
        let mut backend = model.inner.lock().unwrap();
        backend.predict(&sequences).await
    }
}
