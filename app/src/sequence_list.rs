/// Sequence list components: toolbar, items list, and pagination footer.
///
/// These three components are designed to be placed in separate layout zones
/// by the parent `PredictionPage`:
/// - `SequenceToolbar` — fixed top toolbar with summary counts
/// - `SequenceItems` — scrollable list of sequence items with accordion expansion
/// - `SequencePagination` — fixed bottom pagination controls
use std::collections::HashMap;

use dioxus::prelude::*;

use crate::components::badge::{Badge, BadgeVariant};
use crate::components::input::Input;
use crate::components::pagination::{
    Pagination, PaginationContent, PaginationItem, PaginationLink, PaginationNext,
    PaginationPrevious,
};
use crate::inference::{ExportMetadata, Predictions, TaskLabels};
use crate::prediction_controls::PredictionProgress;
use crate::prediction_view::{GlobalTrackStats, ModelPredictionData, PredictionView};
use crate::tooltip::Tooltip;

const SEQ_TRUNCATE_LEN: usize = 60;

/// Status of a single sequence for a specific model in the prediction pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum SequenceStatus {
    Pending,
    Running,
    Done,
    Error(String),
}

/// Prediction result from a single model for a single sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelResult {
    pub model_id: String,
    pub model_label: String,
    pub status: SequenceStatus,
    pub predictions: Option<Predictions>,
}

/// A sequence entry in the list, holding results from zero or more models.
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceEntry {
    pub index: usize,
    pub header: Option<String>,
    pub sequence: String,
    pub model_results: Vec<ModelResult>,
}

impl SequenceEntry {
    /// Find the result for a specific model, if any.
    pub fn result_for_model(&self, model_id: &str) -> Option<&ModelResult> {
        self.model_results.iter().find(|r| r.model_id == model_id)
    }

    /// Status for the current model. Returns `None` if this sequence has
    /// not been predicted with the given model yet.
    pub fn status_for_model(&self, model_id: &str) -> Option<&SequenceStatus> {
        self.result_for_model(model_id).map(|r| &r.status)
    }

    /// Whether any model has completed predictions for this sequence.
    pub fn has_any_predictions(&self) -> bool {
        self.model_results.iter().any(|r| r.predictions.is_some())
    }
}

// ── Toolbar ──

/// Per-model summary for the toolbar tooltip.
struct ModelSummary {
    model_id: String,
    model_label: String,
    done: usize,
    running: usize,
    error: usize,
    pending: usize,
    tasks: Vec<String>,
}

#[component]
pub fn SequenceToolbar(
    entries: Signal<Vec<SequenceEntry>>,
    model_metadata: HashMap<String, ExportMetadata>,
    current_model_id: String,
    is_predicting: bool,
    prediction_completed: usize,
    prediction_total: usize,
    prediction_elapsed_ms: Option<f64>,
) -> Element {
    let entries_read = entries.read();
    let total = entries_read.len();
    let done_count = entries_read
        .iter()
        .filter(|e| {
            e.status_for_model(&current_model_id)
                .map(|s| matches!(s, SequenceStatus::Done))
                .unwrap_or(false)
        })
        .count();
    let error_count = entries_read
        .iter()
        .filter(|e| {
            e.status_for_model(&current_model_id)
                .map(|s| matches!(s, SequenceStatus::Error(_)))
                .unwrap_or(false)
        })
        .count();
    let running_count = entries_read
        .iter()
        .filter(|e| {
            e.status_for_model(&current_model_id)
                .map(|s| matches!(s, SequenceStatus::Running))
                .unwrap_or(false)
        })
        .count();

    // Collect per-model summaries: unique models that have any ModelResult.
    let model_summaries: Vec<ModelSummary> = {
        let mut seen: HashMap<String, ModelSummary> = HashMap::new();
        for entry in entries_read.iter() {
            for r in &entry.model_results {
                let summary = seen.entry(r.model_id.clone()).or_insert_with(|| {
                    let mut tasks: Vec<String> = model_metadata
                        .get(&r.model_id)
                        .map(|m| m.tasks.keys().cloned().collect())
                        .unwrap_or_default();
                    tasks.sort();
                    ModelSummary {
                        model_id: r.model_id.clone(),
                        model_label: r.model_label.clone(),
                        done: 0,
                        running: 0,
                        error: 0,
                        pending: 0,
                        tasks,
                    }
                });
                match &r.status {
                    SequenceStatus::Done => summary.done += 1,
                    SequenceStatus::Running => summary.running += 1,
                    SequenceStatus::Error(_) => summary.error += 1,
                    SequenceStatus::Pending => summary.pending += 1,
                }
            }
        }
        let mut summaries: Vec<ModelSummary> = seen.into_values().collect();
        summaries.sort_by(|a, b| a.model_label.cmp(&b.model_label));
        summaries
    };
    let model_count = model_summaries.len();

    drop(entries_read);

    let tooltip_content = rsx! {
        div {
            class: "flex flex-col gap-2",
            for (si, summary) in model_summaries.iter().enumerate() {
                div {
                    key: "{summary.model_id}",
                    class: "flex flex-col gap-0.5",

                    div {
                        class: "font-medium text-[11px]",
                        "{summary.model_label}"
                    }

                    div {
                        class: "flex items-center gap-2 text-[10px]",
                        if summary.done > 0 {
                            span { class: "text-green-400", "\u{2713} {summary.done} done" }
                        }
                        if summary.running > 0 {
                            span { class: "text-amber-400", "{summary.running} running" }
                        }
                        if summary.error > 0 {
                            span { class: "text-red-400", "{summary.error} error(s)" }
                        }
                        if summary.pending > 0 {
                            span { class: "opacity-50", "{summary.pending} pending" }
                        }
                    }

                    if !summary.tasks.is_empty() {
                        div {
                            class: "text-[10px] opacity-50",
                            "Tasks: {summary.tasks.join(\", \")}"
                        }
                    }
                }

                // Separator between models
                if si + 1 < model_summaries.len() {
                    div {
                        class: "border-t border-border-subtle",
                    }
                }
            }
        }
    };

    let seq_label = if total == 1 { "sequence" } else { "sequences" };
    let model_label = if model_count == 1 { "model" } else { "models" };
    let error_label = if error_count == 1 { "error" } else { "errors" };

    rsx! {
        div {
            class: "flex items-center gap-4 px-5 py-3 border-b border-border-subtle shrink-0",

            // Left: summary counts with middot separators
            div {
                class: "flex items-center gap-2 text-sm shrink-0",

                span { class: "font-medium", "{total} {seq_label}" }

                if model_count > 0 {
                    span { class: "opacity-20", "\u{00B7}" }
                    Tooltip {
                        content: tooltip_content,
                        span {
                            class: "text-blue-400 text-xs cursor-default",
                            "{model_count} {model_label}"
                        }
                    }
                }
                if done_count > 0 {
                    span { class: "opacity-20", "\u{00B7}" }
                    span { class: "text-green-400 text-xs", "{done_count} done" }
                }
                if running_count > 0 {
                    span { class: "opacity-20", "\u{00B7}" }
                    span { class: "text-amber-400 text-xs", "{running_count} running" }
                }
                if error_count > 0 {
                    span { class: "opacity-20", "\u{00B7}" }
                    span { class: "text-red-400 text-xs", "{error_count} {error_label}" }
                }
            }

            // Right: prediction progress bar
            PredictionProgress {
                is_predicting: is_predicting,
                completed: prediction_completed,
                total: prediction_total,
                elapsed_ms: prediction_elapsed_ms,
            }
        }
    }
}

// ── Items ──

#[component]
pub fn SequenceItems(
    page_entries: Vec<SequenceEntry>,
    model_metadata: HashMap<String, ExportMetadata>,
    task_labels: Signal<Option<TaskLabels>>,
    global_track_stats: GlobalTrackStats,
    current_model_id: String,
) -> Element {
    rsx! {
        div {
            class: "flex flex-col gap-1",
            for entry in page_entries.iter() {
                SequenceItem {
                    key: "{entry.index}",
                    entry: entry.clone(),
                    model_metadata: model_metadata.clone(),
                    task_labels: task_labels,
                    global_track_stats: global_track_stats.clone(),
                    current_model_id: current_model_id.clone(),
                }
            }
        }
    }
}

// ── Pagination ──

#[component]
pub fn SequencePagination(
    mut current_page: Signal<usize>,
    mut page_size: Signal<usize>,
    total_pages: usize,
    total: usize,
) -> Element {
    let page = *current_page.read();

    rsx! {
        div {
            class: "flex items-center justify-between px-5 py-3 border-t border-border-subtle shrink-0",

            // Left: page indicator
            span {
                class: "text-xs opacity-50 shrink-0",
                if total_pages > 1 {
                    "Page {page + 1} of {total_pages}"
                }
            }

            // Center: pagination links
            if total_pages > 1 {
                Pagination {
                    PaginationContent {
                        PaginationItem {
                            PaginationPrevious {
                                onclick: move |_| {
                                    let p = *current_page.read();
                                    if p > 0 {
                                        current_page.set(p - 1);
                                    }
                                },
                            }
                        }

                        {render_page_numbers(page, total_pages, current_page)}

                        PaginationItem {
                            PaginationNext {
                                onclick: move |_| {
                                    let p = *current_page.read();
                                    if p + 1 < total_pages {
                                        current_page.set(p + 1);
                                    }
                                },
                            }
                        }
                    }
                }
            }

            // Right: page size control
            div {
                class: "flex items-center gap-2 shrink-0",
                label {
                    class: "text-xs opacity-50",
                    r#for: "page-size",
                    "Per page"
                }
                Input {
                    id: "page-size",
                    r#type: "number",
                    min: "1",
                    max: "100",
                    value: "{page_size}",
                    style: "width: 4.5rem;",
                    oninput: move |e: FormEvent| {
                        if let Ok(v) = e.value().parse::<usize>() {
                            if v >= 1 {
                                page_size.set(v);
                                current_page.set(0);
                            }
                        }
                    },
                }
            }
        }
    }
}

/// Render a windowed set of page number links (max 7 visible).
fn render_page_numbers(current: usize, total: usize, mut page_signal: Signal<usize>) -> Element {
    let max_visible = 7;

    if total <= max_visible {
        return rsx! {
            {(0..total).map(|p| {
                let is_active = p == current;
                rsx! {
                    PaginationItem {
                        key: "{p}",
                        PaginationLink {
                            is_active: is_active,
                            onclick: move |_| page_signal.set(p),
                            "{p + 1}"
                        }
                    }
                }
            })}
        };
    }

    // Window: show first, last, and pages around current
    let half = (max_visible - 2) / 2;
    let start = if current <= half + 1 {
        1
    } else if current >= total - half - 2 {
        total - max_visible + 1
    } else {
        current - half
    };
    let end = (start + max_visible - 2).min(total - 1);

    rsx! {
        // First page always shown
        PaginationItem {
            PaginationLink {
                is_active: current == 0,
                onclick: move |_| page_signal.set(0),
                "1"
            }
        }

        // Ellipsis or pages
        if start > 1 {
            PaginationItem {
                span { class: "px-2 text-xs opacity-40", "..." }
            }
        }

        for p in start..end {
            PaginationItem {
                key: "{p}",
                PaginationLink {
                    is_active: p == current,
                    onclick: move |_| page_signal.set(p),
                    "{p + 1}"
                }
            }
        }

        if end < total - 1 {
            PaginationItem {
                span { class: "px-2 text-xs opacity-40", "..." }
            }
        }

        // Last page always shown
        PaginationItem {
            PaginationLink {
                is_active: current == total - 1,
                onclick: move |_| page_signal.set(total - 1),
                "{total}"
            }
        }
    }
}

// ── Sequence Item (private) ──

#[component]
fn SequenceItem(
    entry: SequenceEntry,
    model_metadata: HashMap<String, ExportMetadata>,
    task_labels: Signal<Option<TaskLabels>>,
    global_track_stats: GlobalTrackStats,
    current_model_id: String,
) -> Element {
    let mut is_expanded = use_signal(|| false);
    let idx = entry.index;
    let display_name = entry
        .header
        .as_deref()
        .unwrap_or("(raw sequence)")
        .to_string();

    let truncated_seq = if entry.sequence.len() > SEQ_TRUNCATE_LEN {
        format!("{}...", &entry.sequence[..SEQ_TRUNCATE_LEN])
    } else {
        entry.sequence.clone()
    };

    // Badge reflects the current model's status
    let current_status = entry.status_for_model(&current_model_id);
    let badge_variant = match current_status {
        Some(SequenceStatus::Pending) => BadgeVariant::Secondary,
        Some(SequenceStatus::Running) => BadgeVariant::Outline,
        Some(SequenceStatus::Done) => BadgeVariant::Primary,
        Some(SequenceStatus::Error(_)) => BadgeVariant::Destructive,
        None => BadgeVariant::Secondary,
    };
    let status_text = match current_status {
        Some(SequenceStatus::Pending) => "pending",
        Some(SequenceStatus::Running) => "running",
        Some(SequenceStatus::Done) => "done",
        Some(SequenceStatus::Error(_)) => "error",
        None => "pending",
    };

    // Count how many models have completed predictions
    let model_count = entry
        .model_results
        .iter()
        .filter(|r| r.predictions.is_some())
        .count();

    let expanded = *is_expanded.read();
    let expand_icon = if expanded { "\u{25BC}" } else { "\u{25B6}" };

    // Build ModelPredictionData for all completed model results.
    let model_preds: Vec<ModelPredictionData> = entry
        .model_results
        .iter()
        .filter(|r| matches!(r.status, SequenceStatus::Done) && r.predictions.is_some())
        .map(|r| ModelPredictionData {
            model_id: r.model_id.clone(),
            model_label: r.model_label.clone(),
            predictions: r.predictions.clone().unwrap(),
            metadata: model_metadata.get(&r.model_id).cloned(),
        })
        .collect();

    rsx! {
        div {
            class: "border border-border-subtle rounded-md overflow-hidden",

            // Clickable header row
            button {
                class: "w-full flex items-center gap-3 p-3 text-left hover:bg-surface-elevated transition-colors cursor-pointer",
                onclick: move |_| {
                    let current = *is_expanded.peek();
                    is_expanded.set(!current);
                },

                // Expand indicator
                span {
                    class: "text-[10px] opacity-40 shrink-0 w-3",
                    "{expand_icon}"
                }

                // Index
                span {
                    class: "text-xs opacity-40 w-8 shrink-0 text-right font-mono",
                    "{idx + 1}"
                }

                // Name + sequence preview
                div {
                    class: "flex-1 min-w-0",
                    div {
                        class: "text-sm font-medium truncate",
                        "{display_name}"
                    }
                    div {
                        class: "text-xs opacity-40 font-mono truncate",
                        "{truncated_seq}"
                    }
                }

                // Length
                span {
                    class: "text-xs opacity-40 shrink-0",
                    "{entry.sequence.len()} aa"
                }

                // Model count badge (when multiple models have results)
                if model_count > 1 {
                    span {
                        class: "text-[10px] opacity-40 shrink-0",
                        "{model_count} models"
                    }
                }

                // Status badge (for current model)
                Badge {
                    variant: badge_variant,
                    "{status_text}"
                }
            }

            // Expanded prediction results (accordion)
            if expanded {
                div {
                    class: "border-t border-border-subtle bg-surface-elevated",

                    if model_preds.is_empty() {
                        div {
                            class: "p-3",
                            match current_status {
                                Some(SequenceStatus::Error(msg)) => rsx! {
                                    span { class: "text-xs text-red-400 break-words", "Error: {msg}" }
                                },
                                Some(SequenceStatus::Running) => rsx! {
                                    span { class: "text-xs text-amber-400", "Running inference..." }
                                },
                                _ => rsx! {
                                    span { class: "text-xs opacity-50", "No predictions yet" }
                                },
                            }
                        }
                    } else {
                        div {
                            class: "p-3",
                            PredictionView {
                                sequence: entry.sequence.clone(),
                                model_predictions: model_preds,
                                task_labels: task_labels.read().clone(),
                                global_track_stats: global_track_stats.clone(),
                            }
                        }
                    }
                }
            }
        }
    }
}
