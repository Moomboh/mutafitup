/// Sequence track viewer for visualizing per-residue and per-protein predictions.
///
/// Inspired by Nightingale sequence tracks, but tailored to our use-case:
/// a horizontally-scrollable grid of colored cells aligned to the amino acid
/// sequence, with sticky track labels and hover tooltips.
///
/// Multi-model support: results from multiple models are grouped by task name.
/// Protein-level tasks show one row per model within each task section.
/// Per-residue tasks share a single scrollable viewer with tracks from all
/// models stacked together, separated by dividers between task groups.
use std::collections::HashMap;

use dioxus::prelude::*;

use crate::inference::{ExportMetadata, Predictions, TaskConfig, TaskLabelInfo, TaskLabels};
use crate::sequence_list::SequenceEntry;
use crate::tooltip::Tooltip;

/// Per-task global min/max for regression tracks, computed across all
/// predicted sequences and all models. Classification tasks are omitted
/// (their legends are data-independent).
pub type GlobalTrackStats = HashMap<String, (f32, f32)>;

/// Compute global min/max for each per-residue regression task across all
/// sequences and all models that have completed predictions.
pub fn compute_global_track_stats(
    model_metadata: &HashMap<String, ExportMetadata>,
    entries: &[SequenceEntry],
) -> GlobalTrackStats {
    let mut stats = GlobalTrackStats::new();

    // Collect all per-residue regression task names across all models.
    let mut regression_tasks: HashMap<String, ()> = HashMap::new();
    for meta in model_metadata.values() {
        for (task_name, task_config) in &meta.tasks {
            if task_config.level == "per_residue" && task_config.problem_type == "regression" {
                regression_tasks.insert(task_name.clone(), ());
            }
        }
    }

    for task_name in regression_tasks.keys() {
        let mut global_min = f32::INFINITY;
        let mut global_max = f32::NEG_INFINITY;
        let mut has_data = false;

        for entry in entries {
            // Only consider values for actual amino acid positions,
            // not trailing EOS/padding positions left over from the
            // tokenizer.
            let aa_len = entry.sequence.chars().count();

            for result in &entry.model_results {
                if let Some(ref preds) = result.predictions {
                    if let Some(values) = preds.get(task_name.as_str()) {
                        for &v in values.iter().take(aa_len) {
                            if v < global_min {
                                global_min = v;
                            }
                            if v > global_max {
                                global_max = v;
                            }
                            has_data = true;
                        }
                    }
                }
            }
        }

        if has_data {
            stats.insert(task_name.clone(), (global_min, global_max));
        }
    }

    stats
}

// ── Constants ──

const CELL_W: u32 = 18;
const CELL_H: u32 = 20;
/// Interval for position ruler tick marks.
const RULER_TICK_INTERVAL: usize = 10;
/// Width of the sticky label column.
const LABEL_COL_W: u32 = 180;

// ── Color palettes ──

/// 20 visually distinct colors from Sasha Trubetskoy's palette, ordered
/// by subway-map frequency so the first N are a good default for any
/// N-class task. <https://sashamaps.net/docs/resources/20-colors/>
const BASE_PALETTE: &[&str] = &[
    "#e6194b", // red
    "#3cb44b", // green
    "#ffe119", // yellow
    "#4363d8", // blue
    "#f58231", // orange
    "#911eb4", // purple
    "#46f0f0", // cyan
    "#f032e6", // magenta
    "#bcf60c", // lime
    "#fabebe", // pink
    "#008080", // teal
    "#e6beff", // lavender
    "#9a6324", // brown
    "#fffac8", // beige
    "#800000", // maroon
    "#aaffc3", // mint
    "#808000", // olive
    "#ffd8b1", // apricot
    "#000075", // navy
    "#808080", // grey
];

/// Dark gray used for "non-binding" labels and as the low end of
/// sequential regression ramps.
const DARK_GRAY: &str = "#374151";
const DARK_GRAY_RGB: (f32, f32, f32) = (55.0, 65.0, 81.0);

/// Diverging color scheme for regression tasks with meaningful zero
/// crossings (e.g. disorder Z-scores).
const DIVERGING_COLD: (f32, f32, f32) = (67.0, 99.0, 216.0); // #4363d8 blue
const DIVERGING_NEUTRAL: (f32, f32, f32) = (55.0, 65.0, 81.0); // #374151 same as DARK_GRAY
const DIVERGING_WARM: (f32, f32, f32) = (230.0, 25.0, 75.0); // #e6194b red

/// Tasks that use a diverging (cold → neutral → warm) color scale
/// centered at zero instead of the default sequential ramp.
const DIVERGING_TASKS: &[&str] = &["disorder"];

/// Bright accent palette for per-residue regression ramps. These are
/// used as the high end of the dark-gray → color sequential scale,
/// chosen for high saturation and good contrast against `DARK_GRAY`.
const REGRESSION_ACCENTS: &[&str] = &[
    "#38bdf8", // sky-400
    "#4ade80", // green-400
    "#fb7185", // rose-400
    "#facc15", // yellow-400
    "#c084fc", // purple-400
    "#fb923c", // orange-400
    "#22d3ee", // cyan-400
    "#f472b6", // pink-400
    "#a3e635", // lime-400
    "#818cf8", // indigo-400
];

// ── Helpers ──

/// Parse a hex color string (`#RRGGBB`) into an `(R, G, B)` float triple.
fn hex_to_rgb(hex: &str) -> (f32, f32, f32) {
    let hex = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(128) as f32;
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(128) as f32;
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(128) as f32;
    (r, g, b)
}

/// Get the bright accent RGB for a regression task, cycling through
/// `REGRESSION_ACCENTS` by index.
fn regression_accent(index: usize) -> (f32, f32, f32) {
    hex_to_rgb(REGRESSION_ACCENTS[index % REGRESSION_ACCENTS.len()])
}

/// Compute softmax over a slice of logits.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        vec![0.0; logits.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}

/// Pick the color for a classification class label.
///
/// Uses semantic label matching so that identical labels always get the
/// same color regardless of task or class index:
///
/// - `Non-binding` → dark gray (consistent across all GPSite tasks)
/// - `*-binding` → shared binding accent
/// - Secondary structure labels (`H`, `E`, `C`, `G`, `I`, `B`, `S`, `T`)
///   are mapped explicitly so `secstr` and `secstr8` share colors for
///   equivalent classes
/// - Unknown labels fall back to palette-by-index
fn classification_color(class_index: usize, class_label: &str) -> &'static str {
    // Extract the leading letter token, e.g. "H" from "H (Alpha helix)"
    let token = class_label.split_whitespace().next().unwrap_or(class_label);

    match token {
        // ── Non-binding (all GPSite tasks) ───────────────────────
        "Non-binding" => DARK_GRAY,

        // ── Binding positive class ───────────────────────────────
        _ if class_label.to_ascii_lowercase().ends_with("-binding") => "#3cb44b", // green

        // ── Secondary structure — exact matches across secstr/secstr8 ──
        "H" => "#e6194b", // red — Alpha helix (secstr + secstr8)
        "E" => "#4363d8", // blue — Extended strand (secstr + secstr8)
        "C" => "#3cb44b", // green — Coil (secstr + secstr8)

        // ── Secondary structure — secstr8 helix relatives ────────
        "G" => "#f58231", // orange — 3-10 helix (helix-adjacent)
        "I" => "#911eb4", // purple — Pi helix (helix-adjacent)

        // ── Secondary structure — secstr8 strand relative ────────
        "B" => "#46f0f0", // cyan — Beta bridge (strand-adjacent)

        // ── Secondary structure — secstr8 coil/turn relatives ────
        "S" => "#ffe119", // yellow — Bend (coil-adjacent)
        "T" => "#f032e6", // magenta — Turn (coil-adjacent)

        // ── Fallback for unknown labels ──────────────────────────
        _ => BASE_PALETTE[class_index % BASE_PALETTE.len()],
    }
}

/// Interpolate on a sequential scale from `DARK_GRAY_RGB` to `target`.
/// `t` should be in 0..=1 where 0=dark gray, 1=target color.
fn sequential_color(t: f32, target: (f32, f32, f32)) -> String {
    let t = t.clamp(0.0, 1.0);
    let r = DARK_GRAY_RGB.0 + t * (target.0 - DARK_GRAY_RGB.0);
    let g = DARK_GRAY_RGB.1 + t * (target.1 - DARK_GRAY_RGB.1);
    let b = DARK_GRAY_RGB.2 + t * (target.2 - DARK_GRAY_RGB.2);
    format!("rgb({}, {}, {})", r as u8, g as u8, b as u8)
}

/// Interpolate on a diverging scale: cold → neutral → warm.
/// `t` should be in 0..=1 where 0=cold, 0.5=neutral, 1=warm.
fn diverging_color(t: f32) -> String {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        // cold → neutral (t goes from 0 to 0.5, map to 0..1)
        let s = t * 2.0;
        (
            DIVERGING_COLD.0 + s * (DIVERGING_NEUTRAL.0 - DIVERGING_COLD.0),
            DIVERGING_COLD.1 + s * (DIVERGING_NEUTRAL.1 - DIVERGING_COLD.1),
            DIVERGING_COLD.2 + s * (DIVERGING_NEUTRAL.2 - DIVERGING_COLD.2),
        )
    } else {
        // neutral → warm (t goes from 0.5 to 1, map to 0..1)
        let s = (t - 0.5) * 2.0;
        (
            DIVERGING_NEUTRAL.0 + s * (DIVERGING_WARM.0 - DIVERGING_NEUTRAL.0),
            DIVERGING_NEUTRAL.1 + s * (DIVERGING_WARM.1 - DIVERGING_NEUTRAL.1),
            DIVERGING_NEUTRAL.2 + s * (DIVERGING_WARM.2 - DIVERGING_NEUTRAL.2),
        )
    };
    format!("rgb({}, {}, {})", r as u8, g as u8, b as u8)
}

/// Map a value to diverging-scale t, centered at zero with symmetric bounds.
fn diverging_t(value: f32, min: f32, max: f32) -> f32 {
    let abs_max = min.abs().max(max.abs());
    if abs_max == 0.0 {
        0.5
    } else {
        ((value / abs_max) + 1.0) / 2.0
    }
}

fn format_display_value(value: f32) -> String {
    format!("{value:.1}")
}

#[derive(Clone, PartialEq)]
enum ResidueDetail {
    Classification {
        predicted_class: usize,
        label: String,
        probabilities: Vec<(String, f32)>,
    },
    Regression {
        value: f32,
        unit: String,
    },
}

// ── Multi-model data types ──

/// Prediction data from a single model for a single sequence.
#[derive(Clone, PartialEq)]
pub struct ModelPredictionData {
    pub model_id: String,
    pub model_label: String,
    pub predictions: Predictions,
    pub metadata: Option<ExportMetadata>,
}

/// A protein-level task group: one task potentially with results from
/// multiple models.
struct ProteinTaskGroup<'a> {
    name: String,
    description: String,
    problem_type: String,
    class_labels: Option<Vec<String>>,
    unit: Option<String>,
    /// (model_label, values) for each model that has this task.
    model_values: Vec<(&'a str, Vec<f32>)>,
}

/// A per-residue task group: one task potentially with track data from
/// multiple models.
struct ResidueTaskGroup {
    name: String,
    description: String,
    problem_type: String,
    num_outputs: usize,
    class_labels: Option<Vec<String>>,
    unit: Option<String>,
    /// (model_label, values) for each model that has this task.
    model_values: Vec<(String, Vec<f32>)>,
}

// ── Multi-model task grouping ──

/// Group tasks across multiple models by task name. Tasks with the same
/// name are merged; each contributing model adds its values to the group.
fn group_tasks_across_models<'a>(
    model_predictions: &'a [ModelPredictionData],
    task_labels: Option<&TaskLabels>,
) -> (Vec<ProteinTaskGroup<'a>>, Vec<ResidueTaskGroup>) {
    // Collect all unique task names (sorted for stable ordering) and their
    // config from the first model that has them.
    let mut task_order: Vec<String> = Vec::new();
    let mut task_configs: HashMap<String, (&TaskConfig, Option<&TaskLabelInfo>)> = HashMap::new();

    for mpd in model_predictions {
        let Some(ref meta) = mpd.metadata else {
            continue;
        };
        let mut task_names: Vec<&String> = meta.tasks.keys().collect();
        task_names.sort();
        for name in task_names {
            if !task_configs.contains_key(name.as_str()) {
                let config = &meta.tasks[name.as_str()];
                let label_info = task_labels.and_then(|tl| tl.get(name.as_str()));
                task_configs.insert(name.clone(), (config, label_info));
                task_order.push(name.clone());
            }
        }
    }

    task_order.sort();

    let mut per_protein: Vec<ProteinTaskGroup<'a>> = Vec::new();
    let mut per_residue: Vec<ResidueTaskGroup> = Vec::new();

    for name in &task_order {
        let (config, label_info) = task_configs[name.as_str()];
        let description = label_info
            .map(|l| l.description.clone())
            .unwrap_or_else(|| name.clone());
        let class_labels = label_info.and_then(|l| l.class_labels.clone());
        let unit = label_info.and_then(|l| l.unit.clone());

        // Gather values from each model that predicted this task.
        let mut model_values_owned: Vec<(String, Vec<f32>)> = Vec::new();
        for mpd in model_predictions {
            if let Some(values) = mpd.predictions.get(name.as_str()) {
                model_values_owned.push((mpd.model_label.clone(), values.clone()));
            }
        }

        if model_values_owned.is_empty() {
            continue;
        }

        if config.level == "per_residue" {
            per_residue.push(ResidueTaskGroup {
                name: name.clone(),
                description,
                problem_type: config.problem_type.clone(),
                num_outputs: config.num_outputs,
                class_labels,
                unit,
                model_values: model_values_owned,
            });
        } else {
            // For protein-level, borrow model labels from the source data.
            let mut model_values_borrowed: Vec<(&'a str, Vec<f32>)> = Vec::new();
            for mpd in model_predictions {
                if let Some(values) = mpd.predictions.get(name.as_str()) {
                    model_values_borrowed.push((&mpd.model_label, values.clone()));
                }
            }
            per_protein.push(ProteinTaskGroup {
                name: name.clone(),
                description,
                problem_type: config.problem_type.clone(),
                class_labels,
                unit,
                model_values: model_values_borrowed,
            });
        }
    }

    (per_protein, per_residue)
}

// ── Components ──

#[component]
pub fn PredictionView(
    sequence: String,
    model_predictions: Vec<ModelPredictionData>,
    task_labels: Option<TaskLabels>,
    global_track_stats: GlobalTrackStats,
) -> Element {
    let (per_protein, per_residue) =
        group_tasks_across_models(&model_predictions, task_labels.as_ref());

    let has_protein = !per_protein.is_empty();
    let has_residue = !per_residue.is_empty();

    let multi_model = model_predictions.len() > 1;

    rsx! {
        div {
            class: "flex flex-col gap-4",

            if has_protein {
                PerProteinSummary {
                    tasks: per_protein
                        .iter()
                        .map(|t| ProteinTaskGroupData {
                            name: t.name.clone(),
                            description: t.description.clone(),
                            problem_type: t.problem_type.clone(),
                            class_labels: t.class_labels.clone(),
                            unit: t.unit.clone(),
                            model_values: t.model_values.iter().map(|(l, v)| (l.to_string(), v.clone())).collect(),
                        })
                        .collect::<Vec<_>>(),
                    multi_model: multi_model,
                }
            }

            if has_residue {
                MultiModelTrackViewer {
                    sequence: sequence.clone(),
                    global_stats: global_track_stats.clone(),
                    task_groups: per_residue
                        .into_iter()
                        .map(|t| ResidueTaskGroupData {
                            name: t.name,
                            description: t.description,
                            problem_type: t.problem_type,
                            num_outputs: t.num_outputs,
                            class_labels: t.class_labels,
                            unit: t.unit,
                            model_values: t.model_values,
                        })
                        .collect::<Vec<_>>(),
                    multi_model: multi_model,
                }
            }
        }
    }
}

// ── Per-protein summary ──

/// Owned data for a protein-level task group (passed as a prop).
#[derive(Clone, PartialEq)]
struct ProteinTaskGroupData {
    name: String,
    description: String,
    problem_type: String,
    class_labels: Option<Vec<String>>,
    unit: Option<String>,
    /// (model_label, values) for each model.
    model_values: Vec<(String, Vec<f32>)>,
}

#[component]
fn PerProteinSummary(tasks: Vec<ProteinTaskGroupData>, multi_model: bool) -> Element {
    rsx! {
        div {
            class: "rounded-md border border-border-subtle bg-surface p-3 flex flex-col gap-3 max-w-md",
            div {
                class: "text-xs font-medium opacity-50 uppercase tracking-wide",
                "Protein-level predictions"
            }
            for (ti, task) in tasks.iter().enumerate() {
                div {
                    key: "{ti}",
                    class: "flex flex-col gap-1",

                    // Task name header (only when multiple tasks)
                    if tasks.len() > 1 {
                        div {
                            class: "text-[11px] opacity-40",
                            "{task.description}"
                        }
                    }

                    for (mi, (model_label, values)) in task.model_values.iter().enumerate() {
                        ProteinTaskRow {
                            key: "{mi}",
                            problem_type: task.problem_type.clone(),
                            description: if tasks.len() <= 1 { task.description.clone() } else { String::new() },
                            values: values.clone(),
                            class_labels: task.class_labels.clone(),
                            unit: task.unit.clone(),
                            model_label: if multi_model { Some(model_label.clone()) } else { None },
                        }
                    }
                }

                // Divider between task groups
                if ti + 1 < tasks.len() {
                    div {
                        class: "border-t border-border-subtle",
                    }
                }
            }
        }
    }
}

#[component]
fn ProteinTaskRow(
    problem_type: String,
    description: String,
    values: Vec<f32>,
    class_labels: Option<Vec<String>>,
    unit: Option<String>,
    model_label: Option<String>,
) -> Element {
    if problem_type == "classification" && values.len() > 1 {
        let probs = softmax(&values);
        let argmax = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let confidence = probs.get(argmax).copied().unwrap_or(0.0);
        let label = class_labels
            .as_ref()
            .and_then(|cl| cl.get(argmax))
            .cloned()
            .unwrap_or_else(|| format!("Class {argmax}"));
        let color = classification_color(argmax, &label);
        let pct = format!("{:.1}%", confidence * 100.0);

        rsx! {
            div {
                class: "flex items-center justify-between gap-3",
                div {
                    class: "flex items-center gap-2 min-w-0",
                    if let Some(ref ml) = model_label {
                        span {
                            class: "text-[10px] opacity-40 shrink-0",
                            "{ml}"
                        }
                    }
                    if !description.is_empty() {
                        span {
                            class: "text-xs opacity-60 shrink-0",
                            "{description}"
                        }
                    }
                }
                div {
                    class: "flex items-center gap-2 shrink-0",
                    span {
                        class: "inline-block w-3 h-3 rounded-sm shrink-0",
                        style: "background-color: {color};",
                    }
                    span {
                        class: "text-sm font-medium",
                        "{label}"
                    }
                    span {
                        class: "text-xs opacity-40",
                        "({pct})"
                    }
                }
            }
        }
    } else {
        let value = values.first().copied().unwrap_or(0.0);
        let unit_str = unit.as_deref().unwrap_or("");
        let value_str = format_display_value(value);

        rsx! {
            div {
                class: "flex items-center justify-between gap-3",
                div {
                    class: "flex items-center gap-2 min-w-0",
                    if let Some(ref ml) = model_label {
                        span {
                            class: "text-[10px] opacity-40 shrink-0",
                            "{ml}"
                        }
                    }
                    if !description.is_empty() {
                        span {
                            class: "text-xs opacity-60 shrink-0",
                            "{description}"
                        }
                    }
                }
                span {
                    class: "text-sm font-medium font-mono shrink-0",
                    "{value_str} {unit_str}"
                }
            }
        }
    }
}

// ── Multi-model sequence track viewer ──

/// Owned data for a per-residue task group (passed as a prop).
#[derive(Clone, PartialEq)]
struct ResidueTaskGroupData {
    name: String,
    description: String,
    problem_type: String,
    num_outputs: usize,
    class_labels: Option<Vec<String>>,
    unit: Option<String>,
    /// (model_label, values) for each model.
    model_values: Vec<(String, Vec<f32>)>,
}

#[derive(Clone, PartialEq)]
struct DecodedTrack {
    name: String,
    /// Short label for the sticky column.
    short_label: String,
    /// Full description including unit (shown in title tooltip and legend).
    full_description: String,
    problem_type: String,
    cells: Vec<CellData>,
    legend: Vec<LegendEntry>,
}

#[derive(Clone, PartialEq)]
struct CellData {
    color: String,
    text_color: &'static str,
    label: String,
    detail: ResidueDetail,
}

#[derive(Clone, PartialEq)]
struct LegendEntry {
    color: String,
    label: String,
}

/// A group of decoded tracks (one per model) for a single task, plus
/// whether it's the last group (used for divider rendering).
struct DecodedTaskGroup {
    tracks: Vec<DecodedTrack>,
}

#[component]
fn MultiModelTrackViewer(
    sequence: String,
    task_groups: Vec<ResidueTaskGroupData>,
    global_stats: GlobalTrackStats,
    multi_model: bool,
) -> Element {
    let seq_chars: Vec<char> = sequence.chars().collect();
    let seq_len = seq_chars.len();
    let grid_w = seq_len as u32 * CELL_W;

    // Sorted regression task names for stable accent color assignment.
    let mut regression_names: Vec<&str> = global_stats.keys().map(|s| s.as_str()).collect();
    regression_names.sort();

    // Decode all task groups into tracks.
    let decoded_groups: Vec<DecodedTaskGroup> = task_groups
        .iter()
        .map(|tg| {
            let accent_idx = regression_names
                .iter()
                .position(|&n| n == tg.name)
                .unwrap_or(0);
            let accent = regression_accent(accent_idx);

            let tracks: Vec<DecodedTrack> = tg
                .model_values
                .iter()
                .map(|(model_label, values)| {
                    let label_prefix = if multi_model {
                        Some(model_label.as_str())
                    } else {
                        None
                    };
                    decode_track(
                        &tg.name,
                        &tg.description,
                        &tg.problem_type,
                        tg.num_outputs,
                        values,
                        tg.class_labels.as_ref(),
                        tg.unit.as_deref(),
                        label_prefix,
                        seq_len,
                        global_stats.get(&tg.name).copied(),
                        accent,
                    )
                })
                .collect();

            DecodedTaskGroup { tracks }
        })
        .collect();

    // Total track rows across all groups.
    let total_track_rows: u32 = decoded_groups.iter().map(|g| g.tracks.len() as u32).sum();
    // 2 header rows (position + sequence) + all track rows.
    let num_rows = 2 + total_track_rows;
    let total_h = num_rows * CELL_H;

    rsx! {
        div {
            class: "flex flex-col gap-3",

            div {
                class: "relative flex border border-border-subtle rounded-md overflow-hidden",

                // Sticky label column
                div {
                    class: "shrink-0 bg-surface z-10 border-r border-border-subtle",
                    style: "width: {LABEL_COL_W}px;",

                    div {
                        class: "flex items-center px-2 text-[10px] opacity-40 font-mono border-b border-border-subtle truncate",
                        style: "height: {CELL_H}px;",
                        "Position"
                    }
                    div {
                        class: "flex items-center px-2 text-[10px] opacity-40 font-mono border-b border-border-subtle truncate",
                        style: "height: {CELL_H}px;",
                        "Sequence"
                    }
                    for (gi, group) in decoded_groups.iter().enumerate() {
                        for (ti, track) in group.tracks.iter().enumerate() {
                            {
                                // Last track in a non-last group gets a thicker
                                // bottom border as a task-group divider.
                                let is_last_in_group = ti == group.tracks.len() - 1;
                                let is_last_group = gi == decoded_groups.len() - 1;
                                let border_class = if is_last_in_group && !is_last_group {
                                    "border-b-2 border-border-subtle"
                                } else {
                                    "border-b border-border-subtle"
                                };

                                rsx! {
                                    div {
                                        key: "{gi}-{ti}",
                                        class: "flex items-center px-2 text-[10px] font-medium {border_class} truncate",
                                        style: "height: {CELL_H}px;",
                                        title: "{track.full_description}",
                                        "{track.short_label}"
                                    }
                                }
                            }
                        }
                    }
                }

                // Scrollable track area
                div {
                    class: "overflow-x-auto overflow-y-hidden flex-1 pb-2",
                    style: "scrollbar-gutter: stable;",

                    div {
                        class: "relative",
                        style: "width: {grid_w}px; height: {total_h}px;",

                        // Position ruler row
                        div {
                            class: "flex",
                            style: "height: {CELL_H}px;",
                            for i in 0..seq_len {
                                {
                                    let pos = i + 1;
                                    let show_label = pos == 1 || pos % RULER_TICK_INTERVAL == 0;
                                    rsx! {
                                        div {
                                            key: "{i}",
                                            class: "shrink-0 flex items-end justify-center text-[9px] font-mono opacity-30 border-b border-border-subtle",
                                            style: "width: {CELL_W}px; height: {CELL_H}px;",
                                            if show_label {
                                                "{pos}"
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Sequence row
                        div {
                            class: "flex",
                            style: "height: {CELL_H}px;",
                            for (i, &ch) in seq_chars.iter().enumerate() {
                                div {
                                    key: "{i}",
                                    class: "shrink-0 flex items-center justify-center text-[11px] font-mono font-medium border-b border-border-subtle",
                                    style: "width: {CELL_W}px; height: {CELL_H}px;",
                                    "{ch}"
                                }
                            }
                        }

                        // Track rows grouped by task
                        for (gi, group) in decoded_groups.iter().enumerate() {
                            for (ti, dt) in group.tracks.iter().enumerate() {
                                {
                                    let is_last_in_group = ti == group.tracks.len() - 1;
                                    let is_last_group = gi == decoded_groups.len() - 1;
                                    let border_class = if is_last_in_group && !is_last_group {
                                        "border-b-2 border-border-subtle"
                                    } else {
                                        "border-b border-border-subtle"
                                    };

                                    rsx! {
                                        div {
                                            key: "{gi}-{ti}",
                                            class: "flex",
                                            style: "height: {CELL_H}px;",
                                            for i in 0..seq_len {
                                                {
                                                    let cell = &dt.cells[i];
                                                    let bg = cell.color.clone();
                                                    let text_color = cell.text_color;
                                                    let cell_text = cell.label.clone();
                                                    let position = i + 1;
                                                    let residue = seq_chars[i];
                                                    let task_name = dt.name.clone();
                                                    let detail = cell.detail.clone();
                                                    let is_last_cell = i == seq_len - 1;

                                                    // Apply group divider on the last cell row
                                                    let cell_border = if is_last_cell {
                                                        border_class
                                                    } else {
                                                        "border-b border-border-subtle"
                                                    };
                                                    let _ = cell_border;

                                                    let tooltip_content = rsx! {
                                                        div {
                                                            class: "font-medium mb-1",
                                                            "Position {position} \u{2014} {residue}"
                                                        }
                                                        div {
                                                            class: "text-[10px] opacity-50 mb-1",
                                                            "{task_name}"
                                                        }

                                                        {match &detail {
                                                            ResidueDetail::Classification { predicted_class: _, label, probabilities } => rsx! {
                                                                div {
                                                                    class: "font-medium mb-1",
                                                                    "Predicted: {label}"
                                                                }
                                                                div {
                                                                    class: "flex flex-col gap-0.5",
                                                                    for (ci, (class_label, prob)) in probabilities.iter().enumerate() {
                                                                        div {
                                                                            class: "flex items-center gap-1.5",
                                                                            span {
                                                                                class: "inline-block w-2 h-2 rounded-sm shrink-0",
                                                                                style: "background-color: {classification_color(ci, class_label)};",
                                                                            }
                                                                            span { class: "opacity-60 truncate flex-1", "{class_label}" }
                                                                            span { class: "font-mono shrink-0", "{prob:.3}" }
                                                                        }
                                                                    }
                                                                }
                                                            },
                                                            ResidueDetail::Regression { value, unit } => rsx! {
                                                                div {
                                                                    class: "flex items-center gap-1.5 font-mono",
                                                                    span {
                                                                        class: "inline-block w-2 h-2 rounded-sm shrink-0",
                                                                        style: "background-color: {bg};",
                                                                    }
                                                                    "Value: {format_display_value(*value)} {unit}"
                                                                }
                                                            },
                                                        }}
                                                    };

                                                    rsx! {
                                                        Tooltip {
                                                            key: "{i}",
                                                            content: tooltip_content,

                                                            div {
                                                                class: "shrink-0 flex items-center justify-center text-[9px] font-mono cursor-crosshair {border_class}",
                                                                style: "width: {CELL_W}px; height: {CELL_H}px; background-color: {bg}; color: {text_color};",
                                                                if !cell_text.is_empty() {
                                                                    "{cell_text}"
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── Decoded track cells ──

/// Decode a track's raw values into per-cell rendering data.
///
/// `global_range` overrides per-sequence min/max for regression color
/// normalization, so all sequences share the same color scale.
/// `accent` is the bright target color for the sequential ramp on
/// regression tracks.
/// `label_prefix` is prepended to the sticky label when showing multiple
/// models (e.g. "ModelA" → "Binding (ModelA)").
fn decode_track(
    task_name: &str,
    description: &str,
    problem_type: &str,
    num_outputs: usize,
    values: &[f32],
    class_labels_opt: Option<&Vec<String>>,
    unit_opt: Option<&str>,
    label_prefix: Option<&str>,
    seq_len: usize,
    global_range: Option<(f32, f32)>,
    accent: (f32, f32, f32),
) -> DecodedTrack {
    let base_label = if description.is_empty() {
        task_name.to_string()
    } else {
        description.to_string()
    };

    let short_label = match label_prefix {
        Some(prefix) => format!("{base_label} ({prefix})"),
        None => base_label.clone(),
    };

    if problem_type == "classification" && num_outputs > 1 {
        let num_classes = num_outputs;
        let class_labels: Vec<String> = (0..num_classes)
            .map(|i| {
                class_labels_opt
                    .and_then(|cl| cl.get(i))
                    .cloned()
                    .unwrap_or_else(|| format!("Class {i}"))
            })
            .collect();

        let cells: Vec<CellData> = (0..seq_len)
            .map(|r| {
                let start = r * num_classes;
                let end = (start + num_classes).min(values.len());
                let logits = if start < values.len() {
                    &values[start..end]
                } else {
                    &[]
                };
                let probs = softmax(logits);
                let argmax = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let color = classification_color(argmax, &class_labels[argmax]).to_string();
                let probabilities: Vec<(String, f32)> = class_labels
                    .iter()
                    .zip(probs.iter())
                    .map(|(l, &p)| (l.clone(), p))
                    .collect();
                CellData {
                    color,
                    text_color: "#ffffff",
                    label: String::new(),
                    detail: ResidueDetail::Classification {
                        predicted_class: argmax,
                        label: class_labels[argmax].clone(),
                        probabilities,
                    },
                }
            })
            .collect();

        let legend: Vec<LegendEntry> = class_labels
            .iter()
            .enumerate()
            .map(|(i, l)| LegendEntry {
                color: classification_color(i, l).to_string(),
                label: l.clone(),
            })
            .collect();

        DecodedTrack {
            name: task_name.to_string(),
            full_description: short_label.clone(),
            short_label,
            problem_type: problem_type.to_string(),
            cells,
            legend,
        }
    } else {
        // Regression: heatmap
        let unit = unit_opt.unwrap_or("").to_string();
        let is_diverging = DIVERGING_TASKS.contains(&task_name);
        let vals: Vec<f32> = (0..seq_len)
            .map(|i| values.get(i).copied().unwrap_or(0.0))
            .collect();
        let (min, max) = global_range.unwrap_or_else(|| {
            let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            (min, max)
        });
        let range = max - min;

        let cells: Vec<CellData> = vals
            .iter()
            .map(|&v| {
                let color = if is_diverging {
                    let t = diverging_t(v, min, max);
                    diverging_color(t)
                } else {
                    let t = if range > 0.0 { (v - min) / range } else { 0.5 };
                    sequential_color(t, accent)
                };
                CellData {
                    color,
                    text_color: "#ffffff",
                    label: String::new(),
                    detail: ResidueDetail::Regression {
                        value: v,
                        unit: unit.clone(),
                    },
                }
            })
            .collect();

        let legend = if is_diverging {
            let abs_max = min.abs().max(max.abs());
            vec![
                LegendEntry {
                    color: diverging_color(0.0),
                    label: format_display_value(-abs_max),
                },
                LegendEntry {
                    color: diverging_color(0.5),
                    label: "0".to_string(),
                },
                LegendEntry {
                    color: diverging_color(1.0),
                    label: format_display_value(abs_max),
                },
            ]
        } else {
            vec![
                LegendEntry {
                    color: sequential_color(0.0, accent),
                    label: format_display_value(min),
                },
                LegendEntry {
                    color: sequential_color(0.5, accent),
                    label: "mid".to_string(),
                },
                LegendEntry {
                    color: sequential_color(1.0, accent),
                    label: format_display_value(max),
                },
            ]
        };

        let full_desc = if unit.is_empty() {
            short_label.clone()
        } else {
            format!("{short_label} ({unit})")
        };

        DecodedTrack {
            name: task_name.to_string(),
            short_label,
            full_description: full_desc,
            problem_type: problem_type.to_string(),
            cells,
            legend,
        }
    }
}

// ── Global legend grid ──

/// Shared legend grid displayed once above pagination, showing one card
/// per task. Classification tasks show colored swatches; regression tasks
/// show a gradient bar with global min/max values.
#[component]
pub fn TrackLegendGrid(
    model_metadata: HashMap<String, ExportMetadata>,
    task_labels: Signal<Option<TaskLabels>>,
    global_track_stats: GlobalTrackStats,
) -> Element {
    let labels_guard = task_labels.read();

    // Merge per-residue tasks across all models, deduplicating by name.
    // For config, take the first model's config encountered for each task.
    let mut residue_task_configs: HashMap<String, &TaskConfig> = HashMap::new();
    for meta in model_metadata.values() {
        for (task_name, task_config) in &meta.tasks {
            if task_config.level == "per_residue" {
                residue_task_configs
                    .entry(task_name.clone())
                    .or_insert(task_config);
            }
        }
    }

    let mut residue_tasks: Vec<String> = residue_task_configs.keys().cloned().collect();
    residue_tasks.sort();

    if residue_tasks.is_empty() {
        return rsx! {};
    }

    // Sorted regression task names for stable accent color assignment.
    let mut regression_names: Vec<&str> = global_track_stats.keys().map(|s| s.as_str()).collect();
    regression_names.sort();

    let mut is_open = use_signal(|| false);
    let open = *is_open.read();
    let toggle_icon = if open { "\u{25BC}" } else { "\u{25B6}" };

    rsx! {
        div {
            class: "border-t border-border-subtle",

            // Toggle header
            button {
                class: "w-full flex items-center gap-2 px-5 py-2 text-left hover:bg-surface-elevated transition-colors cursor-pointer",
                onclick: move |_| {
                    let current = *is_open.peek();
                    is_open.set(!current);
                },
                span {
                    class: "text-[10px] opacity-40 shrink-0 w-3",
                    "{toggle_icon}"
                }
                span {
                    class: "text-xs opacity-50 select-none",
                    "Legend"
                }
            }

            // Collapsible grid
            if open {
                div {
                    class: "grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2 px-5 pb-3",
                    for task_name in residue_tasks.iter() {
                {
                    let config = residue_task_configs[task_name.as_str()];
                    let label_info = labels_guard.as_ref().and_then(|tl| tl.get(task_name.as_str()));
                    let description = label_info
                        .map(|l| l.description.clone())
                        .unwrap_or_else(|| task_name.to_string());
                    let unit = label_info.and_then(|l| l.unit.clone());

                    if config.problem_type == "classification" {
                        let class_labels: Vec<String> = (0..config.num_outputs)
                            .map(|i| {
                                label_info
                                    .and_then(|l| l.class_labels.as_ref())
                                    .and_then(|cl| cl.get(i))
                                    .cloned()
                                    .unwrap_or_else(|| format!("Class {i}"))
                            })
                            .collect();

                        rsx! {
                            div {
                                key: "{task_name}",
                                class: "rounded-md border border-border-subtle bg-surface p-2.5 flex flex-col gap-1.5",
                                span {
                                    class: "text-[10px] font-medium opacity-60 truncate",
                                    title: "{description}",
                                    "{description}"
                                }
                                div {
                                    class: "flex flex-col gap-0.5",
                                    for (i, class_label) in class_labels.iter().enumerate() {
                                        div {
                                            key: "{i}",
                                            class: "flex items-center gap-1.5",
                                            span {
                                                class: "inline-block w-2.5 h-2.5 rounded-sm shrink-0",
                                                style: "background-color: {classification_color(i, class_label)};",
                                            }
                                            span {
                                                class: "text-[10px] opacity-60 truncate",
                                                "{class_label}"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Regression: gradient bar with global min/max
                        let (min_val, max_val) = global_track_stats
                            .get(task_name.as_str())
                            .copied()
                            .unwrap_or((0.0, 1.0));
                        let is_diverging = DIVERGING_TASKS.contains(&task_name.as_str());
                        let unit_str = unit.as_deref().unwrap_or("");
                        let full_desc = if unit_str.is_empty() {
                            description.clone()
                        } else {
                            format!("{description} ({unit_str})")
                        };

                        let (c_low, c_mid, c_high, label_low, label_high) = if is_diverging {
                            let abs_max = min_val.abs().max(max_val.abs());
                            (
                                diverging_color(0.0),
                                diverging_color(0.5),
                                diverging_color(1.0),
                                format!("{:.2}", -abs_max),
                                format!("{abs_max:.2}"),
                            )
                        } else {
                            let accent_idx = regression_names.iter().position(|&n| n == task_name.as_str()).unwrap_or(0);
                            let acc = regression_accent(accent_idx);
                            (
                                sequential_color(0.0, acc),
                                sequential_color(0.5, acc),
                                sequential_color(1.0, acc),
                                format!("{min_val:.2}"),
                                format!("{max_val:.2}"),
                            )
                        };

                        rsx! {
                            div {
                                key: "{task_name}",
                                class: "rounded-md border border-border-subtle bg-surface p-2.5 flex flex-col gap-1.5",
                                span {
                                    class: "text-[10px] font-medium opacity-60 truncate",
                                    title: "{full_desc}",
                                    "{full_desc}"
                                }
                                div {
                                    class: "flex items-center gap-1",
                                    span {
                                        class: "text-[9px] opacity-40 font-mono shrink-0",
                                        "{label_low}"
                                    }
                                    div {
                                        class: "flex-1 h-3 rounded-sm",
                                        style: "background: linear-gradient(to right, {c_low}, {c_mid}, {c_high});",
                                    }
                                    span {
                                        class: "text-[9px] opacity-40 font-mono shrink-0",
                                        "{label_high}"
                                    }
                                }
                            }
                        }
                    }
                }
                    }
                }
            }
        }
    }
}
