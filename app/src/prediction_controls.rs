/// Prediction controls: batch size + predict button (sidebar),
/// and prediction progress bar (toolbar).
use dioxus::prelude::*;

use crate::components::button::Button;
use crate::components::input::Input;
use crate::components::progress::{Progress, ProgressIndicator};

/// Format milliseconds as hh:mm:ss (omits hours when zero).
fn format_duration(ms: f64) -> String {
    let total_secs = (ms / 1000.0).round() as u64;
    let h = total_secs / 3600;
    let m = (total_secs % 3600) / 60;
    let s = total_secs % 60;
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

/// Inline progress bar with completion stats. Designed for the sequence list header.
#[component]
pub fn PredictionProgress(
    is_predicting: bool,
    completed: usize,
    total: usize,
    elapsed_ms: Option<f64>,
) -> Element {
    let progress_value = if total > 0 {
        Some((completed as f64 / total as f64) * 100.0)
    } else {
        None
    };

    let show = is_predicting || completed > 0;
    let is_done = !is_predicting && completed == total && total > 0;

    let eta_str = if is_predicting && completed > 0 {
        elapsed_ms.map(|ms| {
            let remaining = total.saturating_sub(completed);
            let ms_per_seq = ms / completed as f64;
            let eta_ms = ms_per_seq * remaining as f64;
            format_duration(eta_ms)
        })
    } else {
        None
    };

    if !show {
        return rsx! {};
    }

    rsx! {
        div {
            class: "flex flex-col gap-1 min-w-0 flex-1",

            Progress {
                value: progress_value,
                max: 100.0,
                style: "width: 100%;",
                ProgressIndicator {}
            }

            div {
                class: "flex items-center gap-3 text-xs opacity-60 whitespace-nowrap",

                span { "{completed} / {total}" }

                if let Some(ms) = elapsed_ms {
                    span { "Elapsed: {format_duration(ms)}" }
                }

                if let Some(ref eta) = eta_str {
                    span { "ETA: {eta}" }
                }

                if is_done {
                    if let Some(ms) = elapsed_ms {
                        span {
                            {
                                let per_seq = ms / total as f64;
                                let per_seq_str = if per_seq >= 1000.0 {
                                    format!("{:.2}s", per_seq / 1000.0)
                                } else {
                                    format!("{:.0}ms", per_seq)
                                };
                                format!("{per_seq_str}/seq")
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Batch size input + predict button. Designed for the sidebar.
#[component]
pub fn PredictionControls(
    mut batch_size: Signal<usize>,
    on_predict: EventHandler<()>,
    is_predicting: bool,
    can_predict: bool,
) -> Element {
    rsx! {
        div {
            class: "flex flex-col gap-3",

            div {
                class: "flex flex-col gap-1",
                label {
                    class: "text-xs font-medium opacity-60",
                    r#for: "batch-size",
                    "Batch size"
                }
                Input {
                    id: "batch-size",
                    r#type: "number",
                    min: "1",
                    max: "32",
                    value: "{batch_size}",
                    disabled: is_predicting,
                    oninput: move |e: FormEvent| {
                        if let Ok(v) = e.value().parse::<usize>() {
                            if v >= 1 {
                                batch_size.set(v);
                            }
                        }
                    },
                }
            }

            Button {
                class: "w-full",
                onclick: move |_| on_predict.call(()),
                disabled: !can_predict || is_predicting,
                if is_predicting {
                    "Predicting..."
                } else {
                    "Predict all"
                }
            }
        }
    }
}
