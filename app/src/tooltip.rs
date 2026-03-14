/// Self-positioning tooltip that wraps a trigger element.
///
/// The tooltip measures its own wrapper via `MountedData::get_client_rect()`
/// on hover, then positions itself to the right of the trigger (flipping
/// left near the viewport edge) with a connecting arrow.  Content is
/// rendered through a portal so the DOM sits at the root of the tree,
/// escaping all overflow / stacking-context containers.
use std::rc::Rc;

use dioxus::prelude::*;

use crate::portal::PortalIn;

/// Bounding rect returned by `get_client_rect()`, stored for positioning.
#[derive(Clone, Copy)]
struct Rect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

/// Estimated tooltip width for flip calculations (before measurement).
const EST_W: f64 = 220.0;
/// Gap between the trigger edge and the tooltip body.
const GAP: f64 = 8.0;
/// Arrow square side length.
const ARROW: f64 = 10.0;

/// A self-positioning tooltip.
///
/// Wrap a trigger element as `children`.  On hover the popup appears
/// to the right (or left near the viewport edge) with an arrow pointing
/// at the trigger.
///
/// ```rust,ignore
/// Tooltip {
///     content: rsx! { "Hello" },
///     div { "hover me" }
/// }
/// ```
#[component]
pub fn Tooltip(
    /// The popup body shown on hover.
    content: Element,
    /// The trigger element rendered in-place.
    children: Element,
) -> Element {
    let mut el_ref: Signal<Option<Rc<MountedData>>> = use_signal(|| None);
    let mut visible = use_signal(|| false);
    let mut pos = use_signal(|| None::<(Rect, f64, f64)>);

    rsx! {
        // The wrapper must not interfere with the parent's flex layout.
        // `shrink-0` prevents flex shrinking; the wrapper takes its size
        // from the child (the trigger element).
        div {
            class: "shrink-0",
            onmounted: move |evt: MountedEvent| {
                el_ref.set(Some(evt.data()));
            },
            onmouseenter: move |_| {
                visible.set(true);
                spawn(async move {
                    let (ref_rect, vw, vh) = measure(el_ref).await;
                    if let Some(r) = ref_rect {
                        pos.set(Some((r, vw, vh)));
                    }
                });
            },
            onmouseleave: move |_| {
                visible.set(false);
                pos.set(None);
            },
            {children}
        }

        if visible() {
            if let Some((ref_rect, vw, vh)) = *pos.read() {
                {render_popup(ref_rect, vw, vh, content)}
            }
        }
    }
}

/// Measure the trigger element and viewport.
async fn measure(
    el_ref: Signal<Option<Rc<MountedData>>>,
) -> (Option<Rect>, f64, f64) {
    let el = match el_ref.cloned() {
        Some(el) => el,
        None => return (None, 0.0, 0.0),
    };

    let rect = match el.get_client_rect().await {
        Ok(r) => Rect {
            x: r.origin.x,
            y: r.origin.y,
            w: r.size.width,
            h: r.size.height,
        },
        Err(_) => return (None, 0.0, 0.0),
    };

    // Read viewport dimensions via JS (no cross-platform alternative).
    let (vw, vh) = match document::eval("return [window.innerWidth, window.innerHeight];").await {
        Ok(val) => {
            let w = val.get(0).and_then(|v| v.as_f64()).unwrap_or(2000.0);
            let h = val.get(1).and_then(|v| v.as_f64()).unwrap_or(2000.0);
            (w, h)
        }
        Err(_) => (2000.0, 2000.0),
    };

    (Some(rect), vw, vh)
}

/// Render the positioned popup through the portal.
fn render_popup(r: Rect, vw: f64, vh: f64, content: Element) -> Element {
    // Horizontal: prefer right of trigger, flip left if overflowing.
    let flip = r.x + r.w + GAP + EST_W > vw;

    // When flipped, anchor the tooltip's right edge to the trigger's left
    // edge using CSS `right` so the position is independent of the
    // tooltip's rendered width.
    let h_style = if flip {
        let right_offset = (vw - r.x + GAP).max(0.0);
        format!("right: {right_offset}px; left: auto;")
    } else {
        let left_offset = r.x + r.w + GAP;
        format!("left: {left_offset}px; right: auto;")
    };

    // Vertical: center on the trigger, clamped to viewport.
    let est_h = 120.0_f64;
    let tip_y = (r.y + r.h / 2.0 - est_h / 2.0).clamp(4.0, (vh - est_h - 4.0).max(4.0));

    // Arrow: on the edge facing the trigger, vertically at trigger center.
    let arrow_y = (r.y + r.h / 2.0 - tip_y - ARROW / 2.0).clamp(8.0, est_h - 8.0 - ARROW);

    // Arrow CSS — a rotated square with two borders visible.
    let (arrow_x_style, arrow_border) = if flip {
        // Tooltip is LEFT of trigger → arrow on right edge pointing right
        (
            format!("right: -{}px;", ARROW / 2.0),
            "border-right-width: 1px; border-bottom-width: 1px;",
        )
    } else {
        // Tooltip is RIGHT of trigger → arrow on left edge pointing left
        (
            format!("left: -{}px;", ARROW / 2.0),
            "border-left-width: 1px; border-top-width: 1px;",
        )
    };

    rsx! {
        PortalIn {
            div {
                class: "fixed z-[9999] pointer-events-none",
                style: "{h_style} top: {tip_y}px;",

                div {
                    class: "relative rounded-md border border-border-subtle bg-surface-elevated shadow-lg p-2 text-xs",
                    style: "min-width: 180px; max-width: 260px;",

                    // Arrow
                    div {
                        class: "absolute bg-surface-elevated",
                        style: "width: {ARROW}px; height: {ARROW}px; {arrow_x_style} top: {arrow_y}px; transform: rotate(45deg); border-color: var(--color-border-subtle); border-style: solid; {arrow_border}",
                    }

                    {content}
                }
            }
        }
    }
}
