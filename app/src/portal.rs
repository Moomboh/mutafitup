/// Lightweight portal for teleporting element content to a different
/// location in the component tree.
///
/// Usage:
/// 1. Call `use_context_provider(TooltipPortal::new)` in a long-lived
///    ancestor (e.g. the app root).
/// 2. Place `PortalOut {}` where the content should physically appear
///    in the DOM (typically at the root, outside overflow containers).
/// 3. Wrap content with `PortalIn { children }` wherever it is
///    logically produced — its children will render at the `PortalOut`
///    location instead.
///
/// This module intentionally supports only a **single** shared portal
/// (the tooltip portal) to keep the API surface small.
use dioxus::prelude::*;

/// Shared state for the tooltip portal.
///
/// Provided via `use_context_provider` in the app root.
#[derive(Clone, Copy)]
pub struct TooltipPortal {
    content: Signal<Option<Element>>,
}

impl TooltipPortal {
    pub fn new() -> Self {
        Self {
            content: Signal::new(None),
        }
    }
}

/// Send children into the tooltip portal.
///
/// Renders nothing in-place — the children appear wherever
/// `PortalOut` is mounted.
#[component]
pub fn PortalIn(children: Element) -> Element {
    let mut portal = use_context::<TooltipPortal>();

    // Set content on every render (mount AND prop updates).
    // Writing to `portal.content` triggers a re-render of PortalOut
    // (which reads the signal), but does NOT re-trigger PortalIn
    // (which only writes), so there is no render loop.
    portal.content.set(Some(children));

    // Clear the portal when this component unmounts.
    use_drop(move || {
        portal.content.set(None);
    });

    rsx! {}
}

/// Render whatever content was sent via `PortalIn`.
///
/// Place this at the root of the component tree so the rendered
/// DOM sits outside any overflow / transform / stacking-context
/// containers.
#[component]
pub fn PortalOut() -> Element {
    let portal = use_context::<TooltipPortal>();

    // Clone the content out of the signal to avoid borrow lifetime issues.
    let content = (portal.content)();
    match content {
        Some(el) => rsx! { {el} },
        None => rsx! {},
    }
}
