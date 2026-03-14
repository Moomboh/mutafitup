/// Centered modal overlay with backdrop. Renders slotted children inside
/// a dark card, and closes when the backdrop is clicked.
use dioxus::prelude::*;

#[component]
pub fn Modal(open: bool, on_close: EventHandler<()>, children: Element) -> Element {
    if !open {
        return rsx! {};
    }

    rsx! {
        div {
            class: "fixed inset-0 z-50 flex items-center justify-center",

            // Backdrop
            div {
                class: "absolute inset-0 bg-black/60",
                onclick: move |_| on_close.call(()),
            }

            // Content card
            div {
                class: "relative z-10 rounded-lg border border-border-subtle bg-surface-elevated p-6 shadow-xl max-w-md w-full mx-4",
                // Stop clicks inside the card from closing the modal
                onclick: move |evt| evt.stop_propagation(),
                {children}
            }
        }
    }
}
