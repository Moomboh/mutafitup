/// Sequence input component: textarea for pasting + file upload button.
use dioxus::prelude::*;

use crate::components::button::{Button, ButtonVariant};
use crate::components::textarea::Textarea;

#[component]
pub fn SequenceInput(
    mut text_value: Signal<String>,
    on_load: EventHandler<()>,
    disabled: bool,
) -> Element {
    rsx! {
        div {
            class: "flex flex-col gap-3",

            label {
                class: "text-sm font-medium",
                r#for: "seq-input",
                "Protein sequences"
            }

            Textarea {
                id: "seq-input",
                placeholder: "Paste a protein sequence or FASTA here...\n\n>sp|P12345|PROT_HUMAN Example protein\nMAKLVFGPDHELLOWORLD...",
                rows: "6",
                value: "{text_value}",
                disabled: disabled,
                oninput: move |e: FormEvent| {
                    text_value.set(e.value());
                },
            }

            div {
                class: "flex flex-col gap-2",

                // File upload via hidden native <input type="file">
                label {
                    class: "cursor-pointer w-full",
                    input {
                        r#type: "file",
                        accept: ".fasta,.fa,.fna,.txt,.faa",
                        class: "hidden",
                        disabled: disabled,
                        onchange: move |e: FormEvent| {
                            spawn(async move {
                                let files = e.files();
                                if let Some(file) = files.first() {
                                    if let Ok(text) = file.read_string().await {
                                        text_value.set(text);
                                    }
                                }
                            });
                        },
                    }
                    Button {
                        variant: ButtonVariant::Outline,
                        // pointer-events: none so the click goes through to the
                        // parent <label> which triggers the hidden file input.
                        style: "pointer-events: none; width: 100%;",
                        disabled: disabled,
                        "Upload FASTA"
                    }
                }

                Button {
                    onclick: move |_| on_load.call(()),
                    disabled: disabled || text_value.read().trim().is_empty(),
                    style: "width: 100%;",
                    "Load sequences"
                }
            }
        }
    }
}
