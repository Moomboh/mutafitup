"""Snakemake script: generate a model manifest (models.json) for the app.

Reads ``export_metadata.json`` from each ONNX export directory and combines
it with the pipeline config to produce a manifest that the Dioxus app uses
to populate its model selector.
"""

import json
import os

from snakemake.script import snakemake

from mutafitup.hf_naming import BACKBONE_HF_PREFIXES, hf_model_name
from wfutils.logging import get_logger, log_snakemake_info


logger = get_logger(__name__)
log_snakemake_info(logger)

onnx_export_entries = snakemake.params["onnx_export_entries"]
onnx_export_base = snakemake.params["onnx_export_base"]
output_path = str(snakemake.output["manifest"])

# Human-readable labels for training sections (fallback for non-HF backbones)
SECTION_LABELS = {
    "heads_only": "Heads Only",
    "lora": "LoRA",
    "accgrad_lora": "AccGrad LoRA",
    "align_lora": "Align LoRA",
}

# Human-readable short names for known backbones (fallback for non-HF backbones)
BACKBONE_SHORT_NAMES = {
    "esmc_300m": "ESM-C 300M",
    "facebook/esm2_t6_8M_UR50D": "ESM-2 8M",
    "facebook/esm2_t33_650M_UR50D": "ESM-2 650M",
    "Rostlab/prot_t5_xl_uniref50": "ProtT5-XL",
}

VARIANT_LABELS = {
    "best_overall": "best",
    "final": "final",
}

models = []

for entry in onnx_export_entries:
    section = entry["section"]
    run_id = entry["id"]
    variant = entry["variant"]

    model_id = f"{section}/{run_id}/{variant}"
    export_dir = os.path.join(onnx_export_base, model_id)
    metadata_path = os.path.join(export_dir, "export_metadata.json")

    if not os.path.exists(metadata_path):
        logger.warning("Skipping %s: export_metadata.json not found", model_id)
        continue

    with open(metadata_path) as f:
        metadata = json.load(f)

    backbone = metadata["base_checkpoint"]
    tasks = sorted(metadata["tasks"].keys())

    # Use HF naming convention for supported backbones, otherwise fall
    # back to the descriptive label format.
    if backbone in BACKBONE_HF_PREFIXES:
        label = hf_model_name(backbone, section, run_id, variant)
    else:
        backbone_short = BACKBONE_SHORT_NAMES.get(backbone, backbone)
        section_label = SECTION_LABELS.get(section, section)
        variant_label = VARIANT_LABELS.get(variant, variant)

        n_tasks = len(tasks)
        if n_tasks == 1:
            task_summary = tasks[0]
        elif n_tasks <= 3:
            task_summary = ", ".join(tasks)
        else:
            task_summary = f"{n_tasks} tasks"

        label = f"{backbone_short} — {section_label} ({task_summary}, {variant_label})"

    # Models with external data files exceed the 2 GB protobuf limit and
    # cannot be loaded by onnxruntime-web.
    external_data_path = os.path.join(export_dir, "model.onnx.data")
    web_compatible = not os.path.exists(external_data_path)

    models.append(
        {
            "id": model_id,
            "label": label,
            "section": section,
            "run_id": run_id,
            "variant": variant,
            "backbone": backbone,
            "tasks": tasks,
            "web_compatible": web_compatible,
        }
    )

manifest = {"models": models}

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")

logger.info("Wrote model manifest with %d model(s) to %s", len(models), output_path)
