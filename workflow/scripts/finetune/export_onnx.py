"""Snakemake script: export a trained MultitaskModel to ONNX.

Loads the model checkpoint, resolves the tokenizer from the base
backbone, and delegates to :func:`mutafitup.export_onnx.export_to_onnx`.
"""

import os

import torch
from snakemake.script import snakemake

from mutafitup.export_onnx import export_to_onnx
from wfutils.logging import get_logger, log_snakemake_info


logger = get_logger(__name__)
log_snakemake_info(logger)

model_dir = snakemake.input["model_dir"]
output_dir = os.path.dirname(str(snakemake.output["metadata"]))

model_path = os.path.join(model_dir, "model.pt")

# ------------------------------------------------------------------
# Resolve tokenizer from saved checkpoint metadata
# ------------------------------------------------------------------
saved_data = torch.load(model_path, map_location="cpu", weights_only=False)
base_checkpoint = saved_data["config"]["base_checkpoint"]

backbone_class_path = saved_data["config"]["backbone_class"]
module_name, attr_name = backbone_class_path.rsplit(".", 1)
backbone_module = __import__(module_name, fromlist=[attr_name])
backbone_cls = getattr(backbone_module, attr_name)

logger.info("Loading tokenizer for %s from %s", backbone_class_path, base_checkpoint)
_, tokenizer = backbone_cls.from_pretrained(
    checkpoint=base_checkpoint,
    lora_config=None,
)

# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------
onnx_path = export_to_onnx(
    model_path=model_path,
    output_dir=output_dir,
    tokenizer=tokenizer,
    base_checkpoint=base_checkpoint,
    validate=True,
    logger=logger,
)

logger.info("ONNX export complete: %s", onnx_path)
