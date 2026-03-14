"""HuggingFace model naming helpers for mutafitup ONNX exports.

Generates standardised model names that comply with the Cambrian Open
License requirement (Section 4.e) to include "ESM" at the beginning of
derivative work titles.

Naming convention::

    ESMC-300M-mutafitup-{section_short}-{run_suffix}-{variant_short}

Examples::

    >>> hf_model_name("esmc_300m", "accgrad_lora", "esmc_300m_all_r4", "best_overall")
    'ESMC-300M-mutafitup-accgrad-all-r4-best-overall'

    >>> hf_model_name("esmc_300m", "heads_only", "esmc_300m_all_heads_only", "best_overall")
    'ESMC-300M-mutafitup-heads-only-all-heads-only-best-overall'
"""

from __future__ import annotations

# Mapping from config backbone checkpoint name to the HF-compatible
# prefix used in model names.  Only ESM-C 300M is supported for now;
# extend as needed.
BACKBONE_HF_PREFIXES: dict[str, str] = {
    "esmc_300m": "ESMC-300M",
}

# Short identifiers for training sections.
SECTION_SHORT_NAMES: dict[str, str] = {
    "heads_only": "heads-only",
    "lora": "lora",
    "accgrad_lora": "accgrad",
    "align_lora": "align",
}


def _strip_backbone_prefix(backbone: str, run_id: str) -> str:
    """Remove the backbone prefix from *run_id* if present.

    >>> _strip_backbone_prefix("esmc_300m", "esmc_300m_all_r4")
    'all_r4'
    >>> _strip_backbone_prefix("esmc_300m", "other_all_r4")
    'other_all_r4'
    """
    prefix = f"{backbone}_"
    if run_id.startswith(prefix):
        return run_id[len(prefix) :]
    return run_id


def hf_model_name(
    backbone: str,
    section: str,
    run_id: str,
    variant: str,
) -> str:
    """Build the HuggingFace model name for an ONNX export.

    Parameters
    ----------
    backbone:
        Config-level backbone name (e.g. ``"esmc_300m"``).
    section:
        Training section (e.g. ``"accgrad_lora"``).
    run_id:
        Training run id (e.g. ``"esmc_300m_all_r4"``).
    variant:
        Checkpoint variant (e.g. ``"best_overall"``).

    Returns
    -------
    str
        HuggingFace model name, e.g.
        ``"ESMC-300M-mutafitup-accgrad-all-r4-best-overall"``.

    Raises
    ------
    ValueError
        If the backbone is not in :data:`BACKBONE_HF_PREFIXES`.
    """
    if backbone not in BACKBONE_HF_PREFIXES:
        raise ValueError(
            f"Unsupported backbone {backbone!r} for HF naming. "
            f"Supported: {sorted(BACKBONE_HF_PREFIXES)}"
        )

    hf_prefix = BACKBONE_HF_PREFIXES[backbone]
    section_short = SECTION_SHORT_NAMES.get(section, section.replace("_", "-"))
    run_suffix = _strip_backbone_prefix(backbone, run_id).replace("_", "-")
    variant_short = variant.replace("_", "-")

    return f"{hf_prefix}-mutafitup-{section_short}-{run_suffix}-{variant_short}"
