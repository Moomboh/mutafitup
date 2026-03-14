"""Centralised mapping from internal task identifiers to human-readable
display names used in plots, tables, and reports.
"""

TASK_DISPLAY_NAMES = {
    "secstr": "SecStr",
    "secstr8": "SecStr8",
    "rsa": "RSA",
    "disorder": "Disorder",
    "meltome": "Meltome",
    "subloc": "SubLoc",
    "gpsite_dna": "binding DNA",
    "gpsite_rna": "binding RNA",
    "gpsite_pep": "binding Pep",
    "gpsite_pro": "binding Pro",
    "gpsite_atp": "binding ATP",
    "gpsite_hem": "binding Hem",
    "gpsite_zn": "binding Zn",
    "gpsite_ca": "binding Ca",
    "gpsite_mg": "binding Mg",
    "gpsite_mn": "binding Mn",
}


def task_display_name(task_id: str) -> str:
    """Return the display name for *task_id*, falling back to the raw ID."""
    return TASK_DISPLAY_NAMES.get(task_id, task_id)
