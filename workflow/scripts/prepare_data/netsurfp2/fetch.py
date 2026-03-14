"""Download NetSurfP-2.0 data files.

Downloads the original NPZ files from DTU (containing full structural
annotations: Q3, Q8, RSA, ASA, phi/psi, disorder) and the NEW364 CSV
from ProtTrans Dropbox (used as the test set for secondary structure).

NPZ source: https://services.healthtech.dtu.dk/services/NetSurfP-2.0/
CSV source: ProtTrans (Elnaggar et al., 2021)
"""

from pathlib import Path

import pooch
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)

urls = dict(snakemake.params["urls"])
hashes = dict(snakemake.params.get("hashes", {}))

raw_dir = Path(str(snakemake.output["raw_dir"]))
raw_dir.mkdir(parents=True, exist_ok=True)

file_map = {
    "train_npz": "Train_HHblits.npz",
    "cb513_npz": "CB513_HHblits.npz",
    "new364": "NEW364.csv",
}

for key, filename in file_map.items():
    url = urls[key]
    known_hash = hashes.get(key)

    logger.info("Downloading %s -> %s", url, raw_dir / filename)
    pooch.retrieve(
        url,
        known_hash=known_hash,
        fname=filename,
        path=str(raw_dir),
    )

logger.info("Downloaded %d files to %s", len(file_map), raw_dir)
