"""Download GPSite binding-site dataset files from GitHub.

For each ligand type listed in the config, downloads ``{LIGAND}_train.txt``
and ``{LIGAND}_test.txt`` from the pinned commit of the GPSite repository.
"""

from pathlib import Path

import pooch
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)

commit = str(snakemake.params["commit"])
base_url = str(snakemake.params["base_url"])
subsets = list(snakemake.params["subsets"])
hashes = dict(snakemake.params.get("hashes", {}))

raw_dir = Path(str(snakemake.output["raw_dir"]))
raw_dir.mkdir(parents=True, exist_ok=True)

ligands = sorted({s["ligand"] for s in subsets})

for ligand in ligands:
    for split in ("train", "test"):
        filename = f"{ligand}_{split}.txt"
        url = f"{base_url}/{commit}/datasets/{filename}"
        known_hash = hashes.get(filename)

        logger.info("Downloading %s -> %s", url, raw_dir / filename)
        pooch.retrieve(
            url,
            known_hash=known_hash,
            fname=filename,
            path=str(raw_dir),
        )

logger.info("Downloaded %d files to %s", len(ligands) * 2, raw_dir)
