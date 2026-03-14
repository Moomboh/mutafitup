import os
import shutil
import zipfile
from pathlib import Path

import pooch
from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)

url = snakemake.params["url"]
known_hash = snakemake.params["hash"]


zip_path = Path(snakemake.output["tmp_zip"])
extract_to = Path(snakemake.output["tmp_extract_to"])

pooch.retrieve(
    url,
    known_hash,
    fname=zip_path.name,
    path=zip_path.parent,
)

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(extract_to)

src = extract_to / "training data/"
dst = snakemake.output["train_data"]

for name in os.listdir(src):
    src_path = os.path.join(src, name)
    shutil.move(src_path, os.path.join(dst, name))
