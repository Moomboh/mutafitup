"""Foldseek binary resolution for GPU-accelerated workflows.

The bioconda ``foldseek`` package does not include CUDA/GPU support.
When GPU mode is requested (``gpu=1``), this module transparently
downloads the official GPU-enabled build from
``https://mmseqs.com/foldseek/foldseek-linux-gpu.tar.gz``, verifies
its SHA-256 digest, and caches the extracted binary under
``.cache/foldseek_gpu/``.

The pinned hash must be updated whenever a new foldseek release is
adopted.
"""

import hashlib
import logging
import os
import stat
import tarfile
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

FOLDSEEK_GPU_URL = "https://mmseqs.com/foldseek/foldseek-linux-gpu.tar.gz"
FOLDSEEK_GPU_SHA256 = "508aadf2dfc78837fbbe86a7ceaad82f7ecfc7793a5772ab09e61e58248d4ea0"

_DEFAULT_CACHE_DIR = ".cache/foldseek_gpu"


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_foldseek_bin(
    gpu: int = 0,
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> str:
    """Return the path to the ``foldseek`` binary to use.

    When *gpu* is falsy the conda-provided ``foldseek`` (on ``$PATH``)
    is returned.  When *gpu* is truthy the GPU-enabled build is
    downloaded (once) to *cache_dir*, its SHA-256 digest is verified,
    and the absolute path to the extracted binary is returned.

    Parameters
    ----------
    gpu:
        ``0`` for the conda (CPU) binary, ``1`` for the GPU build.
    cache_dir:
        Directory under which the GPU tarball and extracted binary are
        cached.  Relative paths are resolved against the current
        working directory.

    Returns
    -------
    str
        Either ``"foldseek"`` (conda PATH) or the absolute path to the
        cached GPU binary.

    Raises
    ------
    RuntimeError
        If the download fails or the SHA-256 digest does not match.
    """
    if not gpu:
        return "foldseek"

    cache = Path(cache_dir).resolve()
    bin_path = cache / "foldseek" / "bin" / "foldseek"

    if bin_path.exists():
        logger.info("Using cached foldseek GPU binary: %s", bin_path)
        return str(bin_path)

    logger.info(
        "Downloading foldseek GPU build from %s …",
        FOLDSEEK_GPU_URL,
    )

    cache.mkdir(parents=True, exist_ok=True)

    # Download to a temporary file next to the cache dir so we can
    # atomically move the extracted result into place.
    with tempfile.TemporaryDirectory(dir=cache) as tmp_dir:
        tarball = Path(tmp_dir) / "foldseek-linux-gpu.tar.gz"

        try:
            urllib.request.urlretrieve(FOLDSEEK_GPU_URL, tarball)
        except Exception as exc:
            raise RuntimeError(f"Failed to download foldseek GPU build: {exc}") from exc

        actual_hash = _sha256_file(tarball)
        if actual_hash != FOLDSEEK_GPU_SHA256:
            raise RuntimeError(
                f"SHA-256 mismatch for foldseek GPU build.\n"
                f"  Expected: {FOLDSEEK_GPU_SHA256}\n"
                f"  Got:      {actual_hash}\n"
                f"The upstream binary may have been updated.  Verify the "
                f"new hash and update FOLDSEEK_GPU_SHA256 in "
                f"src/wfutils/foldseek.py."
            )

        logger.info("SHA-256 verified.  Extracting …")

        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(path=cache)

    # Ensure the binary is executable.
    if bin_path.exists():
        bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    else:
        raise RuntimeError(f"Extraction succeeded but binary not found at {bin_path}")

    logger.info("Foldseek GPU binary ready: %s", bin_path)
    return str(bin_path)
