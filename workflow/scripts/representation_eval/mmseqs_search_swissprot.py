"""Run MMseqs2 easy-search from benchmark test proteins to Swiss-Prot."""

import tempfile
from pathlib import Path
from subprocess import CalledProcessError, run

from snakemake.script import snakemake

from wfutils import get_logger
from wfutils.logging import log_snakemake_info

logger = get_logger()
log_snakemake_info(logger)


def main() -> None:
    query_fasta = Path(str(snakemake.input["query"]))
    target_fasta = Path(str(snakemake.input["target"]))
    result_tsv = Path(str(snakemake.output["hits"]))
    min_seq_id = float(snakemake.params["min_seq_id"])
    gpu = int(snakemake.params["gpu"])

    result_tsv.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_result_tsv = Path(tmp_dir) / "hits.tsv"
        cmd = [
            "mmseqs",
            "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(raw_result_tsv),
            tmp_dir,
            "--min-seq-id",
            str(min_seq_id),
            "--format-output",
            "query,target,fident,alnlen,qcov,tcov,evalue,bits",
            "--gpu",
            str(gpu),
        ]
        logger.info("Running MMseqs2: %s", " ".join(cmd))
        try:
            run(cmd, check=True)
        except CalledProcessError as exc:
            raise RuntimeError(
                f"MMseqs2 easy-search failed with exit code {exc.returncode}"
            ) from exc

        with raw_result_tsv.open() as src, result_tsv.open("w") as dst:
            dst.write("query\ttarget\tfident\talnlen\tqcov\ttcov\tevalue\tbits\n")
            for line in src:
                dst.write(line)

    logger.info("Swiss-Prot MMseqs hits written to %s", result_tsv)


if __name__ == "__main__":
    main()
