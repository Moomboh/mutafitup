import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

EXCLUDED_PATHS = (Path(".snakemake") / "conda",)


def _is_excluded(path: Path, excluded: tuple[Path, ...]) -> bool:
    try:
        resolved = path.resolve()
    except FileNotFoundError:
        resolved = path

    for excluded_path in excluded:
        try:
            excluded_resolved = excluded_path.resolve()
        except FileNotFoundError:
            excluded_resolved = excluded_path

        if resolved == excluded_resolved or excluded_resolved in resolved.parents:
            return True

    return False


def _get_dir_size(path: Path, excluded: tuple[Path, ...]) -> int:
    total = 0
    for item in path.rglob("*"):
        if _is_excluded(item, excluded):
            continue
        if item.is_file():
            total += item.stat().st_size
    return total


def _get_item_size(path: Path, excluded: tuple[Path, ...]) -> int:
    if _is_excluded(path, excluded):
        return 0
    if path.is_file():
        return path.stat().st_size
    return _get_dir_size(path, excluded)


def _copy_with_progress(
    src: Path,
    dst: Path,
    progress: Progress,
    task: TaskID,
    excluded: tuple[Path, ...],
) -> None:
    if _is_excluded(src, excluded):
        return
    if src.is_file():
        shutil.copy2(src, dst)
        progress.advance(task, src.stat().st_size)
    else:
        dst.mkdir(exist_ok=True)
        for item in src.iterdir():
            _copy_with_progress(item, dst / item.name, progress, task, excluded)


def _get_commit_short() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        value = result.stdout.strip()
        if value:
            return value
    except Exception:
        pass
    return "nohash"


def _build_snapshot_name() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    commit = _get_commit_short()
    return f"{timestamp}_{commit}"


def _copy_results(source: Path, target_root: Path) -> Path:
    if not source.exists() or not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")

    target_root.mkdir(parents=True, exist_ok=True)

    name = _build_snapshot_name()
    dest = target_root / name
    dest.mkdir()

    items = [source]
    cache_dir = Path(".cache")
    if cache_dir.exists() and cache_dir.is_dir():
        items.append(cache_dir)

    snakemake_dir = Path(".snakemake")
    if snakemake_dir.exists() and snakemake_dir.is_dir():
        items.append(snakemake_dir)

    log_dir = Path("log")
    if log_dir.exists() and log_dir.is_dir():
        items.append(log_dir)

    excluded = EXCLUDED_PATHS
    total_size = sum(_get_item_size(item, excluded) for item in items)

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"Copying to {dest.name}", total=total_size)

        for item in items:
            target = dest / item.name
            _copy_with_progress(item, target, progress, task, excluded)

    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir")
    parser.add_argument("--source", default="results")
    args = parser.parse_args(argv)

    source = Path(args.source)
    target_root = Path(args.target_dir)

    dest = _copy_results(source, target_root)
    print(dest)


if __name__ == "__main__":
    main(sys.argv[1:])
