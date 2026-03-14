"""
Logging utilities for Snakemake workflows.

This module provides convenient logging functionality that automatically
integrates with Snakemake's logging system.

Key features:
- Automatic log file detection from Snakemake's ``log:`` directive.
- Tees both ``sys.stdout`` and ``sys.stderr`` into the log file so that
  uncaught tracebacks, CUDA OOM messages, and any other raw output is
  captured alongside structured log messages.
- Log format includes ``filename:lineno`` instead of the logger name
  (which is always ``__main__`` under Snakemake's script wrapper).
"""

import logging
import sys
from pathlib import Path


class _TeeWriter:
    """Write to both the original stream and a log file handle.

    Forwards all attribute access (``fileno``, ``isatty``, ``encoding``, …)
    to the original stream so that third-party code that inspects the
    stream (e.g. ``tqdm``, ``click``) keeps working.
    """

    _tee_active = True  # sentinel used to detect double-wrapping

    def __init__(self, original, log_fh):
        self.original = original
        self.log_fh = log_fh

    def write(self, msg):
        self.original.write(msg)
        try:
            # Skip progress-bar output from the log file.  Progress bars
            # (tqdm, HuggingFace safetensors, etc.) use \r to overwrite the
            # current terminal line.  Normal log lines never contain a bare
            # \r, so this is a reliable heuristic.  We still allow \r\n
            # (Windows-style line endings) through.
            if "\r" not in msg or msg.endswith("\r\n"):
                self.log_fh.write(msg)
                self.log_fh.flush()
        except (OSError, ValueError):
            # Log file may have been closed; don't crash the program.
            pass

    def flush(self):
        self.original.flush()
        try:
            self.log_fh.flush()
        except (OSError, ValueError):
            pass

    def __getattr__(self, name):
        # Forward everything else (fileno, isatty, encoding, …)
        return getattr(self.original, name)


def _patch_stream_handlers(old_stdout, old_stderr):
    """Re-point StreamHandlers that captured pre-tee stdout/stderr.

    Libraries like HuggingFace *transformers* create a
    ``logging.StreamHandler()`` at import time, which captures a
    reference to ``sys.stderr`` before our ``_TeeWriter`` is installed.
    This function patches those handlers so their output also flows
    through the tee into the log file.
    """
    loggers = [logging.root] + [
        logging.getLogger(name) for name in logging.root.manager.loggerDict
    ]
    for lgr in loggers:
        for handler in getattr(lgr, "handlers", []):
            if not isinstance(handler, logging.StreamHandler):
                continue
            if handler.stream is old_stderr:
                handler.stream = sys.stderr
            elif handler.stream is old_stdout:
                handler.stream = sys.stdout


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger that automatically writes to Snakemake's log output.

    On the first call (when a Snakemake log file is available) this
    function also tees ``sys.stdout`` and ``sys.stderr`` into the log
    file so that *any* output — including uncaught tracebacks and native
    library errors — is captured.

    The log format uses ``%(filename)s:%(lineno)d`` rather than the
    logger name, which avoids the unhelpful ``__main__`` that appears
    when Snakemake wraps scripts in temporary files.

    Args:
        name: Optional name for the logger. If None, uses the calling
              module's ``__name__``.

    Returns:
        A configured logger instance.
    """
    try:
        from snakemake.script import snakemake

        log_file = None
        if hasattr(snakemake, "log") and snakemake.log:
            # snakemake.log can be a Namedlist, list, or a plain string
            if isinstance(snakemake.log, (list, tuple)):
                log_file = snakemake.log[0] if snakemake.log else None
            elif isinstance(snakemake.log, str):
                log_file = snakemake.log
            else:
                # Namedlist or similar — treat as iterable
                try:
                    log_file = list(snakemake.log)[0]
                except (TypeError, IndexError):
                    log_file = str(snakemake.log)
    except ImportError:
        log_file = None

    if name is None:
        frame = sys._getframe(1)
        name = frame.f_globals.get("__name__", "wfutils")

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    # ── format: timestamp - file:line - LEVEL - message ──
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # ── tee stdout + stderr into the log file ──
        # Only once: guard against double-wrapping when get_logger() is
        # called more than once in the same process.
        if not getattr(sys.stderr, "_tee_active", False):
            log_fh = open(log_file, "a")
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = _TeeWriter(old_stdout, log_fh)
            sys.stderr = _TeeWriter(old_stderr, log_fh)

            # Patch StreamHandlers that were created before the tee was
            # installed (e.g. by HuggingFace transformers during import).
            # Those handlers captured the *original* stderr/stdout and
            # would bypass the tee entirely.
            _patch_stream_handlers(old_stdout, old_stderr)

        # Single handler writing to stdout (the tee already mirrors to file).
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)

    return logger


def log_snakemake_info(logger: logging.Logger | None = None) -> None:
    """
    Log useful information about the current Snakemake context.

    Args:
        logger: Logger to use. If None, creates a new one.
    """
    if logger is None:
        logger = get_logger()

    try:
        from snakemake.script import snakemake

        logger.info("=== Snakemake Context Info ===")

        if hasattr(snakemake, "rule"):
            logger.info(f"Rule: {snakemake.rule}")

        if hasattr(snakemake, "wildcards"):
            wildcards_dict = dict(snakemake.wildcards)
            if wildcards_dict:
                logger.info(f"Wildcards: {wildcards_dict}")

        if hasattr(snakemake, "input"):
            input_files = list(snakemake.input)
            if input_files:
                logger.info(f"Input files: {input_files}")

        if hasattr(snakemake, "output"):
            output_files = list(snakemake.output)
            if output_files:
                logger.info(f"Output files: {output_files}")

        if hasattr(snakemake, "params"):
            params_dict = dict(snakemake.params)
            if params_dict:
                logger.info(f"Parameters: {params_dict}")

        logger.info("=== End Context Info ===")

    except ImportError:
        logger.warning("Not running in Snakemake context - no context info available")
