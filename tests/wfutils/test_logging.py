"""Tests for wfutils.logging — _TeeWriter and _patch_stream_handlers."""

import io
import logging
import sys

from wfutils.logging import _TeeWriter, _patch_stream_handlers


class TestTeeWriter:
    """Tests for _TeeWriter."""

    def test_writes_to_both_streams(self):
        original = io.StringIO()
        log_fh = io.StringIO()
        tee = _TeeWriter(original, log_fh)

        tee.write("hello\n")

        assert original.getvalue() == "hello\n"
        assert log_fh.getvalue() == "hello\n"

    def test_filters_bare_carriage_return(self):
        """Progress-bar output containing \\r should be filtered from log."""
        original = io.StringIO()
        log_fh = io.StringIO()
        tee = _TeeWriter(original, log_fh)

        tee.write("loading 50%\r")

        assert "loading 50%" in original.getvalue()
        assert log_fh.getvalue() == ""

    def test_allows_windows_line_endings(self):
        """Messages ending with \\r\\n should still be written to log."""
        original = io.StringIO()
        log_fh = io.StringIO()
        tee = _TeeWriter(original, log_fh)

        tee.write("windows line\r\n")

        assert original.getvalue() == "windows line\r\n"
        assert log_fh.getvalue() == "windows line\r\n"

    def test_handles_closed_log_file(self):
        """Writing to a closed log file should not raise."""
        original = io.StringIO()
        log_fh = io.StringIO()
        log_fh.close()
        tee = _TeeWriter(original, log_fh)

        # Should not raise
        tee.write("after close\n")
        assert original.getvalue() == "after close\n"

    def test_tee_active_sentinel(self):
        original = io.StringIO()
        log_fh = io.StringIO()
        tee = _TeeWriter(original, log_fh)

        assert tee._tee_active is True

    def test_flush_both_streams(self):
        original = io.StringIO()
        log_fh = io.StringIO()
        tee = _TeeWriter(original, log_fh)

        tee.write("data")
        tee.flush()  # should not raise


class TestPatchStreamHandlers:
    """Tests for _patch_stream_handlers."""

    def test_patches_handler_pointing_to_old_stderr(self):
        """A StreamHandler created before the tee should be patched."""
        # Simulate "old stderr" and "new (tee'd) stderr"
        old_stderr = io.StringIO()
        new_stderr = io.StringIO()

        # Create a logger with a handler pointing to old_stderr
        test_logger_name = "_test_patch_stderr"
        lgr = logging.getLogger(test_logger_name)
        lgr.handlers.clear()
        handler = logging.StreamHandler(old_stderr)
        lgr.addHandler(handler)

        try:
            assert handler.stream is old_stderr

            # Install "new stderr" on sys.stderr temporarily
            saved = sys.stderr
            sys.stderr = new_stderr
            try:
                _patch_stream_handlers(
                    old_stdout=io.StringIO(),  # dummy
                    old_stderr=old_stderr,
                )
            finally:
                sys.stderr = saved

            assert handler.stream is new_stderr
        finally:
            lgr.handlers.clear()
            logging.root.manager.loggerDict.pop(test_logger_name, None)

    def test_patches_handler_pointing_to_old_stdout(self):
        """A StreamHandler created before the tee should be patched (stdout)."""
        old_stdout = io.StringIO()
        new_stdout = io.StringIO()

        test_logger_name = "_test_patch_stdout"
        lgr = logging.getLogger(test_logger_name)
        lgr.handlers.clear()
        handler = logging.StreamHandler(old_stdout)
        lgr.addHandler(handler)

        try:
            assert handler.stream is old_stdout

            saved = sys.stdout
            sys.stdout = new_stdout
            try:
                _patch_stream_handlers(
                    old_stdout=old_stdout,
                    old_stderr=io.StringIO(),  # dummy
                )
            finally:
                sys.stdout = saved

            assert handler.stream is new_stdout
        finally:
            lgr.handlers.clear()
            logging.root.manager.loggerDict.pop(test_logger_name, None)

    def test_does_not_patch_unrelated_handler(self):
        """A StreamHandler pointing to a different stream should be left alone."""
        unrelated_stream = io.StringIO()
        old_stderr = io.StringIO()

        test_logger_name = "_test_patch_unrelated"
        lgr = logging.getLogger(test_logger_name)
        lgr.handlers.clear()
        handler = logging.StreamHandler(unrelated_stream)
        lgr.addHandler(handler)

        try:
            saved = sys.stderr
            sys.stderr = io.StringIO()
            try:
                _patch_stream_handlers(
                    old_stdout=io.StringIO(),
                    old_stderr=old_stderr,
                )
            finally:
                sys.stderr = saved

            # Handler should still point to the unrelated stream
            assert handler.stream is unrelated_stream
        finally:
            lgr.handlers.clear()
            logging.root.manager.loggerDict.pop(test_logger_name, None)

    def test_patches_root_logger_handler(self):
        """Handlers on the root logger should also be patched."""
        old_stderr = io.StringIO()
        new_stderr = io.StringIO()

        handler = logging.StreamHandler(old_stderr)
        logging.root.addHandler(handler)

        try:
            saved = sys.stderr
            sys.stderr = new_stderr
            try:
                _patch_stream_handlers(
                    old_stdout=io.StringIO(),
                    old_stderr=old_stderr,
                )
            finally:
                sys.stderr = saved

            assert handler.stream is new_stderr
        finally:
            logging.root.removeHandler(handler)
