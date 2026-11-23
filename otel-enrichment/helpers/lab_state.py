"""Tiny helper kept for compatibility with the workshop notebooks.

The Instruqt version wrote sentinel files that the platform's Check read. Here it
is a best-effort no-op: failure to write is ignored."""

import tempfile
from pathlib import Path


def mark_done(path):
    try:
        (Path(tempfile.gettempdir()) / Path(path).name).write_text("ok\n")
    except OSError:
        pass
