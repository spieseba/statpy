import subprocess
from pathlib import Path

import statpy.database
import statpy.fitting
import statpy.log
import statpy.qcd
import statpy.statistics


def _commit_hash():
    """Best-effort git commit hash of the statpy checkout this module lives in.

    Returns ``None`` for non-git installs (e.g. release wheels).
    """
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


__commit__ = _commit_hash()
