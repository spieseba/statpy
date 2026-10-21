import subprocess
from pathlib import Path

import jax

jax.config.update('jax_enable_x64', True)  # enable 64-bit mode

import statpy.database
import statpy.log
import statpy.qcd
import statpy.statistics
import statpy.variable


def _commit_hash() -> str | None:
    """Best-effort git commit hash of the statpy checkout this module lives in.

    ``-dirty`` is appended if tracked files differ from the commit. Returns ``None`` for non-git
    installs (e.g. release wheels).
    """
    cwd = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd, stderr=subprocess.DEVNULL)
        dirty = subprocess.call(['git', 'diff', '--quiet', 'HEAD'], cwd=cwd, stderr=subprocess.DEVNULL) != 0
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit.decode().strip() + ("-dirty" if dirty else "")


__commit__ = _commit_hash()
