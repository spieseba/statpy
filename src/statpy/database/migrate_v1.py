"""CLI: migrate a single retired-v1 JSON statpy database to the current DB format.

Usage:
    python -m statpy.database.migrate_v1 IN.sample[.blinded] OUT.db

Faithful single-file conversion (see :func:`statpy.database.io.load_v1_json`).
"""
import sys

from statpy.database.io import load_v1_json


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 1
    src, dst = argv
    load_v1_json(src).save(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
