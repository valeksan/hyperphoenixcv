"""Operator commands for local SQLite study files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sqlite_store import SQLiteStudyStore, StudyStoreError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hyperphoenixcv-storage")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="run SQLite and application integrity checks")
    check.add_argument("path")
    backup = commands.add_parser("backup", help="create consistent SQLite backup")
    backup.add_argument("path")
    backup.add_argument("destination")
    restore = commands.add_parser("restore", help="restore backup to stopped storage path")
    restore.add_argument("backup")
    restore.add_argument("destination")
    cleanup = commands.add_parser("prune-empty", help="remove abandoned studies with no trials")
    cleanup.add_argument("path")
    args = parser.parse_args(argv)
    try:
        if args.command == "restore":
            print(SQLiteStudyStore.restore_from(args.backup, args.destination))
            return 0
        with SQLiteStudyStore(args.path) as store:
            if args.command == "check":
                report = store.integrity_check()
                print(json.dumps(report, sort_keys=True))
                return 0 if report["ok"] else 1
            if args.command == "backup":
                print(store.backup_to(args.destination))
                return 0
            print(store.prune_empty_studies())
            return 0
    except (OSError, StudyStoreError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
