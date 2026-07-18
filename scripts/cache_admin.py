"""Inspect, verify, export, and safely prune the level cache.

Pruning is dry-run by default and refuses to delete anything outside
cache/levels.  Run ``python scripts/cache_admin.py --help`` for commands.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import level_cache
from paths import ROOT


def captures():
    if not level_cache.CACHE_DIR.exists():
        return []
    result = []
    for path in level_cache.CACHE_DIR.glob("*/*/meta.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            meta = {"capture_id": path.parent.name, "corrupt": True}
        result.append((path.parent, meta))
    return sorted(result, key=lambda item: item[1].get("created_at", ""), reverse=True)


def command_list(args):
    rows = captures()
    if args.level:
        rows = [item for item in rows if item[1].get("level_key") == args.level]
    for path, meta in rows[:args.limit]:
        extra = meta.get("extra") or {}
        print(f"{meta.get('capture_id', path.name)}  left={meta.get('remaining', '?')}  "
              f"scene={extra.get('scene_state', '?')}  executable={extra.get('executable', '?')}  "
              f"at={meta.get('created_at', '?')}")
    print(f"shown={min(len(rows), args.limit)} total={len(rows)}")
    return 0


def command_verify(_args):
    failures = []
    for path, meta in captures():
        if meta.get("corrupt"):
            failures.append((path, "meta.json corrupt"))
            continue
        board_path = path / "board.json"
        if board_path.exists():
            try:
                board = json.loads(board_path.read_text(encoding="utf-8"))
                expected = (meta.get("state_hash") or meta.get("board_hash")) if int(meta.get("schema") or 0) >= 2 else None
                if expected and level_cache.board_hash(board) != expected:
                    failures.append((path, "state_hash mismatch"))
            except Exception as exc:
                failures.append((path, f"board.json invalid: {exc}"))
        for artifact in meta.get("artifacts", []):
            if not (path / artifact).exists():
                failures.append((path, f"missing {artifact}"))
    for path, reason in failures:
        print(f"FAIL {path.relative_to(ROOT)}: {reason}")
    print(f"captures={len(captures())} failures={len(failures)}")
    return 1 if failures else 0


def command_export(args):
    match = next(((path, meta) for path, meta in captures()
                  if meta.get("capture_id") == args.capture_id), None)
    if not match:
        raise SystemExit(f"capture not found: {args.capture_id}")
    path, _meta = match
    output = Path(args.output or f"{args.capture_id}.zip").resolve()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in path.rglob("*"):
            if file.is_file():
                archive.write(file, file.relative_to(path))
    print(output)
    return 0


def command_duplicates(_args):
    groups = defaultdict(list)
    for path, meta in captures():
        observation = meta.get("observation_hash")
        if observation:
            groups[observation].append((path, meta))
    duplicate_groups = [items for items in groups.values() if len(items) > 1]
    for items in duplicate_groups:
        print(f"observation={items[0][1]['observation_hash']} copies={len(items)}")
        for path, meta in items:
            print(f"  {meta.get('capture_id', path.name)}")
    print(f"duplicate_groups={len(duplicate_groups)}")
    return 0


def command_prune(args):
    root = level_cache.CACHE_DIR.resolve()
    by_level = defaultdict(list)
    for path, meta in captures():
        by_level[meta.get("level_key") or path.parent.name].append((path, meta))
    victims = []
    for items in by_level.values():
        items.sort(key=lambda item: item[1].get("created_at", ""), reverse=True)
        victims.extend(items[max(0, args.keep):])
    for path, _meta in victims:
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or resolved == root:
            raise RuntimeError(f"refusing unsafe cache path: {resolved}")
        print(("DELETE " if args.apply else "DRY-RUN ") + str(path.relative_to(ROOT)))
        if args.apply:
            shutil.rmtree(resolved)
    print(f"candidates={len(victims)} applied={bool(args.apply)}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--level")
    listing.add_argument("--limit", type=int, default=30)
    listing.set_defaults(func=command_list)
    verify = sub.add_parser("verify")
    verify.set_defaults(func=command_verify)
    export = sub.add_parser("export")
    export.add_argument("capture_id")
    export.add_argument("--output")
    export.set_defaults(func=command_export)
    duplicates = sub.add_parser("duplicates")
    duplicates.set_defaults(func=command_duplicates)
    prune = sub.add_parser("prune")
    prune.add_argument("--keep", type=int, default=50)
    prune.add_argument("--apply", action="store_true")
    prune.set_defaults(func=command_prune)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
