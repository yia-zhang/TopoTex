#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Git tree safety preflight — run BEFORE any merge/checkout into a
workspace that holds data.

    python scripts/check_git_tree_safety.py <commit-ish> [--repo DIR]

Checks the TARGET tree (what a checkout would write), not the working
tree:

1. every tracked symlink (mode 120000) with its resolved target
2. tracked entries at reserved persistent-data names
3. collisions: target paths that already exist in the workspace as
   real files/dirs/links — with resolved paths and mount points

Exit 1 on ANY tracked symlink or reserved-name entry or collision.
Rationale: on 2026-07-30 a fast-forward checkout of a branch that
tracked symlinks at gitignored data paths silently deleted the real
directories (git treats ignored paths as expendable). This preflight
makes that class of merge impossible to run by accident.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topotex.paths import RESERVED_DATA_NAMES  # noqa: E402


def sh(args, cwd):
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


def mount_of(path: Path) -> str:
    try:
        p = path.resolve()
    except (OSError, RuntimeError):
        p = path.parent.resolve()
    while not p.exists():
        p = p.parent
    dev = os.stat(p).st_dev
    best = "/"
    try:
        for line in open("/proc/self/mounts"):
            mnt = line.split()[1]
            try:
                if os.stat(mnt).st_dev == dev and len(mnt) > len(best):
                    best = mnt
            except OSError:
                continue
    except OSError:
        pass
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("commit", help="target commit-ish to be checked out")
    ap.add_argument("--repo", default=".", help="workspace to protect")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()

    rows = sh(["git", "ls-tree", "-r", args.commit], repo).splitlines()
    problems = []

    symlinks = [r for r in rows if r.startswith("120000")]
    for r in symlinks:
        mode_type_sha, path = r.split("\t", 1)
        sha = mode_type_sha.split()[2]
        target = sh(["git", "cat-file", "-p", sha], repo).strip()
        problems.append(
            f"TRACKED SYMLINK  {path} -> {target}"
            f"  (would be written into {repo / path})"
        )

    top = {r.split("\t", 1)[1].split("/", 1)[0] for r in rows if "\t" in r}
    for name in sorted(top & set(RESERVED_DATA_NAMES)):
        problems.append(f"RESERVED DATA NAME tracked at repo root: {name}")

    # collision report: reserved/symlink paths that exist for real
    check_paths = {r.split("\t", 1)[1] for r in symlinks} | (
        top & set(RESERVED_DATA_NAMES)
    )
    for rel in sorted(check_paths):
        p = repo / rel
        try:
            present = p.exists() or p.is_symlink()
        except OSError:
            present = True
        if present:
            try:
                kind = (
                    "symlink"
                    if p.is_symlink()
                    else "dir"
                    if p.is_dir()
                    else "file"
                )
                resolved = str(p.resolve()) if p.exists() else "(dangling)"
            except (OSError, RuntimeError) as e:
                kind, resolved = "symlink", f"(unresolvable: {e})"
            try:
                mnt = mount_of(p)
            except (OSError, RuntimeError):
                mnt = "?"
            print(
                f"  collision detail: {rel} exists as {kind}; "
                f"resolves to {resolved}; mount {mnt}"
            )
            problems.append(f"COLLISION: {rel} already exists in {repo}")

    if problems:
        print("UNSAFE TARGET TREE:", args.commit)
        for p in problems:
            print("  " + p)
        print(
            "\nRefusing: fix the branch (git rm --cached <paths>) before "
            "merging/checking out into a data-bearing workspace."
        )
        return 1
    print(
        f"OK: {args.commit} tracks no symlinks and no reserved data "
        f"names ({len(rows)} entries checked)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
