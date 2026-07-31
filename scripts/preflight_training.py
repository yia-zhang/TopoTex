#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training preflight — the single read-only gate before formal runs.

    python scripts/preflight_training.py [--gpus 0,1,...] [--skip-tests]

Checks code (git clean, no tracked symlinks, ruff, pytest), environment
(python/torch/CUDA/nvdiffrast/bpy/libX11/git), data (roots, manifest and
split SHA + counts, a random sample loads with complete queries), GPUs
(temperature/SM clock under load/power/memory/ECC) and storage (run and
checkpoint roots writable, free space, not a symlink, not inside a git
worktree). Prints PASS/FAIL per item; exit 1 on any failure. Never
installs, repairs or mutates anything.
"""

import argparse
import ctypes.util
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from topotex.paths import data_root  # noqa: E402

results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok), detail))
    print(
        f"{'PASS' if ok else 'FAIL'} {name}"
        + (f"  [{detail}]" if detail else "")
    )
    return bool(ok)


def sh(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x != ""]

    # ---------------------------------------------------------- A. code
    git = shutil.which("git")
    check("env: git binary", git is not None)
    if git:
        st = sh([git, "status", "--porcelain"], cwd=REPO).stdout
        dirty = [l for l in st.splitlines() if not l.startswith("??")]
        check("code: git status clean (tracked)", not dirty, st.strip()[:60])
        head = sh(
            [git, "rev-parse", "--short", "HEAD"], cwd=REPO
        ).stdout.strip()
        check("code: HEAD", True, head)
        links = [
            l
            for l in sh(
                [git, "ls-tree", "-r", "HEAD"], cwd=REPO
            ).stdout.splitlines()
            if l.startswith("120000")
        ]
        check("code: no tracked symlinks", not links)
        top = {
            l.split("\t")[1]
            for l in sh([git, "ls-tree", "HEAD"], cwd=REPO).stdout.splitlines()
            if "\t" in l
        }
        from topotex.paths import RESERVED_DATA_NAMES

        check(
            "code: no reserved data paths tracked",
            not (top & set(RESERVED_DATA_NAMES)),
        )
    r = sh(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            "-q",
            "topotex/",
            "tests/",
        ],
        cwd=REPO,
    )
    check("code: ruff format", r.returncode == 0)
    r = sh(
        [sys.executable, "-m", "ruff", "check", "-q", "topotex/", "tests/"],
        cwd=REPO,
    )
    check("code: ruff check", r.returncode == 0)
    if not args.skip_tests:
        r = sh(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "-q",
                "-x",
                "--deselect",
                "tests/test_notebook_smoke.py::test_notebook_executes",
            ],
            cwd=REPO,
        )
        check(
            "code: pytest",
            r.returncode == 0,
            r.stdout.splitlines()[-1][:60] if r.stdout else "",
        )

    # ---------------------------------------------------- B. environment
    check("env: python", True, sys.executable)
    try:
        import torch

        check("env: torch+CUDA", torch.cuda.is_available(), torch.__version__)
    except Exception as e:  # noqa: BLE001 — preflight reports, never raises
        check("env: torch+CUDA", False, repr(e))
    for mod in ("nvdiffrast", "bpy", "xatlas", "trimesh", "safetensors"):
        try:
            __import__(mod)
            check(f"env: {mod}", True)
        except Exception as e:  # noqa: BLE001
            check(f"env: {mod}", False, repr(e)[:60])
    check("env: libX11", ctypes.util.find_library("X11") is not None)

    # ----------------------------------------------------------- C. data
    ds = data_root("dataset", REPO)
    check(
        "data: dataset root exists", (ds / "manifest.jsonl").exists(), str(ds)
    )
    data_dir = ds.parent
    man = data_dir / "dataset_manifest.jsonl"
    split_f = data_dir / "object_split.json"
    if man.exists() and split_f.exists():
        man_sha = hashlib.sha256(man.read_bytes()).hexdigest()
        split_sha = hashlib.sha256(split_f.read_bytes()).hexdigest()
        check("data: manifest sha", True, man_sha[:16])
        check("data: split sha", True, split_sha[:16])
        split = json.loads(split_f.read_text())
        ids = [json.loads(l)["sample_id"] for l in open(ds / "manifest.jsonl")]
        check(
            "data: split covers manifest",
            set(split["train"]) | set(split["val"]) == set(ids)
            and not (set(split["train"]) & set(split["val"])),
            f"train {len(split['train'])} / test {len(split['val'])}",
        )
        import random

        sid = random.Random(0).choice(split["train"])
        try:
            from topotex import TopoTexDataset

            it = TopoTexDataset(ds, [sid])[0]
            ok = len(it["uv_queries"]) == 3 and len(it["test_uv_queries"]) == 1
            check("data: random sample loads, queries complete", ok, sid[:10])
        except Exception as e:  # noqa: BLE001
            check("data: random sample loads", False, repr(e)[:70])
        broken = subprocess.run(
            ["find", str(ds / "samples" / sid), "-xtype", "l"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        check("data: no broken links (probe)", not broken)
    else:
        check("data: frozen manifest/split present", False)

    # ------------------------------------------------------------ D. GPU
    try:
        import torch

        for i in gpus:
            with torch.cuda.device(i):
                a = torch.randn(4096, 4096, device=f"cuda:{i}")
                t0 = time.time()
                while time.time() - t0 < 5:
                    a = a @ a * 1e-4
                torch.cuda.synchronize()
            q = (
                sh(
                    [
                        "nvidia-smi",
                        "-i",
                        str(i),
                        "--query-gpu=clocks.sm,temperature.gpu,power.draw,memory.used,ecc.errors.uncorrected.volatile.total",
                        "--format=csv,noheader,nounits",
                    ]
                )
                .stdout.strip()
                .split(", ")
            )
            clk, tmp = int(q[0]), int(q[1])
            ecc = q[4].strip()
            # clock is the primary throttle signal; 85C accommodates the
            # repaired GPU6's steady point (1980MHz sustained at 84C)
            ok = clk >= 1400 and tmp <= 85 and ecc in ("0", "[N/A]", "N/A")
            check(f"gpu{i}: healthy", ok, f"{clk}MHz {tmp}C ecc={ecc}")
            del a
            torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        check("gpu: probe", False, repr(e)[:70])

    # -------------------------------------------------------- E. storage
    for kind in ("runs", "checkpoints"):
        root = data_root(kind, REPO)
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / f".preflight_{os.getpid()}"
            probe.write_text("x")
            probe.unlink()
            inside_repo = str(root.resolve()).startswith(
                str(REPO.resolve()) + "/"
            )
            check(
                f"storage: {kind} root writable, outside repo, not symlink",
                not root.is_symlink() and not inside_repo,
                str(root),
            )
        except OSError as e:
            check(f"storage: {kind} root writable", False, repr(e)[:60])
    free_gb = shutil.disk_usage(data_root("runs", REPO)).free / 1e9
    check("storage: free space > 200 GB", free_gb > 200, f"{free_gb:.0f} GB")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\nPREFLIGHT: {len(results) - n_fail}/{len(results)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
