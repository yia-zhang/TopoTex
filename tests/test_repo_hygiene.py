# -*- coding: utf-8 -*-
"""Repository hygiene: the consolidated tree must not reintroduce
exploratory naming. Banned substrings are assembled at runtime so this
file does not trip its own scan."""
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

# assembled to keep this file clean under its own rule
BANNED = ["arch" + "ive", "o" + "ld", "back" + "up", "de" + "bug",
          "pi" + "lot", "v" + "1", "v" + "2", "leg" + "acy"]
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".sh", ".ipynb", ".jsonl",
                 ".json", ".txt", ".cfg", ".toml"}
# "held" contains one banned pair ("ld") only as part of a word — the scan
# is substring-based on WORD BOUNDARIES for the two shortest tokens to avoid
# false hits inside ordinary English words.
WORDY = {"o" + "ld", "v" + "1", "v" + "2"}


def _tracked_files():
    out = subprocess.check_output(["git", "-C", str(PROJECT), "ls-files"],
                                  text=True)
    return [PROJECT / p for p in out.splitlines()
            if Path(p).suffix in TEXT_SUFFIXES and (PROJECT / p).exists()]


def _scan_text(f):
    """Notebook JSON embeds base64 images whose bytes randomly contain any
    short token; scan only cell sources and textual outputs there."""
    if f.suffix != ".ipynb":
        return f.read_text(errors="ignore").lower()
    import json
    nb = json.loads(f.read_text(errors="ignore"))
    parts = []
    for c in nb.get("cells", []):
        parts.append("".join(c.get("source", [])))
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                parts.append("".join(o.get("text", [])))
            for k, v in o.get("data", {}).items():
                if k.startswith("text/"):
                    parts.append("".join(v))
    return "\n".join(parts).lower()


def test_no_banned_names():
    import re
    offenders = []
    for f in _tracked_files():
        if f.name == Path(__file__).name:
            continue
        text = _scan_text(f)
        for b in BANNED:
            if b in WORDY:
                hit = re.search(r"(?<![a-z0-9])" + re.escape(b)
                                + r"(?![a-z0-9])", text)
                if hit:
                    offenders.append(f"{f.relative_to(PROJECT)}: {b}")
            elif b in text:
                offenders.append(f"{f.relative_to(PROJECT)}: {b}")
    assert not offenders, "banned names in tracked files:\n" + \
        "\n".join(sorted(set(offenders))[:40])


def test_no_dangling_imports():
    """Every tracked python module imports cleanly, and no tracked file
    references a module retired by the consolidation."""
    import importlib
    import re
    retired = ["models.diffusion", "models.texture_" + "dit",
               "datasets.multi_" + "uv_dataset", "datasets.topotex_dataset",
               "datasets.query_sampler", "datasets.geometry",
               "datasets.unitex_mv", "train_multi_" + "uv",
               "surface_query_decoder", "mini_" + "dit",
               "face_mv_" + "attention", "build_source",
               "maskeddiff" + "usion", "texture_generator.diff" + "usion"]
    offenders = []
    for f in _tracked_files():
        if f.suffix not in {".py", ".ipynb"} or f.name == Path(__file__).name:
            continue
        text = _scan_text(f)
        for r in retired:
            if re.search(re.escape(r.lower()) + r"(?![a-z0-9_])", text):
                offenders.append(f"{f.relative_to(PROJECT)}: {r}")
    assert not offenders, "dangling references:\n" + "\n".join(offenders)
    for mod in ("datasets.dataset", "datasets.uv_query", "datasets.mesh_utils",
                "datasets.mv_generator", "datasets.build_dataset",
                "datasets.build_uv_queries", "datasets.merge_manifest",
                "datasets.rasterizer", "models.surface_conditioner",
                "models.texture_generator", "train", "sample", "evaluate"):
        importlib.import_module(mod)
