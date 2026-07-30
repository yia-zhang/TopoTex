# -*- coding: utf-8 -*-
"""Small IO helpers shared by builders and pipelines."""

import hashlib
import json
from pathlib import Path


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def write_json_atomic(path, obj, indent=1):
    """Write JSON via tmp + rename — no truncated files on interruption."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(obj, indent=indent))
    tmp.replace(path)
