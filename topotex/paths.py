# -*- coding: utf-8 -*-
"""Data-path resolution: persistent data lives OUTSIDE the repository.

The 2026-07-30 incident (see docs/safety.md and the forensic
report in the rescue set) was caused by data directories living at
git-visible paths. Storage roots are therefore resolved here, never via
symlinks inside the repository:

    TOPOTEX_DATASET_ROOT      finalized UV-query dataset
    TOPOTEX_SOURCE_ROOT       source samples (mesh / MV / reference)
    TOPOTEX_CHECKPOINT_ROOT   official checkpoints
    TOPOTEX_RUN_ROOT          training run directories

When an environment variable is unset, the historical project-relative
location is used (so a self-contained checkout still works). Model
modules never import this file — pipelines, data builders, CLIs and
tests only.
"""

import os
from pathlib import Path

ENV_VARS = {
    "dataset": "TOPOTEX_DATASET_ROOT",
    "source": "TOPOTEX_SOURCE_ROOT",
    "checkpoints": "TOPOTEX_CHECKPOINT_ROOT",
    "runs": "TOPOTEX_RUN_ROOT",
}
DEFAULTS = {
    "dataset": "output/topotex_dataset",
    "source": "output/topotex_source",
    "checkpoints": "checkpoints",
    "runs": "runs",
}
#: repo path names that must never hold (or link to) persistent data
RESERVED_DATA_NAMES = (
    "dataset",
    "output",
    "checkpoints",
    "workspace",
    "cache",
    "runs",
)


def data_root(kind: str, project_root, relative=None) -> Path:
    """Storage root for `kind`: env override wins, else project-relative
    default (`relative` replaces the built-in default when given)."""
    env = os.environ.get(ENV_VARS[kind], "").strip()
    if env:
        return Path(env)
    return Path(project_root) / (relative or DEFAULTS[kind])


def resolve_cli_root(kind: str, project_root, arg, default) -> Path:
    """Resolve a CLI path argument against the storage policy.

    Absolute args are taken verbatim; a non-default arg stays
    project-relative (explicit user intent); the parser default defers
    to the environment override.
    """
    p = Path(arg)
    if p.is_absolute():
        return p
    if str(arg) != str(default):
        return Path(project_root) / p
    return data_root(kind, project_root, default)
