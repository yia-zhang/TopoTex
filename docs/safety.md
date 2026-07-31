# TOPOTEX Safety Rules (post-incident, permanent)

Root cause of the 2026-07-30 data loss: a fast-forward checkout of a
branch that tracked symlinks at gitignored data paths — git treats
ignored paths as expendable and deleted the real directories. Full
forensics live in the rescue set (`topotex_rescue_2026-07-30/`).

## Storage separation

1. The repository contains **code only**. Persistent data lives at the
   protected root (`/root/youjiaZhang/topotex_data`), resolved via
   `TOPOTEX_*_ROOT` environment variables (`topotex/paths.py`).
   No repository symlink may point at persistent data; reserved names
   (`dataset output checkpoints workspace cache runs`) must never be
   tracked.
2. Temporary worktrees (if ever needed) carry no data and are removed
   after merge. Formal jobs launch from the canonical workspace only;
   the canonical branch is never switched for experiments.

## Mandatory gates

- **Before any merge/checkout into canonical**:
  `python scripts/check_git_tree_safety.py <commit>` (fails on tracked
  symlinks, reserved-name entries, collisions with real paths).
- **Permanent hygiene tests** (`tests/test_repo_hygiene.py`): zero
  tracked symlinks (HEAD + index), no reserved data names at repo root.
- **Before formal training**: `python scripts/preflight_training.py`
  (code/env/data/GPU/storage — read-only; it never installs or
  mutates anything).

## Destructive-command discipline

Before any recursive deletion: print the target's realpath, mount
point, symlink status and size, and confirm it is not under a data
root. Never `rm -rf` a variable, never `find -delete`, never
`rsync --delete`; deletions are explicit allowlists only.

## Environment fragility (this host)

Container overlay (/tmp, /opt, /var, apt binaries) has been observed to
reset/lose binaries (libX11, git). JuiceFS state survives. Keep rescue
copies on JuiceFS, verify tools before long jobs (preflight does), and
never rely on /tmp for durable artifacts.
