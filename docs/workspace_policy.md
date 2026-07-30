# TOPOTEX workspace policy

1. **The only canonical workspace is `/root/youjiaZhang/TopoTex`.**
   It always holds the latest *verified* TOPOTEX code and is the sole
   entry point for dataset building, official training, inference,
   evaluation, notebooks, and GitHub main synchronization.

2. Canonical carries verified code only. Unverified work never lands in
   canonical directly.

3. Temporary worktrees live under
   `/root/youjiaZhang/.worktrees/topotex/<task>` — one task, one clearly
   named branch each. They must not copy large data, and are removed
   immediately after their branch merges (`git worktree remove` +
   `git worktree prune`).

4. Historical versions are kept as git commits/tags, plus (when needed)
   exported source tarballs under `/root/youjiaZhang/topotex_snapshots/`.
   Duplicate source-tree copies under stale names are forbidden.

5. Large assets (source GLBs, datasets, generated MV, checkpoints, runs)
   live at stable fixed paths and are never duplicated per worktree;
   configs point at those fixed paths. Cross-worktree symlinks are a
   development convenience only — never the storage of record.

6. Official training and dataset builds launch from canonical by default.

7. At no point may two directories both look like "the latest TOPOTEX".
   If a worktree survives its task, it is a bug in the process — merge or
   delete it.

## Data isolation (post-incident, 2026-07-30 — mandatory)

8. **The repository contains code only.** Persistent data lives at
   protected roots OUTSIDE the repository, resolved exclusively through
   environment variables (`TOPOTEX_DATASET_ROOT`,
   `TOPOTEX_SOURCE_ROOT`, `TOPOTEX_CHECKPOINT_ROOT`,
   `TOPOTEX_RUN_ROOT` — see `topotex/paths.py`). No symlink may point
   from any repository checkout to persistent data.

9. **Worktrees carry no data**: no dataset/checkpoint mounts, no data
   symlinks. Formal jobs (training, dataset builds, evaluation) start
   only from the canonical workspace; the canonical branch is never
   switched for experiments.

10. **Merge/checkout preflight is mandatory.** Before ANY merge or
    checkout into the canonical workspace run:

        python scripts/check_git_tree_safety.py <target-commit>

    It refuses trees that track symlinks, reserved data names
    (dataset/output/checkpoints/workspace/cache/runs), or entries
    colliding with existing real paths. The permanent hygiene tests
    (`tests/test_repo_hygiene.py::test_no_tracked_symlinks`,
    `::test_no_reserved_data_paths_tracked`) enforce the same
    invariants on every commit.

