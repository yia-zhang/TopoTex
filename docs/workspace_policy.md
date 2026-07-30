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
   `git archive` snapshots under `/root/youjiaZhang/topotex_snapshots/`.
   Source-tree copies named old/backup/etc. are forbidden.

5. Large assets (source GLBs, datasets, generated MV, checkpoints, runs)
   live at stable fixed paths and are never duplicated per worktree;
   configs point at those fixed paths. Cross-worktree symlinks are a
   development convenience only — never the storage of record.

6. Official training and dataset builds launch from canonical by default.

7. At no point may two directories both look like "the latest TOPOTEX".
   If a worktree survives its task, it is a bug in the process — merge or
   delete it.
