# Changelog

## Unreleased — final research repository consolidation (2026-07-30)

- **Recursive data integrity gate.** `datasets/verify_integrity.py` checks
  every official sample after each build-stage finalize: dangling symlinks
  at any depth, unreadable required files, manifest completeness
  (missing/incomplete official samples fail; unmanifested dirs warn),
  builder provenance, and `--deep` content hashes; covered by
  `tests/test_verify_integrity.py`.
- **Query provenance metadata.** `build_uv_queries` records
  `query_schema_version`, `query_builder_commit`, per-query
  `face_id_sha256` / `barycentric_sha256`, and `source_texture_sha256` in
  every `meta.json` (existing samples backfilled additively).
- **Protocol breakpoint documented.** Pre-2026-07-30 partial-axis numbers
  are not comparable to the repaired partial data; all other axes remain
  comparable (`docs/experiment_protocol.md`).

- **One mainline.** Image + Mesh → Face Set Latent `Z_F` → UV Query →
  Flow Matching → Texture; status snapshot in `docs/current_status.md`.
- **Flow matching is the only generator.** The masked-diffusion schedule,
  its retired reference checkpoint, and the `--generator diffusion` branch
  are removed (code recoverable from git history); FM schedule invariants
  covered by `tests/test_flow_matching.py`.
- **Experiments unified.** `experiments/experiment_log.md` +
  `fm_100` / `fm_2k` / `fm_10k` records (commit, config SHA, dataset SHA,
  metrics); the model scaling study lives under
  `experiments/fm_10k/scaling/`; the fm_2k stage-gate decision report is
  tracked at `experiments/fm_2k/stage_gate_report.md`.
- **Historical outputs deleted.** `runs/`, `reports/`, intermediate
  checkpoints, and the superseded 1k UV-query base; kept checkpoints are
  `checkpoints/baseline` (fm_2k) and `checkpoints/fm_100_reference`.

## v0.1.0-fm-baseline — 2026-07-29

Initial public release of the TOPOTEX research codebase.

- **Flow matching is the official texture generator.** Rectified flow
  (velocity prediction on the frozen patchified-transformer backbone,
  50-step Euler sampling) replaces masked diffusion as the default
  (`generator: fm` in `configs/topotex_fm_baseline.yaml`).
- **Diffusion retired to reference status.** The masked-diffusion schedule
  remains loadable only for the previous reference checkpoint
  (`checkpoints/dit_reference`, untracked); no further work on it.
- **Clean repository release.** Single-commit `main` containing only the
  final method: dataset pipeline (8-GPU sharded build, atomic publish,
  resume, manifest-merge validation), frozen training recipe (packed
  face-graph batching K=4, bf16 velocity net, face-count bucket sampler),
  evaluation protocol (UV PSNR, render consistency, held-out UV family,
  UV seam consistency), three maintained notebooks, docs, 47-test suite.
  Historical exploration lives outside this repository; conclusions are
  summarized in `experiments/experiment_log.md`.
