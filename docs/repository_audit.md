# Repository Audit

Full inventory of the workspace: what exists, why, and whether it stays.
The tree was consolidated in commit `6226eb1` (single final method; history
in git + `experiments/experiment_log.md`) and the build/notebook
infrastructure finalized in `5d4a58d`; this audit records the resulting
state and the disposition of everything else on disk.

## Tracked tree (git)

| path | purpose | keep |
|---|---|---|
| `train.py` / `sample.py` / `evaluate.py` | single-command entry points for the frozen pipeline | ✔ |
| `configs/topotex_fm_baseline.yaml` | the one frozen recipe (model dims, generator, query_probs, efficiency settings) | ✔ |
| `datasets/build_dataset.py` | GLB → source sample (gates, six views, atomic publish, 8-GPU sharding) | ✔ |
| `datasets/build_uv_queries.py` | source sample → canonical/alternative/partial/held-out UV queries | ✔ |
| `datasets/merge_manifest.py` | per-rank manifest merge: no-duplicate / no-missing / schema checks | ✔ |
| `datasets/dataset.py` | the one loader (`TopoTexDataset`) | ✔ |
| `datasets/uv_query.py` | face adjacency + connected surface subsets | ✔ |
| `datasets/mesh_utils.py` | rendering/geometry utilities (cameras, rasterized views, rebake) | ✔ |
| `datasets/rasterizer.py` | deterministic UV address rasterizer (+ self-verification) | ✔ |
| `datasets/mv_generator.py` | frozen UniTEX stage-1 adapter (six canonical views) | ✔ |
| `models/surface_conditioner/` | face_tokenizer / image_encoder / face_image_attention / topology_pe / topology_transformer / uv_query_attention / conditioner | ✔ |
| `models/texture_generator/` | dit.py (MiniDiT) + diffusion.py (masked cosine schedule) | ✔ |
| `notebooks/` | Dataset_Inspector / Model_Inspector / Technical_Report — exactly three, all execute clean | ✔ |
| `experiments/experiment_log.md` | the only home for past experiments (date/goal/dataset/model/result/conclusion/commit) | ✔ |
| `experiments/protocol/` | frozen 1k-baseline manifest + split (provenance of the log's first entry) | ✔ |
| `docs/` | architecture.md / technical_report.md / experiment_protocol.md / this audit | ✔ |
| `tests/` | 44 tests: model invariants, dataset gates, training path, notebook smoke, repo hygiene | ✔ |
| `glbs*.jsonl` | source-asset manifests consumed by `build_dataset.py` | ✔ |
| `scripts/build_dataset_8gpu.sh` | 8-GPU build sharder + auto-merge | ✔ |
| `README.md` / `.gitignore` | current-version-only intro; ignores output/ cache/ runs/ checkpoints/ | ✔ |

## Untracked disk artifacts (gitignored)

| path | size | purpose | disposition |
|---|---|---|---|
| `output/topotex_source` → `topotex_dataset_v…` | 8.8 GB | source dataset, 1109 samples (raw data — kept per policy) | keep |
| `output/topotex_dataset` → symlink | 1.2 GB | the 265-sample training dataset (4 UV queries each) | keep |
| `output/topotex_multi_uv_v0` | 19 GB | 1k-mesh UV-query base (999/1000 built) — raw material for the future 1k dataset | keep |
| `output/asset_manifest.jsonl` | 13 MB | TexVerse download manifest (source-of-truth for asset paths) | keep |
| `checkpoints/baseline/` | 848 MB | the ONLY kept checkpoint (final baseline, optimizer+RNG, resumable) + eval.json | keep |
| `dataset/` (TexVerse) | 510 GB | raw downloaded assets — never deleted | keep |
| `paper/` | 241 MB | external reference material (not project code) | keep, ignored |
| `output/asset_scan*.json` and a superseded sample-list jsonl | ~5 MB | regenerable asset-scan intermediates | deleted |
| `output/spot_pipeline/`, `output/generalization_v…_smoke/` | ~70 MB | one-off demo / smoke outputs | deleted |
| `runs/` | — | scratch of training smoke tests; recreated/removed by tests | ignored |

## Removed in consolidation (git history keeps everything)

| what | reason |
|---|---|
| duplicate code paths (per-texel decoder, superseded loaders/builders, exploratory query samplers, DDP scripts) | one final method only; superseded by the query-attention pipeline |
| all intermediate checkpoints (~14 GB of runs) | only the final baseline checkpoint matters; numbers live in the log |
| all experiment report dumps (JSON/md galleries) | migrated to `experiments/experiment_log.md` first, then deleted |
| exploratory notebooks (2 generations of inspectors + validation sections) | replaced by the three maintained notebooks |
| version-suffixed configs and entry-point wrappers | a single frozen config + train/sample/evaluate |

## Guarantees

- `tests/test_repo_hygiene.py` scans every tracked text file for exploratory
  naming and for dangling imports of removed modules — regressions fail CI.
- `tests/test_notebook_smoke.py` executes all three notebooks end to end
  against the real checkpoint.
