# Repository Audit

Full inventory of the workspace: what exists, why, and whether it stays.
The tree was consolidated in `6226eb1`/`029cf60` (single final method;
history in git + `experiments/experiment_log.md`), the **final research
consolidation** (2026-07-30) reduced it to the paper mainline, and the
architecture refactor packaged it (`topotex/`, see
`docs/code_architecture.md`):
Image + Mesh → Face Set Latent `Z_F` → UV Query → Flow Matching → Texture.

## Tracked tree (git)

| path | purpose | keep |
|---|---|---|
| `train.py` / `sample.py` / `evaluate.py` | thin CLIs over `topotex.pipelines` (DDP training, deterministic sampling, sharded evaluation) | ✔ |
| `configs/topotex_fm_baseline.yaml` | frozen 2K recipe (generator fm, packed K=4, bf16, query_probs) | ✔ |
| `configs/topotex_fm_10k.yaml` | official 10K config (token dim 384; scaling-study decision) | ✔ |
| `topotex/data/builder.py` | offline construction CLI (`source`/`queries`/`merge`): gates, six views, atomic publish, 8-GPU sharding, manifest merge | ✔ |
| `topotex/data/dataset.py` | the one loader (`TopoTexDataset`, typed batches) | ✔ |
| `topotex/data/mesh.py` | rendering/geometry utilities (cameras, rasterized views, rebake, seam metric) | ✔ |
| `topotex/data/uv.py` | deterministic UV address rasterizer + connected face subsets | ✔ |
| `topotex/data/multiview.py` | frozen UniTEX stage-1 adapter (six canonical views, offline only) | ✔ |
| `topotex/data/diversity.py` + `statistics.py` | dataset distribution monitors + final stats | ✔ |
| `topotex/data/integrity.py` | recursive post-finalize integrity gate (symlinks, readability, manifest completeness, provenance, deep hashes) | ✔ |
| `topotex/models/` + `topotex/layers/` | face_tokenizer / image_encoder / uv_query / surface_conditioner / flow_matching (MiniDiT) / TopoTexModel; layers: attention / embeddings / topology / flow | ✔ |
| `topotex/config.py` + `topotex/data/schema.py` | typed configs + typed tensor contracts | ✔ |
| `topotex/pipelines/` | training / inference (`TopoTexPipeline`) / evaluation | ✔ |
| `notebooks/` | Dataset_Inspector / Model_Inspector / Technical_Report + Pipeline_Playground (full pipeline on user-supplied GLBs in `workspace/`) — all execute clean | ✔ |
| `experiments/experiment_log.md` | the only home for past experiments (date/goal/dataset/model/result/conclusion/commit) | ✔ |
| `experiments/fm_100/` `fm_2k/` `fm_10k/` | one `record.json` each (commit, config SHA, dataset SHA, metrics); fm_2k also keeps its stage-gate report; fm_10k keeps the scaling-study record | ✔ |
| `docs/` | architecture / technical_report / experiment_protocol / current_status / this audit | ✔ |
| `tests/` | pytest suite: model invariants, FM schedule invariants, dataset gates, training path, notebook smoke, repo hygiene | ✔ |
| `scripts/` | train_8gpu / evaluate_8gpu / run_experiment / build_dataset_8gpu / benchmark_training | ✔ |
| `README.md` / `CHANGELOG.md` / `.gitignore` | current-version-only intro; ignores output/ runs/ reports/ checkpoints/ | ✔ |

## Untracked disk artifacts (gitignored)

| path | purpose | disposition |
|---|---|---|
| `output/topotex_source` → symlink | source samples (10K build target; the training dataset symlinks into it) | keep |
| `output/topotex_dataset` → symlink | the training dataset (4 UV queries per mesh) | keep |
| `output/source_manifests/` | GLB selection manifests (`glbs*.jsonl`, moved from the repo root) + the 10K build input | keep |
| `output/asset_manifest.jsonl` | TexVerse download manifest (source-of-truth for asset paths) | keep |
| `checkpoints/baseline/` | fm_2k official baseline (optimizer+RNG, resumable) + eval.json | keep |
| `checkpoints/fm_100_reference/` | fm_100 record's checkpoint | keep |
| `dataset/` (TexVerse) | raw downloaded assets — never deleted | keep |
| `paper/` | external reference material (not project code) | keep, ignored |
| `workspace/` | user-supplied GLBs + Pipeline_Playground outputs | keep, ignored |
| `runs/` | training run outputs — recreated by training, pruned after records land | ephemeral |
| `reports/` | stage-gate working dirs — decision reports migrate into `experiments/` | ephemeral |

## Removed in the final consolidation (git history keeps everything)

| what | reason |
|---|---|
| `models/texture_generator/diffusion.py` + its tests + the `--generator diffusion` branch | flow matching is the only generator; FM invariants now covered by `tests/test_flow_matching.py` |
| `checkpoints/dit_reference` | retired diffusion reference; numbers preserved in the log and `docs/technical_report.md` |
| `runs/` (~16 GB: fm/scaling intermediates) | finals live in `checkpoints/`, metrics in `experiments/*/record.json` |
| `reports/` galleries + stage-A dumps | the decision report was migrated to `experiments/fm_2k/stage_gate_report.md` |
| the superseded 19 GB UV-query base under `output/` | the training dataset now references only the source dataset (265 affected samples rebuilt with the frozen builder) |
| `experiments/protocol/baseline_1k_*` | superseded 1k protocol (entry retained in the log) |
| `glbs*.jsonl` at the repo root | moved to `output/source_manifests/` (data, not code) |

## Guarantees

- `tests/test_repo_hygiene.py` scans every tracked text file for
  exploratory naming and for dangling imports of removed modules
  (including the retired diffusion schedule) — regressions fail CI.
- `tests/test_notebook_smoke.py` executes all three notebooks end to end
  against the real checkpoint.
