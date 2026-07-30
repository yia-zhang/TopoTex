# TOPOTEX — Current Status

Snapshot of the official research mainline after the final consolidation
(2026-07-30). One pipeline, one generator, one evaluation protocol.

## Official pipeline

```
Image + Mesh
  → Surface Conditioner            (face tokenizer + face–image cross
                                    attention + sparse topology transformer)
  → Face Set Latent  Z_F           (one token per triangle;
                                    topology-indexed, UV-free)
  → Global UV Query Attention      (any UV layout as (face_id, barycentric)
                                    query — canonical / alternative /
                                    partial / held-out)
  → Flow Matching generator        (rectified flow, MiniDiT velocity net,
                                    50-step Euler, valid-mask interior only)
  → Texture
```

Flow matching is the **only** generator (no other schedule exists in the
code; history lives in git and `experiments/experiment_log.md`). The UV
query encoder is the factorized dense implementation (face-address
projection + barycentric MLP + Conv2d patch embedding, Dq = cond_dim/4);
checkpoints from the concat-bottleneck encoder do not load — fail-fast,
no migration.

## Frozen configuration

| config | role |
|---|---|
| `configs/topotex_fm_baseline.yaml` | frozen 2K recipe (token dim 256, packed K=4, bf16, DDP world 8, query_probs 0.5/0.3/0.2) |
| `configs/topotex_fm_10k.yaml` | official 10K config — identical except token dim **384** (scaling-study decision); no further architecture search before the 10K run completes |

## Completed milestones

| milestone | record |
|---|---|
| Face Set Latent validation (invariance, multi-UV consistency, partial queries) | `experiments/experiment_log.md`, `docs/technical_report.md` |
| FM baseline at 100 meshes | `experiments/fm_100/record.json` (checkpoint `checkpoints/fm_100_reference`) |
| FM 2K baseline — Z_F representation scaling validated (held-out +1.2–3.1 dB over fm_100, two disjoint eval groups agree < 0.6 dB) | `experiments/fm_2k/record.json` (+ `stage_gate_report.md`; checkpoint `checkpoints/baseline`) |
| 8-GPU training/eval/build infrastructure (243 mesh-exposures/s) | `scripts/`, log entries |
| Model scaling study — dim384 chosen; held-out flat across capacity (generalization is data-driven) | `experiments/fm_10k/scaling/results.json` |

## In progress — 10K scaling

- 10K source build running (8-GPU sharded, resume-safe, manifest + SHA).
- Strict order after the build:
  10K finalize → **integrity gate** (source + dataset,
  `datasets/verify_integrity.py`, deep) → frozen 9500/500 mesh-level split
  (`experiments/protocol/scaling_10k_split.json`) → dataset diversity
  report → dim384 FM training (**launch only on explicit user confirmation**)
  → 10% / 50% / 100% gates (gallery + unseen-mesh evaluation + query
  consistency) → unseen-mesh evaluation.
- Six required evaluation axes: canonical UV, alternative UV, held-out UV,
  partial query, seam consistency, unseen-mesh validation
  (`experiments/fm_10k/record.json`).
- Protocol breakpoint: pre-2026-07-30 **partial-axis** numbers (fm_100 /
  fm_2k records) are not comparable to the repaired partial data; the
  canonical / alternative / held-out / full-query consistency / seam axes
  remain comparable (`docs/experiment_protocol.md`).

## Frozen until the 10K run completes

- no deleting or moving data directories
- no renaming core modules
- no changes to the dataset schema, `Z_F`, UV Query, or the FM generator
- maintained notebooks: Dataset_Inspector / Model_Inspector /
  Technical_Report + Pipeline_Playground (interactive pipeline on
  user-supplied GLBs); refreshed together once the 10K data is ready

## Entry points

```bash
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_8gpu.sh --run-name <name> ...
python sample.py   --run checkpoints/baseline --n 4
python evaluate.py --run checkpoints/baseline --n 32
bash scripts/build_dataset_8gpu.sh <manifest> output/topotex_source <limit>
```

Notebooks (`Dataset_Inspector` / `Model_Inspector` / `Technical_Report`)
are the maintained dashboards; all three execute clean end to end.

## Data-path policy (post-incident)

Persistent data roots are resolved via environment variables
(`TOPOTEX_DATASET_ROOT`, `TOPOTEX_SOURCE_ROOT`, `TOPOTEX_CHECKPOINT_ROOT`,
`TOPOTEX_RUN_ROOT`; see `topotex/paths.py`). The repository contains code
only — no symlink may point from a checkout to persistent data, no tracked
entry may use a reserved data name, and every merge/checkout into the
canonical workspace must first pass
`python scripts/check_git_tree_safety.py <commit>` (enforced permanently by
`tests/test_repo_hygiene.py`).

## 2026-07-30 data incident (status)

An interrupted fast-forward checkout of a branch that tracked worktree
symlinks deleted the gitignored real data directories (full analysis:
`forensic_report.md` in the rescue set; inventories + admin recovery
request + deterministic rebuild plan alongside it). Rescued and
SHA-verified in quadruplicate: 8,889/9,919 query samples, the frozen
10,032-id build manifest, the 9,419/500 split (seed 20260727), golden
reference tensors. The 10K run (`fm_10k_dim384_factorized`, random init)
starts only after the dataset is restored or rebuilt at an
administrator-approved protected root.

