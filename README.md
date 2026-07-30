# TOPOTEX

Topology-aware texture generation for artist meshes.

## Pipeline

```
Reference Image + Mesh → six generated canonical views
    → Surface Conditioner → Face Set Latent  Z_F  [F, 256]
    → Global UV Query Attention (ANY UV layout as query)
    → Flow Matching Texture Generator (rectified flow) → UV texture
```

## Overview

### Problem

Texture pipelines that address the surface through world coordinates are
structurally ambiguous on artist meshes — self-intersections, coincident
duplicated faces and thin shells put *different* surface points at *the
same* xyz; pipelines that address it through one fixed UV atlas are tied to
a single parameterization. TOPOTEX addresses texels by
`(face_id, barycentric)`: a topological identity that survives coincident
geometry, rigid/scale transforms, and re-parameterization.

`Z_F` is a UV-layout-independent, topology-indexed surface latent — one
token per triangle, built from intrinsic geometry + topology + multi-view
appearance only. Validated properties (details and provenance in
`experiments/experiment_log.md`): a held-out unwrap family transfers with
zero gap, rotation/scale invariance is exact (cos = 1.0), connected partial
surface queries work, and textures generated through different UV
parameterizations agree when rebaked.

## Repository

```
topotex/     the installable package (docs/code_architecture.md)
  config.py    typed configs (YAML validated at load)
  data/        schema+contracts, loader, mesh/uv/multiview, offline builder,
               integrity gate, diversity monitors
  layers/      attention / embeddings / topology / flow (pure tensor)
  models/      face_tokenizer / image_encoder / uv_query /
               surface_conditioner / flow_matching / TopoTexModel
  pipelines/   training / inference (TopoTexPipeline) / evaluation
  utils/       distributed / io / logging
train.py sample.py evaluate.py   thin CLIs over the pipelines
configs/     topotex_fm_baseline.yaml (frozen 2K) + topotex_fm_10k.yaml
scripts/     8-GPU wrappers (torchrun train, sharded eval, dataset build)
notebooks/   Dataset_Inspector / Model_Inspector / Technical_Report
experiments/ experiment_log.md + fm_100 / fm_2k / fm_10k records
docs/        architecture / code_architecture / technical_report / protocol /
             workspace_policy (data isolation rules)
tests/       pytest suite incl. hygiene gates (zero tracked symlinks)
```

Persistent data (datasets, checkpoints, runs) lives OUTSIDE the
repository at roots given by `TOPOTEX_DATASET_ROOT`,
`TOPOTEX_SOURCE_ROOT`, `TOPOTEX_CHECKPOINT_ROOT`, `TOPOTEX_RUN_ROOT`
(fallback: the historical project-relative locations). Before any
merge/checkout into this workspace run
`python scripts/check_git_tree_safety.py <commit>`.

## Dataset

Each sample under `output/topotex_dataset/samples/<sample_id>/`
(schema `topotex_dataset@1`):

| query | type | contents |
|---|---|---|
| `uv_queries/uv_000` | canonical | native GLB parameterization |
| `uv_queries/uv_001` | alternative | xatlas re-unwrap (face order preserved, asserted) |
| `uv_queries/uv_002` | partial | deterministic connected face subset (25/50/75%), re-rasterized address maps, re-baked GT |
| `uv_queries/uv_test` | held-out | Blender Smart UV — **evaluation only**, never trained |

Every query holds `uv_address.safetensors` (`uv_vertices f32`, `uv_faces
i32` — always the full layout so rebake works, `face_id i32 [256,256]`
(-1 = background), `barycentric f16 [3,256,256]`, `valid_mask u8`) plus
`gt_texture.png` baked through `(face, bary) → native UV → sRGB bilinear`.
Loader: `datasets.dataset.TopoTexDataset` (train queries vs
`test_uv_queries` hard-isolated).

```bash
PY=/root/miniconda3/envs/geomae/bin/python

# 1) source ingest: textured GLB -> reference/mesh/six-views/gt sample
$PY -m datasets.build_dataset --input-manifest output/source_manifests/glbs_eligible.jsonl \
    --output output/topotex_source --limit 10
# 8-GPU sharded (ids[rank::8], one UniTEX load per worker, atomic publish,
# resume-skip; per-rank manifest_rank_K.jsonl merged by merge_manifest.py
# with no-duplicate / no-missing / schema checks)
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/build_dataset_8gpu.sh \
    output/source_manifests/glbs_eligible.jsonl output/topotex_source

# 2) UV query set: canonical / alternative / partial / held-out per mesh
$PY -m datasets.build_uv_queries --limit 265
```

The gate accepts only: triangle mesh, single base-color texture, per-vertex
UV inside [0,1], no UV overlap. The six views come from the frozen UniTEX
stage-1 generator (MV + delight); view mapping `[0,3,1,4,2,5]` is verified.

## Architecture (36.9M parameters)

- **Surface Conditioner** (`models/surface_conditioner/`): FaceTokenizer
  (intrinsic triangle geometry — edge/scale ratios, angles, log-area — plus
  random-walk topology PE; no xyz, no UV, no face index) → face–view cross
  attention over the six views → sparse Topology Transformer along the
  shared-edge face adjacency → `Z_F [F, 256]` → **Global UV Query
  Attention**: per-texel `[face_token ‖ bary_encoding]` → 8×8 patchify →
  1024 query tokens (+ learned atlas position embedding) → cross attention
  with K/V = `Z_F` → UV condition `[64, 256, 256]`.
- **Flow Matching Generator** (`models/texture_generator/`): rectified
  flow — `x_tau = (1-tau)·x0 + tau·eps`, the patchified transformer
  (`dit.py`, 256², patch 8, AdaLN-Zero) predicts the velocity field;
  sampling is a 50-step Euler ODE from tau=1 to 0. Noise and loss live
  only inside the UV valid mask. Flow matching is the only generator.

## Training

```bash
# official multi-GPU entry (torchrun DDP; groups sharded across ranks,
# effective batch = K * world_size meshes/step; budget stays in exposures)
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_8gpu.sh \
    --samples 2000 --run-name fm_2k [--resume]
# single-GPU (small runs / smoke)
$PY train.py --samples 265 --run-name baseline [--resume]
$PY sample.py   --run checkpoints/baseline --n 4 [--include-heldout]
```

Training uses the frozen recipe in `configs/topotex_fm_baseline.yaml`: one mesh per
step, one query per step drawn with `query_probs` (canonical 0.5 /
alternative 0.3 / partial 0.2), 2000 exposures per mesh; checkpoints carry
model/optimizer/RNG and `--resume` restores them exactly. Evaluation
reports UV PSNR (canonical / alternative / partial region / held-out), the
partial-vs-full gap, and render consistency `R(M,U0,T0)` vs `R(M,U1,T1)`
over six canonical views.
## Evaluation

```bash
$PY evaluate.py --run checkpoints/baseline --n 10   # -> <run>/eval.json
# sharded across GPUs (rank i evaluates samples[i::world]; auto-merge):
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/evaluate_8gpu.sh checkpoints/baseline 32
```

Every official job goes through `scripts/run_experiment.sh <task> <cmd...>`,
which prints the GPU plan (selected GPUs / CUDA_VISIBLE_DEVICES / task name)
and the currently-busy GPUs before launching — no unplanned multi-job
contention.

Protocol (`docs/experiment_protocol.md`): 50-step sampling, seed 20260727,
one shared `Z_F` per mesh. Metrics: UV PSNR (canonical / alternative /
partial region / held-out family), render PSNR + cross-layout render
consistency over six canonical views, and UV seam consistency (with the GT
seam error as baking-floor reference).

## Benchmarking

`scripts/benchmark_training.py` measures training throughput levers on the
frozen model (fp32 / bf16-on-DiT / face-count-bucketed group batching /
packed face graph). Packed grouping is the dominant lever (+54% meshes/sec
at group 4); numbers in `experiments/experiment_log.md`.

## Notebook

- `notebooks/Dataset_Inspector.ipynb` — input data, the three UV queries +
  held-out (face_id / barycentric / valid / GT), partial-query statistics.
- `notebooks/Model_Inspector.ipynb` — Z_F PCA on the mesh, UV query
  attention trace, same-mesh-different-query generation, render consistency.
- `notebooks/Technical_Report.ipynb` — presentation-grade
  walkthrough of the full pipeline (all shapes/params printed from live
  forwards).

## Docs & history

- `docs/architecture.md` — module-level architecture reference.
- `docs/technical_report.md` — method + frozen baseline numbers.
- `experiments/experiment_log.md` — every past experiment with date, goal,
  config, result, conclusion, and commit. Code for each entry lives in git
  history at the referenced commit.

## Tests

```bash
$PY -m pytest tests/
```

