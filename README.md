# TOPOTEX

Topology-aware texture generation for artist meshes.

## Pipeline

```
Reference Image + Textureless Mesh → six generated canonical views (UniTEX stage-1, offline)
    → Surface Conditioner → Face Set Latent  Z_F  [F, 384]
    → Factorized Dense UV Query Encoder (ANY UV layout as query)
    → Global UV Query Attention → UV condition [64, 256, 256]
    → Flow Matching (rectified flow, Euler-50) → UV texture [3, 256, 256]
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
`experiments/experiment_log.md`): rotation/scale invariance is exact
(cos = 1.0), face-permutation equivariance holds (face ids are pointers,
not semantic embeddings), connected partial surface queries work, and
textures generated through different UV parameterizations agree when
rebaked.

### Protocol in three sentences

**Split by object** — the train/test unit is the source GLB asset
(content-hash verified unique); every query of an object inherits its
split. **Augment by UV query** — native / xatlas / blender_smart layouts
all train (uniform within a 0.80 full-query budget) plus a connected
partial surface query (0.20). **Evaluate on unseen objects** — 500
never-trained assets; a same-object UV family is *not* the
generalization axis.

## Repository

```
topotex/     the installable package (docs/architecture.md)
  config.py    typed configs (YAML validated at load)
  paths.py     data-root resolution via TOPOTEX_*_ROOT env vars
  data/        schema+contracts, loader, mesh/uv/multiview, offline builder,
               integrity gate, diversity/statistics monitors
  layers/      attention / embeddings / topology / flow (pure tensor)
  models/      face_tokenizer / image_encoder / uv_query (factorized) /
               surface_conditioner / flow_matching / TopoTexModel
  pipelines/   training / inference (TopoTexPipeline) / evaluation
  utils/       distributed / io / logging
train.py sample.py evaluate.py   thin CLIs over the pipelines
configs/     topotex_fm_10k.yaml (official dim384) + topotex_fm_baseline.yaml
scripts/     train_8gpu / evaluate_8gpu / build_dataset_8gpu wrappers,
             preflight_training.py, check_git_tree_safety.py
notebooks/   Dataset_Inspector / Model_Inspector / Technical_Report
experiments/ experiment_log.md + baseline / fm_100 / fm_2k / fm_10k records
docs/        architecture / data_layout / experiment_protocol / safety
tests/       pytest suite incl. hygiene gates (zero tracked symlinks)
```

**The repository contains code only.** Persistent data (raw GLBs,
datasets, checkpoints, runs) lives at a protected root outside the
repository, resolved via `TOPOTEX_SOURCE_ROOT` / `TOPOTEX_DATASET_ROOT` /
`TOPOTEX_CHECKPOINT_ROOT` / `TOPOTEX_RUN_ROOT` (see `topotex/paths.py`
and `docs/data_layout.md`). No repository symlink may point at data.
Before any merge/checkout into the workspace:
`python scripts/check_git_tree_safety.py <commit>`; before any formal
run: `python scripts/preflight_training.py` (both enforced — the second
is wired into `scripts/train_8gpu.sh`). Rationale and rules:
`docs/safety.md`.

## Dataset (frozen object-level baseline)

**4,590 complete objects** (TexVerse-1K sources) — train **4,090** /
unseen test **500** (`object_split.json`, seed 20260727). Frozen
hashes: manifest `a6aef671…`, split `21972f1a…`. Per-sample layout and
audited dtypes: `docs/data_layout.md`.

On-disk query dirs are frozen; the semantic mapping is the contract
(`topotex.data.schema.QUERY_SEMANTICS`):

| dir | layout | mask | role |
|---|---|---|---|
| `uv_000` | **native** | full | native GLB parameterization |
| `uv_001` | **xatlas** | full | re-unwrap (face order preserved, asserted) |
| `uv_test` | **blender_smart** | full | Smart-UV — trains as layout augmentation |
| `uv_002` | **partial** (native) | connected_partial | surface mask/query, *not* an unwrap family |

Every query holds `uv_address.safetensors` (`uv_vertices f32`,
`uv_faces i32` — always the full layout so rebake works, `face_id i32
[256,256]` (−1 = background), `barycentric f16 [3,256,256]`,
`valid_mask u8`) plus `gt_texture.png` baked through
`(face, bary) → native UV → sRGB bilinear`. Dataset↔source payload is
hardlinked (zero duplication). Loader: `topotex.data.dataset.TopoTexDataset`.

```bash
PY=/root/miniconda3/envs/geomae/bin/python
# offline construction (source ingest / UV query set / manifest merge):
$PY -m topotex.data.builder source  --input-manifest <ids.jsonl> --output $TOPOTEX_SOURCE_ROOT
$PY -m topotex.data.builder queries --output $TOPOTEX_DATASET_ROOT --limit <N>
$PY -m topotex.data.builder merge   --output $TOPOTEX_SOURCE_ROOT --input-manifest <ids.jsonl>
# deep integrity gate (content-level checks):
$PY -m topotex.data.integrity --root $TOPOTEX_DATASET_ROOT --kind dataset --deep
```

## Architecture (53.4M parameters, dim384)

- **Surface Conditioner**: FaceTokenizer (intrinsic triangle geometry —
  edge/scale ratios, angles, log-area — plus memory-bounded random-walk
  topology PE; no xyz, no UV, no face index) → face–view cross attention
  over the six views → sparse Topology Transformer along the shared-edge
  face adjacency → `Z_F [F, 384]`.
- **Factorized Dense UV Query Encoder** (the only implementation):
  face address `Linear(384, 96)` applied ONCE to `Z_F` then gathered per
  texel; barycentric address `Fourier(27) → MLP(27→96→96)` on valid
  texels only; `LayerNorm(face + bary)` fusion; learned background
  embedding (never indexes `Z_F`); dense query `[96, 256, 256]` →
  `Conv2d(96, 384, 8, 8)` → 1024 UV tokens (+ learned atlas position
  embedding).
- **Global UV Query Attention**: depth 4, heads 8, pre-LN, FFN×4,
  K/V = `Z_F`, no routing, no hard mask → `Linear(384, 8·8·64)` →
  unpatchify → UV condition `[64, 256, 256]` (exact zero outside the
  valid mask).
- **Flow Matching generator** (the only generator): rectified flow —
  `x_tau = (1-tau)·x0 + tau·eps`, a patchified transformer (256², patch
  8, AdaLN-Zero) predicts the velocity field; sampling is a 50-step
  Euler ODE. Noise and loss live only inside the UV valid mask.

Checkpoints from the earlier concat-bottleneck encoder do not load —
fail-fast, no migration, no silent partial load.

## Training

```bash
# official multi-GPU entry (preflight-gated torchrun DDP)
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_8gpu.sh \
    --config configs/topotex_fm_10k.yaml \
    --ids-file $TOPOTEX_DATA/object_split.json \
    --samples 4090 --run-name <run> [--resume]
```

Frozen recipe (`configs/topotex_fm_10k.yaml`): dim384 / Dq96, BF16,
packed face-graph groups K=4 (face-count buckets, topology-PE cache),
query sampling `[native, xatlas, partial, blender_smart] =
[0.2667, 0.2667, 0.2, 0.2667]`, 2,000 exposures per mesh
(`steps = ceil(N·2000 / (world·K))`), random initialization.
Checkpoints carry model/optimizer/RNG plus config, manifest and split
SHAs; `--resume` restores exactly.

**Current official run**: `fm_baseline_dim384_factorized` — 4,088
feasible train objects (two face-count filter escapees excluded at
launch level, provenance in `experiments/baseline/record.json`),
255,500 steps on 8×H800. Acceptance before launch (all green,
`experiments/baseline/acceptance_record.json`): single-object overfit
(loss EMA 1.465→0.006; all four queries 43–44 dB; seam ratio 1.02),
10-object stage (packed batching, resume, no test leakage), 100-object
DDP smoke (median util 96%, provenance verified).

## Evaluation

```bash
$PY evaluate.py --run $TOPOTEX_RUN_ROOT/<run> --n 10          # -> <run>/eval.json
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/evaluate_8gpu.sh $TOPOTEX_RUN_ROOT/<run> 500
```

Protocol (`docs/experiment_protocol.md`): Euler-50, seed 20260727, one
shared `Z_F` per object. Metrics on the 500 unseen objects: per-layout
UV PSNR (native / xatlas / blender_smart / partial region), render
PSNR, cross-layout render consistency `R(M,U0,T0)` vs `R(M,U1,T1)`, and
UV seam consistency against the GT seam floor — plus random32 / worst8
galleries.

## Notebooks

Three maintained notebooks (sources committed output-free; executed
HTML goes to `TOPOTEX_RUN_ROOT`); all tensors come from live forwards:

- `Dataset_Inspector.ipynb` — overview + frozen hashes, random
  train/test object inspection (three full layouts + partial,
  face_id / barycentric / masks), per-sample consistency checks.
- `Model_Inspector.ipynb` — training monitor, factorized-encoder
  intermediates with shape asserts (`Z_F`, face-address table,
  bary-address map, dense query, UV tokens, attention output,
  condition, FM velocity), four-query inference with seam heatmap,
  FM trajectory. `SUBSET="train"|"test"` picks the pool.
- `Technical_Report.ipynb` — presentation-grade walkthrough of the
  full pipeline on one real object.

## Tests

```bash
$PY -m pytest tests/
```

Includes permanent hygiene gates (zero tracked symlinks, no reserved
data names), factorized-encoder contracts (permutation equivariance,
barycentric sensitivity, background isolation, determinism), RWPE
blocked-vs-dense equality, FM schedule invariants, and notebook smokes.

## Docs & history

- `docs/architecture.md` — module-level architecture reference.
- `docs/data_layout.md` — protected data root, semantics, audited formats.
- `docs/safety.md` — data-isolation rules born from the 2026-07-30
  incident (mandatory preflights, destructive-command discipline).
- `experiments/experiment_log.md` — every past experiment with date,
  goal, config, result, conclusion, and commit. Code for each entry
  lives in git history at the referenced commit.
