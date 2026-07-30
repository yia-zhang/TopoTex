# TOPOTEX Experiment Log

Chronological record of the experiments that shaped the final method. Code
for every entry lives in git history at the referenced commit; raw metric
JSONs were intermediates and are not kept — the numbers that matter are
recorded here. Unified evaluation protocol unless noted: DDIM-50,
seed 20260727, shared face-set latent per mesh, six canonical render views.

Run-to-run noise floor: identical-seed reruns (cuDNN nondeterminism) move
sampled metrics by about ±1.5 dB at the 10-mesh scale — differences below
that are not conclusions.

---

## Baseline 1k (generalization protocol)

- **date** 2026-07-27
- **goal** First train/val/test generalization number for the minimal
  pipeline (fixed canonical-UV decoder, pre-query-attention architecture).
- **dataset** 1000 frozen samples (900/50/50 split, seed 20260727;
  manifest + split under `experiments/`), 256² textures, six frozen
  UniTEX views per mesh.
- **model** 23.6M texture DiT + 8.5M surface conditioner (fixed canonical-UV decoder)
- **config** masked cosine
  diffusion, eps-pred, lr 3e-4, high-t emphasis, 2000 exposures/mesh,
  225k DDP steps from random init.
- **result** UV PSNR train 24.72 / val 13.71 / test 14.38 dB.
- **conclusion** Pipeline learns and transfers; the val/test gap set the
  agenda for representation work (surface-indexed conditioning) rather
  than capacity scaling.
- **commit** 00d2351

## Multi-UV query validation

- **date** 2026-07-28
- **goal** Test the core hypothesis: the mesh essence is a Face Set latent
  `Z_F`; a UV parameterization is only a 2D query. Introduce the Global UV
  Query Attention decoder (1024 patch query tokens, K/V = face tokens).
- **dataset** 10→100 meshes × 3 UV parameterizations per mesh (native,
  two xatlas variants), all face-order-preserving (asserted), GT re-baked
  through `(face, barycentric) → native UV`.
- **model** conditioner submodules + Global UV Query Attention decoder, 36.9M
- **config** one random UV query per step; recipe as above.
- **result** Single-mesh overfit: UV PSNR up to 33.2 dB, cross-parameterization
  render consistency 33.4 dB. 10-mesh run: consistency 18.23 dB > GT render
  fidelity 17.10 dB on 8/10 meshes (decoupling signature).
- **conclusion** Textures generated through different UV layouts agree with
  each other more than each agrees with GT — they share one surface signal
  read from `Z_F`. Hypothesis confirmed; decoder adopted.
- **commit** afbfc37

## Held-out UV family

- **date** 2026-07-28
- **goal** Does `Z_F` transfer to a UV family never seen in training?
- **dataset** 100 meshes; 4th query per mesh from Blender Smart UV
  (loader hard-isolates it from training); later re-checked at 265 meshes.
- **model** the 36.9M query-attention pipeline (unchanged)
- **config** zero-shot on the 10-mesh checkpoint; later the 265-mesh run.
- **result** Seen 15.08 vs held-out 15.01 dB (zero gap). At 265-mesh
  scale: gap +0.04 dB / −0.26 dB on two independent 10-mesh groups.
- **conclusion** No drop on an unseen unwrap family at any scale tested —
  layout generalization comes from the face-addressed query architecture.
- **commit** cd9f897 (100-mesh), 0121e34 (265-mesh)

## Rigid transform & scale invariance

- **date** 2026-07-28
- **goal** `Z_F` must not depend on world coordinates: rotation,
  translation, uniform scale.
- **dataset** 10 meshes, random rotation/translation, scales 0.1×–10×.
- **model** the 36.9M query-attention pipeline (unchanged)
- **config** zero-shot face-token cosine + texture/render agreement.
- **result** Rotation/translation exact (cos = 1.0000, texture 83.8 dB).
  Scale initially looked imperfect (cos down to 0.57) — root-caused to a
  measurement artifact: a cached face graph (global scale, edge lengths)
  from the untransformed mesh was reused for the transformed one. The
  intrinsic features are ratio-based and exact; with a graph rebuilt from
  the actual vertices, cos = 1.0000 across all meshes and scales. Fix:
  `encode_faces` validates the cached graph against the current vertices
  and rebuilds on mismatch; retraining with the fix is bit-equivalent on
  the training path (identical query sequence, same loss EMA).
- **conclusion** Full rigid + uniform-scale invariance holds exactly.
  Lesson: a cached face graph is only valid for the vertices it was built
  from.
- **commit** cd9f897 (finding), 0121e34 (root cause + fix)

## Partial query

- **date** 2026-07-28
- **goal** Can a connected subset of the surface be queried alone?
- **dataset** BFS-connected face patches (~25% of faces), masked address
  maps; evaluation region-restricted.
- **model** the 36.9M query-attention pipeline (unchanged)
- **config** zero-shot on checkpoints trained with full queries only.
- **result** Masking exact (10/10 meshes: zero output outside the region);
  in-region 12.5 dB vs 18.6 dB for the full query on the same region —
  a distribution-shift gap, not a masking failure.
- **conclusion** Partial queries work mechanically; closing the fidelity
  gap needs partial queries in training (next entry).
- **commit** cd9f897

## Query training (partial-aware sampling)

- **date** 2026-07-28/29
- **goal** Does query-aware training close the partial-query gap, and at
  what cost?
- **dataset** 100 meshes. Two regimes tested: (a) heavy resampled partial
  curricula (60–75% of steps on random face subsets), (b) the final
  weighted sampling — canonical 0.5 / alternative 0.3 / partial 0.2 with
  one deterministic re-rasterized partial query per mesh in the dataset.
- **model** the 36.9M query-attention pipeline (unchanged)
- **config** identical recipe/budget as the full-query counterpart
  (200k steps, 2000 exposures/mesh); control: canonical-only training.
- **result** Full-query-only baseline: partial-region gap −3.05 dB.
  Heavy curricula: gap reversed (+2.68 / +0.95 dB across two variants —
  replicated) but full-query fidelity paid −3.5 dB. Final weighted
  sampling: gap +2.8 / +1.0 dB on two eval groups with full-query and
  held-out fidelity matching the full-query-only counterpart within the
  noise floor (canonical 18.43/18.98, alternative 17.32/18.27, held-out
  17.09/17.82 dB; render consistency 22.32/19.78 ≥ GT fidelity
  18.70/18.49). Canonical-only control did NOT collapse on unseen layouts
  (16.9–17.3 dB) — multi-UV training buys cross-layout balance, not the
  ability to generalize.
- **conclusion** The cost of partial competence comes from the sampling
  ratio, not from partial queries themselves; 20% deterministic partial
  sampling is effectively free. Adopted into the final recipe. The
  canonical-only result shows UV independence is architectural.
- **commit** 0121e34 (curricula + control), ac38760 (final recipe)


## Training throughput benchmark

- **date** 2026-07-29
- **goal** Establish the scaling levers for larger training runs without
  touching the model: precision, mesh batching, graph packing.
- **dataset** 10 meshes, canonical query, 600 steps per variant (last 540
  measured), group size 4 for the batched variants.
- **model** frozen 36.9M pipeline; execution strategy varies only
  (`scripts/benchmark_training.py`).
- **result** (H800, single GPU; meshes/sec | peak mem | mean power)
  fp32 13.3 | 2.5 GB | 345 W · bf16-on-DiT 14.4 | 2.1 GB | 340 W ·
  face-count-bucketed group DiT batching 15.4 | 6.8 GB | 342 W ·
  packed face graph (tokenizer+topology once per group, per-mesh unit-area
  normalization keeps features numerically identical) 20.6 | 6.8 GB | 339 W.
- **conclusion** Graph packing is the dominant lever (+54% per-mesh
  throughput); the step is launch-bound, not FLOP-bound, so batching the
  graph pathway pays far more than precision. bf16 helps only the DiT
  fraction (+8%, −14% memory). Adopt packed grouping when scaling to 1k.
- **commit** (this commit)

## Texture generator: diffusion vs flow matching

- **date** 2026-07-29
- **goal** Same data, same condition pathway (Surface Conditioner + UV
  Query Attention untouched) — swap only the generator schedule.
- **dataset** 10 meshes, 20k steps, identical seed/recipe; evaluation per
  `docs/experiment_protocol.md`.
- **model** frozen backbone as velocity net; rectified flow (linear path,
  velocity MSE, Euler-50) vs masked cosine diffusion (eps-pred, DDIM-50);
  `models/texture_generator/flow_matching.py`, `train.py --generator fm`.
- **result** Flow matching reconstructs the SEEN distribution far better at
  equal small-scale budget: canonical 30.28 vs 16.65 dB, alternative 27.55
  vs 15.68, partial region 31.41 vs 19.29, render consistency 28.15 vs
  17.74. Held-out family is WORSE: 12.59 vs 14.73 dB (beyond the ±1.5 dB
  noise floor). Masking exact for both.
- **conclusion** At the 10-mesh memorization scale FM converges much
  faster on seen queries but transfers worse to the unseen unwrap family —
  a faster-fitting, possibly more memorizing generator. Promising but NOT
  adopted as baseline; requires a 100+-mesh comparison before any switch.
- **commit** (this commit)


## Flow matching transition (generator decision)

- **date** 2026-07-29
- **goal** Promote the stronger generator to the official baseline after
  the head-to-head comparison; retire the other to reference status.
- **dataset** 10-mesh comparison plus the packed-configuration ladder
  (single-mesh overfit, 10-mesh); 2K-scale run in progress.
- **model** unchanged 36.9M pipeline; rectified flow as the official
  schedule, masked diffusion retained at the time only to load the
  previous reference checkpoint (both removed in the final consolidation;
  the schedule code lives in git history at this commit).
- **result** Under the official packed/bf16 configuration: single-mesh
  overfit canonical 48.97 dB (diffusion ~39); 10-mesh canonical 21.27 /
  alternative 20.65 / partial 21.86 / held-out 14.25 dB with render
  consistency 22.82 ≥ GT fidelity 22.52 (decoupling signature holds).
  Packed K=4 grouping (4× effective batch, 1/4 optimizer steps) trades
  ~9 dB of seen-query memorization against +1.7 dB held-out transfer
  versus single-mesh FM — the right direction for scaling.
- **conclusion** Flow matching is the primary texture generator
  (config default `generator: fm`); the 100-mesh and 2K-scale runs extend
  this baseline. DiT/diffusion is reference-only.
- **commit** (this commit)


## FM 2K scalable baseline

- **date** 2026-07-29
- **goal** Scale the FM baseline to the 2K dataset; validate Z_F
  representation scaling and the multi-GPU training stack end to end.
- **dataset** 1982 meshes (99.35% build success, zero duplicates,
  full-query coverage 1.0), canonical/alternative/partial + held-out
  Smart-UV; manifest sha recorded in checkpoint and record.json.
- **model** unchanged 36.9M pipeline; rectified flow; official efficiency
  config (packed K=4, bf16) on 8-GPU DDP (256 mesh-exposures/step; PE
  cached per group; synchronized size-class sampling).
- **result** 272 min for 2000 exposures/mesh (243 mesh-exp/s; the two
  throughput fixes took 8-GPU DDP from 21 to 243). Stage-A gate at 6.5%
  budget: CONTINUE (all failures were budget underfit; no structural
  defects). Final, two disjoint 32-mesh groups: canonical 25.00/25.52,
  alternative 24.91/25.67, partial 25.42/25.77, held-out 21.56/21.63 dB;
  render consistency 25.98/26.72 (GT fidelity 26.11/27.33); seam
  0.10-0.11 vs GT floor 0.06 (1.7-1.8x, down from 4.4x at Stage-A).
- **conclusion** Z_F representation scaling holds: 100 -> 1982 meshes with
  zero architecture change lifts held-out transfer by +1.2-3.1 dB with
  tight cross-group agreement. Decoupling signature preserved. Remaining
  gap for 10K: unseen-mesh validation split (current held-out axis is the
  UV family, not mesh identity).
- **commit** (this commit)


## Model scaling study (controlled, one variable at a time)

- **date** 2026-07-29
- **goal** Locate the capacity bottleneck before the 10K scale-up — judged
  on all four axes (canonical, held-out UV, render consistency, seam),
  not reconstruction alone.
- **dataset** 2K set, 100 meshes, 2000 exposures/mesh, official DDP
  configuration; evaluation: 16 meshes, standard protocol.
- **model** config knobs only (architecture untouched): conditioner token
  dim 256->384->512; topology depth 4->6; generator hidden 384->512
  (heads 8 so head_dim stays 64).
- **result** (canonical / held-out / consistency / seam)
  base 36.9M: 23.62 / 20.39 / 23.95 / 0.11 ·
  dim384 51.8M: 24.88 / 20.50 / 24.90 / 0.10 ·
  dim512 72.2M: 24.87 / 20.60 / 24.69 / 0.10 ·
  depth6 38.5M: 23.55 / 20.49 / 24.04 / 0.11 ·
  fm512 54.4M: 24.70 / 20.04 / 24.56 / 0.10.
- **conclusion** Token dimension is the only effective knob and saturates
  at 384 (+1.3 dB canonical, +1 dB consistency; 512 adds nothing for 2x
  base params). Topology depth 6 and generator hidden 512 are flat-to-
  negative. Held-out transfer is FLAT across every variant (20.0-20.6) —
  generalization is driven by data scale, not capacity, corroborating the
  100->2K result. Candidate for 10K: dim384; decide after the 10K data
  run confirms the compute budget.
- **commit** (this commit)

---

## Reference baseline (masked diffusion, retired)

- **date** 2026-07-29
- **method** Image + Mesh → six generated canonical views → Surface
  Conditioner (face tokenizer + face-view cross attention + topology
  transformer) → Face Set latent `Z_F` → Global UV Query Attention (any
  UV layout as query) → texture DiT (masked cosine diffusion, DDIM-50).
  36.9M parameters total.
- **dataset** 265 meshes × (canonical / alternative-xatlas / partial
  surface query) + held-out Smart-UV family, 256².
- **training** weighted query sampling 0.5/0.3/0.2, 2000 exposures/mesh.
- **numbers** see `docs/technical_report.md`; reference checkpoint removed
  in the final consolidation (flow matching is the only generator).
- **commit** ac38760, then the consolidation commit (this one).

---

## Final consolidation (paper mainline freeze)

- **date** 2026-07-30
- **decision** dim384 is the official 10K configuration
  (`configs/topotex_fm_10k.yaml`): +1.3 dB reconstruction and better
  consistency at 51.8M params, dim512 saturated, depth6/fm512 flat,
  held-out unchanged everywhere. No further architecture search before
  the 10K run completes. Scaling-study record moved to
  `experiments/fm_10k/scaling/`.
- **cleanup** Single mainline kept: Image + Mesh → Face Set Latent `Z_F` →
  UV Query → Flow Matching → Texture. Masked-diffusion generator deleted
  (schedule + retired reference checkpoint; code recoverable at the
  commit above); FM schedule invariants covered by
  `tests/test_flow_matching.py`. Historical run outputs (`runs/`,
  `reports/`, intermediate checkpoints, the superseded 1k UV-query base)
  removed — every number that matters is in this log and
  `experiments/*/record.json`. Kept checkpoints: `checkpoints/baseline`
  (fm_2k official) and `checkpoints/fm_100_reference`.
- **incident** Deleting the superseded 1k UV-query base broke the 265 oldest
  2K-dataset samples whose `uv_000/uv_001/uv_test` were directory-level
  symlinks into it. Rebuilt with the frozen builder from the intact
  source: canonical / alternative / held-out reproduce the original
  values exactly (byte-identical on the retained reference copy);
  `uv_002` changed on 115 multi-component meshes because the original
  copies predate the connected-subset reseed fix (they kept fewer faces
  than the target fraction). The dataset now matches the current frozen
  builder — the same code the 10K build uses; partial-axis metrics on
  those meshes are therefore not comparable to pre-rebuild numbers.
- **status** current pipeline + next steps in `docs/current_status.md`.

## Factorized dense UV query encoder (promoted without A/B)

- **date** 2026-07-30
- **decision** The factorized dense UV query encoder is the ONLY official
  implementation, promoted directly by project decision — the planned
  controlled A/B against the concat-bottleneck encoder was cancelled
  before any GPU time was spent on it; no numbers exist and none should
  be quoted. Rationale: the fixed 32-dim texel bottleneck is structurally
  too narrow for the dim384 mainline (Dq scales as cond_dim/4 = 96), and
  the 10K run is the priority.
- **architecture** face_address = Linear(D, Dq) applied once to `Z_F`
  then gathered per texel; bary_address = MLP(27, Dq, Dq) over the
  Fourier-encoded barycentrics (valid texels only);
  texel = LayerNorm(face + bary); learned background embedding (never
  indexes `Z_F`); Conv2d(Dq, D, 8, 8) patch embedding; Global UV Query
  Attention (depth 4, heads 8, K=V=`Z_F`) and the output head unchanged.
  dim384 decoder 11.55M params (was 9.95M).
- **compatibility** Concat-bottleneck checkpoints (fm_100/fm_2k
  references, scaling-study runs) do NOT load into the new architecture —
  fail-fast, no migration, no silent partial load. Their numbers remain
  valid history in this log; the 10K run trains from random init.
- **verification** contract tests (shape Dq 64/96, face-permutation
  equivariance, barycentric sensitivity + gradient flow, background
  isolation, partial-mask zeros, determinism), golden upstream
  equivalence (face features bitwise, `Z_F` <= 2.5e-4), packed/DDP/
  resume/fail-fast smokes — all green pre-promotion.

## Data-loss incident and storage isolation (2026-07-30)

- **incident** An interrupted `git merge --ff-only` checkout (13:37:15 to
  13:39:14) deleted the gitignored real data directories while writing
  tracked symlinks from the refactor branch: `checkpoints/` (fm_2k
  baseline, fm_100 reference, golden capture — unrecoverable outside
  filesystem snapshots), `dataset/TexVerse_data` (~255 GB),
  `output/` source layer (9,958 samples) + dataset metadata + 1,071 of
  9,919 query samples. Root cause: tracked symlinks at
  directory-pattern-ignored paths; git treats ignored paths as
  expendable. Full forensics + inventories in the rescue set
  (`topotex_rescue_2026-07-30`, four SHA-verified copies).
- **rescued** 8,889 query-sample dirs (uv_queries + meta), the frozen
  10,032-id → GLB manifest, the 9,419/500 split (seed 20260727), golden
  reference tensors with per-tensor SHA256. Deterministic rebuild plan
  recorded; rebuild target is a protected root outside the repository.
- **hardening** `topotex/paths.py` env-var data roots
  (TOPOTEX_*_ROOT), permanent hygiene tests (zero tracked symlinks,
  no reserved data names), and `scripts/check_git_tree_safety.py`
  preflight required before any merge/checkout into canonical.
  Validation reran green on a synthetic dataset (loader/train/resume/
  DDP/sample/evaluate through env roots).

