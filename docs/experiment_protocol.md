# Experiment Protocol

The fixed evaluation protocol behind every number in
`experiments/experiment_log.md` and `docs/technical_report.md`.
Implementation: `evaluate.py`.

## Sampling

- DDIM, 50 steps, eta 0
- generation seed 20260727 (one fixed generator seed per query)
- one shared Face Set latent `Z_F` per mesh — every query of a mesh is
  decoded from the same encoding pass

## Metrics

| metric | definition |
|---|---|
| UV PSNR | masked PSNR between prediction and baked GT inside the query's `valid_mask`, per query kind (canonical / alternative / held-out) |
| partial region PSNR | UV PSNR of the partial query inside its own valid region |
| partial gap | partial region PSNR − canonical prediction's PSNR restricted to the same region (positive = partial query at least as good) |
| masking exactness | prediction must be exactly zero outside the query's valid region |
| render consistency | `R(M, U0, T0)` vs `R(M, U1, T1)`: both generated textures rebaked through their own layouts, rendered from the six canonical views, PSNR on intersected masks, averaged |
| GT render fidelity | same render comparison between a prediction and its GT texture |
| decoupling signature | render consistency ≥ GT render fidelity — predictions agree with each other at least as well as each matches GT, i.e. they share one surface signal |
| seam consistency | mean RGB difference across UV seams: every shared mesh edge is sampled on both of its UV images (same 3D points, slight interior inset, dilated texture); only edges whose UV images are > 1 texel apart count as seams. Reported with the GT texture's seam error as the baking-floor reference. Native per-vertex layouts have no shared-edge seams by construction — the metric targets re-parameterized layouts (alternative query) |

## Rendering

Six canonical views (front/back/left/right/top/bottom), 384², adaptive
near/far planes from the mesh bounds; textures are rebaked via each query's
own `uv_vertices/uv_faces` with edge dilation before sampling.

## Statistical caution

Identical-seed training reruns (cuDNN nondeterminism) move sampled metrics
by about **±1.5 dB** at the 10-mesh evaluation scale. Differences below
that are noise, not conclusions. Where a claim matters, it is checked on
two disjoint 10-mesh groups.

## Held-out discipline

`uv_test` (Blender Smart UV) is never sampled during training — the loader
separates it structurally (`test_uv_queries`), not by convention.

## Protocol breakpoint (2026-07-30, partial axis only)

The final consolidation rebuilt the UV queries of 265 samples with the
frozen builder; their partial (`uv_002`) queries previously predated the
connected-subset reseed fix and kept fewer faces than the target fraction
on multi-component meshes (115 samples). The official dataset is now
uniformly consistent with the frozen builder — every partial query reaches
its target fraction.

Consequences for comparisons against numbers recorded before 2026-07-30
(fm_100 / fm_2k records and older log entries):

| axis | comparable? |
|---|---|
| canonical UV PSNR | ✔ (queries reproduce exactly) |
| alternative UV PSNR | ✔ (queries reproduce exactly) |
| held-out UV PSNR | ✔ (queries reproduce exactly) |
| full-query render consistency / GT fidelity | ✔ |
| seam consistency | ✔ |
| partial region PSNR / partial gap | ✘ — measured against pre-fix partial data; treat pre/post numbers as different protocols |

All numbers produced from 2026-07-30 onward (including every fm_10k axis)
use the current data and are mutually comparable.

## Data integrity gate

`datasets/verify_integrity.py` runs after source-build finalize and after
UV-query-build finalize (and on demand): recursive dangling-symlink scan,
required-file readability, manifest completeness (every manifest id must be a complete readable sample — FAIL; sample dirs not in the manifest are reported as warnings, since the loader reads the manifest only), builder provenance
(`query_schema_version` / `query_builder_commit`), and `--deep` content
hashes (`face_id`, `barycentric`, source texture) against `meta.json`.
A training or evaluation run must not start on a dataset whose gate fails.
