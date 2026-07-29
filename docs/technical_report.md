# TOPOTEX Technical Report

Method summary and frozen baseline numbers. Interactive version with live
forwards and visualizations: `notebooks/TOPOTEX_Technical_Report.ipynb`.
Full experimental provenance: `experiments/experiment_log.md`.

## Method

TOPOTEX generates UV textures for artist meshes from a reference image.
The core claim: the right conditioning variable is a **Face Set latent
`Z_F`** — one token per triangle — not a function of world coordinates
(ambiguous under coincident geometry) or of one fixed UV atlas (tied to a
single parameterization).

1. **Inputs.** A textured GLB is split into (reference render, textureless
   mesh, ground-truth texture). Six canonical views
   (front/back/left/right/top/bottom) are generated offline by the frozen
   UniTEX stage-1 generator and cached.
2. **Face Set encoding.** Per-face intrinsic geometry (edge-length ratios,
   angles, normalized area — no xyz/UV/index) + random-walk topology PE →
   face tokens; cross attention injects the six views; a sparse topology
   transformer along the shared-edge adjacency yields `Z_F [F, 256]`.
3. **UV as a query.** Any UV layout is rasterized into per-texel
   `(face_id, barycentric)` address maps. Global UV Query Attention turns
   the layout into 1024 patch query tokens that read `Z_F` and emit a UV
   condition map `[64, 256, 256]`.
4. **Texture generation.** Rectified flow: the patchified transformer
   predicts a velocity field over `x_tau = (1-tau)·x0 + tau·eps` (noise and
   loss only inside the valid mask); a 50-step Euler ODE integrates from
   noise to texture. The masked-diffusion schedule is retained only as the
   previous reference generator.

## Dataset

265 meshes (TexVerse, gated: triangle mesh / single base-color texture /
clean per-vertex UV). Per mesh four queries: canonical (native UV),
alternative (xatlas re-unwrap; face order preserved — asserted),
partial (deterministic connected face subset at 25/50/75%, re-rasterized,
GT re-baked), held-out (Blender Smart UV, evaluation only). All GT
supervision is baked through `(face, bary) → native UV → sRGB bilinear`, so
every query supervises the same surface signal.

## Reference numbers

**Previous diffusion reference** (`checkpoints/dit_reference`) — training:
`query_probs = 0.5/0.3/0.2`, 2000 exposures/mesh, 36.9M params.
Evaluation protocol: DDIM-50, seed 20260727, one shared `Z_F` per mesh, six
canonical render views. Numbers from `checkpoints/dit_reference/eval.json`
(first 10 training meshes; a disjoint 10-mesh group reproduces them within
the ±1.5 dB run-to-run noise floor):

| metric | value |
|---|---|
| canonical UV PSNR | 18.43 dB |
| alternative UV PSNR | 17.32 dB |
| partial region PSNR | 21.68 dB (gap vs canonical-same-region: **+2.80 dB**) |
| held-out family UV PSNR | 17.09 dB |
| render consistency R(M,U0,T0) vs R(M,U1,T1) | 22.32 dB |
| GT render fidelity (U1) | 18.70 dB |
| partial masking exact | 10/10 meshes |

Key properties established along the way (provenance in the experiment
log):

- **Layout independence.** A never-trained unwrap family evaluates with
  zero gap; a canonical-only-trained control does not collapse on unseen
  layouts — UV independence is architectural (face-addressed queries),
  query mixing buys cross-layout balance.
- **Exact invariance.** Rotation/translation/uniform-scale leave `Z_F`
  bitwise-stable (cos = 1.0, scales 0.1×–10×) once the face graph is built
  from the transformed vertices (the encoder now guards this).
- **Partial queries are free at 20% sampling.** The partial-region gap
  flips from −3.05 dB (full-query-only training) to +2.8 dB with no loss
  on full-query or held-out fidelity; heavier partial sampling (60–75%)
  pays ~3.5 dB on full queries — the cost comes from the sampling ratio,
  not from partial queries themselves.
- **Decoupling signature.** Textures generated through different UV
  layouts agree with each other at least as well as each matches GT —
  they share one surface signal read from `Z_F`.

## Flow matching baseline (official, scaling in progress)

Same pipeline, rectified-flow generator, official efficiency configuration
(packed face-graph groups K=4, bf16 velocity net). Current ladder
(protocol as above; `checkpoints/baseline` holds the 1982-mesh
checkpoint — the official FM 2K baseline):

| stage | canonical | alternative | partial | held-out | render consistency |
|---|---|---|---|---|---|
| single-mesh overfit | 48.97 | 47.18 | 48.23 | — | 39.77 |
| 10-mesh | 21.27 | 20.65 | 21.86 | 14.25 | 22.82 (GT 22.52) |
| 100-mesh (8-GPU DDP) | 23.62 | 23.17 | 24.73 | 20.39 | 23.95 (GT 24.57) |
| **1982-mesh (official)** | **25.00/25.52** | **24.91/25.67** | **25.42/25.77** | **21.56/21.63** | **25.98/26.72** (GT 26.11/27.33) |

The 1982-mesh row reports two disjoint 32-mesh evaluation groups; seam
consistency is 0.10-0.11 vs the 0.06 GT baking floor. Z_F representation
scaling holds — held-out transfer improves monotonically with dataset
size at zero architecture change.

Packed grouping trades seen-query memorization for held-out transfer
(+1.7 dB vs single-mesh FM) — see the experiment log. The 100-mesh and
2K-scale runs extend this table; dataset statistics and stage gates are
recorded alongside the training runs.
