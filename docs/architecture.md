# TOPOTEX Architecture

One frozen pipeline (36.9M parameters):

```
Reference Image + Mesh
    → six generated canonical views          (offline, frozen UniTEX stage-1)
    → Surface Conditioner                    (models/surface_conditioner/)
        FaceTokenizer  →  Face-View Cross Attention  →  Topology Transformer
    → Face Set Latent  Z_F  [F, 256]
    → Global UV Query Attention              (any UV layout as query)
    → UV condition [64, 256, 256]
    → Texture Generator                      (models/texture_generator/)
        MiniDiT under masked cosine diffusion, DDIM-50
    → UV texture [3, 256, 256]
```

## Design principle

The mesh essence is the **Face Set**. XYZ is a geometric observation
(changed by rigid transforms); a UV atlas is one replaceable 2D addressing
scheme. `(face_id, barycentric)` is the stable identity of a surface point,
so the latent is indexed by faces and every consumer addresses it through
face/bary queries.

## Surface Conditioner

### FaceTokenizer (`face_tokenizer.py`)
Per-face **intrinsic** features only — no world coordinates, no UV, no face
index:
- 3 edge lengths / mesh scale (`sqrt(total area)`)
- 3 edge lengths / perimeter
- 3 corner angles / π (sorted within face → corner-order invariant)
- log-normalized face area, boundary-edge fraction

Concatenated with a random-walk topology PE (k=16, pure graph structure)
and passed through an MLP → face tokens `[F, 256]`. Ratio-based features
make the tokens exactly invariant to rotation, translation, and uniform
scale, and winding-agnostic.

The face graph (shared-edge adjacency, relation features = edge length /
unoriented dihedral / boundary flag) is built by `build_face_graph`. A
cached graph is only valid for the vertices it was built from;
`SurfaceConditioner.encode_faces` validates the cached graph's global scale
against the current vertices and rebuilds on mismatch.

### Multi-view Image Encoder (`image_encoder.py`)
From scratch (no pretrained backbone): conv patch embedding (patch 16) +
2D position embedding + learnable per-view embedding, joint self-attention
over all view tokens → image tokens `[6·256, 256]`.

### Face–View Cross Attention (`face_image_attention.py`)
Q = face tokens, K/V = image tokens: appearance injection into the face
set.

### Topology Transformer (`topology_transformer.py`)
Sparse graph attention along the shared-edge face adjacency only (no
euclidean KNN), with a learned relation bias from (shared edge length,
dihedral, boundary). Output = the canonical Face Set latent `Z_F [F, 256]`.

### Global UV Query Attention (`uv_query_attention.py`)
```
per-texel:  [ face_token(face_id) ‖ bary_encoding(barycentric) ] → texel feature (32)
patchify:   256×256, patch 8 → 32×32 = 1024 query tokens (+ learned atlas pos emb)
4 × CrossBlock:  Q = UV query tokens,  K,V = Z_F   (global, no routing, no masks)
unpatchify: uv_condition [64, 256, 256]  (background zeroed)  + optional RGB head
```
The UV layout enters only through which faces each patch references;
surface content flows exclusively through attention over `Z_F`. This is
what makes the parameterization swappable at inference time — including
partial (face-subset) queries.

## Texture Generator

`MiniDiT` (`dit.py`): pixel-space transformer at 256², patch 8,
AdaLN-Zero, serving as the velocity network; conv embeddings for noisy
image / condition / valid mask, sin-cos 2D positions. `MaskedFlowMatching`
(`flow_matching.py`): rectified flow — linear interpolation path
`x_tau = (1-tau)·x0 + tau·eps`, velocity-MSE loss restricted to the valid
mask (invalid region fixed at 0 through the whole trajectory), 50-step
Euler ODE sampling.

## Training recipe (frozen, `configs/topotex_fm_baseline.yaml`)

One mesh per step; one stored query per step drawn with
`query_probs = [0.5, 0.3, 0.2]` (canonical / alternative / partial);
noise batch 4 with high-t emphasis (half the batch from t ∈ [700, 1000]);
auxiliary RGB L1 (0.1); AdamW lr 3e-4, warmup 100, cosine to 10%;
2000 exposures per mesh. Checkpoints carry model/optimizer/RNG state and
resume exactly.
