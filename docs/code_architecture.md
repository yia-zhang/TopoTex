# TOPOTEX Code Architecture

One installable package (`topotex/`), thin root CLIs, strict layer
boundaries. Produced by the behavior-preserving architecture refactor —
model mathematics, tensor shapes, the dataset schema, checkpoint
state-dict keys, the query sampling recipe, the flow-matching schedule and
the evaluation protocol are all unchanged (verified against a frozen
golden baseline, see `tests/test_golden_equivalence.py`).

## Package dependency diagram

```mermaid
flowchart TD
    CLI["root CLIs<br/>train.py · sample.py · evaluate.py"]
    PIPE["topotex.pipelines<br/>training · inference · evaluation"]
    MODELS["topotex.models<br/>face_tokenizer · image_encoder ·<br/>uv_query · surface_conditioner ·<br/>flow_matching · topotex"]
    LAYERS["topotex.layers<br/>attention · embeddings ·<br/>topology · flow"]
    DATA["topotex.data<br/>dataset · mesh · uv · multiview ·<br/>builder · integrity · diversity"]
    SCHEMA[("topotex.data.schema<br/>typed contracts (leaf)")]
    CFG[("topotex.config<br/>typed configs")]
    UTILS["topotex.utils<br/>distributed · io · logging"]

    CLI --> PIPE
    PIPE --> MODELS
    PIPE --> DATA
    PIPE --> CFG
    PIPE --> UTILS
    MODELS --> LAYERS
    MODELS --> CFG
    LAYERS --> SCHEMA
    DATA --> SCHEMA
    MODELS --> SCHEMA
    DATA -.->|"build_face_graph only"| LAYERS
    classDef data fill:#e8f0fe,stroke:#3366cc;
    classDef train fill:#fff3e0,stroke:#e8821a;
    class SCHEMA,CFG,DATA data;
    class PIPE,MODELS,LAYERS train;
```

Enforced boundaries:

- `data` never imports `models` (face-graph building lives in
  `layers.topology`, a pure tensor function).
- `layers` performs no file IO; modules are torch + `schema` only.
- `models` consume/produce tensors and typed contracts; they never read
  environment variables or change the working directory.
- `pipelines` orchestrate data + models + checkpoints + sampling.
- Root CLI files only parse arguments and call pipelines.
- The UniTEX adapter (`topotex.data.multiview`) is used exclusively by the
  offline builder — never at inference.

## Public API (`import topotex`)

| symbol | role |
|---|---|
| `TopoTexPipeline` | `from_checkpoint(path)` → `pipeline(mesh=…, uv_query=…, images=…, num_steps=50, seed=…)` → `TopoTexOutput` |
| `TopoTexModel` | composed module (`conditioner` + `dit` + `schedule`); `from_checkpoint` / `from_config`; staged `encode` / `condition` / `generate` |
| `TopoTexConfig` | typed aggregate of `DataConfig` / `SurfaceConditionerConfig` / `FlowMatchingConfig` / `TrainingConfig`; `from_yaml` validates, `to_dict` round-trips the frozen flat checkpoint layout |
| `TopoTexDataset` | loader returning `TopoTexBatch` items |
| `TopoTexBatch` `MeshBatch` `UVQueryBatch` `FaceGraph` `ConditionerOutput` `TopoTexOutput` | typed tensor contracts (dict-compatible: `obj["key"]` and `obj.get` still work) |
| `build_models` | frozen factory: flat config dict → `(conditioner, dit)` (seeds torch first) |

```python
from topotex import TopoTexDataset, TopoTexPipeline

pipe = TopoTexPipeline.from_checkpoint("checkpoints/baseline")
item = TopoTexDataset(root, [sid], device="cuda:0").items[0]
z = pipe.encode(item.mesh, item.mv_images, item.graph)   # Z_F, reusable
out = pipe(mesh=item.mesh, uv_query=item.uv_queries[0], face_tokens=z)
out.texture   # uint8 [256, 256, 3]
```

`reference_image` alone is not consumable at inference: the model is
conditioned on six canonical views; generating views from one reference is
the offline UniTEX stage (`topotex.data.multiview`).

## Module responsibilities

| module | responsibility | was |
|---|---|---|
| `topotex/config.py` | typed configs; YAML → dataclasses; frozen flat round-trip | inline `yaml.safe_load` in train.py |
| `topotex/data/schema.py` | schema constants + all typed contracts | scattered dicts |
| `topotex/data/dataset.py` | `TopoTexDataset` loader | `datasets/dataset.py` |
| `topotex/data/mesh.py` | cameras, rasterized views, rebake render, seam metric | `datasets/mesh_utils.py` |
| `topotex/data/uv.py` | deterministic UV address rasterizer + connected face subsets | `datasets/rasterizer.py` + `datasets/uv_query.py` |
| `topotex/data/multiview.py` | frozen UniTEX stage-1 adapter (offline only) | `datasets/mv_generator.py` |
| `topotex/data/builder.py` | offline construction CLI: `source` / `queries` / `merge` subcommands | `datasets/build_dataset.py` + `build_uv_queries.py` + `merge_manifest.py` |
| `topotex/data/integrity.py` | recursive post-finalize integrity gate | `datasets/verify_integrity.py` |
| `topotex/data/diversity.py` `statistics.py` | dataset monitors | `datasets/dataset_diversity.py` / `dataset_statistics.py` |
| `topotex/layers/attention.py` | face–image cross attention | `face_image_attention.py` |
| `topotex/layers/topology.py` | `build_face_graph` + random-walk PE + sparse topology transformer | `face_tokenizer.py`(part) + `topology_pe.py` + `topology_transformer.py` |
| `topotex/layers/embeddings.py` | sincos pos embed, patchify/unpatchify, timestep embed, bary encoding | parts of `dit.py` / `uv_query_attention.py` |
| `topotex/layers/flow.py` | rectified-flow schedule (`MaskedFlowMatching`) | `texture_generator/flow_matching.py` |
| `topotex/models/face_tokenizer.py` | intrinsic features → face tokens | `face_tokenizer.py` |
| `topotex/models/image_encoder.py` | multi-view ViT encoder | `image_encoder.py` |
| `topotex/models/uv_query.py` | Global UV Query Attention decoder | `uv_query_attention.py` |
| `topotex/models/surface_conditioner.py` | Z_F encoder + decoder composition | `conditioner.py` |
| `topotex/models/flow_matching.py` | MiniDiT velocity network | `texture_generator/dit.py` |
| `topotex/models/topotex.py` | `TopoTexModel` + `build_models` factory | `train.py::build_models` |
| `topotex/pipelines/training.py` | frozen DDP training recipe (packed groups, bucket sampler, resume) | `train.py` body |
| `topotex/pipelines/inference.py` | `TopoTexPipeline` + sampling CLI | `sample.py` body |
| `topotex/pipelines/evaluation.py` | closed-loop evaluation protocol | `evaluate.py` body |
| `topotex/utils/*` | `ddp_env`, sha/json IO, structured logging | inline helpers |

## Tensor contracts

Documented on each dataclass in `topotex/data/schema.py` (F faces,
V vertices, E adjacency edges, H = W = 256, Nv = 6 views):

- `MeshBatch`: `vertices f32 [V,3]`, `faces i64 [F,3]`
- `FaceGraph`: `edges i64 [E,2]`, `rel f32 [E,3]`, `boundary f32 [F]`,
  `global_scale f32` (valid only for the vertices it was built from)
- `UVQueryBatch`: `face_id i64 [H,W]` (−1 = background), `barycentric
  f32 [3,H,W]`, `valid_mask bool [H,W]`, `gt_texture f32 [3,H,W]`,
  full-layout `uv_vertices/uv_faces` (numpy, rebake-safe)
- `ConditionerOutput`: `face_tokens f32 [F,D]` (Z_F), `uv_condition
  f32 [1,C,H,W]`, optional `uv_rgb f32 [1,3,H,W]`
- `TopoTexOutput`: `texture u8 [H,W,3]`, `uv_condition`, `face_tokens`

All contracts remain dict-compatible (`__getitem__` / `__setitem__` /
`get`), so the frozen training internals read identically.

## CLI → pipeline call graph

```mermaid
flowchart LR
    T(["python train.py"]) --> TP["pipelines.training.main<br/>DDP · packed groups · resume"]
    S(["python sample.py"]) --> IP["pipelines.inference.main"]
    E(["python evaluate.py"]) --> EP["pipelines.evaluation.main<br/>protocol metrics"]
    B(["python -m topotex.data.builder<br/>source | queries | merge"]) --> BD["data.builder"]
    IP --> PL["TopoTexPipeline"]
    EP --> PL
    TP --> BM["models.topotex.build_models"]
    PL --> TM["TopoTexModel<br/>encode → condition → generate"]
    classDef train fill:#fff3e0,stroke:#e8821a;
    class TP,IP,EP,BD,PL,TM,BM train;
```

`scripts/*.sh` wrap the same entry points (torchrun for
`train_8gpu.sh`, rank-sharded `evaluate_8gpu.sh`, two-stage
`build_dataset_8gpu.sh` → `topotex.data.builder`).

## Numerical-equivalence policy

`checkpoints/golden/` holds pre-refactor captures (face features, Z_F,
query tokens, dense uv_condition, FM loss at fixed t, Euler-50 texture,
1-mesh evaluation metrics) for the frozen baseline checkpoint and a fixed
sample/seed. `tests/test_golden_equivalence.py` re-runs them through the
public API: bitwise for the tokenizer; 10× the measured same-code
run-to-run CUDA-nondeterminism envelope elsewhere (the topology
transformer's scatter kernels are nondeterministic — measured deltas:
Z_F ≤ 2.4e-5, uv_condition ≤ 3.1e-6, texture ≥ 118 dB run-to-run).
