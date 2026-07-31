# TOPOTEX Data Layout (frozen 4.6K object-level baseline)

Protected root (`$TOPOTEX_DATA`, outside the repository; access via
`TOPOTEX_SOURCE_ROOT` / `TOPOTEX_DATASET_ROOT` / `TOPOTEX_CHECKPOINT_ROOT`
/ `TOPOTEX_RUN_ROOT` — see `topotex/paths.py`):

```
topotex_data/
├── raw_glbs/                    10,021 TexVerse-1K GLBs (69 GB, rebuild raw material)
├── source/                      4,602 source samples (frozen; SHA in source_manifest_current.sha256)
│   └── samples/<id>/{mesh.safetensors, mv.safetensors, reference.png, gt_texture.png, meta.json}
├── dataset/                     4,590 ready objects — THE dataset
│   ├── manifest.jsonl           (== ../dataset_manifest.jsonl, sha a6aef671…)
│   └── samples/<id>/
│       ├── mesh.safetensors ─┐
│       ├── mv.safetensors    ├─ HARDLINKS to source (zero duplicated payload)
│       ├── reference.png    ─┘
│       ├── meta.json            provenance: builder commit, per-query hashes, texture sha
│       └── uv_queries/{uv_000,uv_001,uv_002,uv_test}/
│           ├── uv_address.safetensors
│           └── gt_texture.png
├── object_split.json            train 4,090 / unseen test 500 (seed 20260727, sha 21972f1a…)
├── dataset_statistics.json · source_resolution_audit.json · dataset_recovery_report.md
└── runs/                        all run dirs, acceptance artifacts, notebook HTML
```

## Query semantics (directory names are frozen; the mapping is the contract)

| dir | layout | mask | role |
|---|---|---|---|
| `uv_000` | **native** | full | GLB 原生参数化 |
| `uv_001` | **xatlas** | full | 重参数化 |
| `uv_test` | **blender_smart** | full | Smart-UV（训练增强的一员） |
| `uv_002` | **partial** (native) | connected_partial | **surface mask/query — 不是 unwrap family** |

Split unit = source object (GLB asset, content-hash verified unique);
every query of an object inherits its split. Training sampling: full
0.80 uniform over the three layouts + partial 0.20. Generalization is
evaluated on the 500 unseen objects.

## Field dtypes (audited 2026-07-31, storage_audit.json)

| field | dtype / format |
|---|---|
| images (reference/GT/MV) | PNG · uint8 (MV tensor uint8 [6,3,256,256]) |
| tensors | safetensors |
| barycentric | float16 [3,256,256] (valid sum≈1) |
| face_id / faces | int32 (loader casts to int64; −1 = background) |
| valid_mask | uint8 (== face_id ≥ 0) |
| mesh.vertices | float32 [V,3] |

## Resolutions

reference 512² · MV 256² (UniTEX native 512) · native GT 256² ·
**UV query = model = 256²**. Higher-resolution queries must be
re-rasterized from `uv_vertices`/`uv_faces` and re-baked from raw GLB
textures — never resized from 256 maps.

## Measured IO (JuiceFS, 2026-07-31)

~30.5 GB / ~83k files total; full-object random read 14 ms mean
(p95 32 ms); sequential 548 MB/s; smoke100 dataloader wait ≈ 0 ms →
**no physical migration before full training.**
