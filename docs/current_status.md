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

Flow matching is the **only** generator; the masked-diffusion schedule and
its reference checkpoint were removed in the consolidation (code remains in
git history, numbers in `experiments/experiment_log.md`).

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
- Then: frozen 9500/500 mesh-level split
  (`experiments/protocol/scaling_10k_split.json`), 2K-vs-10K dataset
  diversity report, fm_10k training with evaluation gates at 10% / 50% /
  100% (gallery + unseen-mesh evaluation + query consistency).
- Six required evaluation axes: canonical UV, alternative UV, held-out UV,
  partial query, seam consistency, unseen-mesh validation
  (`experiments/fm_10k/record.json`).

## Entry points

```bash
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_8gpu.sh --run-name <name> ...
python sample.py   --run checkpoints/baseline --n 4
python evaluate.py --run checkpoints/baseline --n 32
bash scripts/build_dataset_8gpu.sh <manifest> output/topotex_source <limit>
```

Notebooks (`Dataset_Inspector` / `Model_Inspector` / `Technical_Report`)
are the maintained dashboards; all three execute clean end to end.
