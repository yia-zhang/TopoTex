# Stability Report — `fm_baseline_dim384_factorized` (4.6K run)

**Status: terminated at step 82,400 / 255,500 after a second, recurrent
divergence** (user-preauthorized branch: save evidence, stop, move GPUs to
the 10K dataset build). This run's purpose was to validate the
dim384 + Factorized-Query + FM training recipe; its conclusion is the
frozen recipe below, with each item tagged **[verified]** or
**[untested — inferred from the failure evidence]**.

## Run identity

| item | value |
|---|---|
| run | `fm_baseline_dim384_factorized`, 8×H800, BF16, packed K=4 (32 meshes/step) |
| train set | 4,088 feasible objects (`train_ids_feasible.json` sha `a11adb2c…`) |
| config | `configs/topotex_fm_10k.yaml` (sha `06f62875…`): dim384/Dq96, AdamW(0.9, 0.95) wd=0, cosine → 0.1×, warmup 100, grad-clip 1.0, `ckpt_every` 2000 |
| code | launch `d726ec0` (+ RWPE fix `8150ea1`); recovery guard + `--lr` override `5136027` |
| budget | 255,500 steps = 2,000 exposures/mesh; sustained ~4.4–4.5 steps/s |

## Divergence timeline (two episodes)

**Episode 1 — lr 3.0e-4 cosine.** Healthy from 0 to **83,800** (loss EMA
0.068–0.099). Diverged at **~84,000** (effective lr ≈ 2.3e-4): EMA → 1.4.
Partial self-recovery around 116k (EMA 0.099), re-diverged at 124k, stuck
at EMA ≈ 1.2 through 132k. Grad clipping (1.0) was active throughout.
Evidence: `metrics_diverged_84k.jsonl`, `ckpt_diverged_step132k.pt`.

**Recovery attempt (commit `5136027`).** Spike-rejection guard (post-warmup,
skip `opt.step()` when `loss > max(4×EMA, 0.5)`) + `--lr` CLI override.
Resumed from the healthy step-64,000 checkpoint at lr 1.5e-4.

**Episode 2 — lr 1.5e-4 cosine (effective ≈ 1.2e-4 at failure).** Healthy
from 64,000 to **79,600** (EMA 0.071–0.080, transient batch losses down to
0.017). Then a **ramp divergence** over ~800 steps: EMA 0.077 (79,600) →
0.42 (80,400) → 1.30 (81,200); no recovery through 82,400. The guard fired
exactly once (step 79,917: loss 0.729 > 4×0.156) — by then the EMA itself
had already doubled. Evidence: `metrics_diverged2_79900.jsonl`,
`ckpt_diverged2_step82k.pt`, `divergence2_log_extract.txt`,
`relaunch_full_log_diverged2.log` (all in the run directory).

## Analysis

1. **Halving lr did not buy stability.** Episode 2 failed *earlier*
   (79.9k vs 84k) at roughly half the effective lr (1.2e-4 vs 2.3e-4).
   Pure lr magnitude is not a sufficient explanation.
2. **The RNG state was restored on resume**, so episode 2 replayed the
   same batch order from 64k that episode 1 survived until 84k. Both
   failures landing in the same 80k–84k window is consistent with either
   (a) a pathological packed-batch region in that data-order window
   interacting with optimizer state, or (b) intrinsic edge-of-stability
   behavior of dim384 + bf16 once the loss gets very low (batch losses
   0.017–0.02 immediately precede both blowups). The evidence cannot
   separate (a) from (b); the 10K run reshuffles data anyway, so only
   the operational protections transfer.
3. **Spike guard v1 is verified insufficient against ramps.** It rejects
   isolated spikes; a run of moderately-elevated losses drags the EMA up
   fast enough that `4×EMA` is never exceeded again.
4. **Single rolling `ckpt.pt` is an operational hazard.** The 2,000-step
   rolling save meant the last clean state (78,000) was overwritten by
   poisoned saves (80k, 82k); the best healthy artifact regressed to the
   64k interim copy that existed only by accident.

## What the run *did* validate **[verified]**

- Architecture + data pipeline train correctly at scale: 8-GPU DDP,
  packed K=4 face-graph groups, blocked-sparse RWPE, 4-way query
  sampling, provenance stamping, preflight gates — 0 data faults,
  0 NaN/Inf from the model itself over ~150k total step-attempts.
- Unseen-object behavior improves with training (fixed 8-object
  validation subset, steps 28k → 60k): cross-layout render consistency
  15.4 → 18.1 dB, seam gen/GT ratio 3.92 → 3.50, per-layout PSNR
  converged across all four query types (~11.7 dB at this scale).
- Throughput recipe: ~4.4–4.5 global steps/s sustained (≈ 32 meshes/step),
  median GPU util 96%, BF16, no loader bottleneck.
- Best healthy artifact: `ckpt_step64000_interim.pt` (step 64,000,
  EMA 0.083, lr-3e-4 lineage; 25% of budget).

## Frozen recipe for `fm_10k_dim384_factorized`

Unchanged (all **[verified]**): architecture (dim384/Dq96, factorized
encoder, depth-4/heads-8 query attention, FM), AdamW(0.9, 0.95) wd=0,
grad-clip 1.0, BF16, packed K=4, 8-rank DDP, warmup 100, query sampling
[0.26667, 0.26667, 0.2, 0.26666], 2,000 exposures/mesh.

Changed (each **[untested]**, motivated strictly by the two episodes):

1. **lr 1.0e-4** (cosine → 0.1×, as before). Both failures occurred at
   effective lr ≥ 1.2e-4; 1.0e-4 keeps the entire schedule below the
   lowest observed failure point. This is an envelope argument, not a
   guarantee.
2. **Ramp-aware guard (guard v2)**: keep the spike rejection, and add a
   divergence halt — if loss EMA exceeds 3× the EMA recorded at the last
   checkpoint save, stop stepping, restore the last checkpoint, and skip
   forward past the offending window with a fresh data-order seed
   (logged), instead of continuing to burn GPU in a poisoned state.
3. **Checkpoint retention: keep the last TWO rolling checkpoints**
   (`ckpt.pt` + `ckpt_prev.pt`), so a ramp that contaminates one save
   interval cannot destroy the last clean state.

Items 2–3 are trainer-level operational protections; they change no
model, loss, data, or schedule semantics. They require user sign-off
before the 10K launch (the ~16h dataset build is the review window).

## Disposition

- GPUs released at 19:56; 10K source build takes over immediately
  (task section 2 discipline).
- 4.6K run checkpoints retained: 64k healthy interim + both divergence
  forensics; 10% milestone previews (28k) retained.
- This run is **not** the official baseline result; the object-level
  baseline claim now rests on the forthcoming `fm_10k_dim384_factorized`.
