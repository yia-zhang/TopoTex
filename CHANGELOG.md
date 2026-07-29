# Changelog

## v0.1.0-fm-baseline — 2026-07-29

Initial public release of the TOPOTEX research codebase.

- **Flow matching is the official texture generator.** Rectified flow
  (velocity prediction on the frozen patchified-transformer backbone,
  50-step Euler sampling) replaces masked diffusion as the default
  (`generator: fm` in `configs/topotex_fm_baseline.yaml`).
- **Diffusion retired to reference status.** The masked-diffusion schedule
  remains loadable only for the previous reference checkpoint
  (`checkpoints/dit_reference`, untracked); no further work on it.
- **Clean repository release.** Single-commit `main` containing only the
  final method: dataset pipeline (8-GPU sharded build, atomic publish,
  resume, manifest-merge validation), frozen training recipe (packed
  face-graph batching K=4, bf16 velocity net, face-count bucket sampler),
  evaluation protocol (UV PSNR, render consistency, held-out UV family,
  UV seam consistency), three maintained notebooks, docs, 47-test suite.
  Historical exploration lives outside this repository; conclusions are
  summarized in `experiments/experiment_log.md`.
