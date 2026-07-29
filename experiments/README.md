# Experiments

`experiment_log.md` is the chronological record of every experiment that
shaped the method (goal / dataset / model / result / conclusion / commit).

## Artifact naming convention

Every named training run gets a directory here — `fm_100/`, `fm_2k/`,
`fm_10k/`, ... — containing a single `record.json` with the full
reproduction context:

| field | meaning |
|---|---|
| `commit` | git commit the run was launched from |
| `config` | config file + its sha256 at launch |
| `dataset_manifest_sha256` | sha256 of the dataset manifest consumed |
| `metrics` | final evaluation aggregates (protocol: `docs/experiment_protocol.md`) |
| `checkpoint_path` | where the checkpoint lives (checkpoints are never tracked) |
| `status` | running / complete |

Records are updated once when the run finishes; intermediate artifacts
(galleries, profile logs) stay with the run directory and are not tracked.

`protocol/` holds frozen data-protocol provenance (manifests, splits).
