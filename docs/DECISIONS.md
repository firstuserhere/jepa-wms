# Decision log

These decisions describe the current research program. Change one only with an
explicit dated revision, rationale, and new experiment identity.

| Date | Decision | Consequence |
|---|---|---|
| 2026-08-15 | Reproduce the released DINOv3 + DROID JEPA-WM before extending it. | Baseline scientific hyperparameters remain unchanged; operational changes use overlays. |
| 2026-08-15 | Use the native DINOv3 ViT-L/16 artifact and pinned DINO code. | Converted/safetensors substitutes are rejected; code and bytes are checksummed. |
| 2026-08-15 | Treat dataset, source, encoder, W&B, and config identity as checkpoint state. | Exact resume fails closed when mathematical identity changes. |
| 2026-08-15 | Separate exact resume from continued-pretraining fork. | New data/stages reset optimizer/progress/RNG/W&B and record the parent lineage. |
| 2026-08-15 | Use immutable checkpoint objects with atomic aliases. | `latest`, `best_rollout`, and `best_planning` never point to partial files. |
| 2026-08-15 | Retain three recent independent fallbacks in addition to semantic roles. | A bad promotion or recent corruption does not eliminate all recovery points. |
| 2026-08-15 | Make rollout promotion deterministic and planning promotion checksum-bound. | Best checkpoints are comparable and correspond to the exact evaluated object. |
| 2026-08-15 | Use XYZ endpoint error/DROID Action Score for DROID planning. | The dummy environment's always-true success field cannot select checkpoints. |
| 2026-08-15 | Preserve the released eight-rank planning topology during qualification. | Planning is scientifically comparable but leaves 24 of 32 GPUs idle in that phase. |
| 2026-08-19 | Require online W&B for all training and recovery stages. | A missing/unresumable W&B run is a failed training contract, not a warning. |
| 2026-08-19 | Require continuous model-aware effective MFU for the dashboard. | The existing selected-step profiler is diagnostic only and must be extended before full training. |
| 2026-08-19 | Use immutable dated Git tags as multi-agent bases. | Every Cursor agent works in its own worktree/branch from a resolved commit SHA. |
| 2026-08-19 | Treat every Sky mutation and priority as a fresh operator decision. | Earlier p0 or cancellation approval is not standing authorization. |

## Experiment naming

The matched baseline preserves the released config. Any change to encoder,
input views, action representation, predictor, loss, schedule, batch topology,
data mixture, or sampling is a new experiment. Name the change in the config
and W&B run, retain the baseline as a control, and use `fork` lineage if it
starts from a prior schema-v2 checkpoint.
