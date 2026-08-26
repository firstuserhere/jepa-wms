# Research status

Last audited: 2026-08-25 (America/Los_Angeles).

## Executive state

The implementation is pushed to the research fork and has a draft PR. DROID
and checkpoint volumes exist. Full DROID staging and a two-node managed
recovery/DDP smoke succeeded. The released checkpoint loaded strictly and the
first canonical planning suite exceeded the paper's central DROID Action Score,
but the user cancelled qualification before all eight suites completed.

There is no active JEPA-WM GPU job. No matched DROID training run has started.
There is no combined `QUALIFIED.json` and no real-gradient
`RUNTIME_READY.json`; therefore a full training launch is not yet unlocked by
the repository's own gates.

The local `codex/pantheon-training-readiness` worktree now renders the full
Pantheon task with zero linter errors, current shim-generated InfiniBand,
canonical checkpoint storage, stable W&B identity, continuous `training-v1`
heartbeat, cumulative effective MFU, and measured epoch-boundary recovery.
Those are implementation/test results only until this branch is reviewed,
committed, pushed to an immutable SHA, and exercised on H200s.

A bounded single-node MFU-box profile is now implemented for the exact matched
DROID/DINOv3 training path: one 8xH200 node, the production per-GPU batch and
loader settings, 96 real optimizer updates, two epoch-boundary checkpoints,
five-second W&B system telemetry, worst-rank pipeline diagnostics, and an
integrity-protected `MFU_BOX.json` decision record. This is not yet runtime
evidence. The 32-GPU launch remains gated on two consistent successful boxes.

## Git state

| Item | State |
|---|---|
| Fork | `firstuserhere/jepa-wms` |
| Integration branch | `codex/droid-dinov3-research-infra` |
| Draft PR | `firstuserhere/jepa-wms#1` |
| Meta base | `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` |
| Runtime-qualified source | `caf06234f89beb4cbbf87fd9833a17d6dcc46e44` |
| New-agent base | annotated tag `jepawm-droid-infra-2026-08-19` |

## Durable assets

Both volumes are in the default Sky workspace and were `READY`, not in use, at
the latest check:

| Volume | Size | Mode | Purpose |
|---|---:|---|---|
| `firstuserhere-jepawm-droid` | 8192 GiB | RWX PVC | Staged DROID and Franka validation data |
| `firstuserhere-jepawm-checkpoints` | 2048 GiB | RWX PVC | Immutable DINO/released model artifacts |
| `checkpoints` | shared Pantheon volume | RWX | Per-user/per-experiment checkpoints, receipts, logs, W&B metadata |

The DROID volume should be mounted read-only by training jobs. Do not reuse
another researcher's volume.

## Immutable artifacts

| Artifact | Identity |
|---|---|
| DROID loader-ready dataset | 73,431 episodes; fingerprint `5c93744b881524b7f12e38a6575c5c6a06cac444ebb46848e2fa343e0b570c90` |
| Official DROID source inventory | 74,970 episodes before loader-readiness filtering |
| DINOv3 code | `54694f7627fd815f62a5dcc82944ffa6153bbb76` |
| DINOv3 ViT-L/16 weights | SHA-256 `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035` |
| Released JEPA-WM Hub revision | `9b9c41ef249466630dbf1a20e78391865d07b3b9` |
| Released JEPA-WM checkpoint | SHA-256 `daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa` |

Private artifact URLs and tokens are intentionally absent from this repository.

## Job ledger

### 8448 — DROID staging

- Name: `jepawm-stage-droid-raw`
- Topology: 4 x H200:8
- Priority: p0
- Source: `8be8f4f81cebd3c3db6c55ed365b81936ff3bfb4`
- Outcome: `SUCCEEDED`, zero recoveries
- Evidence: all four shard receipts verified; 74,970 source episodes; 73,431
  loader-ready episodes; manifest fingerprint recorded above.

### 8451 — distributed recovery smoke

- Name: `jepawm-distributed-recovery-smoke`
- Topology: 2 x H200:8, 16 ranks on two distinct hosts
- Priority: p0
- Source: `8be8f4f81cebd3c3db6c55ed365b81936ff3bfb4`
- Outcome: `SUCCEEDED`, one intentional managed recovery
- Evidence: exact 16-rank NCCL collective and recovery marker succeeded.
- W&B: `https://wandb.ai/pantheoninc-pantheon-inc/vjepa_wm/runs/infra-73b69f29a8cfb6e3dbf41b3b`

### 8529 — released DROID qualification

- Name: `jepawm-droid-released-qualification`
- Topology: 4 x H200:8
- Priority: p0
- Source: `caf06234f89beb4cbbf87fd9833a17d6dcc46e44`
- Stable run ID: `released-droid-qual-caf0623-20260819-p0`
- Outcome: `CANCELLED` at the user's request after the first of eight planning
  suites; zero recoveries.
- Model evidence: released epoch-315 predictor loaded with all keys matched.
- Suite 1: all 64 canonical CEM/L2 episodes completed.
- `ep_end_dist_xyz`: `0.03617349954464993`.
- DROID Action Score: `51.06120036428006`.
- Paper reference: `48.2 +/- 1.8`; the observed single-suite result is about
  `+1.59` paper standard deviations and inside the configured 4-sigma gate.
- Rollout W&B: `https://wandb.ai/pantheoninc-pantheon-inc/vjepa_wm/runs/qt9rny6gmk8dzty0etdg0k64`
- Planning W&B: `https://wandb.ai/pantheoninc-pantheon-inc/vjepa_wm/runs/b37nd5k2ohhbmyl6qxtzjw9m`

This is encouraging evidence, not a complete qualification. Seven suites and
the combined receipt were not completed.

## Proven versus unproven

| Claim | Status | Evidence or missing gate |
|---|---|---|
| Full raw DROID is durably staged | Proven | Job 8448 and manifest fingerprint |
| Cross-node torchrun/NCCL works | Proven for 2 nodes/16 ranks | Job 8451 |
| Sky managed recovery restarts the gang | Proven for distributed smoke | Job 8451, one intentional recovery |
| Released predictor and DINO artifacts are compatible | Proven | Job 8529 strict load |
| Canonical planning can match the paper region | Partially proven | First of eight suites in job 8529 |
| Released baseline is fully qualified | Not proven | Missing seven suites and `QUALIFIED.json` |
| Real DROID gradient/save/resume is exact | Not proven | Training smoke never run; no `RUNTIME_READY.json` |
| Current checkpoint retention works on target PVC | Not runtime-proven | Unit-tested only; needs four-save/promotion observation |
| Matched 94,500-update DROID training works | Not proven | Full run never launched |
| Dashboard shows trustworthy live MFU | Implemented, not runtime-proven | Static snapshot validator passes; needs live W&B/H200 evidence |
| Single-node training is efficient enough to scale | Not proven | Needs two successful 8xH200 `MFU_BOX.json` records plus W&B all-GPU review |
| Pantheon full task passes current linter | Proven locally | 0 errors; sole static warning requires a live training-v1 audit |
| Continued pretraining across datasets works at scale | Not proven | Fork mode implemented/unit-tested, not exercised after baseline |

## Required next sequence

1. Review, commit, and push the Pantheon-readiness branch to a new immutable SHA.
2. Re-run released qualification from that SHA and finish all suites
   plus `QUALIFIED.json`.
3. Run the real two-node DROID gradient/save/restart/resume smoke and publish
   `RUNTIME_READY.json`.
4. Launch the matched 4 x 8 H200, 94,500-update baseline only after both receipts
   pass fail-closed verification.
5. Compare fixed rollout, planning, throughput, MFU, and checkpoint evidence to
   the released baseline.
6. Begin continuous pretraining as an explicit forked stage with new data,
   optimizer state, W&B run, and lineage.

Every launch requires fresh explicit user authorization, including the priority.
Past p0 approval is historical evidence, not standing permission.
