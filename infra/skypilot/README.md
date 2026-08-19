# SkyPilot DROID + DINOv3 runbook

This bundle keeps the released DROID scientific config unchanged and layers
only operational requirements on top: strict continuation, required W&B,
pinned DINOv3 artifacts, durable checkpoint roles, and a canonical planning
promotion suite.

The intended order is:

1. Preflight locally (zero cloud mutation).
2. Idempotently stage the official non-stereo DROID raw `1.0.1` release and
   the released Franka validation set in durable GCS or an RWX PVC.
3. Stage and checksum the original native DINOv3 ViT-L/16 artifact.
4. Run the controlled two-node managed-recovery/NCCL/W&B smoke.
5. Qualify the released DROID checkpoint through rollout and planning on the
   same 4-node x 8-GPU topology used for matched training.
6. Run the two-node, 1% DROID gradient/save/recovery smoke.
7. Submit matched training on 4 nodes x 8 H200.

## Current SkyPilot environment: Kubernetes H200 pool

The connected Enterprise API currently exposes Kubernetes compute in the
`default`, `cpu`, and `fellows` workspaces; it does not expose GCP compute.
The `Skypilot` Kubernetes context advertises H200 nodes with 1/2/4/8 GPUs per
node and 128 H200s total. Availability is dynamic (the read-only check on
2026-08-15 reported zero free), so the default explicit `p1` launch priority is what
queues/bids for capacity. Do not use the GCP profile below unless `sky check`
later confirms GCP in the selected workspace.

Kubernetes managed recovery needs durable RWX PVCs. Templates are provided,
but applying them is intentionally manual because creating 10 TiB of storage
is a material external action:

```bash
SKY_WORKSPACE=YOUR_KUBERNETES_WORKSPACE
DROID_VOLUME=YOUR_UNIQUE_DROID_VOLUME_NAME
CHECKPOINT_VOLUME=YOUR_UNIQUE_CHECKPOINT_VOLUME_NAME

sky volumes apply infra/skypilot/volumes/droid_raw_k8s.yaml \
  --name "$DROID_VOLUME" --config "active_workspace=$SKY_WORKSPACE"
sky volumes apply infra/skypilot/volumes/checkpoints_k8s.yaml \
  --name "$CHECKPOINT_VOLUME" --config "active_workspace=$SKY_WORKSPACE"
```

After the DINOv3 access form is accepted, store the private native-weight URL
as the managed Sky secret `DINOV3_WEIGHTS_URL`. Then use the exact full
SHA-256 locally obtained for the downloaded `8aa4cbdd` artifact:

```bash
GIT_URL=https://github.com/YOUR_ORG/jepa-wms.git
GIT_REF=FULL_40_HEX_COMMIT
DINOV3_SHA256=FULL_64_HEX_SHA256
QUALIFICATION_RUN_ID=released-droid-qualification-001
DISTRIBUTED_SMOKE_RUN_ID=cross-node-recovery-smoke-001
TRAIN_SMOKE_RUN_ID=droid-training-recovery-smoke-001

infra/skypilot/launch_k8s.sh stage \
  --droid-volume "$DROID_VOLUME" \
  --git-url "$GIT_URL" --git-ref "$GIT_REF" --workspace "$SKY_WORKSPACE"

infra/skypilot/launch_k8s.sh stage-dinov3 \
  --checkpoint-volume "$CHECKPOINT_VOLUME" \
  --dinov3-weights-sha256 "$DINOV3_SHA256" \
  --git-url "$GIT_URL" --git-ref "$GIT_REF" --workspace "$SKY_WORKSPACE"

infra/skypilot/launch_k8s.sh stage-released \
  --checkpoint-volume "$CHECKPOINT_VOLUME" \
  --git-url "$GIT_URL" --git-ref "$GIT_REF" --workspace "$SKY_WORKSPACE"

infra/skypilot/launch_k8s.sh distributed-smoke \
  --run-id "$DISTRIBUTED_SMOKE_RUN_ID" \
  --checkpoint-volume "$CHECKPOINT_VOLUME" \
  --git-url "$GIT_URL" --git-ref "$GIT_REF" --workspace "$SKY_WORKSPACE"

infra/skypilot/launch_k8s.sh qualify \
  --run-id "$QUALIFICATION_RUN_ID" \
  --droid-volume "$DROID_VOLUME" \
  --checkpoint-volume "$CHECKPOINT_VOLUME" \
  --dinov3-weights-sha256 "$DINOV3_SHA256" \
  --git-url "$GIT_URL" --git-ref "$GIT_REF" --workspace "$SKY_WORKSPACE"

infra/skypilot/launch_k8s.sh train-smoke \
  --run-id "$TRAIN_SMOKE_RUN_ID" \
  --distributed-smoke-run-id "$DISTRIBUTED_SMOKE_RUN_ID" \
  --droid-volume "$DROID_VOLUME" \
  --checkpoint-volume "$CHECKPOINT_VOLUME" \
  --dinov3-weights-sha256 "$DINOV3_SHA256" \
  --git-url "$GIT_URL" --git-ref "$GIT_REF" --workspace "$SKY_WORKSPACE"

infra/skypilot/launch_k8s.sh full \
  --qualification-run-id "$QUALIFICATION_RUN_ID" \
  --runtime-readiness-run-id "$TRAIN_SMOKE_RUN_ID" \
  --droid-volume "$DROID_VOLUME" \
  --checkpoint-volume "$CHECKPOINT_VOLUME" \
  --dinov3-weights-sha256 "$DINOV3_SHA256" \
  --git-url "$GIT_URL" --git-ref "$GIT_REF" --workspace "$SKY_WORKSPACE"
```

`stage` uses four full H200 nodes to mirror disjoint, size-balanced institution
shards of the public filtered raw DROID release into the 8 TiB RWX PVC. Each
rank checksum-checks its shard and publishes an immutable receipt; rank 0
requires all four receipts and then publishes the manifest only after checking
every episode. `stage-dinov3` downloads the private URL quietly, validates the
full checksum, and atomically publishes the artifact into the checkpoint PVC.
`stage-released` downloads Meta's pinned DROID world-model release once through
the managed HF identity, verifies its full SHA-256, and atomically publishes it
for all qualification nodes.
Training copies that verified artifact into each node's local source tree.

Add `--dry-run` to any Kubernetes command to render its provider/storage
profile and run Sky schema validation without checking secrets, creating a
volume, or submitting a job.

## Optional GCP/object-store profile

From the repository root:

```bash
infra/skypilot/launch.sh preflight

# Actual jobs deliberately reject a local workdir upload: SkyPilot excludes
# .git from those uploads, while research checkpoints require the exact commit
# and dirty patch. Commit/push this tree, then use the immutable commit SHA.
GIT_URL=https://github.com/YOUR_ORG/jepa-wms.git
GIT_REF=FULL_40_HEX_COMMIT
SKY_WORKSPACE=YOUR_GCP_ENABLED_WORKSPACE
QUALIFICATION_RUN_ID=released-droid-qualification-001
DISTRIBUTED_SMOKE_RUN_ID=cross-node-recovery-smoke-001
TRAIN_SMOKE_RUN_ID=droid-training-recovery-smoke-001

infra/skypilot/launch.sh stage \
  --droid-store gs://YOUR_DATA_BUCKET/jepawm \
  --git-url "$GIT_URL" --git-ref "$GIT_REF" --workspace "$SKY_WORKSPACE"

infra/skypilot/launch.sh distributed-smoke \
  --run-id "$DISTRIBUTED_SMOKE_RUN_ID" \
  --checkpoint-store gs://YOUR_CHECKPOINT_BUCKET/jepawm \
  --git-url "$GIT_URL" --git-ref "$GIT_REF" --workspace "$SKY_WORKSPACE"

infra/skypilot/launch.sh qualify \
  --run-id "$QUALIFICATION_RUN_ID" \
  --droid-store gs://YOUR_DATA_BUCKET/jepawm \
  --checkpoint-store gs://YOUR_CHECKPOINT_BUCKET/jepawm \
  --dinov3-weights-uri gs://YOUR_ARTIFACT_BUCKET/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --dinov3-weights-sha256 FULL_64_HEX_SHA256 \
  --git-url "$GIT_URL" --git-ref "$GIT_REF" --workspace "$SKY_WORKSPACE"

infra/skypilot/launch.sh train-smoke \
  --run-id "$TRAIN_SMOKE_RUN_ID" \
  --distributed-smoke-run-id "$DISTRIBUTED_SMOKE_RUN_ID" \
  --droid-store gs://YOUR_DATA_BUCKET/jepawm \
  --checkpoint-store gs://YOUR_CHECKPOINT_BUCKET/jepawm \
  --dinov3-weights-uri gs://YOUR_ARTIFACT_BUCKET/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --dinov3-weights-sha256 FULL_64_HEX_SHA256 \
  --git-url "$GIT_URL" --git-ref "$GIT_REF" --workspace "$SKY_WORKSPACE"

infra/skypilot/launch.sh full \
  --qualification-run-id "$QUALIFICATION_RUN_ID" \
  --runtime-readiness-run-id "$TRAIN_SMOKE_RUN_ID" \
  --droid-store gs://YOUR_DATA_BUCKET/jepawm \
  --checkpoint-store gs://YOUR_CHECKPOINT_BUCKET/jepawm \
  --dinov3-weights-uri gs://YOUR_ARTIFACT_BUCKET/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --dinov3-weights-sha256 FULL_64_HEX_SHA256 \
  --git-url "$GIT_URL" --git-ref "$GIT_REF" --workspace "$SKY_WORKSPACE"
```

Add `--dry-run` to any launch command to execute the same local preflight and
URI validation without contacting SkyPilot or submitting a job.

Actual launches require a GitHub URL plus a full 40-hex commit and pass them to
SkyPilot's Git workdir clone. Moving branches/tags and `.git`-less local uploads
are rejected. `setup_training.sh` verifies the remote checkout is exactly that
commit and clean; schema-v2 checkpoints then embed the commit, status, complete
dirty patch (empty for these clean launches), and patch checksum.

Every actual submission contains an explicit CLI priority. The launcher defaults
to `--priority p1` (the case-sensitive spelling in this SkyPilot Enterprise
workspace), which can displace `p2` work. Escalation is deliberate via
`--priority p0`; the YAML files contain no hidden priority setting. `HF_TOKEN` and
`WANDB_API_KEY` are managed SkyPilot secret references; this bundle never
accepts their values on the command line. The Hugging Face identity behind
`HF_TOKEN` must already have accepted the DINOv3 license.

The original DINOv3 ViT-L/16 checkpoint expected by JEPA-WM is the Meta `.pth`
artifact named `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`. The current Hub
conversion exposes `model.safetensors`, whose parameter layout is not silently
interchangeable. Stage the original `.pth` at a non-secret GCS URI and pass its
full SHA-256. Setup refuses missing, non-`8aa4cbdd` or mismatched artifacts. The
DINOv3 code checkout is independently pinned to
`54694f7627fd815f62a5dcc82944ffa6153bbb76`.

The Kubernetes DROID stage uses pinned rclone v1.73.5 to copy and checksum-check
four disjoint institution shards from `gs://gresearch/robotics/droid_raw`,
excluding SVO and stereo MP4 data (about 5.6 TB). Its local `Colon` encoding
maps source U+003A to SharedFS-compatible U+FF1A and records both source and
staged inventory fingerprints; a pre-existing U+FF1A or mapping collision is
rejected. (The optional GCS-to-GCS profile retains idempotent `gsutil rsync`.)
The live `1.0.1` source contains 74,970 trajectory objects. Staging relists that official source after the copy
and requires the staged trajectory IDs to match it exactly, so a partial raw
mirror cannot pass merely by exceeding a loose count threshold. It creates
`DROID/droid_paths.csv` and a fingerprinted
`DROID/droid_manifest.json` under the staged prefix. The full task mounts that
prefix as `DATASET_RO`. Before publishing that manifest, staging verifies every
listed episode through the target mount, including `trajectory.h5`,
metadata, and the referenced left-camera MP4; a listing-only manifest cannot
pass the strict training gate. It also byte-compares every staged `Franka_hf`
file against `facebook/jepa-wms` dataset revision
`6116f042ae7ae4c8e3f1fd2f194f432615664182`, records the canonical file list,
per-file hashes and tree hash, and binds that auxiliary identity into the
combined dataset fingerprint. Checkpoints use a separate parameterized
`MODEL_CHECKPOINT_RW` cached mount with immediate write-back.

The qualification task downloads
`facebook/jepa-wms@9b9c41ef249466630dbf1a20e78391865d07b3b9`'s
`jepa_wm_droid.pth.tar`, requires its published object SHA-256
`daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa`,
writes that checksum beside the durable copy, and runs
the released data-rollout and planning configs. Matched training promotes on
the canonical DROID L2 CEM H3 suite; the other released planners are retained
as qualification-only diagnostics.

`best_rollout` has one fixed scientific identity: primary validation loader 0
from the released config (`Franka_hf`, 5 frames, batch 4, no drop-last,
`exterior_image_2_left`), sampler epoch 0, seed 50234, the first 64 distributed
batches on the fixed 32-rank topology, and visual-L2 horizons 1–4. It runs every
epoch. The full task runs only the canonical CEM planning promotion suite;
additional planners run only in qualification jobs. Planning itself uses only
ranks 0–7 (one physical node), preserving the released rank-dependent episode
partition and random streams exactly; ranks 8–31 wait outside planner
collectives. This synchronous fallback spends less cluster capacity efficiently
than upstream's Slurm-submitted asynchronous jobs, but it does not change the
scientific comparison and it remains recoverable from the immutable pending
checkpoint.

Rank zero publishes `released_droid/QUALIFIED.json` only after both released
evaluations are complete. The receipt binds the released Hub revision and
checkpoint SHA, complete source-matched DROID manifest, native DINOv3 SHA,
source Git commit, fixed rollout result, all planning results, canonical
promotion config, exact 64-episode count, and W&B identity. `full` requires
`--qualification-run-id` and re-verifies that receipt against its own mounted
dataset, DINO bytes, and Git commit before initializing torchrun. A stale,
partial, differently configured, or merely hand-asserted qualification cannot
unlock the allocation.

Each successful role promotion also writes a checksum-bound receipt beside
the immutable object. This is essential for asynchronous planning: the
checkpoint hash remains the exact object that was evaluated, while the receipt
records the metric name, value, direction, complete metric payload, and role.
Role resolution verifies both hashes. DROID's dummy offline environment always
reports episode success, so DROID `best_planning` correctly minimizes XYZ goal
endpoint error and reports the corresponding DROID Action Score; simulator
tasks use actual episode success.

By default, each training/qualification task uses its stable
`SKYPILOT_TASK_ID` as `JEPAWM_RUN_ID`, so managed recovery reuses the exact log,
checkpoint and W&B identity. For a named new experiment add `--run-id NAME`.
To intentionally continue an existing named run, add both `--run-id NAME` and
`--resume`; without that explicit opt-in, an existing or ambiguously populated
run directory is rejected before training.

Managed recovery restarts the entire gang if a node is preempted. The
distributed smoke deliberately exits once with code 42, then verifies that
the same SkyPilot task resumes from durable state. The DROID training smoke is
a second, stronger two-phase gate: it publishes and checksums an epoch-1 v2
`latest` object, triggers managed recovery, strictly restores it with the same
W&B ID, advances through epoch 2, and verifies the final alias checksum,
epoch, and global update. Its two-epoch mathematical config is identical on
both attempts; only the operational checkpoint-boundary stop changes. For a manual cancellation,
use SkyPilot's graceful cancellation option so cached checkpoint writes are
flushed before teardown.

The cross-node smoke now records one H200/hostname record for every one of its
16 ranks and proves that those ranks span two nodes. After the real DROID smoke
strictly resumes from epoch 1/update 8 through epoch 2/update 16, rank zero
loads and validates the resulting schema-v2 checkpoint and publishes
`RUNTIME_READY.json`. That checksum-protected receipt binds both controlled
recoveries, their W&B identities, the source commit, exact dataset manifest,
DINOv3 identity, resume contract and final checkpoint. `full` requires the
training-smoke run ID and verifies this receipt in addition to the released
qualification receipt; neither smoke can be skipped by relying on the runbook
alone.

Exact recovery and continued pretraining are intentionally different modes.
`checkpointing.load_mode: resume` restores the entire mathematical trajectory
and rejects config, dataset, encoder, sampler topology, W&B, or scheduler
changes. A new stage may set `checkpointing.load_mode: fork`,
`checkpointing.fork_optimizer: reset`, and `meta.pretrained_path` to any
schema-v2 object. It loads model components strictly, starts new optimizer,
schedulers, progress, RNG, sampler and W&B state, and records the parent file
SHA, step, dataset, encoder, Git state and W&B run in every child checkpoint's
lineage. After that stage writes its own `latest`, preemption resumes it exactly.

These task files request GCP H200s with `network_tier: best`; they deliberately
do not request `local_disk`, which is unsupported for this GCP resource shape
in the pinned SkyPilot Enterprise client. The local workspace must have GCP
enabled before SkyPilot can complete a provider dry-run; a disabled GCP account
is an external workspace gate, not a task-schema fallback.
