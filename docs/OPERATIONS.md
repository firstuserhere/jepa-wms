# Operations

This is the concise operator contract. Command details live in
`infra/skypilot/README.md`; scientific state lives in `docs/RESEARCH_STATUS.md`.

## Current environment

- SkyPilot Enterprise API: `https://pantheon.svc.skypilot.co`.
- User/workspace at last audit: `kunvar@pantheon.inc`, `default`.
- Compute: Pantheon Kubernetes H200 pool on Crusoe.
- Durable storage: the two named RWX volumes in `docs/RESEARCH_STATUS.md`.
- Managed jobs must use `sky jobs launch`, not an unmanaged cluster launch.

Never copy a private DINO URL, `HF_TOKEN`, or `WANDB_API_KEY` into a command,
YAML, log, issue, PR, checkpoint metadata, or this repository. Refer to managed
secret names only.

## Launch authority

Dry runs and read-only inspection do not require launch authority. Submitting,
resuming, cancelling, changing priority, creating storage, or mutating durable
artifacts does. Obtain explicit authorization for the exact operation.

Priority is a scheduling policy, not a performance hyperparameter:

- use the explicitly approved `p0`-`p4` value on the CLI;
- the spelling is lowercase in this Sky installation;
- do not encode priority inside research YAML;
- never infer ongoing p0 permission from an earlier run.

## Current Pantheon task contract

Before the next launch, render and inspect the final task against the current
cluster guide. As audited on 2026-08-21, a full H200 node should start from
`H200:8`, 160 CPUs, and 1840 GB memory unless a dated measurement justifies a
smaller request. The final task must not retain provider `infra`, `disk_size`,
or ephemeral-storage requests. Multi-node jobs must use the approved
InfiniBand/network shim and prove the actual transport in a smoke.

The Pantheon renderer now removes `infra`, `disk_size`, provider selectors, and
ephemeral-storage requests; emits 160 CPU/1840 GB for each H200:8 pod; mounts
the shared volume exactly as `/checkpoints: checkpoints`; and injects the
checked-in output generated from `modal-skypilot` commit
`32ce29872d49b728ede2f90427f543a4aa64bcea`. The personal 2 TiB volume remains
an immutable model-artifact store; new checkpoints, receipts, logs, and W&B
metadata live under `/checkpoints/$PANTHEON_USER/$EXPERIMENT_TAG`.

## Mandatory W&B and MFU contract

Every training job must initialize W&B online on rank zero and preserve one run
ID across managed recovery. The dashboard contract is `training-v1`:

- summary heartbeat approximately every 10 seconds;
- `contract_version`, `heartbeat_at`, `total_steps`, `active_step`,
  `step_time_s`, `effective_mfu`, and `wandb_url`;
- history containing `train/loss*` and validation loss when available;
- stable step semantics across resume.

Effective MFU uses a representative real DROID operator profile sampled after
model forward/backward and before the transition optimizer. It counts unique
global model work once and divides the cumulative useful-FLOP sum by contiguous
slowest-rank wall time, all allocated GPUs, and the dense H200 SXM BF16 peak of
989 TFLOP/s. Input stalls, collectives, optimizer, logging, validation, and
checkpoint time remain in the denominator. Rank zero refreshes the exact
`training-v1` fields every ten seconds and writes a validator-compatible
`run_metadata/training_v1_snapshot.json` at each checkpoint. The original
compute-active metrics remain separately named diagnostics.

## Go/no-go ladder

The expensive run is gated in this order:

1. Clean, pushed, full 40-hex Git source.
2. Verified DROID/Franka manifest and native DINO/released-checkpoint checksums.
3. Multi-node DDP/NCCL/network/recovery smoke.
4. Complete released-checkpoint qualification and `QUALIFIED.json`.
5. Real DROID gradient/checkpoint/restart/resume smoke and
   `RUNTIME_READY.json`.
6. Full matched training.

Jobs 8448 and 8451 completed steps 2 and 3 for source `8be8f4f...`. Job 8529
partially completed step 4 on source `caf0623...`; it did not publish the
combined receipt. Steps 4-6 remain blocked.

## Safe status commands

Run from the repository root:

```bash
sky api info
sky jobs queue --all --refresh
sky volumes ls
sky jobs logs JOB_ID --status
git status --short --branch
git rev-parse HEAD
```

Do not poll aggressively. Prefer a bounded monitor that records state
transitions, W&B heartbeat age, progress, MFU, checkpoint freshness, recovery
count, and failure signatures. Monitoring must survive the originating agent
task; a Codex conversation is not an operations controller.

## Checkpoint policy

- Rank zero alone publishes shared checkpoint objects.
- Save exactly at every completed epoch boundary. The trainer publishes before
  rollout/planning evaluation, records the slowest-rank epoch-plus-checkpoint
  duration, and fails if it exceeds 300 seconds. Runtime readiness rejects the
  run unless both pre- and post-recovery boundaries meet that measured budget.
- Keep atomic `latest`, semantic best roles, pending planning references, and
  three recent immutable fallbacks.
- Verify checksum and close-to-open visibility from another node.
- Gracefully cancel when possible so cached writes flush; verify durable aliases
  after cancellation.
- `resume` is exact continuation only. Use `fork` for a new dataset/stage.

## Failure response

On a failure, first preserve evidence: job ID, task ID, immutable Git SHA,
recovery count, terminal log, W&B URL/run ID, latest durable checkpoint alias,
and receipt/checksum state. Do not repeatedly relaunch with ad hoc changes.
Classify the failure as code, data, artifact, distributed/network, storage,
telemetry, capacity, or contract mismatch, then create a narrow immutable fix.
