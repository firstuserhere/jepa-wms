# JEPA-WM research infrastructure completion audit

This audit distinguishes code/schema evidence from the runtime evidence that
must exist before a large allocation is scientifically defensible. A local
unit test is not treated as proof that NCCL, PVC semantics, video decoding, or
managed recovery work on the target cluster.

## Requirement evidence

| Requirement | Current evidence | Status needed before full run |
|---|---|---|
| Four-node H200 launch | `droid_train_full.yaml` requests 4×`H200:8`; `render_k8s_task.py` preserves topology and pins `k8s/Skypilot`; both profiles pass Sky schema validation. | K8s full-task resource planning must accept the two named RWX volumes. |
| Cross-node distributed launch | `distributed_smoke.yaml` uses two nodes; `app/torchrun.py` checks local CUDA binding, an all-rank NCCL reduction, exact 16-rank membership, two distinct hostnames and H200 identity on every rank. | Managed smoke must finish once after its controlled restart and publish checksum-protected terminal evidence. |
| Automatic preemption/resume | Managed recovery is configured; run identity is stable across a Sky task; the two-node actual training smoke stops after epoch 1 and must strictly resume through epoch 2. A combined `RUNTIME_READY.json` is mandatory for `full`. | Runtime receipt must bind the terminal distributed smoke and report `strict_resume_verified: true`, update 8→16, the same W&B ID, compatible dataset/DINO/Git/resume contract, and a verified schema-v2 latest object. |
| Durable checkpoint publication | Schema-v2 objects are immutable, file-fsynced, same-directory atomically replaced, SHA-256/size verified; aliases and promotion receipts are independently atomic and checksummed. | Verify close-to-open visibility from another node/PVC mount during the distributed and training smokes. |
| Top-checkpoint retention | Roles are `latest`, `best_rollout`, `best_planning`; GC also retains three newest independent objects and all pending planning candidates. | Observe retention after at least four saves and one promotion on the target PVC. |
| Strict manifests/compatibility | Full resolved config and resume contract; clean pinned source; dataset fingerprint/runtime binding; DINO code/weight identity; exact world size, dataset length, IPE and batch contract. | Stage full data, then pass runtime binding on the mounted PVC. |
| Always-on W&B | Online mode and `WANDB_API_KEY` are required; rank zero persists one W&B ID and `resume="must"` on recovery. | Smoke must create and resume the same online run. |
| Checkpoint contents | Predictor/action/proprio/head state, optimizer, scaler, LR/WD and rollout scheduler state, epoch/update, every-rank Python/NumPy/CPU/CUDA RNG, sampler/loader/dataset RNG state, config, Git patch, dataset, DINO, W&B, lineage and candidate metrics. | Load the epoch-1 artifact during the controlled training recovery smoke. |
| Promotion provenance | Fixed held-out multi-horizon rollout suite; exact checkpoint checksum and evaluation-config checksum in planning registry; complete episode counts; checksum-bound role receipts; one-shot no-gradient released-rollout result; planning is restricted to the released one-node/eight-rank seed topology. A combined `QUALIFIED.json` binds those results to the dataset, DINO bytes, Git commit and W&B identity, and is mandatory for `full`. | Released qualification must publish the combined receipt; the full task must verify it against its mounted inputs before torchrun. |
| Continued pretraining | Explicit schema-v2 `fork` loads model state strictly, resets optimizer/schedulers/progress/RNG/sampler/W&B, and records the complete parent lineage; later failures exactly resume the child stage. | Exercise after the matched baseline, not before it. |
| Full DROID access | Public GCS API confirms `gs://gresearch/robotics/droid_raw/1.0.1` and a live 2026-08-15 inventory of 74,970 trajectory objects; staging filters SVO/stereo, requires an exact source/staged ID match, verifies every required H5/metadata/camera path, and binds pinned Franka bytes. | Create the 8 TiB RWX PVC and complete staging/manifest publication. |
| Released baseline identity | Hub model commit `9b9c41e…` and checkpoint SHA-256 `daa69198…` are both pinned. A local read-only audit verified the 2.7 GB object, epoch 315, 176 predictor tensors, 228,835,328 parameters, 12 blocks, and 7→1024 action projection; all keys strictly load into the predictor constructed from the released DINOv3 DROID config. | Qualification must download, verify, register, and evaluate that exact object on the target runtime. |
| Native DINOv3 identity | Code commit `54694f7…`, native filename identity `8aa4cbdd`, and caller-supplied full SHA-256 are fail-closed. Hub `model.safetensors` is explicitly rejected as a silent substitute. | Accept the Meta access form, add the private URL as `DINOV3_WEIGHTS_URL`, and run `stage-dinov3`. |

## Current external state (2026-08-15)

- `sky check -o json`: `default`, `cpu`, and `fellows` expose Kubernetes
  compute; none exposes GCP compute.
- `sky gpus list H200 --infra kubernetes`: context `Skypilot`, requestable
  counts 1/2/4/8 per node, 128 total H200s, zero free at the most recent
  observation. Availability is dynamic and is not a capacity reservation.
- No user-owned JEPA-WM PVC has been identified. Existing listed PVCs belong to
  other users and must not be reused without explicit coordination.
- No job, PVC, GCS bucket, or paid GPU was created by this work.
- The implementation is still an uncommitted local dirty tree. Actual launch
  deliberately requires a pushed immutable 40-hex Git revision.

## Go/no-go sequence

1. Review and create collision-free 8 TiB data + 2 TiB checkpoint RWX PVCs.
2. Commit/push this tree and pin the resulting commit.
3. Stage DROID and require the complete combined manifest.
4. Stage the native DINOv3 `.pth` and require its full checksum.
5. Pass the two-node NCCL/W&B/managed-recovery smoke.
6. Pass released checkpoint rollout and planning qualification.
7. Pass the two-node real DROID gradient/save/restart/resume smoke.
8. Only then submit the 4×8 H200 matched run with explicit `p1` priority.

Any failed step is a no-go for the next expensive step; it is not converted
into a warning or silently bypassed.
