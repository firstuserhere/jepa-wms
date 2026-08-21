# Architecture

## What the model is

The DROID JEPA-WM is an action-conditioned latent dynamics model. It does not
decode pixels during core training, and it is not an online EMA-target or
patch-masking JEPA training loop. A frozen pretrained visual encoder defines
the feature space; a trainable causal transformer predicts the next visual
patch features under robot actions.

The exact scientific base is:

`configs/vjepa_wm/droid_final_sweep/droid_4fpcs_fps4_r256_dv3vitl_asp1_pred_AdaLN_depth12_noprop_repro_2roll_4n.yaml`

The operational quality overlay is:

`infra/skypilot/droid_dinov3_quality_overlay.yaml`

## Data-to-loss path

```text
DROID raw episode
  -> one left exterior RGB stream + 7-D Cartesian/gripper state
  -> random 4-frame clip at 4 FPS, resized/cropped to 256 x 256
  -> pose differences: dx,dy,dz + relative Euler xyz + gripper delta
  -> frozen DINOv3 ViT-L/16 patch features
  -> 12-block causal action-modulated AdaLN transformer
  -> next-frame latent regression + truncated autoregressive rollout loss
```

### Images and tokens

- Input visual tensor: `B x 4 x 3 x 256 x 256`.
- Training camera key: `left_mp4_path`; the loader resolves the episode's left
  MP4 recording. Validation uses `exterior_image_2_left` from the released
  Franka set.
- Encoder: frozen DINOv3 ViT-L/16, distilled on LVD-1689M.
- Encoder output key: `x_norm_patchtokens`.
- Patch grid: `16 x 16`, so each frame has 256 patch tokens.
- Token width: 1024.
- The world model consumes patch tokens only. It does not feed a CLS token to
  the predictor.
- Frames are encoded independently by flattening time into the image batch,
  then reshaped to `B x T x 1 x 16 x 16 x 1024`.

The native DINO code revision is
`54694f7627fd815f62a5dcc82944ffa6153bbb76`. The expected ViT-L/16 weight SHA-256
is `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`.
Code revision, native filename identity, and full checksum are verified before
training.

### Actions and state

DROID state has seven values: Cartesian XYZ, Euler XYZ, and scalar gripper
closedness. The loader converts consecutive poses into seven-dimensional
actions:

1. XYZ displacement;
2. relative rotation, represented again as XYZ Euler angles;
3. gripper closedness delta.

With `frameskip=1`, `action_skip=1`, and tubelet size 1, the predictor action
dimension remains 7. Actions are not normalized in the released config. A
linear layer maps each 7-D action to the 1024-D predictor width.

The config calls this `action_conditioning: token`, but the AdaLN predictor's
important implementation detail is that the projected per-frame action is the
conditioning vector for every transformer block. Each block turns it into
shift, scale, and gate values for attention and MLP normalization. It is not a
visual patch and is not predicted as an output token. Proprioception is disabled
for this released `noprop` model; state is still used to construct actions and
planning targets.

### Predictor

The predictor is `VisionTransformerAdaLN`:

- 12 transformer blocks;
- width 1024;
- 16 attention heads;
- GELU MLPs;
- rotary spatial/temporal position handling;
- causal/local temporal attention window of 3 with full spatial attention;
- action-driven adaptive LayerNorm modulation in every block;
- output projection back into the frozen encoder's 1024-D feature space.

The released checkpoint audit found 176 predictor tensors and 228,835,328
predictor parameters. Qualification loaded every released predictor key
strictly at epoch 315.

### Objective and rollout

The primary loss is mean squared error in frozen DINOv3 patch-feature space.
Predictions are shifted against the next frame (`shift=1`). The training config
also uses a two-step sequential rollout objective with a context window of 3
and stop-gradient between rollout steps. This trains short autoregressive
dynamics without updating the DINO encoder. Decoder heads are optional and are
not required for world-model training or latent planning.

## Released schedule

| Setting | Value |
|---|---|
| Nodes / GPUs | 4 nodes x 8 GPUs = 32 ranks |
| Per-rank batch | 8 clips |
| Effective global batch | 256 clips |
| Updates per epoch | 300 |
| Epochs | 315 |
| Total optimizer updates | 94,500 |
| Total clip presentations | 24,192,000 |
| Precision | bfloat16 mixed precision |
| Optimizer LR | constant `5e-4`, no warmup |
| Weight decay | `1e-7` to `1e-6` schedule |
| Gradient clipping | 1 |

The repository specifies the mathematical schedule, not Meta's original
hardware, wall-clock duration, dataloader throughput, or cluster load. Do not
claim an original duration from the 315-epoch number.

## Planning path

Planning optimizes action sequences against a latent goal. For the canonical
DROID comparison, CEM searches horizon-3 action sequences and minimizes L2
distance between predicted and goal latent features. The qualification suite
uses 64 episodes.

Only ranks 0-7 participate in in-process planning even when qualification uses
32 ranks. This preserves the released one-node, eight-rank episode partition
and random streams. Ranks 8-31 wait by design. That is scientifically faithful
but utilization-inefficient and must be budgeted separately from training MFU.

DROID's offline dummy environment always reports success, so checkpoint
selection minimizes `ep_end_dist_xyz`. The reported DROID Action Score is:

```text
800 * (0.1 - endpoint_xyz_error), if endpoint_xyz_error < 0.1; otherwise 0
```

## Checkpoint and evaluation flow

Rank zero writes immutable schema-v2 checkpoint objects and atomic aliases:

- `latest` for exact recovery;
- `best_rollout` for a fixed held-out, multi-horizon latent rollout corpus;
- `best_planning` for verified planning results.

Garbage collection retains those roles, pending planning candidates, and three
recent independent checkpoints. A checkpoint contains trainable model state,
optimizer/scaler/schedulers, epoch/update, all-rank RNG, sampler state, resolved
config, Git manifest, dataset identity, DINO identity, W&B identity, lineage,
and candidate metrics.

Planning evaluates an immutable checkpoint checksum and config checksum. A
durable registry retains pending work; result publication and role promotion
are atomic and checksum-bound. `resume` requires mathematical identity. `fork`
strictly loads model weights but resets optimizer, schedulers, progress, RNG,
sampler, and W&B while recording complete parent lineage.

Recovery objects are published only at completed epoch boundaries and before
rollout or planning evaluation. Each boundary records slowest-rank epoch time,
checkpoint-write time, and the 300-second loss budget; runtime qualification
must observe compliant boundaries on both sides of managed recovery. Rank-zero
W&B telemetry uses one stable experiment ID and a ten-second `training-v1`
heartbeat. Effective MFU is cumulative useful model forward/backward FLOPs over
contiguous slowest-rank allocation wall time, so data, optimizer, collectives,
validation, logging, and checkpoint overhead remain visible in the denominator.
