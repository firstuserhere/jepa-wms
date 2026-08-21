# Onboarding

## Mission

The immediate goal is to reproduce Meta's released DINOv3 + DROID JEPA-WM
baseline faithfully, establish trustworthy training and planning baselines,
and only then extend the model through lineage-recorded continuous pretraining.
Quality and scientific comparability are the primary metrics; speed and
utilization are constraints to optimize without changing the experiment.

## Repository identity

| Item | Value |
|---|---|
| Local checkout on the original workstation | `/Users/firstuserhere/Documents/ChatGPT/jepa-wm/jepa-wms` |
| Research fork | `https://github.com/firstuserhere/jepa-wms.git` |
| Meta upstream | `https://github.com/facebookresearch/jepa-wms.git` |
| Integration branch | `codex/droid-dinov3-research-infra` |
| Draft integration PR | `https://github.com/firstuserhere/jepa-wms/pull/1` |
| Upstream base | `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` |
| Last runtime-qualified source snapshot | `caf06234f89beb4cbbf87fd9833a17d6dcc46e44` |
| Immutable onboarding tag | `jepawm-droid-infra-2026-08-19` |

The runtime-qualified SHA is the code used by the partial released-checkpoint
qualification. The onboarding tag includes these documents and is the preferred
base for new development. A new job must pin the exact commit it actually uses;
neither the tag name nor the integration branch is a launch-time substitute for
the resolved 40-hex SHA.

## What this fork adds

Relative to Meta upstream, the fork adds:

- a torchrun-compatible multi-node entry point for SkyPilot;
- exact DROID staging and loader-ready dataset manifests;
- pinned native DINOv3 and released-checkpoint provenance;
- schema-v2 atomic checkpoints with strict resume and explicit fork lineage;
- `latest`, `best_rollout`, and `best_planning` roles plus recent fallbacks;
- immutable planning provenance, delayed-result reconciliation, and promotion;
- released-baseline qualification and runtime-readiness gates;
- managed-recovery, storage, DDP, W&B, and MFU test infrastructure;
- Kubernetes and object-store launch profiles.

The branch is 25 implementation commits ahead of Meta's pinned upstream base
before this documentation commit. See `git log 13cf1d9..HEAD --oneline` for the
authoritative local history.

## Read order

1. `AGENTS.md`
2. `docs/RESEARCH_STATUS.md`
3. `docs/ARCHITECTURE.md`
4. `docs/OPERATIONS.md`
5. `docs/WORKTREES.md`
6. `infra/skypilot/README.md`
7. The released DROID config and the quality overlay named in
   `docs/ARCHITECTURE.md`

## Directory map

| Path | Responsibility |
|---|---|
| `app/vjepa_wm/train.py` | Main train/validate/checkpoint/evaluation lifecycle |
| `app/vjepa_wm/video_wm.py` | Encoder/predictor wrapper, latent loss, and rollout |
| `app/plan_common/models/AdaLN_vit.py` | Action-conditioned AdaLN predictor |
| `app/plan_common/datasets/droid_dset.py` | DROID video/state/action loading |
| `app/torchrun.py` | Recursive config overlay and torchrun launch adapter |
| `configs/vjepa_wm/droid_final_sweep/` | Released scientific configurations |
| `infra/skypilot/` | Staging, qualification, smoke, and full-run infrastructure |
| `src/utils/checkpointing.py` | Durable v2 checkpoint objects, aliases, retention, resume |
| `src/utils/dataset_manifest.py` | Dataset identity and runtime binding |
| `src/utils/planning_promotion.py` | Planning registry, metrics, and role promotion |
| `src/utils/qualification.py` | Released-baseline qualification receipts |
| `src/utils/runtime_readiness.py` | Training/recovery readiness receipt |
| `src/utils/mfu.py` | One-step FLOP/MFU diagnostic utility |
| `src/utils/training_telemetry.py` | Cumulative end-to-end MFU, W&B heartbeat, and training-v1 evidence |
| `tests/` | Unit and infrastructure contract tests |

## Local setup

The upstream project uses Python 3.10, Conda for FFmpeg, and `uv` for Python
dependencies:

```bash
conda create -n jepa-wms python=3.10 ffmpeg=7 -c conda-forge -y
conda activate jepa-wms
uv pip install -e '.[dev]'
python setup_macros.py
```

Do not place credentials in `.env`, YAML, shell history, Git, or documentation.
Sky-managed secrets supply `HF_TOKEN`, `WANDB_API_KEY`, and the private native
DINOv3 URL. DROID itself is public; gated Franka/DINO assets require the
authorized identity.

Useful read-only orientation commands:

```bash
git status --short --branch
git remote -v
git log --oneline --decorate --max-count=30
git diff --stat 13cf1d9c7e476f53c17714d2e0f1dc239a883ce0..HEAD
uv run pytest -q tests/utils tests/infra tests/evals
```

Some tests require the ML runtime and are intentionally skipped on a
controller-only machine. A skip is not proof of GPU behavior.

## First task checklist

- Create a dedicated worktree from the immutable onboarding tag.
- State the exact question, experiment, or invariant your change addresses.
- Identify whether the change is scientific, operational, or observational.
- Inspect the current status and decision log before editing.
- Keep a narrow branch; add tests and update the relevant documentation.
- Hand off a commit SHA, test evidence, and any external evidence separately.
