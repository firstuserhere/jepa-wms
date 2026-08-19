# Agent entry point

This repository is a research codebase, not a generic application. Preserve
scientific comparability, provenance, durable recovery, and explicit operator
control ahead of convenience.

## Read before changing anything

Read these files in order:

1. `docs/ONBOARDING.md` — repository identity, setup, and directory map.
2. `docs/RESEARCH_STATUS.md` — dated evidence, completed jobs, and open gates.
3. `docs/ARCHITECTURE.md` — model, data, loss, planning, and checkpoint flows.
4. `docs/OPERATIONS.md` — Pantheon/SkyPilot rules and launch gates.
5. `docs/WORKTREES.md` — one-agent-per-worktree collaboration workflow.
6. `docs/DECISIONS.md` — decisions that must not be silently reversed.

The detailed infrastructure runbook remains `infra/skypilot/README.md`.

## Source and branch discipline

- Meta upstream is `https://github.com/facebookresearch/jepa-wms.git`.
- The research fork is `https://github.com/firstuserhere/jepa-wms.git`.
- The integration branch is `codex/droid-dinov3-research-infra`.
- The immutable onboarding base is the annotated tag
  `jepawm-droid-infra-2026-08-19`. Resolve and record its full commit SHA before
  starting work or launching compute.
- Never launch from a moving branch, uncommitted tree, or local workdir upload.
  Sky jobs receive a clean GitHub URL and full 40-hex commit.
- Every agent gets its own worktree and branch. Never let two agents write the
  same worktree or branch. Integrate reviewed commits; do not copy an entire
  dirty tree over another agent's work.
- Do not rewrite published history, force-push shared branches, or mutate the
  onboarding tag.

## Research invariants

- The released DROID baseline config is the scientific source of truth:
  `configs/vjepa_wm/droid_final_sweep/droid_4fpcs_fps4_r256_dv3vitl_asp1_pred_AdaLN_depth12_noprop_repro_2roll_4n.yaml`.
- Operational changes belong in overlays, launch tooling, manifests, telemetry,
  checkpointing, or evaluation infrastructure. A scientific hyperparameter
  change is a new experiment and must be named and documented as such.
- The DINOv3 encoder is the native ViT-L/16 artifact, not a converted or
  similarly named substitute. Its code revision and full weight SHA are
  fail-closed provenance fields.
- Exact resume restores the complete mathematical trajectory. Continued
  pretraining uses explicit `fork` mode and records the parent lineage; it is
  not a permissive resume.
- DROID planning selection uses XYZ endpoint error and the derived DROID Action
  Score. The offline dummy environment's success flag is not a valid metric.
- W&B is required for every training run. MFU must be model-aware,
  end-to-end, and continuously visible; never invent MFU for evaluation work.

## External-action safety

Without explicit user authorization for the specific action, do not:

- submit, resume, cancel, or reprioritize a Sky job;
- create, delete, resize, or write to a volume;
- change a managed secret or reveal a token/private DINO URL;
- publish a Git branch, tag, release, PR, checkpoint, or model artifact;
- use `p0` or assume a previous priority approval applies to a later launch.

Read-only status, logs, manifests, Git inspection, and dry-run validation are
allowed. Never print secret values. The two JEPA-WM volumes are durable research
assets; mount the DROID volume read-only for training and let rank zero alone
publish shared checkpoints.

## Definition of done

A change is not complete merely because it imports locally. In proportion to
its scope, provide:

- focused unit tests and `git diff --check`;
- deterministic config/provenance tests for research behavior;
- Sky schema/dry-run validation for infrastructure changes;
- a real multi-node smoke for distributed/storage/recovery claims;
- W&B evidence for telemetry claims;
- a dated update to `docs/RESEARCH_STATUS.md` when external evidence changes.

Do not edit a historical result to look current. Add a new dated observation
and keep the old job ID, immutable Git SHA, artifact checksum, and outcome.
