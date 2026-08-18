#!/usr/bin/env python3
"""Zero-mutation validation for the JEPA-WM SkyPilot launch bundle."""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml

INFRA_DIR = Path(__file__).resolve().parent
REPO_ROOT = INFRA_DIR.parent.parent
TASK_FILES = (
    "distributed_smoke.yaml",
    "droid_stage.yaml",
    "droid_train_smoke.yaml",
    "droid_qualify_released.yaml",
    "droid_train_full.yaml",
)
TRAINING_TASKS = (
    "distributed_smoke.yaml",
    "droid_train_smoke.yaml",
    "droid_qualify_released.yaml",
    "droid_train_full.yaml",
)
REQUIRED_DISTRIBUTED_ENVS = {
    "NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "PYTHONUNBUFFERED": "1",
}


def _assert_managed_secrets(document: dict, filename: str) -> None:
    expected = {"secrets:HF_TOKEN", "secrets:WANDB_API_KEY"}
    actual = set(document.get("secrets", []))
    if actual != expected:
        raise AssertionError(
            f"{filename}: expected managed secret references {sorted(expected)}, found {sorted(actual)}"
        )


def validate_tasks() -> None:
    import sky

    for filename in TASK_FILES:
        task_path = INFRA_DIR / filename
        document = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        if "priority" in document or "priority_class" in document:
            raise AssertionError(f"{filename}: priority must remain an explicit launch CLI flag")
        resources = document["resources"]
        if "local_disk" in resources:
            raise AssertionError(f"{filename}: GCP H200 tasks must not request unsupported local_disk")
        if resources.get("job_recovery") is None:
            raise AssertionError(f"{filename}: managed job recovery is required")
        if filename in TRAINING_TASKS:
            _assert_managed_secrets(document, filename)
            for key, expected in REQUIRED_DISTRIBUTED_ENVS.items():
                if document.get("envs", {}).get(key) != expected:
                    raise AssertionError(f"{filename}: required distributed env {key}={expected}")
        elif document.get("secrets") != ["secrets:HF_TOKEN"]:
            raise AssertionError(f"{filename}: staging should reference only managed HF_TOKEN")
        sky.Task.from_yaml(task_path)

    full = yaml.safe_load((INFRA_DIR / "droid_train_full.yaml").read_text(encoding="utf-8"))
    if full["num_nodes"] != 4 or full["resources"]["accelerators"] != "H200:8":
        raise AssertionError("Full task must request exactly 4 nodes x H200:8")
    if full["resources"].get("network_tier") != "best":
        raise AssertionError("Full task must request network_tier: best")
    if full["resources"].get("disk_size", 0) < 3000:
        raise AssertionError("Full task disk must accommodate the 2.5 TB DROID read-through cache")
    full_run = full["run"]
    first_torchrun = full_run.index("torchrun")
    if full_run.index("qualification_receipt.py verify") >= first_torchrun:
        raise AssertionError("Full task must verify released qualification before torchrun")
    if full_run.index("runtime_readiness.py verify") >= first_torchrun:
        raise AssertionError("Full task must verify both runtime smokes before torchrun")
    mounts = full["file_mounts"]
    if mounts["/mnt/jepawm-datasets"].get("type") != "DATASET_RO":
        raise AssertionError("DROID must be mounted read-only")
    if mounts["/mnt/jepawm-checkpoints"].get("type") != "MODEL_CHECKPOINT_RW":
        raise AssertionError("Checkpoints must use a read-write checkpoint mount")

    distributed_smoke = yaml.safe_load(
        (INFRA_DIR / "distributed_smoke.yaml").read_text(encoding="utf-8")
    )
    if distributed_smoke.get("num_nodes") != 2:
        raise AssertionError("Distributed smoke must exercise cross-node NCCL on exactly two nodes")
    if "${SKYPILOT_TASK_ID:?" not in distributed_smoke.get("run", ""):
        raise AssertionError("Distributed smoke durable state must be isolated by managed task ID")
    if "--expected-num-nodes 2" not in distributed_smoke.get("run", ""):
        raise AssertionError("Distributed smoke must prove that its ranks span two nodes")

    qualification = yaml.safe_load(
        (INFRA_DIR / "droid_qualify_released.yaml").read_text(encoding="utf-8")
    )
    if qualification.get("num_nodes") != 4:
        raise AssertionError("Released rollout qualification must match the 32-rank full-run topology")

    stage = yaml.safe_load((INFRA_DIR / "droid_stage.yaml").read_text(encoding="utf-8"))
    stage_mount = stage["file_mounts"]["/mnt/jepawm-datasets"]
    if (
        stage_mount.get("mode") != "MOUNT"
        or stage_mount.get("config", {}).get("mount", {}).get("read_only") is not True
    ):
        raise AssertionError("DROID staging must verify through a close-to-open read-only target mount")
    stage_run = stage["run"]
    if "build_droid_manifest.py" not in stage_run or "--no-verify-files" in stage_run:
        raise AssertionError("DROID staging must run full per-episode file verification")
    if "--bind-franka-manifest" not in stage_run or "--franka-source-root" not in stage_run:
        raise AssertionError("DROID staging must byte-verify and bind Franka_hf into the combined manifest")
    if "6116f042ae7ae4c8e3f1fd2f194f432615664182" not in stage_run:
        raise AssertionError("Franka_hf Hub dataset revision is not pinned")

    smoke = yaml.safe_load((INFRA_DIR / "droid_train_smoke.yaml").read_text(encoding="utf-8"))
    if smoke.get("num_nodes") != 2:
        raise AssertionError("Real DROID training recovery smoke must exercise two-node DDP")
    smoke_run = smoke["run"]
    if "--checkpoint-recovery-smoke" not in smoke_run or "--smoke-verify-epoch 2" not in smoke_run:
        raise AssertionError("Training smoke must prove strict v2 checkpoint recovery and forward progress")
    if 42 not in smoke["resources"]["job_recovery"].get("recover_on_exit_codes", []):
        raise AssertionError("Training smoke controlled stop must be recoverable")
    if "runtime_readiness.py publish" not in smoke_run:
        raise AssertionError("Training smoke must publish the combined runtime-readiness receipt")

    qualify = yaml.safe_load((INFRA_DIR / "droid_qualify_released.yaml").read_text(encoding="utf-8"))
    qualify_run = qualify["run"]
    if "9b9c41ef249466630dbf1a20e78391865d07b3b9" not in qualify_run:
        raise AssertionError("Released JEPA-WM Hub revision is not pinned")
    if "daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa" not in qualify_run:
        raise AssertionError("Released DROID checkpoint SHA-256 is not pinned")
    if "hf download facebook/jepa-wms" in qualify_run:
        raise AssertionError("Qualification must consume the pre-staged immutable released checkpoint")
    if "$JEPAWM_CKPT/artifacts/releases/jepa_wm_droid-${release_sha256}.pth.tar" not in qualify_run:
        raise AssertionError("Qualification does not consume the shared released-checkpoint artifact")


def validate_kubernetes_profile() -> None:
    import sky

    sys.path.insert(0, str(REPO_ROOT))
    from infra.skypilot.render_k8s_task import render_k8s_task

    for filename in TASK_FILES:
        document = yaml.safe_load((INFRA_DIR / filename).read_text(encoding="utf-8"))
        source_mounts = document.get("file_mounts", {})
        rendered = render_k8s_task(
            document,
            droid_volume=(
                "jepawm-droid-test" if "/mnt/jepawm-datasets" in source_mounts else None
            ),
            checkpoint_volume=(
                "jepawm-checkpoints-test"
                if "/mnt/jepawm-checkpoints" in source_mounts
                else None
            ),
            context="Skypilot",
        )
        if rendered["resources"].get("infra") != "k8s/Skypilot":
            raise AssertionError(f"{filename}: Kubernetes profile did not pin the known context")
        if "file_mounts" in rendered:
            raise AssertionError(f"{filename}: Kubernetes profile retained a cloud-bucket mount")
        sky.Task.from_yaml_config(copy.deepcopy(rendered))

    dinov3_stage = yaml.safe_load((INFRA_DIR / "dinov3_stage_k8s.yaml").read_text(encoding="utf-8"))
    dinov3_rendered = render_k8s_task(
        dinov3_stage,
        droid_volume=None,
        checkpoint_volume="jepawm-checkpoints-test",
        context="Skypilot",
    )
    if dinov3_rendered.get("secrets") != ["secrets:DINOV3_WEIGHTS_URL"]:
        raise AssertionError("Native DINOv3 staging URL must remain a managed secret")
    sky.Task.from_yaml_config(copy.deepcopy(dinov3_rendered))

    released_stage = yaml.safe_load(
        (INFRA_DIR / "released_droid_stage_k8s.yaml").read_text(encoding="utf-8")
    )
    released_rendered = render_k8s_task(
        released_stage,
        droid_volume=None,
        checkpoint_volume="jepawm-checkpoints-test",
        context="Skypilot",
    )
    if released_rendered.get("secrets") != ["secrets:HF_TOKEN"]:
        raise AssertionError("Released DROID staging must use only the managed HF token")
    released_run = released_rendered.get("run", "")
    if "9b9c41ef249466630dbf1a20e78391865d07b3b9" not in released_run:
        raise AssertionError("Released DROID staging Hub revision is not pinned")
    if "daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa" not in released_run:
        raise AssertionError("Released DROID staging checksum is not pinned")
    sky.Task.from_yaml_config(copy.deepcopy(released_rendered))

    expected_volumes = {
        "droid_raw_k8s.yaml": ("8Ti", "ReadWriteMany"),
        "checkpoints_k8s.yaml": ("2Ti", "ReadWriteMany"),
    }
    for filename, (size, access_mode) in expected_volumes.items():
        volume = yaml.safe_load((INFRA_DIR / "volumes" / filename).read_text(encoding="utf-8"))
        if volume.get("type") != "k8s-pvc" or volume.get("infra") != "k8s/Skypilot":
            raise AssertionError(f"{filename}: expected a PVC on the known Kubernetes context")
        if volume.get("size") != size or volume.get("config", {}).get("access_mode") != access_mode:
            raise AssertionError(f"{filename}: expected {size} RWX capacity")


def validate_overlays() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from app.torchrun import load_config

    quality = load_config(INFRA_DIR / "droid_dinov3_quality_overlay.yaml")
    if quality["model"]["visual_encoder"]["enc_version"] != "dinov3_vitl16":
        raise AssertionError("Quality overlay must inherit DINOv3 ViT-L/16")
    if quality["model"]["predictor"]["pred_depth"] != 12:
        raise AssertionError("Quality overlay must inherit predictor depth 12")
    if quality["optimization"]["transition_model"]["num_epochs"] != 315:
        raise AssertionError("Quality overlay unexpectedly changed the matched training duration")
    if quality["data"]["loader"]["batch_size"] != 8:
        raise AssertionError("Quality overlay unexpectedly changed the per-rank batch size")
    if quality["data"]["loader"]["persistent_workers"] is not False:
        raise AssertionError("Strict continuation requires persistent_workers: false")
    if quality["logging"]["wandb"].get("required") is not True:
        raise AssertionError("W&B must be required")
    if quality["logging"]["wandb"].get("required_online") is not True:
        raise AssertionError("W&B must be online, not merely locally buffered")
    checkpointing = quality["checkpointing"]
    if not checkpointing.get("enabled") or not checkpointing.get("strict_continuation"):
        raise AssertionError("Strict resumable checkpointing must be enabled")
    if not checkpointing.get("require_dataset_manifest"):
        raise AssertionError("DROID manifest must be required")
    if not checkpointing.get("require_source_inventory"):
        raise AssertionError("DROID manifest must exactly match the live official trajectory inventory")
    if not checkpointing.get("require_git_manifest"):
        raise AssertionError("Complete Git source provenance must be required")
    if checkpointing.get("keep_recent") != 3:
        raise AssertionError("Quality training must retain three independent recent fallbacks")
    expected_promotion = {
        "enabled": True,
        "every_epochs": 1,
        "loader_index": 0,
        "sampler_epoch": 0,
        "seed": 50234,
        "max_batches": 64,
        "horizons": [1, 2, 3, 4],
        "aggregation": "mean",
        "metric_base": "data_traj/val_rollout/visual_l2_loss",
    }
    if checkpointing.get("rollout_promotion") != expected_promotion:
        raise AssertionError("best_rollout promotion corpus is not the fixed Franka_hf suite")
    expected_revision = "54694f7627fd815f62a5dcc82944ffa6153bbb76"
    if quality["model"]["visual_encoder"].get("pretrain_enc_repo_revision") != expected_revision:
        raise AssertionError("DINOv3 repository revision is not pinned")
    encoder = quality["model"]["visual_encoder"]
    if encoder.get("pretrain_enc_sha256") != "${DINOV3_WEIGHTS_SHA256}":
        raise AssertionError("DINOv3 weights must be bound to a full caller-supplied SHA-256")
    if quality["folder"] != "${JEPAWM_LOGS}/${JEPAWM_RUN_ID}" or quality["checkpoint_folder"] != (
        "${JEPAWM_CKPT}/${JEPAWM_RUN_ID}"
    ):
        raise AssertionError("Quality outputs must be isolated by JEPAWM_RUN_ID")

    load_config(INFRA_DIR / "droid_dinov3_smoke_overlay.yaml")
    released_rollout = load_config(INFRA_DIR / "released_rollout_overlay.yaml")
    released_planning = load_config(INFRA_DIR / "released_planning_overlay.yaml")
    if released_rollout["meta"].get("rollout_only_eval_mode") is not True:
        raise AssertionError("Released rollout qualification must use the explicit one-shot mode")
    if released_rollout["meta"].get("light_eval_only_mode") is not True:
        raise AssertionError("Released rollout qualification must remain gradient-free")
    released_schedule = released_rollout["optimization"]["transition_model"]
    if released_schedule.get("num_epochs") != 1 or released_schedule.get("iterations_per_epoch") != 1:
        raise AssertionError("Released rollout qualification must execute exactly one evaluation epoch")
    if released_planning["evals"].get("promotion_eval_index", 0) != 0:
        raise AssertionError("Released planning qualification must promote only canonical CEM L2")
    if len(released_planning["evals"].get("eval_cfg_paths", [])) != 8:
        raise AssertionError("Released planning qualification must retain all eight released diagnostics")


def validate_local_syntax() -> None:
    if not (REPO_ROOT / "uv.lock").is_file():
        raise AssertionError("uv.lock is required for frozen reproducible worker installs")
    smoke_setup = (INFRA_DIR / "setup_smoke.sh").read_text(encoding="utf-8")
    if '"numpy==2.2.6"' not in smoke_setup:
        raise AssertionError(
            "Distributed smoke setup must install pinned NumPy for torch all_gather_object"
        )
    launch_text = (INFRA_DIR / "launch.sh").read_text(encoding="utf-8")
    if 'priority_class="p1"' not in launch_text:
        raise AssertionError("P1 must remain the default launch priority")
    if '--priority "$priority_class"' not in launch_text:
        raise AssertionError("Launch priority must remain an explicit, validated CLI setting")
    for required_flag in ('--git-url "$git_url"', '--git-ref "$git_ref"', '--workspace "$sky_workspace"'):
        if required_flag not in launch_text:
            raise AssertionError(f"Actual launches must provide {required_flag}")
    k8s_launch_text = (INFRA_DIR / "launch_k8s.sh").read_text(encoding="utf-8")
    if 'priority_class="p1"' not in k8s_launch_text:
        raise AssertionError("Kubernetes launches must default to p1")
    if '--priority "$priority_class"' not in k8s_launch_text:
        raise AssertionError("Kubernetes launch priority must be an explicit, validated CLI setting")
    if 'check -w "$sky_workspace" -o json' not in k8s_launch_text:
        raise AssertionError("Kubernetes launches must verify workspace capability")
    scripts = sorted(INFRA_DIR.glob("*.sh"))
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
    for filename in (*TASK_FILES, "dinov3_stage_k8s.yaml", "released_droid_stage_k8s.yaml"):
        document = yaml.safe_load((INFRA_DIR / filename).read_text(encoding="utf-8"))
        for block_name in ("setup", "run"):
            block = document.get(block_name)
            if block:
                subprocess.run(
                    ["bash", "-n"],
                    input=block,
                    text=True,
                    check=True,
                )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(REPO_ROOT / "app" / "torchrun.py"),
            str(INFRA_DIR / "stage_droid.py"),
            str(INFRA_DIR / "render_k8s_task.py"),
            str(INFRA_DIR / "qualification_receipt.py"),
            str(INFRA_DIR / "runtime_readiness.py"),
            str(REPO_ROOT / "src" / "utils" / "qualification.py"),
            str(REPO_ROOT / "src" / "utils" / "runtime_readiness.py"),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    validate_tasks()
    validate_kubernetes_profile()
    validate_overlays()
    validate_local_syntax()
    if not args.quiet:
        print(
            "Preflight passed: GCP and Kubernetes task profiles, two PVC templates, "
            "4 overlays, SkyPilot schema, launch invariants, and local syntax."
        )
        print("No cloud resources, storage, secrets, or jobs were accessed.")


if __name__ == "__main__":
    main()
