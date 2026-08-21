#!/usr/bin/env python3
"""Render a checked SkyPilot task for Pantheon's Kubernetes H200 pool.

The scientific command remains in the source task.  This renderer changes only
provider/storage plumbing: GCP bucket mounts become named RWX PVC mounts,
Pantheon's canonical shared checkpoint volume is mounted at ``/checkpoints``,
and current modal-skypilot InfiniBand settings are injected for full-node
multi-node jobs. CPU-only tasks retain the dedicated-pool thread limits.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml


VOLUME_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9.]{0,61}[a-z0-9])?")
DATASET_MOUNT = "/mnt/jepawm-datasets"
ARTIFACT_MOUNT = "/mnt/jepawm-checkpoints"
CHECKPOINT_MOUNT = "/checkpoints"
CHECKPOINT_VOLUME = "checkpoints"
IB_ARTIFACT = Path(__file__).with_name("modal_skypilot_ib_32ce2987.yaml")
PANTHEON_USER = "kunvar@pantheon.inc"
CPU_THREAD_ENVS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "POLARS_MAX_THREADS",
)


def _cpu_thread_count(value: Any) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return str(value)
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([1-9][0-9]*)\+?\s*", value)
        if match:
            return match.group(1)
    return None


def _validate_volume_name(value: str) -> str:
    if not VOLUME_NAME.fullmatch(value):
        raise ValueError(
            "Sky volume names must be lowercase DNS-style identifiers of at most 63 characters"
        )
    return value


def _validate_experiment_tag(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Experiment tags must be 1-128 filesystem-safe characters")
    return value


def _load_infiniband_artifact() -> dict[str, Any]:
    artifact = yaml.safe_load(IB_ARTIFACT.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise TypeError(f"Invalid modal-skypilot InfiniBand artifact: {IB_ARTIFACT}")
    commit = artifact.get("source_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("InfiniBand artifact has no immutable modal-skypilot source commit")
    return artifact


def render_k8s_task(
    document: dict[str, Any],
    *,
    droid_volume: str | None,
    checkpoint_volume: str | None,
    context: str,
    experiment_tag: str | None = None,
    pantheon_user: str = PANTHEON_USER,
    training: bool = False,
) -> dict[str, Any]:
    rendered = dict(document)
    resources = dict(rendered.get("resources", {}))
    resources.pop("cloud", None)
    resources.pop("use_spot", None)
    resources.pop("network_tier", None)
    resources.pop("disk_tier", None)
    resources.pop("disk_size", None)
    resources.pop("infra", None)
    cpu_only = not resources.get("accelerators")
    accelerator = str(resources.get("accelerators", ""))
    accelerator_match = re.fullmatch(r"H200:([1-8])", accelerator)
    if not cpu_only and accelerator_match is None:
        raise ValueError("Pantheon GPU tasks must request H200:1 through H200:8")
    if accelerator_match is not None:
        gpu_count = int(accelerator_match.group(1))
        resources["cpus"] = 20 * gpu_count
        resources["memory"] = 230 * gpu_count
    recovery = resources.get("job_recovery")
    if isinstance(recovery, dict):
        recovery = dict(recovery)
        recovery["strategy"] = "FAILOVER"
        resources["job_recovery"] = recovery
    rendered["resources"] = resources

    source_mounts = rendered.pop("file_mounts", {}) or {}
    volumes = dict(rendered.get("volumes", {}) or {})
    if DATASET_MOUNT in source_mounts or DATASET_MOUNT in volumes:
        if not droid_volume:
            raise ValueError("This task requires --droid-volume")
        volumes[DATASET_MOUNT] = _validate_volume_name(droid_volume)
    if ARTIFACT_MOUNT in source_mounts or ARTIFACT_MOUNT in volumes:
        if not checkpoint_volume:
            raise ValueError("This task requires --checkpoint-volume")
        volumes[ARTIFACT_MOUNT] = _validate_volume_name(checkpoint_volume)
    if not cpu_only:
        volumes[CHECKPOINT_MOUNT] = CHECKPOINT_VOLUME
    if volumes:
        rendered["volumes"] = volumes

    envs = dict(rendered.get("envs", {}) or {})
    envs.pop("DROID_STORE_URI", None)
    envs.pop("CHECKPOINT_STORE_URI", None)
    if cpu_only:
        thread_count = _cpu_thread_count(resources.get("cpus"))
        if thread_count is not None:
            for variable in CPU_THREAD_ENVS:
                envs.setdefault(variable, thread_count)
    if droid_volume:
        envs["DROID_VOLUME_NAME"] = droid_volume
    if DATASET_MOUNT in volumes:
        envs["JEPAWM_STORAGE_BACKEND"] = "pvc"
    if ARTIFACT_MOUNT in volumes:
        envs["DINOV3_WEIGHTS_SOURCE_PATH"] = (
            f"{ARTIFACT_MOUNT}/artifacts/dinov3/"
            "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )
        envs["DINOV3_WEIGHTS_URI"] = ""
    if accelerator_match is not None:
        if not isinstance(pantheon_user, str) or "@" not in pantheon_user:
            raise ValueError("Pantheon GPU tasks require the researcher's email identity")
        tag = _validate_experiment_tag(
            experiment_tag or str(envs.get("EXPERIMENT_TAG", "REPLACE_WITH_EXPERIMENT_TAG"))
        )
        envs.update(
            {
                "PANTHEON_USER": pantheon_user,
                "EXPERIMENT_TAG": tag,
                "WANDB_RUN_ID": tag,
                "WANDB_RESUME": "allow",
                "WANDB_MODE": "online",
                "WANDB_PROJECT": str(envs.get("WANDB_PROJECT") or "vjepa_wm"),
                "JEPAWM_PANTHEON_ROOT": f"/checkpoints/{pantheon_user}/{tag}",
            }
        )
        labels = dict(resources.get("labels", {}) or {})
        if training:
            labels.pop("telemetry-contract", None)
            labels.pop("telemetry-opt-out-reason", None)
        else:
            labels["telemetry-contract"] = "opt-out"
            labels["telemetry-opt-out-reason"] = "jepa-wm-qualification-or-infrastructure"
        resources["labels"] = labels

    num_nodes = int(rendered.get("num_nodes", 1))
    config = dict(rendered.get("config", {}) or {})
    kubernetes_config = dict(config.get("kubernetes", {}) or {})
    if num_nodes > 1 and accelerator == "H200:8":
        ib = _load_infiniband_artifact()
        ib_envs = dict(ib["envs"])
        for key, expected in ib_envs.items():
            existing = envs.get(key)
            if existing is not None and str(existing) != str(expected):
                raise ValueError(
                    f"Task overrides modal-skypilot InfiniBand setting {key}: {existing!r} != {expected!r}"
                )
        envs.update(ib_envs)
        envs["PANTHEON_INFINIBAND_SOURCE"] = f"modal-skypilot@{ib['source_commit']}"
        pod_config = dict(ib["pod_config"])
        metadata = dict(pod_config.get("metadata", {}) or {})
        annotations = dict(metadata.get("annotations", {}) or {})
        # Preserve machine-readable provenance in the final submitted object.
        # The value intentionally names the RDMA contract while the current
        # shim requests its HCAs through nvidia.com/hostdev.
        annotations["pantheon.inc/modal-skypilot-ib-source"] = (
            f"rdma/modal-skypilot@{ib['source_commit']}"
        )
        metadata["annotations"] = annotations
        pod_config["metadata"] = metadata
        if kubernetes_config.get("pod_config") not in (None, pod_config):
            raise ValueError("Task pod_config conflicts with current modal-skypilot InfiniBand output")
        kubernetes_config["pod_config"] = pod_config
    rendered["envs"] = envs
    rendered["resources"] = resources
    rendered["api_server_access"] = False

    kubernetes_config.setdefault("provision_timeout", 3600)
    config["kubernetes"] = kubernetes_config
    rendered["config"] = config
    return rendered


def atomic_yaml_dump(document: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(document, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--droid-volume")
    parser.add_argument("--checkpoint-volume")
    parser.add_argument("--context", default="Skypilot")
    parser.add_argument("--experiment-tag")
    parser.add_argument("--pantheon-user", default=PANTHEON_USER)
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise TypeError("Sky task must be a YAML mapping")
    rendered = render_k8s_task(
        document,
        droid_volume=args.droid_volume,
        checkpoint_volume=args.checkpoint_volume,
        context=args.context,
        experiment_tag=args.experiment_tag,
        pantheon_user=args.pantheon_user,
        training=args.training,
    )
    atomic_yaml_dump(rendered, args.output)


if __name__ == "__main__":
    main()
