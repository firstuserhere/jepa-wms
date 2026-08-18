#!/usr/bin/env python3
"""Render a checked SkyPilot task for the repository's Kubernetes H200 pool.

The scientific command remains in the source task.  This renderer changes only
provider/storage plumbing: GCP bucket mounts become named RWX PVC mounts, and
provider-specific resource selectors become the known Kubernetes context.
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
CHECKPOINT_MOUNT = "/mnt/jepawm-checkpoints"


def _validate_volume_name(value: str) -> str:
    if not VOLUME_NAME.fullmatch(value):
        raise ValueError(
            "Sky volume names must be lowercase DNS-style identifiers of at most 63 characters"
        )
    return value


def render_k8s_task(
    document: dict[str, Any],
    *,
    droid_volume: str | None,
    checkpoint_volume: str | None,
    context: str,
) -> dict[str, Any]:
    rendered = dict(document)
    resources = dict(rendered.get("resources", {}))
    resources.pop("cloud", None)
    resources.pop("use_spot", None)
    resources.pop("network_tier", None)
    resources.pop("disk_tier", None)
    resources["infra"] = f"k8s/{context}"
    # On this Sky build Kubernetes disk_size requests pod ephemeral storage.
    # Dataset/checkpoint bytes live on PVCs, so only setup/cache headroom is needed.
    resources["disk_size"] = 100
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
    if CHECKPOINT_MOUNT in source_mounts or CHECKPOINT_MOUNT in volumes:
        if not checkpoint_volume:
            raise ValueError("This task requires --checkpoint-volume")
        volumes[CHECKPOINT_MOUNT] = _validate_volume_name(checkpoint_volume)
    if volumes:
        rendered["volumes"] = volumes

    envs = dict(rendered.get("envs", {}) or {})
    envs.pop("DROID_STORE_URI", None)
    envs.pop("CHECKPOINT_STORE_URI", None)
    if droid_volume:
        envs["DROID_VOLUME_NAME"] = droid_volume
    if DATASET_MOUNT in volumes:
        envs["JEPAWM_STORAGE_BACKEND"] = "pvc"
    if CHECKPOINT_MOUNT in volumes:
        envs["DINOV3_WEIGHTS_SOURCE_PATH"] = (
            f"{CHECKPOINT_MOUNT}/artifacts/dinov3/"
            "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )
        envs["DINOV3_WEIGHTS_URI"] = ""
    rendered["envs"] = envs
    rendered["api_server_access"] = False

    kubernetes_config = dict(rendered.get("config", {}).get("kubernetes", {}) or {})
    kubernetes_config.setdefault("provision_timeout", 3600)
    config = dict(rendered.get("config", {}) or {})
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
    )
    atomic_yaml_dump(rendered, args.output)


if __name__ == "__main__":
    main()
