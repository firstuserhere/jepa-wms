"""Fail-closed evidence for the released DROID qualification gate.

The matched training job must not rely on a human remembering that a prior
qualification looked healthy in W&B.  This module combines the independently
verified rollout and planning artifacts into one checksum-protected receipt,
then verifies that receipt against the exact dataset, DINOv3 weights, and Git
revision used by a prospective training job.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

from src.utils.checkpointing import CheckpointManager, CheckpointRef, atomic_json_dump, sha256_file
from src.utils.planning_promotion import (
    canonical_sha256,
    load_complete_planning_result,
    load_planning_evaluation_registry,
)


QUALIFICATION_SCHEMA_VERSION = 1
RELEASED_DROID_REPOSITORY = "facebook/jepa-wms"
RELEASED_DROID_REVISION = "9b9c41ef249466630dbf1a20e78391865d07b3b9"
RELEASED_DROID_FILENAME = "jepa_wm_droid.pth.tar"
RELEASED_DROID_SHA256 = "daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa"
DINOV3_REPOSITORY_REVISION = "54694f7627fd815f62a5dcc82944ffa6153bbb76"
# Canonical CEM-L2 result for the released/best JEPA-WM in Table 1 of
# https://arxiv.org/abs/2512.24497. Promotion still minimizes action error;
# this independent success-rate gate detects evaluation/data drift.
PUBLISHED_DROID_SUCCESS_PERCENT = 48.2
PUBLISHED_DROID_SUCCESS_STD_PERCENT = 1.8
PUBLISHED_DROID_MAX_ABS_Z = 4.0

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class QualificationError(RuntimeError):
    """Raised when released-checkpoint qualification evidence is incomplete."""


def _load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot read qualification input {source}: {error}") from error
    if not isinstance(document, dict):
        raise QualificationError(f"qualification input is not a JSON object: {source}")
    return document


def _validate_sha256(value: str, label: str) -> str:
    digest = str(value).lower()
    if not _SHA256.fullmatch(digest):
        raise QualificationError(f"{label} must be a full lowercase SHA-256 digest")
    return digest


def _validate_git_commit(value: str) -> str:
    commit = str(value).lower()
    if not _GIT_COMMIT.fullmatch(commit):
        raise QualificationError("source Git revision must be an immutable 40-hex commit")
    return commit


def _verified_dataset_manifest(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    manifest = _load_json(path)
    fingerprint = manifest.get("dataset_fingerprint")
    verification = manifest.get("verification", {})
    if not isinstance(fingerprint, str) or not fingerprint:
        raise QualificationError("DROID manifest has no dataset_fingerprint")
    for required in ("complete", "files_checked", "source_listing_matches_staged"):
        if verification.get(required) is not True:
            raise QualificationError(f"DROID manifest verification.{required} is not true")
    return manifest, sha256_file(path)


def _read_identity_file(path: str | os.PathLike[str], label: str) -> str:
    try:
        identity = Path(path).read_text(encoding="utf-8").strip()
    except OSError as error:
        raise QualificationError(f"cannot read {label} identity: {error}") from error
    if not identity or any(character.isspace() for character in identity):
        raise QualificationError(f"{label} identity is empty or malformed")
    return identity


def _verify_rollout_result(
    path: str | os.PathLike[str], *, dataset_fingerprint: str
) -> dict[str, Any]:
    result_path = Path(path).resolve()
    result = _load_json(result_path)
    if result.get("status") != "complete" or result.get("artifact_kind") != "released_rollout_qualification":
        raise QualificationError("released rollout qualification is not complete")
    if result.get("dataset_fingerprint") != dataset_fingerprint:
        raise QualificationError("released rollout qualification used a different dataset fingerprint")
    reference_data = result.get("checkpoint")
    if not isinstance(reference_data, Mapping):
        raise QualificationError("released rollout result has no immutable checkpoint reference")
    reference = CheckpointRef.from_dict(reference_data)
    if reference.sha256 != RELEASED_DROID_SHA256:
        raise QualificationError("released rollout used the wrong released checkpoint")
    manager_root = result_path.parent.parent
    CheckpointManager(manager_root).verify(reference)
    selection = result.get("selection", {})
    if (
        selection.get("mode") != "min"
        or not isinstance(selection.get("metric_value"), (int, float))
        or isinstance(selection.get("metric_value"), bool)
        or int(selection.get("count", 0)) <= 0
    ):
        raise QualificationError("released rollout selection metric is incomplete")
    if not str(selection.get("metric_name", "")).startswith(
        "data_traj/val_rollout/visual_l2_loss/"
    ):
        raise QualificationError("released rollout selected an unexpected metric")
    if not result.get("wandb_run_id"):
        raise QualificationError("released rollout result has no W&B run identity")
    promotion_config_sha256 = result.get("promotion_config_sha256")
    if not isinstance(promotion_config_sha256, str) or not _SHA256.fullmatch(promotion_config_sha256):
        raise QualificationError("released rollout result has no promotion configuration checksum")
    return {
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "checkpoint": reference.to_dict(),
        "promotion_config_sha256": promotion_config_sha256,
        "selection": selection,
        "wandb_run_id": result["wandb_run_id"],
    }


def _verify_planning_registry(path: str | os.PathLike[str]) -> dict[str, Any]:
    registry_path = Path(path).resolve()
    registry = load_planning_evaluation_registry(registry_path)
    if not registry.get("promotion_eval_config_sha256"):
        raise QualificationError("planning registry has no promotion configuration identity")
    qualifying_results = []
    expected_eval_count = 0
    for checkpoint in registry["checkpoints"].values():
        if checkpoint["checkpoint"]["sha256"] != RELEASED_DROID_SHA256:
            raise QualificationError("planning registry contains a different released checkpoint")
        if checkpoint.get("status") != "complete":
            raise QualificationError("planning registry contains an incomplete checkpoint evaluation")
        checkpoint_path = Path(checkpoint["checkpoint"]["path"])
        if sha256_file(checkpoint_path) != RELEASED_DROID_SHA256:
            raise QualificationError("planning checkpoint bytes do not match the released checkpoint")
        for evaluation in checkpoint["evaluations"].values():
            expected_eval_count += 1
            if evaluation.get("status") != "complete":
                raise QualificationError("planning registry contains an incomplete evaluation")
            result = load_complete_planning_result(evaluation["result_path"])
            if result["checkpoint"]["sha256"] != RELEASED_DROID_SHA256:
                raise QualificationError("planning result used the wrong released checkpoint")
            if result["evaluation"]["expected_episode_count"] != 64:
                raise QualificationError("released DROID planning must evaluate exactly 64 episodes")
            if evaluation.get("promotion_eligible"):
                qualifying_results.append(result)
    if expected_eval_count <= 0 or len(qualifying_results) != 1:
        raise QualificationError("planning qualification requires exactly one promotion-eligible result")
    primary = qualifying_results[0]
    if primary["evaluation"]["config_sha256"] != registry["promotion_eval_config_sha256"]:
        raise QualificationError("planning promotion result does not match the registry configuration")
    if (primary["selection"]["metric"], primary["selection"]["mode"]) != (
        "ep_end_dist_xyz",
        "min",
    ):
        raise QualificationError("DROID planning qualification must minimize ep_end_dist_xyz")
    success = primary.get("metrics", {}).get("episode_success")
    if (
        not isinstance(success, (int, float))
        or isinstance(success, bool)
        or not math.isfinite(float(success))
        or not 0.0 <= float(success) <= 1.0
    ):
        raise QualificationError("DROID planning qualification has no valid episode_success")
    success_percent = 100.0 * float(success)
    delta_percent = success_percent - PUBLISHED_DROID_SUCCESS_PERCENT
    absolute_z = abs(delta_percent) / PUBLISHED_DROID_SUCCESS_STD_PERCENT
    published_comparison = {
        "source": "arXiv:2512.24497 Table 1",
        "metric": "episode_success_percent",
        "reported_mean": PUBLISHED_DROID_SUCCESS_PERCENT,
        "reported_std": PUBLISHED_DROID_SUCCESS_STD_PERCENT,
        "observed": success_percent,
        "delta": delta_percent,
        "absolute_z": absolute_z,
        "max_absolute_z": PUBLISHED_DROID_MAX_ABS_Z,
        "within_reproduction_band": absolute_z <= PUBLISHED_DROID_MAX_ABS_Z,
    }
    if not published_comparison["within_reproduction_band"]:
        raise QualificationError(
            "released DROID checkpoint does not reproduce the published planning result: "
            f"observed={success_percent:.3f}%, expected={PUBLISHED_DROID_SUCCESS_PERCENT:.3f}% "
            f"(+/- {PUBLISHED_DROID_MAX_ABS_Z:.1f}*{PUBLISHED_DROID_SUCCESS_STD_PERCENT:.3f}%)"
        )
    return {
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "registry_integrity_sha256": registry["integrity_sha256"],
        "expected_eval_count": expected_eval_count,
        "promotion_eval_config_sha256": registry["promotion_eval_config_sha256"],
        "primary_result_path": primary["evaluation"]["result_path"],
        "primary_result_integrity_sha256": primary["integrity_sha256"],
        "selection": primary["selection"],
        "metrics": primary["metrics"],
        "published_comparison": published_comparison,
    }


def publish_released_droid_qualification(
    *,
    rollout_result_path: str | os.PathLike[str],
    planning_registry_path: str | os.PathLike[str],
    dataset_manifest_path: str | os.PathLike[str],
    dinov3_weights_path: str | os.PathLike[str],
    planning_wandb_run_id_path: str | os.PathLike[str],
    source_git_commit: str,
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify all qualification evidence and atomically publish one receipt."""

    source_git_commit = _validate_git_commit(source_git_commit)
    dataset, dataset_manifest_sha256 = _verified_dataset_manifest(dataset_manifest_path)
    dinov3_sha256 = _validate_sha256(sha256_file(dinov3_weights_path), "DINOv3 weights")
    rollout = _verify_rollout_result(
        rollout_result_path,
        dataset_fingerprint=dataset["dataset_fingerprint"],
    )
    planning = _verify_planning_registry(planning_registry_path)
    planning["wandb_run_id"] = _read_identity_file(
        planning_wandb_run_id_path,
        "planning W&B run",
    )
    receipt = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "kind": "released-droid-pipeline-qualification",
        "status": "complete",
        "source_git_commit": source_git_commit,
        "released_checkpoint": {
            "repository": RELEASED_DROID_REPOSITORY,
            "revision": RELEASED_DROID_REVISION,
            "filename": RELEASED_DROID_FILENAME,
            "sha256": RELEASED_DROID_SHA256,
        },
        "dataset": {
            "fingerprint": dataset["dataset_fingerprint"],
            "manifest_path": str(Path(dataset_manifest_path).resolve()),
            "manifest_sha256": dataset_manifest_sha256,
        },
        "encoder": {
            "repository_revision": DINOV3_REPOSITORY_REVISION,
            "weights_path": str(Path(dinov3_weights_path).resolve()),
            "weights_sha256": dinov3_sha256,
        },
        "rollout": rollout,
        "planning": planning,
    }
    receipt["integrity_sha256"] = canonical_sha256(receipt)
    atomic_json_dump(receipt, output_path)
    return receipt


def load_released_droid_qualification(path: str | os.PathLike[str]) -> dict[str, Any]:
    receipt = _load_json(path)
    if receipt.get("schema_version") != QUALIFICATION_SCHEMA_VERSION:
        raise QualificationError("unsupported qualification receipt schema")
    if receipt.get("kind") != "released-droid-pipeline-qualification" or receipt.get("status") != "complete":
        raise QualificationError("released DROID qualification receipt is not complete")
    integrity = receipt.pop("integrity_sha256", None)
    if integrity != canonical_sha256(receipt):
        raise QualificationError("released DROID qualification receipt integrity mismatch")
    receipt["integrity_sha256"] = integrity
    released = receipt.get("released_checkpoint", {})
    if released != {
        "repository": RELEASED_DROID_REPOSITORY,
        "revision": RELEASED_DROID_REVISION,
        "filename": RELEASED_DROID_FILENAME,
        "sha256": RELEASED_DROID_SHA256,
    }:
        raise QualificationError("qualification receipt names an unexpected released checkpoint")
    return receipt


def verify_released_droid_qualification(
    path: str | os.PathLike[str],
    *,
    dataset_manifest_path: str | os.PathLike[str],
    dinov3_weights_path: str | os.PathLike[str],
    source_git_commit: str,
) -> dict[str, Any]:
    """Require a receipt produced by the exact prospective training inputs."""

    receipt = load_released_droid_qualification(path)
    source_git_commit = _validate_git_commit(source_git_commit)
    dataset, dataset_manifest_sha256 = _verified_dataset_manifest(dataset_manifest_path)
    dinov3_sha256 = _validate_sha256(sha256_file(dinov3_weights_path), "DINOv3 weights")
    if receipt["source_git_commit"] != source_git_commit:
        raise QualificationError("qualification and training source Git commits differ")
    if receipt["dataset"] != {
        "fingerprint": dataset["dataset_fingerprint"],
        "manifest_path": str(Path(dataset_manifest_path).resolve()),
        "manifest_sha256": dataset_manifest_sha256,
    }:
        raise QualificationError("qualification and training dataset manifests differ")
    if receipt["encoder"]["repository_revision"] != DINOV3_REPOSITORY_REVISION:
        raise QualificationError("qualification used a different DINOv3 code revision")
    if receipt["encoder"]["weights_sha256"] != dinov3_sha256:
        raise QualificationError("qualification and training DINOv3 weights differ")
    return receipt
