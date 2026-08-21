"""Checksum-protected evidence that distributed recovery is ready for a full run."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.utils.checkpointing import CheckpointManager, atomic_json_dump, sha256_file, validate_checkpoint_v2
from src.utils.planning_promotion import canonical_sha256
from src.utils.training_telemetry import validate_telemetry_snapshot_document


RUNTIME_READINESS_SCHEMA_VERSION = 1
DINOV3_REPOSITORY_REVISION = "54694f7627fd815f62a5dcc82944ffa6153bbb76"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class RuntimeReadinessError(RuntimeError):
    """Raised when smoke evidence is incomplete or incompatible."""


def _load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeReadinessError(f"cannot read runtime evidence {source}: {error}") from error
    if not isinstance(document, dict):
        raise RuntimeReadinessError(f"runtime evidence is not a JSON object: {source}")
    return document


def _load_integrity_document(path: str | os.PathLike[str], kind: str) -> dict[str, Any]:
    document = _load_json(path)
    stored = document.pop("integrity_sha256", None)
    if stored != canonical_sha256(document):
        raise RuntimeReadinessError(f"{kind} evidence integrity checksum mismatch")
    document["integrity_sha256"] = stored
    if document.get("schema_version") != 1 or document.get("kind") != kind:
        raise RuntimeReadinessError(f"unexpected {kind} evidence schema")
    return document


def _validate_git_commit(value: str) -> str:
    commit = str(value).lower()
    if not _GIT_COMMIT.fullmatch(commit):
        raise RuntimeReadinessError("runtime readiness requires an immutable 40-hex Git commit")
    return commit


def _verified_dataset(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    manifest = _load_json(path)
    verification = manifest.get("verification", {})
    if not manifest.get("dataset_fingerprint"):
        raise RuntimeReadinessError("DROID manifest has no dataset fingerprint")
    for key in ("complete", "files_checked", "source_listing_matches_staged"):
        if verification.get(key) is not True:
            raise RuntimeReadinessError(f"DROID manifest verification.{key} is not true")
    return manifest, sha256_file(path)


def _verify_distributed_smoke(path: str | os.PathLike[str], source_git_commit: str) -> dict[str, Any]:
    state = _load_integrity_document(path, "distributed-recovery-smoke")
    if state.get("phase") != "complete" or state.get("recovered_from_controlled_failure") is not True:
        raise RuntimeReadinessError("distributed smoke did not complete a controlled managed recovery")
    if state.get("world_size") != 16 or state.get("observed_node_count") != 2:
        raise RuntimeReadinessError("distributed smoke must prove exactly two nodes and sixteen ranks")
    evidence = state.get("rank_evidence")
    if not isinstance(evidence, list) or len(evidence) != 16:
        raise RuntimeReadinessError("distributed smoke rank evidence is incomplete")
    if sorted(item.get("rank") for item in evidence) != list(range(16)):
        raise RuntimeReadinessError("distributed smoke rank identities are incomplete")
    if len({item.get("hostname") for item in evidence}) != 2:
        raise RuntimeReadinessError("distributed smoke did not observe two distinct nodes")
    if any("h200" not in str(item.get("gpu_name", "")).lower() for item in evidence):
        raise RuntimeReadinessError("distributed smoke did not run every rank on an H200")
    if state.get("source_git_commit") != source_git_commit:
        raise RuntimeReadinessError("distributed smoke used a different source commit")
    if not state.get("wandb_run_id"):
        raise RuntimeReadinessError("distributed smoke has no W&B run identity")
    return state


def _verify_training_smoke(
    state_path: str | os.PathLike[str],
    checkpoint_root: str | os.PathLike[str],
    *,
    source_git_commit: str,
    dataset_fingerprint: str,
    dinov3_weights_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _load_integrity_document(state_path, "training-checkpoint-recovery-smoke")
    if state.get("phase") != "complete" or state.get("strict_resume_verified") is not True:
        raise RuntimeReadinessError("real training smoke did not complete strict checkpoint recovery")
    first = state.get("first_checkpoint", {})
    final = state.get("final_checkpoint", {})
    if (first.get("epoch"), first.get("global_update")) != (1, 8):
        raise RuntimeReadinessError("training smoke first checkpoint is not the fixed epoch-1/update-8 boundary")
    if (final.get("epoch"), final.get("global_update")) != (2, 16):
        raise RuntimeReadinessError("training smoke did not advance to fixed epoch-2/update-16")
    for key, expected in (
        ("checkpoint_schema_version", 2),
        ("source_git_commit", source_git_commit),
        ("dataset_fingerprint", dataset_fingerprint),
        ("encoder_repo_revision", DINOV3_REPOSITORY_REVISION),
        ("encoder_weights_sha256", dinov3_weights_sha256),
    ):
        if first.get(key) != expected or final.get(key) != expected:
            raise RuntimeReadinessError(f"training smoke checkpoint {key} is incompatible")
    if not first.get("wandb_run_id") or first["wandb_run_id"] != final.get("wandb_run_id"):
        raise RuntimeReadinessError("training smoke did not preserve its W&B identity")
    if first.get("resume_contract_sha256") != final.get("resume_contract_sha256"):
        raise RuntimeReadinessError("training smoke changed its mathematical resume contract")
    for label, checkpoint_evidence in (("first", first), ("final", final)):
        if checkpoint_evidence.get("epoch_boundary_only") is not True:
            raise RuntimeReadinessError(f"training smoke {label} checkpoint was not epoch-boundary-only")
        elapsed = checkpoint_evidence.get("epoch_boundary_seconds")
        budget = checkpoint_evidence.get("checkpoint_loss_budget_seconds")
        if (
            not isinstance(elapsed, (int, float))
            or not isinstance(budget, (int, float))
            or elapsed <= 0
            or budget <= 0
            or elapsed > budget
            or budget > 300
            or checkpoint_evidence.get("checkpoint_within_loss_budget") is not True
        ):
            raise RuntimeReadinessError(
                f"training smoke {label} checkpoint does not prove the five-minute work-loss budget"
            )
        if not _SHA256.fullmatch(str(checkpoint_evidence.get("telemetry_snapshot_sha256", ""))):
            raise RuntimeReadinessError(f"training smoke {label} checkpoint has no telemetry snapshot digest")
        telemetry_path = checkpoint_evidence.get("telemetry_snapshot_path")
        if not telemetry_path or sha256_file(telemetry_path) != checkpoint_evidence["telemetry_snapshot_sha256"]:
            raise RuntimeReadinessError(f"training smoke {label} telemetry snapshot changed")
        telemetry_document = _load_json(telemetry_path)
        heartbeat_at = telemetry_document.get("summary", {}).get(
            "pantheon_telemetry/heartbeat_at"
        )
        try:
            validate_telemetry_snapshot_document(
                telemetry_document,
                expected_active_step=int(checkpoint_evidence["global_update"]),
                now=float(heartbeat_at),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeReadinessError(
                f"training smoke {label} telemetry snapshot is invalid: {error}"
            ) from error
        if checkpoint_evidence.get("telemetry_wandb_url") is None:
            raise RuntimeReadinessError(f"training smoke {label} has no canonical W&B URL")

    manager = CheckpointManager(checkpoint_root)
    alias = manager.read_alias("latest", verify=True)
    if alias is None:
        raise RuntimeReadinessError("training smoke has no verified latest checkpoint")
    reference = alias["checkpoint"]
    for key in ("object_id", "sha256", "global_update", "epoch"):
        if reference.get(key) != final.get(key):
            raise RuntimeReadinessError(f"training smoke latest checkpoint disagrees on {key}")
    checkpoint = manager.load("latest", map_location="cpu", verify=False)
    validate_checkpoint_v2(checkpoint)
    manifest = checkpoint["manifest"]
    if manifest["git"]["commit"] != source_git_commit:
        raise RuntimeReadinessError("training smoke checkpoint Git manifest differs")
    if manifest["dataset"]["dataset_fingerprint"] != dataset_fingerprint:
        raise RuntimeReadinessError("training smoke checkpoint dataset manifest differs")
    if manifest["encoder"]["weights_sha256"] != dinov3_weights_sha256:
        raise RuntimeReadinessError("training smoke checkpoint DINOv3 identity differs")
    return state, reference


def publish_runtime_readiness(
    *,
    distributed_smoke_path: str | os.PathLike[str],
    training_smoke_path: str | os.PathLike[str],
    training_checkpoint_root: str | os.PathLike[str],
    dataset_manifest_path: str | os.PathLike[str],
    dinov3_weights_path: str | os.PathLike[str],
    source_git_commit: str,
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    source_git_commit = _validate_git_commit(source_git_commit)
    dataset, dataset_manifest_sha256 = _verified_dataset(dataset_manifest_path)
    dinov3_weights_sha256 = sha256_file(dinov3_weights_path)
    if not _SHA256.fullmatch(dinov3_weights_sha256):  # pragma: no cover - sha256_file contract
        raise RuntimeReadinessError("invalid DINOv3 weight checksum")
    distributed = _verify_distributed_smoke(distributed_smoke_path, source_git_commit)
    training, latest_reference = _verify_training_smoke(
        training_smoke_path,
        training_checkpoint_root,
        source_git_commit=source_git_commit,
        dataset_fingerprint=dataset["dataset_fingerprint"],
        dinov3_weights_sha256=dinov3_weights_sha256,
    )
    receipt = {
        "schema_version": RUNTIME_READINESS_SCHEMA_VERSION,
        "kind": "jepa-wm-runtime-readiness",
        "status": "complete",
        "source_git_commit": source_git_commit,
        "dataset": {
            "fingerprint": dataset["dataset_fingerprint"],
            "manifest_path": str(Path(dataset_manifest_path).resolve()),
            "manifest_sha256": dataset_manifest_sha256,
        },
        "encoder": {
            "repository_revision": DINOV3_REPOSITORY_REVISION,
            "weights_sha256": dinov3_weights_sha256,
        },
        "distributed_smoke": {
            "path": str(Path(distributed_smoke_path).resolve()),
            "file_sha256": sha256_file(distributed_smoke_path),
            "integrity_sha256": distributed["integrity_sha256"],
            "run_id": distributed["run_id"],
            "wandb_run_id": distributed["wandb_run_id"],
            "world_size": distributed["world_size"],
            "observed_node_count": distributed["observed_node_count"],
        },
        "training_smoke": {
            "path": str(Path(training_smoke_path).resolve()),
            "file_sha256": sha256_file(training_smoke_path),
            "integrity_sha256": training["integrity_sha256"],
            "wandb_run_id": training["final_checkpoint"]["wandb_run_id"],
            "first_checkpoint": training["first_checkpoint"],
            "final_checkpoint": training["final_checkpoint"],
            "verified_latest_reference": latest_reference,
        },
    }
    receipt["integrity_sha256"] = canonical_sha256(receipt)
    atomic_json_dump(receipt, output_path)
    return receipt


def load_runtime_readiness(path: str | os.PathLike[str]) -> dict[str, Any]:
    receipt = _load_json(path)
    integrity = receipt.pop("integrity_sha256", None)
    if integrity != canonical_sha256(receipt):
        raise RuntimeReadinessError("runtime-readiness receipt integrity mismatch")
    receipt["integrity_sha256"] = integrity
    if (
        receipt.get("schema_version") != RUNTIME_READINESS_SCHEMA_VERSION
        or receipt.get("kind") != "jepa-wm-runtime-readiness"
        or receipt.get("status") != "complete"
    ):
        raise RuntimeReadinessError("runtime-readiness receipt is not complete")
    return receipt


def verify_runtime_readiness(
    path: str | os.PathLike[str],
    *,
    dataset_manifest_path: str | os.PathLike[str],
    dinov3_weights_path: str | os.PathLike[str],
    source_git_commit: str,
) -> dict[str, Any]:
    receipt = load_runtime_readiness(path)
    source_git_commit = _validate_git_commit(source_git_commit)
    dataset, dataset_manifest_sha256 = _verified_dataset(dataset_manifest_path)
    dinov3_weights_sha256 = sha256_file(dinov3_weights_path)
    if receipt["source_git_commit"] != source_git_commit:
        raise RuntimeReadinessError("runtime smokes and full training source commits differ")
    if receipt["dataset"] != {
        "fingerprint": dataset["dataset_fingerprint"],
        "manifest_path": str(Path(dataset_manifest_path).resolve()),
        "manifest_sha256": dataset_manifest_sha256,
    }:
        raise RuntimeReadinessError("runtime smokes and full training dataset manifests differ")
    if receipt["encoder"] != {
        "repository_revision": DINOV3_REPOSITORY_REVISION,
        "weights_sha256": dinov3_weights_sha256,
    }:
        raise RuntimeReadinessError("runtime smokes and full training DINOv3 identities differ")
    for evidence_key in ("distributed_smoke", "training_smoke"):
        evidence = receipt[evidence_key]
        if sha256_file(evidence["path"]) != evidence["file_sha256"]:
            raise RuntimeReadinessError(f"{evidence_key} evidence changed after readiness publication")
    return receipt
