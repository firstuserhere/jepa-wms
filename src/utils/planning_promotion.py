# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Provenance and promotion utilities for asynchronous planning evaluations.

Planning evaluations can finish in a different order from the checkpoints that
created them.  The helpers in this module therefore never infer a checkpoint
from ``latest``.  A result is tied to an immutable checkpoint path and digest,
and reconciliation considers all complete, comparable results before updating
the ``best_planning`` role.

This module intentionally has no torch or OmegaConf dependency.  It is safe to
use from training launchers, evaluation workers, and small recovery utilities.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import inspect
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable


PLANNING_PROVENANCE_SCHEMA_VERSION = 1
PLANNING_RESULT_SCHEMA_VERSION = 1
ROLE_ALIAS_SCHEMA_VERSION = 1
PLANNING_REGISTRY_SCHEMA_VERSION = 1
PLANNING_RESULT_SUFFIX = ".planning-result.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MUTABLE_CHECKPOINT_NAMES = {"latest", "best", "best-rollout", "best-planning"}
_EVAL_CONFIG_RUNTIME_KEYS = {
    "planning_provenance",
    "folder",
    "checkpoint_folder",
    "work_dir",
    "tag",
    "rank",
    "world_size",
    "device",
    "active_ranks",
    "num_active_gpus",
    "local_seed",
    "task_indices",
    "episodes_per_task",
    "action_dim",
    "frameskip",
    "tasks",
    "use_fsdp",
}


class PlanningProvenanceError(ValueError):
    """Raised when planning provenance or a completed result is inconsistent."""


class PlanningPromotionError(RuntimeError):
    """Raised when planning results cannot be compared or promoted safely."""


class PlanningDrainTimeout(TimeoutError):
    """Raised when a required final planning-evaluation drain times out."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _normalise_sha256(value: str) -> str:
    digest = str(value).lower().removeprefix("sha256:")
    if not _SHA256_RE.fullmatch(digest):
        raise PlanningProvenanceError("checkpoint_sha256 must be a 64-character SHA-256 hex digest")
    return digest


def _safe_id(value: str) -> str:
    safe = _SAFE_ID_RE.sub("-", str(value)).strip("-.")
    if not safe:
        raise PlanningProvenanceError("eval_id must contain at least one filename-safe character")
    return safe


def _plain(value: Any, *, nonfinite_to_none: bool = False) -> Any:
    """Convert common config/metric containers into canonical JSON values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item, nonfinite_to_none=nonfinite_to_none) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item, nonfinite_to_none=nonfinite_to_none) for item in value]
    if isinstance(value, set):
        return sorted(_plain(item, nonfinite_to_none=nonfinite_to_none) for item in value)
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            if nonfinite_to_none:
                return None
            raise PlanningProvenanceError("non-finite values are not valid in canonical planning metadata")
        return value

    # numpy scalar values expose item() and appear in evaluation metrics.  Do
    # not import numpy here merely to identify them.
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is not value:
            return _plain(scalar, nonfinite_to_none=nonfinite_to_none)
    return str(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 digest of a canonical JSON representation."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def evaluation_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash the semantic planning configuration, excluding run identity.

    Checkpoint paths, output paths, epoch tags, and distributed runtime fields
    must not make otherwise identical evaluations incomparable.  Model and
    planner hyperparameters, task definitions, episode counts, seeds, and
    resource topology remain part of the digest.
    """

    payload = _plain(config)
    if not isinstance(payload, dict):
        raise PlanningProvenanceError("planning evaluation config must be a mapping")
    for key in _EVAL_CONFIG_RUNTIME_KEYS:
        payload.pop(key, None)
    model_kwargs = payload.get("model_kwargs")
    if isinstance(model_kwargs, dict):
        model_kwargs.pop("checkpoint", None)
    return canonical_sha256(payload)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 16 * 1024 * 1024) -> str:
    """Stream a file once and return its SHA-256 digest."""

    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        while True:
            chunk = checkpoint_file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def assert_immutable_checkpoint_path(path: str | os.PathLike[str]) -> str:
    """Reject conventional mutable role paths such as ``latest.pth.tar``."""

    if not os.fspath(path):
        raise PlanningProvenanceError("planning checkpoint path is required")
    checkpoint_path = _absolute_path(path)
    basename = Path(checkpoint_path).name.lower()
    tokens = {token for token in re.split(r"[^a-z0-9]+", basename) if token}
    combined_tokens = tokens | {"-".join(pair) for pair in zip(basename.split("-"), basename.split("-")[1:])}
    if _MUTABLE_CHECKPOINT_NAMES & combined_tokens or "latest" in tokens:
        raise PlanningProvenanceError(
            f"planning checkpoint path must be immutable, not a mutable role alias: {checkpoint_path}"
        )
    return checkpoint_path


def default_selection(task_name: str) -> tuple[str, str]:
    """Return the checkpoint-selection metric and direction for a task."""

    if "droid" in str(task_name).lower():
        # The DROID Action Score is derived from terminal xyz error only;
        # orientation and gripper closure must not influence selection.
        return "ep_end_dist_xyz", "min"
    return "episode_success", "max"


def build_planning_provenance(
    eval_config: Mapping[str, Any],
    *,
    checkpoint_id: str,
    checkpoint_sha256: str,
    checkpoint_path: str | os.PathLike[str],
    result_dir: str | os.PathLike[str],
    expected_tasks: Sequence[str],
    expected_episodes_per_task: int | Mapping[str, int],
    task_name: str,
    eval_id: str | None = None,
    checkpoint_step: int | None = None,
    selection_metric: str | None = None,
    selection_mode: str | None = None,
    promotion_eligible: bool = True,
) -> dict[str, Any]:
    """Build strict provenance embedded in a planning job configuration."""

    if not str(checkpoint_id):
        raise PlanningProvenanceError("checkpoint_id is required")
    checkpoint_path = assert_immutable_checkpoint_path(checkpoint_path)
    digest = _normalise_sha256(checkpoint_sha256)
    tasks = [str(task) for task in expected_tasks]
    if not tasks or len(tasks) != len(set(tasks)):
        raise PlanningProvenanceError("expected_tasks must be a non-empty list of unique task names")

    if isinstance(expected_episodes_per_task, Mapping):
        episode_counts = {str(task): int(count) for task, count in expected_episodes_per_task.items()}
    else:
        episode_counts = {task: int(expected_episodes_per_task) for task in tasks}
    if set(episode_counts) != set(tasks) or any(count <= 0 for count in episode_counts.values()):
        raise PlanningProvenanceError("expected episode counts must be positive and specified for every task")

    config_digest = evaluation_config_sha256(eval_config)
    if eval_id is None:
        eval_id = f"{_safe_id(checkpoint_id)}-{config_digest[:16]}"
    else:
        eval_id = _safe_id(eval_id)
    result_path = _absolute_path(Path(result_dir) / f"{eval_id}{PLANNING_RESULT_SUFFIX}")

    default_metric, default_mode = default_selection(task_name)
    metric = selection_metric or default_metric
    mode = selection_mode or default_mode
    if mode not in {"min", "max"}:
        raise PlanningProvenanceError("selection_mode must be 'min' or 'max'")
    if "droid" in str(task_name).lower() and (metric, mode) != ("ep_end_dist_xyz", "min"):
        raise PlanningProvenanceError("DROID planning promotion must minimize ep_end_dist_xyz")

    provenance = {
        "schema_version": PLANNING_PROVENANCE_SCHEMA_VERSION,
        "checkpoint_id": str(checkpoint_id),
        "checkpoint_sha256": digest,
        "checkpoint_path": checkpoint_path,
        "checkpoint_step": None if checkpoint_step is None else int(checkpoint_step),
        "eval_id": eval_id,
        "eval_config_sha256": config_digest,
        "result_path": result_path,
        "task_name": str(task_name),
        "expected_tasks": tasks,
        "expected_task_count": len(tasks),
        "expected_episode_counts": episode_counts,
        "expected_episode_count": sum(episode_counts.values()),
        "selection_metric": metric,
        "selection_mode": mode,
        "promotion_eligible": bool(promotion_eligible),
    }
    validate_planning_provenance(provenance, eval_config=eval_config)
    return provenance


def validate_planning_provenance(
    provenance: Mapping[str, Any], eval_config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate and normalize strict planning provenance."""

    data = _plain(provenance)
    if data.get("schema_version") != PLANNING_PROVENANCE_SCHEMA_VERSION:
        raise PlanningProvenanceError("unsupported planning provenance schema_version")
    required = {
        "checkpoint_id",
        "checkpoint_sha256",
        "checkpoint_path",
        "eval_id",
        "eval_config_sha256",
        "result_path",
        "task_name",
        "expected_tasks",
        "expected_task_count",
        "expected_episode_counts",
        "expected_episode_count",
        "selection_metric",
        "selection_mode",
    }
    missing = sorted(key for key in required if key not in data or data[key] in (None, ""))
    if missing:
        raise PlanningProvenanceError(f"planning provenance is missing required fields: {', '.join(missing)}")

    data["checkpoint_sha256"] = _normalise_sha256(data["checkpoint_sha256"])
    data["checkpoint_path"] = assert_immutable_checkpoint_path(data["checkpoint_path"])
    data["result_path"] = _absolute_path(data["result_path"])
    data["eval_id"] = _safe_id(data["eval_id"])
    if not _SHA256_RE.fullmatch(str(data["eval_config_sha256"])):
        raise PlanningProvenanceError("eval_config_sha256 must be a 64-character SHA-256 hex digest")
    if data["selection_mode"] not in {"min", "max"}:
        raise PlanningProvenanceError("selection_mode must be 'min' or 'max'")
    if "droid" in str(data["task_name"]).lower() and (
        data["selection_metric"],
        data["selection_mode"],
    ) != ("ep_end_dist_xyz", "min"):
        raise PlanningProvenanceError("DROID planning promotion must minimize ep_end_dist_xyz")

    tasks = [str(task) for task in data["expected_tasks"]]
    counts = {str(task): int(count) for task, count in data["expected_episode_counts"].items()}
    if (
        not tasks
        or len(tasks) != len(set(tasks))
        or len(tasks) != int(data["expected_task_count"])
        or set(tasks) != set(counts)
    ):
        raise PlanningProvenanceError("expected task names/counts do not agree")
    if any(count <= 0 for count in counts.values()) or sum(counts.values()) != int(data["expected_episode_count"]):
        raise PlanningProvenanceError("expected episode counts do not agree")

    result_name = Path(data["result_path"]).name
    if data["eval_id"] not in result_name:
        raise PlanningProvenanceError("result_path must be specific to eval_id")
    if eval_config is not None:
        actual_config_digest = evaluation_config_sha256(eval_config)
        if actual_config_digest != data["eval_config_sha256"]:
            raise PlanningProvenanceError(
                "planning evaluation config checksum mismatch: "
                f"expected {data['eval_config_sha256']}, got {actual_config_digest}"
            )
    return data


def verify_planning_checkpoint(
    provenance: Mapping[str, Any], actual_checkpoint_path: str | os.PathLike[str]
) -> str:
    """Verify that an evaluation will load the exact immutable checkpoint."""

    data = validate_planning_provenance(provenance)
    actual_path = Path(actual_checkpoint_path).expanduser()
    declared_path = Path(data["checkpoint_path"]).expanduser()
    if declared_path.is_symlink():
        raise PlanningProvenanceError("planning checkpoint path must not be a mutable symbolic link")
    if not actual_path.is_file():
        raise PlanningProvenanceError(f"planning checkpoint does not exist: {actual_path}")
    if actual_path.resolve() != declared_path.resolve():
        raise PlanningProvenanceError(
            f"model checkpoint path {actual_path.resolve()} does not match provenance {declared_path.resolve()}"
        )
    actual_digest = sha256_file(actual_path)
    if actual_digest != data["checkpoint_sha256"]:
        raise PlanningProvenanceError(
            f"planning checkpoint checksum mismatch: expected {data['checkpoint_sha256']}, got {actual_digest}"
        )
    return actual_digest


def build_complete_planning_result(
    provenance: Mapping[str, Any],
    metrics: Mapping[str, Any],
    observed_episode_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Build a complete result only after all expected tasks were gathered."""

    prov = validate_planning_provenance(provenance)
    observed = {str(task): int(count) for task, count in observed_episode_counts.items()}
    expected = {str(task): int(count) for task, count in prov["expected_episode_counts"].items()}
    if observed != expected:
        raise PlanningProvenanceError(
            f"planning result is incomplete: expected episodes {expected}, observed {observed}"
        )

    clean_metrics = _plain(metrics, nonfinite_to_none=True)
    metric_name = prov["selection_metric"]
    metric_value = clean_metrics.get(metric_name)
    if not isinstance(metric_value, (int, float)) or isinstance(metric_value, bool):
        raise PlanningProvenanceError(f"selection metric {metric_name!r} is missing or non-finite")
    if "droid" in str(prov["task_name"]).lower():
        xyz_error = float(clean_metrics["ep_end_dist_xyz"])
        # Paper metric: retain xyz error for deterministic ranking because the
        # clipped score ties every model at zero once E >= 0.1.
        clean_metrics["droid_action_score"] = 800.0 * (0.1 - xyz_error) if xyz_error < 0.1 else 0.0

    result = {
        "schema_version": PLANNING_RESULT_SCHEMA_VERSION,
        "kind": "planning-evaluation-result",
        "status": "complete",
        "completed_at": _utc_now(),
        "checkpoint": {
            "id": prov["checkpoint_id"],
            "sha256": prov["checkpoint_sha256"],
            "path": prov["checkpoint_path"],
            "step": prov.get("checkpoint_step"),
        },
        "evaluation": {
            "id": prov["eval_id"],
            "config_sha256": prov["eval_config_sha256"],
            "result_path": prov["result_path"],
            "task_name": prov["task_name"],
            "expected_tasks": prov["expected_tasks"],
            "expected_task_count": prov["expected_task_count"],
            "expected_episode_counts": expected,
            "expected_episode_count": prov["expected_episode_count"],
            "observed_tasks": list(observed),
            "observed_task_count": len(observed),
            "observed_episode_counts": observed,
            "observed_episode_count": sum(observed.values()),
            "promotion_eligible": prov.get("promotion_eligible", True),
        },
        "selection": {
            "metric": metric_name,
            "mode": prov["selection_mode"],
            "value": float(metric_value),
        },
        "metrics": clean_metrics,
    }
    result["integrity_sha256"] = canonical_sha256(result)
    return result


def _atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(_plain(payload), output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some object-store FUSE implementations do not support fsync on
            # directories.  The file was still atomically replaced and fsynced.
            pass
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
    return str(destination)


def write_complete_planning_result(result: Mapping[str, Any]) -> str:
    """Atomically publish a checksum-protected, complete planning result."""

    validated = validate_complete_planning_result(result)
    return _atomic_write_json(validated["evaluation"]["result_path"], validated)


def validate_complete_planning_result(
    result: Mapping[str, Any], *, actual_path: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """Validate result completeness, integrity, counts, and DROID semantics."""

    data = _plain(result, nonfinite_to_none=True)
    if data.get("schema_version") != PLANNING_RESULT_SCHEMA_VERSION:
        raise PlanningProvenanceError("unsupported planning result schema_version")
    if data.get("kind") != "planning-evaluation-result" or data.get("status") != "complete":
        raise PlanningProvenanceError("planning result is not complete")
    stored_integrity = data.pop("integrity_sha256", None)
    if stored_integrity != canonical_sha256(data):
        raise PlanningProvenanceError("planning result integrity checksum mismatch")
    data["integrity_sha256"] = stored_integrity

    checkpoint = data.get("checkpoint", {})
    if not checkpoint.get("id"):
        raise PlanningProvenanceError("planning result checkpoint id is missing")
    _normalise_sha256(checkpoint.get("sha256", ""))
    assert_immutable_checkpoint_path(checkpoint.get("path", ""))
    evaluation = data.get("evaluation", {})
    if not evaluation.get("id") or _safe_id(evaluation["id"]) != evaluation["id"]:
        raise PlanningProvenanceError("planning result evaluation id is invalid")
    if not _SHA256_RE.fullmatch(str(evaluation.get("config_sha256", ""))):
        raise PlanningProvenanceError("planning result evaluation config checksum is invalid")
    expected_tasks = set(evaluation.get("expected_tasks", []))
    observed_tasks = set(evaluation.get("observed_tasks", []))
    expected_counts = evaluation.get("expected_episode_counts", {})
    observed_counts = evaluation.get("observed_episode_counts", {})
    if not expected_tasks or expected_tasks != observed_tasks or expected_counts != observed_counts:
        raise PlanningProvenanceError("planning result task or episode counts are incomplete")
    if any(not isinstance(count, int) or isinstance(count, bool) or count <= 0 for count in expected_counts.values()):
        raise PlanningProvenanceError("planning result episode counts must be positive integers")
    if len(expected_tasks) != evaluation.get("expected_task_count"):
        raise PlanningProvenanceError("planning result expected_task_count is inconsistent")
    if len(observed_tasks) != evaluation.get("observed_task_count"):
        raise PlanningProvenanceError("planning result observed_task_count is inconsistent")
    if sum(expected_counts.values()) != evaluation.get("expected_episode_count"):
        raise PlanningProvenanceError("planning result expected_episode_count is inconsistent")
    if sum(observed_counts.values()) != evaluation.get("observed_episode_count"):
        raise PlanningProvenanceError("planning result observed_episode_count is inconsistent")

    selection = data.get("selection", {})
    metric = selection.get("metric")
    mode = selection.get("mode")
    value = selection.get("value")
    if mode not in {"min", "max"} or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PlanningProvenanceError("planning result selection is invalid")
    if data.get("metrics", {}).get(metric) != value:
        raise PlanningProvenanceError("planning result selection value does not match metrics")
    if "droid" in str(evaluation.get("task_name", "")).lower() and (metric, mode) != (
        "ep_end_dist_xyz",
        "min",
    ):
        raise PlanningProvenanceError("DROID planning results must minimize ep_end_dist_xyz")

    declared_path = _absolute_path(evaluation.get("result_path", ""))
    if evaluation["id"] not in Path(declared_path).name:
        raise PlanningProvenanceError("planning result path is not specific to its evaluation id")
    if actual_path is not None and _absolute_path(actual_path) != declared_path:
        raise PlanningProvenanceError(
            f"planning result path does not match provenance: {actual_path} != {declared_path}"
        )
    return data


def load_complete_planning_result(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a published planning result and validate its provenance/integrity."""

    with open(path, "r", encoding="utf-8") as result_file:
        result = json.load(result_file)
    return validate_complete_planning_result(result, actual_path=path)


def _alias_integrity(alias: Mapping[str, Any]) -> str:
    payload = dict(alias)
    payload.pop("integrity_sha256", None)
    return canonical_sha256(payload)


def write_atomic_role_alias(
    alias_path: str | os.PathLike[str], role: str, result: Mapping[str, Any]
) -> dict[str, Any]:
    """Atomically point a checkpoint role at the exact result checkpoint."""

    completed = validate_complete_planning_result(result)
    alias = {
        "schema_version": ROLE_ALIAS_SCHEMA_VERSION,
        "kind": "checkpoint-role-alias",
        "role": str(role),
        "updated_at": _utc_now(),
        "checkpoint": completed["checkpoint"],
        "promotion": {
            "evaluation_id": completed["evaluation"]["id"],
            "evaluation_config_sha256": completed["evaluation"]["config_sha256"],
            "result_path": completed["evaluation"]["result_path"],
            "selection": completed["selection"],
            "metrics": completed["metrics"],
            "completed_at": completed["completed_at"],
        },
    }
    alias["integrity_sha256"] = _alias_integrity(alias)
    _atomic_write_json(alias_path, alias)
    return alias


def load_role_alias(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate an atomic checkpoint role alias."""

    with open(path, "r", encoding="utf-8") as alias_file:
        alias = json.load(alias_file)
    if alias.get("schema_version") != ROLE_ALIAS_SCHEMA_VERSION or alias.get("kind") != "checkpoint-role-alias":
        raise PlanningPromotionError(f"invalid checkpoint role alias: {path}")
    if alias.get("integrity_sha256") != _alias_integrity(alias):
        raise PlanningPromotionError(f"checkpoint role alias integrity checksum mismatch: {path}")
    _normalise_sha256(alias.get("checkpoint", {}).get("sha256", ""))
    assert_immutable_checkpoint_path(alias.get("checkpoint", {}).get("path", ""))
    return alias


def _result_rank(result: Mapping[str, Any]) -> tuple[Any, ...]:
    selection = result["selection"]
    value = float(selection["value"])
    primary = value if selection["mode"] == "min" else -value
    step = result["checkpoint"].get("step")
    # Prefer a later step for exact metric ties, then use stable identifiers so
    # the winner is independent of result arrival order.
    step_rank = -int(step) if step is not None else 0
    return primary, step_rank, str(result["checkpoint"]["id"]), str(result["evaluation"]["id"])


def _alias_as_result(alias: Mapping[str, Any]) -> dict[str, Any]:
    promotion = alias["promotion"]
    return {
        "checkpoint": alias["checkpoint"],
        "evaluation": {
            "id": promotion["evaluation_id"],
            "config_sha256": promotion["evaluation_config_sha256"],
            "result_path": promotion["result_path"],
        },
        "selection": promotion["selection"],
        "metrics": promotion["metrics"],
    }


def _discover_result_paths(
    source: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
) -> tuple[list[Path], Path]:
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.is_dir():
            return sorted(path.rglob(f"*{PLANNING_RESULT_SUFFIX}")), path
        return [path], path.parent
    paths = sorted(Path(path) for path in source)
    if not paths:
        return [], Path.cwd()
    try:
        common = Path(os.path.commonpath([str(path.parent) for path in paths]))
    except ValueError:
        common = paths[0].parent
    return paths, common


def _invoke_manager(manager: Any, role: str, result: Mapping[str, Any]) -> Any:
    method = getattr(manager, "promote", None)
    if not callable(method):
        raise PlanningPromotionError("checkpoint manager does not expose a callable promote method")
    checkpoint = result["checkpoint"]
    checkpoint_path = Path(checkpoint["path"]).resolve()
    checkpoint_reference = None
    if "reference" in inspect.signature(method).parameters:
        manager_root = getattr(manager, "root", None)
        if manager_root is None:
            raise PlanningPromotionError("checkpoint manager with a reference API must expose its root path")
        try:
            relative_path = checkpoint_path.relative_to(Path(manager_root).resolve()).as_posix()
        except ValueError as error:
            raise PlanningPromotionError(
                f"planning checkpoint is outside the checkpoint manager root: {checkpoint_path}"
            ) from error
        if checkpoint.get("step") is None:
            raise PlanningPromotionError("checkpoint_step is required for CheckpointManager promotion")
        checkpoint_reference = {
            "object_id": checkpoint_path.name,
            "relative_path": relative_path,
            "sha256": checkpoint["sha256"],
            "size_bytes": checkpoint_path.stat().st_size,
            "global_update": int(checkpoint["step"]),
            "epoch": None,
            "created_at": result["completed_at"],
            "metadata": {
                "planning_eval_id": result["evaluation"]["id"],
                "planning_eval_config_sha256": result["evaluation"]["config_sha256"],
                "planning_result_path": result["evaluation"]["result_path"],
            },
        }
    manager_metrics = dict(result["metrics"])
    manager_metrics["_planning_eval_config_sha256"] = result["evaluation"]["config_sha256"]
    manager_metrics["_planning_eval_id"] = result["evaluation"]["id"]
    manager_metrics["_planning_result_path"] = result["evaluation"]["result_path"]
    available = {
        "role": role,
        "checkpoint_id": checkpoint["id"],
        "checkpoint_path": checkpoint["path"],
        "source_path": checkpoint["path"],
        "checkpoint_checksum": checkpoint["sha256"],
        "checksum": checkpoint["sha256"],
        "reference": checkpoint_reference,
        "metric_name": result["selection"]["metric"],
        "metric_value": result["selection"]["value"],
        "mode": result["selection"]["mode"],
        "metrics": manager_metrics,
        "promotion": result["selection"],
        "provenance": result,
    }
    signature = inspect.signature(method)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    kwargs = (
        available
        if accepts_kwargs
        else {key: value for key, value in available.items() if key in signature.parameters}
    )
    missing = [
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.default is inspect.Parameter.empty
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]
    if missing:
        raise PlanningPromotionError(
            "checkpoint manager promote signature is unsupported; missing arguments: " + ", ".join(missing)
        )
    return method(**kwargs)


def reconcile_planning_results(
    results: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    manager: Any | None = None,
    promote: Callable[[str, str, Mapping[str, Any]], Any] | None = None,
    role: str = "best_planning",
    alias_path: str | os.PathLike[str] | None = None,
    expected_eval_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Reconcile out-of-order results and promote the globally best checkpoint.

    ``promote`` is a deliberately small integration seam with the signature
    ``promote(checkpoint_path, role, metrics)``.  Alternatively, ``manager``
    may expose a keyword-oriented ``promote`` method.  When neither is given,
    the atomic role alias is the promotion mechanism.
    """

    if manager is not None and promote is not None:
        raise PlanningPromotionError("pass either manager or promote, not both")
    result_paths, base_dir = _discover_result_paths(results)
    loaded: list[dict[str, Any]] = []
    seen_eval_ids: dict[str, str] = {}
    for path in result_paths:
        result = load_complete_planning_result(path)
        if not result["evaluation"].get("promotion_eligible", True):
            continue
        config_digest = result["evaluation"]["config_sha256"]
        if expected_eval_config_sha256 is not None and config_digest != expected_eval_config_sha256:
            continue
        eval_id = result["evaluation"]["id"]
        integrity = result["integrity_sha256"]
        if eval_id in seen_eval_ids and seen_eval_ids[eval_id] != integrity:
            raise PlanningPromotionError(f"conflicting complete results have the same eval_id: {eval_id}")
        seen_eval_ids[eval_id] = integrity
        loaded.append(result)

    if alias_path is None:
        if manager is not None and callable(getattr(manager, "alias_path", None)):
            alias_path = manager.alias_path(role)
        else:
            alias_path = base_dir / "roles" / f"{role}.json"
    alias_path = Path(alias_path)
    current_alias = None
    if manager is None and alias_path.exists():
        current_alias = load_role_alias(alias_path)
        if current_alias["role"] != role:
            raise PlanningPromotionError(
                f"checkpoint role alias at {alias_path} is for {current_alias['role']!r}, expected {role!r}"
            )

    if not loaded:
        return {
            "promoted": False,
            "winner": _alias_as_result(current_alias) if current_alias else None,
            "role": role,
            "alias_path": str(alias_path),
        }

    config_digests = {result["evaluation"]["config_sha256"] for result in loaded}
    selection_schemas = {(result["selection"]["metric"], result["selection"]["mode"]) for result in loaded}
    if len(config_digests) != 1:
        raise PlanningPromotionError(
            "planning results use different evaluation configs; pass expected_eval_config_sha256 before comparing"
        )
    if len(selection_schemas) != 1:
        raise PlanningPromotionError("planning results use different selection metrics and cannot be compared")

    winner = min(loaded, key=_result_rank)
    incumbent = _alias_as_result(current_alias) if current_alias else None
    if incumbent is not None:
        if incumbent["evaluation"]["config_sha256"] != winner["evaluation"]["config_sha256"]:
            raise PlanningPromotionError("best_planning alias was produced by an incomparable evaluation config")
        incumbent_schema = (incumbent["selection"]["metric"], incumbent["selection"]["mode"])
        winner_schema = (winner["selection"]["metric"], winner["selection"]["mode"])
        if incumbent_schema != winner_schema:
            raise PlanningPromotionError("best_planning alias uses an incomparable selection metric")
        if _result_rank(incumbent) <= _result_rank(winner):
            return {
                "promoted": False,
                "winner": incumbent,
                "role": role,
                "alias_path": str(alias_path),
            }

    if manager is not None:
        read_alias = getattr(manager, "read_alias", None)
        if callable(read_alias):
            managed_incumbent = read_alias(role, verify=True)
            if managed_incumbent is not None:
                incumbent_metrics = managed_incumbent.get("promotion", {}).get("metrics", {})
                incumbent_config = incumbent_metrics.get("_planning_eval_config_sha256")
                if incumbent_config != winner["evaluation"]["config_sha256"]:
                    raise PlanningPromotionError(
                        "managed best_planning alias was produced by an unknown or incomparable evaluation config"
                    )
        promoted = bool(_invoke_manager(manager, role, winner))
        return {
            "promoted": promoted,
            "winner": winner,
            "role": role,
            "alias_path": str(alias_path),
        }
    if promote is not None:
        promote(winner["checkpoint"]["path"], role, winner["metrics"])
    alias = write_atomic_role_alias(alias_path, role, winner)
    return {
        "promoted": True,
        "winner": winner,
        "role": role,
        "alias_path": str(alias_path),
        "alias": alias,
    }


def _new_planning_registry() -> dict[str, Any]:
    return {
        "schema_version": PLANNING_REGISTRY_SCHEMA_VERSION,
        "kind": "planning-evaluation-registry",
        "updated_at": _utc_now(),
        "promotion_eval_config_sha256": None,
        "checkpoints": {},
    }


def _registry_integrity(registry: Mapping[str, Any]) -> str:
    payload = dict(registry)
    payload.pop("integrity_sha256", None)
    return canonical_sha256(payload)


def _validate_planning_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    data = _plain(registry, nonfinite_to_none=True)
    if data.get("schema_version") != PLANNING_REGISTRY_SCHEMA_VERSION:
        raise PlanningPromotionError("unsupported planning evaluation registry schema_version")
    if data.get("kind") != "planning-evaluation-registry" or not isinstance(data.get("checkpoints"), dict):
        raise PlanningPromotionError("invalid planning evaluation registry")
    stored_integrity = data.pop("integrity_sha256", None)
    if stored_integrity is not None and stored_integrity != canonical_sha256(data):
        raise PlanningPromotionError("planning evaluation registry integrity checksum mismatch")
    if stored_integrity is not None:
        data["integrity_sha256"] = stored_integrity

    promotion_digest = data.get("promotion_eval_config_sha256")
    if promotion_digest is not None and not _SHA256_RE.fullmatch(str(promotion_digest)):
        raise PlanningPromotionError("planning registry promotion config checksum is invalid")
    for checkpoint_id, entry in data["checkpoints"].items():
        checkpoint = entry.get("checkpoint", {})
        if checkpoint.get("id") != checkpoint_id:
            raise PlanningPromotionError("planning registry checkpoint key/id mismatch")
        _normalise_sha256(checkpoint.get("sha256", ""))
        assert_immutable_checkpoint_path(checkpoint.get("path", ""))
        evaluations = entry.get("evaluations")
        expected_eval_ids = entry.get("expected_eval_ids")
        if not isinstance(evaluations, dict) or not isinstance(expected_eval_ids, list):
            raise PlanningPromotionError(f"planning registry checkpoint {checkpoint_id} has invalid evaluations")
        if set(evaluations) != set(expected_eval_ids) or len(expected_eval_ids) != len(set(expected_eval_ids)):
            raise PlanningPromotionError(f"planning registry checkpoint {checkpoint_id} evaluation IDs disagree")
        if entry.get("launch_state") not in {"registered", "launched"}:
            raise PlanningPromotionError(f"planning registry checkpoint {checkpoint_id} has an invalid launch state")
        if entry.get("execution_mode") not in {"synchronous", "asynchronous", "controller"}:
            raise PlanningPromotionError(f"planning registry checkpoint {checkpoint_id} has an invalid execution mode")
        for eval_id, evaluation in evaluations.items():
            if evaluation.get("eval_id") != eval_id:
                raise PlanningPromotionError("planning registry evaluation key/id mismatch")
            if not _SHA256_RE.fullmatch(str(evaluation.get("config_sha256", ""))):
                raise PlanningPromotionError(f"planning registry evaluation {eval_id} has an invalid config checksum")
            if evaluation.get("status") not in {"pending", "complete"}:
                raise PlanningPromotionError(f"planning registry evaluation {eval_id} has an invalid status")
        if entry.get("status") not in {"pending", "results_complete", "complete"}:
            raise PlanningPromotionError(f"planning registry checkpoint {checkpoint_id} has an invalid status")
    return data


@contextmanager
def _planning_registry_lock(registry_path: str | os.PathLike[str]):
    lock_path = Path(f"{os.fspath(registry_path)}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):  # pragma: no cover - non-POSIX or limited FUSE
            fcntl = None
        try:
            yield
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


def _read_planning_registry(path: str | os.PathLike[str]) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        return _new_planning_registry()
    try:
        with registry_path.open("r", encoding="utf-8") as registry_file:
            registry = json.load(registry_file)
    except (OSError, json.JSONDecodeError) as error:
        raise PlanningPromotionError(f"cannot read planning evaluation registry {registry_path}: {error}") from error
    return _validate_planning_registry(registry)


def load_planning_evaluation_registry(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and verify a durable planning-evaluation registry."""

    return _read_planning_registry(path)


def _write_planning_registry(path: str | os.PathLike[str], registry: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(registry)
    payload.pop("integrity_sha256", None)
    payload["updated_at"] = _utc_now()
    payload["integrity_sha256"] = _registry_integrity(payload)
    _atomic_write_json(path, payload)
    return payload


def register_planning_evaluations(
    registry_path: str | os.PathLike[str],
    provenances: Sequence[Mapping[str, Any]],
    *,
    execution_mode: str = "controller",
) -> dict[str, Any]:
    """Register every expected eval ID before an asynchronous launch.

    Registration is all-or-nothing for a checkpoint.  Retried registration is
    idempotent only when the complete expected set is identical.
    """

    if not provenances:
        raise PlanningPromotionError("at least one planning provenance record is required")
    if execution_mode not in {"synchronous", "asynchronous", "controller"}:
        raise PlanningPromotionError("execution_mode must be synchronous, asynchronous, or controller")
    normalized = [validate_planning_provenance(provenance) for provenance in provenances]
    checkpoint_keys = {
        (
            item["checkpoint_id"],
            item["checkpoint_sha256"],
            item["checkpoint_path"],
            item.get("checkpoint_step"),
        )
        for item in normalized
    }
    if len(checkpoint_keys) != 1:
        raise PlanningPromotionError("one registry launch may reference only one immutable checkpoint")
    checkpoint_id, checkpoint_sha256, checkpoint_path, checkpoint_step = next(iter(checkpoint_keys))
    eval_ids = [item["eval_id"] for item in normalized]
    if len(eval_ids) != len(set(eval_ids)):
        raise PlanningPromotionError("planning eval IDs must be unique within a checkpoint launch")
    promotion_digests = {
        item["eval_config_sha256"] for item in normalized if item.get("promotion_eligible", True)
    }
    if len(promotion_digests) > 1:
        raise PlanningPromotionError(
            "only one comparable eval-config checksum may be eligible for best_planning promotion"
        )
    promotion_digest = next(iter(promotion_digests), None)

    with _planning_registry_lock(registry_path):
        registry = _read_planning_registry(registry_path)
        existing_promotion_digest = registry.get("promotion_eval_config_sha256")
        if promotion_digest is not None:
            if existing_promotion_digest is None:
                registry["promotion_eval_config_sha256"] = promotion_digest
            elif existing_promotion_digest != promotion_digest:
                raise PlanningPromotionError(
                    "best_planning evaluation config changed within one registry; start a new comparison registry"
                )

        evaluation_records = {
            item["eval_id"]: {
                "eval_id": item["eval_id"],
                "config_sha256": item["eval_config_sha256"],
                "result_path": item["result_path"],
                "promotion_eligible": bool(item.get("promotion_eligible", True)),
                "status": "pending",
                "completed_at": None,
                "result_integrity_sha256": None,
            }
            for item in normalized
        }
        checkpoint_record = {
            "checkpoint": {
                "id": checkpoint_id,
                "sha256": checkpoint_sha256,
                "path": checkpoint_path,
                "step": checkpoint_step,
            },
            "registered_at": _utc_now(),
            "released_at": None,
            "launched_at": None,
            "launch_state": "registered",
            "execution_mode": execution_mode,
            "status": "pending",
            "expected_eval_ids": eval_ids,
            "evaluations": evaluation_records,
        }
        existing = registry["checkpoints"].get(checkpoint_id)
        if existing is not None:
            comparable_existing = copy.deepcopy(existing)
            comparable_existing.pop("registered_at", None)
            comparable_existing.pop("released_at", None)
            comparable_existing.pop("launched_at", None)
            comparable_existing.pop("launch_state", None)
            comparable_existing.pop("status", None)
            comparable_new = copy.deepcopy(checkpoint_record)
            comparable_new.pop("registered_at", None)
            comparable_new.pop("released_at", None)
            comparable_new.pop("launched_at", None)
            comparable_new.pop("launch_state", None)
            comparable_new.pop("status", None)
            # Preserve completed state on an exact retry; reject attempts to
            # silently add or remove expected evaluations.
            existing_evaluations = comparable_existing.get("evaluations", {})
            for record in existing_evaluations.values():
                record.pop("status", None)
                record.pop("completed_at", None)
                record.pop("result_integrity_sha256", None)
                record.pop("selection", None)
            for record in comparable_new.get("evaluations", {}).values():
                record.pop("status", None)
                record.pop("completed_at", None)
                record.pop("result_integrity_sha256", None)
                record.pop("selection", None)
            if comparable_existing != comparable_new:
                raise PlanningPromotionError(
                    f"checkpoint {checkpoint_id} was already registered with a different expected eval set"
                )
        else:
            registry["checkpoints"][checkpoint_id] = checkpoint_record
        return _write_planning_registry(registry_path, registry)


def mark_planning_evaluations_launched(
    registry_path: str | os.PathLike[str], checkpoint_id: str
) -> dict[str, Any]:
    """Durably mark that all registered eval jobs for a checkpoint were launched."""

    with _planning_registry_lock(registry_path):
        registry = _read_planning_registry(registry_path)
        try:
            checkpoint = registry["checkpoints"][str(checkpoint_id)]
        except KeyError as error:
            raise PlanningPromotionError(
                f"cannot mark unregistered planning checkpoint as launched: {checkpoint_id}"
            ) from error
        if checkpoint["launch_state"] == "registered":
            checkpoint["launch_state"] = "launched"
            checkpoint["launched_at"] = _utc_now()
        return _write_planning_registry(registry_path, registry)


def _verify_registered_result(
    checkpoint_record: Mapping[str, Any],
    evaluation_record: Mapping[str, Any],
) -> dict[str, Any] | None:
    result_path = Path(evaluation_record["result_path"])
    if not result_path.exists():
        return None
    result = load_complete_planning_result(result_path)
    checkpoint = checkpoint_record["checkpoint"]
    result_checkpoint = result["checkpoint"]
    for key, result_key in (("id", "id"), ("sha256", "sha256"), ("path", "path")):
        if result_checkpoint[result_key] != checkpoint[key]:
            raise PlanningPromotionError(
                f"planning result {result_path} checkpoint {result_key} does not match its registry entry"
            )
    evaluation = result["evaluation"]
    if evaluation["id"] != evaluation_record["eval_id"]:
        raise PlanningPromotionError(f"planning result {result_path} eval ID does not match its registry entry")
    if evaluation["config_sha256"] != evaluation_record["config_sha256"]:
        raise PlanningPromotionError(
            f"planning result {result_path} config checksum does not match its registry entry"
        )
    if bool(evaluation.get("promotion_eligible", True)) != bool(evaluation_record["promotion_eligible"]):
        raise PlanningPromotionError(
            f"planning result {result_path} promotion eligibility does not match its registry entry"
        )
    return result


def poll_planning_evaluations(
    registry_path: str | os.PathLike[str],
    *,
    manager: Any | None = None,
    role: str = "best_planning",
) -> dict[str, Any]:
    """Verify arrived results, promote comparable winners, then release pins.

    A checkpoint remains pending in ``CheckpointManager`` until every eval ID
    registered for it has a complete, checksum-protected result.
    """

    with _planning_registry_lock(registry_path):
        registry = _read_planning_registry(registry_path)
        newly_verified: list[str] = []
        for checkpoint_record in registry["checkpoints"].values():
            if checkpoint_record["status"] == "complete":
                continue
            for evaluation_record in checkpoint_record["evaluations"].values():
                if evaluation_record["status"] == "complete":
                    continue
                result = _verify_registered_result(checkpoint_record, evaluation_record)
                if result is None:
                    continue
                evaluation_record["status"] = "complete"
                evaluation_record["completed_at"] = result["completed_at"]
                evaluation_record["result_integrity_sha256"] = result["integrity_sha256"]
                evaluation_record["selection"] = result["selection"]
                newly_verified.append(result["evaluation"]["id"])
            if all(item["status"] == "complete" for item in checkpoint_record["evaluations"].values()):
                checkpoint_record["status"] = "results_complete"

        # Persist verified terminal state before mutating checkpoint roles.  A
        # crash after this point is safely repairable by an idempotent poll.
        registry = _write_planning_registry(registry_path, registry)

        promotion_digest = registry.get("promotion_eval_config_sha256")
        eligible_result_paths = [
            evaluation["result_path"]
            for checkpoint in registry["checkpoints"].values()
            for evaluation in checkpoint["evaluations"].values()
            if evaluation["status"] == "complete"
            and evaluation["promotion_eligible"]
            and evaluation["config_sha256"] == promotion_digest
        ]
        reconciliation = None
        if eligible_result_paths:
            reconciliation = reconcile_planning_results(
                eligible_result_paths,
                manager=manager,
                role=role,
                expected_eval_config_sha256=promotion_digest,
                alias_path=(Path(registry_path).parent / "roles" / f"{role}.json") if manager is None else None,
            )

        released: list[str] = []
        for checkpoint_id, checkpoint_record in registry["checkpoints"].items():
            if checkpoint_record["status"] != "results_complete":
                continue
            if manager is not None:
                manager.unmark_pending(checkpoint_id)
            checkpoint_record["status"] = "complete"
            checkpoint_record["released_at"] = _utc_now()
            released.append(checkpoint_id)
        registry = _write_planning_registry(registry_path, registry)
        if released and manager is not None:
            manager.garbage_collect()

        pending = [
            checkpoint_id
            for checkpoint_id, checkpoint in registry["checkpoints"].items()
            if checkpoint["status"] != "complete"
        ]
        expected_eval_count = sum(
            len(checkpoint["evaluations"]) for checkpoint in registry["checkpoints"].values()
        )
        complete_eval_count = sum(
            evaluation["status"] == "complete"
            for checkpoint in registry["checkpoints"].values()
            for evaluation in checkpoint["evaluations"].values()
        )
        return {
            "pending_checkpoint_ids": pending,
            "released_checkpoint_ids": released,
            "newly_verified_eval_ids": newly_verified,
            "expected_eval_count": expected_eval_count,
            "complete_eval_count": complete_eval_count,
            "promotion": reconciliation,
            "registry_path": os.fspath(registry_path),
        }


def drain_planning_evaluations(
    registry_path: str | os.PathLike[str],
    *,
    manager: Any | None = None,
    role: str = "best_planning",
    timeout_seconds: float,
    poll_interval_seconds: float = 10.0,
    checkpoint_ids: Iterable[str] | None = None,
    raise_on_timeout: bool = False,
    _clock: Callable[[], float] = time.monotonic,
    _sleep: Callable[[float], Any] = time.sleep,
) -> dict[str, Any]:
    """Block a controller until registered planning results are terminal.

    Evaluation ranks only publish result files.  This controller-side drain is
    the sole owner of checkpoint promotion, pending-pin release, and garbage
    collection.
    """

    timeout_seconds = float(timeout_seconds)
    poll_interval_seconds = float(poll_interval_seconds)
    if timeout_seconds < 0 or poll_interval_seconds <= 0:
        raise ValueError("timeout_seconds must be non-negative and poll_interval_seconds must be positive")
    targets = set(str(item) for item in checkpoint_ids) if checkpoint_ids is not None else None
    start = _clock()
    deadline = start + timeout_seconds
    while True:
        report = poll_planning_evaluations(registry_path, manager=manager, role=role)
        registry = load_planning_evaluation_registry(registry_path)
        known_ids = set(registry["checkpoints"])
        if targets is not None and not targets <= known_ids:
            missing = sorted(targets - known_ids)
            raise PlanningPromotionError(f"drain requested unregistered checkpoint IDs: {missing}")
        pending = set(report["pending_checkpoint_ids"])
        pending_targets = pending if targets is None else pending & targets
        now = _clock()
        if not pending_targets:
            return {
                **report,
                "drained": True,
                "timed_out": False,
                "elapsed_seconds": max(0.0, now - start),
            }
        if now >= deadline:
            timed_out_report = {
                **report,
                "drained": False,
                "timed_out": True,
                "elapsed_seconds": max(0.0, now - start),
                "pending_checkpoint_ids": sorted(pending_targets),
            }
            if raise_on_timeout:
                raise PlanningDrainTimeout(
                    "timed out waiting for verified planning results for checkpoints: "
                    + ", ".join(sorted(pending_targets))
                )
            return timed_out_report
        _sleep(min(poll_interval_seconds, max(0.0, deadline - now)))
