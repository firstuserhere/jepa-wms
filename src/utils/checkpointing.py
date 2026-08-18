# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Durable, verifiable training checkpoints.

Checkpoint files are immutable objects.  Human-readable roles (``latest``,
``best_rollout`` and ``best_planning``) are small, atomically replaced JSON
aliases that point at those objects.  This avoids copying a multi-gigabyte
checkpoint merely to promote it and, more importantly, means a reader never
observes a role pointing at a partially-written object.

This module deliberately has no import-time torch dependency.  Manifest,
retention and integrity tooling can therefore run on storage/controller nodes
without installing the training stack.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as _datetime
import enum
import fnmatch
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_FORMAT = "jepa-wm-training-checkpoint"
CHECKPOINT_ROLES = frozenset(("latest", "best_rollout", "best_planning"))

# These settings change where/how a run executes, but not the mathematical
# training trajectory. Dataset and encoder identities live in their own
# checksummed manifests and are intentionally not inferred from path strings.
DEFAULT_RESUME_EXCLUDED_PATHS = (
    "folder",
    "checkpoint_folder",
    "nodes",
    "tasks_per_node",
    "cpus_per_task",
    "cluster",
    "cluster.*",
    "resources",
    "resources.*",
    "runtime",
    "runtime.*",
    "logging",
    "logging.*",
    "meta.load_checkpoint",
    "meta.read_checkpoint",
    "meta.checkpoint_path",
    "meta.resume_from",
    "meta.output_dir",
    "meta.log_dir",
    "meta.pretrained_path",
    "data.root",
    "data.root_path",
    "data.dataset_root",
    "data.dataset_path",
    "data.manifest_path",
    "data.paths_path",
)

_RESERVED_CHECKPOINT_KEYS = frozenset(("schema_version", "format", "manifest", "state", "progress", "rng", "sampler"))
_OBJECT_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class CheckpointError(RuntimeError):
    """Base exception for durable-checkpoint failures."""


class CheckpointIntegrityError(CheckpointError):
    """A checkpoint is missing, truncated, or has the wrong digest."""


class CheckpointSchemaError(CheckpointError):
    """A checkpoint or alias does not satisfy the v2 schema."""


class ResumeContractError(CheckpointError):
    """The current run is not compatible with a saved training state."""


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_directory(directory: Path) -> None:
    """Persist a directory entry after ``os.replace`` when supported."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some object-store FUSE implementations do not implement directory
        # fsync. The file itself was still fsynced before the atomic replace.
        pass
    finally:
        os.close(descriptor)


def _temporary_path(destination: Path) -> tuple[int, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    return descriptor, Path(name)


def atomic_json_dump(value: Any, destination: os.PathLike[str] | str) -> Path:
    """Write JSON using fsync followed by an atomic same-directory replace."""

    destination = Path(destination)
    descriptor, temporary = _temporary_path(destination)
    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1  # ownership transferred to stream
        with stream:
            json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_torch_save(value: Any, destination: os.PathLike[str] | str, **save_kwargs: Any) -> Path:
    """Durably save with ``torch.save`` and atomically publish the result."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised on controller-only installs
        raise CheckpointError("atomic_torch_save requires PyTorch") from error

    destination = Path(destination)
    descriptor, temporary = _temporary_path(destination)
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1  # ownership transferred to stream
        with stream:
            torch.save(value, stream, **save_kwargs)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_copy(source: os.PathLike[str] | str, destination: os.PathLike[str] | str) -> Path:
    """Copy a file and atomically publish the complete copy."""

    source = Path(source)
    destination = Path(destination)
    descriptor, temporary = _temporary_path(destination)
    try:
        output_stream = os.fdopen(descriptor, "wb")
        descriptor = -1  # ownership transferred to output_stream
        with source.open("rb") as input_stream, output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=16 * 1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        temporary.unlink(missing_ok=True)
        raise
    return destination


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_integrity(path: os.PathLike[str] | str) -> dict[str, Any]:
    path = Path(path)
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def verify_file_integrity(
    path: os.PathLike[str] | str,
    *,
    sha256: str,
    size_bytes: int,
) -> bool:
    path = Path(path)
    if not path.is_file():
        raise CheckpointIntegrityError(f"checkpoint object does not exist: {path}")
    actual_size = path.stat().st_size
    if actual_size != int(size_bytes):
        raise CheckpointIntegrityError(
            f"checkpoint size mismatch for {path}: expected {size_bytes}, got {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != sha256:
        raise CheckpointIntegrityError(
            f"checkpoint SHA256 mismatch for {path}: expected {sha256}, got {actual_sha256}"
        )
    return True


def _to_primitive(value: Any) -> Any:
    """Convert a resolved config/manifest to deterministic JSON values."""

    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True, enum_to_str=True)
    except ImportError:
        pass

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, enum.Enum):
        return _to_primitive(value.value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_primitive(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_primitive(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_primitive(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_to_primitive(item) for item in value]
        return sorted(items, key=canonical_json_dumps)

    # NumPy scalar types expose item(); calling it on arbitrary containers is
    # undesirable, so only accept scalar results.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar = item_method()
        except (TypeError, ValueError):
            scalar = value
        if scalar is not value:
            return _to_primitive(scalar)

    module = type(value).__module__.split(".", 1)[0]
    if module == "torch" and type(value).__name__ in ("dtype", "device"):
        return str(value)
    raise TypeError(f"cannot canonically serialize {type(value).__module__}.{type(value).__qualname__}")


def canonical_json_dumps(value: Any) -> str:
    return json.dumps(
        _to_primitive(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def config_hash(config: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(config).encode("utf-8")).hexdigest()


def _is_excluded(path: tuple[str, ...], patterns: Sequence[str]) -> bool:
    dotted = ".".join(path)
    return any(fnmatch.fnmatchcase(dotted, pattern) for pattern in patterns)


def _without_excluded_paths(value: Any, patterns: Sequence[str], path: tuple[str, ...] = ()) -> Any:
    primitive = _to_primitive(value)
    if isinstance(primitive, Mapping):
        result = {}
        for key, item in primitive.items():
            child_path = path + (str(key),)
            if not _is_excluded(child_path, patterns):
                result[str(key)] = _without_excluded_paths(item, patterns, child_path)
        return result
    if isinstance(primitive, list):
        return [_without_excluded_paths(item, patterns, path + (str(index),)) for index, item in enumerate(primitive)]
    return primitive


def build_resume_contract(
    resolved_config: Any,
    *,
    excluded_paths: Iterable[str] = DEFAULT_RESUME_EXCLUDED_PATHS,
) -> dict[str, Any]:
    patterns = tuple(sorted(set(excluded_paths)))
    mathematical_config = _without_excluded_paths(resolved_config, patterns)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "algorithm": "sha256",
        "sha256": config_hash(mathematical_config),
        "config": mathematical_config,
        "excluded_paths": list(patterns),
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(item, child))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(_flatten(item, child))
        return flattened
    return {prefix or "<root>": value}


def resume_contract_diff(saved_config: Any, current_config: Any) -> list[dict[str, Any]]:
    saved_flat = _flatten(_to_primitive(saved_config))
    current_flat = _flatten(_to_primitive(current_config))
    differences = []
    for path in sorted(set(saved_flat) | set(current_flat)):
        saved_value = saved_flat.get(path, {"__missing__": True})
        current_value = current_flat.get(path, {"__missing__": True})
        if saved_value != current_value:
            differences.append({"path": path, "saved": saved_value, "current": current_value})
    return differences


def validate_resume_contract(
    saved_contract: Mapping[str, Any],
    current_resolved_config: Any,
    *,
    excluded_paths: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Raise if a current config cannot exactly continue a saved trajectory."""

    if "resume_contract" in saved_contract:
        saved_contract = saved_contract["resume_contract"]
    required = ("sha256", "config", "excluded_paths")
    missing = [key for key in required if key not in saved_contract]
    if missing:
        raise ResumeContractError(f"saved resume contract is missing: {', '.join(missing)}")
    saved_config = _to_primitive(saved_contract["config"])
    saved_sha256 = str(saved_contract["sha256"])
    if config_hash(saved_config) != saved_sha256:
        raise ResumeContractError("saved resume contract is internally inconsistent (config hash mismatch)")

    patterns = tuple(excluded_paths) if excluded_paths is not None else tuple(saved_contract["excluded_paths"])
    current_contract = build_resume_contract(current_resolved_config, excluded_paths=patterns)
    if current_contract["sha256"] != saved_sha256:
        differences = resume_contract_diff(saved_config, current_contract["config"])
        preview = "; ".join(
            f"{item['path']}: {item['saved']!r} -> {item['current']!r}" for item in differences[:12]
        )
        if len(differences) > 12:
            preview += f"; ... and {len(differences) - 12} more"
        raise ResumeContractError(f"resume config is incompatible: {preview}")
    return current_contract


def get_git_manifest(repo_root: os.PathLike[str] | str | None = None) -> dict[str, Any]:
    """Capture the exact source state, including staged and untracked files.

    ``git diff HEAD`` does not include untracked files.  Merely recording their
    names is insufficient for reproducing a dirty research run, so each
    untracked file is represented as a binary-safe ``git diff --no-index``
    patch as well.  Large generated artifacts should therefore be ignored by
    the repository rather than placed inside the source tree.
    """

    working_directory = Path(repo_root or Path(__file__).resolve().parents[2])

    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", os.fspath(working_directory), *arguments],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")

    try:
        root = Path(run("rev-parse", "--show-toplevel").strip())
        commit = run("rev-parse", "HEAD").strip()
        branch = run("rev-parse", "--abbrev-ref", "HEAD").strip()
        status = run("status", "--porcelain=v1", "--untracked-files=all")
        dirty_patch = run("diff", "--binary", "HEAD", "--")
        untracked_raw = subprocess.check_output(
            ["git", "-C", os.fspath(working_directory), "ls-files", "--others", "--exclude-standard", "-z"],
            stderr=subprocess.DEVNULL,
        )
        untracked_files = [
            item.decode("utf-8", errors="surrogateescape")
            for item in untracked_raw.split(b"\0")
            if item
        ]
        untracked_patches = []
        for relative_path in untracked_files:
            result = subprocess.run(
                ["git", "diff", "--binary", "--no-index", "--", "/dev/null", relative_path],
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            # git diff returns 1 when differences are present.  Any other
            # non-zero status means the source snapshot is incomplete.
            if result.returncode not in (0, 1):
                error_text = result.stderr.decode("utf-8", errors="replace").strip()
                raise subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    output=result.stdout,
                    stderr=error_text,
                )
            untracked_patches.append(result.stdout.decode("utf-8", errors="replace"))
        dirty_patch += "".join(untracked_patches)
        return {
            "available": True,
            "root": os.fspath(root),
            "commit": commit,
            "branch": branch,
            "dirty": bool(status),
            "status_porcelain": status,
            "dirty_patch": dirty_patch,
            "dirty_patch_sha256": hashlib.sha256(dirty_patch.encode("utf-8")).hexdigest(),
            "untracked_files": untracked_files,
            "untracked_files_in_patch": True,
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def create_checkpoint_manifest(
    *,
    resolved_config: Any,
    dataset_manifest: Mapping[str, Any],
    encoder_manifest: Mapping[str, Any],
    wandb_run_id: str,
    promotion_metrics: Mapping[str, Any] | None = None,
    lineage: Sequence[Mapping[str, Any]] | None = None,
    git_manifest: Mapping[str, Any] | None = None,
    repo_root: os.PathLike[str] | str | None = None,
    require_git_manifest: bool = False,
    excluded_resume_paths: Iterable[str] = DEFAULT_RESUME_EXCLUDED_PATHS,
) -> dict[str, Any]:
    if not wandb_run_id or not str(wandb_run_id).strip():
        raise ValueError("wandb_run_id is required for a v2 checkpoint manifest")
    if not dataset_manifest:
        raise ValueError("dataset_manifest is required for a v2 checkpoint manifest")
    if not encoder_manifest:
        raise ValueError("encoder_manifest is required for a v2 checkpoint manifest")
    config = _to_primitive(resolved_config)
    source_manifest = (
        _to_primitive(git_manifest) if git_manifest is not None else get_git_manifest(repo_root)
    )
    if require_git_manifest:
        if not source_manifest.get("available"):
            raise ValueError(
                "A complete Git source manifest is required, but repository metadata is unavailable: "
                f"{source_manifest.get('error', 'unknown error')}"
            )
        required_source_fields = (
            "commit",
            "dirty",
            "dirty_patch",
            "dirty_patch_sha256",
            "status_porcelain",
        )
        missing_source_fields = [
            key for key in required_source_fields if key not in source_manifest
        ]
        if missing_source_fields:
            raise ValueError(
                "Git source manifest is incomplete: " + ", ".join(missing_source_fields)
            )
        expected_patch_sha256 = hashlib.sha256(
            str(source_manifest["dirty_patch"]).encode("utf-8")
        ).hexdigest()
        if source_manifest["dirty_patch_sha256"] != expected_patch_sha256:
            raise ValueError("Git dirty patch checksum is internally inconsistent")
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "format": CHECKPOINT_FORMAT,
        "created_at": _utc_now(),
        "resolved_config": config,
        "config_sha256": config_hash(config),
        "resume_contract": build_resume_contract(config, excluded_paths=excluded_resume_paths),
        "git": source_manifest,
        "dataset": _to_primitive(dataset_manifest),
        "encoder": _to_primitive(encoder_manifest),
        "wandb": {"run_id": str(wandb_run_id)},
        "promotion_metrics": _to_primitive(promotion_metrics or {}),
        "lineage": _to_primitive(lineage or []),
    }


def build_checkpoint_v2(
    *,
    training_state: Mapping[str, Any],
    epoch: int,
    global_update: int,
    rng_state: Mapping[str, Any],
    sampler_state: Mapping[str, Any],
    manifest: Mapping[str, Any],
    legacy_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "format": CHECKPOINT_FORMAT,
        "manifest": dict(manifest),
        "state": dict(training_state),
        "progress": {"epoch": int(epoch), "global_update": int(global_update)},
        "rng": dict(rng_state),
        "sampler": dict(sampler_state),
    }
    if legacy_fields:
        overlap = _RESERVED_CHECKPOINT_KEYS.intersection(legacy_fields)
        if overlap:
            raise ValueError(f"legacy_fields may not replace reserved v2 fields: {sorted(overlap)}")
        checkpoint.update(legacy_fields)
    validate_checkpoint_v2(checkpoint)
    return checkpoint


def validate_checkpoint_v2(checkpoint: Mapping[str, Any]) -> bool:
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointSchemaError(f"expected checkpoint schema v2, got {checkpoint.get('schema_version')!r}")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointSchemaError(f"unexpected checkpoint format: {checkpoint.get('format')!r}")
    for key in ("manifest", "state", "progress", "rng", "sampler"):
        if not isinstance(checkpoint.get(key), Mapping):
            raise CheckpointSchemaError(f"checkpoint field {key!r} must be a mapping")

    state = checkpoint["state"]
    missing_state = [key for key in ("predictor", "optimizer", "scaler", "schedulers") if key not in state]
    if missing_state:
        raise CheckpointSchemaError(f"checkpoint training state is missing: {', '.join(missing_state)}")
    schedulers = state["schedulers"]
    if not isinstance(schedulers, Mapping) or not {"lr", "weight_decay"}.issubset(schedulers):
        raise CheckpointSchemaError("checkpoint must contain lr and weight_decay scheduler states")
    progress = checkpoint["progress"]
    if "epoch" not in progress or "global_update" not in progress:
        raise CheckpointSchemaError("checkpoint progress must contain epoch and global_update")
    manifest = checkpoint["manifest"]
    for key in (
        "resolved_config",
        "config_sha256",
        "resume_contract",
        "git",
        "dataset",
        "encoder",
        "wandb",
        "promotion_metrics",
    ):
        if key not in manifest:
            raise CheckpointSchemaError(f"checkpoint manifest is missing {key!r}")
    if config_hash(manifest["resolved_config"]) != manifest["config_sha256"]:
        raise CheckpointSchemaError("checkpoint resolved config hash does not match")
    return True


def is_v2_checkpoint(checkpoint: Mapping[str, Any]) -> bool:
    return (
        checkpoint.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
        and checkpoint.get("format") == CHECKPOINT_FORMAT
    )


def training_state_from_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized state from either v2 or the original flat format."""

    if is_v2_checkpoint(checkpoint):
        return dict(checkpoint["state"])
    state = dict(checkpoint)
    if "optimizer" not in state and "opt" in state:
        state["optimizer"] = state["opt"]
    return state


def checkpoint_epoch(checkpoint: Mapping[str, Any], default: int = 0) -> int:
    if is_v2_checkpoint(checkpoint):
        return int(checkpoint["progress"]["epoch"])
    return int(checkpoint.get("epoch", default))


def checkpoint_global_update(checkpoint: Mapping[str, Any], default: int = 0) -> int:
    if is_v2_checkpoint(checkpoint):
        return int(checkpoint["progress"]["global_update"])
    return int(checkpoint.get("global_update", default))


def capture_rng_state(rank: int | None = None) -> dict[str, Any]:
    """Capture Python, NumPy and torch CPU/CUDA RNGs for one rank."""

    try:
        import numpy as np
        import torch
    except ImportError as error:  # pragma: no cover - dependencies are required for training
        raise CheckpointError("capture_rng_state requires NumPy and PyTorch") from error

    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    return {
        "rank": int(rank) if rank is not None else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": cuda_state,
    }


def gather_rng_states(local_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Collect one RNG snapshot per distributed rank on every rank."""

    try:
        import torch.distributed as dist
    except ImportError as error:  # pragma: no cover
        raise CheckpointError("gather_rng_states requires PyTorch") from error

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    local = dict(local_state) if local_state is not None else capture_rng_state(rank=rank)
    local["rank"] = rank
    if distributed:
        gathered: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    return {
        "world_size": len(gathered),
        "states": {str(state["rank"]): state for state in gathered},
    }


def restore_rng_state(state: Mapping[str, Any], *, strict_cuda: bool = True) -> None:
    """Restore one rank's RNG snapshot."""

    try:
        import numpy as np
        import torch
    except ImportError as error:  # pragma: no cover
        raise CheckpointError("restore_rng_state requires NumPy and PyTorch") from error

    for key in ("python", "numpy", "torch_cpu", "torch_cuda_all"):
        if key not in state:
            raise CheckpointSchemaError(f"RNG state is missing {key!r}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda_all"]
    if cuda_states is not None:
        if not torch.cuda.is_available():
            if strict_cuda:
                raise CheckpointError("checkpoint has CUDA RNG state but CUDA is unavailable")
        else:
            expected = torch.cuda.device_count()
            if strict_cuda and len(cuda_states) != expected:
                raise CheckpointError(
                    f"CUDA RNG device-count mismatch: checkpoint has {len(cuda_states)}, runtime has {expected}"
                )
            torch.cuda.set_rng_state_all(cuda_states)


def restore_rank_rng_state(
    rng_bundle: Mapping[str, Any],
    *,
    rank: int | None = None,
    strict_world_size: bool = True,
    strict_cuda: bool = True,
) -> None:
    """Select and restore the current rank from ``gather_rng_states`` output."""

    try:
        import torch.distributed as dist
    except ImportError as error:  # pragma: no cover
        raise CheckpointError("restore_rank_rng_state requires PyTorch") from error

    distributed = dist.is_available() and dist.is_initialized()
    current_rank = dist.get_rank() if rank is None and distributed else int(rank or 0)
    current_world_size = dist.get_world_size() if distributed else 1
    saved_world_size = int(rng_bundle.get("world_size", len(rng_bundle.get("states", {}))))
    if strict_world_size and current_world_size != saved_world_size:
        raise CheckpointError(
            f"RNG world-size mismatch: checkpoint has {saved_world_size}, runtime has {current_world_size}"
        )
    try:
        state = rng_bundle["states"][str(current_rank)]
    except KeyError as error:
        raise CheckpointSchemaError(f"checkpoint has no RNG state for rank {current_rank}") from error
    restore_rng_state(state, strict_cuda=strict_cuda)


def should_promote(
    candidate_value: float,
    incumbent_value: float | None,
    *,
    mode: str,
    candidate_step: int | None = None,
    incumbent_step: int | None = None,
    tie_break: str = "newer",
) -> bool:
    """Compare a metric, using an explicit deterministic tie-break policy."""

    if mode not in ("min", "max"):
        raise ValueError("mode must be 'min' or 'max'")
    if tie_break not in ("newer", "older", "keep", "candidate"):
        raise ValueError("tie_break must be 'newer', 'older', 'keep', or 'candidate'")
    candidate = float(candidate_value)
    if math.isnan(candidate):
        return False
    if incumbent_value is None or math.isnan(float(incumbent_value)):
        return True
    incumbent = float(incumbent_value)
    if (mode == "max" and candidate > incumbent) or (mode == "min" and candidate < incumbent):
        return True
    if candidate != incumbent:
        return False
    if tie_break == "candidate":
        return True
    if tie_break == "keep" or candidate_step is None or incumbent_step is None:
        return False
    if tie_break == "newer":
        return int(candidate_step) > int(incumbent_step)
    return int(candidate_step) < int(incumbent_step)


@dataclass(frozen=True)
class CheckpointRef:
    object_id: str
    relative_path: str
    sha256: str
    size_bytes: int
    global_update: int
    epoch: int | None
    created_at: str
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointRef":
        required = ("object_id", "relative_path", "sha256", "size_bytes", "global_update", "created_at")
        missing = [key for key in required if key not in value]
        if missing:
            raise CheckpointSchemaError(f"checkpoint reference is missing: {', '.join(missing)}")
        return cls(
            object_id=str(value["object_id"]),
            relative_path=str(value["relative_path"]),
            sha256=str(value["sha256"]),
            size_bytes=int(value["size_bytes"]),
            global_update=int(value["global_update"]),
            epoch=int(value["epoch"]) if value.get("epoch") is not None else None,
            created_at=str(value["created_at"]),
            metadata=dict(value.get("metadata", {})),
        )


@contextmanager
def _advisory_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):  # pragma: no cover - non-POSIX or limited FUSE
            fcntl = None
        try:
            yield
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


class CheckpointManager:
    """Manage immutable checkpoint objects and atomic role aliases."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        *,
        prefix: str = "jepa",
        keep_recent: int = 0,
    ) -> None:
        if not _OBJECT_NAME_RE.fullmatch(prefix):
            raise ValueError("checkpoint prefix may contain only letters, digits, '.', '_' and '-'")
        if int(keep_recent) < 0:
            raise ValueError("keep_recent must be non-negative")
        self.root = Path(root)
        self.prefix = prefix
        self.keep_recent = int(keep_recent)
        self.objects_dir = self.root / "objects"
        self.aliases_dir = self.root / "aliases"
        self.state_dir = self.root / "state"
        self.promotions_dir = self.state_dir / "promotions"
        for directory in (self.objects_dir, self.aliases_dir, self.state_dir, self.promotions_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _validate_role(self, role: str) -> None:
        if role not in CHECKPOINT_ROLES:
            raise ValueError(f"unknown checkpoint role {role!r}; expected one of {sorted(CHECKPOINT_ROLES)}")

    def _object_name(self, global_update: int) -> str:
        return f"{self.prefix}-step-{int(global_update):012d}-{uuid.uuid4().hex[:12]}.pth.tar"

    def _new_ref(
        self,
        path: Path,
        *,
        global_update: int,
        epoch: int | None,
        metadata: Mapping[str, Any] | None,
    ) -> CheckpointRef:
        integrity = file_integrity(path)
        return CheckpointRef(
            object_id=path.name,
            relative_path=path.relative_to(self.root).as_posix(),
            sha256=integrity["sha256"],
            size_bytes=integrity["size_bytes"],
            global_update=int(global_update),
            epoch=int(epoch) if epoch is not None else None,
            created_at=_utc_now(),
            metadata=_to_primitive(metadata or {}),
        )

    def save(
        self,
        checkpoint: Any,
        *,
        global_update: int,
        epoch: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        pending: bool = False,
    ) -> CheckpointRef:
        path = self.objects_dir / self._object_name(global_update)
        if path.exists():  # UUID collisions are fantastically unlikely, but never overwrite an object.
            raise CheckpointError(f"immutable checkpoint object already exists: {path}")
        atomic_torch_save(checkpoint, path)
        reference = self._new_ref(path, global_update=global_update, epoch=epoch, metadata=metadata)
        if pending:
            self.mark_pending(reference)
        return reference

    def register_existing(
        self,
        source: os.PathLike[str] | str,
        *,
        global_update: int,
        epoch: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        pending: bool = False,
    ) -> CheckpointRef:
        """Copy an existing checkpoint into the immutable object store."""

        destination = self.objects_dir / self._object_name(global_update)
        atomic_copy(source, destination)
        reference = self._new_ref(destination, global_update=global_update, epoch=epoch, metadata=metadata)
        if pending:
            self.mark_pending(reference)
        return reference

    def _path_for_ref(self, reference: CheckpointRef) -> Path:
        path = (self.root / reference.relative_path).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as error:
            raise CheckpointSchemaError(f"checkpoint reference escapes root: {reference.relative_path}") from error
        if path.parent != self.objects_dir.resolve():
            raise CheckpointSchemaError(f"checkpoint object is outside objects directory: {reference.relative_path}")
        if path.name != reference.object_id:
            raise CheckpointSchemaError("checkpoint object_id does not match relative_path")
        return path

    def verify(self, reference: CheckpointRef | Mapping[str, Any]) -> Path:
        reference = reference if isinstance(reference, CheckpointRef) else CheckpointRef.from_dict(reference)
        path = self._path_for_ref(reference)
        verify_file_integrity(path, sha256=reference.sha256, size_bytes=reference.size_bytes)
        return path

    def alias_path(self, role: str) -> Path:
        self._validate_role(role)
        return self.aliases_dir / f"{role}.json"

    def read_alias(self, role: str, *, verify: bool = True) -> dict[str, Any] | None:
        path = self.alias_path(role)
        if not path.exists():
            return None
        try:
            alias = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointSchemaError(f"cannot read checkpoint alias {path}: {error}") from error
        if alias.get("schema_version") != CHECKPOINT_SCHEMA_VERSION or alias.get("role") != role:
            raise CheckpointSchemaError(f"invalid {role!r} checkpoint alias")
        reference = CheckpointRef.from_dict(alias.get("checkpoint", {}))
        if verify:
            self.verify(reference)
            receipt = alias.get("promotion_receipt")
            if not isinstance(receipt, Mapping):
                raise CheckpointSchemaError(f"{role!r} checkpoint alias has no promotion receipt")
            receipt_path = (self.root / str(receipt.get("relative_path", ""))).resolve()
            try:
                receipt_path.relative_to(self.promotions_dir.resolve())
            except ValueError as error:
                raise CheckpointSchemaError("promotion receipt escapes its state directory") from error
            verify_file_integrity(
                receipt_path,
                sha256=str(receipt.get("sha256", "")),
                size_bytes=int(receipt.get("size_bytes", -1)),
            )
            try:
                receipt_document = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise CheckpointSchemaError(f"cannot read promotion receipt {receipt_path}: {error}") from error
            if (
                receipt_document.get("role") != role
                or receipt_document.get("checkpoint") != reference.to_dict()
                or receipt_document.get("promotion") != alias.get("promotion")
            ):
                raise CheckpointSchemaError("promotion receipt does not match its checkpoint alias")
        return alias

    def _write_promotion_receipt(
        self,
        role: str,
        reference: CheckpointRef,
        promotion: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Publish the metric evidence for one role without modifying the checkpoint object."""

        receipt_path = self.promotions_dir / f"{reference.object_id}.{role}.json"
        atomic_json_dump(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "role": role,
                "recorded_at": _utc_now(),
                "checkpoint": reference.to_dict(),
                "promotion": _to_primitive(promotion),
            },
            receipt_path,
        )
        integrity = file_integrity(receipt_path)
        return {
            "relative_path": receipt_path.relative_to(self.root).as_posix(),
            **integrity,
        }

    def promote(
        self,
        role: str,
        reference: CheckpointRef | Mapping[str, Any],
        *,
        metric_name: str | None = None,
        metric_value: float | None = None,
        mode: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        tie_break: str = "newer",
        force: bool = False,
        clear_pending: bool = False,
    ) -> bool:
        self._validate_role(role)
        reference = reference if isinstance(reference, CheckpointRef) else CheckpointRef.from_dict(reference)
        self.verify(reference)
        if role != "latest":
            if metric_name is None or metric_value is None or mode not in ("min", "max"):
                raise ValueError(
                    "best checkpoint promotion requires metric_name, metric_value and mode ('min' or 'max')"
                )
        elif mode is not None and mode not in ("min", "max"):
            raise ValueError("mode must be 'min', 'max', or None")

        lock_path = self.state_dir / f"{role}.lock"
        with _advisory_lock(lock_path):
            incumbent = self.read_alias(role, verify=True)
            promote = force or incumbent is None
            if not promote and role == "latest":
                incumbent_ref = CheckpointRef.from_dict(incumbent["checkpoint"])
                promote = reference.global_update >= incumbent_ref.global_update
            elif not promote:
                incumbent_ref = CheckpointRef.from_dict(incumbent["checkpoint"])
                incumbent_promotion = incumbent.get("promotion", {})
                if incumbent_promotion.get("metric_name") != metric_name or incumbent_promotion.get("mode") != mode:
                    raise CheckpointSchemaError(
                        f"cannot compare {role} using {metric_name!r}/{mode!r}; incumbent uses "
                        f"{incumbent_promotion.get('metric_name')!r}/{incumbent_promotion.get('mode')!r}"
                    )
                promote = should_promote(
                    float(metric_value),
                    incumbent_promotion.get("metric_value"),
                    mode=str(mode),
                    candidate_step=reference.global_update,
                    incumbent_step=incumbent_ref.global_update,
                    tie_break=tie_break,
                )
            if not promote:
                return False

            promotion = {
                "metric_name": metric_name,
                "metric_value": float(metric_value) if metric_value is not None else None,
                "mode": mode,
                "tie_break": tie_break,
                "metrics": _to_primitive(metrics or {}),
            }
            promotion_receipt = self._write_promotion_receipt(role, reference, promotion)
            atomic_json_dump(
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "role": role,
                    "updated_at": _utc_now(),
                    "checkpoint": reference.to_dict(),
                    "promotion": promotion,
                    "promotion_receipt": promotion_receipt,
                },
                self.alias_path(role),
            )
        if clear_pending:
            self.unmark_pending(reference.object_id)
        return True

    def resolve(self, role: str, *, verify: bool = True) -> Path:
        alias = self.read_alias(role, verify=verify)
        if alias is None:
            raise FileNotFoundError(f"checkpoint role {role!r} has not been assigned")
        return self._path_for_ref(CheckpointRef.from_dict(alias["checkpoint"]))

    def load(self, role: str, *, map_location: Any = "cpu", verify: bool = True, **load_kwargs: Any) -> Any:
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise CheckpointError("CheckpointManager.load requires PyTorch") from error
        path = self.resolve(role, verify=verify)
        if "weights_only" not in load_kwargs:
            load_kwargs["weights_only"] = False
        try:
            return torch.load(path, map_location=map_location, **load_kwargs)
        except TypeError:
            load_kwargs.pop("weights_only", None)
            return torch.load(path, map_location=map_location, **load_kwargs)

    def repair_embedded_promotions(self, *, source_role: str = "latest") -> dict[str, Any]:
        """Idempotently repair semantic roles from metrics embedded in a published alias.

        Checkpoint publication intentionally makes ``latest`` visible before
        the optional best-role aliases.  A process loss in that narrow window
        must not discard a scientifically valid rollout promotion: its exact
        metric payload is already checksum-bound in the immutable checkpoint
        reference metadata.  Planning promotions are reconciled separately
        from their durable evaluation registry.
        """

        alias = self.read_alias(source_role, verify=True)
        if alias is None:
            return {"source_role": source_role, "repaired": {}}
        reference = CheckpointRef.from_dict(alias["checkpoint"])
        embedded = reference.metadata.get("promotion_metrics", {})
        if not isinstance(embedded, Mapping):
            raise CheckpointSchemaError("checkpoint promotion_metrics metadata must be a mapping")

        repaired: dict[str, Any] = {}
        rollout = embedded.get("best_rollout")
        if rollout is not None:
            if not isinstance(rollout, Mapping):
                raise CheckpointSchemaError("embedded best_rollout metric must be a mapping")
            metric_name = rollout.get("metric_name")
            metric_value = rollout.get("metric_value")
            mode = rollout.get("mode", "min")
            if not isinstance(metric_name, str) or not metric_name:
                raise CheckpointSchemaError("embedded best_rollout has no metric_name")
            if not isinstance(metric_value, (int, float)) or isinstance(metric_value, bool):
                raise CheckpointSchemaError("embedded best_rollout has no numeric metric_value")
            if mode not in ("min", "max"):
                raise CheckpointSchemaError("embedded best_rollout mode must be 'min' or 'max'")
            repaired["best_rollout"] = {
                "promoted": self.promote(
                    "best_rollout",
                    reference,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    mode=mode,
                    metrics=rollout,
                    tie_break="older",
                ),
                "checkpoint": reference.to_dict(),
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "mode": mode,
            }
        return {"source_role": source_role, "repaired": repaired}

    @property
    def pending_path(self) -> Path:
        return self.state_dir / "pending.json"

    def _read_pending(self) -> dict[str, Any]:
        if not self.pending_path.exists():
            return {"schema_version": CHECKPOINT_SCHEMA_VERSION, "candidates": {}}
        try:
            pending = json.loads(self.pending_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointSchemaError(f"cannot read pending checkpoint registry: {error}") from error
        if pending.get("schema_version") != CHECKPOINT_SCHEMA_VERSION or not isinstance(
            pending.get("candidates"), dict
        ):
            raise CheckpointSchemaError("invalid pending checkpoint registry")
        return pending

    def mark_pending(self, reference: CheckpointRef | Mapping[str, Any], *, reason: str | None = None) -> None:
        reference = reference if isinstance(reference, CheckpointRef) else CheckpointRef.from_dict(reference)
        self.verify(reference)
        with _advisory_lock(self.state_dir / "pending.lock"):
            pending = self._read_pending()
            pending["candidates"][reference.object_id] = {
                "checkpoint": reference.to_dict(),
                "reason": reason,
                "marked_at": _utc_now(),
            }
            atomic_json_dump(pending, self.pending_path)

    def unmark_pending(self, reference_or_id: CheckpointRef | str) -> bool:
        object_id = reference_or_id.object_id if isinstance(reference_or_id, CheckpointRef) else str(reference_or_id)
        with _advisory_lock(self.state_dir / "pending.lock"):
            pending = self._read_pending()
            existed = pending["candidates"].pop(object_id, None) is not None
            if existed:
                atomic_json_dump(pending, self.pending_path)
            return existed

    def garbage_collect(
        self,
        *,
        keep_object_ids: Iterable[str] = (),
        keep_recent: int | None = None,
        dry_run: bool = False,
    ) -> list[Path]:
        """Remove objects not referenced by roles, recency, or pending evaluation."""

        recent_count = self.keep_recent if keep_recent is None else int(keep_recent)
        if recent_count < 0:
            raise ValueError("keep_recent must be non-negative")

        retained = set(keep_object_ids)
        for role in CHECKPOINT_ROLES:
            alias = self.read_alias(role, verify=True)
            if alias is not None:
                retained.add(str(alias["checkpoint"]["object_id"]))
        pending = self._read_pending()
        retained.update(pending["candidates"])

        object_paths = sorted(self.objects_dir.glob(f"{self.prefix}-step-*.pth.tar"))
        if recent_count:
            retained.update(path.name for path in object_paths[-recent_count:])

        removed = []
        for path in object_paths:
            if path.name not in retained:
                removed.append(path)
                if not dry_run:
                    path.unlink()
                    for receipt in self.promotions_dir.glob(f"{path.name}.*.json"):
                        receipt.unlink()
        if removed and not dry_run:
            _fsync_directory(self.objects_dir)
            _fsync_directory(self.promotions_dir)
        return removed
