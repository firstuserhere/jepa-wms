#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""``torchrun`` entrypoint for local and SkyPilot distributed jobs.

The upstream launcher predates ``torchrun`` and starts local child processes
itself.  This module consumes the rank environment created by ``torchrun`` and
starts exactly one JEPA-WM worker in each process.  It also supports small YAML
overlays so the released scientific configs remain unchanged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


def _document_integrity(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("integrity_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _with_integrity(document: dict[str, Any]) -> dict[str, Any]:
    payload = dict(document)
    payload["integrity_sha256"] = _document_integrity(payload)
    return payload


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; lists and scalar values are replaced."""

    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_base_config(config_path: Path, base_config: str) -> Path:
    candidate = Path(base_config).expanduser()
    if candidate.is_absolute():
        return candidate
    relative_to_overlay = (config_path.parent / candidate).resolve()
    if relative_to_overlay.exists():
        return relative_to_overlay
    return (Path.cwd() / candidate).resolve()


def load_config(config_path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a normal config or a recursive ``base_config``/``overrides`` overlay."""

    import yaml

    resolved_path = Path(config_path).expanduser().resolve()
    seen = set() if _seen is None else _seen
    if resolved_path in seen:
        chain = " -> ".join(str(item) for item in [*seen, resolved_path])
        raise ValueError(f"Config overlay cycle detected: {chain}")
    seen.add(resolved_path)
    try:
        with resolved_path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        if not isinstance(document, dict):
            raise TypeError(f"Config must contain a YAML mapping: {resolved_path}")

        if "base_config" not in document:
            return document

        unexpected = set(document) - {"base_config", "overrides"}
        if unexpected:
            raise ValueError(
                f"Overlay {resolved_path} has unexpected top-level keys: {sorted(unexpected)}; "
                "put all changes under 'overrides'"
            )
        base_config = document["base_config"]
        overrides = document.get("overrides", {})
        if not isinstance(base_config, str) or not isinstance(overrides, dict):
            raise TypeError("Overlay requires a string 'base_config' and mapping 'overrides'")
        base_path = _resolve_base_config(resolved_path, base_config)
        return _deep_merge(load_config(base_path, seen), overrides)
    finally:
        seen.remove(resolved_path)


def _require_environment(names: list[str]) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Required environment variables are unavailable: {', '.join(missing)}")


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _prepare_run_identity(params: dict[str, Any], *, resume_existing: bool) -> None:
    """Fail closed before a new task can merge with an existing durable run."""

    if int(os.environ.get("RANK", "0")) != 0:
        return
    run_id = os.environ.get("JEPAWM_RUN_ID", "")
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise RuntimeError("JEPAWM_RUN_ID must be a non-empty, filesystem-safe run identifier")
    folder = Path(params["folder"]).expanduser().resolve()
    checkpoint_folder = Path(params.get("checkpoint_folder", params["folder"])).expanduser().resolve()
    for label, path in (("folder", folder), ("checkpoint_folder", checkpoint_folder)):
        if run_id not in path.parts:
            raise RuntimeError(f"Resolved {label} is not scoped by JEPAWM_RUN_ID")

    metadata_dir = checkpoint_folder / "run_metadata"
    identity_path = metadata_dir / "launch_identity.json"
    # Sky's ID is stable across managed recovery.  A local process has no such
    # durable identity, so never treat a later local invocation as an implicit
    # recovery of the first.
    task_id = os.environ.get("SKYPILOT_TASK_ID") or f"local-process-{os.getpid()}"
    existing_identity = None
    if identity_path.exists():
        with identity_path.open("r", encoding="utf-8") as stream:
            existing_identity = json.load(stream)
        if existing_identity.get("run_id") != run_id:
            raise RuntimeError("Durable launch identity does not match JEPAWM_RUN_ID")
        same_managed_task = existing_identity.get("origin_task_id") == task_id
        if not same_managed_task and not resume_existing:
            raise RuntimeError(
                "Run already belongs to another task; pass --resume-existing only for intentional continuation"
            )
        return

    def has_entries(path: Path) -> bool:
        return path.exists() and any(path.iterdir())

    if (has_entries(folder) or has_entries(checkpoint_folder)) and not resume_existing:
        raise RuntimeError(
            "Run directory already exists without an unambiguous task identity; refusing to merge. "
            "Use a new JEPAWM_RUN_ID or pass --resume-existing for intentional continuation."
        )
    metadata_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        identity_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "origin_task_id": task_id,
        },
    )


def _checkpoint_snapshot(params: dict[str, Any]) -> dict[str, Any]:
    """Verify the latest alias/object and return continuation evidence."""

    from src.utils.checkpointing import CheckpointManager, validate_checkpoint_v2

    prefix = params.get("logging", {}).get("write_tag", "jepa") or "jepa"
    checkpoint_folder = Path(params.get("checkpoint_folder", params["folder"])).expanduser().resolve()
    manager = CheckpointManager(checkpoint_folder, prefix=prefix)
    alias = manager.read_alias("latest", verify=True)
    if alias is None:
        raise FileNotFoundError("latest checkpoint alias has not been published")
    checkpoint = manager.load("latest", map_location="cpu", verify=True)
    validate_checkpoint_v2(checkpoint)
    reference = alias["checkpoint"]
    manifest = checkpoint["manifest"]
    timing_path = checkpoint_folder / "run_metadata" / "epoch_boundary_timing.json"
    telemetry_path = (
        checkpoint_folder
        / "run_metadata"
        / f"training_v1_step-{int(checkpoint['progress']['global_update']):012d}.json"
    )
    try:
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise FileNotFoundError(
            "latest checkpoint is not yet accompanied by epoch timing and training-v1 telemetry"
        ) from error
    if timing.get("epoch") != int(checkpoint["progress"]["epoch"]):
        raise RuntimeError("epoch-boundary timing does not describe the latest checkpoint")
    summary = telemetry.get("summary", {})
    if summary.get("pantheon_telemetry/contract_version") != "training-v1":
        raise RuntimeError("latest checkpoint has no valid training-v1 telemetry snapshot")
    if summary.get("pantheon_telemetry/active_step") != int(checkpoint["progress"]["global_update"]):
        raise RuntimeError("training-v1 telemetry does not describe the latest checkpoint")
    telemetry_sha256 = hashlib.sha256(telemetry_path.read_bytes()).hexdigest()
    return {
        "epoch": int(checkpoint["progress"]["epoch"]),
        "global_update": int(checkpoint["progress"]["global_update"]),
        "object_id": reference["object_id"],
        "sha256": reference["sha256"],
        "wandb_run_id": manifest["wandb"]["run_id"],
        "checkpoint_schema_version": int(checkpoint["schema_version"]),
        "source_git_commit": manifest["git"]["commit"],
        "dataset_fingerprint": manifest["dataset"]["dataset_fingerprint"],
        "encoder_repo_revision": manifest["encoder"]["repo_revision"],
        "encoder_weights_sha256": manifest["encoder"]["weights_sha256"],
        "resume_contract_sha256": manifest["resume_contract"]["sha256"],
        "epoch_boundary_only": timing.get("epoch_boundary_only"),
        "epoch_boundary_seconds": timing.get("epoch_boundary_seconds"),
        "checkpoint_seconds": timing.get("checkpoint_seconds"),
        "checkpoint_loss_budget_seconds": timing.get("loss_budget_seconds"),
        "checkpoint_within_loss_budget": timing.get("within_loss_budget"),
        "telemetry_snapshot_path": str(telemetry_path),
        "telemetry_snapshot_sha256": telemetry_sha256,
        "telemetry_effective_mfu": summary.get("pantheon_telemetry/effective_mfu"),
        "telemetry_wandb_url": summary.get("pantheon_telemetry/wandb_url"),
    }


def _start_checkpoint_recovery_probe(
    params: dict[str, Any], *, stop_after_epoch: int, exit_code: int
) -> tuple[Path, dict[str, Any] | None]:
    """Arm a rank-zero epoch-boundary stop, or validate its durable first phase."""

    checkpoint_folder = Path(params.get("checkpoint_folder", params["folder"])).expanduser().resolve()
    state_path = checkpoint_folder / "run_metadata" / "training_recovery_smoke.json"
    prior = None
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as stream:
            prior = json.load(stream)
    if prior and prior.get("phase") == "complete":
        # A later receipt-publication step may have been interrupted after the
        # expensive training smoke completed. Treat the terminal smoke state as
        # idempotent so managed recovery can retry only the publication gate.
        return state_path, prior
    if prior and prior.get("phase") == "armed":
        snapshot = _checkpoint_snapshot(params)
        if snapshot != prior.get("first_checkpoint"):
            raise RuntimeError("Recovery target changed after the controlled phase-one stop")
        return state_path, prior

    if int(os.environ.get("RANK", "0")) == 0:

        def monitor() -> None:
            while True:
                try:
                    snapshot = _checkpoint_snapshot(params)
                except (FileNotFoundError, OSError, RuntimeError, ValueError):
                    time.sleep(0.1)
                    continue
                if snapshot["epoch"] < stop_after_epoch:
                    time.sleep(0.1)
                    continue
                if snapshot["epoch"] != stop_after_epoch:
                    _atomic_write_json(
                        state_path,
                        _with_integrity(
                            {"schema_version": 1, "phase": "failed", "observed_checkpoint": snapshot}
                        ),
                    )
                    os._exit(43)
                _atomic_write_json(
                    state_path,
                    _with_integrity(
                        {
                            "schema_version": 1,
                            "kind": "training-checkpoint-recovery-smoke",
                            "phase": "armed",
                            "first_checkpoint": snapshot,
                        }
                    ),
                )
                os._exit(exit_code)

        threading.Thread(target=monitor, name="checkpoint-recovery-probe", daemon=True).start()
    return state_path, None


def _complete_checkpoint_recovery_probe(
    params: dict[str, Any], state_path: Path, prior: dict[str, Any] | None, *, verify_epoch: int
) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    if not prior or prior.get("phase") != "armed":
        raise RuntimeError("Training completed without exercising the controlled recovery boundary")
    first = prior["first_checkpoint"]
    final = _checkpoint_snapshot(params)
    if final["epoch"] < verify_epoch or final["global_update"] <= first["global_update"]:
        raise RuntimeError("Recovered training did not advance through the required epoch/update")
    if final["wandb_run_id"] != first["wandb_run_id"]:
        raise RuntimeError("Recovered training changed W&B run identity")
    for provenance_key in (
        "source_git_commit",
        "dataset_fingerprint",
        "encoder_repo_revision",
        "encoder_weights_sha256",
        "resume_contract_sha256",
    ):
        if final[provenance_key] != first[provenance_key]:
            raise RuntimeError(f"Recovered training changed {provenance_key}")
    _atomic_write_json(
        state_path,
        _with_integrity(
            {
                "schema_version": 1,
                "kind": "training-checkpoint-recovery-smoke",
                "phase": "complete",
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "first_checkpoint": first,
                "final_checkpoint": final,
                "strict_resume_verified": True,
            }
        ),
    )


def _stable_wandb_id(task_id: str) -> str:
    return "infra-" + hashlib.sha256(f"jepa-wm:{task_id}".encode("utf-8")).hexdigest()[:24]


def run_smoke(args: argparse.Namespace) -> int:
    """Exercise CUDA, collectives, managed recovery, durable state, and W&B."""

    import torch
    import torch.distributed as dist
    import wandb

    from src.utils.distributed import init_distributed

    world_size, rank = init_distributed(nccl_timeout_minutes=args.nccl_timeout_minutes)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the distributed smoke test")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible")
    torch.cuda.set_device(local_rank)
    gpu_name = torch.cuda.get_device_name(local_rank)
    if args.expected_gpu and args.expected_gpu.lower() not in gpu_name.lower():
        raise RuntimeError(f"Expected GPU containing {args.expected_gpu!r}, found {gpu_name!r}")

    rank_value = torch.tensor(float(rank), device=f"cuda:{local_rank}")
    dist.all_reduce(rank_value, op=dist.ReduceOp.SUM)
    expected_sum = world_size * (world_size - 1) / 2
    if rank_value.item() != expected_sum:
        raise RuntimeError(f"Collective check failed: expected {expected_sum}, got {rank_value.item()}")

    source_git_commit = os.environ.get("GIT_COMMIT_HASH", "").lower()
    if not re.fullmatch(r"[0-9a-f]{40}", source_git_commit):
        raise RuntimeError("distributed smoke requires SkyPilot's immutable GIT_COMMIT_HASH")
    local_evidence = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "gpu_name": gpu_name,
    }
    rank_evidence = [None] * world_size
    dist.all_gather_object(rank_evidence, local_evidence)
    if sorted(item["rank"] for item in rank_evidence) != list(range(world_size)):
        raise RuntimeError("distributed smoke did not gather exactly one record from every rank")
    observed_nodes = len({item["hostname"] for item in rank_evidence})
    if args.expected_num_nodes and observed_nodes != args.expected_num_nodes:
        raise RuntimeError(
            f"Expected {args.expected_num_nodes} distributed nodes, observed {observed_nodes}"
        )

    task_id = os.environ.get("SKYPILOT_TASK_ID", "local-smoke")
    state_path = Path(args.smoke_state).expanduser().resolve()
    should_fail = False
    prior_phase = None
    if rank == 0:
        previous = None
        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as stream:
                previous = json.load(stream)
        same_attempt = isinstance(previous, dict) and previous.get("task_id") == task_id
        prior_phase = previous.get("phase") if same_attempt else None
        should_fail = bool(args.smoke_fail_once and prior_phase not in {"armed", "complete"})

        wandb_run = wandb.init(
            project=args.wandb_project,
            id=_stable_wandb_id(task_id),
            resume="allow",
            job_type="distributed-infrastructure-smoke",
            config={
                "world_size": world_size,
                "expected_gpu": args.expected_gpu,
                "managed_recovery_probe": bool(args.smoke_fail_once),
            },
        )
        wandb_run.log(
            {
                "smoke/collective_ok": 1,
                "smoke/world_size": world_size,
                "smoke/recovered": int(prior_phase == "armed"),
            }
        )
        wandb_run.finish()

        _atomic_write_json(
            state_path,
            _with_integrity(
                {
                    "schema_version": 1,
                    "kind": "distributed-recovery-smoke",
                    "task_id": task_id,
                    "run_id": os.environ.get("JEPAWM_RUN_ID", task_id),
                    "phase": "armed" if should_fail else "complete",
                    "completed_at": (
                        dt.datetime.now(dt.timezone.utc).isoformat() if not should_fail else None
                    ),
                    "source_git_commit": source_git_commit,
                    "world_size": world_size,
                    "observed_node_count": observed_nodes,
                    "rank_evidence": rank_evidence,
                    "wandb_run_id": _stable_wandb_id(task_id),
                    "recovered_from_controlled_failure": prior_phase == "armed",
                }
            ),
        )

    decision = torch.tensor(int(should_fail), device=f"cuda:{local_rank}")
    dist.broadcast(decision, src=0)
    dist.barrier()
    should_fail = bool(decision.item())
    dist.destroy_process_group()
    return args.recovery_exit_code if should_fail else 0


def _write_resolved_config(params: dict[str, Any]) -> None:
    import yaml

    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    folder = Path(params["folder"]).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / "params-pretrain.yaml"
    descriptor, temporary = tempfile.mkstemp(prefix=".params-pretrain.", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(params, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run_training(args: argparse.Namespace) -> int:
    from app.scaffold import main as app_main
    from src.utils.yaml_utils import expand_env_vars

    params = expand_env_vars(load_config(args.fname))
    if not isinstance(params.get("folder"), str):
        raise ValueError("Resolved training config must define a string 'folder'")
    if "app" not in params:
        raise ValueError("Resolved training config must define 'app'")
    _prepare_run_identity(params, resume_existing=args.resume_existing)
    probe_state = None
    probe_prior = None
    if args.checkpoint_recovery_smoke:
        probe_state, probe_prior = _start_checkpoint_recovery_probe(
            params,
            stop_after_epoch=args.smoke_stop_after_epoch,
            exit_code=args.recovery_exit_code,
        )
        if probe_prior and probe_prior.get("phase") == "complete":
            return 0
    _write_resolved_config(params)
    app_main(params["app"], args=params)
    if args.checkpoint_recovery_smoke:
        _complete_checkpoint_recovery_probe(
            params,
            probe_state,
            probe_prior,
            verify_epoch=args.smoke_verify_epoch,
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fname", help="Training YAML or base_config/overrides overlay")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Intentionally continue a pre-existing JEPAWM_RUN_ID owned by another managed task",
    )
    parser.add_argument(
        "--require-env",
        action="append",
        default=[],
        metavar="NAME",
        help="Fail closed when a required managed secret/environment variable is absent",
    )
    parser.add_argument("--smoke", action="store_true", help="Run collectives/recovery/W&B smoke checks")
    parser.add_argument("--smoke-state", help="Durable JSON state used by --smoke")
    parser.add_argument("--smoke-fail-once", action="store_true", help="Exit once to exercise managed recovery")
    parser.add_argument("--recovery-exit-code", type=int, default=42)
    parser.add_argument("--expected-gpu", default="H200")
    parser.add_argument("--expected-num-nodes", type=int)
    parser.add_argument("--wandb-project", default="vjepa_wm")
    parser.add_argument("--nccl-timeout-minutes", type=int, default=15)
    parser.add_argument(
        "--checkpoint-recovery-smoke",
        action="store_true",
        help="Exercise a controlled epoch-boundary v2 checkpoint stop and strict managed recovery",
    )
    parser.add_argument("--smoke-stop-after-epoch", type=int, default=1)
    parser.add_argument("--smoke-verify-epoch", type=int, default=2)
    parsed = parser.parse_args()
    if parsed.smoke and not parsed.smoke_state:
        parser.error("--smoke requires --smoke-state")
    if not parsed.smoke and not parsed.fname:
        parser.error("training requires --fname")
    if parsed.checkpoint_recovery_smoke and parsed.smoke:
        parser.error("--checkpoint-recovery-smoke is a training-only mode")
    if parsed.smoke_stop_after_epoch < 1 or parsed.smoke_verify_epoch <= parsed.smoke_stop_after_epoch:
        parser.error("checkpoint recovery epochs must be positive and strictly increasing")
    if not 0 <= parsed.recovery_exit_code <= 255:
        parser.error("--recovery-exit-code must be in [0, 255]")
    if parsed.expected_num_nodes is not None and parsed.expected_num_nodes <= 0:
        parser.error("--expected-num-nodes must be positive")
    return parsed


def main() -> int:
    args = parse_args()
    for name in args.require_env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid environment variable name: {name!r}")
    _require_environment(args.require_env)
    return run_smoke(args) if args.smoke else run_training(args)


if __name__ == "__main__":
    sys.exit(main())
