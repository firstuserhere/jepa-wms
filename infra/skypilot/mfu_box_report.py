#!/usr/bin/env python3
"""Publish a checksum-protected decision record for a short MFU box."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.utils.checkpointing import atomic_json_dump, sha256_file
from src.utils.planning_promotion import canonical_sha256
from src.utils.training_telemetry import validate_telemetry_snapshot_document


def build_report(
    snapshot: Mapping[str, Any],
    *,
    snapshot_path: str | Path,
    expected_gpus: int = 8,
    minimum_cumulative_mfu: float = 0.30,
    minimum_window_mfu: float = 0.40,
    minimum_gpu_region_fraction: float = 0.80,
    maximum_batch_wait_fraction: float = 0.10,
) -> dict[str, Any]:
    summary = snapshot.get("summary", {})
    heartbeat = float(summary.get("pantheon_telemetry/heartbeat_at"))
    validate_telemetry_snapshot_document(snapshot, now=heartbeat)
    diagnostics = snapshot.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("MFU boxing snapshot has no bottleneck diagnostics")

    cumulative_mfu = float(summary["pantheon_telemetry/effective_mfu"])
    window_mfu = float(diagnostics["perf/effective_mfu_window"])
    step_time = float(diagnostics["perf/step_time_s"])
    gpu_region_fraction = float(diagnostics["perf/gpu_train_region_fraction"])
    batch_wait_fraction = float(diagnostics["perf/batch_wait_fraction"])
    gpu_count = int(snapshot["mfu_terms"]["gpu_count"])
    active_step = int(summary["pantheon_telemetry/active_step"])
    total_steps = int(summary["pantheon_telemetry/total_steps"])
    oom_kills = int(diagnostics.get("system/cgroup_memory_events_oom_kill", 0))
    checks = {
        "box_completed": active_step == total_steps,
        "expected_gpu_count": gpu_count == expected_gpus,
        "cumulative_effective_mfu": cumulative_mfu >= minimum_cumulative_mfu,
        "post_warmup_window_mfu": window_mfu >= minimum_window_mfu,
        "gpu_train_region_fraction": gpu_region_fraction >= minimum_gpu_region_fraction,
        "batch_wait_fraction": batch_wait_fraction <= maximum_batch_wait_fraction,
        "no_cgroup_oom_kill": oom_kills == 0,
    }
    scale_candidate = all(checks.values())
    report = {
        "schema_version": 1,
        "kind": "jepa-wm-single-node-mfu-box",
        "status": "scale-candidate" if scale_candidate else "needs-optimization",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scale_candidate": scale_candidate,
        "hardware_gpu_utilization_review_required": True,
        "checks": checks,
        "thresholds": {
            "expected_gpus": expected_gpus,
            "minimum_cumulative_mfu": minimum_cumulative_mfu,
            "minimum_window_mfu": minimum_window_mfu,
            "minimum_gpu_region_fraction": minimum_gpu_region_fraction,
            "maximum_batch_wait_fraction": maximum_batch_wait_fraction,
        },
        "metrics": {
            "cumulative_effective_mfu": cumulative_mfu,
            "post_warmup_window_mfu": window_mfu,
            "step_time_s": step_time,
            "gpu_train_region_fraction": gpu_region_fraction,
            "batch_wait_fraction": batch_wait_fraction,
            "gpu_count": gpu_count,
            "active_step": active_step,
            "total_steps": total_steps,
            "cgroup_oom_kill_count": oom_kills,
            "wandb_url": summary["pantheon_telemetry/wandb_url"],
        },
        "telemetry_snapshot": {
            "path": str(Path(snapshot_path).resolve()),
            "sha256": sha256_file(snapshot_path),
        },
    }
    report["integrity_sha256"] = canonical_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--minimum-cumulative-mfu", type=float, default=0.30)
    parser.add_argument("--minimum-window-mfu", type=float, default=0.40)
    parser.add_argument("--minimum-gpu-region-fraction", type=float, default=0.80)
    parser.add_argument("--maximum-batch-wait-fraction", type=float, default=0.10)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    report = build_report(
        snapshot,
        snapshot_path=args.snapshot,
        expected_gpus=args.expected_gpus,
        minimum_cumulative_mfu=args.minimum_cumulative_mfu,
        minimum_window_mfu=args.minimum_window_mfu,
        minimum_gpu_region_fraction=args.minimum_gpu_region_fraction,
        maximum_batch_wait_fraction=args.maximum_batch_wait_fraction,
    )
    atomic_json_dump(report, args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
