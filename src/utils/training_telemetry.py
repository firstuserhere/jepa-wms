"""Pantheon training-v1 telemetry and cumulative effective MFU accounting."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.parse import urlparse


TELEMETRY_CONTRACT_VERSION = "training-v1"
MFU_FORMULA_ID = "jepa-wm-profiled-useful-ops"
MFU_FORMULA_VERSION = "2"


@dataclass(frozen=True)
class EffectiveMFUSnapshot:
    active_step: int
    total_steps: int
    step_time_s: float
    effective_mfu: float
    model_flops: float
    elapsed_e2e_s: float
    gpu_count: int
    peak_flops_per_gpu: float
    timed_wall_coverage: float
    actual_global_samples: int

    def history(self) -> dict[str, float | int]:
        samples_per_second = (
            self.actual_global_samples / self.elapsed_e2e_s if self.elapsed_e2e_s > 0 else 0.0
        )
        return {
            "perf/effective_mfu": self.effective_mfu,
            "perf/model_flops": self.model_flops,
            "perf/elapsed_e2e_s": self.elapsed_e2e_s,
            "perf/gpu_count": self.gpu_count,
            "perf/peak_flops_per_gpu": self.peak_flops_per_gpu,
            "perf/timed_wall_coverage": self.timed_wall_coverage,
            "perf/actual_global_samples": self.actual_global_samples,
            "train/samples_per_second": samples_per_second,
        }


class EffectiveMFUAccumulator:
    """Ratio-of-sums effective MFU over one contiguous training allocation.

    The wall denominator starts before the first batch and therefore includes
    input stalls, collectives, optimizer work, logging, evaluation, and
    checkpoint publication. Samples seen before the representative FLOP profile
    is available are retained and charged retroactively once the fixed
    per-sample useful-work value is known.
    """

    def __init__(
        self,
        *,
        total_steps: int,
        gpu_count: int,
        peak_flops_per_gpu: float,
        started_at: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if total_steps <= 0 or gpu_count <= 0 or peak_flops_per_gpu <= 0:
            raise ValueError("MFU totals, GPU count, and peak FLOP/s must be positive")
        self.total_steps = int(total_steps)
        self.gpu_count = int(gpu_count)
        self.peak_flops_per_gpu = float(peak_flops_per_gpu)
        self._clock = clock
        self.started_at = float(clock() if started_at is None else started_at)
        self._flops_per_sample: float | None = None
        self._actual_global_samples = 0

    @property
    def flops_per_sample(self) -> float | None:
        return self._flops_per_sample

    def set_flops_per_sample(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Profiled useful FLOPs/sample must be finite and positive")
        if self._flops_per_sample is not None and not math.isclose(
            self._flops_per_sample, value, rel_tol=1e-9
        ):
            raise ValueError("The useful-FLOP ledger changed within one training run")
        self._flops_per_sample = value

    def record_samples(self, actual_global_samples: int) -> None:
        if actual_global_samples <= 0:
            raise ValueError("Actual global samples must be positive")
        self._actual_global_samples += int(actual_global_samples)

    def snapshot(
        self,
        *,
        active_step: int,
        step_time_s: float,
        elapsed_e2e_s: float | None = None,
        timed_wall_coverage: float = 1.0,
    ) -> EffectiveMFUSnapshot:
        elapsed = float(self._clock() - self.started_at if elapsed_e2e_s is None else elapsed_e2e_s)
        if elapsed <= 0 or not math.isfinite(elapsed):
            raise ValueError("End-to-end elapsed time must be finite and positive")
        if step_time_s <= 0 or not math.isfinite(step_time_s):
            raise ValueError("Step time must be finite and positive")
        if not 0.0 <= timed_wall_coverage <= 1.0:
            raise ValueError("Timed-wall coverage must be in [0, 1]")
        model_flops = (
            0.0
            if self._flops_per_sample is None
            else self._flops_per_sample * self._actual_global_samples
        )
        denominator = elapsed * self.gpu_count * self.peak_flops_per_gpu
        effective_mfu = model_flops / denominator
        if not math.isfinite(effective_mfu) or effective_mfu < 0:
            raise ValueError("Effective MFU is not finite and non-negative")
        if effective_mfu > 1:
            raise ValueError(
                f"Impossible effective MFU {effective_mfu:.4f}; audit useful FLOPs and peak basis"
            )
        return EffectiveMFUSnapshot(
            active_step=int(active_step),
            total_steps=self.total_steps,
            step_time_s=float(step_time_s),
            effective_mfu=effective_mfu,
            model_flops=model_flops,
            elapsed_e2e_s=elapsed,
            gpu_count=self.gpu_count,
            peak_flops_per_gpu=self.peak_flops_per_gpu,
            timed_wall_coverage=float(timed_wall_coverage),
            actual_global_samples=self._actual_global_samples,
        )


class PantheonWandbHeartbeat:
    """Refresh training-v1 summary fields from rank zero during long phases."""

    def __init__(
        self,
        run: Any,
        *,
        total_steps: int,
        interval_seconds: float = 10.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if run is None or not getattr(run, "url", None):
            raise ValueError("A canonical W&B run URL is required")
        if total_steps <= 0 or interval_seconds <= 0:
            raise ValueError("Heartbeat totals and interval must be positive")
        self.run = run
        self.total_steps = int(total_steps)
        self.interval_seconds = float(interval_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._latest: dict[str, Any] = {
            "active_step": 0,
            "step_time_s": 1e-9,
            "effective_mfu": 0.0,
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def update(self, snapshot: EffectiveMFUSnapshot) -> None:
        with self._lock:
            self._latest = {
                "active_step": snapshot.active_step,
                "step_time_s": snapshot.step_time_s,
                "effective_mfu": snapshot.effective_mfu,
            }
        self.publish()

    def publish(self) -> None:
        with self._lock:
            latest = dict(self._latest)
        self.run.summary.update(
            {
                "pantheon_telemetry/contract_version": TELEMETRY_CONTRACT_VERSION,
                "pantheon_telemetry/heartbeat_at": float(self._clock()),
                "pantheon_telemetry/total_steps": self.total_steps,
                "pantheon_telemetry/active_step": int(latest["active_step"]),
                "pantheon_telemetry/step_time_s": float(latest["step_time_s"]),
                "pantheon_telemetry/effective_mfu": float(latest["effective_mfu"]),
                "pantheon_telemetry/wandb_url": str(self.run.url),
            }
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        self.publish()

        def loop() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.publish()

        self._thread = threading.Thread(target=loop, name="pantheon-wandb-heartbeat", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.publish()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
            self._thread = None


def telemetry_provenance(
    *,
    git_commit: str,
    world_size: int,
    peak_flops_per_gpu: float,
    timed_wall_coverage: float,
) -> Mapping[str, Any]:
    return {
        "formula_id": MFU_FORMULA_ID,
        "formula_version": MFU_FORMULA_VERSION,
        "operator_ledger": (
            "torch.utils.flop_counter representative real DROID step sampled after "
            "forward/backward and before transition optimizer"
        ),
        "git_commit": git_commit,
        "gpu_sku": "NVIDIA H200",
        "gpu_form_factor": "SXM",
        "precision": "bf16",
        "sparsity": "dense",
        "peak_flops_per_gpu": float(peak_flops_per_gpu),
        "peak_source": "NVIDIA-H200-SXM-dense-BF16-2026-08",
        "world_size": int(world_size),
        "timing_scope": "end-to-end-slowest-rank-contiguous-allocation",
        "timed_wall_coverage": float(timed_wall_coverage),
        "numerator_scope": "unique global model forward+backward work; frozen encoder forward included",
        "recomputation_policy": "profiled executed model graph; transition optimizer excluded",
    }


def telemetry_snapshot_document(
    snapshot: EffectiveMFUSnapshot,
    *,
    wandb_url: str,
    provenance: Mapping[str, Any],
    validation_started: bool,
    heartbeat_at: float | None = None,
) -> dict[str, Any]:
    """Build the validator-compatible durable telemetry evidence document."""

    history_keys = [
        "train/global_step",
        "train/loss",
        "perf/effective_mfu",
        "perf/model_flops",
        "perf/elapsed_e2e_s",
        "perf/gpu_count",
        "perf/peak_flops_per_gpu",
        "perf/timed_wall_coverage",
        "perf/actual_global_samples",
        "train/samples_per_second",
    ]
    if validation_started:
        history_keys.append("val/loss")
    return {
        "summary": {
            "pantheon_telemetry/contract_version": TELEMETRY_CONTRACT_VERSION,
            "pantheon_telemetry/heartbeat_at": float(time.time() if heartbeat_at is None else heartbeat_at),
            "pantheon_telemetry/total_steps": snapshot.total_steps,
            "pantheon_telemetry/active_step": snapshot.active_step,
            "pantheon_telemetry/step_time_s": snapshot.step_time_s,
            "pantheon_telemetry/effective_mfu": snapshot.effective_mfu,
            "pantheon_telemetry/wandb_url": str(wandb_url),
        },
        "history_keys": history_keys,
        "validation_started": bool(validation_started),
        "mfu_terms": {
            "elapsed_s": snapshot.elapsed_e2e_s,
            "gpu_count": snapshot.gpu_count,
            "useful_flops_by_precision": {"bf16": snapshot.model_flops},
            "peak_flops_per_gpu": {"bf16": snapshot.peak_flops_per_gpu},
        },
        "provenance": dict(provenance),
    }


def validate_telemetry_snapshot_document(
    document: Mapping[str, Any],
    *,
    expected_active_step: int | None = None,
    now: float | None = None,
    max_heartbeat_age_seconds: float = 30.0,
) -> None:
    """Fail closed on the runtime subset of the training-v1 contract."""

    summary = document.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("training-v1 snapshot has no summary mapping")
    required_summary = {
        "pantheon_telemetry/contract_version",
        "pantheon_telemetry/heartbeat_at",
        "pantheon_telemetry/total_steps",
        "pantheon_telemetry/active_step",
        "pantheon_telemetry/step_time_s",
        "pantheon_telemetry/effective_mfu",
        "pantheon_telemetry/wandb_url",
    }
    missing = sorted(required_summary - set(summary))
    if missing:
        raise ValueError(f"training-v1 summary fields are missing: {missing}")
    if summary["pantheon_telemetry/contract_version"] != TELEMETRY_CONTRACT_VERSION:
        raise ValueError("unexpected Pantheon telemetry contract version")
    active_step = summary["pantheon_telemetry/active_step"]
    total_steps = summary["pantheon_telemetry/total_steps"]
    if (
        isinstance(active_step, bool)
        or not isinstance(active_step, int)
        or isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or not 0 <= active_step <= total_steps
    ):
        raise ValueError("training-v1 progress fields are invalid")
    if expected_active_step is not None and active_step != expected_active_step:
        raise ValueError("training-v1 active step does not match the checkpoint")
    heartbeat_at = float(summary["pantheon_telemetry/heartbeat_at"])
    current = time.time() if now is None else float(now)
    if not math.isfinite(heartbeat_at) or not -5 <= current - heartbeat_at <= max_heartbeat_age_seconds:
        raise ValueError("training-v1 heartbeat is stale or has invalid clock skew")
    step_time = float(summary["pantheon_telemetry/step_time_s"])
    effective_mfu = float(summary["pantheon_telemetry/effective_mfu"])
    if not math.isfinite(step_time) or step_time <= 0:
        raise ValueError("training-v1 step time is invalid")
    if not math.isfinite(effective_mfu) or not 0 <= effective_mfu <= 1:
        raise ValueError("training-v1 effective MFU is invalid")
    parsed = urlparse(str(summary["pantheon_telemetry/wandb_url"]))
    if parsed.scheme != "https" or not parsed.netloc or "/runs/" not in parsed.path:
        raise ValueError("training-v1 snapshot has no canonical W&B URL")

    history = document.get("history_keys")
    if not isinstance(history, list) or not any(str(key).startswith("train/loss") for key in history):
        raise ValueError("training-v1 snapshot has no training-loss history")
    for key in (
        "train/global_step",
        "perf/effective_mfu",
        "perf/model_flops",
        "perf/elapsed_e2e_s",
        "perf/gpu_count",
        "perf/peak_flops_per_gpu",
        "perf/timed_wall_coverage",
        "perf/actual_global_samples",
    ):
        if key not in history:
            raise ValueError(f"training-v1 history key is missing: {key}")
    if document.get("validation_started") is True and "val/loss" not in history:
        raise ValueError("training-v1 snapshot is missing val/loss after validation")

    terms = document.get("mfu_terms")
    provenance = document.get("provenance")
    if not isinstance(terms, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("training-v1 snapshot is missing raw MFU terms or provenance")
    elapsed = float(terms["elapsed_s"])
    gpu_count = int(terms["gpu_count"])
    useful = float(terms["useful_flops_by_precision"]["bf16"])
    peak = float(terms["peak_flops_per_gpu"]["bf16"])
    recomputed = useful / (elapsed * gpu_count * peak)
    if not math.isclose(recomputed, effective_mfu, rel_tol=1e-3, abs_tol=1e-4):
        raise ValueError("training-v1 effective MFU does not match its raw terms")
    if float(provenance.get("timed_wall_coverage", 0.0)) < 0.98:
        raise ValueError("training-v1 timed wall coverage is below 98%")
    timing_scope = str(provenance.get("timing_scope", "")).lower()
    if "end-to-end" not in timing_scope or "slowest" not in timing_scope:
        raise ValueError("training-v1 timing scope is not end-to-end slowest-rank")
