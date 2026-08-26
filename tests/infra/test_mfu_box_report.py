import json

from infra.skypilot.mfu_box_report import build_report


def _snapshot(*, cumulative_mfu=0.35, window_mfu=0.45, gpu_fraction=0.9, batch_wait=0.1):
    elapsed = 10.0
    gpu_count = 8
    peak = 989e12
    return {
        "summary": {
            "pantheon_telemetry/contract_version": "training-v1",
            "pantheon_telemetry/heartbeat_at": 1234.5,
            "pantheon_telemetry/total_steps": 96,
            "pantheon_telemetry/active_step": 96,
            "pantheon_telemetry/step_time_s": 2.0,
            "pantheon_telemetry/effective_mfu": cumulative_mfu,
            "pantheon_telemetry/wandb_url": "https://wandb.ai/entity/project/runs/mfu-box",
        },
        "history_keys": [
            "train/global_step",
            "train/loss",
            "perf/effective_mfu",
            "perf/model_flops",
            "perf/elapsed_e2e_s",
            "perf/gpu_count",
            "perf/peak_flops_per_gpu",
            "perf/timed_wall_coverage",
            "perf/actual_global_samples",
        ],
        "validation_started": False,
        "mfu_terms": {
            "elapsed_s": elapsed,
            "gpu_count": gpu_count,
            "useful_flops_by_precision": {"bf16": cumulative_mfu * elapsed * gpu_count * peak},
            "peak_flops_per_gpu": {"bf16": peak},
        },
        "provenance": {
            "formula_id": "jepa-wm-profiled-useful-ops",
            "formula_version": "2",
            "git_commit": "a" * 40,
            "gpu_sku": "NVIDIA H200",
            "gpu_form_factor": "SXM",
            "precision": "bf16",
            "sparsity": "dense",
            "peak_flops_per_gpu": peak,
            "peak_source": "NVIDIA-H200-SXM-dense-BF16-2026-08",
            "world_size": gpu_count,
            "timing_scope": "end-to-end-slowest-rank-contiguous-allocation",
            "timed_wall_coverage": 1.0,
        },
        "diagnostics": {
            "perf/effective_mfu_window": window_mfu,
            "perf/step_time_s": 2.0,
            "perf/gpu_train_region_fraction": gpu_fraction,
            "perf/batch_wait_s": batch_wait,
            "perf/batch_wait_fraction": batch_wait / 2.0,
            "system/cgroup_memory_events_oom_kill": 0,
        },
    }


def test_box_report_marks_only_measured_candidate(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(_snapshot()), encoding="utf-8")
    report = build_report(_snapshot(), snapshot_path=path)

    assert report["status"] == "scale-candidate"
    assert report["scale_candidate"] is True
    assert report["hardware_gpu_utilization_review_required"] is True
    assert len(report["integrity_sha256"]) == 64


def test_box_report_identifies_data_wait_bottleneck(tmp_path):
    snapshot = _snapshot(batch_wait=0.5)
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    report = build_report(snapshot, snapshot_path=path)

    assert report["status"] == "needs-optimization"
    assert report["checks"]["batch_wait_fraction"] is False
