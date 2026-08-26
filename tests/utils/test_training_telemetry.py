import pytest

from src.utils.training_telemetry import (
    EffectiveMFUAccumulator,
    PantheonWandbHeartbeat,
    TELEMETRY_CONTRACT_VERSION,
    read_cgroup_memory_metrics,
    telemetry_provenance,
    telemetry_snapshot_document,
    validate_telemetry_snapshot_document,
)


class FakeRun:
    def __init__(self):
        self.url = "https://wandb.ai/entity/project/runs/experiment-001"
        self.summary = {}


def test_effective_mfu_is_ratio_of_cumulative_sums_and_charges_warmup_wall():
    accumulator = EffectiveMFUAccumulator(
        total_steps=100,
        gpu_count=8,
        peak_flops_per_gpu=100.0,
        started_at=0.0,
        clock=lambda: 10.0,
    )
    accumulator.record_samples(16)
    accumulator.record_samples(16)
    accumulator.set_flops_per_sample(100.0)
    snapshot = accumulator.snapshot(active_step=2, step_time_s=5.0, elapsed_e2e_s=10.0)

    assert snapshot.model_flops == 3200.0
    assert snapshot.effective_mfu == pytest.approx(0.4)
    assert snapshot.actual_global_samples == 32
    assert snapshot.history()["train/samples_per_second"] == pytest.approx(3.2)


def test_impossible_mfu_fails_instead_of_clamping():
    accumulator = EffectiveMFUAccumulator(
        total_steps=1,
        gpu_count=1,
        peak_flops_per_gpu=1.0,
        started_at=0.0,
        clock=lambda: 1.0,
    )
    accumulator.set_flops_per_sample(2.0)
    accumulator.record_samples(1)
    with pytest.raises(ValueError, match="Impossible effective MFU"):
        accumulator.snapshot(active_step=1, step_time_s=1.0, elapsed_e2e_s=1.0)


def test_training_v1_heartbeat_uses_exact_summary_fields():
    run = FakeRun()
    accumulator = EffectiveMFUAccumulator(
        total_steps=10,
        gpu_count=2,
        peak_flops_per_gpu=100.0,
        started_at=0.0,
        clock=lambda: 2.0,
    )
    accumulator.set_flops_per_sample(50.0)
    accumulator.record_samples(4)
    snapshot = accumulator.snapshot(active_step=1, step_time_s=2.0, elapsed_e2e_s=2.0)
    heartbeat = PantheonWandbHeartbeat(run, total_steps=10, clock=lambda: 1234.5)
    heartbeat.update(snapshot)

    assert run.summary == {
        "pantheon_telemetry/contract_version": TELEMETRY_CONTRACT_VERSION,
        "pantheon_telemetry/heartbeat_at": 1234.5,
        "pantheon_telemetry/total_steps": 10,
        "pantheon_telemetry/active_step": 1,
        "pantheon_telemetry/step_time_s": 2.0,
        "pantheon_telemetry/effective_mfu": 0.5,
        "pantheon_telemetry/wandb_url": run.url,
    }


def test_snapshot_contains_independently_recomputable_raw_terms():
    run = FakeRun()
    accumulator = EffectiveMFUAccumulator(
        total_steps=10,
        gpu_count=2,
        peak_flops_per_gpu=989e12,
        started_at=0.0,
        clock=lambda: 2.0,
    )
    accumulator.set_flops_per_sample(0.25 * 2.0 * 2 * 989e12 / 4)
    accumulator.record_samples(4)
    snapshot = accumulator.snapshot(active_step=1, step_time_s=2.0, elapsed_e2e_s=2.0)
    provenance = telemetry_provenance(
        git_commit="a" * 40,
        world_size=2,
        peak_flops_per_gpu=989e12,
        timed_wall_coverage=1.0,
    )

    document = telemetry_snapshot_document(
        snapshot,
        wandb_url=run.url,
        provenance=provenance,
        validation_started=True,
        diagnostics={"perf/effective_mfu_window": 0.3},
        heartbeat_at=1234.5,
    )

    terms = document["mfu_terms"]
    recomputed = terms["useful_flops_by_precision"]["bf16"] / (
        terms["elapsed_s"] * terms["gpu_count"] * terms["peak_flops_per_gpu"]["bf16"]
    )
    assert recomputed == pytest.approx(0.25)
    assert "train/loss" in document["history_keys"]
    assert "val/loss" in document["history_keys"]
    assert document["diagnostics"]["perf/effective_mfu_window"] == 0.3
    validate_telemetry_snapshot_document(document, expected_active_step=1, now=1234.5)

    document["mfu_terms"]["useful_flops_by_precision"]["bf16"] *= 2
    with pytest.raises(ValueError, match="does not match"):
        validate_telemetry_snapshot_document(document, expected_active_step=1, now=1234.5)


def test_cgroup_memory_metrics_use_anon_and_events(tmp_path):
    (tmp_path / "memory.current").write_text("1200\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text("2400\n", encoding="utf-8")
    (tmp_path / "memory.stat").write_text("anon 700\nfile 400\nkernel 100\n", encoding="utf-8")
    (tmp_path / "memory.events").write_text(
        "low 0\nhigh 2\nmax 3\noom 1\noom_kill 1\n", encoding="utf-8"
    )

    assert read_cgroup_memory_metrics(tmp_path) == {
        "system/cgroup_memory_current_bytes": 1200,
        "system/cgroup_memory_limit_bytes": 2400,
        "system/cgroup_memory_anon_bytes": 700,
        "system/cgroup_memory_file_bytes": 400,
        "system/cgroup_memory_events_high": 2,
        "system/cgroup_memory_events_max": 3,
        "system/cgroup_memory_events_oom": 1,
        "system/cgroup_memory_events_oom_kill": 1,
    }


def test_missing_cgroup_files_are_nonfatal(tmp_path):
    assert read_cgroup_memory_metrics(tmp_path) == {}
