import pytest

from src.utils.mfu import training_performance_stats


def test_training_performance_stats_uses_dense_peak_and_global_batch():
    stats = training_performance_stats(
        flops_per_sample=1.0e12,
        local_batch_size=8,
        world_size=32,
        gpu_elapsed_ms=8.0,
        wall_elapsed_ms=10.0,
        peak_dense_tflops=1000.0,
    )

    assert stats["perf/mfu_dense"] == pytest.approx(0.8)
    assert stats["perf/mfu_dense_gpu_active"] == pytest.approx(1.0)
    assert stats["perf/global_samples_per_sec"] == pytest.approx(25_600.0)
    assert stats["perf/gpu_duty_cycle"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "field,value",
    [
        ("flops_per_sample", 0.0),
        ("local_batch_size", 0),
        ("world_size", 0),
        ("gpu_elapsed_ms", -1.0),
        ("wall_elapsed_ms", float("nan")),
        ("peak_dense_tflops", 0.0),
    ],
)
def test_training_performance_stats_rejects_invalid_inputs(field, value):
    kwargs = {
        "flops_per_sample": 1.0,
        "local_batch_size": 1,
        "world_size": 1,
        "gpu_elapsed_ms": 1.0,
        "wall_elapsed_ms": 1.0,
        "peak_dense_tflops": 1.0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        training_performance_stats(**kwargs)
