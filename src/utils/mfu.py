"""Model FLOPs utilization calculations for training telemetry."""

import math


def training_performance_stats(
    *,
    flops_per_sample: float,
    local_batch_size: int,
    world_size: int,
    gpu_elapsed_ms: float,
    wall_elapsed_ms: float,
    peak_dense_tflops: float,
) -> dict[str, float]:
    """Calculate dense MFU and throughput for one distributed training step.

    ``flops_per_sample`` is measured from the actual forward/backward step.
    Peak throughput must be the *dense* hardware figure for the active dtype;
    sparse marketing throughput would under-report MFU by two.
    """

    values = (flops_per_sample, gpu_elapsed_ms, wall_elapsed_ms, peak_dense_tflops)
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in values):
        raise ValueError("MFU inputs must be finite and positive")
    if local_batch_size <= 0 or world_size <= 0:
        raise ValueError("MFU batch size and world size must be positive")

    step_flops = float(flops_per_sample) * int(local_batch_size)
    gpu_seconds = float(gpu_elapsed_ms) / 1000.0
    wall_seconds = float(wall_elapsed_ms) / 1000.0
    achieved_wall_tflops = step_flops / wall_seconds / 1.0e12
    achieved_gpu_tflops = step_flops / gpu_seconds / 1.0e12

    return {
        "perf/mfu_dense": achieved_wall_tflops / float(peak_dense_tflops),
        "perf/mfu_dense_gpu_active": achieved_gpu_tflops / float(peak_dense_tflops),
        "perf/achieved_tflops_per_gpu": achieved_wall_tflops,
        "perf/achieved_tflops_per_gpu_active": achieved_gpu_tflops,
        "perf/flops_per_sample": float(flops_per_sample),
        "perf/step_flops_per_gpu": step_flops,
        "perf/peak_dense_tflops_per_gpu": float(peak_dense_tflops),
        "perf/global_samples_per_sec": int(local_batch_size) * int(world_size) / wall_seconds,
        "perf/gpu_step_time_ms": float(gpu_elapsed_ms),
        "perf/wall_step_time_ms": float(wall_elapsed_ms),
        "perf/gpu_duty_cycle": gpu_seconds / wall_seconds,
    }
