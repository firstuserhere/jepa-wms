import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.utils.checkpointing import (
    CheckpointManager,
    build_checkpoint_v2,
    create_checkpoint_manifest,
    sha256_file,
)
from src.utils.planning_promotion import canonical_sha256
from src.utils.runtime_readiness import (
    DINOV3_REPOSITORY_REVISION,
    RuntimeReadinessError,
    publish_runtime_readiness,
    verify_runtime_readiness,
)


def _write_integrity(path: Path, document: dict):
    document = dict(document)
    document["integrity_sha256"] = canonical_sha256(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _runtime_inputs(tmp_path: Path):
    source_commit = "c" * 40
    dataset_manifest = tmp_path / "droid_manifest.json"
    dataset = {
        "dataset_fingerprint": "droid-runtime-fixture",
        "verification": {
            "complete": True,
            "files_checked": True,
            "source_listing_matches_staged": True,
        },
    }
    dataset_manifest.write_text(json.dumps(dataset), encoding="utf-8")
    weights = tmp_path / "dinov3.pth"
    weights.write_bytes(b"runtime-dinov3-fixture")
    weights_sha = sha256_file(weights)

    distributed_state = tmp_path / "distributed_recovery.json"
    rank_evidence = [
        {
            "rank": rank,
            "local_rank": rank % 8,
            "hostname": f"node-{rank // 8}",
            "gpu_name": "NVIDIA H200",
        }
        for rank in range(16)
    ]
    _write_integrity(
        distributed_state,
        {
            "schema_version": 1,
            "kind": "distributed-recovery-smoke",
            "task_id": "distributed-task",
            "run_id": "distributed-run",
            "phase": "complete",
            "completed_at": "2026-08-15T00:00:00+00:00",
            "source_git_commit": source_commit,
            "world_size": 16,
            "observed_node_count": 2,
            "rank_evidence": rank_evidence,
            "wandb_run_id": "distributed-wandb",
            "recovered_from_controlled_failure": True,
        },
    )

    checkpoint_root = tmp_path / "training-checkpoints"
    manager = CheckpointManager(checkpoint_root, prefix="jepa")
    manifest = create_checkpoint_manifest(
        resolved_config={"model": {"depth": 12}, "optimization": {"iterations_per_epoch": 8}},
        dataset_manifest=dataset,
        encoder_manifest={
            "repo_revision": DINOV3_REPOSITORY_REVISION,
            "weights_sha256": weights_sha,
        },
        wandb_run_id="training-wandb",
        git_manifest={"available": True, "commit": source_commit, "dirty_patch": ""},
    )
    checkpoint = build_checkpoint_v2(
        training_state={
            "predictor": {"weight": 1},
            "optimizer": {"state": {}},
            "scaler": None,
            "schedulers": {"lr": {"step": 16}, "weight_decay": {"step": 16}},
        },
        epoch=2,
        global_update=16,
        rng_state={"world_size": 16, "states": {}},
        sampler_state={"epoch": 2},
        manifest=manifest,
    )
    reference = manager.save(checkpoint, epoch=2, global_update=16)
    manager.promote("latest", reference)

    def write_telemetry(step: int) -> Path:
        path = checkpoint_root / "run_metadata" / f"training_v1_step-{step:012d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        elapsed = float(step)
        peak = 989e12
        useful = 0.25 * elapsed * 16 * peak
        path.write_text(
            json.dumps(
                {
                    "summary": {
                        "pantheon_telemetry/contract_version": "training-v1",
                        "pantheon_telemetry/heartbeat_at": 1_800_000_000.0,
                        "pantheon_telemetry/total_steps": 16,
                        "pantheon_telemetry/active_step": step,
                        "pantheon_telemetry/step_time_s": 1.0,
                        "pantheon_telemetry/effective_mfu": 0.25,
                        "pantheon_telemetry/wandb_url": (
                            "https://wandb.ai/entity/project/runs/training-wandb"
                        ),
                    },
                    "history_keys": [
                        "train/global_step",
                        "train/loss",
                        "val/loss",
                        "perf/effective_mfu",
                        "perf/model_flops",
                        "perf/elapsed_e2e_s",
                        "perf/gpu_count",
                        "perf/peak_flops_per_gpu",
                        "perf/timed_wall_coverage",
                        "perf/actual_global_samples",
                    ],
                    "validation_started": True,
                    "mfu_terms": {
                        "elapsed_s": elapsed,
                        "gpu_count": 16,
                        "useful_flops_by_precision": {"bf16": useful},
                        "peak_flops_per_gpu": {"bf16": peak},
                    },
                    "provenance": {
                        "formula_id": "test",
                        "formula_version": "1",
                        "git_commit": source_commit,
                        "gpu_sku": "NVIDIA H200",
                        "gpu_form_factor": "SXM",
                        "precision": "bf16",
                        "sparsity": "dense",
                        "peak_flops_per_gpu": peak,
                        "peak_source": "test",
                        "world_size": 16,
                        "timing_scope": "end-to-end-slowest-rank",
                        "timed_wall_coverage": 1.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    first_telemetry = write_telemetry(8)
    final_telemetry = write_telemetry(16)

    common_snapshot = {
        "wandb_run_id": "training-wandb",
        "checkpoint_schema_version": 2,
        "source_git_commit": source_commit,
        "dataset_fingerprint": dataset["dataset_fingerprint"],
        "encoder_repo_revision": DINOV3_REPOSITORY_REVISION,
        "encoder_weights_sha256": weights_sha,
        "resume_contract_sha256": manifest["resume_contract"]["sha256"],
        "epoch_boundary_only": True,
        "epoch_boundary_seconds": 120.0,
        "checkpoint_seconds": 10.0,
        "checkpoint_loss_budget_seconds": 300.0,
        "checkpoint_within_loss_budget": True,
        "telemetry_effective_mfu": 0.25,
        "telemetry_wandb_url": "https://wandb.ai/entity/project/runs/training-wandb",
    }
    first = {
        **common_snapshot,
        "epoch": 1,
        "global_update": 8,
        "object_id": "first-checkpoint.pth.tar",
        "sha256": "f" * 64,
        "telemetry_snapshot_path": str(first_telemetry),
        "telemetry_snapshot_sha256": sha256_file(first_telemetry),
    }
    final = {
        **common_snapshot,
        "epoch": 2,
        "global_update": 16,
        "object_id": reference.object_id,
        "sha256": reference.sha256,
        "telemetry_snapshot_path": str(final_telemetry),
        "telemetry_snapshot_sha256": sha256_file(final_telemetry),
    }
    training_state = tmp_path / "training_recovery_smoke.json"
    _write_integrity(
        training_state,
        {
            "schema_version": 1,
            "kind": "training-checkpoint-recovery-smoke",
            "phase": "complete",
            "completed_at": "2026-08-15T00:10:00+00:00",
            "first_checkpoint": first,
            "final_checkpoint": final,
            "strict_resume_verified": True,
        },
    )
    return {
        "distributed_smoke_path": distributed_state,
        "training_smoke_path": training_state,
        "training_checkpoint_root": checkpoint_root,
        "dataset_manifest_path": dataset_manifest,
        "dinov3_weights_path": weights,
        "source_git_commit": source_commit,
        "output_path": tmp_path / "RUNTIME_READY.json",
    }


def test_runtime_readiness_binds_both_recoveries_and_training_inputs(tmp_path: Path):
    inputs = _runtime_inputs(tmp_path)

    receipt = publish_runtime_readiness(**inputs)
    verified = verify_runtime_readiness(
        inputs["output_path"],
        dataset_manifest_path=inputs["dataset_manifest_path"],
        dinov3_weights_path=inputs["dinov3_weights_path"],
        source_git_commit=inputs["source_git_commit"],
    )

    assert verified["integrity_sha256"] == receipt["integrity_sha256"]
    assert verified["distributed_smoke"]["world_size"] == 16
    assert verified["training_smoke"]["final_checkpoint"]["global_update"] == 16


def test_runtime_readiness_rejects_evidence_changed_after_publication(tmp_path: Path):
    inputs = _runtime_inputs(tmp_path)
    publish_runtime_readiness(**inputs)
    inputs["distributed_smoke_path"].write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeReadinessError, match="changed after readiness publication"):
        verify_runtime_readiness(
            inputs["output_path"],
            dataset_manifest_path=inputs["dataset_manifest_path"],
            dinov3_weights_path=inputs["dinov3_weights_path"],
            source_git_commit=inputs["source_git_commit"],
        )
