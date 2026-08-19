import json
from pathlib import Path

import pytest

import src.utils.qualification as qualification
from src.utils.checkpointing import CheckpointManager, sha256_file
from src.utils.planning_promotion import (
    build_complete_planning_result,
    build_planning_provenance,
    mark_planning_evaluations_launched,
    poll_planning_evaluations,
    register_planning_evaluations,
    write_complete_planning_result,
)
from src.utils.qualification import (
    QualificationError,
    publish_released_droid_qualification,
    verify_released_droid_qualification,
)


def _qualification_inputs(tmp_path: Path, monkeypatch):
    source = tmp_path / "released.pth.tar"
    source.write_bytes(b"released-checkpoint-test-bytes")
    released_sha = sha256_file(source)
    monkeypatch.setattr(qualification, "RELEASED_DROID_SHA256", released_sha)

    dataset_manifest = tmp_path / "droid_manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "dataset_fingerprint": "droid-fixture-fingerprint",
                "verification": {
                    "complete": True,
                    "files_checked": True,
                    "source_listing_matches_staged": True,
                },
            }
        ),
        encoding="utf-8",
    )
    weights = tmp_path / "dinov3.pth"
    weights.write_bytes(b"dinov3-fixture")

    rollout_root = tmp_path / "rollout"
    rollout_manager = CheckpointManager(rollout_root, prefix="jepa")
    rollout_reference = rollout_manager.register_existing(source, global_update=0)
    rollout_result = rollout_root / "qualification_results" / "released_rollout.json"
    rollout_result.parent.mkdir(parents=True)
    rollout_result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "artifact_kind": "released_rollout_qualification",
                "checkpoint": rollout_reference.to_dict(),
                "dataset_fingerprint": "droid-fixture-fingerprint",
                "promotion_config_sha256": "1" * 64,
                "selection": {
                    "metric_name": "data_traj/val_rollout/visual_l2_loss/mean_h1_h2_h3_h4",
                    "metric_value": 0.25,
                    "mode": "min",
                    "count": 2048,
                },
                "wandb_run_id": "rollout-wandb-id",
            }
        ),
        encoding="utf-8",
    )

    planning_root = tmp_path / "planning"
    planning_manager = CheckpointManager(planning_root, prefix="jepa")
    planning_reference = planning_manager.register_existing(source, global_update=0, pending=True)
    checkpoint_path = planning_manager.verify(planning_reference)
    config = {
        "meta": {"eval_episodes": 64, "seed": 1},
        "nodes": 1,
        "tasks_per_node": 8,
        "tasks": ["droid-base"],
        "planner": {"planner_name": "cem", "horizon": 3},
    }
    provenance = build_planning_provenance(
        config,
        checkpoint_id=planning_reference.object_id,
        checkpoint_sha256=planning_reference.sha256,
        checkpoint_path=checkpoint_path,
        result_dir=planning_root / "planning_results",
        expected_tasks=["droid-base"],
        expected_episodes_per_task=64,
        task_name="droid-base",
        checkpoint_step=0,
    )
    registry = planning_root / "planning_results" / "registry.json"
    register_planning_evaluations(registry, [provenance], execution_mode="synchronous")
    mark_planning_evaluations_launched(registry, planning_reference.object_id)
    result = build_complete_planning_result(
        provenance,
        metrics={"ep_end_dist_xyz": 0.04, "episode_success": 0.482},
        observed_episode_counts={"droid-base": 64},
    )
    write_complete_planning_result(result)
    poll_planning_evaluations(registry, manager=planning_manager)
    planning_wandb_id = planning_root / "run_metadata" / "wandb_run_id.txt"
    planning_wandb_id.parent.mkdir(parents=True)
    planning_wandb_id.write_text("planning-wandb-id\n", encoding="utf-8")

    return {
        "rollout_result_path": rollout_result,
        "planning_registry_path": registry,
        "dataset_manifest_path": dataset_manifest,
        "dinov3_weights_path": weights,
        "planning_wandb_run_id_path": planning_wandb_id,
        "source_git_commit": "a" * 40,
        "output_path": tmp_path / "QUALIFIED.json",
    }


def test_released_qualification_receipt_binds_all_training_inputs(tmp_path: Path, monkeypatch):
    inputs = _qualification_inputs(tmp_path, monkeypatch)

    receipt = publish_released_droid_qualification(**inputs)
    verified = verify_released_droid_qualification(
        inputs["output_path"],
        dataset_manifest_path=inputs["dataset_manifest_path"],
        dinov3_weights_path=inputs["dinov3_weights_path"],
        source_git_commit=inputs["source_git_commit"],
    )

    assert verified["integrity_sha256"] == receipt["integrity_sha256"]
    assert verified["planning"]["expected_eval_count"] == 1
    assert verified["planning"]["selection"] == {
        "metric": "ep_end_dist_xyz",
        "mode": "min",
        "value": 0.04,
    }
    comparison = verified["planning"]["published_comparison"]
    assert comparison["observed"] == pytest.approx(48.2)
    assert comparison["within_reproduction_band"] is True


def test_released_qualification_receipt_rejects_changed_inputs(tmp_path: Path, monkeypatch):
    inputs = _qualification_inputs(tmp_path, monkeypatch)
    publish_released_droid_qualification(**inputs)

    inputs["dinov3_weights_path"].write_bytes(b"different-dinov3-weights")
    with pytest.raises(QualificationError, match="weights differ"):
        verify_released_droid_qualification(
            inputs["output_path"],
            dataset_manifest_path=inputs["dataset_manifest_path"],
            dinov3_weights_path=inputs["dinov3_weights_path"],
            source_git_commit=inputs["source_git_commit"],
        )

    with pytest.raises(QualificationError, match="Git commits differ"):
        verify_released_droid_qualification(
            inputs["output_path"],
            dataset_manifest_path=inputs["dataset_manifest_path"],
            dinov3_weights_path=tmp_path / "dinov3.pth",
            source_git_commit="b" * 40,
        )


def test_released_qualification_rejects_published_baseline_drift(tmp_path: Path, monkeypatch):
    inputs = _qualification_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(qualification, "PUBLISHED_DROID_SUCCESS_PERCENT", 80.0)

    with pytest.raises(QualificationError, match="does not reproduce the published"):
        publish_released_droid_qualification(**inputs)
