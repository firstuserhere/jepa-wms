from pathlib import Path

import pytest

from app.torchrun import _prepare_run_identity, _require_environment, load_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_overlay_recursively_merges_without_mutating_base(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text(
        """
app: vjepa_wm
folder: /base
data:
  loader:
    batch_size: 8
    num_workers: 16
    persistent_workers: true
model:
  predictor:
    pred_depth: 12
""",
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        """
base_config: base.yaml
overrides:
  folder: /durable
  data:
    loader:
      persistent_workers: false
""",
        encoding="utf-8",
    )

    resolved = load_config(overlay)

    assert resolved["folder"] == "/durable"
    assert resolved["data"]["loader"] == {
        "batch_size": 8,
        "num_workers": 16,
        "persistent_workers": False,
    }
    assert resolved["model"]["predictor"]["pred_depth"] == 12
    assert "base_config" not in resolved


def test_overlay_can_inherit_from_an_overlay(tmp_path: Path):
    (tmp_path / "base.yaml").write_text("value: 1\nnested:\n  keep: yes\n", encoding="utf-8")
    (tmp_path / "middle.yaml").write_text(
        "base_config: base.yaml\noverrides:\n  nested:\n    middle: 2\n",
        encoding="utf-8",
    )
    (tmp_path / "top.yaml").write_text(
        "base_config: middle.yaml\noverrides:\n  value: 3\n",
        encoding="utf-8",
    )

    resolved = load_config(tmp_path / "top.yaml")

    assert resolved == {"value": 3, "nested": {"keep": True, "middle": 2}}


def test_overlay_rejects_changes_outside_overrides(tmp_path: Path):
    (tmp_path / "base.yaml").write_text("value: 1\n", encoding="utf-8")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("base_config: base.yaml\noverrides: {}\nvalue: 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected top-level keys"):
        load_config(overlay)


def test_required_environment_error_never_contains_secret_value(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "do-not-print-this")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as error:
        _require_environment(["HF_TOKEN", "WANDB_API_KEY"])

    assert "WANDB_API_KEY" in str(error.value)
    assert "do-not-print-this" not in str(error.value)


def test_run_identity_allows_same_managed_task_recovery(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("JEPAWM_RUN_ID", "run-123")
    monkeypatch.setenv("SKYPILOT_TASK_ID", "sky-task-1")
    params = {
        "folder": str(tmp_path / "logs" / "run-123"),
        "checkpoint_folder": str(tmp_path / "checkpoints" / "run-123"),
    }

    _prepare_run_identity(params, resume_existing=False)
    _prepare_run_identity(params, resume_existing=False)

    monkeypatch.setenv("SKYPILOT_TASK_ID", "sky-task-2")
    with pytest.raises(RuntimeError, match="another task"):
        _prepare_run_identity(params, resume_existing=False)
    _prepare_run_identity(params, resume_existing=True)


def test_run_identity_rejects_ambiguous_existing_directory(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("JEPAWM_RUN_ID", "run-123")
    monkeypatch.setenv("SKYPILOT_TASK_ID", "sky-task-2")
    folder = tmp_path / "logs" / "run-123"
    folder.mkdir(parents=True)
    (folder / "old-log.csv").write_text("existing", encoding="utf-8")
    params = {
        "folder": str(folder),
        "checkpoint_folder": str(tmp_path / "checkpoints" / "run-123"),
    }

    with pytest.raises(RuntimeError, match="refusing to merge"):
        _prepare_run_identity(params, resume_existing=False)

    _prepare_run_identity(params, resume_existing=True)


def test_released_rollout_is_one_shot_full_corpus_no_gradient_qualification():
    resolved = load_config(REPO_ROOT / "infra/skypilot/released_rollout_overlay.yaml")

    assert resolved["meta"]["rollout_only_eval_mode"] is True
    assert resolved["meta"]["light_eval_only_mode"] is True
    assert resolved["meta"]["quick_debug"] is False
    assert resolved["optimization"]["transition_model"]["num_epochs"] == 1
    assert resolved["optimization"]["transition_model"]["iterations_per_epoch"] == 1
    assert resolved["data"]["custom"].get("filter_first_episodes") is None
    assert resolved["checkpointing"]["rollout_promotion"] == {
        "enabled": True,
        "every_epochs": 1,
        "loader_index": 0,
        "sampler_epoch": 0,
        "seed": 50234,
        "max_batches": 64,
        "horizons": [1, 2, 3, 4],
        "aggregation": "mean",
        "metric_base": "data_traj/val_rollout/visual_l2_loss",
    }


@pytest.mark.parametrize(
    "overlay",
    [
        "released_planning_overlay.yaml",
        "droid_dinov3_quality_overlay.yaml",
    ],
)
def test_droid_planning_preserves_released_eight_rank_topology(overlay: str):
    resolved = load_config(REPO_ROOT / "infra/skypilot" / overlay)

    assert resolved["evals"]["separate"] is False
    assert resolved["evals"]["inprocess_world_size"] == 8
    assert resolved["evals"]["nodes"] == 1
    assert resolved["evals"]["eval_episodes"] == 64


def test_full_task_requires_verified_released_qualification_before_torchrun():
    import yaml

    full_task = yaml.safe_load(
        (REPO_ROOT / "infra/skypilot/droid_train_full.yaml").read_text(encoding="utf-8")
    )
    run = full_task["run"]
    verify_position = run.index("qualification_receipt.py verify")
    runtime_verify_position = run.index("runtime_readiness.py verify")
    torchrun_position = run.index("torchrun")

    assert verify_position < torchrun_position
    assert runtime_verify_position < torchrun_position
    assert "JEPAWM_QUALIFICATION_RUN_ID" in run
    assert "JEPAWM_RUNTIME_READINESS_RUN_ID" in run
    assert "--dataset-manifest" in run
    assert "--dinov3-weights" in run
    assert "--git-commit" in run


def test_qualification_task_publishes_receipt_after_planning():
    import yaml

    qualification_task = yaml.safe_load(
        (REPO_ROOT / "infra/skypilot/droid_qualify_released.yaml").read_text(encoding="utf-8")
    )
    run = qualification_task["run"]

    assert run.rindex("qualification_receipt.py publish") > run.index("--master_port=29601")
    assert "--planning-wandb-run-id-file" in run


def test_runtime_smokes_publish_two_node_recovery_evidence():
    import yaml

    distributed_task = yaml.safe_load(
        (REPO_ROOT / "infra/skypilot/distributed_smoke.yaml").read_text(encoding="utf-8")
    )
    training_task = yaml.safe_load(
        (REPO_ROOT / "infra/skypilot/droid_train_smoke.yaml").read_text(encoding="utf-8")
    )

    assert distributed_task["num_nodes"] == 2
    assert "--expected-num-nodes 2" in distributed_task["run"]
    assert "JEPAWM_RUN_ID" in distributed_task["run"]
    assert training_task["num_nodes"] == 2
    assert "JEPAWM_DISTRIBUTED_SMOKE_RUN_ID" in training_task["run"]
    assert training_task["run"].rindex("runtime_readiness.py publish") > training_task["run"].index(
        "torchrun"
    )


def test_distributed_smoke_installs_numpy_for_object_collectives():
    setup = (REPO_ROOT / "infra/skypilot/setup_smoke.sh").read_text(encoding="utf-8")

    assert '"torch==2.7.0"' in setup
    assert '"numpy==2.2.6"' in setup
