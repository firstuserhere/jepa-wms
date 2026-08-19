from pathlib import Path

import yaml

from infra.skypilot.render_k8s_task import render_k8s_task


REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_task():
    return {
        "name": "test",
        "num_nodes": 4,
        "resources": {
            "cloud": "gcp",
            "accelerators": "H200:8",
            "disk_size": 3000,
            "disk_tier": "best",
            "network_tier": "best",
            "use_spot": True,
            "job_recovery": {"strategy": "EAGER_NEXT_REGION", "max_restarts_on_errors": 3},
        },
        "envs": {"DROID_STORE_URI": "gs://data", "CHECKPOINT_STORE_URI": "gs://checkpoints"},
        "file_mounts": {
            "/mnt/jepawm-datasets": {"source": "${DROID_STORE_URI}"},
            "/mnt/jepawm-checkpoints": {"source": "${CHECKPOINT_STORE_URI}"},
        },
    }


def test_k8s_profile_replaces_only_provider_and_storage_plumbing():
    rendered = render_k8s_task(
        _base_task(),
        droid_volume="jepawm-droid",
        checkpoint_volume="jepawm-checkpoints",
        context="Skypilot",
    )

    assert rendered["num_nodes"] == 4
    assert rendered["resources"]["accelerators"] == "H200:8"
    assert rendered["resources"]["infra"] == "k8s/Skypilot"
    assert rendered["resources"]["disk_size"] == 100
    assert rendered["resources"]["job_recovery"]["strategy"] == "FAILOVER"
    assert "cloud" not in rendered["resources"]
    assert "file_mounts" not in rendered
    assert rendered["volumes"] == {
        "/mnt/jepawm-datasets": "jepawm-droid",
        "/mnt/jepawm-checkpoints": "jepawm-checkpoints",
    }
    assert rendered["envs"]["JEPAWM_STORAGE_BACKEND"] == "pvc"
    assert rendered["envs"]["DINOV3_WEIGHTS_SOURCE_PATH"].startswith(
        "/mnt/jepawm-checkpoints/artifacts/"
    )
    assert rendered["api_server_access"] is False


def test_k8s_profile_rejects_missing_or_unsafe_volume_names():
    task = _base_task()
    try:
        render_k8s_task(task, droid_volume=None, checkpoint_volume="checkpoints", context="Skypilot")
    except ValueError as error:
        assert "droid-volume" in str(error)
    else:
        raise AssertionError("missing DROID volume was accepted")

    try:
        render_k8s_task(task, droid_volume="DROID VOLUME", checkpoint_volume="checkpoints", context="Skypilot")
    except ValueError as error:
        assert "DNS-style" in str(error)
    else:
        raise AssertionError("unsafe DROID volume was accepted")


def test_droid_staging_uses_four_full_h200_nodes_for_parallel_transfer():
    task = yaml.safe_load(
        (REPO_ROOT / "infra/skypilot/droid_stage.yaml").read_text(encoding="utf-8")
    )

    assert task["num_nodes"] == 4
    assert task["resources"]["accelerators"] == "H200:8"
    assert task["resources"]["cpus"] == "64+"
    assert task["resources"]["network_tier"] == "best"
    assert task["resources"]["memory"] == "256+"


def test_cpu_only_profile_omits_disk_and_caps_library_threads():
    task = _base_task()
    task["num_nodes"] = 1
    task["resources"].pop("accelerators")
    task["resources"]["cpus"] = "16+"
    task["envs"]["OMP_NUM_THREADS"] = "16"

    rendered = render_k8s_task(
        task,
        droid_volume="jepawm-droid",
        checkpoint_volume="jepawm-checkpoints",
        context="Skypilot",
    )

    assert "disk_size" not in rendered["resources"]
    assert {name: rendered["envs"][name] for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "POLARS_MAX_THREADS",
    )} == {
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
        "OPENBLAS_NUM_THREADS": "16",
        "NUMEXPR_NUM_THREADS": "16",
        "RAYON_NUM_THREADS": "16",
        "POLARS_MAX_THREADS": "16",
    }
