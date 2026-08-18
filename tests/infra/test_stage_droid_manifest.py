import json
from pathlib import Path

import pytest

import infra.skypilot.stage_droid as stage_droid
from infra.skypilot.stage_droid import (
    FRANKA_REVISION,
    bind_franka_manifest,
    encode_source_episode_id,
    encode_source_episode_ids,
    list_episode_ids_local,
    write_artifacts,
)


def _write_verified_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset_fingerprint": "a" * 64,
                "verification": {"files_checked": True, "complete": True},
            }
        ),
        encoding="utf-8",
    )


def test_franka_tree_is_verified_and_bound_into_combined_fingerprint(tmp_path: Path):
    source = tmp_path / "source"
    staged = tmp_path / "staged"
    source.mkdir()
    staged.mkdir()
    (source / "episode.h5").write_bytes(b"trajectory")
    (staged / "episode.h5").write_bytes(b"trajectory")
    manifest_path = tmp_path / "manifest.json"
    _write_verified_manifest(manifest_path)

    bind_franka_manifest(manifest_path, staged, source)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    auxiliary = manifest["auxiliary_datasets"]["Franka_hf"]
    assert auxiliary["hub_revision"] == FRANKA_REVISION
    assert auxiliary["files"][0]["path"] == "episode.h5"
    assert auxiliary["verification"]["source_tree_matches_staged"] is True
    assert manifest["dataset_fingerprint"] != manifest["primary_dataset_fingerprint"]


def test_franka_tree_mismatch_is_rejected(tmp_path: Path):
    source = tmp_path / "source"
    staged = tmp_path / "staged"
    source.mkdir()
    staged.mkdir()
    (source / "episode.h5").write_bytes(b"expected")
    (staged / "episode.h5").write_bytes(b"different")
    manifest_path = tmp_path / "manifest.json"
    _write_verified_manifest(manifest_path)

    with pytest.raises(RuntimeError, match="does not byte-match"):
        bind_franka_manifest(manifest_path, staged, source)


def test_local_pvc_episode_index_is_mount_independent(tmp_path: Path):
    root = tmp_path / "mounted"
    episode = root / "droid_raw" / "1.0.1" / "lab" / "day" / "episode"
    episode.mkdir(parents=True)
    (episode / "trajectory.h5").write_bytes(b"trajectory")
    output = tmp_path / "output"

    assert list_episode_ids_local(root) == ["lab/day/episode"]
    write_artifacts(
        target_uri=None,
        target_root=root,
        staged_identity="sky-volume://droid-volume/droid_raw/1.0.1",
        expected_source_root_uri=None,
        mount_root="/mnt/jepawm-datasets",
        min_episodes=1,
        output_dir=output,
    )

    manifest = json.loads((output / "droid_index_manifest.json").read_text(encoding="utf-8"))
    assert manifest["staged_uri"] == "sky-volume://droid-volume/droid_raw/1.0.1"
    assert manifest["episode_count"] == 1
    assert (output / "droid_paths.csv").read_text(encoding="utf-8").startswith(
        "/mnt/jepawm-datasets/droid_raw/1.0.1/lab/day/episode "
    )


def test_staged_inventory_is_bound_to_exact_live_source(tmp_path: Path, monkeypatch):
    root = tmp_path / "mounted"
    episode = root / "droid_raw" / "1.0.1" / "lab" / "episode"
    episode.mkdir(parents=True)
    (episode / "trajectory.h5").write_bytes(b"trajectory")
    output = tmp_path / "output"
    monkeypatch.setattr(stage_droid, "list_episode_ids_gcs", lambda _uri: ["lab/episode"])

    write_artifacts(
        target_uri=None,
        target_root=root,
        staged_identity="sky-volume://droid-volume/droid_raw/1.0.1",
        expected_source_root_uri="gs://gresearch/robotics",
        mount_root="/mnt/jepawm-datasets",
        min_episodes=1,
        output_dir=output,
    )

    manifest = json.loads((output / "droid_index_manifest.json").read_text(encoding="utf-8"))
    assert manifest["verification"]["source_listing_matches_staged"] is True
    assert manifest["source_inventory"]["episode_count"] == 1


def test_colon_episode_ids_are_encoded_and_bound_to_original_source(tmp_path: Path, monkeypatch):
    root = tmp_path / "mounted"
    staged_id = "lab/day/Fri_Aug_18_11：43：44_2023"
    episode = root / "droid_raw" / "1.0.1" / staged_id
    episode.mkdir(parents=True)
    (episode / "trajectory.h5").write_bytes(b"trajectory")
    output = tmp_path / "output"
    source_id = "lab/day/Fri_Aug_18_11:43:44_2023"
    monkeypatch.setattr(stage_droid, "list_episode_ids_gcs", lambda _uri: [source_id])

    write_artifacts(
        target_uri=None,
        target_root=root,
        staged_identity="sky-volume://droid-volume/droid_raw/1.0.1",
        expected_source_root_uri="gs://gresearch/robotics",
        mount_root="/mnt/jepawm-datasets",
        min_episodes=1,
        output_dir=output,
    )

    manifest = json.loads((output / "droid_index_manifest.json").read_text(encoding="utf-8"))
    assert encode_source_episode_id(source_id) == staged_id
    assert manifest["path_encoding"]["rclone_version"] == "v1.73.5"
    assert manifest["source_inventory"]["canonical_episode_ids_sha256"] != manifest[
        "canonical_episode_ids_sha256"
    ]
    assert manifest["source_inventory"]["staged_episode_ids_sha256"] == manifest[
        "canonical_episode_ids_sha256"
    ]


def test_path_encoding_rejects_reserved_fullwidth_colon():
    with pytest.raises(ValueError, match="reserved U\\+FF1A"):
        encode_source_episode_ids(["lab/A：B"])


def test_staged_inventory_rejects_missing_source_episode(tmp_path: Path, monkeypatch):
    root = tmp_path / "mounted"
    episode = root / "droid_raw" / "1.0.1" / "lab" / "episode"
    episode.mkdir(parents=True)
    (episode / "trajectory.h5").write_bytes(b"trajectory")
    monkeypatch.setattr(
        stage_droid,
        "list_episode_ids_gcs",
        lambda _uri: ["lab/episode", "lab/missing-episode"],
    )

    with pytest.raises(RuntimeError, match="does not match the live official source"):
        write_artifacts(
            target_uri=None,
            target_root=root,
            staged_identity="sky-volume://droid-volume/droid_raw/1.0.1",
            expected_source_root_uri="gs://gresearch/robotics",
            mount_root="/mnt/jepawm-datasets",
            min_episodes=1,
            output_dir=tmp_path / "output",
        )
