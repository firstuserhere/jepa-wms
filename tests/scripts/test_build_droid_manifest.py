import hashlib
import json
from pathlib import Path

import pytest

from src.scripts.build_droid_manifest import atomic_write_json, build_manifest


def _make_episode(root: Path, relative: str) -> Path:
    episode = root / relative
    video_dir = episode / "recordings" / "MP4"
    video_dir.mkdir(parents=True)
    (episode / "trajectory.h5").write_bytes(b"trajectory")
    (video_dir / "cam-left.mp4").write_bytes(b"video")
    (episode / "metadata.json").write_text(
        json.dumps({"left_mp4_path": "recordings/MP4/cam-left.mp4"}), encoding="utf-8"
    )
    return episode


def test_manifest_is_mount_path_independent(tmp_path):
    fingerprints = []
    for mount_name in ("mount-a", "mount-b"):
        root = tmp_path / mount_name / "1.0.1"
        first = _make_episode(root, "lab/day/success/episode-1")
        second = _make_episode(root, "lab/day/success/episode-2")
        paths = tmp_path / f"{mount_name}.csv"
        paths.write_text(f"{second}\n{first}\n", encoding="utf-8")
        manifest = build_manifest(paths, root)
        assert manifest["verification"]["complete"] is True
        assert manifest["episode_count"] == 2
        fingerprints.append(manifest["dataset_fingerprint"])
    assert fingerprints[0] == fingerprints[1]


def test_manifest_rejects_path_outside_root(tmp_path):
    root = tmp_path / "root"
    outside = _make_episode(tmp_path / "other", "episode")
    paths = tmp_path / "paths.csv"
    paths.write_text(f"{outside}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside DROID root"):
        build_manifest(paths, root)


def test_manifest_records_missing_required_file(tmp_path):
    root = tmp_path / "root"
    episode = _make_episode(root, "episode")
    (episode / "trajectory.h5").unlink()
    paths = tmp_path / "paths.csv"
    paths.write_text(f"{episode}\n", encoding="utf-8")
    manifest = build_manifest(paths, root)
    assert manifest["verification"]["complete"] is False
    assert manifest["verification"]["missing_episode_count"] == 1


def test_manifest_can_publish_an_accounted_loader_ready_subset(tmp_path):
    root = tmp_path / "root"
    valid = _make_episode(root, "valid")
    no_metadata = _make_episode(root, "no-metadata")
    (no_metadata / "metadata.json").unlink()
    no_video = _make_episode(root, "no-video")
    (no_video / "recordings" / "MP4" / "cam-left.mp4").unlink()
    paths = tmp_path / "source-paths.csv"
    paths.write_text(f"{valid}\n{no_metadata}\n{no_video}\n", encoding="utf-8")
    filtered_paths = tmp_path / "loader-ready.csv"

    manifest = build_manifest(
        paths,
        root,
        filter_loader_unusable=True,
        filtered_paths_output=filtered_paths,
        published_path_list_path=Path("/mnt/jepawm-datasets/DROID/droid_paths.csv"),
    )

    assert manifest["verification"]["complete"] is True
    assert manifest["source_episode_count"] == 3
    assert manifest["episode_count"] == 1
    assert manifest["verification"]["verified_episode_count"] == 1
    assert manifest["verification"]["filtered_episode_count"] == 2
    assert manifest["verification"]["unfilterable_error_count"] == 0
    assert manifest["verification"]["error_category_counts"] == {
        "missing_camera_video": 1,
        "no_metadata_json": 1,
    }
    assert manifest["loader_ready_filter"]["all_source_episodes_accounted_for"] is True
    assert manifest["path_list"] == "/mnt/jepawm-datasets/DROID/droid_paths.csv"
    assert filtered_paths.read_text(encoding="utf-8") == f"{valid} 0\n"


def test_loader_ready_filter_does_not_accept_invalid_metadata(tmp_path):
    root = tmp_path / "root"
    episode = _make_episode(root, "invalid-json")
    (episode / "metadata.json").write_text("{", encoding="utf-8")
    paths = tmp_path / "source-paths.csv"
    paths.write_text(f"{episode}\n", encoding="utf-8")

    manifest = build_manifest(
        paths,
        root,
        filter_loader_unusable=True,
        filtered_paths_output=tmp_path / "loader-ready.csv",
    )

    assert manifest["verification"]["complete"] is False
    assert manifest["verification"]["unfilterable_error_count"] == 1
    assert manifest["verification"]["error_category_counts"] == {"invalid_metadata_json": 1}
    assert manifest["loader_ready_filter"]["all_source_episodes_accounted_for"] is False


def test_loader_ready_subset_fingerprint_is_mount_independent(tmp_path):
    fingerprints = []
    for mount_name in ("mount-a", "mount-b"):
        root = tmp_path / mount_name
        valid = _make_episode(root, "valid")
        rejected = _make_episode(root, "no-metadata")
        (rejected / "metadata.json").unlink()
        paths = tmp_path / f"{mount_name}.csv"
        paths.write_text(f"{valid}\n{rejected}\n", encoding="utf-8")
        manifest = build_manifest(
            paths,
            root,
            filter_loader_unusable=True,
            filtered_paths_output=tmp_path / f"{mount_name}-loader-ready.csv",
            published_path_list_path=Path("/mnt/jepawm-datasets/DROID/droid_paths.csv"),
        )
        fingerprints.append(manifest["dataset_fingerprint"])

    assert fingerprints[0] == fingerprints[1]


def test_listing_only_manifest_is_never_marked_complete(tmp_path):
    root = tmp_path / "root"
    episode = _make_episode(root, "episode")
    paths = tmp_path / "paths.csv"
    paths.write_text(f"{episode}\n", encoding="utf-8")

    manifest = build_manifest(paths, root, verify_files=False)

    assert manifest["verification"]["files_checked"] is False
    assert manifest["verification"]["complete"] is False


def test_manifest_binds_matching_official_source_inventory(tmp_path):
    root = tmp_path / "root"
    episode = _make_episode(root, "lab/episode")
    paths = tmp_path / "paths.csv"
    paths.write_text(f"{episode}\n", encoding="utf-8")
    episode_hash = hashlib.sha256(b"lab/episode\n").hexdigest()
    source_index = tmp_path / "source-index.json"
    source_index.write_text(
        json.dumps(
            {
                "episode_count": 1,
                "canonical_episode_ids_sha256": episode_hash,
                "source_inventory": {
                    "root_uri": "gs://gresearch/robotics/droid_raw/1.0.1",
                    "episode_count": 1,
                    "canonical_episode_ids_sha256": episode_hash,
                },
                "verification": {"source_listing_matches_staged": True},
            }
        ),
        encoding="utf-8",
    )

    manifest = build_manifest(paths, root, source_index_manifest=source_index)

    assert manifest["verification"]["source_listing_matches_staged"] is True
    assert manifest["source_inventory"]["episode_count"] == 1

    source_document = json.loads(source_index.read_text(encoding="utf-8"))
    source_document["episode_count"] = 2
    source_index.write_text(json.dumps(source_document), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        build_manifest(paths, root, source_index_manifest=source_index)


def test_manifest_binds_encoded_staged_ids_to_original_source_inventory(tmp_path):
    root = tmp_path / "root"
    staged_id = "lab/Fri_Aug_18_11：43：44_2023"
    episode = _make_episode(root, staged_id)
    paths = tmp_path / "paths.csv"
    paths.write_text(f"{episode}\n", encoding="utf-8")
    source_id = "lab/Fri_Aug_18_11:43:44_2023"
    source_hash = hashlib.sha256(f"{source_id}\n".encode()).hexdigest()
    staged_hash = hashlib.sha256(f"{staged_id}\n".encode()).hexdigest()
    source_index = tmp_path / "source-index.json"
    source_index.write_text(
        json.dumps(
            {
                "episode_count": 1,
                "canonical_episode_ids_sha256": staged_hash,
                "path_encoding": {
                    "scheme": "rclone-local-colon",
                    "rclone_version": "v1.73.5",
                },
                "source_inventory": {
                    "root_uri": "gs://gresearch/robotics/droid_raw/1.0.1",
                    "episode_count": 1,
                    "canonical_episode_ids_sha256": source_hash,
                    "staged_episode_ids_sha256": staged_hash,
                },
                "verification": {"source_listing_matches_staged": True},
            }
        ),
        encoding="utf-8",
    )

    manifest = build_manifest(paths, root, source_index_manifest=source_index)

    assert manifest["verification"]["source_listing_matches_staged"] is True
    assert manifest["canonical_episode_ids_sha256"] == staged_hash
    assert manifest["source_inventory"]["canonical_episode_ids_sha256"] == source_hash
    assert manifest["path_encoding"]["scheme"] == "rclone-local-colon"


def test_atomic_json_write(tmp_path):
    output = tmp_path / "nested" / "manifest.json"
    atomic_write_json({"value": 3}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 3}
    assert not list(output.parent.glob("*.tmp"))
