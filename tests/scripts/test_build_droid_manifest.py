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


def test_atomic_json_write(tmp_path):
    output = tmp_path / "nested" / "manifest.json"
    atomic_write_json({"value": 3}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 3}
    assert not list(output.parent.glob("*.tmp"))
