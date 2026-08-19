import hashlib
import json
from pathlib import Path

import pytest

from src.utils.dataset_manifest import (
    DatasetManifestError,
    verify_auxiliary_runtime_binding,
    verify_droid_runtime_binding,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_droid_runtime_binding_detects_path_list_mutation(tmp_path):
    root = tmp_path / "droid_raw" / "1.0.1"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    path_list = tmp_path / "droid_paths.csv"
    path_list.write_text(f"{root / 'a'} 0\n{root / 'b'} 1\n", encoding="utf-8")
    canonical = hashlib.sha256(b"a\nb\n").hexdigest()
    manifest = {
        "droid_root": str(root),
        "path_list_sha256": _sha(path_list),
        "episode_count": 2,
        "canonical_episode_ids_sha256": canonical,
    }
    result = verify_droid_runtime_binding(manifest, [str(path_list)])
    assert result["canonical_episode_ids_sha256"] == canonical

    path_list.write_text(f"{root / 'a'} 0\n", encoding="utf-8")
    with pytest.raises(DatasetManifestError, match="path-list checksum"):
        verify_droid_runtime_binding(manifest, [str(path_list)])


def test_auxiliary_runtime_binding_verifies_every_file(tmp_path):
    root = tmp_path / "franka"
    root.mkdir()
    first = root / "episode.h5"
    second = root / "metadata.json"
    first.write_bytes(b"trajectory")
    second.write_text("{}", encoding="utf-8")
    records = [
        {"path": "episode.h5", "size": first.stat().st_size, "sha256": _sha(first)},
        {"path": "metadata.json", "size": second.stat().st_size, "sha256": _sha(second)},
    ]
    canonical = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )
    manifest = {
        "auxiliary_datasets": {
            "Franka_hf": {
                "files": records,
                "tree_sha256": hashlib.sha256(canonical).hexdigest(),
            }
        }
    }
    assert verify_auxiliary_runtime_binding(manifest, "Franka_hf", root)["file_count"] == 2
    first.write_bytes(b"corrupt")
    with pytest.raises(DatasetManifestError, match="differs"):
        verify_auxiliary_runtime_binding(manifest, "Franka_hf", root)
