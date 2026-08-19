# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Runtime binding checks for staged research datasets.

Staging verification is useful only if the path list and held-out files seen by
the trainer are the same bytes described by the published manifest.  These
checks are intentionally independent of the local mount prefix.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class DatasetManifestError(RuntimeError):
    """The mounted dataset does not match its declared manifest."""


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _droid_episode_paths(path_list: Path) -> list[Path]:
    episodes: list[Path] = []
    with path_list.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if not fields:
                raise DatasetManifestError(f"Empty DROID record at {path_list}:{line_number}")
            episodes.append(Path(fields[0]))
    return episodes


def verify_droid_runtime_binding(
    manifest: Mapping[str, Any], dataset_paths: Sequence[str]
) -> dict[str, Any]:
    """Verify the exact DROID path-list bytes and canonical episode identity."""

    if len(dataset_paths) != 1:
        raise DatasetManifestError(
            f"A DROID manifest must bind exactly one training path list, got {len(dataset_paths)}"
        )
    path_list = Path(dataset_paths[0]).expanduser().resolve()
    if not path_list.is_file():
        raise DatasetManifestError(f"Mounted DROID path list is missing: {path_list}")
    actual_path_sha = _sha256_file(path_list)
    if actual_path_sha != manifest.get("path_list_sha256"):
        raise DatasetManifestError(
            "Mounted DROID path-list checksum differs from the staged manifest: "
            f"{actual_path_sha} != {manifest.get('path_list_sha256')}"
        )

    episodes = _droid_episode_paths(path_list)
    if len(episodes) != int(manifest.get("episode_count", -1)):
        raise DatasetManifestError(
            f"Mounted DROID episode count is {len(episodes)}, expected {manifest.get('episode_count')}"
        )
    manifest_root = manifest.get("droid_root")
    if not manifest_root:
        raise DatasetManifestError("DROID manifest has no droid_root")
    root = Path(str(manifest_root)).expanduser().resolve()
    canonical_ids = []
    for episode in episodes:
        try:
            canonical_ids.append(episode.expanduser().resolve().relative_to(root).as_posix())
        except ValueError as error:
            raise DatasetManifestError(
                f"DROID path-list entry escapes the manifested root {root}: {episode}"
            ) from error
    if len(canonical_ids) != len(set(canonical_ids)):
        raise DatasetManifestError("Mounted DROID path list contains duplicate episode identifiers")
    canonical_digest = hashlib.sha256(
        "".join(f"{episode_id}\n" for episode_id in sorted(canonical_ids)).encode("utf-8")
    ).hexdigest()
    if canonical_digest != manifest.get("canonical_episode_ids_sha256"):
        raise DatasetManifestError(
            "Mounted DROID episode identity differs from the staged manifest: "
            f"{canonical_digest} != {manifest.get('canonical_episode_ids_sha256')}"
        )
    return {
        "path_list": str(path_list),
        "path_list_sha256": actual_path_sha,
        "episode_count": len(episodes),
        "canonical_episode_ids_sha256": canonical_digest,
    }


def verify_auxiliary_runtime_binding(
    manifest: Mapping[str, Any], name: str, root: str | Path
) -> dict[str, Any]:
    """Byte-verify a small held-out dataset used to promote checkpoints."""

    auxiliary = manifest.get("auxiliary_datasets", {}).get(name)
    if not isinstance(auxiliary, Mapping):
        raise DatasetManifestError(f"Dataset manifest has no auxiliary identity for {name}")
    root_path = Path(root).expanduser().resolve()
    records = auxiliary.get("files")
    if not isinstance(records, list) or not records:
        raise DatasetManifestError(f"Auxiliary manifest for {name} has no file records")
    observed_records = []
    for record in records:
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise DatasetManifestError(f"Unsafe {name} manifest path: {relative}")
        path = root_path / relative
        if not path.is_file():
            raise DatasetManifestError(f"Missing {name} file: {path}")
        observed = {
            "path": relative.as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        if observed != record:
            raise DatasetManifestError(f"{name} file differs from its manifest: {path}")
        observed_records.append(observed)
    canonical = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for record in observed_records
    )
    tree_sha256 = hashlib.sha256(canonical).hexdigest()
    if tree_sha256 != auxiliary.get("tree_sha256"):
        raise DatasetManifestError(
            f"Mounted {name} tree checksum differs from its manifest: "
            f"{tree_sha256} != {auxiliary.get('tree_sha256')}"
        )
    return {
        "root": str(root_path),
        "file_count": len(observed_records),
        "tree_sha256": tree_sha256,
    }
