#!/usr/bin/env python3
"""Build a reproducible manifest for the raw DROID dataset used by JEPA-WM.

The training loader consumes a whitespace-delimited list of raw episode
directories.  This utility fingerprints the *relative* episode identifiers so
the same dataset mounted at a different local path has the same identity, and
optionally verifies the files the loader will open.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = 2
DEFAULT_SOURCE_URI = "gs://gresearch/robotics/droid_raw/1.0.1"
DEFAULT_DATASET_VERSION = "1.0.1"
LOADER_READY_FILTER_POLICY = "jepa-wm-left-mp4-loader-ready-v1"
FILTERABLE_ERROR_CODES = frozenset({"no_metadata_json", "missing_camera_video"})


class EpisodeVerificationError(Exception):
    """A deterministic loader-readiness failure with a stable manifest code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _sha256_bytes(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    def chunks():
        with path.open("rb") as stream:
            while block := stream.read(chunk_size):
                yield block

    return _sha256_bytes(chunks())


def read_episode_paths(path_list: Path) -> list[Path]:
    episodes: list[Path] = []
    with path_list.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            value = line.split()[0]
            if not value:
                raise ValueError(f"Empty path at {path_list}:{line_number}")
            episodes.append(Path(value))
    if not episodes:
        raise ValueError(f"No DROID episode paths found in {path_list}")
    return episodes


def canonical_episode_ids(episodes: list[Path], droid_root: Path) -> list[str]:
    root = droid_root.resolve()
    ids: list[str] = []
    for episode in episodes:
        resolved = episode.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Episode path is outside DROID root: {episode} (root={root})") from exc
        ids.append(relative.as_posix())
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"DROID path list contains duplicate episodes: {preview}")
    return ids


def _load_metadata(episode: Path) -> dict:
    json_files = sorted(episode.glob("*.json"))
    if not json_files:
        raise EpisodeVerificationError("no_metadata_json", f"No metadata JSON in {episode}")
    with json_files[0].open("r", encoding="utf-8") as stream:
        return json.load(stream)


def verify_episode(episode: Path, camera_key: str) -> dict[str, int | str]:
    if not episode.is_dir():
        raise EpisodeVerificationError("missing_episode_directory", f"Missing episode directory: {episode}")
    trajectory = episode / "trajectory.h5"
    if not trajectory.is_file():
        raise EpisodeVerificationError("missing_trajectory", f"Missing trajectory.h5: {trajectory}")
    metadata = _load_metadata(episode)
    if camera_key not in metadata:
        raise EpisodeVerificationError("missing_camera_key", f"Metadata in {episode} has no {camera_key!r}")
    video_name = str(metadata[camera_key]).split("recordings/MP4/")[-1]
    video = episode / "recordings" / "MP4" / video_name
    if not video.is_file():
        raise EpisodeVerificationError("missing_camera_video", f"Missing camera video referenced by metadata: {video}")
    if "stereo" in video.name.lower():
        raise EpisodeVerificationError(
            "stereo_camera_video", f"Configured camera unexpectedly references a stereo video: {video}"
        )
    return {
        "trajectory_bytes": trajectory.stat().st_size,
        "video_bytes": video.stat().st_size,
        "camera_video": video.name,
    }


def build_manifest(
    path_list: Path,
    droid_root: Path,
    *,
    source_uri: str = DEFAULT_SOURCE_URI,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    camera_key: str = "left_mp4_path",
    verify_files: bool = True,
    source_index_manifest: Path | None = None,
    filter_loader_unusable: bool = False,
    filtered_paths_output: Path | None = None,
    published_path_list_path: Path | None = None,
) -> dict:
    if filter_loader_unusable and not verify_files:
        raise ValueError("Loader-ready filtering requires full file verification")
    if filter_loader_unusable and filtered_paths_output is None:
        raise ValueError("Loader-ready filtering requires filtered_paths_output")
    episodes = read_episode_paths(path_list)
    source_episode_ids = canonical_episode_ids(episodes, droid_root)
    source_identity_lines = [f"{item}\n".encode("utf-8") for item in sorted(source_episode_ids)]

    missing: list[dict[str, str]] = []
    usable_episodes: list[Path] = []
    usable_episode_ids: list[str] = []
    verified = 0
    trajectory_bytes = 0
    video_bytes = 0
    if verify_files:
        for episode_id, episode in zip(source_episode_ids, episodes):
            try:
                stats = verify_episode(episode, camera_key)
                trajectory_bytes += int(stats["trajectory_bytes"])
                video_bytes += int(stats["video_bytes"])
                verified += 1
                usable_episodes.append(episode)
                usable_episode_ids.append(episode_id)
            except EpisodeVerificationError as exc:
                missing.append({"episode_id": episode_id, "code": exc.code, "error": str(exc)})
            except json.JSONDecodeError as exc:
                missing.append({"episode_id": episode_id, "code": "invalid_metadata_json", "error": str(exc)})
            except (FileNotFoundError, OSError) as exc:
                missing.append({"episode_id": episode_id, "code": "file_read_error", "error": str(exc)})
    else:
        usable_episodes = episodes
        usable_episode_ids = source_episode_ids

    unfilterable = [item for item in missing if item["code"] not in FILTERABLE_ERROR_CODES]
    filtered = [item for item in missing if item["code"] in FILTERABLE_ERROR_CODES] if filter_loader_unusable else []
    if filter_loader_unusable:
        _atomic_write_path_list(usable_episodes, filtered_paths_output)
        effective_path_list = filtered_paths_output
        episode_ids = usable_episode_ids
    else:
        effective_path_list = path_list
        episode_ids = source_episode_ids
    identity_lines = [f"{item}\n".encode("utf-8") for item in sorted(episode_ids)]

    source_inventory = None
    path_encoding = None
    source_listing_matches_staged = None
    if source_index_manifest is not None:
        with source_index_manifest.open("r", encoding="utf-8") as stream:
            source_index = json.load(stream)
        source_inventory = source_index.get("source_inventory")
        path_encoding = source_index.get("path_encoding") or {
            "scheme": "identity",
            "schema_version": 1,
        }
        expected_source_hash = _sha256_bytes(source_identity_lines)
        source_staged_hash = (
            source_inventory.get("staged_episode_ids_sha256") if isinstance(source_inventory, dict) else None
        )
        source_listing_matches_staged = bool(
            source_index.get("verification", {}).get("source_listing_matches_staged")
            and source_index.get("episode_count") == len(source_episode_ids)
            and source_index.get("canonical_episode_ids_sha256") == expected_source_hash
            and isinstance(source_inventory, dict)
            and source_inventory.get("episode_count") == len(source_episode_ids)
            and (source_staged_hash or source_inventory.get("canonical_episode_ids_sha256")) == expected_source_hash
            and isinstance(path_encoding, dict)
        )
        if not source_listing_matches_staged:
            raise ValueError("Source inventory manifest does not match the verified staged episode list")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "DROID raw non-stereo HD",
        "dataset_version": dataset_version,
        "license": "CC-BY-4.0",
        "source_uri": source_uri.rstrip("/"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "droid_root": str(droid_root.resolve()),
        "path_list": str((published_path_list_path or effective_path_list).resolve()),
        "path_list_sha256": sha256_file(effective_path_list),
        "source_path_list_sha256": sha256_file(path_list),
        "canonical_episode_ids_sha256": _sha256_bytes(identity_lines),
        "episode_count": len(episode_ids),
        "source_canonical_episode_ids_sha256": _sha256_bytes(source_identity_lines),
        "source_episode_count": len(source_episode_ids),
        "camera_key": camera_key,
        "source_inventory": source_inventory,
        "path_encoding": path_encoding,
        "verification": {
            "files_checked": verify_files,
            "verified_episode_count": verified,
            "missing_episode_count": len(missing),
            "filtered_episode_count": len(filtered),
            "unfilterable_error_count": len(unfilterable),
            "error_category_counts": dict(sorted(collections.Counter(item["code"] for item in missing).items())),
            "required_trajectory_bytes": trajectory_bytes,
            "required_camera_video_bytes": video_bytes,
            "source_listing_matches_staged": source_listing_matches_staged,
            # A listing-only pass is useful for diagnostics but is never a
            # complete research manifest.  Training also checks files_checked.
            "complete": verify_files and not unfilterable and (not missing or filter_loader_unusable),
            "errors": missing,
        },
    }
    if filter_loader_unusable:
        exclusion_lines = [
            f"{item['episode_id']}\t{item['code']}\n".encode("utf-8")
            for item in sorted(filtered, key=lambda item: (item["episode_id"], item["code"]))
        ]
        manifest["loader_ready_filter"] = {
            "policy": LOADER_READY_FILTER_POLICY,
            "camera_key": camera_key,
            "allowed_exclusion_codes": sorted(FILTERABLE_ERROR_CODES),
            "excluded_episode_count": len(filtered),
            "excluded_episode_records_sha256": _sha256_bytes(exclusion_lines),
            "all_source_episodes_accounted_for": len(episode_ids) + len(filtered) == len(source_episode_ids),
        }
        manifest["verification"]["complete"] = bool(
            manifest["verification"]["complete"]
            and manifest["loader_ready_filter"]["all_source_episodes_accounted_for"]
        )
    identity = {
        key: manifest[key]
        for key in (
            "schema_version",
            "dataset",
            "dataset_version",
            "source_uri",
            "canonical_episode_ids_sha256",
            "episode_count",
            "camera_key",
        )
    }
    identity["source_episode_count"] = manifest["source_episode_count"]
    identity["source_canonical_episode_ids_sha256"] = manifest["source_canonical_episode_ids_sha256"]
    if filter_loader_unusable:
        identity["loader_ready_filter"] = manifest["loader_ready_filter"]
    if source_inventory is not None:
        identity["source_inventory"] = {
            key: source_inventory[key]
            for key in (
                "root_uri",
                "episode_count",
                "canonical_episode_ids_sha256",
                "staged_episode_ids_sha256",
            )
            if key in source_inventory
        }
        identity["path_encoding"] = path_encoding
    manifest["dataset_fingerprint"] = _sha256_bytes(
        [json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")]
    )
    return manifest


def _atomic_write_path_list(episodes: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for index, episode in enumerate(episodes):
                stream.write(f"{episode} {index}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(document: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True, type=Path, help="Generated droid_paths.csv")
    parser.add_argument("--droid-root", required=True, type=Path, help="Mounted droid_raw/1.0.1 directory")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON manifest")
    parser.add_argument("--source-uri", default=DEFAULT_SOURCE_URI)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--camera-key", default="left_mp4_path")
    parser.add_argument("--source-index-manifest", type=Path)
    parser.add_argument(
        "--filter-loader-unusable",
        action="store_true",
        help="Publish only episodes that the configured raw loader can open; record every exclusion",
    )
    parser.add_argument("--filtered-paths-output", type=Path)
    parser.add_argument(
        "--published-path-list-path",
        type=Path,
        help="Stable mounted path to record in the manifest for the filtered path list",
    )
    parser.add_argument(
        "--no-verify-files",
        action="store_true",
        help="Fingerprint only; do not check trajectory/video files (not valid for a full launch gate)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        args.paths,
        args.droid_root,
        source_uri=args.source_uri,
        dataset_version=args.dataset_version,
        camera_key=args.camera_key,
        verify_files=not args.no_verify_files,
        source_index_manifest=args.source_index_manifest,
        filter_loader_unusable=args.filter_loader_unusable,
        filtered_paths_output=args.filtered_paths_output,
        published_path_list_path=args.published_path_list_path,
    )
    atomic_write_json(manifest, args.output)
    if not manifest["verification"]["complete"]:
        raise SystemExit(
            "DROID verification did not produce a complete loader-ready manifest; "
            f"unusable={manifest['verification']['missing_episode_count']}, "
            f"unfilterable={manifest['verification']['unfilterable_error_count']}; "
            f"see {args.output}"
        )
    print(
        f"DROID manifest: {manifest['episode_count']} episodes, "
        f"fingerprint={manifest['dataset_fingerprint']}, output={args.output}"
    )


if __name__ == "__main__":
    main()
