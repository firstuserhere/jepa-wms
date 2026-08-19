#!/usr/bin/env python3
"""Generate the JEPA-WM path list and fingerprint for staged DROID raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

GS_URI = re.compile(r"^gs://[a-z0-9][a-z0-9._-]{1,221}[a-z0-9](?:/[A-Za-z0-9._~!$&'()+,;=:@%/-]+)?$")
FRANKA_REPOSITORY = "facebook/jepa-wms"
FRANKA_REVISION = "6116f042ae7ae4c8e3f1fd2f194f432615664182"
RCLONE_VERSION = "v1.73.5"
RCLONE_LOCAL_ENCODING = "Slash,Colon,InvalidUtf8,Dot"
ASCII_COLON = ":"
FULLWIDTH_COLON = "："
RCLONE_PARTIAL_NAME = re.compile(r".+\.[0-9a-f]{8}\.partial$")


def encode_source_episode_id(episode_id: str) -> str:
    """Return the collision-safe local ID produced by rclone's Colon encoding.

    Crusoe SharedFS rejects U+003A in path components.  We deliberately reject a
    source ID that already contains rclone's U+FF1A replacement instead of guessing
    at an escape rule; this keeps the source-to-staged mapping auditable and bijective.
    """

    if FULLWIDTH_COLON in episode_id:
        raise ValueError(
            "Official DROID episode ID already contains the reserved U+FF1A "
            f"path-encoding character: {episode_id!r}"
        )
    return episode_id.replace(ASCII_COLON, FULLWIDTH_COLON)


def encode_source_episode_ids(episode_ids: list[str]) -> list[str]:
    encoded = [encode_source_episode_id(episode_id) for episode_id in episode_ids]
    if len(encoded) != len(set(encoded)):
        collisions: dict[str, list[str]] = {}
        for source, staged in zip(episode_ids, encoded):
            collisions.setdefault(staged, []).append(source)
        preview = [values for values in collisions.values() if len(values) > 1][:3]
        raise ValueError(f"DROID path encoding is not bijective; collisions={preview}")
    return encoded


def remove_stale_rclone_partials(root: Path) -> list[str]:
    """Remove only rclone's randomized local temporary files below ``root``.

    Interrupted copies can leave ``NAME.<8 lowercase hex>.partial`` files.
    A subsequent ``rclone copy`` correctly ignores these unrelated destination
    objects, while strict ``rclone check`` rejects them.  Cleanup is therefore
    explicit, confined to one already-validated institution directory, and
    rejects symlinks rather than following or deleting them.
    """

    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Rclone partial cleanup root is not a directory: {root}")
    removed: list[str] = []
    for path in sorted(root.rglob("*.partial")):
        if not RCLONE_PARTIAL_NAME.fullmatch(path.name):
            continue
        if path.is_symlink():
            raise RuntimeError(f"Refusing to remove symlink matching rclone partial pattern: {path}")
        if not path.is_file():
            raise RuntimeError(f"Rclone partial candidate is not a regular file: {path}")
        relative = path.relative_to(root).as_posix()
        path.unlink()
        removed.append(relative)
    return removed


def _path_encoding_descriptor(encoded: bool) -> dict[str, object]:
    if not encoded:
        return {"scheme": "identity", "schema_version": 1}
    return {
        "scheme": "rclone-local-colon",
        "schema_version": 1,
        "implementation": "rclone",
        "rclone_version": RCLONE_VERSION,
        "local_encoding": RCLONE_LOCAL_ENCODING,
        "source_codepoint": "U+003A",
        "staged_codepoint": "U+FF1A",
        "preexisting_staged_codepoint_rejected": True,
        "reversible_for_verified_source_inventory": True,
    }


def validate_gs_uri(value: str) -> str:
    value = value.rstrip("/")
    if not GS_URI.fullmatch(value) or any(character.isspace() for character in value):
        raise ValueError(f"Expected a GCS bucket/prefix URI, got {value!r}")
    return value


def _anonymous_rclone_droid_remote(target_uri: str) -> str:
    target_uri = validate_gs_uri(target_uri)
    if target_uri != "gs://gresearch/robotics":
        raise ValueError("Anonymous rclone listing is restricted to the official DROID source")
    return ":gcs,anonymous=true:gresearch/robotics/droid_raw/1.0.1"


def list_episode_ids_gcs(target_uri: str) -> list[str]:
    root_uri = f"{target_uri}/droid_raw/1.0.1"
    rclone = os.environ.get("JEPAWM_RCLONE")
    if rclone and target_uri.rstrip("/") == "gs://gresearch/robotics":
        listing_command = [
            rclone,
            "lsf",
            _anonymous_rclone_droid_remote(target_uri),
            "--recursive",
            "--files-only",
            "--include",
            "**/trajectory.h5",
            "--fast-list",
        ]
        prefix = ""
    else:
        listing_command = ["gsutil", "ls", "-r", f"{root_uri}/**/trajectory.h5"]
        prefix = root_uri + "/"
    listing = subprocess.Popen(
        listing_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert listing.stdout is not None
    episode_ids = []
    for raw_line in listing.stdout:
        object_uri = raw_line.strip()
        if not object_uri.endswith("/trajectory.h5") or not object_uri.startswith(prefix):
            continue
        episode_id = object_uri[len(prefix) : -len("/trajectory.h5")]
        if any(character.isspace() for character in episode_id):
            raise ValueError(f"DROID episode path contains whitespace: {episode_id!r}")
        episode_ids.append(episode_id)
    stderr = "" if listing.stderr is None else listing.stderr.read()
    return_code = listing.wait()
    if return_code:
        raise RuntimeError(
            f"DROID source listing failed with exit code {return_code}: {stderr[-2000:]}"
        )
    return sorted(set(episode_ids))


def list_episode_ids_local(target_root: Path) -> list[str]:
    """List a staged PVC/filesystem tree without depending on object-store credentials."""

    root = (target_root / "droid_raw" / "1.0.1").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Staged DROID root is missing: {root}")
    episode_ids = []
    for trajectory in root.rglob("trajectory.h5"):
        if trajectory.is_symlink() or not trajectory.is_file():
            continue
        episode_id = trajectory.parent.relative_to(root).as_posix()
        if any(character.isspace() for character in episode_id):
            raise ValueError(f"DROID episode path contains whitespace: {episode_id!r}")
        episode_ids.append(episode_id)
    return sorted(set(episode_ids))


def _episode_ids_sha256(episode_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for episode_id in episode_ids:
        digest.update(f"{episode_id}\n".encode("utf-8"))
    return digest.hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _verified_tree(root: Path) -> tuple[list[dict[str, object]], str, int]:
    if not root.is_dir():
        raise FileNotFoundError(f"Auxiliary dataset root is missing: {root}")
    records: list[dict[str, object]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Auxiliary dataset contains an unresolved symlink: {path}")
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        size = path.stat().st_size
        records.append({"path": relative_path, "size": size, "sha256": _sha256_file(path)})
        total_bytes += size
    if not records:
        raise RuntimeError(f"Auxiliary dataset contains no files: {root}")
    canonical = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for record in records
    )
    return records, hashlib.sha256(canonical).hexdigest(), total_bytes


def bind_franka_manifest(manifest_path: Path, staged_root: Path, source_root: Path) -> None:
    """Bind exact, verified Franka_hf bytes into the combined dataset identity."""

    source_files, source_tree_sha256, source_bytes = _verified_tree(source_root)
    staged_files, staged_tree_sha256, staged_bytes = _verified_tree(staged_root)
    if source_files != staged_files or source_tree_sha256 != staged_tree_sha256:
        raise RuntimeError("Staged Franka_hf tree does not byte-match the pinned Hub checkout")

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    verification = manifest.get("verification", {})
    if verification.get("files_checked") is not True or verification.get("complete") is not True:
        raise RuntimeError("Refusing to augment an unverified DROID manifest")
    raw_fingerprint = manifest.get("dataset_fingerprint")
    if not raw_fingerprint:
        raise RuntimeError("DROID manifest has no primary dataset fingerprint")

    auxiliary_identity = {
        "dataset": "Franka_hf",
        "hub_repository": FRANKA_REPOSITORY,
        "hub_repository_type": "dataset",
        "hub_revision": FRANKA_REVISION,
        "subdirectory": "franka_custom",
        "file_count": len(staged_files),
        "total_bytes": staged_bytes,
        "tree_sha256": staged_tree_sha256,
        "files": staged_files,
        "verification": {
            "files_checked": True,
            "complete": True,
            "source_tree_matches_staged": True,
            "source_total_bytes": source_bytes,
        },
    }
    combined_identity = {
        "schema_version": 1,
        "primary_dataset_fingerprint": raw_fingerprint,
        "auxiliary_dataset": {
            key: auxiliary_identity[key]
            for key in (
                "dataset",
                "hub_repository",
                "hub_repository_type",
                "hub_revision",
                "subdirectory",
                "file_count",
                "total_bytes",
                "tree_sha256",
            )
        },
    }
    manifest["primary_dataset_fingerprint"] = raw_fingerprint
    manifest["auxiliary_datasets"] = {"Franka_hf": auxiliary_identity}
    manifest["dataset_fingerprint"] = hashlib.sha256(
        json.dumps(combined_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    verification["auxiliary_files_checked"] = True
    verification["auxiliary_complete"] = True
    manifest["verification"] = verification

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, manifest_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_artifacts(
    *,
    target_uri: str | None,
    target_root: Path | None,
    staged_identity: str | None,
    expected_source_root_uri: str | None,
    mount_root: str,
    min_episodes: int,
    output_dir: Path,
) -> None:
    if (target_uri is None) == (target_root is None):
        raise ValueError("Specify exactly one of target_uri and target_root")
    episode_ids = (
        list_episode_ids_gcs(target_uri)
        if target_uri is not None
        else list_episode_ids_local(target_root)
    )
    if len(episode_ids) < min_episodes:
        raise RuntimeError(f"Only {len(episode_ids)} DROID episodes found; expected at least {min_episodes}")

    source_inventory = None
    path_encoding = _path_encoding_descriptor(target_root is not None)
    if expected_source_root_uri is not None:
        expected_source_root_uri = validate_gs_uri(expected_source_root_uri)
        source_episode_ids = list_episode_ids_gcs(expected_source_root_uri)
        expected_staged_ids = (
            sorted(encode_source_episode_ids(source_episode_ids))
            if target_root is not None
            else source_episode_ids
        )
        if episode_ids != expected_staged_ids:
            staged = set(episode_ids)
            source = set(expected_staged_ids)
            missing = sorted(source - staged)
            extra = sorted(staged - source)
            raise RuntimeError(
                "Staged DROID trajectory inventory does not match the live official source: "
                f"missing={len(missing)} {missing[:3]}, extra={len(extra)} {extra[:3]}"
            )
        source_inventory = {
            "root_uri": f"{expected_source_root_uri}/droid_raw/1.0.1",
            "episode_count": len(source_episode_ids),
            "canonical_episode_ids_sha256": _episode_ids_sha256(source_episode_ids),
            "staged_episode_ids_sha256": _episode_ids_sha256(expected_staged_ids),
            "listed_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    mount_root = mount_root.rstrip("/")
    identity = hashlib.sha256()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths_file = output_dir / "droid_paths.csv"
    with paths_file.open("w", encoding="utf-8") as stream:
        for index, episode_id in enumerate(episode_ids):
            identity.update(f"{episode_id}\n".encode("utf-8"))
            stream.write(f"{mount_root}/droid_raw/1.0.1/{episode_id} {index}\n")

    index_manifest = {
        "schema_version": 1,
        "dataset": "DROID raw non-stereo HD",
        "dataset_version": "1.0.1",
        "source_uri": "gs://gresearch/robotics/droid_raw/1.0.1",
        "staged_uri": (
            f"{target_uri}/droid_raw/1.0.1"
            if target_uri is not None
            else str(staged_identity or target_root)
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(episode_ids),
        "camera_key": "left_mp4_path",
        "copy_exclude_regex": r".*SVO.*|.*stereo.*\.mp4$",
        "path_encoding": path_encoding,
        "canonical_episode_ids_sha256": identity.hexdigest(),
        "source_inventory": source_inventory,
        "path_list_sha256": hashlib.sha256(paths_file.read_bytes()).hexdigest(),
        "verification": {
            "files_checked": False,
            "rsync_completed": True,
            "trajectory_objects_listed": len(episode_ids),
            "minimum_episode_gate": min_episodes,
            "source_listing_matches_staged": source_inventory is not None,
            "complete": False,
        },
    }
    index_file = output_dir / "droid_index_manifest.json"
    index_file.write_text(json.dumps(index_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Staged DROID index contains {len(episode_ids)} episodes; full file verification is still required")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--target-uri", type=validate_gs_uri)
    target.add_argument("--target-root", type=Path)
    parser.add_argument(
        "--staged-identity",
        help="Mount-independent identity such as sky-volume://NAME/droid_raw/1.0.1",
    )
    parser.add_argument("--mount-root", default="/mnt/jepawm-datasets")
    parser.add_argument("--min-episodes", type=int, default=50_000)
    parser.add_argument(
        "--expected-source-root-uri",
        help="Official GCS root whose trajectory inventory must exactly match the staged copy",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bind-franka-manifest", type=Path)
    parser.add_argument("--franka-staged-root", type=Path)
    parser.add_argument("--franka-source-root", type=Path)
    parser.add_argument("--cleanup-rclone-partials", type=Path)
    args = parser.parse_args()
    if args.cleanup_rclone_partials is not None:
        removed = remove_stale_rclone_partials(args.cleanup_rclone_partials)
        print(f"Removed {len(removed)} stale rclone partial file(s)")
        return
    if args.bind_franka_manifest is not None:
        if args.franka_staged_root is None or args.franka_source_root is None:
            parser.error("--bind-franka-manifest requires both Franka roots")
        bind_franka_manifest(
            args.bind_franka_manifest,
            args.franka_staged_root,
            args.franka_source_root,
        )
        return
    if (args.target_uri is None and args.target_root is None) or args.output_dir is None:
        parser.error("index generation requires one staging target and --output-dir")
    write_artifacts(
        target_uri=args.target_uri,
        target_root=args.target_root,
        staged_identity=args.staged_identity,
        expected_source_root_uri=args.expected_source_root_uri,
        mount_root=args.mount_root,
        min_episodes=args.min_episodes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
