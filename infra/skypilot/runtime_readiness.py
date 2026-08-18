#!/usr/bin/env python3
"""Publish or verify the distributed runtime-readiness receipt."""

from __future__ import annotations

import argparse
import json

from src.utils.runtime_readiness import publish_runtime_readiness, verify_runtime_readiness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--distributed-smoke", required=True)
    publish.add_argument("--training-smoke", required=True)
    publish.add_argument("--training-checkpoint-root", required=True)
    publish.add_argument("--dataset-manifest", required=True)
    publish.add_argument("--dinov3-weights", required=True)
    publish.add_argument("--git-commit", required=True)
    publish.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", required=True)
    verify.add_argument("--dataset-manifest", required=True)
    verify.add_argument("--dinov3-weights", required=True)
    verify.add_argument("--git-commit", required=True)
    args = parser.parse_args()

    if args.command == "publish":
        result = publish_runtime_readiness(
            distributed_smoke_path=args.distributed_smoke,
            training_smoke_path=args.training_smoke,
            training_checkpoint_root=args.training_checkpoint_root,
            dataset_manifest_path=args.dataset_manifest,
            dinov3_weights_path=args.dinov3_weights,
            source_git_commit=args.git_commit,
            output_path=args.output,
        )
    else:
        result = verify_runtime_readiness(
            args.receipt,
            dataset_manifest_path=args.dataset_manifest,
            dinov3_weights_path=args.dinov3_weights,
            source_git_commit=args.git_commit,
        )
    print(json.dumps({"status": result["status"], "integrity_sha256": result["integrity_sha256"]}))


if __name__ == "__main__":
    main()
