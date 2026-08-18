#!/usr/bin/env python3
"""Publish or verify the released-DROID qualification receipt."""

from __future__ import annotations

import argparse
import json

from src.utils.qualification import (
    publish_released_droid_qualification,
    verify_released_droid_qualification,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish")
    publish.add_argument("--rollout-result", required=True)
    publish.add_argument("--planning-registry", required=True)
    publish.add_argument("--dataset-manifest", required=True)
    publish.add_argument("--dinov3-weights", required=True)
    publish.add_argument("--planning-wandb-run-id-file", required=True)
    publish.add_argument("--git-commit", required=True)
    publish.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", required=True)
    verify.add_argument("--dataset-manifest", required=True)
    verify.add_argument("--dinov3-weights", required=True)
    verify.add_argument("--git-commit", required=True)
    args = parser.parse_args()

    if args.command == "publish":
        result = publish_released_droid_qualification(
            rollout_result_path=args.rollout_result,
            planning_registry_path=args.planning_registry,
            dataset_manifest_path=args.dataset_manifest,
            dinov3_weights_path=args.dinov3_weights,
            planning_wandb_run_id_path=args.planning_wandb_run_id_file,
            source_git_commit=args.git_commit,
            output_path=args.output,
        )
    else:
        result = verify_released_droid_qualification(
            args.receipt,
            dataset_manifest_path=args.dataset_manifest,
            dinov3_weights_path=args.dinov3_weights,
            source_git_commit=args.git_commit,
        )
    print(json.dumps({"status": result["status"], "integrity_sha256": result["integrity_sha256"]}))


if __name__ == "__main__":
    main()
